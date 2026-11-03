from contextlib import closing, redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from img2answer import cli
from img2answer.question_index import QuestionIndexError, index_questions, record_revision
from img2answer.vector_store import (
    ChromaImageVectorStore, ChromaQuestionVectorStore, PillowHashEmbeddingModel,
    QuestionIndexUnavailableError, question_embedding_signature, resolve_question_collection,
)
from question_helpers import QuestionFixture


class RecordingQuestionVectors:
    def __init__(self):
        self.records = {}
        self.upserts = 0
        self.fail_upsert = False
        self.fail_delete = False

    def document_ids(self, document_id, signature):
        return {key for key, value in self.records.items()
                if value.metadata["document_id"] == document_id
                and value.metadata["embedding_signature"] == signature
                and value.metadata["image_role"] == "question_crop"}

    def upsert_images(self, records):
        for record in records:
            self.records[record.image_id] = record
            if self.fail_upsert:
                raise RuntimeError("partial write")
        self.upserts += 1

    def delete_ids(self, ids):
        if self.fail_delete:
            raise RuntimeError("cleanup failed")
        for image_id in ids:
            self.records.pop(image_id, None)


class QuestionIndexTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.fixture = QuestionFixture(self.root)
        self.vectors = RecordingQuestionVectors()
        self.factory = Mock(return_value=self.vectors)

    def index(self, **kwargs):
        return index_questions(self.fixture.store.path, self.fixture.document_id, self.root / "chroma",
                               vector_store_factory=self.factory, **kwargs)

    def test_index_uses_only_questions_and_persists_revision_and_image_digest(self):
        before = [dict(row) for row in self.fixture.store.fetch_all("questions")]
        with patch("img2answer.ocr.OCRSettings.from_env", side_effect=AssertionError("credentials")):
            result = self.index()
        self.assertEqual(result["indexed"], 2)
        self.assertEqual(result["deleted"], 0)
        self.assertEqual(result["chroma_collection"], "question_candidates")
        with self.fixture.store.question_connection() as connection:
            records = self.fixture.store.question_records(connection)
        for record in records:
            vector = self.vectors.records[record["image_id"]]
            self.assertEqual(vector.metadata["record_revision"], record_revision(record))
            self.assertEqual(vector.metadata["image_sha256"], hashlib.sha256(
                Path(record["output_path"]).read_bytes()).hexdigest())
            self.assertNotIn("answer", vector.metadata)
            self.assertNotIn(str(self.root), vector.document)
        self.assertEqual(before, [dict(row) for row in self.fixture.store.fetch_all("questions")])

    def test_repeat_and_shrink_and_zero_candidates_only_replace_one_document(self):
        self.index()
        other = QuestionFixture(self.root, "other")
        index_questions(other.store.path, "other", self.root / "chroma", vector_store_factory=self.factory)
        other_ids = {q["image_id"] for q in other.questions}
        self.index()
        self.assertEqual(len(self.vectors.records), 4)
        self.fixture.update("DELETE FROM question_images WHERE id = ?", (self.fixture.questions[1]["image_id"],))
        self.assertEqual(self.index()["deleted"], 1)
        self.fixture.clear_questions()
        with patch("img2answer.question_index.create_embedding_model", side_effect=AssertionError("empty model")):
            result = self.index()
        self.assertEqual((result["indexed"], result["deleted"]), (0, 1))
        self.assertEqual(set(self.vectors.records), other_ids)

    def test_unknown_document_and_missing_database_do_not_create_or_clear(self):
        self.index()
        before = deepcopy(self.vectors.records)
        for database, document in ((self.root / "missing.sqlite3", "sample"),
                                   (self.fixture.store.path, "unknown")):
            with self.subTest(database=database, document=document), self.assertRaises(QuestionIndexError):
                index_questions(database, document, self.root / "chroma", vector_store_factory=self.factory)
        self.assertFalse((self.root / "missing.sqlite3").exists())
        self.assertEqual(before, self.vectors.records)

    def test_missing_corrupt_or_wrong_size_image_never_publishes(self):
        self.index()
        before = deepcopy(self.vectors.records)
        path = Path(self.fixture.questions[1]["image"]["output_path"])
        original = path.read_bytes()
        for value in (None, b"not an image", self.fixture.snapshot.image):
            with self.subTest(value=None if value is None else len(value)):
                if value is None:
                    path.unlink()
                else:
                    path.write_bytes(value)
                self.factory.reset_mock()
                with self.assertRaises(QuestionIndexError):
                    self.index()
                self.factory.assert_not_called()
                self.assertEqual(before, self.vectors.records)
                path.write_bytes(original)

    def test_missing_question_schema_is_not_upgraded(self):
        self.fixture.update("DROP TABLE questions")
        with self.assertRaises(QuestionIndexError):
            self.index()
        self.factory.assert_not_called()
        with closing(sqlite3.connect(self.fixture.store.path)) as connection:
            self.assertFalse(connection.execute("SELECT name FROM sqlite_master WHERE name = 'questions'").fetchall())

    def test_final_publication_holds_sqlite_write_lock_without_changing_rows(self):
        def factory(*args, **kwargs):
            with closing(sqlite3.connect(self.fixture.store.path, timeout=0)) as connection:
                with self.assertRaisesRegex(sqlite3.OperationalError, "locked"):
                    connection.execute("UPDATE source_documents SET updated_at = updated_at")
            return self.vectors
        self.factory.side_effect = factory
        self.index()
        # The write lock has been released after successful publication.
        self.fixture.update("UPDATE source_documents SET updated_at = updated_at")

    def test_invalid_or_inconsistent_embedding_preserves_old_index(self):
        self.index()
        before = deepcopy(self.vectors.records)
        for values in (([], []), ([float("nan")], [0]), ([float("inf")], [0]),
                       ([0.1], [0.2, 0.3]), ([True], [False])):
            with self.subTest(values=values), self.assertRaises(QuestionIndexError):
                self.index(embedding_model=Mock(embed_image=Mock(side_effect=values)))
            self.assertEqual(before, self.vectors.records)

    def test_embedding_failure_cleans_snapshot_and_does_not_publish(self):
        paths = []
        def fail(path):
            paths.append(path)
            raise RuntimeError("model failed")
        with self.assertRaises(QuestionIndexError):
            self.index(embedding_model=Mock(embed_image=fail))
        self.factory.assert_not_called()
        self.assertTrue(paths)
        self.assertFalse(paths[0].parent.exists())

    def test_reextract_or_reimport_or_picture_mutation_during_embedding_rejects_publish(self):
        for change in (self.fixture.extract, self.fixture.import_run,
                       lambda: Image.new("RGB", (self.fixture.questions[0]["image"]["width"],
                                                  self.fixture.questions[0]["image"]["height"]), "white").save(
                           self.fixture.questions[0]["image"]["output_path"])):
            with self.subTest(change=change):
                self.fixture.extract()
                self.factory.reset_mock()
                model = PillowHashEmbeddingModel()
                done = False
                def embed(path):
                    nonlocal done
                    result = model.embed_image(path)
                    if not done:
                        done = True
                        change()
                    return result
                with self.assertRaises(QuestionIndexError):
                    self.index(embedding_model=Mock(embed_image=embed))
                self.factory.assert_not_called()

    def test_snapshot_bytes_are_the_same_bytes_hashed_and_embedded(self):
        embedded = []
        model = PillowHashEmbeddingModel()
        def embed(path):
            embedded.append(path.read_bytes())
            return model.embed_image(path)
        self.index(embedding_model=Mock(embed_image=embed))
        self.assertEqual([hashlib.sha256(data).hexdigest() for data in embedded],
                         [record.metadata["image_sha256"] for record in self.vectors.records.values()])

    def test_partial_publish_and_cleanup_failures_can_be_retried(self):
        self.vectors.fail_upsert = True
        with self.assertRaisesRegex(QuestionIndexError, "可能部分写入"):
            self.index()
        self.vectors.fail_upsert = False
        self.assertEqual(self.index()["indexed"], 2)
        self.fixture.clear_questions()
        self.vectors.fail_delete = True
        with self.assertRaisesRegex(QuestionIndexError, "可能部分写入"):
            self.index()
        self.vectors.fail_delete = False
        self.assertEqual(self.index()["deleted"], 2)

    def test_cli_dispatches_and_reports_safe_errors(self):
        out, err = io.StringIO(), io.StringIO()
        with patch("img2answer.question_index.index_questions", return_value={"indexed": 2}) as run, \
                redirect_stdout(out), redirect_stderr(err):
            self.assertEqual(cli.main(["index-questions", "--database", "sample.db", "--document-id", "sample",
                                       "--embedding-model", "openclip", "--chroma-collection", "custom"]), 0)
        self.assertEqual(run.call_args.kwargs["embedding_model_name"], "openclip")
        self.assertIn("indexed=2", out.getvalue())
        with redirect_stderr(err):
            self.assertEqual(cli.main(["index-questions", "--database", str(self.root / "missing.db"),
                                       "--document-id", "sample"]), 1)
        self.assertNotIn(str(self.root), err.getvalue())


class QuestionCollectionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_missing_index_read_does_not_create_directory_or_collection(self):
        path = self.root / "missing"
        with self.assertRaises(QuestionIndexUnavailableError):
            ChromaQuestionVectorStore(path)
        self.assertFalse(path.exists())
        legacy = ChromaImageVectorStore(self.root / "chroma")
        with self.assertRaises(QuestionIndexUnavailableError):
            ChromaQuestionVectorStore(self.root / "chroma")
        self.assertEqual([value.name for value in legacy.client.list_collections()], ["question_images"])

    def test_page_collection_is_not_adopted_or_modified(self):
        legacy = ChromaImageVectorStore(self.root / "chroma")
        for create in (True, False):
            with self.subTest(create=create), self.assertRaises(QuestionIndexUnavailableError):
                ChromaQuestionVectorStore(self.root / "chroma", "question_images", create=create)
        self.assertIsNone(legacy.client.get_collection("question_images").metadata)

    def test_model_and_schema_mismatch_rejected(self):
        store = ChromaQuestionVectorStore(self.root / "chroma", create=True)
        with self.assertRaises(QuestionIndexUnavailableError):
            ChromaQuestionVectorStore(self.root / "chroma", embedding_model_name="openclip", create=True)
        store.collection.modify(metadata={**store.collection.metadata, "index_schema_version": 2})
        with self.assertRaises(QuestionIndexUnavailableError):
            ChromaQuestionVectorStore(self.root / "chroma")

    def test_names_and_signatures_are_separate(self):
        self.assertEqual(resolve_question_collection(None, "openclip"), "question_candidates_openclip")
        self.assertEqual(resolve_question_collection("custom", "openclip"), "custom")
        self.assertNotEqual(question_embedding_signature("pillow-hash"), question_embedding_signature("openclip"))
        with self.assertRaises(ValueError):
            resolve_question_collection(None, "unknown")

    def test_missing_collection_on_chroma_05_without_not_found_error(self):
        path = self.root / "chroma"
        path.mkdir()
        (path / "chroma.sqlite3").touch()
        client = Mock()
        client.get_collection.side_effect = ValueError("Collection does not exist")
        client.create_collection.return_value = SimpleNamespace(metadata={
            "index_kind": "question_candidates", "index_schema_version": 1,
            "embedding_signature": question_embedding_signature("pillow-hash"),
        })
        with patch("chromadb.errors", SimpleNamespace()), patch("chromadb.PersistentClient", return_value=client):
            with self.assertRaises(QuestionIndexUnavailableError):
                ChromaQuestionVectorStore(path)
            client.create_collection.assert_not_called()
            ChromaQuestionVectorStore(path, create=True)
            client.create_collection.assert_called_once()


if __name__ == "__main__":
    unittest.main()
