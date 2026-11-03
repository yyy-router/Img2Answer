from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from PIL import Image

from img2answer.question_index import index_questions
from img2answer.search import fetch_question_detail, search_similar_questions
from img2answer.vector_store import ImageSearchMatch, PillowHashEmbeddingModel
from question_helpers import QuestionFixture
from test_question_index import RecordingQuestionVectors


class SearchQuestionVectors(RecordingQuestionVectors):
    def __init__(self):
        super().__init__()
        self.extra = []
        self.limits = []

    def count(self):
        return len(self.extra) + len(self.records)

    def query_similar_images(self, embedding, top_k):
        self.limits.append(top_k)
        return (self.extra + [ImageSearchMatch(record.image_id, float(index), record.metadata, record.document)
                              for index, record in enumerate(self.records.values())])[:top_k]


class QuestionSearchTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.fixture = QuestionFixture(self.root)
        self.query = self.root / "query.png"
        self.query.write_bytes(Path(self.fixture.questions[0]["image"]["output_path"]).read_bytes())
        self.vectors = SearchQuestionVectors()
        self.index()

    def index(self):
        return index_questions(self.fixture.store.path, "sample", self.root / "chroma",
                               vector_store_factory=Mock(return_value=self.vectors))

    def search(self, top_k=5):
        return search_similar_questions(self.query,
                                        sqlite_store=self.fixture.store, vector_store=self.vectors,
                                        embedding_model=PillowHashEmbeddingModel(),
                                        embedding_model_name="pillow-hash", top_k=top_k)

    def test_current_question_fields_null_answer_and_paths_not_exposed(self):
        first, second = self.search()
        self.assertEqual(first["answer"], "A")
        self.assertIsNone(second["answer"])
        self.assertEqual(first["options"], dict(A="first", B="second", C="third", D="fourth"))
        self.assertEqual(first["parse_status"], "needs_review")
        self.assertIn("missing_answer", second["review_reasons"])
        self.assertEqual(first["page_number"], 1)
        self.assertEqual(first["analysis_text"], "示例")
        self.assertNotIn(str(self.root), str(first))
        self.assertNotIn("raw_text", first)
        self.assertIn("raw_text", fetch_question_detail(self.fixture.store, first["question_id"]))

    def test_stale_top_hits_do_not_occupy_valid_top_k(self):
        self.vectors.extra = [ImageSearchMatch(f"deleted-{i}", -10.0 + i, {}, None) for i in range(3)]
        results = self.search(top_k=2)
        self.assertEqual([result["rank"] for result in results], [1, 2])
        self.assertEqual(self.vectors.limits, [2, 4, 5])

    def test_duplicate_hits_are_deduplicated_and_ranked(self):
        record = next(iter(self.vectors.records.values()))
        self.vectors.extra = [ImageSearchMatch(record.image_id, -1.0, record.metadata, None)]
        self.assertEqual(len(self.search(top_k=3)), 2)

    def test_reextract_same_ids_rejects_old_vectors_until_reindexed(self):
        ids = set(self.vectors.records)
        self.fixture.extract()
        self.assertEqual(ids, {q["image_id"] for q in self.fixture.questions})
        self.assertEqual(self.search(), [])
        self.index()
        self.assertEqual(len(self.search()), 2)

    def test_reimport_immediately_invalidates_old_vectors(self):
        self.fixture.import_run()
        self.assertEqual(self.search(), [])
        self.assertIsNone(fetch_question_detail(self.fixture.store, self.fixture.questions[0]["id"]))

    def test_changed_missing_and_corrupt_image_are_filtered(self):
        question = self.fixture.questions[0]
        path = Path(question["image"]["output_path"])
        original = path.read_bytes()
        data = io.BytesIO()
        Image.new("RGB", (question["image"]["width"], question["image"]["height"]), "white").save(data, "PNG")
        for value in (data.getvalue(), b"broken", None):
            with self.subTest(value=None if value is None else len(value)):
                if value is None:
                    path.unlink()
                else:
                    path.write_bytes(value)
                results = self.search()
                self.assertEqual([result["question_id"] for result in results], [self.fixture.questions[1]["id"]])
                self.assertIsNotNone(fetch_question_detail(self.fixture.store, question["id"]))
                path.write_bytes(original)

    def test_wrong_metadata_is_never_attached_to_current_answer(self):
        vector = next(iter(self.vectors.records.values()))
        original = deepcopy(vector.metadata)
        for key in ("index_schema_version", "embedding_signature", "question_id", "image_id", "document_id",
                    "image_role", "section", "record_revision", "image_sha256"):
            with self.subTest(key=key):
                vector.metadata[key] = "wrong"
                self.assertEqual(len(self.search()), 1)
                vector.metadata.clear()
                vector.metadata.update(original)

    def test_corrupt_stored_json_is_a_data_error(self):
        self.fixture.update("UPDATE questions SET options_json = 'broken' WHERE id = ?",
                            (self.fixture.questions[0]["id"],))
        with self.assertRaises(ValueError):
            self.search()
        with self.assertRaises(ValueError):
            fetch_question_detail(self.fixture.store, self.fixture.questions[0]["id"])

    def test_source_hash_or_image_role_changes_invalidate_current_answers(self):
        self.fixture.update("UPDATE source_documents SET sha256 = ?", ("b" * 64,))
        self.assertEqual(self.search(), [])
        self.assertIsNone(fetch_question_detail(self.fixture.store, self.fixture.questions[0]["id"]))
        self.fixture.update("UPDATE source_documents SET sha256 = ?", ("a" * 64,))
        self.fixture.update("UPDATE question_images SET image_role = 'crop_candidate' WHERE id = ?",
                            (self.fixture.questions[0]["image_id"],))
        self.assertEqual(len(self.search()), 1)

    def test_empty_collection_and_invalid_top_k(self):
        self.vectors.records.clear()
        self.assertEqual(self.search(), [])
        with self.assertRaises(ValueError):
            self.search(top_k=0)


if __name__ == "__main__":
    unittest.main()
