from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.parse import quote

from fastapi.testclient import TestClient
from PIL import Image

from img2answer import cli
from img2answer.api import ApiSettings, create_app
from img2answer.question_index import index_questions
from img2answer.vector_store import (
    ChromaImageVectorStore, ChromaQuestionVectorStore, EmbeddedImage, PillowHashEmbeddingModel,
)
from question_helpers import QuestionFixture
from test_question_search import SearchQuestionVectors


class QuestionApiTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.fixture = QuestionFixture(self.root)
        self.vectors = SearchQuestionVectors()
        index_questions(self.fixture.store.path, "sample", self.root / "chroma",
                        vector_store_factory=Mock(return_value=self.vectors))
        self.factory = Mock(return_value=PillowHashEmbeddingModel())
        self.vector_factory = Mock(return_value=self.vectors)
        self.settings = ApiSettings(database=self.fixture.store.path, chroma_dir=self.root / "chroma",
                                    image_root=self.root)
        self.app = create_app(settings=self.settings, question_vector_store_factory=self.vector_factory,
                              embedding_model_factory=self.factory)
        self.client = TestClient(self.app)
        self.query = Path(self.fixture.questions[0]["image"]["output_path"]).read_bytes()

    def search(self, **kwargs):
        return self.client.post("/search/question", files={"file": ("query.png", self.query, "image/png")}, **kwargs)

    def detail(self, index=0):
        return self.client.get("/questions/" + quote(self.fixture.questions[index]["id"], safe=""))

    def test_upload_returns_question_fields_and_versioned_preview(self):
        response = self.search(data={"top_k": "2"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["chroma_collection"], "question_candidates")
        self.assertEqual(len(body["results"]), 2)
        first = body["results"][0]
        preview = self.client.get(first["image"]["preview_url"])
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.content, self.query)
        self.assertEqual(preview.headers["content-type"], "image/png")
        self.assertEqual(preview.headers["cache-control"], "no-store")
        self.assertNotIn(str(self.root), response.text)
        self.vector_factory.assert_called_once_with(self.settings.chroma_dir, collection_name="question_candidates",
                                                    embedding_model_name="pillow-hash", create=False)

    def test_detail_and_preview_do_not_load_model_or_vectors(self):
        detail = self.detail()
        self.assertEqual(detail.status_code, 200)
        self.assertIn("raw_text", detail.json())
        self.assertEqual(self.client.get(detail.json()["image"]["preview_url"]).status_code, 200)
        self.factory.assert_not_called()
        self.vector_factory.assert_not_called()
        self.assertEqual(self.client.get("/questions/unknown").status_code, 404)
        self.assertEqual(self.client.get("/image-content/unknown").status_code, 404)

    def test_old_preview_revision_rejected_after_reextract(self):
        old_url = self.detail().json()["image"]["preview_url"]
        self.fixture.extract()
        self.assertEqual(self.client.get(old_url).status_code, 409)
        new_url = self.detail().json()["image"]["preview_url"]
        self.assertEqual(self.client.get(new_url).status_code, 200)
        self.assertEqual(self.search().json()["results"], [])

    def test_invalid_uploads_and_top_k_do_not_initialize_model(self):
        cases = [({}, 422), ({"files": {"file": ("empty.png", b"")}}, 400),
                 ({"files": {"file": ("bad.png", b"not image")}}, 400)]
        for args, status in cases:
            with self.subTest(args=args):
                self.assertEqual(self.client.post("/search/question", **args).status_code, status)
        self.assertEqual(self.search(data={"top_k": "0"}).status_code, 400)
        self.assertEqual(self.search(data={"top_k": "oops"}).status_code, 422)
        self.factory.assert_not_called()
        self.vector_factory.assert_not_called()

    def test_request_body_limit_runs_before_multipart_for_new_route(self):
        with patch("starlette.formparsers.SpooledTemporaryFile") as spool:
            response = self.client.post("/search/question", content=b"small", headers={"content-length": "99999999"})
        self.assertEqual(response.status_code, 413)
        spool.assert_not_called()
        self.factory.assert_not_called()

    def test_file_limit_and_query_cleanup(self):
        app = create_app(settings=ApiSettings(database=self.fixture.store.path, max_upload_bytes=128),
                         embedding_model_factory=self.factory, question_vector_store_factory=self.vector_factory)
        paths = []
        original = tempfile.TemporaryDirectory
        def temporary(**kwargs):
            value = original(dir=self.root, **kwargs)
            paths.append(Path(value.name))
            return value
        with patch("img2answer.api.tempfile.TemporaryDirectory", side_effect=temporary):
            response = TestClient(app).post("/search/question", files={"file": ("q.png", self.query + b"x" * 256)})
        self.assertEqual(response.status_code, 413)
        self.assertTrue(paths)
        self.assertTrue(all(not value.exists() for value in paths))
        self.factory.assert_not_called()

    def test_query_cleanup_after_model_error_and_no_internal_paths_in_response(self):
        paths = []
        def fail(path):
            paths.append(path)
            raise RuntimeError(str(self.root))
        self.factory.return_value = Mock(embed_image=fail)
        with self.assertLogs("img2answer.api", level="ERROR"):
            response = self.search()
        self.assertEqual(response.status_code, 500)
        self.assertNotIn(str(self.root), response.text)
        self.assertTrue(paths)
        self.assertFalse(paths[0].parent.exists())

    def test_concurrent_searches_share_one_model(self):
        with ThreadPoolExecutor(max_workers=3) as executor:
            responses = list(executor.map(lambda _: self.search(), range(3)))
        self.assertTrue(all(response.status_code == 200 for response in responses))
        self.factory.assert_called_once()

    def test_missing_database_schema_and_collection_are_503(self):
        cases = [(self.root / "missing.db", None), (self.fixture.store.path, "schema")]
        for database, change in cases:
            if change:
                self.fixture.update("DROP TABLE questions")
            client = TestClient(create_app(settings=ApiSettings(database=database),
                                           embedding_model_factory=self.factory))
            for method, path, kwargs in (("get", "/questions/unknown", {}),
                                         ("get", "/image-content/unknown", {}),
                                         ("post", "/search/question", {"files": {"file": ("q.png", self.query)}})):
                with self.subTest(database=database, path=path):
                    response = getattr(client, method)(path, **kwargs)
                    self.assertEqual(response.status_code, 503)
                    self.assertNotIn(str(self.root), response.text)
        self.assertFalse((self.root / "missing.db").exists())
        self.factory.assert_not_called()

    def test_missing_collection_does_not_initialize_model_or_create_storage(self):
        app = create_app(settings=self.settings, embedding_model_factory=self.factory)
        response = TestClient(app).post("/search/question", files={"file": ("q.png", self.query)})
        self.assertEqual(response.status_code, 503)
        self.assertFalse(self.settings.chroma_dir.exists())
        self.factory.assert_not_called()

    def test_corrupt_fields_and_storage_errors_are_sanitized_500(self):
        self.fixture.update("UPDATE questions SET options_json = 'invalid' WHERE id = ?",
                            (self.fixture.questions[0]["id"],))
        with self.assertLogs("img2answer.api", level="ERROR"):
            for response in (self.search(), self.detail()):
                self.assertEqual(response.status_code, 500)
                self.assertNotIn(str(self.root), response.text)
        self.vector_factory.side_effect = RuntimeError(str(self.root))
        with self.assertLogs("img2answer.api", level="ERROR"):
            self.assertEqual(self.search().status_code, 500)

    def test_missing_preview_still_allows_text_detail(self):
        Path(self.fixture.questions[0]["image"]["output_path"]).unlink()
        detail = self.detail()
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(self.client.get(detail.json()["image"]["preview_url"]).status_code, 404)
        self.assertEqual(len(self.search().json()["results"]), 1)

    def test_preview_rejects_outside_root_directory_nonimage_and_fake_extension(self):
        question = self.fixture.questions[0]
        directory = self.root / "directory"
        directory.mkdir()
        fake = self.root / "fake.png"
        fake.write_text("a secret, not an image", encoding="utf-8")
        restricted = self.root / "allowed"
        restricted.mkdir()
        outside_client = TestClient(create_app(settings=ApiSettings(database=self.fixture.store.path,
                                                                    image_root=restricted)))
        image_url = "/image-content/" + quote(question["image_id"], safe="")
        self.assertEqual(outside_client.get(image_url).status_code, 404)
        for path in (directory, fake, self.root / "missing.png"):
            with self.subTest(path=path):
                self.fixture.update("UPDATE question_images SET output_path = ? WHERE id = ?",
                                    (str(path), question["image_id"]))
                response = self.client.get(image_url)
                self.assertEqual(response.status_code, 404)
                self.assertNotIn(str(path), response.text)

    def test_jpeg_content_determined_by_bytes_and_unicode_ids(self):
        unicode_fixture = QuestionFixture(self.root, "中文题库")
        question = unicode_fixture.questions[0]
        path = self.root / "jpeg-without-extension"
        Image.new("RGB", (question["image"]["width"], question["image"]["height"])).save(path, "JPEG")
        self.fixture.update("UPDATE question_images SET output_path = ? WHERE id = ?", (str(path), question["image_id"]))
        detail = self.client.get("/questions/" + quote(question["id"], safe=""))
        self.assertEqual(detail.status_code, 200)
        preview = self.client.get(detail.json()["image"]["preview_url"])
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.headers["content-type"], "image/jpeg")

    def test_directory_link_cannot_escape_allowed_root(self):
        allowed, outside = self.root / "allowed", self.root / "outside"
        allowed.mkdir()
        outside.mkdir()
        (outside / "question.png").write_bytes(self.query)
        link = allowed / "link"
        if os.name == "nt":
            import _winapi
            _winapi.CreateJunction(str(outside), str(link))
        else:
            link.symlink_to(outside, target_is_directory=True)
        try:
            question = self.fixture.questions[0]
            self.fixture.update("UPDATE question_images SET output_path = ? WHERE id = ?",
                                (str(link / "question.png"), question["image_id"]))
            app = create_app(settings=ApiSettings(database=self.fixture.store.path, image_root=allowed))
            response = TestClient(app).get("/image-content/" + quote(question["image_id"], safe=""))
            self.assertEqual(response.status_code, 404)
            self.assertTrue((outside / "question.png").is_file())
        finally:
            if os.name == "nt":
                link.rmdir()
            else:
                link.unlink()

    def test_animated_png_cannot_be_served_as_a_static_question(self):
        question = self.fixture.questions[0]
        size = (question["image"]["width"], question["image"]["height"])
        path = Path(question["image"]["output_path"])
        Image.new("RGB", size, "white").save(path, "PNG", save_all=True,
                                             append_images=[Image.new("RGB", size, "black")])
        self.assertEqual(self.client.get(self.detail().json()["image"]["preview_url"]).status_code, 404)

    def test_openapi_exposes_only_upload_and_top_k(self):
        schema = self.client.get("/openapi.json").json()
        ref = schema["paths"]["/search/question"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"]
        self.assertEqual(set(schema["components"]["schemas"][ref.rsplit("/", 1)[-1]]["properties"]), {"file", "top_k"})

    def test_new_configuration_is_separate_and_respects_precedence(self):
        (self.root / ".env").write_text("IMG2ANSWER_QUESTION_CHROMA_COLLECTION=from_file\n"
                                         "IMG2ANSWER_IMAGE_ROOT=local/images\n", encoding="utf-8")
        with patch("img2answer.api.Path.cwd", return_value=self.root), patch.dict(os.environ, {}, clear=True):
            settings = ApiSettings.from_env()
            self.assertEqual(settings.question_chroma_collection, "from_file")
            self.assertIsNone(settings.chroma_collection)
            self.assertEqual(settings.image_root, Path("local/images"))
            self.assertNotIn("IMG2ANSWER_IMAGE_ROOT", os.environ)
            with patch.dict(os.environ, {"IMG2ANSWER_QUESTION_CHROMA_COLLECTION": "from_process"}):
                self.assertEqual(ApiSettings.from_env().question_chroma_collection, "from_process")


class QuestionEndToEndTests(unittest.TestCase):
    def test_extract_index_http_search_preview_reextract_and_reimport(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            fixture = QuestionFixture(root)
            legacy = ChromaImageVectorStore(root / "chroma")
            legacy.upsert_images([EmbeddedImage("legacy-page", [0.5] * 64,
                                                 {"document_id": "sample", "image_role": "crop_candidate"}, "page")])
            output = io.StringIO()
            with redirect_stdout(output):
                status = cli.main(["index-questions", "--database", str(fixture.store.path),
                                   "--document-id", "sample", "--chroma-dir", str(root / "chroma")])
            self.assertEqual(status, 0)
            self.assertIn("indexed=2", output.getvalue())
            client = TestClient(create_app(settings=ApiSettings(database=fixture.store.path, chroma_dir=root / "chroma",
                                                               image_root=root)))
            query = Path(fixture.questions[0]["image"]["output_path"]).read_bytes()
            def search():
                return client.post("/search/question", files={"file": ("question.png", query)}, data={"top_k": 2})
            response = search()
            self.assertEqual(response.status_code, 200)
            results = response.json()["results"]
            self.assertEqual(results[0]["question_id"], fixture.questions[0]["id"])
            self.assertEqual(results[0]["answer"], "A")
            self.assertIsNone(results[1]["answer"])
            old_preview = results[0]["image"]["preview_url"]
            self.assertEqual(client.get(old_preview).content, query)
            fixture.extract()
            self.assertEqual(search().json()["results"], [])
            self.assertEqual(client.get(old_preview).status_code, 409)
            index_questions(fixture.store.path, "sample", root / "chroma")
            self.assertEqual(len(search().json()["results"]), 2)
            fixture.import_run()
            self.assertEqual(search().json()["results"], [])
            self.assertEqual(client.get(old_preview).status_code, 404)
            self.assertEqual(index_questions(fixture.store.path, "sample", root / "chroma")["deleted"], 2)
            self.assertEqual(ChromaQuestionVectorStore(root / "chroma").count(), 0)
            self.assertEqual(legacy.collection.count(), 1)


if __name__ == "__main__":
    unittest.main()
