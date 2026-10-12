import io
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import tempfile
from threading import Barrier
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image

from img2answer.api import ApiSettings, create_app
from img2answer.config import SectionConfig
from img2answer.graphic_crop import crop_pages
from img2answer.pdf_document import inspect_pdf
from img2answer.render_pages import render_section
from img2answer.report import SectionReport, build_report, write_report
from img2answer.search import fetch_question_image_detail
from img2answer.store import SQLiteStore
from img2answer.vector_store import (
    ChromaImageVectorStore,
    ImageSearchMatch,
    PillowHashEmbeddingModel,
    embed_question_images,
)

from helpers import create_sample_pdf


class RecordingSearchVectorStore:
    matches: list[ImageSearchMatch] = []
    instances: list["RecordingSearchVectorStore"] = []

    def __init__(self, persist_dir: str | Path, collection_name: str = "question_images"):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.last_embedding: list[float] | None = None
        self.last_top_k: int | None = None
        self.__class__.instances.append(self)

    def query_similar_images(self, embedding: list[float], top_k: int) -> list[ImageSearchMatch]:
        self.last_embedding = embedding
        self.last_top_k = top_k
        return self.matches[:top_k]


class RecordingEmbeddingModel:
    def __init__(self, error: Exception | None = None):
        self.paths: list[Path] = []
        self.contents: list[bytes] = []
        self.error = error

    def embed_image(self, image_path: str | Path) -> list[float]:
        path = Path(image_path)
        self.paths.append(path)
        self.contents.append(path.read_bytes())
        if self.error is not None:
            raise self.error
        return [0.1, 0.2, 0.3]


class ApiSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        cwd_patch = patch("img2answer.api.Path.cwd", return_value=self.root)
        cwd_patch.start()
        self.addCleanup(cwd_patch.stop)

    def test_environment_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = ApiSettings.from_env()

        self.assertEqual(settings.database, Path("data/processed/img2answer.sqlite3"))
        self.assertEqual(settings.chroma_dir, Path("data/processed/chroma"))
        self.assertIsNone(settings.chroma_collection)
        self.assertEqual(settings.embedding_model, "pillow-hash")
        self.assertEqual(settings.top_k, 5)
        self.assertEqual(settings.max_upload_bytes, 10 * 1024 * 1024)

    def test_upload_limit_configuration_from_file_and_environment(self) -> None:
        (self.root / ".env").write_text("IMG2ANSWER_MAX_UPLOAD_BYTES=2048\n", encoding="utf-8")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ApiSettings.from_env().max_upload_bytes, 2048)
        with patch.dict(os.environ, {"IMG2ANSWER_MAX_UPLOAD_BYTES": "4096"}, clear=True):
            self.assertEqual(ApiSettings.from_env().max_upload_bytes, 4096)

    def test_rejects_invalid_upload_limit_configuration(self) -> None:
        for value in ("0", "-1", "abc"):
            with self.subTest(value=value), patch.dict(os.environ, {
                "IMG2ANSWER_MAX_UPLOAD_BYTES": value,
            }, clear=True):
                with self.assertRaises(ValueError):
                    ApiSettings.from_env()
        with self.assertRaises(ValueError):
            ApiSettings(max_upload_bytes=0)

    def test_create_app_reads_dotenv_without_changing_process_environment(self) -> None:
        (self.root / ".env").write_text(
            "IMG2ANSWER_DATABASE=private/questions.sqlite3\n"
            "IMG2ANSWER_CHROMA_DIR=private/vectors\n"
            "IMG2ANSWER_CHROMA_COLLECTION=custom_images\n"
            "IMG2ANSWER_EMBEDDING_MODEL=openclip\n"
            "IMG2ANSWER_TOP_K=3\n",
            encoding="utf-8",
        )
        with patch.dict(os.environ, {}, clear=True):
            settings = create_app().state.settings
            self.assertNotIn("IMG2ANSWER_DATABASE", os.environ)
            self.assertNotIn("IMG2ANSWER_EMBEDDING_MODEL", os.environ)

        self.assertEqual(settings.database, Path("private/questions.sqlite3"))
        self.assertEqual(settings.chroma_dir, Path("private/vectors"))
        self.assertEqual(settings.chroma_collection, "custom_images")
        self.assertEqual(settings.embedding_model, "openclip")
        self.assertEqual(settings.top_k, 3)

    def test_process_environment_overrides_dotenv_values(self) -> None:
        (self.root / ".env").write_text(
            "IMG2ANSWER_DATABASE=file/questions.sqlite3\n"
            "IMG2ANSWER_CHROMA_DIR=file/vectors\n"
            "IMG2ANSWER_CHROMA_COLLECTION=file_images\n"
            "IMG2ANSWER_EMBEDDING_MODEL=openclip\n"
            "IMG2ANSWER_TOP_K=3\n",
            encoding="utf-8",
        )
        with patch.dict(os.environ, {
            "IMG2ANSWER_DATABASE": "environment/questions.sqlite3",
            "IMG2ANSWER_CHROMA_DIR": "environment/vectors",
            "IMG2ANSWER_CHROMA_COLLECTION": "environment_images",
            "IMG2ANSWER_EMBEDDING_MODEL": "pillow-hash",
            "IMG2ANSWER_TOP_K": "2",
        }, clear=True):
            settings = ApiSettings.from_env()

        self.assertEqual(settings.database, Path("environment/questions.sqlite3"))
        self.assertEqual(settings.chroma_dir, Path("environment/vectors"))
        self.assertEqual(settings.chroma_collection, "environment_images")
        self.assertEqual(settings.embedding_model, "pillow-hash")
        self.assertEqual(settings.top_k, 2)

    def test_empty_dotenv_uses_defaults(self) -> None:
        (self.root / ".env").touch()
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ApiSettings.from_env(), ApiSettings())

    def test_partial_dotenv_supports_quotes_comments_unicode_and_bom(self) -> None:
        (self.root / ".env").write_text(
            '# Local paths\nIMG2ANSWER_DATABASE="data/\u9898\u5e93 sample/questions.sqlite3"\n'
            "IMG2ANSWER_TOP_K=2 # result count\n"
            "IMG2ANSWER_CHROMA_COLLECTION=\n"
            "IMG2ANSWER_CHROMA_DIR\n",
            encoding="utf-8-sig",
        )
        with patch.dict(os.environ, {}, clear=True):
            settings = ApiSettings.from_env()

        self.assertEqual(settings.database, Path("data/\u9898\u5e93 sample/questions.sqlite3"))
        self.assertEqual(settings.top_k, 2)
        self.assertEqual(settings.chroma_dir, ApiSettings().chroma_dir)
        self.assertIsNone(settings.chroma_collection)
        self.assertEqual(settings.embedding_model, "pillow-hash")

    def test_dotenv_changes_are_read_on_next_app_creation(self) -> None:
        env_path = self.root / ".env"
        with patch.dict(os.environ, {}, clear=True):
            env_path.write_text("IMG2ANSWER_TOP_K=2\n", encoding="utf-8")
            first_settings = create_app().state.settings
            env_path.write_text("IMG2ANSWER_TOP_K=3\n", encoding="utf-8")
            second_settings = create_app().state.settings

        self.assertEqual(first_settings.top_k, 2)
        self.assertEqual(second_settings.top_k, 3)

    def test_explicit_settings_bypass_invalid_dotenv(self) -> None:
        (self.root / ".env").write_text("IMG2ANSWER_TOP_K=invalid\n", encoding="utf-8")
        settings = ApiSettings(top_k=2)

        self.assertIs(create_app(settings=settings).state.settings, settings)

    def test_rejects_invalid_values_in_dotenv(self) -> None:
        for value in ("IMG2ANSWER_TOP_K=0", "IMG2ANSWER_TOP_K=-1", "IMG2ANSWER_TOP_K=abc",
                      "IMG2ANSWER_EMBEDDING_MODEL=unknown"):
            with self.subTest(value=value), patch.dict(os.environ, {}, clear=True):
                (self.root / ".env").write_text(value + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    create_app()

    def test_create_app_reads_environment_settings(self) -> None:
        with patch.dict(os.environ, {
            "IMG2ANSWER_DATABASE": "private/questions.sqlite3",
            "IMG2ANSWER_CHROMA_DIR": "private/vectors",
            "IMG2ANSWER_CHROMA_COLLECTION": "custom_images",
            "IMG2ANSWER_EMBEDDING_MODEL": "openclip",
            "IMG2ANSWER_TOP_K": "3",
        }, clear=True):
            settings = create_app().state.settings

        self.assertEqual(settings.database, Path("private/questions.sqlite3"))
        self.assertEqual(settings.chroma_dir, Path("private/vectors"))
        self.assertEqual(settings.chroma_collection, "custom_images")
        self.assertEqual(settings.embedding_model, "openclip")
        self.assertEqual(settings.top_k, 3)

    def test_rejects_invalid_server_settings(self) -> None:
        for top_k in ("0", "-1", "abc"):
            with self.subTest(top_k=top_k), patch.dict(os.environ, {
                "IMG2ANSWER_TOP_K": top_k,
            }, clear=True):
                with self.assertRaises(ValueError):
                    create_app()
        with self.assertRaises(ValueError):
            ApiSettings(embedding_model="unknown")


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        RecordingSearchVectorStore.matches = []
        RecordingSearchVectorStore.instances.clear()

    def test_model_is_loaded_once_for_sequential_requests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / "images.sqlite3")
            store.initialize()
            model = RecordingEmbeddingModel()
            factory = Mock(return_value=model)
            client = TestClient(create_app(
                settings=ApiSettings(database=store.path, embedding_model="openclip"),
                vector_store_factory=RecordingSearchVectorStore,
                embedding_model_factory=factory,
            ))
            factory.assert_not_called()

            for _ in range(2):
                response = client.post(
                    "/search/image", files={"file": ("query.png", _image_bytes(), "image/png")},
                )
                self.assertEqual(response.status_code, 200)

            factory.assert_called_once_with("openclip")
            self.assertEqual(len(model.paths), 2)
            self.assertTrue(all(not path.exists() for path in model.paths))

    def test_model_is_loaded_once_for_concurrent_requests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / "images.sqlite3")
            store.initialize()
            factory = Mock(side_effect=lambda name: RecordingEmbeddingModel())
            app = create_app(
                settings=ApiSettings(database=store.path, embedding_model="openclip"),
                vector_store_factory=RecordingSearchVectorStore,
                embedding_model_factory=factory,
            )
            barrier = Barrier(4, timeout=10)
            with TestClient(app) as client:
                def search(_):
                    barrier.wait()
                    return client.post(
                        "/search/image", files={"file": ("query.png", _image_bytes(), "image/png")},
                    ).status_code

                with ThreadPoolExecutor(max_workers=4) as executor:
                    statuses = list(executor.map(search, range(4)))

            self.assertEqual(statuses, [200] * 4)
            factory.assert_called_once_with("openclip")

    def test_model_initialization_failure_can_be_retried(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / "images.sqlite3")
            store.initialize()
            factory = Mock(side_effect=[RuntimeError("load failed"), RecordingEmbeddingModel()])
            client = TestClient(create_app(
                settings=ApiSettings(database=store.path, embedding_model="openclip"),
                vector_store_factory=RecordingSearchVectorStore,
                embedding_model_factory=factory,
            ))
            with self.assertLogs("img2answer.api", level="ERROR"):
                response = client.post(
                    "/search/image", files={"file": ("query.png", _image_bytes(), "image/png")},
                )
            self.assertEqual(response.status_code, 500)
            for _ in range(2):
                response = client.post(
                    "/search/image", files={"file": ("query.png", _image_bytes(), "image/png")},
                )
                self.assertEqual(response.status_code, 200)
            self.assertEqual(factory.call_count, 2)

    def test_model_cache_is_not_shared_between_apps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / "images.sqlite3")
            store.initialize()
            models = [RecordingEmbeddingModel(), RecordingEmbeddingModel()]
            factory = Mock(side_effect=models)
            for _ in range(2):
                client = TestClient(create_app(
                    settings=ApiSettings(database=store.path, embedding_model="openclip"),
                    vector_store_factory=RecordingSearchVectorStore,
                    embedding_model_factory=factory,
                ))
                response = client.post(
                    "/search/image", files={"file": ("query.png", _image_bytes(), "image/png")},
                )
                self.assertEqual(response.status_code, 200)
            self.assertEqual(factory.call_count, 2)
            self.assertEqual([len(model.paths) for model in models], [1, 1])

    def test_search_image_returns_results_and_cleans_upload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store, image_id = _create_store_with_one_image(root)
            counts_before = store.counts()
            image_bytes = _image_bytes()
            model = RecordingEmbeddingModel()
            model_factory = Mock(return_value=model)
            RecordingSearchVectorStore.matches = [
                ImageSearchMatch(
                    image_id=image_id,
                    distance=0.25,
                    metadata={"image_id": image_id, "document_id": "sample"},
                    document="crop.png",
                )
            ]
            client = TestClient(create_app(
                settings=ApiSettings(
                    database=store.path,
                    chroma_dir=root / "chroma",
                    embedding_model="openclip",
                ),
                vector_store_factory=RecordingSearchVectorStore,
                embedding_model_factory=model_factory,
            ))

            response = client.post(
                "/search/image",
                files={"file": ("query.png", image_bytes, "image/png")},
                data={"top_k": "1"},
            )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["query_image"], "query.png")
            self.assertEqual(payload["top_k"], 1)
            self.assertEqual(payload["embedding_model"], "openclip")
            self.assertEqual(payload["chroma_collection"], "question_images_openclip")
            self.assertEqual(len(payload["results"]), 1)
            self.assertEqual(payload["results"][0]["image_id"], image_id)
            self.assertEqual(payload["results"][0]["record"]["id"], image_id)
            self.assertEqual(payload["results"][0]["detail"]["image_id"], image_id)
            vector_store = RecordingSearchVectorStore.instances[0]
            self.assertEqual(vector_store.persist_dir, root / "chroma")
            self.assertEqual(vector_store.collection_name, "question_images_openclip")
            self.assertEqual(vector_store.last_embedding, [0.1, 0.2, 0.3])
            self.assertEqual(vector_store.last_top_k, 1)
            model_factory.assert_called_once_with("openclip")
            self.assertEqual(model.contents, [image_bytes])
            self.assertEqual(len(model.paths), 1)
            self.assertFalse(model.paths[0].parent.exists())
            self.assertNotIn(str(model.paths[0]), response.text)
            self.assertEqual(store.counts(), counts_before)

    def test_server_defaults_cannot_be_overridden_by_request_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SQLiteStore(root / "img2answer.sqlite3")
            store.initialize()
            model_factory = Mock(return_value=RecordingEmbeddingModel())
            client = TestClient(create_app(
                settings=ApiSettings(
                    database=store.path,
                    chroma_dir=root / "chroma",
                    chroma_collection="custom_images",
                    embedding_model="openclip",
                    top_k=3,
                ),
                vector_store_factory=RecordingSearchVectorStore,
                embedding_model_factory=model_factory,
            ))

            response = client.post(
                "/search/image",
                files={"file": ("query.png", _image_bytes(), "image/png")},
                data={
                    "image_path": str(root / "missing.png"),
                    "database": str(root / "client.sqlite3"),
                    "chroma_dir": str(root / "client-chroma"),
                    "chroma_collection": "client_images",
                    "embedding_model": "pillow-hash",
                },
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["top_k"], 3)
            self.assertEqual(response.json()["results"], [])
            self.assertEqual(response.json()["embedding_model"], "openclip")
            self.assertEqual(response.json()["chroma_collection"], "custom_images")
            self.assertEqual(RecordingSearchVectorStore.instances[0].persist_dir, root / "chroma")
            model_factory.assert_called_once_with("openclip")
            self.assertFalse((root / "client.sqlite3").exists())
            self.assertFalse((root / "client-chroma").exists())

    def test_search_requires_upload_instead_of_old_json_request(self) -> None:
        client = TestClient(create_app(settings=ApiSettings()))

        response = client.post("/search/image", json={"image_path": "query.png"})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"][0]["loc"], ["body", "file"])

    def test_search_rejects_invalid_top_k_before_loading_backends(self) -> None:
        model_factory = Mock()
        client = TestClient(create_app(
            settings=ApiSettings(),
            embedding_model_factory=model_factory,
        ))
        for top_k, status in (("0", 400), ("-1", 400), ("abc", 422)):
            with self.subTest(top_k=top_k):
                response = client.post(
                    "/search/image",
                    files={"file": ("query.png", _image_bytes(), "image/png")},
                    data={"top_k": top_k},
                )
                self.assertEqual(response.status_code, status)
        model_factory.assert_not_called()
        self.assertEqual(RecordingSearchVectorStore.instances, [])

    def test_search_rejects_empty_invalid_and_corrupt_uploads_and_cleans_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            created_dirs: list[Path] = []
            temporary_directory = tempfile.TemporaryDirectory

            def make_query_directory(**kwargs):
                temporary = temporary_directory(dir=root, **kwargs)
                created_dirs.append(Path(temporary.name))
                return temporary

            model_factory = Mock()
            client = TestClient(create_app(
                settings=ApiSettings(),
                embedding_model_factory=model_factory,
            ))
            for content in (b"", b"not an image", _image_bytes()[:40]):
                with self.subTest(content=content), patch(
                    "img2answer.api.tempfile.TemporaryDirectory", side_effect=make_query_directory,
                ):
                    response = client.post(
                        "/search/image",
                        files={"file": ("query.png", content, "image/png")},
                    )
                    self.assertEqual(response.status_code, 400)
            model_factory.assert_not_called()
            self.assertEqual(len(created_dirs), 3)
            self.assertTrue(all(not path.exists() for path in created_dirs))

    def test_upload_filename_and_mime_type_do_not_determine_server_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SQLiteStore(root / "img2answer.sqlite3")
            store.initialize()
            model = RecordingEmbeddingModel()
            client = TestClient(create_app(
                settings=ApiSettings(database=store.path, chroma_dir=root / "chroma"),
                vector_store_factory=RecordingSearchVectorStore,
                embedding_model_factory=lambda name: model,
            ))

            response = client.post(
                "/search/image",
                files={"file": ("../../query.txt", _image_bytes(), "text/plain")},
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["query_image"], "query.txt")
            self.assertEqual(model.paths[0].name, "query-image")
            self.assertFalse(model.paths[0].exists())

    def test_backend_failure_cleans_upload_and_hides_internal_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SQLiteStore(root / "img2answer.sqlite3")
            store.initialize()
            model = RecordingEmbeddingModel(error=RuntimeError(f"backend failed at {root / 'chroma'}"))
            client = TestClient(create_app(
                settings=ApiSettings(database=store.path, chroma_dir=root / "chroma"),
                vector_store_factory=RecordingSearchVectorStore,
                embedding_model_factory=lambda name: model,
            ))

            with self.assertLogs("img2answer.api", level="ERROR"):
                response = client.post(
                    "/search/image",
                    files={"file": ("query.png", _image_bytes(), "image/png")},
                )

            self.assertEqual(response.status_code, 500)
            self.assertNotIn(str(root), response.text)
            self.assertNotIn(str(model.paths[0]), response.text)
            self.assertFalse(model.paths[0].parent.exists())

    def test_image_detail_uses_server_database_without_query_parameter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store, image_id = _create_store_with_one_image(root)
            client = TestClient(create_app(settings=ApiSettings(database=store.path)))

            response = client.get(f"/images/{image_id}")
            override_response = client.get(
                f"/images/{image_id}", params={"database": str(root / "client.sqlite3")},
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(override_response.json(), response.json())
            payload = response.json()
            expected = fetch_question_image_detail(store, image_id)
            self.assertIsNotNone(expected)
            self.assertEqual(payload["image_id"], image_id)
            self.assertEqual(payload["document_id"], "sample")
            self.assertEqual(payload["bbox"], expected.bbox)
            self.assertEqual(payload["report_path"], expected.report_path)
            self.assertFalse((root / "client.sqlite3").exists())

    def test_image_detail_returns_404_for_missing_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteStore(Path(directory) / "img2answer.sqlite3")
            store.initialize()
            client = TestClient(create_app(settings=ApiSettings(database=store.path)))

            response = client.get("/images/missing")

            self.assertEqual(response.status_code, 404)

    def test_missing_server_database_is_not_created_or_exposed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "missing.sqlite3"
            client = TestClient(create_app(settings=ApiSettings(database=database)))

            response = client.get("/images/missing")
            search_response = client.post(
                "/search/image",
                files={"file": ("query.png", _image_bytes(), "image/png")},
            )

            for result in (response, search_response):
                self.assertEqual(result.status_code, 503)
                self.assertNotIn(str(database), result.text)
            self.assertFalse(database.exists())

    def test_openapi_exposes_only_public_request_fields(self) -> None:
        schema = create_app(settings=ApiSettings()).openapi()
        search = schema["paths"]["/search/image"]["post"]
        self.assertNotIn("parameters", search)
        body = search["requestBody"]["content"]["multipart/form-data"]["schema"]
        component = schema["components"]["schemas"][body["$ref"].rsplit("/", 1)[-1]]
        self.assertEqual(set(component["properties"]), {"file", "top_k"})
        self.assertEqual(component["required"], ["file"])
        detail = schema["paths"]["/images/{image_id}"]["get"]
        self.assertEqual([parameter["name"] for parameter in detail["parameters"]], ["image_id"])

    def test_uploaded_crop_searches_real_chroma_without_indexing_query(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            store, image_id = _create_store_with_one_image(root)
            vector_store = ChromaImageVectorStore(root / "chroma")
            embedding_model = PillowHashEmbeddingModel()
            embed_question_images(
                "sample", store.fetch_question_images(),
                vector_store=vector_store, embedding_model=embedding_model,
            )
            original_count = vector_store.collection.count()
            crop_path = Path(store.fetch_question_image(image_id)["output_path"])
            client = TestClient(create_app(
                settings=ApiSettings(database=store.path, chroma_dir=root / "chroma"),
                vector_store_factory=lambda path, **kwargs: vector_store,
                embedding_model_factory=lambda name: embedding_model,
            ))

            response = client.post(
                "/search/image",
                files={"file": ("crop.png", crop_path.read_bytes(), "image/png")},
                data={"top_k": "1"},
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["results"][0]["image_id"], image_id)
            self.assertAlmostEqual(response.json()["results"][0]["distance"], 0.0, places=5)
            self.assertEqual(vector_store.collection.count(), original_count)


def _image_bytes() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(stream, format="PNG")
    return stream.getvalue()


def _create_store_with_one_image(root: Path) -> tuple[SQLiteStore, str]:
    pdf_path = root / "sample.pdf"
    create_sample_pdf(pdf_path)
    metadata = inspect_pdf(pdf_path)
    section = SectionConfig(name="graphic_reasoning", page_from=1, page_to=1)
    rendered_pages = render_section(pdf_path, section, root / "pages", dpi=96)
    candidates = crop_pages(rendered_pages, root / "crops", min_area_ratio=0.001)
    report = build_report(
        document_id="sample",
        metadata=metadata,
        sections=[
            SectionReport(
                section=section.name,
                page_from=section.page_from,
                page_to=section.page_to,
                rendered_pages=len(rendered_pages),
                crop_candidates=len(candidates),
            )
        ],
        crop_candidates=candidates,
    )
    report_path = write_report(report, root / "reports" / "sample-report.json")
    store = SQLiteStore(root / "img2answer.sqlite3")
    store.initialize()
    store.replace_document_run(
        document_id="sample",
        metadata=metadata,
        section_candidates={section.name: candidates},
        report_path=report_path,
        report=report,
    )
    return store, store.fetch_question_images()[0]["id"]


if __name__ == "__main__":
    unittest.main()
