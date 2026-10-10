import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from img2answer import cli as cli_module
from img2answer.cli import main
from img2answer.config import ConfigError
from img2answer.store import SQLiteStore
from img2answer.vector_store import EmbeddedImage, ImageSearchMatch

from helpers import create_sample_pdf


class RecordingChromaImageVectorStore:
    instances: list["RecordingChromaImageVectorStore"] = []

    def __init__(self, persist_dir: str | Path, collection_name: str = "question_images"):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.records: list[EmbeddedImage] = []
        self.deleted_documents: list[tuple[str, str | None]] = []
        self.__class__.instances.append(self)

    def delete_document_images(self, document_id: str, embedding_model_name: str | None = None) -> None:
        self.deleted_documents.append((document_id, embedding_model_name))

    def upsert_images(self, records: list[EmbeddedImage]) -> None:
        self.records.extend(records)


class RecordingSearchChromaImageVectorStore:
    matches: list[ImageSearchMatch] = []
    instances: list["RecordingSearchChromaImageVectorStore"] = []

    def __init__(self, persist_dir: str | Path, collection_name: str = "question_images"):
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.__class__.instances.append(self)

    def delete_document_images(self, document_id: str, embedding_model_name: str | None = None) -> None:
        raise AssertionError("search CLI should not delete vectors")

    def upsert_images(self, records: list[EmbeddedImage]) -> None:
        raise AssertionError("search CLI should not upsert vectors")

    def query_similar_images(self, embedding: list[float], top_k: int) -> list[ImageSearchMatch]:
        return self.matches[:top_k]


class ConstantEmbeddingModel:
    def embed_image(self, image_path: str | Path) -> list[float]:
        return [0.1, 0.2, 0.3]


class CliTests(unittest.TestCase):
    def test_cli_writes_report_and_sqlite_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "sample.pdf"
            create_sample_pdf(pdf_path)
            config_path = root / "sections.yml"
            output_dir = root / "processed"
            database_path = output_dir / "img2answer.sqlite3"
            config_path.write_text(
                f"""
documents:
  sample:
    path: {pdf_path.as_posix()}
    sections:
      graphic_reasoning:
        page_from: 1
        page_to: 1
""".strip(),
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(output_dir),
                    "--database",
                    str(database_path),
                    "--dpi",
                    "96",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "reports" / "sample-report.json").exists())
            store = SQLiteStore(database_path)
            counts = store.counts()
            self.assertEqual(counts.source_documents, 1)
            self.assertEqual(counts.question_images, 1)
            self.assertEqual(counts.process_reports, 1)

    def test_cli_defaults_database_to_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "sample.pdf"
            create_sample_pdf(pdf_path)
            config_path = root / "sections.yml"
            output_dir = root / "custom-output"
            config_path.write_text(
                f"""
documents:
  sample:
    path: {pdf_path.as_posix()}
    sections:
      graphic_reasoning:
        page_from: 1
        page_to: 1
""".strip(),
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(output_dir),
                    "--dpi",
                    "96",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "img2answer.sqlite3").exists())

    def test_cli_failed_run_does_not_write_partial_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "sample.pdf"
            create_sample_pdf(pdf_path)
            config_path = root / "sections.yml"
            output_dir = root / "processed"
            database_path = output_dir / "img2answer.sqlite3"
            config_path.write_text(
                f"""
documents:
  sample:
    path: {pdf_path.as_posix()}
    sections:
      graphic_reasoning:
        page_from: 1
        page_to: 3
""".strip(),
                encoding="utf-8",
            )

            with self.assertRaises(ConfigError):
                main(
                    [
                        "--config",
                        str(config_path),
                        "--output-dir",
                        str(output_dir),
                        "--database",
                        str(database_path),
                        "--dpi",
                        "96",
                    ]
                )

            store = SQLiteStore(database_path)
            counts = store.counts()
            self.assertEqual(counts.source_documents, 0)
            self.assertEqual(counts.question_images, 0)
            self.assertEqual(counts.process_reports, 0)

    def test_cli_can_embed_images_into_vector_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "sample.pdf"
            create_sample_pdf(pdf_path)
            config_path = root / "sections.yml"
            output_dir = root / "processed"
            config_path.write_text(
                f"""
documents:
  sample:
    path: {pdf_path.as_posix()}
    sections:
      graphic_reasoning:
        page_from: 1
        page_to: 1
""".strip(),
                encoding="utf-8",
            )
            RecordingChromaImageVectorStore.instances.clear()

            with patch.object(cli_module, "ChromaImageVectorStore", RecordingChromaImageVectorStore):
                exit_code = main(
                    [
                        "--config",
                        str(config_path),
                        "--output-dir",
                        str(output_dir),
                        "--dpi",
                        "96",
                        "--embed-images",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(RecordingChromaImageVectorStore.instances), 1)
            vector_store = RecordingChromaImageVectorStore.instances[0]
            self.assertEqual(vector_store.persist_dir, output_dir / "chroma")
            self.assertEqual(vector_store.collection_name, "question_images")
            self.assertEqual(vector_store.deleted_documents, [("sample", "pillow-hash")])
            self.assertEqual(len(vector_store.records), 1)
            self.assertEqual(vector_store.records[0].metadata["document_id"], "sample")
            self.assertEqual(vector_store.records[0].metadata["embedding_model"], "pillow-hash")

    def test_cli_passes_embedding_model_to_vector_ingestion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "sample.pdf"
            create_sample_pdf(pdf_path)
            config_path = root / "sections.yml"
            output_dir = root / "processed"
            config_path.write_text(
                f"""
documents:
  sample:
    path: {pdf_path.as_posix()}
    sections:
      graphic_reasoning:
        page_from: 1
        page_to: 1
""".strip(),
                encoding="utf-8",
            )
            created_models = []

            def create_model(name: str) -> ConstantEmbeddingModel:
                created_models.append(name)
                return ConstantEmbeddingModel()

            RecordingChromaImageVectorStore.instances.clear()
            with patch.object(cli_module, "ChromaImageVectorStore", RecordingChromaImageVectorStore):
                with patch.object(cli_module, "create_embedding_model", side_effect=create_model):
                    exit_code = main(
                        [
                            "--config",
                            str(config_path),
                            "--output-dir",
                            str(output_dir),
                            "--dpi",
                            "96",
                            "--embed-images",
                            "--embedding-model",
                            "openclip",
                        ]
                    )

            self.assertEqual(exit_code, 0)
            self.assertEqual(created_models, ["openclip"])
            self.assertEqual(len(RecordingChromaImageVectorStore.instances), 1)
            self.assertEqual(RecordingChromaImageVectorStore.instances[0].collection_name, "question_images_openclip")
            self.assertEqual(RecordingChromaImageVectorStore.instances[0].deleted_documents, [("sample", "openclip")])

    def test_cli_search_image_outputs_json_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "sample.pdf"
            create_sample_pdf(pdf_path)
            config_path = root / "sections.yml"
            output_dir = root / "processed"
            config_path.write_text(
                f"""
documents:
  sample:
    path: {pdf_path.as_posix()}
    sections:
      graphic_reasoning:
        page_from: 1
        page_to: 1
""".strip(),
                encoding="utf-8",
            )
            self.assertEqual(
                main(
                    [
                        "--config",
                        str(config_path),
                        "--output-dir",
                        str(output_dir),
                        "--dpi",
                        "96",
                    ]
                ),
                0,
            )
            store = SQLiteStore(output_dir / "img2answer.sqlite3")
            image = store.fetch_question_images()[0]
            RecordingSearchChromaImageVectorStore.matches = [
                ImageSearchMatch(
                    image_id=image["id"],
                    distance=0.25,
                    metadata={
                        "image_id": image["id"],
                        "document_id": image["document_id"],
                        "section": image["section"],
                    },
                    document=image["output_path"],
                )
            ]
            stdout = io.StringIO()
            created_models = []

            def create_model(name: str) -> ConstantEmbeddingModel:
                created_models.append(name)
                return ConstantEmbeddingModel()

            RecordingSearchChromaImageVectorStore.instances.clear()
            with patch.object(cli_module, "ChromaImageVectorStore", RecordingSearchChromaImageVectorStore):
                with patch.object(cli_module, "create_embedding_model", side_effect=create_model):
                    with redirect_stdout(stdout):
                        exit_code = main(
                            [
                                "search-image",
                                "--image",
                                image["output_path"],
                                "--database",
                                str(output_dir / "img2answer.sqlite3"),
                                "--chroma-dir",
                                str(output_dir / "chroma"),
                                "--top-k",
                                "1",
                                "--embedding-model",
                                "openclip",
                            ]
                        )

            self.assertEqual(exit_code, 0)
            self.assertEqual(created_models, ["openclip"])
            self.assertEqual(RecordingSearchChromaImageVectorStore.instances[-1].collection_name, "question_images_openclip")
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["top_k"], 1)
            self.assertEqual(payload["embedding_model"], "openclip")
            self.assertEqual(len(payload["results"]), 1)
            self.assertEqual(payload["results"][0]["image_id"], image["id"])
            self.assertEqual(payload["results"][0]["record"]["id"], image["id"])
            self.assertEqual(payload["results"][0]["detail"]["image_id"], image["id"])
            self.assertEqual(payload["results"][0]["detail"]["document_id"], "sample")
            self.assertEqual(payload["results"][0]["detail"]["section"], "graphic_reasoning")
            self.assertEqual(payload["results"][0]["detail"]["report_path"], str(output_dir / "reports" / "sample-report.json"))


if __name__ == "__main__":
    unittest.main()
