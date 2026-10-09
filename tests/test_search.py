import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from PIL import Image

from img2answer.search import fetch_question_image_detail, search_similar_images
from img2answer.store import SQLiteStore
from img2answer.vector_store import ImageSearchMatch, PillowHashEmbeddingModel

from helpers import create_sample_pdf
from img2answer.config import SectionConfig
from img2answer.graphic_crop import crop_pages
from img2answer.pdf_document import inspect_pdf
from img2answer.render_pages import render_section
from img2answer.report import SectionReport, build_report, write_report


class RecordingSearchStore:
    def __init__(self, matches: list[ImageSearchMatch]):
        self.matches = matches
        self.last_embedding: list[float] | None = None
        self.last_top_k: int | None = None

    def query_similar_images(self, embedding: list[float], top_k: int) -> list[ImageSearchMatch]:
        self.last_embedding = embedding
        self.last_top_k = top_k
        return self.matches[:top_k]


class FailingSearchStore:
    def query_similar_images(self, embedding: list[float], top_k: int) -> list[ImageSearchMatch]:
        raise AssertionError("vector search should not be called")


class RecordingEmbeddingModel:
    def __init__(self, embedding: list[float]):
        self.embedding = embedding
        self.called = False

    def embed_image(self, image_path: str | Path) -> list[float]:
        self.called = True
        return self.embedding


class FailingEmbeddingModel:
    def embed_image(self, image_path: str | Path) -> list[float]:
        raise AssertionError("embedding should not be called")


class ImageSearchTests(unittest.TestCase):
    def test_search_similar_images_returns_sqlite_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store, image_id = _create_store_with_one_image(root)
            query_image = root / "query.png"
            Image.new("RGB", (16, 16), "white").save(query_image)
            vector_store = RecordingSearchStore(
                [
                    ImageSearchMatch(
                        image_id=image_id,
                        distance=0.125,
                        metadata={
                            "image_id": image_id,
                            "document_id": "sample",
                            "section": "graphic_reasoning",
                            "image_role": "crop_candidate",
                            "output_path": "crop.png",
                        },
                        document="crop.png",
                    )
                ]
            )

            results = search_similar_images(
                query_image,
                sqlite_store=store,
                vector_store=vector_store,
                embedding_model=PillowHashEmbeddingModel(),
                top_k=1,
            )

            self.assertEqual(vector_store.last_top_k, 1)
            self.assertIsNotNone(vector_store.last_embedding)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].rank, 1)
            self.assertEqual(results[0].image_id, image_id)
            self.assertEqual(results[0].distance, 0.125)
            self.assertIsNotNone(results[0].record)
            self.assertEqual(results[0].record["id"], image_id)
            self.assertIsNotNone(results[0].detail)
            self.assertEqual(results[0].detail["image_id"], image_id)
            self.assertEqual(results[0].detail["document_id"], "sample")
            self.assertEqual(results[0].detail["section"], "graphic_reasoning")
            self.assertEqual(results[0].detail["bbox"], json.loads(results[0].record["bbox_json"]))
            self.assertEqual(results[0].detail["report_path"], str(root / "reports" / "sample-report.json"))

    def test_search_similar_images_keeps_missing_sqlite_record_visible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query_image = root / "query.png"
            Image.new("RGB", (16, 16), "white").save(query_image)
            store = SQLiteStore(root / "img2answer.sqlite3")
            store.initialize()
            vector_store = RecordingSearchStore(
                [
                    ImageSearchMatch(
                        image_id="missing",
                        distance=0.25,
                        metadata={"image_id": "missing", "document_id": "sample"},
                        document="missing.png",
                    )
                ]
            )

            results = search_similar_images(
                query_image,
                sqlite_store=store,
                vector_store=vector_store,
                embedding_model=PillowHashEmbeddingModel(),
                top_k=5,
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].image_id, "missing")
            self.assertIsNone(results[0].record)
            self.assertIsNone(results[0].detail)

    def test_fetch_question_image_detail_returns_report_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store, image_id = _create_store_with_one_image(root)

            detail = fetch_question_image_detail(store, image_id)

            self.assertIsNotNone(detail)
            self.assertEqual(detail.image_id, image_id)
            self.assertEqual(detail.document_id, "sample")
            self.assertEqual(detail.section, "graphic_reasoning")
            self.assertEqual(detail.image_role, "crop_candidate")
            self.assertEqual(detail.bbox, json.loads(store.fetch_question_image(image_id)["bbox_json"]))
            self.assertGreater(detail.width, 0)
            self.assertGreater(detail.height, 0)
            self.assertEqual(detail.report_path, str(root / "reports" / "sample-report.json"))

    def test_fetch_question_image_detail_allows_missing_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store, image_id = _create_store_with_one_image(root)
            connection = sqlite3.connect(store.path)
            try:
                connection.execute("DELETE FROM process_reports")
                connection.commit()
            finally:
                connection.close()

            detail = fetch_question_image_detail(store, image_id)

            self.assertIsNotNone(detail)
            self.assertIsNone(detail.report_path)

    def test_fetch_question_image_detail_rejects_invalid_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store, image_id = _create_store_with_one_image(root)
            connection = sqlite3.connect(store.path)
            try:
                connection.execute(
                    "UPDATE question_images SET bbox_json = ? WHERE id = ?",
                    ("not-json", image_id),
                )
                connection.commit()
            finally:
                connection.close()

            with self.assertRaisesRegex(ValueError, "invalid bbox_json"):
                fetch_question_image_detail(store, image_id)

    def test_search_similar_images_rejects_missing_query_image_before_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SQLiteStore(root / "img2answer.sqlite3")
            store.initialize()

            with self.assertRaisesRegex(FileNotFoundError, "query image does not exist"):
                search_similar_images(
                    root / "missing.png",
                    sqlite_store=store,
                    vector_store=FailingSearchStore(),
                    embedding_model=FailingEmbeddingModel(),
                    top_k=1,
                )

    def test_search_similar_images_rejects_zero_top_k_before_search(self) -> None:
        self._assert_invalid_top_k_rejected_before_search(0)

    def test_search_similar_images_rejects_negative_top_k_before_search(self) -> None:
        self._assert_invalid_top_k_rejected_before_search(-1)

    def test_search_similar_images_rejects_empty_embedding_before_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query_image = root / "query.png"
            Image.new("RGB", (16, 16), "white").save(query_image)
            store = SQLiteStore(root / "img2answer.sqlite3")
            store.initialize()
            embedding_model = RecordingEmbeddingModel([])

            with self.assertRaisesRegex(ValueError, "empty embedding"):
                search_similar_images(
                    query_image,
                    sqlite_store=store,
                    vector_store=FailingSearchStore(),
                    embedding_model=embedding_model,
                    top_k=1,
                )

            self.assertTrue(embedding_model.called)

    def _assert_invalid_top_k_rejected_before_search(self, top_k: int) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query_image = root / "query.png"
            Image.new("RGB", (16, 16), "white").save(query_image)
            store = SQLiteStore(root / "img2answer.sqlite3")
            store.initialize()
            embedding_model = FailingEmbeddingModel()

            with self.assertRaisesRegex(ValueError, "top_k must be greater than 0"):
                search_similar_images(
                    query_image,
                    sqlite_store=store,
                    vector_store=FailingSearchStore(),
                    embedding_model=embedding_model,
                    top_k=top_k,
                )


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
