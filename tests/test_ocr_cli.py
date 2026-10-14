from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from img2answer import cli
from img2answer.ocr import OCRError
from img2answer.store import SQLiteStore

from helpers import create_sample_pdf
from test_ocr import RecordingOCRClient, settings


class OCRCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.pdf = self.root / "sample.pdf"
        create_sample_pdf(self.pdf)
        self.output = self.root / "processed"
        self.config = self.root / "sections.yml"
        self.config.write_text(
            json.dumps({"documents": {"sample": {"path": str(self.pdf), "sections": {"graphics": {"page_from": 1, "page_to": 2}}}}}),
            encoding="utf-8",
        )
        self.args = ["--config", str(self.config), "--output-dir", str(self.output), "--dpi", "96"]
        self.client = RecordingOCRClient()
        self.report_path = self.output / "reports" / "sample-report.json"

    def run_cli(self, *extra):
        with redirect_stdout(io.StringIO()):
            return cli.main([*self.args, *extra])

    def test_default_does_not_load_settings_or_sdk(self):
        with patch.object(cli.OCRSettings, "from_env", side_effect=AssertionError("not enabled")), patch.object(cli, "AliyunOCRClient") as factory:
            self.assertEqual(self.run_cli(), 0)
            factory.assert_not_called()
        self.assertNotIn("ocr_pages", json.loads(self.report_path.read_text(encoding="utf-8")))
        self.assertFalse((self.output / "ocr").exists())

    def test_cli_writes_report_and_sqlite_reference_then_uses_cache(self):
        with patch.object(cli.OCRSettings, "from_env", return_value=settings()), patch.object(cli, "AliyunOCRClient", return_value=self.client) as factory:
            self.assertEqual(self.run_cli("--ocr"), 0)
            factory.assert_called_once()
            first = json.loads(self.report_path.read_text(encoding="utf-8"))
            self.assertEqual(len(first["ocr_pages"]), 2)
            self.assertEqual([page["page_number"] for page in first["ocr_pages"]], [1, 2])
            for page in first["ocr_pages"]:
                self.assertEqual(page["section"], "graphics")
                self.assertFalse(page["cache_hit"])
                self.assertEqual(page["status"], "recognized")
                self.assertTrue(Path(page["result_path"]).is_file())
                self.assertTrue(Path(page["source_page"]).is_file())
            self.assertEqual(self.run_cli("--ocr"), 0)
            second = json.loads(self.report_path.read_text(encoding="utf-8"))
            self.assertTrue(all(page["cache_hit"] for page in second["ocr_pages"]))
            self.assertEqual(len(self.client.calls), 2)
            self.assertEqual(self.run_cli("--ocr", "--ocr-refresh"), 0)
            self.assertEqual(len(self.client.calls), 4)
        store = SQLiteStore(self.output / "img2answer.sqlite3")
        self.assertEqual(store.counts().process_reports, 1)
        self.assertEqual(store.fetch_all("process_reports")[0]["report_path"], str(self.report_path))
        self.assertEqual(store.counts().source_documents, 1)

    def test_refresh_requires_opt_in(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
            self.run_cli("--ocr-refresh")
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("必须与 --ocr", stderr.getvalue())
        self.assertFalse(self.output.exists())

    def test_later_page_range_keeps_original_pdf_page_number(self):
        self.config.write_text(
            json.dumps({"documents": {"sample": {"path": str(self.pdf), "sections": {"graphics": {"page_from": 2, "page_to": 2}}}}}),
            encoding="utf-8",
        )
        with patch.object(cli.OCRSettings, "from_env", return_value=settings()), patch.object(cli, "AliyunOCRClient", return_value=self.client):
            self.assertEqual(self.run_cli("--ocr"), 0)
        pages = json.loads(self.report_path.read_text(encoding="utf-8"))["ocr_pages"]
        self.assertEqual(pages[0]["page_number"], 2)
        self.assertTrue(pages[0]["result_path"].endswith("page-0002.json"))

    def test_later_document_failure_does_not_roll_back_completed_document(self):
        document = {"path": str(self.pdf), "sections": {"graphics": {"page_from": 1, "page_to": 1}}}
        self.config.write_text(json.dumps({"documents": {"first": document, "second": document}}), encoding="utf-8")
        real_recognize = self.client.recognize

        def fail_second_document(stream):
            if self.client.calls:
                raise OCRError("模拟后续文档失败")
            return real_recognize(stream)

        with patch.object(cli.OCRSettings, "from_env", return_value=settings()), patch.object(cli, "AliyunOCRClient", return_value=self.client):
            with patch.object(self.client, "recognize", side_effect=fail_second_document), redirect_stderr(io.StringIO()):
                self.assertEqual(self.run_cli("--ocr"), 1)
        self.assertTrue((self.output / "reports" / "first-report.json").exists())
        self.assertFalse((self.output / "reports" / "second-report.json").exists())
        store = SQLiteStore(self.output / "img2answer.sqlite3")
        self.assertEqual([row["id"] for row in store.fetch_all("source_documents")], ["first"])

    def test_configuration_errors_happen_before_side_effects(self):
        stderr = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), patch.object(Path, "cwd", return_value=self.root), redirect_stderr(stderr):
            self.assertEqual(self.run_cli("--ocr"), 1)
        self.assertIn("ALIBABA_CLOUD_ACCESS_KEY_ID", stderr.getvalue())
        self.assertFalse(self.output.exists())

    def test_failure_does_not_overwrite_existing_report_metadata_or_vectors(self):
        self.assertEqual(self.run_cli(), 0)
        store = SQLiteStore(self.output / "img2answer.sqlite3")
        before = {table: [dict(row) for row in store.fetch_all(table)] for table in ("source_documents", "question_images", "process_reports")}
        report = self.report_path.read_bytes()
        self.client.error = OCRError("模拟云端失败")
        with patch.object(cli.OCRSettings, "from_env", return_value=settings()), patch.object(cli, "AliyunOCRClient", return_value=self.client), patch.object(cli, "ChromaImageVectorStore"), patch.object(cli, "create_embedding_model"), patch.object(cli, "embed_question_images") as embed, redirect_stderr(io.StringIO()):
            self.assertEqual(self.run_cli("--ocr", "--embed-images"), 1)
            embed.assert_not_called()
        self.assertEqual(self.report_path.read_bytes(), report)
        for table, rows in before.items():
            self.assertEqual([dict(row) for row in store.fetch_all(table)], rows)
        self.assertEqual(len(self.client.calls), 1)

    def test_failure_retains_completed_page_and_next_run_resumes(self):
        real_recognize = self.client.recognize

        def fail_second_page(stream):
            if len(self.client.calls) == 1:
                raise OCRError("模拟第二页失败")
            return real_recognize(stream)

        with patch.object(cli.OCRSettings, "from_env", return_value=settings()), patch.object(cli, "AliyunOCRClient", return_value=self.client):
            with patch.object(self.client, "recognize", side_effect=fail_second_page), redirect_stderr(io.StringIO()):
                self.assertEqual(self.run_cli("--ocr"), 1)
            self.assertTrue((self.output / "ocr" / "sample" / "graphics" / "page-0001.json").exists())
            self.assertFalse(self.report_path.exists())
            self.assertEqual(SQLiteStore(self.output / "img2answer.sqlite3").counts().source_documents, 0)
            self.assertEqual(self.run_cli("--ocr"), 0)
        pages = json.loads(self.report_path.read_text(encoding="utf-8"))["ocr_pages"]
        self.assertTrue(pages[0]["cache_hit"])
        self.assertFalse(pages[1]["cache_hit"])
        self.assertEqual(len(self.client.calls), 2)


if __name__ == "__main__":
    unittest.main()
