from pathlib import Path
import tempfile
import unittest

from img2answer.cli import main
from img2answer.config import ConfigError
from img2answer.store import SQLiteStore

from helpers import create_sample_pdf


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


if __name__ == "__main__":
    unittest.main()
