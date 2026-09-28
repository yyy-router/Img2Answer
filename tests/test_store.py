from pathlib import Path
import tempfile
import unittest

from img2answer.config import SectionConfig
from img2answer.graphic_crop import crop_pages
from img2answer.pdf_document import inspect_pdf
from img2answer.render_pages import render_section
from img2answer.report import SectionReport, build_report, write_report
from img2answer.store import SQLiteStore

from helpers import create_sample_pdf


class StoreTests(unittest.TestCase):
    def test_store_persists_documents_images_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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
            store.upsert_source_document("sample", metadata)
            store.replace_crop_candidates(
                document_id="sample",
                section=section.name,
                candidates=candidates,
            )
            store.replace_crop_candidates(
                document_id="sample",
                section=section.name,
                candidates=candidates,
            )
            store.insert_process_report(document_id="sample", report_path=report_path, report=report)

            counts = store.counts()
            self.assertEqual(counts.source_documents, 1)
            self.assertEqual(counts.question_images, 1)
            self.assertEqual(counts.process_reports, 1)

            document = store.fetch_all("source_documents")[0]
            self.assertEqual(document["id"], "sample")
            self.assertEqual(document["page_count"], 2)

            image = store.fetch_all("question_images")[0]
            self.assertEqual(image["document_id"], "sample")
            self.assertEqual(image["image_role"], "crop_candidate")


if __name__ == "__main__":
    unittest.main()

