from pathlib import Path
import tempfile
import unittest

import pymupdf

from img2answer.config import SectionConfig
from img2answer.graphic_crop import crop_pages
from img2answer.pdf_document import inspect_pdf, sha256_file, validate_section
from img2answer.render_pages import render_section
from img2answer.report import SectionReport, build_report, write_report


class PdfPipelineTests(unittest.TestCase):
    def test_inspect_render_crop_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "sample.pdf"
            _create_sample_pdf(pdf_path)

            metadata = inspect_pdf(pdf_path)
            self.assertEqual(metadata.page_count, 2)
            self.assertEqual(metadata.sha256, sha256_file(pdf_path))

            section = SectionConfig(name="graphic_reasoning", page_from=1, page_to=1)
            validate_section(section, metadata.page_count)

            rendered_pages = render_section(pdf_path, section, root / "pages", dpi=96)
            self.assertEqual(len(rendered_pages), 1)
            self.assertTrue(rendered_pages[0].exists())

            candidates = crop_pages(rendered_pages, root / "crops", min_area_ratio=0.001)
            self.assertEqual(len(candidates), 1)
            self.assertTrue(candidates[0].output_path.exists())
            self.assertGreater(candidates[0].width, 10)
            self.assertGreater(candidates[0].height, 10)

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
            self.assertTrue(report_path.exists())
            self.assertEqual(report["processed_sections"][0]["crop_candidates"], 1)


def _create_sample_pdf(path: Path) -> None:
    document = pymupdf.open()
    page = document.new_page(width=240, height=180)
    page.draw_rect(pymupdf.Rect(60, 50, 180, 130), color=(0, 0, 0), width=2)
    page.draw_line(pymupdf.Point(60, 50), pymupdf.Point(180, 130), color=(0, 0, 0), width=2)
    page.insert_text(pymupdf.Point(30, 30), "1. graphic sample", fontsize=10)
    document.new_page(width=240, height=180)
    document.save(path)
    document.close()


if __name__ == "__main__":
    unittest.main()

