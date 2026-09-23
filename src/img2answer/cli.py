from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .graphic_crop import crop_pages
from .pdf_document import inspect_pdf, validate_section
from .render_pages import render_section
from .report import SectionReport, build_report, write_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render configured PDF sections and create crop candidates.")
    parser.add_argument("--config", required=True, help="Path to a JSON/YAML section config.")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory for local artifacts.")
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI.")
    args = parser.parse_args(argv)

    project_config = load_config(args.config)
    output_root = Path(args.output_dir)

    for document_config in project_config.documents:
        metadata = inspect_pdf(document_config.path)
        section_reports: list[SectionReport] = []
        all_candidates = []

        for section in document_config.sections:
            validate_section(section, metadata.page_count)
            page_dir = output_root / "pages" / document_config.document_id / section.name
            crop_dir = output_root / "crops" / document_config.document_id / section.name

            rendered_pages = render_section(metadata.path, section, page_dir, dpi=args.dpi)
            candidates = crop_pages(rendered_pages, crop_dir)
            all_candidates.extend(candidates)
            section_reports.append(
                SectionReport(
                    section=section.name,
                    page_from=section.page_from,
                    page_to=section.page_to,
                    rendered_pages=len(rendered_pages),
                    crop_candidates=len(candidates),
                )
            )

        report = build_report(
            document_id=document_config.document_id,
            metadata=metadata,
            sections=section_reports,
            crop_candidates=all_candidates,
        )
        report_path = output_root / "reports" / f"{document_config.document_id}-report.json"
        write_report(report, report_path)
        print(report_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

