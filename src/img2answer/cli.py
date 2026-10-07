from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .graphic_crop import crop_pages
from .pdf_document import inspect_pdf, validate_section
from .render_pages import render_section
from .report import SectionReport, build_report, write_report
from .store import SQLiteStore
from .vector_store import ChromaImageVectorStore, PillowHashEmbeddingModel, embed_question_images


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render configured PDF sections and create crop candidates.")
    parser.add_argument("--config", required=True, help="Path to a JSON/YAML section config.")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory for local artifacts.")
    parser.add_argument(
        "--database",
        default=None,
        help="SQLite database path for local metadata. Defaults to <output-dir>/img2answer.sqlite3.",
    )
    parser.add_argument("--embed-images", action="store_true", help="Write crop candidate image vectors to ChromaDB.")
    parser.add_argument(
        "--chroma-dir",
        default=None,
        help="ChromaDB persistence directory. Defaults to <output-dir>/chroma.",
    )
    parser.add_argument(
        "--chroma-collection",
        default="question_images",
        help="ChromaDB collection name for image vectors.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI.")
    args = parser.parse_args(argv)

    project_config = load_config(args.config)
    output_root = Path(args.output_dir)
    database_path = Path(args.database) if args.database else output_root / "img2answer.sqlite3"
    store = SQLiteStore(database_path)
    store.initialize()
    vector_store = None
    embedding_model = None
    if args.embed_images:
        chroma_dir = Path(args.chroma_dir) if args.chroma_dir else output_root / "chroma"
        vector_store = ChromaImageVectorStore(chroma_dir, collection_name=args.chroma_collection)
        embedding_model = PillowHashEmbeddingModel()

    for document_config in project_config.documents:
        metadata = inspect_pdf(document_config.path)
        section_reports: list[SectionReport] = []
        all_candidates = []
        section_candidates = {}

        for section in document_config.sections:
            validate_section(section, metadata.page_count)
            page_dir = output_root / "pages" / document_config.document_id / section.name
            crop_dir = output_root / "crops" / document_config.document_id / section.name

            rendered_pages = render_section(metadata.path, section, page_dir, dpi=args.dpi)
            candidates = crop_pages(rendered_pages, crop_dir)
            all_candidates.extend(candidates)
            section_candidates[section.name] = candidates
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
        store.replace_document_run(
            document_id=document_config.document_id,
            metadata=metadata,
            section_candidates=section_candidates,
            report_path=report_path,
            report=report,
        )
        print(report_path)
        if vector_store is not None and embedding_model is not None:
            image_rows = store.fetch_question_images(document_config.document_id)
            embedding_result = embed_question_images(
                document_config.document_id,
                image_rows,
                vector_store=vector_store,
                embedding_model=embedding_model,
            )
            print(
                "image_vectors "
                f"document_id={document_config.document_id} "
                f"embedded={embedding_result.embedded} "
                f"skipped_missing_files={embedding_result.skipped_missing_files}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

