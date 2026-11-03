from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

from .config import SectionConfig, load_config
from .graphic_crop import crop_pages
from .ocr import AliyunOCRClient, OCRError, OCRSettings, recognize_page
from .pdf_document import inspect_pdf, validate_section
from .render_pages import render_section
from .report import SectionReport, build_report, write_report
from .search import search_similar_images
from .store import SQLiteStore
from .vector_store import ChromaImageVectorStore, create_embedding_model, embed_question_images


DEFAULT_CHROMA_COLLECTION = "question_images"


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] == "extract-questions":
        return extract_questions_main(args_list[1:])
    if args_list and args_list[0] == "index-questions":
        return index_questions_main(args_list[1:])
    if args_list and args_list[0] == "search-image":
        return search_image_main(args_list[1:])
    try:
        return process_documents_main(args_list)
    except OCRError as exc:
        print(
            f"OCR 处理失败：{exc}。导入已停止；当前文档本次元数据未写入，"
            "此前成功的文档、页面缓存及渲染文件不会回滚。",
            file=sys.stderr,
        )
        return 1


def process_documents_main(argv: list[str]) -> int:
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
        default=None,
        help=(
            "ChromaDB collection name for image vectors. Defaults to question_images for pillow-hash "
            "and question_images_openclip for openclip."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default="pillow-hash",
        help="Image embedding model to use when --embed-images is enabled: pillow-hash or openclip.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI.")
    parser.add_argument("--ocr", action="store_true", help="使用阿里云 OCR 识别配置页段；会上传页面图片。")
    parser.add_argument("--ocr-refresh", action="store_true", help="配合 --ocr 跳过缓存重新识别；可能增加费用。")
    args = parser.parse_args(argv)
    if args.ocr_refresh and not args.ocr:
        parser.error("--ocr-refresh 必须与 --ocr 一起使用")

    project_config = load_config(args.config)
    ocr_settings = OCRSettings.from_env() if args.ocr else None
    ocr_client = AliyunOCRClient(ocr_settings) if ocr_settings is not None else None
    output_root = Path(args.output_dir)
    database_path = Path(args.database) if args.database else output_root / "img2answer.sqlite3"
    store = SQLiteStore(database_path)
    store.initialize()
    vector_store = None
    embedding_model = None
    if args.embed_images:
        chroma_dir = Path(args.chroma_dir) if args.chroma_dir else output_root / "chroma"
        chroma_collection = resolve_chroma_collection(args.chroma_collection, args.embedding_model)
        vector_store = ChromaImageVectorStore(chroma_dir, collection_name=chroma_collection)
        embedding_model = create_embedding_model(args.embedding_model)

    for document_config in project_config.documents:
        metadata = inspect_pdf(document_config.path)
        section_reports: list[SectionReport] = []
        all_candidates = []
        section_candidates = {}
        ocr_pages = [] if args.ocr else None
        rendered_sections: list[tuple[SectionConfig, list[Path]]] = []

        for section in document_config.sections:
            validate_section(section, metadata.page_count)
            page_dir = output_root / "pages" / document_config.document_id / section.name

            rendered_pages = render_section(metadata.path, section, page_dir, dpi=args.dpi)
            if ocr_client is not None and ocr_settings is not None:
                for page_number, page_path in enumerate(rendered_pages, start=section.page_from):
                    ocr_result = recognize_page(
                        page_path,
                        output_root / "ocr" / document_config.document_id / section.name / f"page-{page_number:04d}.json",
                        client=ocr_client,
                        min_confidence=ocr_settings.min_confidence,
                        refresh=args.ocr_refresh,
                    )
                    ocr_pages.append({
                        "section": section.name,
                        "page_number": page_number,
                        "source_page": str(page_path),
                        "result_path": str(ocr_result.result_path),
                        "status": ocr_result.status,
                        "cache_hit": ocr_result.cache_hit,
                        "block_count": ocr_result.block_count,
                        "needs_review": ocr_result.needs_review,
                        "image_sha256": ocr_result.image_sha256,
                        "result_sha256": ocr_result.result_sha256,
                    })
            rendered_sections.append((section, rendered_pages))

        # Keep existing crops intact until every section's render/OCR phase succeeds.
        for section, rendered_pages in rendered_sections:
            crop_dir = output_root / "crops" / document_config.document_id / section.name
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
            ocr_pages=ocr_pages,
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
                embedding_model_name=args.embedding_model,
            )
            print(
                "image_vectors "
                f"document_id={document_config.document_id} "
                f"embedded={embedding_result.embedded} "
                f"skipped_missing_files={embedding_result.skipped_missing_files}"
            )

    return 0


def extract_questions_main(argv: list[str]) -> int:
    from .question_extract import ExtractionError, extract_questions

    parser = argparse.ArgumentParser(description="离线切分单题候选并写入 SQLite，所有结果均需核对。")
    parser.add_argument("--report", required=True, help="包含 OCR 快照绑定的成功导入报告。")
    parser.add_argument("--database", required=True, help="对应导入的现有 SQLite 数据库。")
    parser.add_argument("--output-dir", required=True, help="单题截图与核对文件的输出根目录。")
    parser.add_argument("--min-confidence", type=float, default=80, help="0 至 100 的核对阈值，默认 80。")
    args = parser.parse_args(argv)
    try:
        result = extract_questions(args.report, args.database, args.output_dir, min_confidence=args.min_confidence)
    except ExtractionError as exc:
        print(f"单题提取失败：{exc}", file=sys.stderr)
        return 1
    print(f"提取完成：候选 {result['candidate_count']} 道，待核对 {result['needs_review_count']} 道，"
          f"跳过页面 {result['skipped_page_count']} 页。")
    print(f"运行目录：{result['run_dir']}")
    print(f"核对文件：{result['extraction_path']}")
    return 0


def index_questions_main(argv: list[str]) -> int:
    from .question_index import QuestionIndexError, index_questions

    parser = argparse.ArgumentParser(description="为当前文档的单题截图建立独立向量索引。")
    parser.add_argument("--database", required=True)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--chroma-dir", default="data/processed/chroma")
    parser.add_argument("--embedding-model", choices=("pillow-hash", "openclip"), default="pillow-hash")
    parser.add_argument("--chroma-collection", default=None)
    parser.add_argument("--verbose", action="store_true", help="失败时显示完整异常链，可能包含本地路径。")
    args = parser.parse_args(argv)
    try:
        result = index_questions(args.database, args.document_id, args.chroma_dir,
                                 embedding_model_name=args.embedding_model, collection_name=args.chroma_collection)
    except QuestionIndexError as exc:
        print(f"单题索引失败：{exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exception(exc, file=sys.stderr)
        else:
            print("可加 --verbose 查看完整异常链。", file=sys.stderr)
        return 1
    print("单题索引完成：" + " ".join(f"{key}={value}" for key, value in result.items()))
    return 0


def search_image_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Search similar crop candidate images.")
    parser.add_argument("--image", required=True, help="Query image path.")
    parser.add_argument(
        "--database",
        default="data/processed/img2answer.sqlite3",
        help="SQLite database path for local metadata.",
    )
    parser.add_argument(
        "--chroma-dir",
        default="data/processed/chroma",
        help="ChromaDB persistence directory.",
    )
    parser.add_argument(
        "--chroma-collection",
        default=None,
        help=(
            "ChromaDB collection name for image vectors. Defaults to question_images for pillow-hash "
            "and question_images_openclip for openclip."
        ),
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of similar images to return.")
    parser.add_argument(
        "--embedding-model",
        default="pillow-hash",
        help="Image embedding model to use for the query image: pillow-hash or openclip.",
    )
    args = parser.parse_args(argv)

    sqlite_store = SQLiteStore(args.database)
    chroma_collection = resolve_chroma_collection(args.chroma_collection, args.embedding_model)
    vector_store = ChromaImageVectorStore(args.chroma_dir, collection_name=chroma_collection)
    results = search_similar_images(
        args.image,
        sqlite_store=sqlite_store,
        vector_store=vector_store,
        embedding_model=create_embedding_model(args.embedding_model),
        top_k=args.top_k,
    )
    print(
        json.dumps(
            {
                "query_image": str(Path(args.image)),
                "top_k": args.top_k,
                "embedding_model": args.embedding_model,
                "results": [result.__dict__ for result in results],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def resolve_chroma_collection(collection_name: str | None, embedding_model_name: str) -> str:
    if collection_name:
        return collection_name
    normalized_model = embedding_model_name.strip().lower()
    if normalized_model == "pillow-hash":
        return DEFAULT_CHROMA_COLLECTION
    return f"{DEFAULT_CHROMA_COLLECTION}_{normalized_model.replace('-', '_')}"


if __name__ == "__main__":
    raise SystemExit(main())

