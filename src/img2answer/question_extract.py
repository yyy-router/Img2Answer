from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Any

from PIL import Image, ImageDraw

from .ocr import OCRError, OCRSnapshot, read_ocr_snapshot, result_sha256
from .store import ImportChangedError, SQLiteStore


SECTION = "graphic_reasoning"
_PREFIX = re.compile(r"^[ \t]*([1-9][0-9]{0,3})[.．、)）](?![ \t]*[0-9])[ \t]*")
_MARKER = re.compile(
    r"(?<!\w)(?P<answer>正确答案|答案)[ \t]*[:：]?|(?<!\w)(?P<analysis>解析)[ \t]*[:：]?|"
    r"(?<![A-Za-z0-9])(?P<option>[A-D])[.．、)）:：][ \t]*|"
    r"^[ \t]*(?P<bare_option>[A-D])[ \t]+(?=\S)",
    re.MULTILINE,
)


class ExtractionError(RuntimeError):
    """A local extraction error safe to display without question text or credentials."""


@dataclass
class _Line:
    blocks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def top(self) -> float:
        return min(block["bbox"][1] for block in self.blocks)

    @property
    def bottom(self) -> float:
        return max(block["bbox"][3] for block in self.blocks)


def _lines(blocks: list[dict[str, Any]]) -> list[_Line]:
    lines: list[_Line] = []
    for block in sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0], b["text"])):
        top, bottom = block["bbox"][1], block["bbox"][3]
        matching = [line for line in lines
                    if max(bottom - top, line.bottom - line.top) <= 2 * min(bottom - top, line.bottom - line.top)
                    and min(bottom, line.bottom) - max(top, line.top)
                    >= min(bottom - top, line.bottom - line.top) * 0.5]
        if matching:
            min(matching, key=lambda line: abs((line.top + line.bottom) - (top + bottom))).blocks.append(block)
        else:
            lines.append(_Line([block]))
    for line in lines:
        line.blocks.sort(key=lambda b: (b["bbox"][0], b["bbox"][1], b["text"]))
    return sorted(lines, key=lambda line: (line.top, line.blocks[0]["bbox"][0]))


def _text(lines: list[_Line]) -> str:
    rows = []
    for line in lines:
        text = ""
        previous = None
        for block in line.blocks:
            if previous is not None:
                gap = block["bbox"][0] - previous["bbox"][2]
                height = max(block["bbox"][3] - block["bbox"][1],
                             previous["bbox"][3] - previous["bbox"][1])
                # Distant fragments must not turn an answer label into an unrelated value.
                text += "\n" if gap > height * 2 else " "
            text += block["text"]
            previous = block
        rows.append(text)
    return "\n".join(rows)


def _fields(raw_text: str) -> dict[str, Any]:
    body = _PREFIX.sub("", raw_text, count=1).strip()
    markers = []
    for marker in _MARKER.finditer(body):
        if marker.lastgroup in {"option", "bare_option"} and markers and markers[-1].lastgroup == "answer":
            between = body[markers[-1].end():marker.start()]
            if "\n" not in between and not between.strip():
                continue
        markers.append(marker)
    stem = body[:markers[0].start()].strip() if markers else body
    options: dict[str, str] = {}
    duplicate_options: set[str] = set()
    answers: list[str | None] = []
    analysis = ""
    for index, marker in enumerate(markers):
        if marker.lastgroup == "analysis":
            analysis = body[marker.end():].strip()
            break
        end = markers[index + 1].start() if index + 1 < len(markers) else len(body)
        value = body[marker.end():end].strip()
        if marker.lastgroup in {"option", "bare_option"}:
            label = marker.group(marker.lastgroup)
            if label in options:
                duplicate_options.add(label)
            options[label] = value
        else:
            # A value on a later line is deliberately not associated with this label.
            first_line = body[marker.end():end].split("\n", 1)[0].strip()
            match = re.fullmatch(r"([A-D])[ \t]*[。．.;；]?", first_line)
            answers.append(match[1] if match else None)
    for label in duplicate_options:
        options.pop(label, None)
    options = {label: value for label, value in options.items() if value}
    reasons = []
    answer = answers[0] if answers and None not in answers and len(set(answers)) == 1 else None
    if not stem:
        reasons.append("missing_stem_text")
    if not options:
        reasons.append("missing_options")
    if answer is None:
        reasons.append("missing_answer")
    if len({value for value in answers if value is not None}) > 1:
        reasons.append("conflicting_answer")
    if None in answers:
        reasons.append("invalid_answer")
    if duplicate_options:
        reasons.append("conflicting_options")
    return dict(stem_text=stem, options=options, answer=answer, analysis_text=analysis, review_reasons=reasons)


def _content_mask(image: Image.Image) -> Image.Image:
    return image.convert("L").point(lambda value: 255 if value < 245 else 0, mode="1")


def extract_page(snapshot: OCRSnapshot, *, page_number: int, min_confidence: float = 80) -> dict[str, Any]:
    """Compute candidate boxes and text from one validated, immutable page snapshot."""
    blocks = snapshot.result["blocks"]
    lines = _lines(blocks)
    page: dict[str, Any] = {"page_number": page_number, "status": "skipped", "warnings": [],
                            "candidates": [], "skipped_candidates": []}
    anchors = []
    for line_index, line in enumerate(lines):
        for block in line.blocks:
            # One box containing several anchors has no usable per-question geometry.
            if sum(_PREFIX.match(row) is not None for row in block["text"].splitlines()) > 1:
                page["warnings"].append("ambiguous_boundary")
                return page
            match = _PREFIX.match(block["text"])
            if match:
                anchors.append((line_index, block, match[1], block["text"][match.end():].strip()))
    with Image.open(io.BytesIO(snapshot.image)) as original:
        mask = _content_mask(original)
    if not anchors:
        page["warnings"].append("no_question_anchor" if mask.getbbox() else "empty_page")
        return page
    numbers = [int(anchor[2]) for anchor in anchors]
    xs = [anchor[1]["bbox"][0] for anchor in anchors]
    conflicts = []
    if len({anchor[0] for anchor in anchors}) != len(anchors):
        conflicts.append("multiple_anchors_on_line")
    if len(set(numbers)) != len(numbers):
        conflicts.append("duplicate_question_number")
    if any(after < before for before, after in zip(numbers, numbers[1:])):
        conflicts.append("question_number_order")
    if max(xs) > snapshot.width * 0.25 or max(xs) - min(xs) > snapshot.width * 0.05:
        conflicts.append("unsupported_columns")
    if conflicts:
        page["warnings"].extend(conflicts)
        return page
    boundaries = [math.floor(lines[anchor[0]].top) for anchor in anchors] + [snapshot.height]
    if any(block["bbox"][1] < edge < block["bbox"][3]
           for edge in boundaries[:-1] for block in blocks):
        page["warnings"].append("ambiguous_boundary")
        return page
    if boundaries[0] > 0 and mask.crop((0, 0, snapshot.width, boundaries[0])).getbbox():
        page["warnings"].append("unassigned_prefix")
    for index, (_, anchor_block, question_no, inline_text) in enumerate(anchors):
        top, bottom = boundaries[index:index + 2]
        used = [block for block in blocks if top <= block["bbox"][1] and block["bbox"][3] <= bottom]
        region = mask.crop((0, top, snapshot.width, bottom))
        content = region.copy()
        if not inline_text:
            box = anchor_block["bbox"]
            ImageDraw.Draw(content).rectangle(
                (max(0, math.floor(box[0]) - 2), max(0, math.floor(box[1]) - top - 2),
                 min(snapshot.width, math.ceil(box[2]) + 2), min(bottom - top, math.ceil(box[3]) - top + 2)),
                fill=0,
            )
        bbox = region.getbbox()
        if bbox is None or (not inline_text and content.getbbox() is None):
            page["skipped_candidates"].append({"question_no": question_no, "anchor_index": index + 1,
                                                "reason": "no_question_content"})
            continue
        left, local_top, right, local_bottom = bbox
        bbox = [max(0, left - 8), max(top, top + local_top - 8),
                min(snapshot.width, right + 8), min(bottom, top + local_bottom + 8)]
        raw_text = _text(_lines(used))
        fields = _fields(raw_text)
        reasons = ["auto_extracted", *fields.pop("review_reasons")]
        if any(block["confidence"] is None for block in used):
            reasons.append("missing_confidence")
        if any(block["confidence"] is not None and block["confidence"] < min_confidence for block in used):
            reasons.append("low_confidence")
        if index > 0 and numbers[index] != numbers[index - 1] + 1:
            reasons.append("question_number_gap")
        if index == len(anchors) - 1:
            reasons.append("open_page_end")
        page["candidates"].append(dict(anchor_index=index + 1, question_no=question_no, bbox=bbox,
                                         raw_text=raw_text, parse_status="needs_review", review_reasons=reasons,
                                         **fields))
    page["status"] = "candidates" if page["candidates"] else "skipped"
    return page


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _digest(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _reject_constant(value: str) -> None:
    raise ValueError("invalid JSON number")


def _preflight(report_path: Path, store: SQLiteStore, output_root: Path):
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes.decode("utf-8-sig"), parse_constant=_reject_constant)
    if not isinstance(report, dict) or not _nonempty(report.get("document_id")) \
            or not _digest(report.get("sha256")) or type(report.get("page_count")) is not int \
            or report["page_count"] < 1 or not isinstance(report.get("errors"), list) or report["errors"]:
        raise ExtractionError("处理报告结构无效或不是成功导入报告")
    if not _nonempty(report.get("imported_at")):
        raise ExtractionError("旧报告缺少导入快照绑定，请重新执行原有导入 / OCR 流程，可复用有效缓存")
    state = store.read_import_state(report["document_id"])
    if state is None or state.sha256 != report["sha256"] or state.page_count != report["page_count"] \
            or Path(state.report_path).resolve() != report_path.resolve() \
            or state.source_updated_at != report["imported_at"] or state.report_created_at != report["imported_at"]:
        raise ExtractionError("报告与数据库当前导入版本不匹配，请选择对应数据集或重新完成导入")
    sections = report.get("processed_sections")
    if not isinstance(sections, list) or not sections or any(not isinstance(s, dict) for s in sections):
        raise ExtractionError("报告缺少有效页段")
    names = [s.get("section") for s in sections]
    if any(not _nonempty(name) for name in names) or len(set(names)) != len(names):
        raise ExtractionError("报告页段名称无效或重复")
    targets = [section for section in sections if section["section"] == SECTION]
    if len(targets) != 1:
        raise ExtractionError("报告缺少 graphic_reasoning 目标页段")
    section = targets[0]
    if type(section.get("page_from")) is not int or type(section.get("page_to")) is not int \
            or not 1 <= section["page_from"] <= section["page_to"] <= report["page_count"]:
        raise ExtractionError("目标页段页码范围无效")
    pages = report.get("ocr_pages")
    if not isinstance(pages, list) or any(not isinstance(page, dict) for page in pages):
        raise ExtractionError("报告缺少有效 OCR 页面引用")
    target_pages = [page for page in pages if page.get("section") == SECTION]
    page_numbers = [page.get("page_number") for page in target_pages]
    expected = list(range(section["page_from"], section["page_to"] + 1))
    if any(type(number) is not int for number in page_numbers) or sorted(page_numbers) != expected:
        raise ExtractionError("目标 OCR 页码重复、缺失或不在页段范围内")
    protected = [report_path.parent.parent / name for name in ("pages", "crops", "ocr", "reports")]
    snapshots = []
    seen_images: set[Path] = set()
    seen_results: set[Path] = set()
    for page in sorted(target_pages, key=lambda item: item["page_number"]):
        if not _nonempty(page.get("source_page")) or not _nonempty(page.get("result_path")):
            raise ExtractionError(f"第 {page['page_number']} 页缺少图片或 OCR 路径")
        if not _digest(page.get("image_sha256")) or not _digest(page.get("result_sha256")):
            raise ExtractionError("旧报告缺少逐页快照绑定，请重新执行原有导入 / OCR 流程，可复用有效缓存")
        image_path, result_path = Path(page["source_page"]).resolve(), Path(page["result_path"]).resolve()
        if image_path in seen_images or result_path in seen_results:
            raise ExtractionError("目标页面重复引用同一图片或 OCR 路径")
        seen_images.add(image_path)
        seen_results.add(result_path)
        protected.extend([image_path.parent, result_path.parent])
        try:
            snapshot = read_ocr_snapshot(image_path, result_path)
        except OCRError:
            raise ExtractionError(f"第 {page['page_number']} 页图片或 OCR 缓存无效，请重新完成原有导入 / OCR 流程") from None
        if snapshot.result["image_sha256"] != page["image_sha256"] \
                or result_sha256(snapshot.result) != page["result_sha256"]:
            raise ExtractionError(f"第 {page['page_number']} 页快照与成功导入报告不一致，请重新完成导入")
        snapshots.append((page, snapshot))
    if any(output_root.is_relative_to(path.resolve()) for path in protected):
        raise ExtractionError("单题输出目录不能位于原有 pages、crops、ocr 或 reports 目录中")
    return report, report_bytes, state, snapshots, [name for name in names if name != SECTION]


def _cleanup(run_dir: Path, output_root: Path, paths: list[Path]) -> None:
    if run_dir.resolve().parent != output_root or not run_dir.resolve().is_relative_to(output_root):
        raise OSError("unsafe cleanup target")
    for path in reversed(paths):
        if path.resolve().parent != run_dir.resolve():
            raise OSError("unsafe cleanup artifact")
        path.unlink(missing_ok=True)
    run_dir.rmdir()


def extract_questions(report_path: str | Path, database: str | Path, output_dir: str | Path,
                      *, min_confidence: float = 80) -> dict[str, Any]:
    if type(min_confidence) not in (int, float) or not 0 <= min_confidence <= 100 \
            or not math.isfinite(min_confidence):
        raise ExtractionError("核对阈值必须为 0 至 100 的有限数值")
    report_path, database, output_root = Path(report_path), Path(database), Path(output_dir).resolve()
    if not database.is_file():
        raise ExtractionError("SQLite 数据库文件不存在，请选择对应的成功导入数据集")
    store = SQLiteStore(database)
    run_dir = None
    written: list[Path] = []
    try:
        report, report_bytes, state, snapshots, skipped_sections = _preflight(report_path, store, output_root)
        pages = []
        questions = []
        for reference, snapshot in snapshots:
            page = extract_page(snapshot, page_number=reference["page_number"], min_confidence=min_confidence)
            page.update(source_page=reference["source_page"], ocr_result_path=reference["result_path"],
                        image_sha256=reference["image_sha256"], result_sha256=reference["result_sha256"])
            for candidate in page.pop("candidates"):
                ordinal = candidate.pop("anchor_index")
                question_id = f"{report['document_id']}:{SECTION}:page-{reference['page_number']:04d}:question-{ordinal:03d}"
                bbox = candidate.pop("bbox")
                questions.append(dict(
                    id=question_id, document_id=report["document_id"], section=SECTION,
                    page_number=reference["page_number"], image_id=f"{question_id}:image",
                    source_sha256=report["sha256"], ocr_result_path=reference["result_path"],
                    image=dict(source_page_path=reference["source_page"], bbox=bbox,
                               width=bbox[2] - bbox[0], height=bbox[3] - bbox[1]),
                    **candidate,
                ))
            pages.append(page)
        # Nothing on disk is changed until every page has passed validation and parsing.
        output_root.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="run-", dir=output_root))
        by_page = {reference["page_number"]: snapshot for reference, snapshot in snapshots}
        for question in questions:
            ordinal = question["id"].rsplit("question-", 1)[1]
            path = run_dir / f"page-{question['page_number']:04d}-question-{ordinal}.png"
            written.append(path)
            with Image.open(io.BytesIO(by_page[question["page_number"]].image)) as image:
                image.crop(question["image"]["bbox"]).save(path)
            question["image"]["output_path"] = str(path)
        extraction_path = run_dir / "extraction.json"
        result = dict(
            schema_version=1, document_id=report["document_id"], section=SECTION,
            source_sha256=report["sha256"], report_path=str(report_path), imported_at=report["imported_at"],
            report_sha256=hashlib.sha256(report_bytes).hexdigest(), min_confidence=min_confidence,
            pages=pages, questions=questions, skipped_sections=skipped_sections,
            candidate_count=len(questions), needs_review_count=len(questions),
            skipped_page_count=sum(page["status"] == "skipped" for page in pages),
            run_dir=str(run_dir), extraction_path=str(extraction_path),
        )
        written.append(extraction_path)
        extraction_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        if report_path.read_bytes() != report_bytes:
            raise ExtractionError("提取期间导入报告已变化，请重新运行单题提取")
        store.initialize()
        store.replace_extracted_questions(state, questions)
        return result
    except Exception as exc:
        if run_dir is not None:
            try:
                _cleanup(run_dir, output_root, written)
            except OSError:
                raise ExtractionError(f"提取失败且本轮产物未完全清理，请检查运行目录：{run_dir}") from None
        if isinstance(exc, ExtractionError):
            raise
        if isinstance(exc, ImportChangedError):
            raise ExtractionError(str(exc)) from None
        if isinstance(exc, (OSError, ValueError, TypeError, sqlite3.Error, RecursionError)):
            raise ExtractionError("输入读取、候选处理或写入失败，请检查文件、报告和数据库；历史候选未替换") from None
        raise
