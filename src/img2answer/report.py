from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Any

from .graphic_crop import CropCandidate
from .pdf_document import PdfMetadata


@dataclass(frozen=True)
class SectionReport:
    section: str
    page_from: int
    page_to: int
    rendered_pages: int
    crop_candidates: int


def build_report(
    *,
    document_id: str,
    metadata: PdfMetadata,
    sections: list[SectionReport],
    crop_candidates: list[CropCandidate],
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "source_path": str(metadata.path),
        "sha256": metadata.sha256,
        "page_count": metadata.page_count,
        "processed_sections": [asdict(section) for section in sections],
        "crop_candidates": [
            {
                "source_page": str(candidate.source_page),
                "output_path": str(candidate.output_path),
                "bbox": list(candidate.bbox),
                "width": candidate.width,
                "height": candidate.height,
            }
            for candidate in crop_candidates
        ],
        "warnings": warnings or [],
        "errors": errors or [],
    }


def write_report(report: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

