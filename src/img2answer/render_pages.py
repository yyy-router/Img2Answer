from __future__ import annotations

from pathlib import Path

import pymupdf

from .config import SectionConfig
from .pdf_document import validate_section


def render_section(
    pdf_path: str | Path,
    section: SectionConfig,
    output_dir: str | Path,
    *,
    dpi: int = 300,
) -> list[Path]:
    if dpi <= 0:
        raise ValueError("dpi must be positive")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rendered: list[Path] = []
    with pymupdf.open(str(pdf_path)) as document:
        validate_section(section, document.page_count)
        zoom = dpi / 72
        matrix = pymupdf.Matrix(zoom, zoom)
        for page_index in section.zero_based_range():
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = output_path / f"page-{page_index + 1:04d}.png"
            pixmap.save(str(image_path))
            rendered.append(image_path)
    return rendered

