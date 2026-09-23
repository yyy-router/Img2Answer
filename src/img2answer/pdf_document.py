from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import pymupdf

from .config import ConfigError, SectionConfig


@dataclass(frozen=True)
class PdfMetadata:
    path: Path
    sha256: str
    page_count: int


def inspect_pdf(path: str | Path) -> PdfMetadata:
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF path is not a file: {pdf_path}")

    try:
        with pymupdf.open(str(pdf_path)) as document:
            page_count = document.page_count
    except Exception as exc:
        raise ValueError(f"could not open PDF: {pdf_path}") from exc

    return PdfMetadata(path=pdf_path, sha256=sha256_file(pdf_path), page_count=page_count)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_section(section: SectionConfig, page_count: int) -> None:
    if section.page_to > page_count:
        raise ConfigError(
            f"section '{section.name}' ends at page {section.page_to}, but PDF has {page_count} pages"
        )

