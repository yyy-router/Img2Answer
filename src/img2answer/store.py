from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from contextlib import contextmanager
import json
import sqlite3
from datetime import datetime, timezone

from .graphic_crop import CropCandidate
from .pdf_document import PdfMetadata


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class StoreCounts:
    source_documents: int
    question_images: int
    process_reports: int


class SQLiteStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS source_documents (
                    id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    page_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS question_images (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    section TEXT NOT NULL,
                    image_role TEXT NOT NULL,
                    source_page_path TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    bbox_json TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES source_documents(id)
                );

                CREATE TABLE IF NOT EXISTS process_reports (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    report_path TEXT NOT NULL,
                    page_count INTEGER NOT NULL,
                    rendered_pages INTEGER NOT NULL,
                    crop_candidates INTEGER NOT NULL,
                    warnings_json TEXT NOT NULL,
                    errors_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES source_documents(id)
                );
                """
            )

    def upsert_source_document(self, document_id: str, metadata: PdfMetadata) -> None:
        now = utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO source_documents (
                    id, source_path, sha256, page_count, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    source_path = excluded.source_path,
                    sha256 = excluded.sha256,
                    page_count = excluded.page_count,
                    updated_at = excluded.updated_at
                """,
                (
                    document_id,
                    str(metadata.path),
                    metadata.sha256,
                    metadata.page_count,
                    now,
                    now,
                ),
            )

    def replace_crop_candidates(
        self,
        *,
        document_id: str,
        section: str,
        candidates: list[CropCandidate],
    ) -> None:
        now = utc_now_iso()
        with self._connect() as connection:
            for candidate in candidates:
                image_id = crop_candidate_id(document_id, section, candidate)
                connection.execute(
                    """
                    INSERT OR REPLACE INTO question_images (
                        id,
                        document_id,
                        section,
                        image_role,
                        source_page_path,
                        output_path,
                        bbox_json,
                        width,
                        height,
                        created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        image_id,
                        document_id,
                        section,
                        "crop_candidate",
                        str(candidate.source_page),
                        str(candidate.output_path),
                        json.dumps(list(candidate.bbox), ensure_ascii=False),
                        candidate.width,
                        candidate.height,
                        now,
                    ),
                )

    def insert_process_report(
        self,
        *,
        document_id: str,
        report_path: str | Path,
        report: dict[str, Any],
    ) -> str:
        now = utc_now_iso()
        report_id = f"{document_id}:{now}"
        rendered_pages = sum(section.get("rendered_pages", 0) for section in report["processed_sections"])
        crop_candidates = sum(section.get("crop_candidates", 0) for section in report["processed_sections"])
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO process_reports (
                    id,
                    document_id,
                    report_path,
                    page_count,
                    rendered_pages,
                    crop_candidates,
                    warnings_json,
                    errors_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report_id,
                    document_id,
                    str(report_path),
                    report["page_count"],
                    rendered_pages,
                    crop_candidates,
                    json.dumps(report.get("warnings", []), ensure_ascii=False),
                    json.dumps(report.get("errors", []), ensure_ascii=False),
                    now,
                ),
            )
        return report_id

    def counts(self) -> StoreCounts:
        with self._connect() as connection:
            return StoreCounts(
                source_documents=_table_count(connection, "source_documents"),
                question_images=_table_count(connection, "question_images"),
                process_reports=_table_count(connection, "process_reports"),
            )

    def fetch_all(self, table: str) -> list[sqlite3.Row]:
        if table not in {"source_documents", "question_images", "process_reports"}:
            raise ValueError(f"unsupported table: {table}")
        with self._connect() as connection:
            return list(connection.execute(f"SELECT * FROM {table} ORDER BY id"))

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()


def crop_candidate_id(document_id: str, section: str, candidate: CropCandidate) -> str:
    return f"{document_id}:{section}:{candidate.source_page.stem}:crop-001"


def _table_count(connection: sqlite3.Connection, table: str) -> int:
    cursor = connection.execute(f"SELECT COUNT(*) FROM {table}")
    return int(cursor.fetchone()[0])
