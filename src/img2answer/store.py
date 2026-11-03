from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence
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


@dataclass(frozen=True)
class ImportState:
    document_id: str
    sha256: str
    page_count: int
    source_updated_at: str
    report_created_at: str
    report_path: str


class ImportChangedError(ValueError):
    """The import changed after an extraction took its snapshot."""


class QuestionDataUnavailableError(RuntimeError):
    """Question data cannot be read without an import or schema upgrade."""


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

                CREATE TABLE IF NOT EXISTS questions (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES source_documents(id),
                    section TEXT NOT NULL,
                    page_number INTEGER NOT NULL CHECK (page_number > 0),
                    question_no TEXT NOT NULL,
                    image_id TEXT NOT NULL UNIQUE REFERENCES question_images(id) ON DELETE CASCADE,
                    source_sha256 TEXT NOT NULL,
                    ocr_result_path TEXT NOT NULL,
                    raw_text TEXT NOT NULL,
                    stem_text TEXT NOT NULL,
                    options_json TEXT NOT NULL,
                    answer TEXT CHECK (answer IS NULL OR answer IN ('A', 'B', 'C', 'D')),
                    analysis_text TEXT NOT NULL,
                    parse_status TEXT NOT NULL CHECK (parse_status = 'needs_review'),
                    review_reasons_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def replace_document_run(
        self,
        *,
        document_id: str,
        metadata: PdfMetadata,
        section_candidates: dict[str, list[CropCandidate]],
        report_path: str | Path,
        report: dict[str, Any],
    ) -> str:
        now = report.get("imported_at") or utc_now_iso()
        report_id = document_id
        rendered_pages = sum(section.get("rendered_pages", 0) for section in report["processed_sections"])
        crop_candidates = sum(section.get("crop_candidates", 0) for section in report["processed_sections"])
        with self._connect() as connection:
            self._upsert_source_document(connection, document_id, metadata, now)
            connection.execute("DELETE FROM question_images WHERE document_id = ?", (document_id,))
            for section, candidates in section_candidates.items():
                for candidate in candidates:
                    image_id = crop_candidate_id(document_id, section, candidate)
                    connection.execute(
                        """
                        INSERT INTO question_images (
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

            connection.execute("DELETE FROM process_reports WHERE document_id = ?", (document_id,))
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
        if table not in {"source_documents", "question_images", "process_reports", "questions"}:
            raise ValueError(f"unsupported table: {table}")
        with self._connect() as connection:
            return list(connection.execute(f"SELECT * FROM {table} ORDER BY id"))

    def read_import_state(self, document_id: str) -> ImportState | None:
        with self._connect(read_only=True) as connection:
            return self._import_state(connection, document_id)

    @staticmethod
    def _import_state(connection: sqlite3.Connection, document_id: str) -> ImportState | None:
        row = connection.execute(
            """SELECT s.id, s.sha256, s.page_count, s.updated_at, p.created_at, p.report_path
               FROM source_documents s JOIN process_reports p ON p.document_id = s.id
               WHERE s.id = ?""", (document_id,),
        ).fetchone()
        return ImportState(*row) if row is not None else None

    def replace_extracted_questions(self, expected_import: ImportState, questions: Sequence[dict[str, Any]]) -> None:
        """Replace one section only after checking the import inside the write transaction."""
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if self._import_state(connection, expected_import.document_id) != expected_import:
                raise ImportChangedError("来源导入版本已变化，请重新运行单题提取")
            previous = dict(connection.execute(
                "SELECT id, created_at FROM questions WHERE document_id = ? AND section = 'graphic_reasoning'",
                (expected_import.document_id,),
            ).fetchall())
            connection.execute(
                "DELETE FROM question_images WHERE document_id = ? AND section = 'graphic_reasoning' "
                "AND image_role = 'question_crop'", (expected_import.document_id,),
            )
            now = utc_now_iso()
            for question in questions:
                if question["document_id"] != expected_import.document_id or question["section"] != "graphic_reasoning" \
                        or question["source_sha256"] != expected_import.sha256:
                    raise ValueError("单题来源与导入版本不一致")
                image = question["image"]
                connection.execute(
                    """INSERT INTO question_images
                       (id, document_id, section, image_role, source_page_path, output_path,
                        bbox_json, width, height, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (question["image_id"], question["document_id"], question["section"], "question_crop",
                     image["source_page_path"], image["output_path"], json.dumps(image["bbox"]),
                     image["width"], image["height"], now),
                )
                connection.execute(
                    """INSERT INTO questions
                       (id, document_id, section, page_number, question_no, image_id, source_sha256,
                        ocr_result_path, raw_text, stem_text, options_json, answer, analysis_text,
                        parse_status, review_reasons_json, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (question["id"], question["document_id"], question["section"], question["page_number"],
                     question["question_no"], question["image_id"], question["source_sha256"],
                     question["ocr_result_path"], question["raw_text"], question["stem_text"],
                     json.dumps(question["options"], ensure_ascii=False), question["answer"],
                     question["analysis_text"], "needs_review", json.dumps(question["review_reasons"]),
                     previous.get(question["id"], now), now),
                )

    def fetch_question(self, question_id: str) -> sqlite3.Row | None:
        with self._connect(read_only=True) as connection:
            return connection.execute("SELECT * FROM questions WHERE id = ?", (question_id,)).fetchone()

    def fetch_question_by_image(self, image_id: str) -> sqlite3.Row | None:
        with self._connect(read_only=True) as connection:
            return connection.execute("SELECT * FROM questions WHERE image_id = ?", (image_id,)).fetchone()

    @contextmanager
    def question_connection(self, *, for_update: bool = False) -> Iterator[sqlite3.Connection]:
        """Keep question reads or final index publication on one SQLite snapshot."""
        if not self.path.is_file():
            raise QuestionDataUnavailableError("题目数据库不可用，请先完成导入和单题提取")
        with self._connect(read_only=not for_update, existing=True) as connection:
            connection.execute("BEGIN IMMEDIATE" if for_update else "BEGIN")
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            if not {"questions", "question_images", "source_documents", "process_reports"} <= tables:
                raise QuestionDataUnavailableError("单题数据表不可用，请先完成导入和单题提取")
            yield connection

    @staticmethod
    def question_records(connection: sqlite3.Connection, *, document_id: str | None = None,
                         question_id: str | None = None, image_id: str | None = None) -> list[dict[str, Any]]:
        predicates, parameters = [], []
        for column, value in (("document_id", document_id), ("id", question_id), ("image_id", image_id)):
            if value is not None:
                predicates.append(f"q.{column} = ?")
                parameters.append(value)
        where = " AND ".join(predicates) if predicates else "1 = 1"
        rows = connection.execute(
            """SELECT q.*, i.id AS joined_image_id, i.document_id AS image_document_id,
                      i.section AS image_section, i.image_role, i.source_page_path, i.output_path,
                      i.bbox_json, i.width, i.height, i.created_at AS image_created_at,
                      s.sha256 AS import_sha256, s.page_count AS import_page_count,
                      s.updated_at AS source_updated_at, p.created_at AS report_created_at,
                      p.report_path, p.errors_json AS import_errors_json
               FROM questions q
               LEFT JOIN question_images i ON i.id = q.image_id
               LEFT JOIN source_documents s ON s.id = q.document_id
               LEFT JOIN process_reports p ON p.document_id = q.document_id
               WHERE """ + where + " ORDER BY q.id", parameters,
        )
        return [dict(row) for row in rows]

    @staticmethod
    def question_document_state(connection: sqlite3.Connection, document_id: str) -> ImportState:
        state = SQLiteStore._import_state(connection, document_id)
        if state is None:
            raise ValueError("文档不存在或缺少成功导入记录")
        errors = json.loads(connection.execute(
            "SELECT errors_json FROM process_reports WHERE document_id = ?", (document_id,),
        ).fetchone()[0])
        if errors != []:
            raise ValueError("文档缺少有效成功导入记录")
        return state

    def fetch_question_images(self, document_id: str | None = None) -> list[sqlite3.Row]:
        with self._connect() as connection:
            if document_id is None:
                return list(connection.execute("SELECT * FROM question_images ORDER BY id"))
            return list(
                connection.execute(
                    "SELECT * FROM question_images WHERE document_id = ? ORDER BY id",
                    (document_id,),
                )
            )

    def fetch_question_image(self, image_id: str) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                "SELECT * FROM question_images WHERE id = ?",
                (image_id,),
            ).fetchone()

    def fetch_question_image_detail(self, image_id: str) -> sqlite3.Row | None:
        with self._connect() as connection:
            return connection.execute(
                """
                SELECT
                    question_images.id AS image_id,
                    question_images.document_id,
                    question_images.section,
                    question_images.image_role,
                    question_images.source_page_path,
                    question_images.output_path,
                    question_images.bbox_json,
                    question_images.width,
                    question_images.height,
                    process_reports.report_path
                FROM question_images
                LEFT JOIN process_reports
                    ON process_reports.document_id = question_images.document_id
                WHERE question_images.id = ?
                """,
                (image_id,),
            ).fetchone()

    @contextmanager
    def _connect(self, *, read_only: bool = False, existing: bool = False) -> Iterator[sqlite3.Connection]:
        mode = "ro" if read_only else "rw"
        connection = (sqlite3.connect(f"{self.path.resolve().as_uri()}?mode={mode}", uri=True)
                      if read_only or existing else sqlite3.connect(self.path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _upsert_source_document(
        self,
        connection: sqlite3.Connection,
        document_id: str,
        metadata: PdfMetadata,
        now: str,
    ) -> None:
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


def crop_candidate_id(document_id: str, section: str, candidate: CropCandidate) -> str:
    return f"{document_id}:{section}:{candidate.source_page.stem}:crop-001"


def _table_count(connection: sqlite3.Connection, table: str) -> int:
    cursor = connection.execute(f"SELECT COUNT(*) FROM {table}")
    return int(cursor.fetchone()[0])
