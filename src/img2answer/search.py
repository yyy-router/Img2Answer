from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

from PIL import Image

from .question_index import (current_question, decoded_question_fields, read_question_image, record_revision)
from .store import SQLiteStore
from .vector_store import ImageEmbeddingModel, ImageSearchMatch, question_embedding_signature


class ImageSearchStore(Protocol):
    def query_similar_images(self, embedding: list[float], top_k: int) -> list[ImageSearchMatch]:
        """Return nearest image vectors for one query embedding."""


@dataclass(frozen=True)
class ImageSearchResult:
    rank: int
    image_id: str
    distance: float | None
    metadata: dict[str, str]
    record: dict[str, Any] | None
    detail: dict[str, Any] | None


@dataclass(frozen=True)
class QuestionImageDetail:
    image_id: str
    document_id: str
    section: str
    image_role: str
    source_page_path: str
    output_path: str
    bbox: list[int]
    width: int
    height: int
    report_path: str | None


def search_similar_images(
    query_image: str | Path,
    *,
    sqlite_store: SQLiteStore,
    vector_store: ImageSearchStore,
    embedding_model: ImageEmbeddingModel,
    top_k: int,
) -> list[ImageSearchResult]:
    image_path = Path(query_image)
    if not image_path.exists():
        raise FileNotFoundError(f"query image does not exist: {image_path}")
    if top_k < 1:
        raise ValueError("top_k must be greater than 0")

    embedding = embedding_model.embed_image(image_path)
    if not embedding:
        raise ValueError(f"empty embedding for query image: {image_path}")

    matches = vector_store.query_similar_images(embedding, top_k=top_k)
    results: list[ImageSearchResult] = []
    for rank, match in enumerate(matches, start=1):
        row = sqlite_store.fetch_question_image(match.image_id)
        detail = fetch_question_image_detail(sqlite_store, match.image_id)
        results.append(
            ImageSearchResult(
                rank=rank,
                image_id=match.image_id,
                distance=match.distance,
                metadata=match.metadata,
                record=dict(row) if row is not None else None,
                detail=detail.__dict__ if detail is not None else None,
            )
        )
    return results


def fetch_question_image_detail(sqlite_store: SQLiteStore, image_id: str) -> QuestionImageDetail | None:
    row = sqlite_store.fetch_question_image_detail(image_id)
    if row is None:
        return None

    try:
        bbox = json.loads(row["bbox_json"])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid bbox_json for image: {image_id}") from exc
    if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(value, int) for value in bbox):
        raise ValueError(f"invalid bbox_json for image: {image_id}")

    return QuestionImageDetail(
        image_id=row["image_id"],
        document_id=row["document_id"],
        section=row["section"],
        image_role=row["image_role"],
        source_page_path=row["source_page_path"],
        output_path=row["output_path"],
        bbox=bbox,
        width=row["width"],
        height=row["height"],
        report_path=row["report_path"],
    )


def question_detail(record: dict[str, Any], *, include_raw: bool = False) -> dict[str, Any]:
    fields = decoded_question_fields(record)
    revision = record_revision(record)
    result = {key: record[key] for key in (
        "document_id", "section", "page_number", "question_no", "image_id", "stem_text",
        "answer", "analysis_text", "parse_status",
    )}
    result.update(question_id=record["id"], record_revision=revision,
                  options=fields["options_json"], review_reasons=fields["review_reasons_json"],
                  image={"width": record["width"], "height": record["height"], "bbox": fields["bbox_json"],
                         "preview_url": f"/image-content/{quote(record['image_id'], safe='')}?revision={revision}"})
    if include_raw:
        result["raw_text"] = record["raw_text"]
    return result


def fetch_question_detail(sqlite_store: SQLiteStore, question_id: str) -> dict[str, Any] | None:
    with sqlite_store.question_connection() as connection:
        records = sqlite_store.question_records(connection, question_id=question_id)
        if not records or not current_question(records[0]):
            return None
        return question_detail(records[0], include_raw=True)


def search_similar_questions(query_image: str | Path, *, sqlite_store: SQLiteStore, vector_store: Any,
                             embedding_model: ImageEmbeddingModel, embedding_model_name: str,
                             top_k: int) -> list[dict[str, Any]]:
    if top_k < 1:
        raise ValueError("top_k must be greater than 0")
    signature = question_embedding_signature(embedding_model_name)
    valid: dict[str, dict[str, Any]] = {}
    checked: set[str] = set()
    count = vector_store.count()
    embedding = embedding_model.embed_image(query_image) if count else None
    # A single read transaction prevents fields from different extraction runs being combined.
    with sqlite_store.question_connection() as connection:
        if count == 0:
            return []
        limit = min(top_k, count)
        while True:
            matches = vector_store.query_similar_images(embedding, top_k=limit)
            for match in matches:
                if match.image_id in checked:
                    continue
                checked.add(match.image_id)
                records = sqlite_store.question_records(connection, image_id=match.image_id)
                if not records or not current_question(records[0]):
                    continue
                record = records[0]
                revision = record_revision(record)
                expected = {"index_schema_version": "1", "embedding_signature": signature,
                            "question_id": record["id"], "image_id": record["image_id"],
                            "document_id": record["document_id"], "section": record["section"],
                            "image_role": "question_crop", "record_revision": revision}
                if any(match.metadata.get(key) != value for key, value in expected.items()):
                    continue
                try:
                    digest = hashlib.sha256(read_question_image(record)).hexdigest()
                except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
                    continue
                if match.metadata.get("image_sha256") != digest:
                    continue
                valid[match.image_id] = {**question_detail(record), "distance": match.distance}
            if len(valid) >= top_k or limit >= count:
                break
            limit = min(count, max(limit + 1, limit * 2))
    results = sorted(valid.values(), key=lambda value: (
        value["distance"] is None, value["distance"] if value["distance"] is not None else 0, value["image_id"],
    ))[:top_k]
    return [{"rank": rank, **result} for rank, result in enumerate(results, start=1)]
