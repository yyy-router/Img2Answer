from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Protocol

from .store import SQLiteStore
from .vector_store import ImageEmbeddingModel, ImageSearchMatch


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
    if not isinstance(bbox, list) or not all(isinstance(value, int) for value in bbox):
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
