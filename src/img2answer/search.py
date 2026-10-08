from __future__ import annotations

from dataclasses import dataclass
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
        results.append(
            ImageSearchResult(
                rank=rank,
                image_id=match.image_id,
                distance=match.distance,
                metadata=match.metadata,
                record=dict(row) if row is not None else None,
            )
        )
    return results
