from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

from PIL import Image


class ImageEmbeddingModel(Protocol):
    def embed_image(self, image_path: str | Path) -> list[float]:
        """Return an embedding vector for one local image."""


class ImageVectorStore(Protocol):
    def upsert_images(self, records: Sequence["EmbeddedImage"]) -> None:
        """Insert or replace image vectors by id."""


@dataclass(frozen=True)
class EmbeddedImage:
    image_id: str
    embedding: list[float]
    metadata: dict[str, str]
    document: str


@dataclass(frozen=True)
class ImageEmbeddingResult:
    embedded: int
    skipped_missing_files: int


class PillowHashEmbeddingModel:
    """Small deterministic image embedding for local POC runs."""

    def __init__(self, size: tuple[int, int] = (8, 8)):
        self.size = size

    def embed_image(self, image_path: str | Path) -> list[float]:
        with Image.open(image_path) as image:
            grayscale = image.convert("L").resize(self.size)
            pixels = grayscale.tobytes()
        return [round(1.0 - (pixel / 255.0), 6) for pixel in pixels]


class ChromaImageVectorStore:
    def __init__(self, persist_dir: str | Path, collection_name: str = "question_images"):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "ChromaDB is required when --embed-images is enabled. "
                "Install the vector dependencies from environment.yml or with: pip install 'chromadb>=0.5'"
            ) from exc

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def upsert_images(self, records: Sequence[EmbeddedImage]) -> None:
        if not records:
            return
        self.collection.upsert(
            ids=[record.image_id for record in records],
            embeddings=[record.embedding for record in records],
            metadatas=[record.metadata for record in records],
            documents=[record.document for record in records],
        )


def embed_question_images(
    question_images: Sequence[Any],
    *,
    vector_store: ImageVectorStore,
    embedding_model: ImageEmbeddingModel,
) -> ImageEmbeddingResult:
    embedded_records: list[EmbeddedImage] = []
    skipped_missing_files = 0

    for row in question_images:
        image_path = Path(_row_value(row, "output_path"))
        if not image_path.exists():
            skipped_missing_files += 1
            continue

        embedding = embedding_model.embed_image(image_path)
        if not embedding:
            raise ValueError(f"empty embedding for image: {image_path}")

        image_id = str(_row_value(row, "id"))
        embedded_records.append(
            EmbeddedImage(
                image_id=image_id,
                embedding=embedding,
                metadata={
                    "image_id": image_id,
                    "document_id": str(_row_value(row, "document_id")),
                    "section": str(_row_value(row, "section")),
                    "image_role": str(_row_value(row, "image_role")),
                    "output_path": str(image_path),
                },
                document=str(image_path),
            )
        )

    vector_store.upsert_images(embedded_records)
    return ImageEmbeddingResult(
        embedded=len(embedded_records),
        skipped_missing_files=skipped_missing_files,
    )


def _row_value(row: Any, key: str) -> Any:
    return row[key]
