from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any, Protocol, Sequence

from PIL import Image


class ImageEmbeddingModel(Protocol):
    def embed_image(self, image_path: str | Path) -> list[float]:
        """Return an embedding vector for one local image."""


class ImageVectorStore(Protocol):
    def delete_document_images(self, document_id: str, embedding_model_name: str | None = None) -> None:
        """Remove image vectors for one document."""

    def upsert_images(self, records: Sequence["EmbeddedImage"]) -> None:
        """Insert or replace image vectors by id."""

    def query_similar_images(self, embedding: Sequence[float], top_k: int) -> list["ImageSearchMatch"]:
        """Return nearest image vectors for one query embedding."""


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


@dataclass(frozen=True)
class ImageSearchMatch:
    image_id: str
    distance: float | None
    metadata: dict[str, str]
    document: str | None


class PillowHashEmbeddingModel:
    """Small deterministic image embedding for local POC runs."""

    def __init__(self, size: tuple[int, int] = (8, 8)):
        self.size = size

    def embed_image(self, image_path: str | Path) -> list[float]:
        with Image.open(image_path) as image:
            grayscale = image.convert("L").resize(self.size)
            pixels = grayscale.tobytes()
        return [round(1.0 - (pixel / 255.0), 6) for pixel in pixels]


class OpenCLIPImageEmbeddingModel:
    """OpenCLIP image embedding model for local visual retrieval."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
    ):
        try:
            open_clip = importlib.import_module("open_clip")
            torch = importlib.import_module("torch")
        except ImportError as exc:
            raise RuntimeError(
                "OpenCLIP dependencies are required when --embedding-model openclip is used. "
                "Install them with: pip install -e '.[openclip]'"
            ) from exc

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.model.eval()

    def embed_image(self, image_path: str | Path) -> list[float]:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            image_tensor = self.preprocess(rgb).unsqueeze(0).to(self.device)

        with self.torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return [float(value) for value in features.squeeze(0).detach().cpu().tolist()]


def create_embedding_model(name: str = "pillow-hash") -> ImageEmbeddingModel:
    normalized = name.strip().lower()
    if normalized == "pillow-hash":
        return PillowHashEmbeddingModel()
    if normalized == "openclip":
        return OpenCLIPImageEmbeddingModel()
    raise ValueError(f"unsupported embedding model: {name}")


class QuestionIndexUnavailableError(RuntimeError):
    """An existing compatible question index is required."""


def question_embedding_signature(name: str) -> str:
    signatures = {
        "pillow-hash": "pillow-hash:8x8:L:inverse-255:round-6:v1",
        "openclip": "openclip:ViT-B-32:laion2b_s34b_b79k:l2:v1",
    }
    try:
        return signatures[name]
    except KeyError:
        raise ValueError("unsupported question embedding model") from None


def resolve_question_collection(collection_name: str | None, model_name: str) -> str:
    question_embedding_signature(model_name)
    return collection_name or ("question_candidates" if model_name == "pillow-hash"
                               else "question_candidates_openclip")


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

    def delete_document_images(self, document_id: str, embedding_model_name: str | None = None) -> None:
        if embedding_model_name is None:
            self.collection.delete(where={"document_id": document_id})
            return
        self.collection.delete(
            where={
                "$and": [
                    {"document_id": document_id},
                    {"embedding_model": embedding_model_name},
                ]
            }
        )

    def upsert_images(self, records: Sequence[EmbeddedImage]) -> None:
        if not records:
            return
        self.collection.upsert(
            ids=[record.image_id for record in records],
            embeddings=[record.embedding for record in records],
            metadatas=[record.metadata for record in records],
            documents=[record.document for record in records],
        )

    def query_similar_images(self, embedding: Sequence[float], top_k: int) -> list[ImageSearchMatch]:
        if top_k < 1:
            raise ValueError("top_k must be greater than 0")
        result = self.collection.query(
            query_embeddings=[list(embedding)],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        documents = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        matches: list[ImageSearchMatch] = []
        for index, image_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            document = documents[index] if index < len(documents) else None
            distance = distances[index] if index < len(distances) else None
            matches.append(
                ImageSearchMatch(
                    image_id=str(image_id),
                    distance=float(distance) if distance is not None else None,
                    metadata={str(key): str(value) for key, value in metadata.items()},
                    document=str(document) if document is not None else None,
                )
            )
        return matches


class ChromaQuestionVectorStore(ChromaImageVectorStore):
    """A separately identified collection; searches never create a missing index."""

    def __init__(self, persist_dir: str | Path, collection_name: str = "question_candidates", *,
                 embedding_model_name: str = "pillow-hash", create: bool = False):
        import chromadb
        from chromadb import errors

        # Chroma 0.5 reports a missing collection as ValueError; later versions use typed errors.
        missing_collection_errors = (ValueError,) + tuple(
            getattr(errors, name) for name in ("NotFoundError", "InvalidCollectionException")
            if hasattr(errors, name)
        )

        expected = {"index_kind": "question_candidates", "index_schema_version": 1,
                    "embedding_signature": question_embedding_signature(embedding_model_name)}
        self.persist_dir = Path(persist_dir)
        if not create and not (self.persist_dir / "chroma.sqlite3").is_file():
            raise QuestionIndexUnavailableError("单题索引不可用，请先建立单题索引")
        if create:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except missing_collection_errors:
            if not create:
                raise QuestionIndexUnavailableError("单题索引不可用，请先建立单题索引") from None
            self.collection = self.client.create_collection(name=collection_name, metadata=expected)
        if any((self.collection.metadata or {}).get(key) != value for key, value in expected.items()):
            raise QuestionIndexUnavailableError("单题索引用途、版本或模型不匹配，请使用独立 collection 重建")

    def count(self) -> int:
        return self.collection.count()

    def document_ids(self, document_id: str, signature: str) -> set[str]:
        result = self.collection.get(where={"$and": [
            {"document_id": document_id}, {"image_role": "question_crop"},
            {"section": "graphic_reasoning"},
            {"embedding_signature": signature},
        ]}, include=[])
        return set(result["ids"])

    def delete_ids(self, ids: Sequence[str]) -> None:
        if ids:
            self.collection.delete(ids=list(ids))


def embed_question_images(
    document_id: str,
    question_images: Sequence[Any],
    *,
    vector_store: ImageVectorStore,
    embedding_model: ImageEmbeddingModel,
    embedding_model_name: str = "pillow-hash",
) -> ImageEmbeddingResult:
    embedded_records: list[EmbeddedImage] = []
    skipped_missing_files = 0
    vector_store.delete_document_images(document_id, embedding_model_name=embedding_model_name)

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
                    "embedding_model": embedding_model_name,
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
