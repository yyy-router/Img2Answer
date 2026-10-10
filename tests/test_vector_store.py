from pathlib import Path
import importlib
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from img2answer.vector_store import (
    ChromaImageVectorStore,
    EmbeddedImage,
    OpenCLIPImageEmbeddingModel,
    PillowHashEmbeddingModel,
    create_embedding_model,
    embed_question_images,
)


class RecordingVectorStore:
    def __init__(self) -> None:
        self.records: list[EmbeddedImage] = []
        self.deleted_documents: list[tuple[str, str | None]] = []

    def delete_document_images(self, document_id: str, embedding_model_name: str | None = None) -> None:
        self.deleted_documents.append((document_id, embedding_model_name))
        self.records = [
            record
            for record in self.records
            if not (
                record.metadata["document_id"] == document_id
                and (embedding_model_name is None or record.metadata.get("embedding_model") == embedding_model_name)
            )
        ]

    def upsert_images(self, records: list[EmbeddedImage]) -> None:
        self.records.extend(records)


class FakeNoGrad:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None


class FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class FakeTorch:
    cuda = FakeCuda()

    @staticmethod
    def no_grad() -> FakeNoGrad:
        return FakeNoGrad()


class FakeTensor:
    def unsqueeze(self, dimension: int) -> "FakeTensor":
        self.unsqueeze_dimension = dimension
        return self

    def to(self, device: str) -> "FakeTensor":
        self.device = device
        return self


class FakeFeatures:
    def norm(self, dim: int, keepdim: bool) -> float:
        self.norm_args = (dim, keepdim)
        return 1.0

    def __truediv__(self, other: float) -> "FakeFeatures":
        self.divisor = other
        return self

    def squeeze(self, dimension: int) -> "FakeFeatures":
        self.squeeze_dimension = dimension
        return self

    def detach(self) -> "FakeFeatures":
        return self

    def cpu(self) -> "FakeFeatures":
        return self

    def tolist(self) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeModel:
    def __init__(self) -> None:
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True

    def encode_image(self, image_tensor: FakeTensor) -> FakeFeatures:
        self.image_tensor = image_tensor
        return FakeFeatures()


class FakeOpenClip:
    def __init__(self) -> None:
        self.model = FakeModel()
        self.calls: list[tuple[str, str, str]] = []

    def create_model_and_transforms(self, model_name: str, *, pretrained: str, device: str):
        self.calls.append((model_name, pretrained, device))
        return self.model, None, self.preprocess

    @staticmethod
    def preprocess(image) -> FakeTensor:
        return FakeTensor()


class VectorStoreTests(unittest.TestCase):
    def test_create_embedding_model_defaults_to_pillow_hash(self) -> None:
        model = create_embedding_model()

        self.assertIsInstance(model, PillowHashEmbeddingModel)

    def test_create_embedding_model_rejects_unknown_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported embedding model"):
            create_embedding_model("unknown")

    def test_openclip_model_reports_missing_dependencies(self) -> None:
        def raise_openclip_import(name: str):
            if name == "open_clip":
                raise ImportError("missing open_clip")
            return importlib.import_module(name)

        with patch("img2answer.vector_store.importlib.import_module", side_effect=raise_openclip_import):
            with self.assertRaisesRegex(RuntimeError, "OpenCLIP dependencies are required"):
                OpenCLIPImageEmbeddingModel()

    def test_openclip_model_embeds_image_with_loaded_dependencies(self) -> None:
        fake_openclip = FakeOpenClip()

        def fake_import(name: str):
            if name == "open_clip":
                return fake_openclip
            if name == "torch":
                return FakeTorch()
            return importlib.import_module(name)

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.png"
            Image.new("RGB", (16, 16), "black").save(image_path)

            with patch("img2answer.vector_store.importlib.import_module", side_effect=fake_import):
                model = OpenCLIPImageEmbeddingModel(model_name="fake-model", pretrained="fake-weights")
                embedding = model.embed_image(image_path)

        self.assertEqual(fake_openclip.calls, [("fake-model", "fake-weights", "cpu")])
        self.assertTrue(fake_openclip.model.eval_called)
        self.assertEqual(embedding, [0.1, 0.2, 0.3])

    def test_pillow_hash_embedding_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.png"
            Image.new("RGB", (16, 16), "black").save(image_path)
            model = PillowHashEmbeddingModel()

            first = model.embed_image(image_path)
            second = model.embed_image(image_path)

            self.assertEqual(first, second)
            self.assertEqual(len(first), 64)

    def test_embed_question_images_upserts_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "crop.png"
            Image.new("RGB", (16, 16), "white").save(image_path)
            vector_store = RecordingVectorStore()

            result = embed_question_images(
                "sample",
                [
                    {
                        "id": "sample:graphic_reasoning:page-0001:crop-001",
                        "document_id": "sample",
                        "section": "graphic_reasoning",
                        "image_role": "crop_candidate",
                        "output_path": str(image_path),
                    }
                ],
                vector_store=vector_store,
                embedding_model=PillowHashEmbeddingModel(),
            )

            self.assertEqual(result.embedded, 1)
            self.assertEqual(result.skipped_missing_files, 0)
            self.assertEqual(vector_store.deleted_documents, [("sample", "pillow-hash")])
            self.assertEqual(len(vector_store.records), 1)
            record = vector_store.records[0]
            self.assertEqual(record.image_id, "sample:graphic_reasoning:page-0001:crop-001")
            self.assertEqual(record.metadata["document_id"], "sample")
            self.assertEqual(record.metadata["section"], "graphic_reasoning")
            self.assertEqual(record.metadata["image_role"], "crop_candidate")
            self.assertEqual(record.metadata["output_path"], str(image_path))
            self.assertEqual(record.metadata["embedding_model"], "pillow-hash")

    def test_embed_question_images_skips_missing_files(self) -> None:
        vector_store = RecordingVectorStore()

        result = embed_question_images(
            "sample",
            [
                {
                    "id": "missing",
                    "document_id": "sample",
                    "section": "graphic_reasoning",
                    "image_role": "crop_candidate",
                    "output_path": "missing.png",
                }
            ],
            vector_store=vector_store,
            embedding_model=PillowHashEmbeddingModel(),
        )

        self.assertEqual(result.embedded, 0)
        self.assertEqual(result.skipped_missing_files, 1)
        self.assertEqual(vector_store.deleted_documents, [("sample", "pillow-hash")])
        self.assertEqual(vector_store.records, [])

    def test_embed_question_images_removes_obsolete_document_vectors(self) -> None:
        vector_store = RecordingVectorStore()
        vector_store.records.append(
            EmbeddedImage(
                image_id="obsolete",
                embedding=[0.1],
                metadata={
                    "image_id": "obsolete",
                    "document_id": "sample",
                    "section": "graphic_reasoning",
                    "image_role": "crop_candidate",
                    "output_path": "obsolete.png",
                    "embedding_model": "pillow-hash",
                },
                document="obsolete.png",
            )
        )
        vector_store.records.append(
            EmbeddedImage(
                image_id="other-model",
                embedding=[0.2],
                metadata={
                    "image_id": "other-model",
                    "document_id": "sample",
                    "section": "graphic_reasoning",
                    "image_role": "crop_candidate",
                    "output_path": "other-model.png",
                    "embedding_model": "openclip",
                },
                document="other-model.png",
            )
        )

        result = embed_question_images(
            "sample",
            [],
            vector_store=vector_store,
            embedding_model=PillowHashEmbeddingModel(),
        )

        self.assertEqual(result.embedded, 0)
        self.assertEqual(vector_store.deleted_documents, [("sample", "pillow-hash")])
        self.assertEqual([record.image_id for record in vector_store.records], ["other-model"])

    def test_chroma_vector_store_persists_embeddings_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            store = ChromaImageVectorStore(root / "chroma", collection_name="test_question_images")
            store.delete_document_images("sample")
            store.upsert_images(
                [
                    EmbeddedImage(
                        image_id="sample:graphic_reasoning:page-0001:crop-001",
                        embedding=[0.1, 0.2, 0.3],
                        metadata={
                            "image_id": "sample:graphic_reasoning:page-0001:crop-001",
                            "document_id": "sample",
                            "section": "graphic_reasoning",
                            "image_role": "crop_candidate",
                            "output_path": "crop.png",
                            "embedding_model": "pillow-hash",
                        },
                        document="crop.png",
                    )
                ]
            )

            reopened = ChromaImageVectorStore(root / "chroma", collection_name="test_question_images")
            self.assertEqual(reopened.collection.count(), 1)
            result = reopened.collection.get(
                ids=["sample:graphic_reasoning:page-0001:crop-001"],
                include=["metadatas", "documents"],
            )

            self.assertEqual(result["ids"], ["sample:graphic_reasoning:page-0001:crop-001"])
            self.assertEqual(result["metadatas"][0]["document_id"], "sample")
            self.assertEqual(result["metadatas"][0]["section"], "graphic_reasoning")
            self.assertEqual(result["metadatas"][0]["embedding_model"], "pillow-hash")
            self.assertEqual(result["documents"], ["crop.png"])

            matches = reopened.query_similar_images([0.1, 0.2, 0.3], top_k=1)
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0].image_id, "sample:graphic_reasoning:page-0001:crop-001")
            self.assertEqual(matches[0].metadata["document_id"], "sample")
            self.assertEqual(matches[0].document, "crop.png")

            reopened.delete_document_images("sample")
            self.assertEqual(reopened.collection.count(), 0)

    def test_chroma_delete_document_images_can_scope_to_embedding_model(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
            root = Path(directory)
            store = ChromaImageVectorStore(root / "chroma", collection_name="test_question_images_models")
            store.upsert_images(
                [
                    EmbeddedImage(
                        image_id="sample:pillow",
                        embedding=[0.1, 0.2, 0.3],
                        metadata={
                            "image_id": "sample:pillow",
                            "document_id": "sample",
                            "section": "graphic_reasoning",
                            "image_role": "crop_candidate",
                            "output_path": "pillow.png",
                            "embedding_model": "pillow-hash",
                        },
                        document="pillow.png",
                    ),
                    EmbeddedImage(
                        image_id="sample:openclip",
                        embedding=[0.2, 0.3, 0.4],
                        metadata={
                            "image_id": "sample:openclip",
                            "document_id": "sample",
                            "section": "graphic_reasoning",
                            "image_role": "crop_candidate",
                            "output_path": "openclip.png",
                            "embedding_model": "openclip",
                        },
                        document="openclip.png",
                    ),
                ]
            )

            store.delete_document_images("sample", embedding_model_name="pillow-hash")
            result = store.collection.get(include=["metadatas"])

            self.assertEqual(result["ids"], ["sample:openclip"])
            self.assertEqual(result["metadatas"][0]["embedding_model"], "openclip")


if __name__ == "__main__":
    unittest.main()
