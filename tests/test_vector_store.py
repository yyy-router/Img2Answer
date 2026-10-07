from pathlib import Path
import tempfile
import unittest

from PIL import Image

from img2answer.vector_store import (
    ChromaImageVectorStore,
    EmbeddedImage,
    PillowHashEmbeddingModel,
    embed_question_images,
)


class RecordingVectorStore:
    def __init__(self) -> None:
        self.records: list[EmbeddedImage] = []
        self.deleted_document_ids: list[str] = []

    def delete_document_images(self, document_id: str) -> None:
        self.deleted_document_ids.append(document_id)
        self.records = [record for record in self.records if record.metadata["document_id"] != document_id]

    def upsert_images(self, records: list[EmbeddedImage]) -> None:
        self.records.extend(records)


class VectorStoreTests(unittest.TestCase):
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
            self.assertEqual(vector_store.deleted_document_ids, ["sample"])
            self.assertEqual(len(vector_store.records), 1)
            record = vector_store.records[0]
            self.assertEqual(record.image_id, "sample:graphic_reasoning:page-0001:crop-001")
            self.assertEqual(record.metadata["document_id"], "sample")
            self.assertEqual(record.metadata["section"], "graphic_reasoning")
            self.assertEqual(record.metadata["image_role"], "crop_candidate")
            self.assertEqual(record.metadata["output_path"], str(image_path))

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
        self.assertEqual(vector_store.deleted_document_ids, ["sample"])
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
                },
                document="obsolete.png",
            )
        )

        result = embed_question_images(
            "sample",
            [],
            vector_store=vector_store,
            embedding_model=PillowHashEmbeddingModel(),
        )

        self.assertEqual(result.embedded, 0)
        self.assertEqual(vector_store.deleted_document_ids, ["sample"])
        self.assertEqual(vector_store.records, [])

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
            self.assertEqual(result["documents"], ["crop.png"])

            reopened.delete_document_images("sample")
            self.assertEqual(reopened.collection.count(), 0)


if __name__ == "__main__":
    unittest.main()
