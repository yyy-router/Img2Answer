from pathlib import Path
import tempfile
import unittest

from PIL import Image

from img2answer.vector_store import EmbeddedImage, PillowHashEmbeddingModel, embed_question_images


class RecordingVectorStore:
    def __init__(self) -> None:
        self.records: list[EmbeddedImage] = []

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
        self.assertEqual(vector_store.records, [])


if __name__ == "__main__":
    unittest.main()
