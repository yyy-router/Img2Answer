from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Callable

from PIL import Image

from .store import SQLiteStore
from .vector_store import (
    ChromaQuestionVectorStore, EmbeddedImage, ImageEmbeddingModel, create_embedding_model,
    question_embedding_signature, resolve_question_collection,
)


class QuestionIndexError(RuntimeError):
    """A safe summary of an indexing failure."""


def current_question(record: dict[str, Any]) -> bool:
    linked = (
        record["section"] == "graphic_reasoning" and record["image_role"] == "question_crop"
        and record["image_id"] == record["joined_image_id"]
        and record["document_id"] == record["image_document_id"]
        and record["section"] == record["image_section"]
        and record["source_sha256"] == record["import_sha256"]
        and record["source_updated_at"] is not None and record["report_created_at"] is not None
        and bool(record["report_path"])
        and 1 <= record["page_number"] <= record["import_page_count"]
    )
    if not linked:
        return False
    errors = json.loads(record["import_errors_json"])
    if not isinstance(errors, list) or any(not isinstance(error, str) for error in errors):
        raise ValueError("Invalid import errors")
    return not errors


def decoded_question_fields(record: dict[str, Any]) -> dict[str, Any]:
    options = json.loads(record["options_json"])
    reasons = json.loads(record["review_reasons_json"])
    bbox = json.loads(record["bbox_json"])
    if (not isinstance(options, dict)
            or any(key not in {"A", "B", "C", "D"} or not isinstance(value, str)
                   for key, value in options.items())
            or not isinstance(reasons, list) or any(not isinstance(value, str) for value in reasons)
            or not isinstance(bbox, list) or len(bbox) != 4
            or any(type(value) is not int for value in bbox)
            or not (0 <= bbox[0] < bbox[2] and 0 <= bbox[1] < bbox[3])
            or (record["width"], record["height"]) != (bbox[2] - bbox[0], bbox[3] - bbox[1])
            or record["parse_status"] != "needs_review"
            or record["answer"] not in {None, "A", "B", "C", "D"}):
        raise ValueError("Invalid structured question fields")
    return {"options_json": options, "review_reasons_json": reasons, "bbox_json": bbox}


def record_revision(record: dict[str, Any]) -> str:
    normalized = {**record, **decoded_question_fields(record)}
    data = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def validate_question_image(data: bytes, record: dict[str, Any]) -> str:
    with Image.open(io.BytesIO(data)) as image:
        if (image.format not in {"PNG", "JPEG"} or getattr(image, "n_frames", 1) != 1
                or image.size != (record["width"], record["height"])):
            raise ValueError("Invalid question image")
        mime = "image/png" if image.format == "PNG" else "image/jpeg"
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        image.load()
    return mime


def read_question_image(record: dict[str, Any]) -> bytes:
    data = Path(record["output_path"]).read_bytes()
    validate_question_image(data, record)
    return data


def index_questions(database: str | Path, document_id: str, chroma_dir: str | Path, *,
                    embedding_model_name: str = "pillow-hash", collection_name: str | None = None,
                    embedding_model: ImageEmbeddingModel | None = None,
                    vector_store_factory: Callable[..., ChromaQuestionVectorStore] = ChromaQuestionVectorStore,
                    ) -> dict[str, Any]:
    """Prepare all embeddings first; publish only while their SQLite version is current."""
    publishing = False
    try:
        signature = question_embedding_signature(embedding_model_name)
        collection_name = resolve_question_collection(collection_name, embedding_model_name)
        store = SQLiteStore(database)
        with store.question_connection() as connection:
            state = store.question_document_state(connection, document_id)
            records = store.question_records(connection, document_id=document_id)
        if any(not current_question(record) for record in records):
            raise QuestionIndexError("题目与当前来源关联无效，请重新提取")
        revisions = {record["id"]: record_revision(record) for record in records}
        hashes: dict[str, str] = {}
        vectors = []
        dimension = None
        with tempfile.TemporaryDirectory(prefix="img2answer-index-") as directory:
            paths = []
            for ordinal, record in enumerate(records):
                data = read_question_image(record)
                hashes[record["id"]] = hashlib.sha256(data).hexdigest()
                path = Path(directory) / f"question-{ordinal}.image"
                path.write_bytes(data)
                paths.append(path)
            model = (embedding_model or create_embedding_model(embedding_model_name)) if records else None
            for record, path in zip(records, paths):
                try:
                    embedding = list(model.embed_image(path))
                finally:
                    path.unlink(missing_ok=True)
                if (not embedding or any(isinstance(value, bool) or not isinstance(value, (int, float))
                                         or not math.isfinite(value) for value in embedding)
                        or dimension is not None and dimension != len(embedding)):
                    raise QuestionIndexError("模型返回无效向量，原索引未更新")
                dimension = len(embedding)
                vectors.append(EmbeddedImage(
                    image_id=record["image_id"], embedding=embedding, document=record["id"],
                    metadata={
                        "index_schema_version": "1", "question_id": record["id"],
                        "image_id": record["image_id"], "document_id": document_id,
                        "section": record["section"], "image_role": record["image_role"],
                        "record_revision": revisions[record["id"]], "image_sha256": hashes[record["id"]],
                        "embedding_signature": signature, "embedding_model": embedding_model_name,
                    },
                ))
        with store.question_connection(for_update=True) as connection:
            current = store.question_records(connection, document_id=document_id)
            if (store.question_document_state(connection, document_id) != state
                    or any(not current_question(record) for record in current)
                    or {record["id"]: record_revision(record) for record in current} != revisions
                    or any(hashlib.sha256(read_question_image(record)).hexdigest() != hashes[record["id"]]
                           for record in current)):
                raise QuestionIndexError("索引期间来源、题目或截图已变化，请重新运行索引")
            vector_store = vector_store_factory(chroma_dir, collection_name=collection_name,
                                               embedding_model_name=embedding_model_name, create=True)
            old_ids = vector_store.document_ids(document_id, signature)
            stale = sorted(old_ids - {record.image_id for record in vectors})
            publishing = True
            for offset in range(0, len(vectors), 256):
                vector_store.upsert_images(vectors[offset:offset + 256])
            for offset in range(0, len(stale), 256):
                vector_store.delete_ids(stale[offset:offset + 256])
        return {"document_id": document_id, "embedding_model": embedding_model_name,
                "chroma_collection": collection_name, "indexed": len(vectors), "deleted": len(stale)}
    except QuestionIndexError:
        raise
    except Exception as exc:
        if publishing:
            raise QuestionIndexError("单题索引发布失败，可能部分写入；请重新运行索引恢复") from exc
        raise QuestionIndexError("单题索引准备失败，请检查数据库、图片、模型和 collection；尚未发布向量") from exc
