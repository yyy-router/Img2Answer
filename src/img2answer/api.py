from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sqlite3
import tempfile
from threading import Lock
from typing import BinaryIO, Callable

from dotenv import dotenv_values
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from starlette.staticfiles import StaticFiles

from .cli import resolve_chroma_collection
from .question_index import current_question, record_revision, validate_question_image
from .search import fetch_question_detail, fetch_question_image_detail, search_similar_images, search_similar_questions
from .store import QuestionDataUnavailableError, SQLiteStore
from .vector_store import (
    ChromaImageVectorStore, ChromaQuestionVectorStore, ImageEmbeddingModel,
    QuestionIndexUnavailableError, create_embedding_model, resolve_question_collection,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ApiSettings:
    database: Path = Path("data/processed/img2answer.sqlite3")
    chroma_dir: Path = Path("data/processed/chroma")
    chroma_collection: str | None = None
    embedding_model: str = "pillow-hash"
    top_k: int = 5
    max_upload_bytes: int = 10 * 1024 * 1024
    question_chroma_collection: str | None = None
    image_root: Path = Path("data/processed")
    web_dist: Path = Path("frontend/dist")

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("IMG2ANSWER_TOP_K must be greater than 0")
        if self.max_upload_bytes < 1:
            raise ValueError("IMG2ANSWER_MAX_UPLOAD_BYTES must be greater than 0")
        if self.embedding_model not in {"pillow-hash", "openclip"}:
            raise ValueError("IMG2ANSWER_EMBEDDING_MODEL must be pillow-hash or openclip")

    @classmethod
    def from_env(cls) -> ApiSettings:
        file_values = dotenv_values(Path.cwd() / ".env", encoding="utf-8-sig", interpolate=False)
        values = {key: value for key, value in file_values.items() if value is not None}
        values.update(os.environ)
        return cls(
            database=Path(values.get("IMG2ANSWER_DATABASE", "data/processed/img2answer.sqlite3")),
            chroma_dir=Path(values.get("IMG2ANSWER_CHROMA_DIR", "data/processed/chroma")),
            chroma_collection=values.get("IMG2ANSWER_CHROMA_COLLECTION") or None,
            embedding_model=values.get("IMG2ANSWER_EMBEDDING_MODEL", "pillow-hash"),
            top_k=int(values.get("IMG2ANSWER_TOP_K", "5")),
            max_upload_bytes=int(values.get("IMG2ANSWER_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))),
            question_chroma_collection=values.get("IMG2ANSWER_QUESTION_CHROMA_COLLECTION") or None,
            image_root=Path(values.get("IMG2ANSWER_IMAGE_ROOT") or "data/processed"),
            web_dist=Path(values.get("IMG2ANSWER_WEB_DIST") or "frontend/dist"),
        )


StoreFactory = Callable[[str | Path], SQLiteStore]
VectorStoreFactory = Callable[..., ChromaImageVectorStore]
EmbeddingModelFactory = Callable[[str], ImageEmbeddingModel]


class UploadRequestLimitMiddleware:
    def __init__(self, app: ASGIApp, max_body_bytes: int):
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return

        rejection = JSONResponse(status_code=413, content={"detail": "Request body exceeds configured size limit"})
        for name, value in scope["headers"]:
            if name == b"content-length":
                try:
                    declared_size = int(value)
                except ValueError:
                    continue
                if declared_size > self.max_body_bytes:
                    await rejection(scope, receive, send)
                    return

        # Bound the body before the multipart parser can spool any files to disk.
        buffered_body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(buffered_body) + len(chunk) > self.max_body_bytes:
                await rejection(scope, receive, send)
                return
            buffered_body.extend(chunk)
            if not message.get("more_body", False):
                break

        body = bytes(buffered_body)
        buffered_body.clear()
        replayed = False

        async def replay_receive() -> Message:
            nonlocal replayed
            if replayed:
                return await receive()
            replayed = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay_receive, send)


def create_app(
    *,
    settings: ApiSettings | None = None,
    store_factory: StoreFactory = SQLiteStore,
    vector_store_factory: VectorStoreFactory = ChromaImageVectorStore,
    embedding_model_factory: EmbeddingModelFactory = create_embedding_model,
    question_vector_store_factory: Callable[..., ChromaQuestionVectorStore] = ChromaQuestionVectorStore,
) -> FastAPI:
    app = FastAPI(title="Img2Answer API", version="0.1.0")
    app.state.settings = settings if settings is not None else ApiSettings.from_env()
    app.state.store_factory = store_factory
    app.state.vector_store_factory = vector_store_factory
    app.state.question_vector_store_factory = question_vector_store_factory
    app.state.embedding_model_factory = embedding_model_factory
    app.state.embedding_model = None
    app.state.embedding_model_lock = Lock()
    app.add_middleware(
        UploadRequestLimitMiddleware,
        max_body_bytes=app.state.settings.max_upload_bytes + 64 * 1024,
    )

    @app.post("/search/image")
    def search_image(
        request: Request,
        file: UploadFile = File(...),
        top_k: int | None = Form(None),
    ) -> dict:
        config = request.app.state.settings
        result_count = config.top_k if top_k is None else top_k
        if result_count < 1:
            raise HTTPException(status_code=400, detail="top_k must be greater than 0")
        collection_name = resolve_chroma_collection(config.chroma_collection, config.embedding_model)

        try:
            # Existing embedding models accept paths; uploaded files live only for this request.
            with tempfile.TemporaryDirectory(prefix="img2answer-query-") as directory:
                image_path = Path(directory) / "query-image"
                with image_path.open("wb") as destination:
                    _copy_uploaded_image(file.file, destination, max_bytes=config.max_upload_bytes)
                _validate_uploaded_image(image_path)

                sqlite_store = _create_sqlite_store(request.app.state.store_factory, config.database)
                vector_store = request.app.state.vector_store_factory(
                    config.chroma_dir,
                    collection_name=collection_name,
                )
                embedding_model = _get_embedding_model(request.app)
                results = search_similar_images(
                    image_path,
                    sqlite_store=sqlite_store,
                    vector_store=vector_store,
                    embedding_model=embedding_model,
                    top_k=result_count,
                )
        except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
            logger.exception("Image search failed")
            raise HTTPException(status_code=500, detail="Image search failed; check server logs") from exc

        return {
            "query_image": (file.filename or "upload").replace("\\", "/").rsplit("/", 1)[-1],
            "top_k": result_count,
            "embedding_model": config.embedding_model,
            "chroma_collection": collection_name,
            "results": [result.__dict__ for result in results],
        }

    @app.get("/images/{image_id:path}")
    def get_image_detail(image_id: str, request: Request) -> dict:
        try:
            sqlite_store = _create_sqlite_store(
                request.app.state.store_factory, request.app.state.settings.database,
            )
            detail = fetch_question_image_detail(sqlite_store, image_id)
        except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
            logger.exception("Image detail lookup failed")
            raise HTTPException(status_code=500, detail="Image detail lookup failed; check server logs") from exc
        if detail is None:
            raise HTTPException(status_code=404, detail=f"image not found: {image_id}")
        return detail.__dict__

    @app.post("/search/question")
    def search_question(request: Request, file: UploadFile = File(...), top_k: int | None = Form(None)) -> dict:
        config = request.app.state.settings
        result_count = config.top_k if top_k is None else top_k
        if result_count < 1:
            raise HTTPException(status_code=400, detail="top_k must be greater than 0")
        collection_name = resolve_question_collection(config.question_chroma_collection, config.embedding_model)
        try:
            with tempfile.TemporaryDirectory(prefix="img2answer-query-") as directory:
                image_path = Path(directory) / "query-image"
                with image_path.open("wb") as destination:
                    _copy_uploaded_image(file.file, destination, max_bytes=config.max_upload_bytes)
                _validate_uploaded_image(image_path)
                store = _create_sqlite_store(request.app.state.store_factory, config.database)
                # Check schema before opening Chroma or loading a model.
                with store.question_connection():
                    pass
                vectors = request.app.state.question_vector_store_factory(
                    config.chroma_dir, collection_name=collection_name,
                    embedding_model_name=config.embedding_model, create=False,
                )
                results = search_similar_questions(
                    image_path, sqlite_store=store, vector_store=vectors,
                    embedding_model=_get_embedding_model(request.app),
                    embedding_model_name=config.embedding_model, top_k=result_count,
                )
        except HTTPException:
            raise
        except (QuestionDataUnavailableError, QuestionIndexUnavailableError) as exc:
            raise HTTPException(status_code=503, detail="Question data or index unavailable; import and index questions") from exc
        except Exception as exc:
            logger.exception("Question search failed")
            raise HTTPException(status_code=500, detail="Question search failed; check server logs") from exc
        return {"query_image": (file.filename or "upload").replace("\\", "/").rsplit("/", 1)[-1],
                "top_k": result_count, "embedding_model": config.embedding_model,
                "chroma_collection": collection_name, "results": results}

    @app.get("/questions/{question_id:path}")
    def get_question_detail(question_id: str, request: Request) -> dict:
        try:
            store = _create_sqlite_store(request.app.state.store_factory, request.app.state.settings.database)
            detail = fetch_question_detail(store, question_id)
        except HTTPException:
            raise
        except QuestionDataUnavailableError as exc:
            raise HTTPException(status_code=503, detail="Question data unavailable; import and extract questions") from exc
        except Exception as exc:
            logger.exception("Question detail lookup failed")
            raise HTTPException(status_code=500, detail="Question detail lookup failed; check server logs") from exc
        if detail is None:
            raise HTTPException(status_code=404, detail="Question not found")
        return detail

    @app.get("/image-content/{image_id:path}")
    def get_image_content(image_id: str, request: Request, revision: str | None = None) -> Response:
        config = request.app.state.settings
        try:
            store = _create_sqlite_store(request.app.state.store_factory, config.database)
            with store.question_connection() as connection:
                records = store.question_records(connection, image_id=image_id)
                if not records or not current_question(records[0]):
                    raise HTTPException(status_code=404, detail="Image not found")
                record = records[0]
                current_revision = record_revision(record)
                if revision is not None and revision != current_revision:
                    raise HTTPException(status_code=409, detail="Question changed; refresh question details")
                try:
                    root = config.image_root.resolve(strict=True)
                    path = Path(record["output_path"]).resolve(strict=True)
                    if not path.is_relative_to(root) or not path.is_file():
                        raise ValueError("Image outside permitted root")
                    data = path.read_bytes()
                    mime = validate_question_image(data, record)
                except (OSError, ValueError, RuntimeError, SyntaxError, Image.DecompressionBombError):
                    raise HTTPException(status_code=404, detail="Image not found") from None
        except HTTPException:
            raise
        except QuestionDataUnavailableError as exc:
            raise HTTPException(status_code=503, detail="Question data unavailable; import and extract questions") from exc
        except Exception as exc:
            logger.exception("Question image lookup failed")
            raise HTTPException(status_code=500, detail="Image lookup failed; check server logs") from exc
        return Response(content=data, media_type=mime, headers={"Cache-Control": "no-store"})

    web_dist = app.state.settings.web_dist.resolve()
    if (web_dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=web_dist / "assets"), name="web-assets")

    @app.get("/", include_in_schema=False)
    def get_web_app() -> FileResponse:
        index = web_dist / "index.html"
        if not index.is_file():
            raise HTTPException(status_code=503, detail=(
                "前端尚未构建，请执行 npm --prefix frontend ci 和 npm --prefix frontend run build，然后重启服务。"
            ))
        return FileResponse(index, media_type="text/html", headers={"Cache-Control": "no-store"})

    return app


def _get_embedding_model(app: FastAPI) -> ImageEmbeddingModel:
    with app.state.embedding_model_lock:
        if app.state.embedding_model is None:
            app.state.embedding_model = app.state.embedding_model_factory(app.state.settings.embedding_model)
        return app.state.embedding_model


def _copy_uploaded_image(source: BinaryIO, destination: BinaryIO, *, max_bytes: int) -> None:
    copied_bytes = 0
    while chunk := source.read(min(64 * 1024, max_bytes - copied_bytes + 1)):
        copied_bytes += len(chunk)
        if copied_bytes > max_bytes:
            raise HTTPException(status_code=413, detail="Uploaded image exceeds configured size limit")
        destination.write(chunk)


def _validate_uploaded_image(image_path: Path) -> None:
    if image_path.stat().st_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            image.load()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc


def _create_sqlite_store(store_factory: StoreFactory, database: str | Path) -> SQLiteStore:
    if not Path(database).is_file():
        raise HTTPException(status_code=503, detail="Server database is unavailable")
    return store_factory(database)


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run("img2answer.api:app", host="127.0.0.1", port=8000)
