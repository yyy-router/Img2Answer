import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException
from PIL import Image

import img2answer.api as api
from img2answer.store import SQLiteStore
from img2answer.vector_store import PillowHashEmbeddingModel


class EmptyVectorStore:
    def __init__(self, path, **kwargs):
        pass

    def query_similar_images(self, embedding, top_k):
        return []


class UploadLimitTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        store = SQLiteStore(self.root / "images.sqlite3")
        store.initialize()
        self.limit = 256
        self.factory = Mock(return_value=PillowHashEmbeddingModel())
        self.app = api.create_app(
            settings=api.ApiSettings(database=store.path, max_upload_bytes=self.limit),
            vector_store_factory=EmptyVectorStore,
            embedding_model_factory=self.factory,
        )

    async def test_declared_oversize_is_rejected_before_reading_or_parsing(self) -> None:
        with patch("starlette.formparsers.SpooledTemporaryFile") as spool:
            messages, reads = await _invoke_app(
                self.app, [b"not read"], headers=[(b"content-length", b"9999999")],
            )

        self.assertEqual(messages[0]["status"], 413)
        self.assertEqual(reads, 0)
        spool.assert_not_called()
        self.factory.assert_not_called()

    async def test_actual_size_is_checked_without_a_trustworthy_content_length(self) -> None:
        body = _multipart_body(b"x" * (self.limit + 64 * 1024))
        chunks = [body[:200], body[200:]]
        for headers in ([], [(b"content-length", b"1")], [(b"content-length", b"invalid")]):
            with self.subTest(headers=headers), patch("starlette.formparsers.SpooledTemporaryFile") as spool:
                messages, reads = await _invoke_app(self.app, chunks, headers=headers)
                self.assertEqual(messages[0]["status"], 413)
                self.assertEqual(reads, 2)
                spool.assert_not_called()
        self.factory.assert_not_called()

    async def test_valid_chunked_upload_without_content_length(self) -> None:
        body = _multipart_body(_image_bytes())
        messages, reads = await _invoke_app(self.app, [body[:100], body[100:]])

        self.assertEqual(messages[0]["status"], 200)
        self.assertEqual(reads, 2)
        self.assertEqual(json.loads(messages[-1]["body"])["query_image"], "query.png")
        self.factory.assert_called_once()

    async def test_file_exactly_at_limit_is_allowed(self) -> None:
        image_bytes = _image_bytes()
        payload = image_bytes + b"x" * (self.limit - len(image_bytes))
        messages, _ = await _invoke_app(self.app, [_multipart_body(payload)])

        self.assertEqual(messages[0]["status"], 200)
        self.factory.assert_called_once()

    async def test_file_above_limit_cleans_query_and_parser_files(self) -> None:
        payload = _image_bytes() + b"x" * (self.limit + 1 - len(_image_bytes()))
        query_dirs: list[Path] = []
        parser_files = []
        temporary_directory = tempfile.TemporaryDirectory
        spooled_temporary_file = tempfile.SpooledTemporaryFile

        def make_query_directory(**kwargs):
            directory = temporary_directory(dir=self.root, **kwargs)
            query_dirs.append(Path(directory.name))
            return directory

        def make_parser_file(**kwargs):
            file = spooled_temporary_file(**kwargs)
            parser_files.append(file)
            return file

        with patch("img2answer.api.tempfile.TemporaryDirectory", side_effect=make_query_directory), patch(
            "starlette.formparsers.SpooledTemporaryFile", side_effect=make_parser_file,
        ):
            messages, _ = await _invoke_app(self.app, [_multipart_body(payload)])

        self.assertEqual(messages[0]["status"], 413)
        self.assertTrue(query_dirs)
        self.assertTrue(all(not directory.exists() for directory in query_dirs))
        self.assertTrue(parser_files)
        self.assertTrue(all(file.closed for file in parser_files))
        self.factory.assert_not_called()

    async def test_request_exactly_at_limit_is_allowed(self) -> None:
        request_limit = self.limit + 64 * 1024
        base_body = _multipart_body(_image_bytes(), padding=b"")
        body = _multipart_body(_image_bytes(), padding=b"x" * (request_limit - len(base_body)))
        self.assertEqual(len(body), request_limit)
        messages, _ = await _invoke_app(
            self.app, [body], headers=[(b"content-length", str(len(body)).encode())],
        )

        self.assertEqual(messages[0]["status"], 200)

    async def test_request_one_byte_over_limit_is_rejected(self) -> None:
        request_limit = self.limit + 64 * 1024
        base_body = _multipart_body(_image_bytes(), padding=b"")
        body = _multipart_body(_image_bytes(), padding=b"x" * (request_limit + 1 - len(base_body)))
        with patch("starlette.formparsers.SpooledTemporaryFile") as spool:
            messages, _ = await _invoke_app(self.app, [body])

        self.assertEqual(messages[0]["status"], 413)
        spool.assert_not_called()
        self.factory.assert_not_called()

    async def test_disconnect_during_upload_does_not_parse_or_load_model(self) -> None:
        with patch("starlette.formparsers.SpooledTemporaryFile") as spool:
            messages, reads = await _invoke_app(self.app, [b"partial upload"], disconnect=True)

        self.assertEqual(messages, [])
        self.assertEqual(reads, 2)
        spool.assert_not_called()
        self.factory.assert_not_called()

    async def test_detail_requests_do_not_load_model_or_read_upload_body(self) -> None:
        messages, reads = await _invoke_app(
            self.app, [b"not read"], method="GET", path="/images/missing",
            headers=[(b"content-length", b"9999999")],
        )

        self.assertEqual(messages[0]["status"], 404)
        self.assertEqual(reads, 0)
        self.factory.assert_not_called()

    def test_bounded_copy_stops_before_writing_excess_bytes(self) -> None:
        source = io.BytesIO(b"x" * 100001)
        destination = io.BytesIO()
        with self.assertRaises(HTTPException) as error:
            api._copy_uploaded_image(source, destination, max_bytes=100000)

        self.assertEqual(error.exception.status_code, 413)
        self.assertLessEqual(destination.tell(), 100000)
        self.assertLessEqual(source.tell(), 100001)


async def _invoke_app(app, chunks, *, headers=(), method="POST", path="/search/image", disconnect=False):
    events = [
        {"type": "http.request", "body": chunk, "more_body": index < len(chunks) - 1 or disconnect}
        for index, chunk in enumerate(chunks)
    ]
    if disconnect:
        events.append({"type": "http.disconnect"})
    received = iter(events)
    messages = []
    reads = 0

    async def receive():
        nonlocal reads
        reads += 1
        return next(received)

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "scheme": "http", "method": method, "path": path, "raw_path": path.encode(),
        "root_path": "", "query_string": b"",
        "headers": [(b"content-type", b"multipart/form-data; boundary=upload-test"), *headers],
        "server": ("127.0.0.1", 8000), "client": ("127.0.0.1", 54321),
    }
    await app(scope, receive, send)
    return messages, reads


def _multipart_body(payload: bytes, *, padding: bytes | None = None) -> bytes:
    body = (
        b'--upload-test\r\nContent-Disposition: form-data; name="file"; filename="query.png"\r\n'
        b"Content-Type: image/png\r\n\r\n" + payload + b"\r\n"
    )
    if padding is not None:
        body += b'--upload-test\r\nContent-Disposition: form-data; name="padding"\r\n\r\n' + padding + b"\r\n"
    return body + b"--upload-test--\r\n"


def _image_bytes() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(stream, format="PNG")
    return stream.getvalue()


if __name__ == "__main__":
    unittest.main()
