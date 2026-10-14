import builtins
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import traceback
import unittest
from unittest.mock import patch

from PIL import Image

from img2answer import ocr
from img2answer.ocr import AliyunOCRClient, OCRError, OCRSettings, normalize_response, recognize_page


def sample_response(width=100, height=120, confidence=95, text="示例题干"):
    words = [] if not text else [{
        "word": text,
        "prob": confidence,
        "pos": [{"x": 1, "y": 2}, {"x": width - 1, "y": 2},
                {"x": width - 1, "y": height - 1}, {"x": 1, "y": height - 1}],
    }]
    data = {"content": text, "width": width, "height": height,
            "orgWidth": width, "orgHeight": height, "prism_wordsInfo": words}
    return {"RequestId": "request-123", "Data": json.dumps(data, ensure_ascii=False)}


class RecordingOCRClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []
        self.streams = []

    def recognize(self, stream):
        self.streams.append(stream)
        data = stream.read()
        self.calls.append(data)
        if self.error is not None:
            raise self.error
        if self.response is not None:
            return self.response
        with Image.open(io.BytesIO(data)) as image:
            return sample_response(*image.size)


def settings(**kwargs):
    return OCRSettings(access_key_id="local-dummy-id", access_key_secret="local-dummy-secret", **kwargs)


class OCRSettingsTests(unittest.TestCase):
    def test_settings_hide_both_credentials(self):
        config = settings()
        self.assertNotIn(config.access_key_id, repr(config))
        self.assertNotIn(config.access_key_secret, str(config))
        self.assertEqual(config.connect_timeout_seconds, 5)
        self.assertEqual(config.read_timeout_seconds, 30)

    def test_config_boundaries_and_invalid_values(self):
        for confidence in (0, 80.5, 100):
            self.assertEqual(settings(min_confidence=confidence).min_confidence, confidence)
        invalid = [
            {"connect_timeout_seconds": 0}, {"connect_timeout_seconds": True},
            {"read_timeout_seconds": -1}, {"read_timeout_seconds": 1.5},
            {"min_confidence": -1}, {"min_confidence": 101},
            {"min_confidence": float("nan")}, {"min_confidence": float("inf")},
            {"min_confidence": True}, {"min_confidence": "80"}, {"min_confidence": 10 ** 400},
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(OCRError):
                settings(**values)
        for key in ("access_key_id", "access_key_secret"):
            values = {"access_key_id": "id", "access_key_secret": "secret", key: " "}
            with self.subTest(key=key), self.assertRaises(OCRError):
                OCRSettings(**values)

    def test_dotenv_bom_quotes_priority_and_no_environment_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text(
                '# 本地配置\nALIBABA_CLOUD_ACCESS_KEY_ID="file-id"\n'
                'ALIBABA_CLOUD_ACCESS_KEY_SECRET="${NOT_EXPANDED}"\n'
                'IMG2ANSWER_OCR_CONNECT_TIMEOUT_SECONDS=7\nIMG2ANSWER_OCR_MIN_CONFIDENCE=90.5\n',
                encoding="utf-8-sig",
            )
            with patch.object(Path, "cwd", return_value=root), patch.dict(os.environ, {"ALIBABA_CLOUD_ACCESS_KEY_ID": "env-id"}, clear=True):
                before = dict(os.environ)
                config = OCRSettings.from_env()
                self.assertEqual(config.access_key_id, "env-id")
                self.assertEqual(config.access_key_secret, "${NOT_EXPANDED}")
                self.assertEqual(config.connect_timeout_seconds, 7)
                self.assertEqual(config.min_confidence, 90.5)
                self.assertEqual(dict(os.environ), before)

    def test_absent_dotenv_and_invalid_config_are_safe(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(Path, "cwd", return_value=Path(directory)):
            with patch.dict(os.environ, {}, clear=True), self.assertRaises(OCRError):
                OCRSettings.from_env()
            environment = {"ALIBABA_CLOUD_ACCESS_KEY_ID": "id", "ALIBABA_CLOUD_ACCESS_KEY_SECRET": "secret"}
            with patch.dict(os.environ, environment, clear=True):
                self.assertEqual(OCRSettings.from_env().min_confidence, 80)
            for value in ("", "secret-value", "0", "-1", "1.2"):
                with self.subTest(value=value), patch.dict(os.environ, {**environment, "IMG2ANSWER_OCR_READ_TIMEOUT_SECONDS": value}, clear=True):
                    with self.assertRaises(OCRError) as raised:
                        OCRSettings.from_env()
                    self.assertNotIn("secret-value", str(raised.exception))


class OCRResponseTests(unittest.TestCase):
    def test_normalizes_multiple_blocks_and_uses_polygon_not_rotated_box(self):
        response = sample_response()
        data = json.loads(response["Data"])
        word = data["prism_wordsInfo"][0]
        word.update({"x": -300, "y": -300, "angle": -89})
        word["pos"] = [{"x": 10, "y": 10}, {"x": 50, "y": 5}, {"x": 60, "y": 30}, {"x": 20, "y": 35}]
        data["prism_wordsInfo"].append(deepcopy(word))
        response["Data"] = json.dumps(data)
        result = normalize_response(response, width=100, height=120)
        self.assertEqual(result["blocks"][0]["bbox"], [10, 5, 60, 35])
        self.assertEqual(len(result["blocks"]), 2)
        self.assertEqual(result["blocks"][0]["confidence"], 95)
        self.assertEqual(result["raw_data"], data)

    def test_empty_and_missing_confidence_are_not_invented(self):
        result = normalize_response(sample_response(text=""), width=100, height=120)
        self.assertEqual(result["blocks"], [])
        response = sample_response()
        data = json.loads(response["Data"])
        del data["prism_wordsInfo"][0]["prob"]
        response["Data"] = json.dumps(data)
        result = normalize_response(response, width=100, height=120)
        self.assertIsNone(result["blocks"][0]["confidence"])

    def test_rejects_bad_outer_responses_and_json(self):
        for response in (None, [], {}, {"Code": "noPermission"}, {"Data": {}}, {"Data": "{"}, {"Data": "[]"}, {"Data": "NaN"}):
            with self.subTest(response=response), self.assertRaises(OCRError):
                normalize_response(response, width=100, height=120)

    def test_rejects_bad_dimensions_content_and_geometry(self):
        original = json.loads(sample_response()["Data"])
        bad_data = []
        for key, value in (("width", 99), ("height", True), ("orgWidth", 99), ("angle", 90), ("content", None), ("content", ""), ("prism_wordsInfo", []), ("prism_wordsInfo", {})):
            bad_data.append({**deepcopy(original), key: value})
        for key, value in (("word", None), ("pos", []), ("prob", -1), ("prob", 101), ("prob", True), ("prob", "99"), ("prob", float("inf")), ("prob", 10 ** 400)):
            data = deepcopy(original)
            data["prism_wordsInfo"][0][key] = value
            bad_data.append(data)
        for x in (-1, 101, True, float("nan"), 10 ** 400):
            data = deepcopy(original)
            data["prism_wordsInfo"][0]["pos"][0]["x"] = x
            bad_data.append(data)
        data = deepcopy(original)
        data["prism_wordsInfo"][0]["pos"] = [{"x": 1, "y": 1}] * 4
        bad_data.append(data)
        for index, data in enumerate(bad_data):
            with self.subTest(index=index), self.assertRaises(OCRError):
                normalize_response({"Data": json.dumps(data)}, width=100, height=120)


class OCRCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.image = self.root / "page.png"
        Image.new("RGB", (100, 120), "white").save(self.image)
        self.result = self.root / "ocr" / "page.json"
        self.client = RecordingOCRClient()

    def run_page(self, **kwargs):
        return recognize_page(self.image, self.result, client=self.client, **kwargs)

    def test_cache_reuses_exact_image_and_recomputes_review_threshold(self):
        first = self.run_page()
        second = self.run_page(min_confidence=99)
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertFalse(first.needs_review)
        self.assertTrue(second.needs_review)
        self.assertEqual(first.status, "recognized")
        self.assertEqual(len(self.client.calls), 1)
        saved = json.loads(self.result.read_text(encoding="utf-8"))
        self.assertEqual(saved["image_sha256"], hashlib.sha256(self.client.calls[0]).hexdigest())
        self.assertNotIn("min_confidence", saved)
        self.assertTrue(all(stream.closed for stream in self.client.streams))

    def test_input_content_and_request_identity_invalidate_cache(self):
        self.run_page()
        Image.new("RGB", (100, 120), "black").save(self.image)
        self.assertFalse(self.run_page().cache_hit)
        for key, value in (("action", "Other"), ("schema_version", 2), ("endpoint", "unknown"), ("request_options", {"need_rotate": True}), ("provider", "other"), ("api_version", "old")):
            cached = json.loads(self.result.read_text(encoding="utf-8"))
            cached[key] = value
            self.result.write_text(json.dumps(cached), encoding="utf-8")
            self.assertFalse(self.run_page().cache_hit)
        Image.new("RGB", (101, 120), "black").save(self.image)
        self.assertFalse(self.run_page().cache_hit)

    def test_corrupt_or_tampered_cache_is_not_used(self):
        self.result.parent.mkdir()
        for content in ("{", "null", "[]", '\ufffd'):
            self.result.write_text(content, encoding="utf-8")
            self.assertFalse(self.run_page().cache_hit)
        for mutate in (lambda cached: cached["blocks"][0].update(text="changed"), lambda cached: cached.update(unexpected="field"), lambda cached: cached.update(request_options={"need_rotate": 0})):
            cached = json.loads(self.result.read_text(encoding="utf-8"))
            mutate(cached)
            self.result.write_text(json.dumps(cached), encoding="utf-8")
            self.assertFalse(self.run_page().cache_hit)

    def test_refresh_failure_keeps_valid_cache_and_closes_stream(self):
        self.run_page()
        original = self.result.read_bytes()
        self.client.error = OCRError("模拟超时")
        with self.assertRaises(OCRError):
            self.run_page(refresh=True)
        self.assertEqual(self.result.read_bytes(), original)
        self.assertTrue(self.client.streams[-1].closed)
        self.assertTrue(self.run_page().cache_hit)
        self.assertEqual(len(self.client.calls), 2)

    def test_stale_cache_is_not_returned_when_new_recognition_fails(self):
        self.run_page()
        original = self.result.read_bytes()
        Image.new("RGB", (100, 120), "black").save(self.image)
        self.client.error = OCRError("模拟失败")
        with self.assertRaises(OCRError):
            self.run_page()
        self.assertEqual(self.result.read_bytes(), original)

    def test_successful_refresh_and_empty_result(self):
        self.run_page()
        self.client.response = sample_response(text="")
        result = self.run_page(refresh=True)
        self.assertEqual(result.status, "empty")
        self.assertTrue(result.needs_review)
        self.assertEqual(result.block_count, 0)
        self.assertTrue(self.run_page().cache_hit)
        self.assertEqual(len(self.client.calls), 2)

    def test_bad_response_does_not_write_cache(self):
        self.client.response = {"Data": "invalid"}
        with self.assertRaises(OCRError):
            self.run_page()
        self.assertFalse(self.result.exists())

    def test_atomic_replace_failure_cleans_temporary_file(self):
        self.run_page()
        original = self.result.read_bytes()
        with patch.object(ocr.os, "replace", side_effect=OSError("simulated")), self.assertRaises(OCRError):
            self.run_page(refresh=True)
        self.assertEqual(self.result.read_bytes(), original)
        self.assertEqual(list(self.result.parent.glob("*.tmp")), [])

    def test_cleanup_failure_is_reported_without_raw_filesystem_diagnostics(self):
        self.run_page()
        original = self.result.read_bytes()
        with patch.object(ocr.os, "replace", side_effect=OSError("simulated")), patch.object(Path, "unlink", side_effect=OSError("dummy-secret")):
            with self.assertRaises(OCRError) as raised:
                self.run_page(refresh=True)
        self.assertNotIn("dummy-secret", "".join(traceback.format_exception(raised.exception)))
        self.assertEqual(self.result.read_bytes(), original)

    def test_submitted_bytes_match_hash_even_if_source_changes_during_call(self):
        original = self.image.read_bytes()
        real_recognize = self.client.recognize

        def change_source(stream):
            Image.new("RGB", (100, 120), "black").save(self.image)
            return real_recognize(stream)

        with patch.object(self.client, "recognize", side_effect=change_source):
            self.run_page()
        cached = json.loads(self.result.read_text(encoding="utf-8"))
        self.assertEqual(self.client.calls[0], original)
        self.assertEqual(cached["image_sha256"], hashlib.sha256(original).hexdigest())
        self.assertFalse(self.run_page().cache_hit)

    def test_invalid_images_fail_before_client_call(self):
        for data in (b"", b"not an image", self.image.read_bytes()[:50]):
            self.image.write_bytes(data)
            with self.assertRaises(OCRError):
                self.run_page()
        for size in ((15, 20), (20, 15), (8192, 20), (20, 8192), (1000, 20)):
            Image.new("RGB", size).save(self.image)
            with self.subTest(size=size), self.assertRaises(OCRError):
                self.run_page()
        Image.new("RGB", (100, 120)).save(self.image, format="GIF")
        with self.assertRaises(OCRError):
            self.run_page()
        self.assertEqual(self.client.calls, [])

    def test_image_size_limits_and_disguised_extensions(self):
        original = self.image.read_bytes()
        with patch.object(ocr, "MAX_IMAGE_BYTES", len(original)):
            self.assertEqual(self.run_page().status, "recognized")
        with patch.object(ocr, "MAX_IMAGE_BYTES", len(original) - 1), self.assertRaises(OCRError):
            self.run_page(refresh=True)
        disguised = self.root / "not-a-png.txt"
        disguised.write_bytes(original)
        self.assertTrue(recognize_page(disguised, self.result, client=self.client).cache_hit)
        self.assertEqual(len(self.client.calls), 1)

    def test_static_jpeg_and_multiframe_png(self):
        Image.new("RGB", (16, 16), "white").save(self.image, format="JPEG")
        self.assertFalse(self.run_page().cache_hit)
        Image.new("RGB", (20, 20), "white").save(self.image, format="PNG", save_all=True, append_images=[Image.new("RGB", (20, 20), "black")])
        with self.assertRaises(OCRError):
            self.run_page()
        self.assertEqual(len(self.client.calls), 1)


class AliyunAdapterTests(unittest.TestCase):
    def test_real_sdk_transport_submits_once_and_normalizes_response(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "page.png"
            Image.new("RGB", (100, 120), "white").save(image_path)
            response_body = io.BytesIO(json.dumps(sample_response()).encode("utf-8"))
            with patch("alibabacloud_tea_openapi.client.DaraCore.do_action") as transport:
                transport.return_value.status_code = 200
                transport.return_value.headers = {}
                transport.return_value.body = response_body
                client = AliyunOCRClient(settings())
                result = recognize_page(image_path, root / "page.json", client=client)
                self.assertEqual(result.status, "recognized")
                self.assertEqual(transport.call_count, 1)
                request, runtime = transport.call_args.args
                self.assertEqual(request.protocol, "HTTPS")
                self.assertEqual(request.headers["host"], ocr.OCR_ENDPOINT)
                self.assertEqual(request.body, image_path.read_bytes())
                self.assertEqual(runtime["readTimeout"], 30000)

    def test_wrapped_sdk_timeout_is_classified_without_retries(self):
        with patch("alibabacloud_tea_openapi.client.DaraCore.do_action", side_effect=TimeoutError("dummy-secret")) as transport:
            client = AliyunOCRClient(settings())
            with self.assertRaises(OCRError) as raised:
                client.recognize(io.BytesIO(b"image"))
            self.assertEqual(transport.call_count, 1)
            self.assertIn("请求超时", str(raised.exception))
            self.assertNotIn("dummy-secret", str(raised.exception))

    def test_initialization_failure_hides_sdk_configuration(self):
        config = settings()
        with patch("alibabacloud_ocr_api20210707.client.Client", side_effect=RuntimeError(config.access_key_secret)):
            with self.assertRaises(OCRError) as raised:
                AliyunOCRClient(config)
        self.assertNotIn(config.access_key_secret, "".join(traceback.format_exception(raised.exception)))

    def test_missing_sdk_has_a_safe_installation_message(self):
        original_import = builtins.__import__

        def missing_sdk(name, *args, **kwargs):
            if name.startswith("alibabacloud_ocr_api20210707"):
                raise ImportError("simulated")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=missing_sdk), self.assertRaises(OCRError):
            AliyunOCRClient(settings())

    def test_real_sdk_models_receive_binary_stream_and_bounded_runtime(self):
        from alibabacloud_ocr_api20210707.models import RecognizeAdvancedResponse, RecognizeAdvancedResponseBody

        with patch("alibabacloud_ocr_api20210707.client.Client") as sdk:
            sdk.return_value.recognize_advanced_with_options.return_value = RecognizeAdvancedResponse(
                body=RecognizeAdvancedResponseBody(data=sample_response()["Data"], request_id="request-123"),
            )
            client = AliyunOCRClient(settings())
            with io.BytesIO(b"image") as stream:
                response = client.recognize(stream)
                request, runtime = sdk.return_value.recognize_advanced_with_options.call_args.args
                self.assertIs(request.body, stream)
                self.assertFalse(request.need_rotate)
                self.assertIsNone(request.url)
                self.assertFalse(runtime.autoretry)
                self.assertEqual(runtime.connect_timeout, 5000)
                self.assertEqual(runtime.read_timeout, 30000)
            self.assertEqual(response["RequestId"], "request-123")
            self.assertEqual(sdk.call_args.args[0].endpoint, ocr.OCR_ENDPOINT)
            self.assertEqual(sdk.call_args.args[0].protocol, "HTTPS")

    def test_errors_are_safe_and_do_not_retry(self):
        from alibabacloud_ocr_api20210707.models import RecognizeAdvancedResponse, RecognizeAdvancedResponseBody

        codes = ("InvalidAccessKeyId.NotFound", "QuotaExceeded", "Throttling", "ServiceNotOpened", "ReadTimeout", "unknown")
        config = settings()
        for code in codes:
            with self.subTest(code=code), patch("alibabacloud_ocr_api20210707.client.Client") as sdk:
                error = RuntimeError(f"{config.access_key_id} {config.access_key_secret}")
                error.code = code
                error.data = {"RequestId": "safe-request-id"}
                sdk.return_value.recognize_advanced_with_options.side_effect = error
                client = AliyunOCRClient(config)
                with self.assertRaises(OCRError) as raised:
                    client.recognize(io.BytesIO(b"image"))
                formatted = "".join(traceback.format_exception(raised.exception))
                self.assertNotIn(config.access_key_id, formatted)
                self.assertNotIn(config.access_key_secret, formatted)
                self.assertIn("safe-request-id", str(raised.exception))
                self.assertEqual(sdk.return_value.recognize_advanced_with_options.call_count, 1)
        with patch("alibabacloud_ocr_api20210707.client.Client") as sdk:
            sdk.return_value.recognize_advanced_with_options.return_value = RecognizeAdvancedResponse(
                body=RecognizeAdvancedResponseBody(code="noPermission", message=config.access_key_secret, request_id=config.access_key_id),
            )
            with self.assertRaises(OCRError) as raised:
                AliyunOCRClient(config).recognize(io.BytesIO(b"image"))
            self.assertNotIn(config.access_key_id, str(raised.exception))
            self.assertNotIn(config.access_key_secret, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
