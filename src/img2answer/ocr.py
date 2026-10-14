from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, BinaryIO

from PIL import Image


OCR_ENDPOINT = "ocr-api.cn-hangzhou.aliyuncs.com"
MAX_IMAGE_BYTES = 10_000_000
REQUEST_OPTIONS = {"need_rotate": False}


class OCRError(RuntimeError):
    """An OCR failure with a message safe to display without SDK diagnostics."""


@dataclass(frozen=True)
class OCRSettings:
    access_key_id: str = field(repr=False)
    access_key_secret: str = field(repr=False)
    connect_timeout_seconds: int = 5
    read_timeout_seconds: int = 30
    min_confidence: float = 80

    def __post_init__(self) -> None:
        for name, value in (
            ("ALIBABA_CLOUD_ACCESS_KEY_ID", self.access_key_id),
            ("ALIBABA_CLOUD_ACCESS_KEY_SECRET", self.access_key_secret),
        ):
            if not isinstance(value, str) or not value.strip():
                raise OCRError(f"配置项 {name} 必须非空")
        for name, value in (
            ("IMG2ANSWER_OCR_CONNECT_TIMEOUT_SECONDS", self.connect_timeout_seconds),
            ("IMG2ANSWER_OCR_READ_TIMEOUT_SECONDS", self.read_timeout_seconds),
        ):
            if type(value) is not int or value <= 0:
                raise OCRError(f"配置项 {name} 必须是正整数")
        if not _is_number(self.min_confidence) or not 0 <= self.min_confidence <= 100:
            raise OCRError("配置项 IMG2ANSWER_OCR_MIN_CONFIDENCE 必须是 0 至 100 的有限数值")

    @classmethod
    def from_env(cls) -> OCRSettings:
        try:
            from dotenv import dotenv_values
        except ImportError:
            raise OCRError('缺少 OCR 依赖，请在项目 conda 环境安装 ".[ocr]"') from None
        try:
            file_values = dotenv_values(Path.cwd() / ".env", encoding="utf-8-sig", interpolate=False)
        except (OSError, UnicodeError):
            raise OCRError("无法读取项目根目录的 .env 配置") from None
        values = {key: value for key, value in file_values.items() if value is not None}
        values.update(os.environ)

        def parse(name: str, default: str, converter: Any) -> Any:
            try:
                return converter(values.get(name, default))
            except (TypeError, ValueError, OverflowError):
                raise OCRError(f"配置项 {name} 的数值格式无效") from None

        return cls(
            access_key_id=values.get("ALIBABA_CLOUD_ACCESS_KEY_ID", ""),
            access_key_secret=values.get("ALIBABA_CLOUD_ACCESS_KEY_SECRET", ""),
            connect_timeout_seconds=parse("IMG2ANSWER_OCR_CONNECT_TIMEOUT_SECONDS", "5", int),
            read_timeout_seconds=parse("IMG2ANSWER_OCR_READ_TIMEOUT_SECONDS", "30", int),
            min_confidence=parse("IMG2ANSWER_OCR_MIN_CONFIDENCE", "80", float),
        )


class AliyunOCRClient:
    def __init__(self, settings: OCRSettings):
        try:
            from alibabacloud_ocr_api20210707.client import Client
            from alibabacloud_ocr_api20210707.models import RecognizeAdvancedRequest
            from alibabacloud_tea_openapi.models import Config
            from alibabacloud_tea_util.models import RuntimeOptions
        except ImportError:
            raise OCRError('缺少阿里云 OCR SDK，请在项目 conda 环境安装 ".[ocr]"') from None
        self.settings = settings
        self._request_type = RecognizeAdvancedRequest
        config = Config(
            access_key_id=settings.access_key_id,
            access_key_secret=settings.access_key_secret,
            endpoint=OCR_ENDPOINT,
            protocol="HTTPS",
        )
        try:
            self._client = Client(config)
            self._runtime = RuntimeOptions(
                connect_timeout=settings.connect_timeout_seconds * 1000,
                read_timeout=settings.read_timeout_seconds * 1000,
                autoretry=False,
            )
        except Exception:
            raise OCRError("无法初始化阿里云 OCR SDK，请检查依赖与本地配置") from None

    def recognize(self, image: BinaryIO) -> dict[str, Any]:
        request = self._request_type(body=image, **REQUEST_OPTIONS)
        try:
            response = self._client.recognize_advanced_with_options(request, self._runtime)
        except Exception as exc:
            data = getattr(exc, "data", None)
            request_id = data.get("RequestId") if isinstance(data, dict) else None
            raise _cloud_error(getattr(exc, "code", ""), request_id, exc, self.settings) from None
        try:
            body = response.body.to_map()
        except (AttributeError, TypeError):
            raise OCRError("阿里云 OCR 响应结构无效") from None
        if body.get("Code"):
            raise _cloud_error(body["Code"], body.get("RequestId"), None, self.settings)
        return body


def _safe_request_id(value: Any) -> str | None:
    if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9-]{1,128}", value):
        return value
    return None


def _cloud_error(code: Any, request_id: Any, exc: Exception | None, settings: OCRSettings) -> OCRError:
    normalized = code.lower() if isinstance(code, str) else ""
    inner_exception = getattr(exc, "inner_exception", None)
    if isinstance(inner_exception, Exception):
        exc = inner_exception
    if isinstance(exc, TimeoutError) or "timeout" in normalized or "timeout" in type(exc).__name__.lower():
        message = "阿里云 OCR 请求超时；云端可能已处理，请核对调用记录后续跑"
    elif any(part in normalized for part in ("accesskey", "signature", "permission", "forbidden")):
        message = "阿里云 OCR 权限校验失败，请检查 RAM 授权和本地凭证"
    elif any(part in normalized for part in ("arrears", "quota", "balance", "unpaid")):
        message = "阿里云 OCR 服务额度或余额不足，请检查服务控制台"
    elif any(part in normalized for part in ("notopened", "notenabled", "servicenot")):
        message = "阿里云 OCR 服务未开通，请检查服务控制台"
    elif any(part in normalized for part in ("throttl", "ratelimit")):
        message = "阿里云 OCR 请求被限流，请稍后续跑"
    elif isinstance(exc, (OSError, ConnectionError)):
        message = "阿里云 OCR 网络请求失败，请检查网络后续跑"
    else:
        message = "阿里云 OCR 调用失败，请检查服务状态和调用记录"
    safe_id = _safe_request_id(request_id)
    if safe_id and not any(secret in safe_id for secret in (settings.access_key_id, settings.access_key_secret)):
        message += f"（请求 ID：{safe_id}）"
    return OCRError(message)


def _is_number(value: Any) -> bool:
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _json_constant(value: str) -> None:
    raise ValueError("non-finite JSON number")


def normalize_response(response: dict[str, Any], *, width: int, height: int) -> dict[str, Any]:
    if not isinstance(response, dict) or response.get("Code"):
        raise OCRError("阿里云 OCR 返回错误状态")
    data_string = response.get("Data")
    if not isinstance(data_string, str):
        raise OCRError("阿里云 OCR 的 Data 必须是 JSON 字符串")
    try:
        data = json.loads(data_string, parse_constant=_json_constant)
    except (ValueError, RecursionError):
        raise OCRError("阿里云 OCR 的 Data 不是有效 JSON") from None
    if not isinstance(data, dict):
        raise OCRError("阿里云 OCR 的 Data 结构无效")
    for key, expected in (("width", width), ("height", height), ("orgWidth", width), ("orgHeight", height)):
        if key in {"orgWidth", "orgHeight"} and key not in data:
            continue
        if type(data.get(key)) is not int or data[key] != expected:
            raise OCRError("阿里云 OCR 返回尺寸与原图不一致，不能直接使用文字坐标")
    if "angle" in data and (not _is_number(data["angle"]) or data["angle"] != 0):
        raise OCRError("阿里云 OCR 返回了未经支持的页面旋转")
    content = data.get("content")
    words = data.get("prism_wordsInfo")
    if not isinstance(content, str) or not isinstance(words, list) or bool(content.strip()) != bool(words):
        raise OCRError("阿里云 OCR 文字与文字块结构无效")

    blocks = []
    for word in words:
        if not isinstance(word, dict) or not isinstance(word.get("word"), str) or not word["word"].strip():
            raise OCRError("阿里云 OCR 文字块内容无效")
        points = word.get("pos")
        if not isinstance(points, list) or len(points) != 4:
            raise OCRError("阿里云 OCR 文字块必须包含四点坐标")
        polygon = []
        for point in points:
            if not isinstance(point, dict):
                raise OCRError("阿里云 OCR 文字坐标无效")
            x, y = point.get("x"), point.get("y")
            if not _is_number(x) or not _is_number(y) or not 0 <= x <= width or not 0 <= y <= height:
                raise OCRError("阿里云 OCR 文字坐标无效或超出原图边界")
            polygon.append([x, y])
        xs, ys = [point[0] for point in polygon], [point[1] for point in polygon]
        bbox = [min(xs), min(ys), max(xs), max(ys)]
        if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
            raise OCRError("阿里云 OCR 文字边界框面积必须为正")
        confidence = word.get("prob")
        if confidence is not None and (not _is_number(confidence) or not 0 <= confidence <= 100):
            raise OCRError("阿里云 OCR 置信度无效")
        blocks.append({"text": word["word"], "polygon": polygon, "bbox": bbox, "confidence": confidence})
    return {
        "request_id": _safe_request_id(response.get("RequestId")),
        "content": content,
        "blocks": blocks,
        "raw_data": data,
    }


@dataclass(frozen=True)
class OCRPageResult:
    result_path: Path
    status: str
    cache_hit: bool
    block_count: int
    needs_review: bool


def _read_image(path: Path) -> tuple[bytes, int, int]:
    try:
        with path.open("rb") as stream:
            data = stream.read(MAX_IMAGE_BYTES + 1)
    except OSError:
        raise OCRError("无法读取 OCR 页面图片") from None
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise OCRError("OCR 图片不能为空且不能超过 10,000,000 字节")
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            if image.format not in {"PNG", "JPEG"} or getattr(image, "n_frames", 1) != 1:
                raise OCRError("OCR 仅接受静态 PNG 或 JPEG 图片")
            if not 15 < width < 8192 or not 15 < height < 8192 or max(width, height) / min(width, height) >= 50:
                raise OCRError("OCR 图片尺寸超限，请调整渲染 DPI 后重新运行")
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image.load()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        raise OCRError("OCR 页面图片无效或已损坏") from None
    return data, width, height


def _cache_identity(image: bytes, width: int, height: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "provider": "aliyun",
        "action": "RecognizeAdvanced",
        "api_version": "2021-07-07",
        "endpoint": OCR_ENDPOINT,
        "request_options": dict(REQUEST_OPTIONS),
        "image_sha256": hashlib.sha256(image).hexdigest(),
        "image_width": width,
        "image_height": height,
    }


def _load_cache(path: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    try:
        cached = json.loads(path.read_text(encoding="utf-8"), parse_constant=_json_constant)
        if not isinstance(cached, dict):
            return None
        normalized = normalize_response(
            {"RequestId": cached.get("request_id"), "Data": json.dumps(cached["raw_data"], allow_nan=False)},
            width=identity["image_width"],
            height=identity["image_height"],
        )
        expected = {**identity, **normalized}
        # Check both raw and normalized fields; Python equality alone treats false as zero.
        if json.dumps(cached, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
            return None
        return expected
    except (OSError, ValueError, TypeError, KeyError, OCRError, RecursionError):
        return None


def _write_cache(path: Path, result: dict[str, Any]) -> None:
    temporary_path = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary_path, path)
        temporary_path = None
    except (OSError, ValueError, TypeError):
        raise OCRError("无法保存 OCR 结果；未替换历史缓存") from None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                raise OCRError("无法清理 OCR 临时文件，请检查输出目录权限") from None


def recognize_page(
    image_path: str | Path,
    result_path: str | Path,
    *,
    client: AliyunOCRClient,
    min_confidence: float = 80,
    refresh: bool = False,
) -> OCRPageResult:
    if not _is_number(min_confidence) or not 0 <= min_confidence <= 100:
        raise OCRError("OCR 核对阈值必须是 0 至 100 的有限数值")
    image, width, height = _read_image(Path(image_path))
    identity = _cache_identity(image, width, height)
    output_path = Path(result_path)
    cached = None if refresh else _load_cache(output_path, identity)
    cache_hit = cached is not None
    if cached is None:
        # Submit the exact bytes that were validated and hashed, even if the source changes.
        with io.BytesIO(image) as stream:
            response = client.recognize(stream)
        cached = {**identity, **normalize_response(response, width=width, height=height)}
        _write_cache(output_path, cached)
    blocks = cached["blocks"]
    return OCRPageResult(
        result_path=output_path,
        status="recognized" if blocks else "empty",
        cache_hit=cache_hit,
        block_count=len(blocks),
        needs_review=not blocks or any(
            block["confidence"] is None or block["confidence"] < min_confidence
            for block in blocks
        ),
    )
