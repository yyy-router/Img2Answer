from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a section configuration is invalid."""


@dataclass(frozen=True)
class SectionConfig:
    name: str
    page_from: int
    page_to: int

    def zero_based_range(self) -> range:
        return range(self.page_from - 1, self.page_to)


@dataclass(frozen=True)
class DocumentConfig:
    document_id: str
    path: Path
    sections: tuple[SectionConfig, ...]


@dataclass(frozen=True)
class ProjectConfig:
    documents: tuple[DocumentConfig, ...]


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    raw = _load_raw_config(config_path)
    documents = raw.get("documents")
    if not isinstance(documents, dict) or not documents:
        raise ConfigError("config must contain a non-empty 'documents' mapping")

    parsed_documents: list[DocumentConfig] = []
    for document_id, document_raw in documents.items():
        if not isinstance(document_raw, dict):
            raise ConfigError(f"document '{document_id}' must be a mapping")

        source_path = document_raw.get("path")
        if not isinstance(source_path, str) or not source_path.strip():
            raise ConfigError(f"document '{document_id}' must define a non-empty path")

        sections_raw = document_raw.get("sections")
        if not isinstance(sections_raw, dict) or not sections_raw:
            raise ConfigError(f"document '{document_id}' must define sections")

        sections = tuple(_parse_section(name, value, document_id) for name, value in sections_raw.items())
        parsed_documents.append(
            DocumentConfig(
                document_id=str(document_id),
                path=_resolve_config_path(config_path, source_path),
                sections=sections,
            )
        )

    return ProjectConfig(documents=tuple(parsed_documents))


def _load_raw_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file does not exist: {path}")

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yml", ".yaml"}:
        data = yaml.safe_load(text)
    else:
        raise ConfigError("config file must be .json, .yml, or .yaml")

    if not isinstance(data, dict):
        raise ConfigError("config root must be a mapping")
    return data


def _parse_section(name: str, value: Any, document_id: str) -> SectionConfig:
    if not isinstance(value, dict):
        raise ConfigError(f"section '{name}' in document '{document_id}' must be a mapping")
    page_from = _parse_positive_int(value.get("page_from"), "page_from", name, document_id)
    page_to = _parse_positive_int(value.get("page_to"), "page_to", name, document_id)
    if page_from > page_to:
        raise ConfigError(f"section '{name}' in document '{document_id}' has page_from > page_to")
    return SectionConfig(name=str(name), page_from=page_from, page_to=page_to)


def _parse_positive_int(value: Any, field: str, section: str, document_id: str) -> int:
    if not isinstance(value, int) or value < 1:
        raise ConfigError(
            f"section '{section}' in document '{document_id}' must define positive integer {field}"
        )
    return value


def _resolve_config_path(config_path: Path, source_path: str) -> Path:
    path = Path(source_path)
    if path.is_absolute():
        return path
    return path.resolve()
