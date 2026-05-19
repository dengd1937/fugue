"""src/ragline/config/__init__.py — Ragline 配置定义、Pydantic 校验与 YAML 加载。"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ragline.api.types import RaglineConfigError

logger = logging.getLogger(__name__)

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")

__all__ = [
    "GraphConfig",
    "IngestConfig",
    "ProviderConfig",
    "RaglineConfig",
    "dump_yaml",
    "expand_env_vars",
    "load_yaml",
]


# ---------------------------------------------------------------------------
# Dataclasses（spec 4.3）
# ---------------------------------------------------------------------------


@dataclass
class GraphConfig:
    transforms: list[str | list[str]] = field(default_factory=lambda: ["rewrite"])
    n_rewrites: int = 3
    max_queries: int = 20
    retrievers: list[str] = field(default_factory=lambda: ["vector", "bm25"])
    route_strategy: Literal["all", "intent"] = "all"
    retriever_weights: dict[str, float] = field(default_factory=dict)
    grade_threshold: float = 0.6
    grade_strategy: Literal["score", "llm", "hybrid"] = "score"
    score_normalizers: dict[str, float] = field(default_factory=lambda: {"bm25": 20.0})
    fallback_chain: list[str] = field(default_factory=list)
    max_retries: int = 1
    processors: list[str] = field(default_factory=lambda: ["rrf", "rerank"])
    top_k: int = 3
    gen_mode: Literal["basic", "citation"] = "basic"
    temperature: float = 0.7


@dataclass
class IngestConfig:
    parser: str = "auto"
    chunker: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64
    collection_name: str = "default"
    persist_dir: str = "./.ragline"


@dataclass
class ProviderConfig:
    # LLM
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_timeout: float = 60.0
    llm_max_retries: int = 2
    llm_concurrency: int = 10
    # Embedding
    embedding_base_url: str | None = None  # None 复用 llm_base_url
    embedding_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_timeout: float = 30.0
    embedding_max_retries: int = 2
    embedding_concurrency: int = 5
    embedding_batch_size: int = 64
    # Reranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_timeout: float = 30.0
    reranker_device: Literal["cpu", "cuda", "auto"] = "auto"


@dataclass
class RaglineConfig:
    graph: GraphConfig = field(default_factory=GraphConfig)
    ingest: IngestConfig = field(default_factory=IngestConfig)
    providers: ProviderConfig = field(default_factory=ProviderConfig)


# ---------------------------------------------------------------------------
# Pydantic v2 校验 Schema
# ---------------------------------------------------------------------------


class GraphConfigSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transforms: list[str | list[str]] = Field(
        default_factory=lambda: ["rewrite"]  # type: ignore[arg-type]
    )
    n_rewrites: int = Field(default=3, ge=1, le=20)
    max_queries: int = Field(default=20, ge=1, le=100)
    retrievers: list[str] = Field(default_factory=lambda: ["vector", "bm25"])
    route_strategy: Literal["all", "intent"] = "all"
    retriever_weights: dict[str, float] = Field(default_factory=dict)
    grade_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    grade_strategy: Literal["score", "llm", "hybrid"] = "score"
    score_normalizers: dict[str, float] = Field(default_factory=lambda: {"bm25": 20.0})
    fallback_chain: list[str] = Field(default_factory=list)
    max_retries: int = Field(default=1, ge=0, le=10)
    processors: list[str] = Field(default_factory=lambda: ["rrf", "rerank"])
    top_k: int = Field(default=3, ge=1, le=100)
    gen_mode: Literal["basic", "citation"] = "basic"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    def to_dataclass(self) -> GraphConfig:
        return GraphConfig(**self.model_dump())


class IngestConfigSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parser: str = "auto"
    chunker: str = "recursive"
    chunk_size: int = Field(default=512, ge=1, le=65536)
    chunk_overlap: int = Field(default=64, ge=0, le=8192)
    collection_name: str = "default"
    persist_dir: str = "./.ragline"

    def to_dataclass(self) -> IngestConfig:
        return IngestConfig(**self.model_dump())


class ProviderConfigSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # LLM
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_timeout: float = Field(default=60.0, ge=0.1, le=600.0)
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    llm_concurrency: int = Field(default=10, ge=1, le=200)
    # Embedding
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_model: str = "text-embedding-3-small"
    embedding_timeout: float = Field(default=30.0, ge=0.1, le=600.0)
    embedding_max_retries: int = Field(default=2, ge=0, le=10)
    embedding_concurrency: int = Field(default=5, ge=1, le=200)
    embedding_batch_size: int = Field(default=64, ge=1, le=4096)
    # Reranker
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_timeout: float = Field(default=30.0, ge=0.1, le=600.0)
    reranker_device: Literal["cpu", "cuda", "auto"] = "auto"

    def to_dataclass(self) -> ProviderConfig:
        return ProviderConfig(**self.model_dump())


class RaglineConfigSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    graph: GraphConfigSchema = Field(default_factory=GraphConfigSchema)
    ingest: IngestConfigSchema = Field(default_factory=IngestConfigSchema)
    providers: ProviderConfigSchema = Field(default_factory=ProviderConfigSchema)

    def to_dataclass(self) -> RaglineConfig:
        return RaglineConfig(
            graph=self.graph.to_dataclass(),
            ingest=self.ingest.to_dataclass(),
            providers=self.providers.to_dataclass(),
        )


# ---------------------------------------------------------------------------
# YAML loader + env 展开
# ---------------------------------------------------------------------------


def expand_env_vars(text: str) -> str:
    """${VAR_NAME} → os.environ['VAR_NAME']；未定义时保留原字符串并 warning。"""

    def replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        if var_name in os.environ:
            return os.environ[var_name]
        logger.warning("Environment variable '%s' not defined; keeping placeholder", var_name)
        return match.group(0)

    return _ENV_VAR_PATTERN.sub(replace, text)


def load_yaml(path: str | Path) -> RaglineConfig:
    """读 YAML → ${env_var} 展开 → Pydantic 校验 → dataclass。

    失败抛 RaglineConfigError，消息含字段路径与原因。
    """
    p = Path(path)
    try:
        raw_text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise RaglineConfigError(f"Failed to read config file '{p}': {e}") from e

    expanded = expand_env_vars(raw_text)

    try:
        raw_data: Any = yaml.safe_load(expanded)
    except yaml.YAMLError as e:
        raise RaglineConfigError(f"Invalid YAML syntax in '{p}': {e}") from e

    if raw_data is None:
        raw_data = {}

    if not isinstance(raw_data, dict):
        raise RaglineConfigError(f"Config file '{p}' must contain a top-level mapping, got {type(raw_data).__name__}")

    try:
        schema = RaglineConfigSchema.model_validate(raw_data)
    except ValidationError as e:
        errors = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"])
            errors.append(f"  - {loc}: {err['msg']}")
        raise RaglineConfigError(f"Config validation failed for '{p}':\n" + "\n".join(errors)) from e

    return schema.to_dataclass()


def dump_yaml(config: RaglineConfig, path: str | Path) -> None:
    """将 RaglineConfig 序列化为 YAML（用于配置导出/快照）。"""
    data = asdict(config)
    p = Path(path)
    p.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
