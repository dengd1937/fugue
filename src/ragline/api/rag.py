"""src/ragline/api/rag.py — Ragline 主入口 class。"""

import logging
import os
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Any

from ragline.api.ingest import IngestPipeline
from ragline.api.types import (
    IngestResult,
    QueryResult,
    RaglineConfigError,
)
from ragline.config import GraphConfig, RaglineConfig, load_yaml
from ragline.engine.graph import build_rag_graph  # allowed via .importlinter ignore_imports
from ragline.handlers.chunkers import register_chunkers
from ragline.handlers.generators import register_generators
from ragline.handlers.graders import register_graders
from ragline.handlers.parsers import register_parsers
from ragline.handlers.processors import register_processors
from ragline.handlers.retrievers import register_retrievers
from ragline.handlers.transforms import register_transforms
from ragline.providers.bm25 import BM25Provider
from ragline.providers.embedding import EmbeddingClient
from ragline.providers.llm import LLMClient
from ragline.providers.reranker.base import Reranker
from ragline.providers.reranker.bge import BGEReranker
from ragline.providers.vector_store.base import VectorStore
from ragline.providers.vector_store.chroma import ChromaVectorStore
from ragline.registry import (
    chunker_registry,
    discover_plugins,
    generator_registry,
    grader_registry,
    parser_registry,
    processor_registry,
    retriever_registry,
    transform_registry,
)

logger = logging.getLogger(__name__)


_BUILTIN_TRANSFORMS = {"rewrite", "hyde", "step_back"}
_BUILTIN_RETRIEVERS = {"vector", "bm25"}
_BUILTIN_PROCESSORS = {"rrf", "rerank"}
_BUILTIN_GENERATORS = {"basic", "citation"}


class _LazyReranker:
    """Lazy reranker proxy: 第一次调用 rerank() 时才加载 BGE 模型。"""

    def __init__(self, model_name: str, device: str, timeout: float) -> None:
        self._model_name = model_name
        self._device = device
        self._timeout = timeout
        self._real: Reranker | None = None

    def _ensure_loaded(self) -> Reranker:
        if self._real is None:
            self._real = BGEReranker(
                model_name=self._model_name,
                device=self._device,  # type: ignore[arg-type]
                timeout=self._timeout,
            )
        return self._real

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        return self._ensure_loaded().rerank(query, documents, top_k=top_k)

    def close(self) -> None:
        if self._real is not None:
            self._real.close()
            self._real = None


def _load_dotenv(env_file: Path) -> None:
    """简易 dotenv 加载器（不依赖 python-dotenv）。"""
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


class RAG(AbstractContextManager["RAG"]):
    """Ragline 主入口。

    .. warning::
        **MVP 0.x 限制**：Registry 是全局单例，**同一进程内只支持一个 RAG 实例**。
        实例化第二个 RAG（不同 provider 配置）会静默覆盖前一个的 transforms/generators
        闭包，结果不可预测。多 RAG 场景请使用独立进程。

        实例化时会检测此覆盖并发 warning。

    .. warning::
        Registry 写入操作非线程安全；所有注册应在第一次 RAG() 实例化前完成。
    """

    def __init__(
        self,
        config: RaglineConfig | None = None,
        *,
        collection_name: str | None = None,
        env_file: str | Path | None = None,
    ) -> None:
        # 1. .env 加载
        env_path = Path(env_file) if env_file else Path(".env")
        _load_dotenv(env_path)

        # 2. config 默认 + collection 覆盖
        cfg = config if config is not None else RaglineConfig()
        if collection_name:
            # IngestConfig 是 dataclass(frozen=False)，直接赋值
            cfg.ingest.collection_name = collection_name
        self._config = cfg

        # 3. 检测 multi-RAG-instance Registry 覆盖
        self._warn_if_handlers_already_registered()

        # 4. 初始化 providers
        api_key = cfg.providers.llm_api_key or os.environ.get("OPENAI_API_KEY", "")
        self._llm_client = LLMClient(
            base_url=cfg.providers.llm_base_url,
            api_key=api_key,
            model=cfg.providers.llm_model,
            timeout=cfg.providers.llm_timeout,
            max_retries=cfg.providers.llm_max_retries,
            max_concurrent=cfg.providers.llm_concurrency,
        )
        embedding_api_key = cfg.providers.embedding_api_key or api_key or ""
        embedding_base_url = cfg.providers.embedding_base_url or cfg.providers.llm_base_url
        self._embedding_client = EmbeddingClient(
            base_url=embedding_base_url,
            api_key=embedding_api_key,
            model=cfg.providers.embedding_model,
            timeout=cfg.providers.embedding_timeout,
            max_retries=cfg.providers.embedding_max_retries,
            max_concurrent=cfg.providers.embedding_concurrency,
            batch_size=cfg.providers.embedding_batch_size,
        )
        self._vector_store: VectorStore = ChromaVectorStore(
            persist_dir=cfg.ingest.persist_dir,
            collection_name=cfg.ingest.collection_name,
        )
        self._reranker = _LazyReranker(
            model_name=cfg.providers.reranker_model,
            device=cfg.providers.reranker_device,
            timeout=cfg.providers.reranker_timeout,
        )
        self._bm25 = BM25Provider()

        # 5. 注册内置 handlers（含无依赖的 graders/parsers/chunkers）
        register_graders()
        register_parsers()
        register_chunkers()
        register_transforms(self._llm_client)
        register_retrievers(self._vector_store, self._embedding_client, self._bm25)
        register_processors(self._reranker)
        register_generators(self._llm_client)

        # 6. 扫描第三方插件
        discover_plugins()

        # 7. fail-fast 配置校验
        self._validate_config()

        # 8. BM25 启动重建（若启用）
        self._bootstrap_bm25()

        # 9. 构建图
        self._graph = build_rag_graph()

        # 10. 用于 ingest
        self._ingest_pipeline = IngestPipeline(
            config=self._config,
            embedding_client=self._embedding_client,
            vector_store=self._vector_store,
            bm25_provider=self._bm25,
        )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        collection_name: str | None = None,
        env_file: str | Path | None = None,
        **overrides: Any,
    ) -> "RAG":
        """从 YAML 文件加载 RaglineConfig 后实例化 RAG。

        overrides 用 dot path 覆盖任意层级（例如 graph__n_rewrites=5；
        因为 Python kwargs 不允许 dot，用双下划线代替）。
        """
        cfg = load_yaml(path)
        for key, value in overrides.items():
            section, _, field = key.partition("__")
            if not field:
                raise RaglineConfigError(f"overrides key '{key}' must use dot path like 'graph__n_rewrites'")
            target = getattr(cfg, section, None)
            if target is None:
                raise RaglineConfigError(f"unknown config section '{section}'")
            if not hasattr(target, field):
                raise RaglineConfigError(f"unknown field '{field}' in section '{section}'")
            setattr(target, field, value)
        return cls(cfg, collection_name=collection_name, env_file=env_file)

    def ingest(
        self,
        sources: str | Path | Any,
        *,
        show_progress: bool = True,
    ) -> IngestResult:
        """委托给 IngestPipeline.run。"""
        result = self._ingest_pipeline.run(sources, show_progress=show_progress)
        # ingest 后 BM25 已 update（pipeline 内已处理）
        return result

    def query(
        self,
        question: str,
        *,
        graph_override: GraphConfig | None = None,
    ) -> QueryResult:
        """对 graph 发起一次查询。"""
        effective_cfg = graph_override if graph_override is not None else self._config.graph
        initial_state: dict[str, Any] = {
            "original_query": question,
            "rewritten_queries": [],
            "documents": [],
            "grade_score": 0.0,
            "grade_decision": "insufficient",
            "source": "kb",
            "retry_count": 0,
            "retrieval_history": [],
            "ranked_documents": [],
            "answer": "",
        }
        configurable = {
            "transforms": effective_cfg.transforms,
            "n_rewrites": effective_cfg.n_rewrites,
            "max_queries": effective_cfg.max_queries,
            "retrievers": effective_cfg.retrievers,
            "route_strategy": effective_cfg.route_strategy,
            "retriever_weights": effective_cfg.retriever_weights,
            "grade_threshold": effective_cfg.grade_threshold,
            "grade_strategy": effective_cfg.grade_strategy,
            "score_normalizers": effective_cfg.score_normalizers,
            "fallback_chain": effective_cfg.fallback_chain,
            "max_retries": effective_cfg.max_retries,
            "processors": effective_cfg.processors,
            "top_k": effective_cfg.top_k,
            "gen_mode": effective_cfg.gen_mode,
            "temperature": effective_cfg.temperature,
        }
        final_state = self._graph.invoke(initial_state, {"configurable": configurable})
        return QueryResult(
            answer=final_state.get("answer", ""),
            ranked_documents=final_state.get("ranked_documents", []),
            grade_score=final_state.get("grade_score", 0.0),
            grade_decision=final_state.get("grade_decision", "insufficient"),
            rewritten_queries=final_state.get("rewritten_queries", []),
            retrieval_rounds=len(final_state.get("retrieval_history", [])),
        )

    def close(self) -> None:
        """显式释放所有 providers 资源。"""
        try:
            self._reranker.close()
        except Exception as e:
            logger.warning("Reranker close failed: %s", e)
        try:
            self._vector_store.close()
        except Exception as e:
            logger.warning("VectorStore close failed: %s", e)
        try:
            self._llm_client.close()
        except Exception as e:
            logger.warning("LLMClient close failed: %s", e)
        try:
            self._embedding_client.close()
        except Exception as e:
            logger.warning("EmbeddingClient close failed: %s", e)
        try:
            self._bm25.close()
        except Exception as e:
            logger.warning("BM25 close failed: %s", e)

    def __enter__(self) -> "RAG":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    # 内部方法 ---------------------------------------------------------

    def _warn_if_handlers_already_registered(self) -> None:
        """检测前一个 RAG 实例已注册过内置 handlers，发 warning。"""
        existing_names = set(transform_registry.names())
        if _BUILTIN_TRANSFORMS & existing_names:
            logger.warning(
                "Multiple RAG instances in same process will share Registry "
                "singleton. Previous handlers will be overwritten. MVP 仅支持单 "
                "RAG 实例 per process；多实例场景请使用独立进程。"
            )

    def _validate_config(self) -> None:
        """fail-fast 配置校验，一次性收集所有问题。"""
        errors: list[str] = []
        cfg = self._config

        # API key 校验
        api_key = cfg.providers.llm_api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            errors.append("OPENAI_API_KEY missing: set env var or providers.llm_api_key")

        # persist_dir 可写
        persist_dir = Path(cfg.ingest.persist_dir)
        # 父目录若不存在创建（chroma 自己会创建子目录）；这里只检查父目录可写性
        parent = persist_dir if persist_dir.exists() else persist_dir.parent
        if parent.exists() and not os.access(parent, os.W_OK):
            errors.append(f"persist_dir '{persist_dir}' parent not writable")

        # transforms：顶层分支可以是 str 或 list[str]
        for branch in cfg.graph.transforms:
            names = [branch] if isinstance(branch, str) else list(branch)
            for name in names:
                if not transform_registry.has(name):
                    errors.append(f"transform '{name}' not registered. Available: {transform_registry.names()}")

        for name in cfg.graph.retrievers:
            if not retriever_registry.has(name):
                errors.append(f"retriever '{name}' not registered. Available: {retriever_registry.names()}")

        for name in cfg.graph.processors:
            if not processor_registry.has(name):
                errors.append(f"processor '{name}' not registered. Available: {processor_registry.names()}")

        if not grader_registry.has(cfg.graph.grade_strategy):
            errors.append(f"grader '{cfg.graph.grade_strategy}' not registered. Available: {grader_registry.names()}")

        if not generator_registry.has(cfg.graph.gen_mode):
            errors.append(f"generator '{cfg.graph.gen_mode}' not registered. Available: {generator_registry.names()}")

        # ingest.parser
        if cfg.ingest.parser != "auto" and not parser_registry.has(cfg.ingest.parser):
            errors.append(f"parser '{cfg.ingest.parser}' not registered. Available: {parser_registry.names()}")

        if not chunker_registry.has(cfg.ingest.chunker):
            errors.append(f"chunker '{cfg.ingest.chunker}' not registered. Available: {chunker_registry.names()}")

        # fallback_chain 中每个 source 必须是已注册 retriever 名
        for source in cfg.graph.fallback_chain:
            if not retriever_registry.has(source):
                errors.append(
                    f"fallback_chain source '{source}' not registered as retriever. "
                    f"Available: {retriever_registry.names()}"
                )

        if errors:
            raise RaglineConfigError("Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    def _bootstrap_bm25(self) -> None:
        """若 retrievers 含 'bm25'，从 vector_store 分批拉取已 ingest 的 chunks
        重建 BM25 索引（避免大语料 OOM）。"""
        if "bm25" not in self._config.graph.retrievers:
            return
        all_chunks = []
        for batch in self._vector_store.iter_chunks(batch_size=1000):
            all_chunks.extend(batch)
        if all_chunks:
            self._bm25.rebuild(all_chunks)
            logger.info("Bootstrapped BM25 index with %d chunks", len(all_chunks))
