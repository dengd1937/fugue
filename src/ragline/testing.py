"""src/ragline/testing.py — 为外部消费者提供测试工具（Fake 实现与上下文管理器）。

此模块是 ragline 的公开测试 API，供下游包在单元测试中使用，
无需真实的 LLM/Embedding 网络调用或 Registry 副作用。
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

from ragline.registry import (
    chunker_registry,
    generator_registry,
    grader_registry,
    parser_registry,
    processor_registry,
    retriever_registry,
    transform_registry,
)

__all__ = ["FakeLLM", "FakeEmbedding", "isolated_registries", "mock_rag_providers"]

# ---------------------------------------------------------------------------
# FakeLLM
# ---------------------------------------------------------------------------


class FakeLLM:
    """LLMClient 的测试替身，无网络调用。

    Attributes:
        answer: complete() 返回的字符串，可在运行时修改。
        calls: 记录每次 complete() 调用的 (prompt, kwargs) 元组列表。
        close_calls: close() 被调用的次数。
    """

    def __init__(self, answer: str = "fake answer") -> None:
        self.answer = answer
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.close_calls: int = 0

    def complete(self, prompt: str, *, temperature: float = 0.7) -> str:
        """模拟 LLMClient.complete，记录调用并返回 self.answer。"""
        self.calls.append((prompt, {"temperature": temperature}))
        return self.answer

    def close(self) -> None:
        """模拟 close，递增计数器，幂等。"""
        self.close_calls += 1


# ---------------------------------------------------------------------------
# FakeEmbedding
# ---------------------------------------------------------------------------


class FakeEmbedding:
    """EmbeddingClient 的测试替身，无网络调用。

    Attributes:
        dim: 每个向量的维度，默认 4。
        calls: 记录每次 embed() 调用的 texts 列表。
        close_calls: close() 被调用的次数。
    """

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []
        self.close_calls: int = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        """模拟 EmbeddingClient.embed，记录调用并返回固定向量。"""
        self.calls.append(texts)
        return [[0.1] * self.dim for _ in texts]

    def close(self) -> None:
        """模拟 close，递增计数器，幂等。"""
        self.close_calls += 1


# ---------------------------------------------------------------------------
# isolated_registries
# ---------------------------------------------------------------------------


@contextmanager
def isolated_registries() -> Generator[None, None, None]:
    """上下文管理器：进入时清空所有 7 个全局 registry，退出时完全恢复。

    保存原始的 name → handler 映射，退出时清空并重新注册。
    使用 try/finally 保证即使异常也能恢复。

    Example::

        with isolated_registries():
            rag = RAG()  # 从空 registry 开始注册内置 handlers
    """
    _registries = [
        transform_registry,
        retriever_registry,
        processor_registry,
        grader_registry,
        generator_registry,
        parser_registry,
        chunker_registry,
    ]

    # snapshot：保存每个 registry 的 name → handler 映射
    snapshots: list[dict[str, Any]] = []
    for reg in _registries:
        snapshot: dict[str, Any] = {name: reg.get(name) for name in reg.names()}
        snapshots.append(snapshot)

    # 清空所有 registry
    for reg in _registries:
        for name in reg.names():
            reg.unregister(name)

    try:
        yield
    finally:
        # 恢复：先清空（yield 期间可能注册了新的），再重新注册原 handlers
        for reg, snapshot in zip(_registries, snapshots, strict=True):
            for name in reg.names():
                reg.unregister(name)
            for name, handler in snapshot.items():
                reg.register(name, handler)


# ---------------------------------------------------------------------------
# mock_rag_providers
# ---------------------------------------------------------------------------


@contextmanager
def mock_rag_providers(
    llm: FakeLLM | None = None,
    embedding: FakeEmbedding | None = None,
) -> Generator[tuple[FakeLLM, FakeEmbedding], None, None]:
    """上下文管理器：patch ragline.api.rag 中的 LLMClient 与 EmbeddingClient，
    使 RAG() 实例化时使用 fake 实现，同时隔离全局 registries。

    Args:
        llm: 自定义 FakeLLM；未提供则自动创建。
        embedding: 自定义 FakeEmbedding；未提供则自动创建。

    Yields:
        (fake_llm, fake_embedding) 元组，供测试断言使用。

    Example::

        with mock_rag_providers() as (llm, embedding):
            rag = RAG()
            rag.query("hello")
            assert llm.calls
    """
    fake_llm = llm if llm is not None else FakeLLM()
    fake_embedding = embedding if embedding is not None else FakeEmbedding()

    # 构建 ChromaVectorStore 的 fake（满足 VectorStore Protocol）
    fake_vector_store = MagicMock()
    fake_vector_store.similarity_search.return_value = []
    fake_vector_store.iter_chunks.return_value = iter([])
    fake_vector_store.stats.return_value = {"num_chunks": 0}

    with (
        isolated_registries(),
        patch("ragline.api.rag.LLMClient", return_value=fake_llm) as _llm_patch,
        patch("ragline.api.rag.EmbeddingClient", return_value=fake_embedding) as _emb_patch,
        patch(
            "ragline.api.rag.ChromaVectorStore",
            return_value=fake_vector_store,
        ),
        # 设置 OPENAI_API_KEY 以通过 fail-fast 校验
        patch.dict(
            "os.environ",
            {"OPENAI_API_KEY": "fake-key-for-testing"},
            clear=False,
        ),
    ):
        yield (fake_llm, fake_embedding)
