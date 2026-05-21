"""tests/conftest.py — 顶层 pytest 共享 fixture。

提供两个可选 fixture 供测试按需使用：

- ``isolated_registries_fx``：在测试期间隔离所有全局 Registry，退出后完全恢复。
- ``mock_rag_providers_fx``：patch LLMClient / EmbeddingClient 为 Fake 实现，
  并隔离 Registry，yield (FakeLLM, FakeEmbedding) 元组供断言使用。

两个 fixture 均**不** autouse，避免污染未使用它们的测试。
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

from ragline.testing import FakeEmbedding, FakeLLM, isolated_registries, mock_rag_providers


@pytest.fixture
def isolated_registries_fx() -> Generator[None, None, None]:
    """隔离所有全局 Registry，退出后完全恢复。"""
    with isolated_registries():
        yield


@pytest.fixture
def mock_rag_providers_fx() -> Generator[tuple[FakeLLM, FakeEmbedding], None, None]:
    """patch LLMClient / EmbeddingClient 为 Fake 实现，yield (FakeLLM, FakeEmbedding)。"""
    with mock_rag_providers() as fakes:
        yield fakes
