"""tests/e2e/conftest.py — E2E 测试共享 fixtures + auto marker。"""

import os
from pathlib import Path

import pytest

from fugue import ProviderConfig

# E2E provider 默认走 OpenRouter（OpenAI 兼容，chat + embedding 同一端点）。
# 可通过环境变量覆盖以切回 OpenAI 原生或其他兼容端点。
_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_LLM_MODEL = "openai/gpt-4o-mini"
_DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """所有 tests/e2e/ 下的测试自动加 e2e marker。"""
    for item in items:
        if "tests/e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)


@pytest.fixture(scope="session")
def openai_key() -> str:
    """读取 OPENAI_API_KEY，未设置时 skip 整个测试。"""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set — skipping e2e tests")
    return key


@pytest.fixture(scope="session")
def e2e_provider(openai_key: str) -> ProviderConfig:
    """共享 ProviderConfig：默认 OpenRouter，可经环境变量覆盖。

    - base_url:        FUGUE_E2E_BASE_URL
    - llm_model:       FUGUE_E2E_LLM_MODEL
    - embedding_model: FUGUE_E2E_EMBEDDING_MODEL
    - api_key:         OPENAI_API_KEY（经 openai_key fixture，未设置则 skip）

    embedding 复用同一 base_url + api_key（RAG 内部回退逻辑）。
    """
    base_url = os.environ.get("FUGUE_E2E_BASE_URL", _DEFAULT_BASE_URL)
    return ProviderConfig(
        llm_base_url=base_url,
        llm_api_key=openai_key,
        llm_model=os.environ.get("FUGUE_E2E_LLM_MODEL", _DEFAULT_LLM_MODEL),
        embedding_model=os.environ.get("FUGUE_E2E_EMBEDDING_MODEL", _DEFAULT_EMBEDDING_MODEL),
    )


@pytest.fixture(scope="session")
def e2e_fixtures_dir() -> Path:
    """E2E fixtures 目录。"""
    return Path(__file__).parent.parent / "fixtures" / "e2e"
