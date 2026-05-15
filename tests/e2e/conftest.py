"""tests/e2e/conftest.py — E2E 测试共享 fixtures + auto marker。"""

import os
from pathlib import Path

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
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
def e2e_fixtures_dir() -> Path:
    """E2E fixtures 目录。"""
    return Path(__file__).parent.parent / "fixtures" / "e2e"
