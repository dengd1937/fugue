"""
测试 pyproject.toml 的 [project] 元数据字段 + CI 覆盖率门控配置。
TDD：先写测试看 RED，再实现看 GREEN。
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).parent.parent.parent
PYPROJECT_PATH = ROOT / "pyproject.toml"
CI_YML_PATH = ROOT / ".github" / "workflows" / "ci.yml"


@pytest.fixture(scope="module")
def pyproject() -> dict[str, Any]:
    with PYPROJECT_PATH.open("rb") as f:
        return tomllib.load(f)


@pytest.fixture(scope="module")
def project(pyproject: dict[str, Any]) -> dict[str, Any]:
    return dict(pyproject["project"])


@pytest.fixture(scope="module")
def ci_content() -> str:
    return CI_YML_PATH.read_text(encoding="utf-8")


# ── 场景 1：authors ──────────────────────────────────────────────────────────


def test_authors(project: dict[str, Any]) -> None:
    authors = project["authors"]
    assert len(authors) == 1
    assert authors[0]["name"] == "dengdi"
    assert authors[0]["email"] == "dengdi1803@gmail.com"


# ── 场景 2：license ──────────────────────────────────────────────────────────


def test_license(project: dict[str, Any]) -> None:
    assert project["license"] == "MIT"


# ── 场景 3：license-files ────────────────────────────────────────────────────


def test_license_files(project: dict[str, Any]) -> None:
    assert project["license-files"] == ["LICENSE"]


# ── 场景 4：keywords ─────────────────────────────────────────────────────────


def test_keywords(project: dict[str, Any]) -> None:
    expected = ["rag", "llm", "langgraph", "retrieval", "embedding", "chromadb", "bm25"]
    assert project["keywords"] == expected


# ── 场景 5：classifiers ──────────────────────────────────────────────────────

REQUIRED_CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Typing :: Typed",
]


def test_classifiers(project: dict[str, Any]) -> None:
    classifiers = set(project["classifiers"])
    for cls in REQUIRED_CLASSIFIERS:
        assert cls in classifiers, f"缺少 classifier: {cls!r}"


# ── 场景 6：urls ─────────────────────────────────────────────────────────────


def test_urls(project: dict[str, Any]) -> None:
    urls = project["urls"]
    assert urls["Homepage"] == "https://github.com/dengd1937/ragline"
    assert urls["Repository"] == "https://github.com/dengd1937/ragline"
    assert urls["Issues"] == "https://github.com/dengd1937/ragline/issues"
    assert urls["Changelog"] == "https://github.com/dengd1937/ragline/blob/main/CHANGELOG.md"


# ── 场景 7：未破坏既有非依赖字段 ──────────────────────────────────────────────


def test_existing_fields_intact(project: dict[str, Any]) -> None:
    assert project["name"] == "ragline"
    assert project["version"] == "0.1.0"
    assert project["requires-python"] == ">=3.12"


# ── 场景 8：addopts 含 --cov-fail-under=96 ───────────────────────────────────


def test_addopts_cov_fail_under(pyproject: dict[str, Any]) -> None:
    addopts: str = pyproject["tool"]["pytest"]["ini_options"]["addopts"]
    assert "--cov-fail-under=96" in addopts


# ── 场景 9：ci.yml 含 --cov-fail-under=96 且不含 --cov-fail-under=90 ─────────


def test_ci_cov_fail_under(ci_content: str) -> None:
    assert "--cov-fail-under=96" in ci_content
    assert "--cov-fail-under=90" not in ci_content
