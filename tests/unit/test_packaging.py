"""
测试 pyproject.toml 的 [project] 元数据字段 + CI 覆盖率门控配置。
TDD：先写测试看 RED，再实现看 GREEN。
"""

from __future__ import annotations

import subprocess
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


# ── LICENSE 文件测试 ──────────────────────────────────────────────────────────

LICENSE_PATH = ROOT / "LICENSE"


@pytest.fixture(scope="module")
def license_text() -> str:
    return LICENSE_PATH.read_text(encoding="utf-8")


# ── 场景 10：LICENSE 文件存在 ────────────────────────────────────────────────


def test_license_file_exists() -> None:
    assert LICENSE_PATH.exists(), "项目根目录缺少 LICENSE 文件"
    assert LICENSE_PATH.is_file(), "LICENSE 不是普通文件"


# ── 场景 11：LICENSE 第一行为 MIT License ────────────────────────────────────


def test_license_first_line(license_text: str) -> None:
    first_line = license_text.splitlines()[0]
    assert first_line == "MIT License", f"LICENSE 第一行应为 'MIT License'，实际为 {first_line!r}"


# ── 场景 12：LICENSE 含版权行 ────────────────────────────────────────────────


def test_license_copyright(license_text: str) -> None:
    assert "Copyright (c) 2026 dengdi" in license_text


# ── 场景 13：LICENSE 含 MIT 关键短语 ─────────────────────────────────────────


def test_license_mit_phrases(license_text: str) -> None:
    required_phrases = [
        "Permission is hereby granted, free of charge",
        "WITHOUT WARRANTY OF ANY KIND",
        "INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY",
    ]
    for phrase in required_phrases:
        assert phrase in license_text, f"LICENSE 缺少关键短语: {phrase!r}"


# ── 场景 14：LICENSE 行数在 21-23 之间 ──────────────────────────────────────


def test_license_line_count(license_text: str) -> None:
    lines = license_text.splitlines()
    count = len(lines)
    assert 21 <= count <= 23, f"LICENSE 行数应在 21-23 之间，实际为 {count}"


REQUIRED_CORE_PREFIXES = [
    "langgraph",
    "langchain-core",
    "openai",
    "pydantic",
    "pyyaml",
    "rank-bm25",
    "chromadb",
]

# ── 场景 15-20：transformers 迁移到 [bge] extra ───────────────────────────────


# ── 场景 15：core deps 不含 transformers ─────────────────────────────────────


def test_core_deps_no_transformers(project: dict[str, Any]) -> None:
    core_deps: list[str] = project["dependencies"]
    for dep in core_deps:
        assert not dep.lower().startswith("transformers"), (
            f"transformers 不应出现在 core dependencies 中，实际发现: {dep!r}"
        )


# ── 场景 16：bge extra 包含 transformers>=4.44.2,<5 ──────────────────────────


def test_bge_extra_contains_transformers(pyproject: dict[str, Any]) -> None:
    bge_deps: list[str] = pyproject["project"]["optional-dependencies"]["bge"]
    assert any(dep.startswith("transformers>=") for dep in bge_deps), (
        f"[bge] extra 应包含以 'transformers>=' 开头的依赖，实际内容: {bge_deps}"
    )


# ── 场景 17：bge extra 仍包含 FlagEmbedding>=1.2 ─────────────────────────────


def test_bge_extra_still_has_flag_embedding(pyproject: dict[str, Any]) -> None:
    bge_deps: list[str] = pyproject["project"]["optional-dependencies"]["bge"]
    assert any(dep.startswith("FlagEmbedding>=1.2") for dep in bge_deps), (
        f"[bge] extra 应仍包含 FlagEmbedding>=1.2，实际内容: {bge_deps}"
    )


# ── 场景 18：all extra 仍引用 ragline[server,bge,chroma,pdf] ─────────────────


def test_all_extra_unchanged(pyproject: dict[str, Any]) -> None:
    all_deps: list[str] = pyproject["project"]["optional-dependencies"]["all"]
    assert "ragline[server,bge,chroma,pdf]" in all_deps, (
        f"[all] extra 应包含 'ragline[server,bge,chroma,pdf]'，实际内容: {all_deps}"
    )


# ── 场景 19：core deps 仍包含所有其他必要依赖 ────────────────────────────────


def test_core_deps_contain_required_packages(project: dict[str, Any]) -> None:
    core_deps: list[str] = project["dependencies"]
    for prefix in REQUIRED_CORE_PREFIXES:
        assert any(dep.lower().startswith(prefix.lower()) for dep in core_deps), (
            f"core dependencies 缺少以 {prefix!r} 开头的依赖"
        )


# ── 场景 20：uv lock --check 退出码为 0 ──────────────────────────────────────


def test_uv_lock_check() -> None:
    try:
        result = subprocess.run(
            ["uv", "lock", "--check"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError:
        pytest.skip("uv 未安装，跳过 lock 一致性检查")
    assert result.returncode == 0, f"uv lock --check 退出码非 0\nstdout: {result.stdout}\nstderr: {result.stderr}"
