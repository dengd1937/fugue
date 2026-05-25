"""
测试 pyproject.toml 的 [project] 元数据字段 + CI 覆盖率门控配置。
TDD：先写测试看 RED，再实现看 GREEN。
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path
from typing import Any, Final

import pytest

import ragline

ROOT = Path(__file__).parent.parent.parent
PYPROJECT_PATH = ROOT / "pyproject.toml"
CI_YML_PATH = ROOT / ".github" / "workflows" / "ci.yml"

# ── 预编译正则（模块级）──────────────────────────────────────────────────────
_H4_RE = re.compile(r"^#### (.+)$", re.MULTILINE)
_SECTION_BOUND_RE = re.compile(r"^#{2,3} ", re.MULTILINE)
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:[.-].+)?$")


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


# ── CHANGELOG.md 测试 ─────────────────────────────────────────────────────────

CHANGELOG_PATH = ROOT / "CHANGELOG.md"


@pytest.fixture(scope="module")
def changelog_text() -> str:
    if not CHANGELOG_PATH.exists():
        pytest.skip("CHANGELOG.md 不存在")
    return CHANGELOG_PATH.read_text(encoding="utf-8")


# ── 场景 21：CHANGELOG.md 存在 ───────────────────────────────────────────────


def test_changelog_exists() -> None:
    assert CHANGELOG_PATH.exists(), "项目根目录缺少 CHANGELOG.md 文件"
    assert CHANGELOG_PATH.is_file(), "CHANGELOG.md 不是普通文件"


# ── 场景 22：第一行以 # Changelog 开头 ──────────────────────────────────────


def test_changelog_first_line(changelog_text: str) -> None:
    first_line = changelog_text.splitlines()[0]
    assert first_line.startswith("# Changelog"), f"CHANGELOG.md 第一行应以 '# Changelog' 开头，实际为 {first_line!r}"


# ── 场景 23：含 Keep a Changelog 子串 ───────────────────────────────────────


def test_changelog_keep_a_changelog(changelog_text: str) -> None:
    assert "Keep a Changelog" in changelog_text, "CHANGELOG.md 应引用 'Keep a Changelog'"


# ── 场景 24：含 Semantic Versioning 子串 ─────────────────────────────────────


def test_changelog_semantic_versioning(changelog_text: str) -> None:
    assert "Semantic Versioning" in changelog_text, "CHANGELOG.md 应引用 'Semantic Versioning'"


# ── 场景 25：含 0.1.0 节标题（固定日期 2026-05-24）────────────────────────────


def test_changelog_version_section(changelog_text: str) -> None:
    assert "## [0.1.0] - 2026-05-24" in changelog_text, "CHANGELOG.md 应含 '## [0.1.0] - 2026-05-24' 节标题"


# ── 场景 26：0.1.0 节下含 ### Added 段标题 ──────────────────────────────────


def test_changelog_added_section(changelog_text: str) -> None:
    version_idx = changelog_text.index("## [0.1.0]")
    section_after = changelog_text[version_idx:]
    assert "### Added" in section_after, "## [0.1.0] 节下应含 '### Added' 段标题"


# ── 场景 27：### Added 段下含 7 个 #### 子模块标题及对应关键词 ──────────────


_SUBMODULE_KEYWORDS: Final[list[tuple[str, tuple[str, ...]]]] = [
    ("顶层 API", ("RAG", "RaglineConfig")),
    ("检索引擎", ("LangGraph", "query_transform")),
    ("内置 Handlers", ("transforms", "retrievers", "rrf")),
    ("Providers", ("LLM", "Embedding", "ChromaDB")),
    ("HTTP Server", ("FastAPI", "[server]")),
    ("对外测试支持", ("FakeLLM", "ragline.testing")),
    ("工程基线", ("py.typed", "coverage", "import-linter")),
]


def test_changelog_submodule_keywords(changelog_text: str) -> None:
    """### Added 下 7 个子模块标题均存在，且各自后续内容含对应关键词（宽松匹配：每个子模块至少含一个关键词）。"""
    added_idx = changelog_text.index("### Added")
    # 取 ### Added 之后的内容（截止到下一个同级 ### 或 ## 节）
    rest = changelog_text[added_idx + len("### Added") :]
    # 找下一个同级或父级 ### / ## 节的位置作为 Added 段的结束（不匹配 ####）
    next_section = _SECTION_BOUND_RE.search(rest)
    added_body = rest[: next_section.start()] if next_section is not None else rest

    # 验证 7 个 #### 子模块标题存在
    h4_titles = _H4_RE.findall(added_body)
    assert len(h4_titles) >= 7, f"### Added 下应含至少 7 个 #### 子模块标题，实际找到 {len(h4_titles)} 个: {h4_titles}"

    # 对每个子模块验证关键词（宽松匹配：含任一关键词即可）
    for module_hint, keywords in _SUBMODULE_KEYWORDS:
        # 找对应子模块块
        pattern = rf"(?s)#### [^\n]*{re.escape(module_hint.split('（')[0].strip())}[^\n]*\n(.*?)(?=\n####|\Z)"
        match = re.search(pattern, added_body)
        assert match is not None, f"### Added 下未找到含 '{module_hint}' 的 #### 子模块标题"
        block_text = match.group(0)
        assert any(kw in block_text for kw in keywords), (
            f"子模块 '{module_hint}' 的内容应含关键词 {keywords} 之一，实际内容片段: {block_text[:200]!r}"
        )


# ── 场景 28：含 ### Notes 段，说明 0.x API 可能调整 ─────────────────────────


def test_changelog_notes_section(changelog_text: str) -> None:
    assert "### Notes" in changelog_text, "CHANGELOG.md 应含 '### Notes' 段"
    notes_idx = changelog_text.index("### Notes")
    notes_body = changelog_text[notes_idx:]
    assert "0.x" in notes_body, "### Notes 段应提及 '0.x' API 可能调整"


# ── 场景 29：ragline.__version__ 属性存在且格式正确 ──────────────────────────


def test_version_attribute_exists() -> None:
    assert hasattr(ragline, "__version__"), "ragline 应暴露 __version__ 属性"
    assert isinstance(ragline.__version__, str), "__version__ 应为 str 类型"
    assert _SEMVER_RE.match(ragline.__version__), f"__version__ 应符合 SemVer 格式，实际值: {ragline.__version__!r}"


# ── 场景 30：ragline.__version__ 与 pyproject.toml 中的版本一致 ──────────────


def test_version_matches_pyproject(pyproject: dict[str, Any]) -> None:
    assert ragline.__version__ == pyproject["project"]["version"], (
        f"ragline.__version__ ({ragline.__version__!r}) 应与 pyproject.toml 中的版本 "
        f"({pyproject['project']['version']!r}) 一致"
    )
