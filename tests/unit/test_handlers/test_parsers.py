"""tests/unit/test_handlers/test_parsers.py — Parsers 单元测试。"""

from pathlib import Path

import pypdf
import pytest

from fugue.handlers.parsers import (
    auto_parser,
    markdown_parser,
    pdf_parser,
    text_parser,
)

FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# PDF fixture（session 级，动态生成，避免 commit 二进制）
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_pdf(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """生成最小 2 页 PDF 用于测试。"""
    pdf_path = tmp_path_factory.mktemp("pdf") / "sample.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    return pdf_path


# ---------------------------------------------------------------------------
# clean_parser_registry fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_parser_registry():
    from fugue.registry import parser_registry

    saved = {n: parser_registry.get(n) for n in parser_registry.names()}
    for n in list(parser_registry.names()):
        parser_registry.unregister(n)
    yield parser_registry
    for n in list(parser_registry.names()):
        parser_registry.unregister(n)
    for n, fn in saved.items():
        parser_registry.register(n, fn)


# ---------------------------------------------------------------------------
# 测试 1: markdown_parser
# ---------------------------------------------------------------------------


def test_markdown_parser_basic() -> None:
    path = FIXTURES_DIR / "sample.md"
    result = markdown_parser(path)

    assert isinstance(result, list)
    assert len(result) == 1

    doc = result[0]
    assert isinstance(doc.source_path, Path)
    assert doc.source_path == path
    assert doc.content != ""
    assert doc.metadata == {"format": "markdown"}


# ---------------------------------------------------------------------------
# 测试 2: text_parser
# ---------------------------------------------------------------------------


def test_text_parser_basic() -> None:
    path = FIXTURES_DIR / "sample.txt"
    result = text_parser(path)

    assert isinstance(result, list)
    assert len(result) == 1

    doc = result[0]
    assert isinstance(doc.source_path, Path)
    assert doc.source_path == path
    assert doc.content != ""
    assert doc.metadata == {"format": "text"}


# ---------------------------------------------------------------------------
# 测试 3: pdf_parser（用 session fixture 生成的 PDF）
# ---------------------------------------------------------------------------


def test_pdf_parser_basic(sample_pdf: Path) -> None:
    result = pdf_parser(sample_pdf)

    assert isinstance(result, list)
    assert len(result) == 2  # 两页 blank PDF

    for doc in result:
        assert doc.metadata.get("format") == "pdf"
        assert "page" in doc.metadata
        assert isinstance(doc.source_path, Path)


# ---------------------------------------------------------------------------
# 测试 4: auto_parser 分派 .md
# ---------------------------------------------------------------------------


def test_auto_parser_dispatches_md() -> None:
    path = FIXTURES_DIR / "sample.md"
    auto_result = auto_parser(path)
    direct_result = markdown_parser(path)

    assert len(auto_result) == len(direct_result)
    assert auto_result[0].content == direct_result[0].content
    assert auto_result[0].metadata == direct_result[0].metadata


# ---------------------------------------------------------------------------
# 测试 5: auto_parser 分派 .txt
# ---------------------------------------------------------------------------


def test_auto_parser_dispatches_txt() -> None:
    path = FIXTURES_DIR / "sample.txt"
    auto_result = auto_parser(path)
    direct_result = text_parser(path)

    assert len(auto_result) == len(direct_result)
    assert auto_result[0].content == direct_result[0].content
    assert auto_result[0].metadata == direct_result[0].metadata


# ---------------------------------------------------------------------------
# 测试 6: auto_parser 分派 .pdf
# ---------------------------------------------------------------------------


def test_auto_parser_dispatches_pdf(sample_pdf: Path) -> None:
    auto_result = auto_parser(sample_pdf)
    direct_result = pdf_parser(sample_pdf)

    assert len(auto_result) == len(direct_result)
    assert auto_result[0].metadata == direct_result[0].metadata


# ---------------------------------------------------------------------------
# 测试 7: auto_parser 未知扩展抛 ValueError
# ---------------------------------------------------------------------------


def test_auto_parser_unknown_extension(tmp_path: Path) -> None:
    unknown = tmp_path / "file.xyz"
    unknown.write_text("content")

    with pytest.raises(ValueError, match="unsupported extension"):
        auto_parser(unknown)


# ---------------------------------------------------------------------------
# 测试 8: markdown_parser 空文件
# ---------------------------------------------------------------------------


def test_markdown_parser_empty_file() -> None:
    path = FIXTURES_DIR / "sample_empty.md"
    result = markdown_parser(path)

    assert len(result) == 1
    assert result[0].content == ""


# ---------------------------------------------------------------------------
# 测试 9: register_parsers 注册（clean_parser_registry）
# ---------------------------------------------------------------------------


def test_register_parsers_registers_all(clean_parser_registry) -> None:
    from fugue.handlers.parsers import register_parsers

    register_parsers()

    assert clean_parser_registry.has("markdown")
    assert clean_parser_registry.has("text")
    assert clean_parser_registry.has("pdf")
    assert clean_parser_registry.has("auto")


# ---------------------------------------------------------------------------
# 测试 10: .markdown 扩展也支持
# ---------------------------------------------------------------------------


def test_auto_parser_dotmarkdown_extension(tmp_path: Path) -> None:
    f = tmp_path / "file.markdown"
    f.write_text("# 标题\n内容")

    result = auto_parser(f)

    assert len(result) == 1
    assert result[0].metadata == {"format": "markdown"}


# ---------------------------------------------------------------------------
# 测试 11: 扩展名大小写不敏感（.MD）
# ---------------------------------------------------------------------------


def test_auto_parser_uppercase_extension(tmp_path: Path) -> None:
    f = tmp_path / "file.MD"
    f.write_text("# 大写扩展\n测试内容")

    result = auto_parser(f)

    assert len(result) == 1
    assert result[0].metadata == {"format": "markdown"}
