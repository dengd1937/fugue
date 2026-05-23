"""tests/unit/test_handlers/test_parsers.py — Parsers 单元测试。"""

from pathlib import Path

import pypdf
import pytest

from ragline.handlers.parsers import (
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


def test_register_parsers_registers_all(isolated_registries_fx) -> None:
    from ragline.handlers.parsers import register_parsers
    from ragline.registry import parser_registry

    register_parsers()

    assert parser_registry.has("markdown")
    assert parser_registry.has("text")
    assert parser_registry.has("pdf")
    assert parser_registry.has("auto")


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


# ---------------------------------------------------------------------------
# 测试 12: pdf_parser 缺失 pypdf 时抛 ImportError（patch require 接缝）
# ---------------------------------------------------------------------------


def test_pdf_parser_missing_pypdf_raises(tmp_path: Path) -> None:
    from unittest.mock import patch

    dummy_pdf = tmp_path / "dummy.pdf"
    dummy_pdf.write_bytes(b"")

    with (
        patch(
            "ragline.handlers.parsers.pdf.require",
            side_effect=ImportError("可选依赖 'pypdf' 未安装。请运行: pip install 'ragline[pdf]'"),
        ),
        pytest.raises(ImportError) as exc_info,
    ):
        pdf_parser(dummy_pdf)

    assert "pip install 'ragline[pdf]'" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 测试 13: pdf.py 模块顶层 namespace 无 pypdf 属性（已移除 eager import）
# ---------------------------------------------------------------------------


def test_pdf_module_no_toplevel_pypdf_import() -> None:
    import ragline.handlers.parsers.pdf as pdfmod

    assert not hasattr(pdfmod, "pypdf")
