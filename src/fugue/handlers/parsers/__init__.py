"""src/fugue/handlers/parsers/__init__.py — 注册 + auto 分派。"""

from pathlib import Path

from fugue.api.types import ParsedDocument
from fugue.handlers.parsers.markdown_text import markdown_parser, text_parser
from fugue.handlers.parsers.pdf import pdf_parser
from fugue.registry import parser_registry  # noqa: E402

_EXTENSION_PARSERS = {
    ".md": markdown_parser,
    ".markdown": markdown_parser,
    ".txt": text_parser,
    ".pdf": pdf_parser,
}


def auto_parser(path: Path) -> list[ParsedDocument]:
    """按扩展名分派 parser。未知扩展抛 ValueError。"""
    ext = path.suffix.lower()
    parser = _EXTENSION_PARSERS.get(ext)
    if parser is None:
        raise ValueError(
            f"unsupported extension '{ext}' for file '{path}'. "
            f"Supported: {sorted(_EXTENSION_PARSERS.keys())}"
        )
    return parser(path)


def register_parsers() -> None:
    """注册 markdown / text / pdf / auto 到 parser_registry。无依赖。"""
    parser_registry.register("markdown", markdown_parser)
    parser_registry.register("text", text_parser)
    parser_registry.register("pdf", pdf_parser)
    parser_registry.register("auto", auto_parser)


# 模块 import 即注册（无依赖直接执行副作用）
register_parsers()


__all__ = [
    "auto_parser",
    "markdown_parser",
    "pdf_parser",
    "register_parsers",
    "text_parser",
]
