"""src/fugue/handlers/parsers/markdown_text.py — Markdown 与纯文本 parser。"""

from pathlib import Path

from fugue.api.types import ParsedDocument


def markdown_parser(path: Path) -> list[ParsedDocument]:
    """读取 .md 文件，返回单个 ParsedDocument。content 为原文。"""
    content = path.read_text(encoding="utf-8")
    return [
        ParsedDocument(
            source_path=path,
            content=content,
            metadata={"format": "markdown"},
        )
    ]


def text_parser(path: Path) -> list[ParsedDocument]:
    """读取 .txt 文件，返回单个 ParsedDocument。"""
    content = path.read_text(encoding="utf-8")
    return [
        ParsedDocument(
            source_path=path,
            content=content,
            metadata={"format": "text"},
        )
    ]
