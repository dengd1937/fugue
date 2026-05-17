"""src/fugue/handlers/parsers/pdf.py — PDF parser 基于 pypdf。"""

from pathlib import Path

from fugue._optional import require
from fugue.api.types import ParsedDocument


def pdf_parser(path: Path) -> list[ParsedDocument]:
    """用 pypdf 解析 PDF 文件，每页一个 ParsedDocument。"""
    pypdf = require("pypdf", extra="pdf")
    reader = pypdf.PdfReader(str(path))
    results: list[ParsedDocument] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        results.append(
            ParsedDocument(
                source_path=path,
                content=text,
                metadata={"format": "pdf", "page": i},
            )
        )
    return results
