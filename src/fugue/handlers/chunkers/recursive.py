"""src/fugue/handlers/chunkers/recursive.py — 递归字符切分。"""

import hashlib
from typing import Any

from fugue.api.types import Chunk, ParsedDocument

# 递归分隔符，按优先级降级
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _hash_id(*parts: str) -> str:
    """计算 sha1 hex 前 16 位作为 ID。"""
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()[:16]


def _recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> list[str]:
    """递归切分文本到 ≤ chunk_size 大小，含 chunk_overlap 重叠。"""
    if len(text) <= chunk_size:
        return [text] if text else []

    # 选择第一个有效分隔符
    separator = ""
    remaining_seps = separators
    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            separator = sep
            remaining_seps = separators[i + 1 :]
            break

    if separator == "":
        # 字符级切分（最末降级），含 overlap
        return _char_split_with_overlap(text, chunk_size, chunk_overlap)

    splits = text.split(separator)
    # 合并小片段使得每个 chunk 接近 chunk_size，必要时递归切分大片段
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for piece in splits:
        piece_len = len(piece) + (len(separator) if current else 0)
        if current_len + piece_len > chunk_size and current:
            # 当前 batch 已达上限，flush
            merged = separator.join(current)
            if len(merged) <= chunk_size:
                chunks.append(merged)
            else:
                # 进一步递归切分这个超长合并块
                chunks.extend(
                    _recursive_split(merged, chunk_size, chunk_overlap, remaining_seps)
                )
            # 启用 overlap：保留尾部 overlap 字符作下一 batch 起点
            overlap_text = merged[-chunk_overlap:] if chunk_overlap > 0 and merged else ""
            current = [overlap_text] if overlap_text else []
            current_len = len(overlap_text)
        if len(piece) > chunk_size:
            # 单 piece 超长，先 flush 当前再递归切分该 piece
            if current:
                chunks.append(separator.join(current))
                current = []
                current_len = 0
            chunks.extend(
                _recursive_split(piece, chunk_size, chunk_overlap, remaining_seps)
            )
        else:
            current.append(piece)
            current_len += piece_len
    if current:
        chunks.append(separator.join(current))
    return [c for c in chunks if c]


def _char_split_with_overlap(
    text: str, chunk_size: int, chunk_overlap: int
) -> list[str]:
    """字符级滑动窗口切分，固定步长 = chunk_size - chunk_overlap。"""
    if not text:
        return []
    chunks: list[str] = []
    step = max(1, chunk_size - chunk_overlap)
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks


def recursive_chunker(
    parsed_docs: list[ParsedDocument],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    **kwargs: Any,  # noqa: ARG001
) -> list[Chunk]:
    """递归按分隔符切分 ParsedDocument 列表为 Chunk 列表。

    chunk_id = sha1(source_path + chunk_index + content[:128])[:16]
    parent_id = sha1(source_path)[:16]（同 source 共享）
    metadata 继承 ParsedDocument.metadata + chunk_index + source_path
    """
    if not parsed_docs:
        return []

    result: list[Chunk] = []
    for parsed in parsed_docs:
        source_path = str(parsed.source_path)
        parent_id = _hash_id(source_path)
        text_chunks = _recursive_split(
            parsed.content, chunk_size, chunk_overlap, SEPARATORS
        )
        for idx, content in enumerate(text_chunks):
            chunk_id = _hash_id(source_path, str(idx), content[:128])
            metadata = {
                **parsed.metadata,
                "chunk_index": idx,
                "source_path": source_path,
            }
            result.append(
                Chunk(
                    chunk_id=chunk_id,
                    parent_id=parent_id,
                    content=content,
                    metadata=metadata,
                )
            )
    return result
