"""rag/handlers/generators.py — 答案生成 handler 实现。"""

from __future__ import annotations

from typing import Any

from rag.handlers.transforms import _default_llm
from rag.registry import generator_registry
from rag.types import Document

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPT_BASIC = "基于以下上下文回答问题。\n\n上下文：\n{context}\n\n问题：{query}"

PROMPT_COT = (
    "请按照以下步骤推理并回答问题：\n"
    "1. 仔细阅读上下文\n"
    "2. 逐步思考（step-by-step reasoning）\n"
    "3. 给出最终答案\n\n"
    "上下文：\n{context}\n\n"
    "问题：{query}\n\n"
    "请逐步推理："
)

PROMPT_CITATION = (
    "基于以下带编号的参考文档回答问题，并在回答中引用相关文档编号（如 [N]）。\n\n"
    "参考文档：\n{context}\n\n"
    "问题：{query}"
)

# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


def _build_basic_context(docs: list[Document]) -> str:
    """拼接文档内容为基础上下文。"""
    return "\n\n".join(doc["content"] for doc in docs)


def _build_citation_context(docs: list[Document]) -> str:
    """拼接带编号的文档内容。"""
    return "\n\n".join(f"[{i + 1}] {doc['content']}" for i, doc in enumerate(docs))


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


def build_prompt(gen_mode: str, query: str, docs: list[Document]) -> str:
    """根据生成模式构建 prompt 字符串。未知模式 fallback 到 basic。"""
    if gen_mode == "citation":
        context = _build_citation_context(docs)
        return PROMPT_CITATION.format(context=context, query=query)
    elif gen_mode == "cot":
        context = _build_basic_context(docs)
        return PROMPT_COT.format(context=context, query=query)
    else:
        # basic 或未知模式
        context = _build_basic_context(docs)
        return PROMPT_BASIC.format(context=context, query=query)


# ---------------------------------------------------------------------------
# Generate 函数
# ---------------------------------------------------------------------------


def basic_generate_fn(
    docs: list[Document],
    query: str,
    temperature: float,
    llm_client: Any = _default_llm,
) -> str:
    """基础生成模式。"""
    prompt = build_prompt("basic", query, docs)
    return str(llm_client.invoke(prompt, temperature=temperature))


def cot_generate_fn(
    docs: list[Document],
    query: str,
    temperature: float,
    llm_client: Any = _default_llm,
) -> str:
    """Chain-of-Thought 推理生成模式。"""
    prompt = build_prompt("cot", query, docs)
    return str(llm_client.invoke(prompt, temperature=temperature))


def citation_generate_fn(
    docs: list[Document],
    query: str,
    temperature: float,
    llm_client: Any = _default_llm,
) -> str:
    """带引用编号的生成模式。"""
    prompt = build_prompt("citation", query, docs)
    return str(llm_client.invoke(prompt, temperature=temperature))


# ---------------------------------------------------------------------------
# 注册
# ---------------------------------------------------------------------------

generator_registry.register("basic", basic_generate_fn)
generator_registry.register("cot", cot_generate_fn)
generator_registry.register("citation", citation_generate_fn)
