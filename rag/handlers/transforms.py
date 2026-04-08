"""rag/handlers/transforms.py — 查询转换 handler 实现。"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from rag.registry import transform_registry
from rag.types import TransformResult

# ---------------------------------------------------------------------------
# LLM Protocol
# ---------------------------------------------------------------------------


class LLMProtocol(Protocol):
    """LLM 客户端协议定义（鸭子类型，供类型检查用）。"""

    def invoke(self, prompt: str, temperature: float | None = None) -> str: ...


# ---------------------------------------------------------------------------
# Lazy default LLM（import 时不实例化，避免无 API key 时崩溃）
# ---------------------------------------------------------------------------


class _LazyDefaultLLM:
    """延迟构造的默认 LLM 客户端。"""

    def __init__(self) -> None:
        self._llm: Any | None = None

    def _get_llm(self) -> Any:
        if self._llm is None:
            from langchain_openai import ChatOpenAI  # 延迟 import

            self._llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        return self._llm

    def invoke(self, prompt: str, temperature: float | None = None) -> str:
        llm = self._get_llm()
        if temperature is not None:
            llm = llm.bind(temperature=temperature)
        response = llm.invoke(prompt)
        # LangChain 返回 AIMessage，取 .content
        return str(response.content) if hasattr(response, "content") else str(response)


_default_llm: _LazyDefaultLLM = _LazyDefaultLLM()

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPT_REWRITE = (
    "请将以下查询改写为 {n} 个不同的表达方式，以提升检索覆盖度。\n"
    "以编号列表返回，每行格式为 '1. 改写后查询'，不要其他内容。\n\n"
    "原始查询：{query}"
)

PROMPT_DECOMPOSE = (
    "请将以下复杂问题分解为 {n} 个独立的子问题，每个子问题可单独检索回答。\n"
    "以编号列表返回，每行格式为 '1. 子问题'，不要其他内容。\n\n"
    "原始问题：{query}"
)

PROMPT_HYDE = (
    "请为以下查询生成 {n} 个假设性文档片段，这些片段如果存在应能很好地回答该查询。\n"
    "以编号列表返回，每行格式为 '1. 假设文档内容'，不要其他内容。\n\n"
    "查询：{query}"
)

PROMPT_STEP_BACK = (
    "请将以下具体问题抽象为 {n} 个更高层次的通用问题，以便检索到更广泛的背景知识。\n"
    "以编号列表返回，每行格式为 '1. 抽象问题'，不要其他内容。\n\n"
    "具体问题：{query}"
)

PROMPT_SELF_QUERY = (
    "请分析以下查询，提取结构化过滤条件。\n"
    '以 JSON 格式返回，格式为：{{"query": "语义查询部分", "filter": {{...过滤条件...}}}}\n'
    "如没有过滤条件，filter 为 null。只返回 JSON，不要其他内容。\n\n"
    "查询：{query}"
)

# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


def _parse_numbered_lines(text: str) -> list[str]:
    """解析 LLM 返回的编号列表格式 '1. xxx\\n2. yyy'。

    - 仅解析 '数字. ' 开头的行
    - 空字符串或无匹配项返回空列表
    - 失败时 graceful 返回空列表
    """
    if not text or not text.strip():
        return []
    pattern = re.compile(r"^\d+\.\s+(.+)$")
    results: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if match:
            results.append(match.group(1).strip())
    return results


# ---------------------------------------------------------------------------
# Transform 函数
# ---------------------------------------------------------------------------


def rewrite_fn(
    queries: list[str],
    n: int,
    llm_client: Any = _default_llm,
) -> list[str]:
    """将查询改写为 n 个不同表达方式。"""
    results: list[str] = []
    for query in queries:
        prompt = PROMPT_REWRITE.format(n=n, query=query)
        response = llm_client.invoke(prompt)
        results.extend(_parse_numbered_lines(response))
    return results


def decompose_fn(
    queries: list[str],
    n: int,
    llm_client: Any = _default_llm,
) -> list[str]:
    """将查询分解为 n 个子问题。"""
    results: list[str] = []
    for query in queries:
        prompt = PROMPT_DECOMPOSE.format(n=n, query=query)
        response = llm_client.invoke(prompt)
        results.extend(_parse_numbered_lines(response))
    return results


def hyde_fn(
    queries: list[str],
    n: int,
    llm_client: Any = _default_llm,
) -> list[str]:
    """生成假设性文档（HyDE）。"""
    results: list[str] = []
    for query in queries:
        prompt = PROMPT_HYDE.format(n=n, query=query)
        response = llm_client.invoke(prompt)
        results.extend(_parse_numbered_lines(response))
    return results


def step_back_fn(
    queries: list[str],
    n: int,
    llm_client: Any = _default_llm,
) -> list[str]:
    """生成抽象的 Step-Back 问题。"""
    results: list[str] = []
    for query in queries:
        prompt = PROMPT_STEP_BACK.format(n=n, query=query)
        response = llm_client.invoke(prompt)
        results.extend(_parse_numbered_lines(response))
    return results


def self_query_fn(
    queries: list[str],
    n: int,
    llm_client: Any = _default_llm,
) -> list[TransformResult]:
    """提取结构化过滤条件（Self-Query）。"""
    results: list[TransformResult] = []
    for query in queries:
        prompt = PROMPT_SELF_QUERY.format(query=query)
        response = llm_client.invoke(prompt)
        try:
            parsed = json.loads(response)
            if not isinstance(parsed, dict) or "query" not in parsed:
                continue
            results.append(
                TransformResult(
                    query=parsed["query"],
                    metadata_filter=parsed.get("filter"),
                )
            )
        except (json.JSONDecodeError, ValueError):
            continue
    return results


# ---------------------------------------------------------------------------
# 注册
# ---------------------------------------------------------------------------

transform_registry.register("rewrite", rewrite_fn)
transform_registry.register("decompose", decompose_fn)
transform_registry.register("hyde", hyde_fn)
transform_registry.register("step_back", step_back_fn)
transform_registry.register("self_query", self_query_fn)
