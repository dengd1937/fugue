"""rag/handlers/intent_router.py — 查询意图路由，决定调用哪些 retriever。"""

from __future__ import annotations

import json
from typing import Any

from rag.handlers.transforms import _default_llm

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

PROMPT_INTENT_ROUTE = (
    "根据用户查询，从以下可用检索器列表中选择最合适的检索器。\n"
    "可用检索器：{available}\n\n"
    '以 JSON 数组格式返回选中的检索器名称列表，例如 ["vector", "es"]。\n'
    "只返回 JSON 数组，不要其他内容。\n\n"
    "用户查询：{query}"
)

# ---------------------------------------------------------------------------
# intent_router
# ---------------------------------------------------------------------------


def intent_router(
    query: str,
    available_retrievers: list[str],
    llm_client: Any = _default_llm,
) -> list[str]:
    """根据查询意图选择 available_retrievers 的子集。

    - LLM 返回 JSON 数组，过滤非 available_retrievers 中的项
    - JSON 解析失败时 fallback 到全集
    - 返回值全不在 available_retrievers 时 fallback 到全集
    """
    if not available_retrievers:
        return []

    prompt = PROMPT_INTENT_ROUTE.format(
        available=json.dumps(available_retrievers, ensure_ascii=False),
        query=query,
    )
    response = llm_client.invoke(prompt)

    try:
        parsed = json.loads(response)
        if not isinstance(parsed, list):
            return list(available_retrievers)
        # 过滤：只保留字符串且在 available 中的项
        filtered = [
            item for item in parsed if isinstance(item, str) and item in available_retrievers
        ]
        if not filtered:
            return list(available_retrievers)
        return filtered
    except (json.JSONDecodeError, ValueError):
        return list(available_retrievers)
