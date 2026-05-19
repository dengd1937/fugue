"""src/ragline/engine/nodes/prepare_fallback.py — 切换到下一个 fallback source。"""

from typing import Any

from langchain_core.runnables import RunnableConfig

from ragline.engine.runtime import get_config
from ragline.engine.state import RAGState


def prepare_fallback(state: RAGState, config: RunnableConfig) -> dict[str, Any]:
    """从 cfg.fallback_chain[retry_count] 取下一个 source。retry_count++。

    不重置 documents（query_transform 节点用 Overwrite([]) 处理）。
    """
    cfg = get_config(config)
    retry_count = state.get("retry_count", 0)
    # 调用前已被 route_after_grade 保护：retry_count < len(fallback_chain)
    next_source = cfg.fallback_chain[retry_count]
    return {
        "source": next_source,
        "retry_count": retry_count + 1,
    }
