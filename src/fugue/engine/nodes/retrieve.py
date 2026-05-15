"""src/fugue/engine/nodes/retrieve.py — 单个检索任务节点（best-effort）。"""

import logging
from typing import Any

from fugue.engine.state import RetrieveInput
from fugue.registry import retriever_registry

logger = logging.getLogger("fugue.engine.nodes.retrieve")


def retrieve(state: RetrieveInput) -> dict[str, Any]:
    """单个检索任务。best-effort：捕获异常返回空 documents 并记录 log。

    检索器返回的 Document.source 字段会被强制覆盖为 retriever_name
    （确保 RRF 等下游处理用一致的 source 标识）。
    """
    try:
        fn = retriever_registry.get(state["retriever_name"])
        docs = fn(query=state["query"], metadata_filter=state.get("metadata_filter"))
        # 强制设置 source = retriever_name
        normalized = [{**d, "source": state["retriever_name"]} for d in docs]
        return {"documents": normalized}
    except Exception as e:
        logger.error(
            "Retriever '%s' failed for query '%s': %s",
            state["retriever_name"],
            state["query"][:50],
            e,
        )
        return {"documents": []}
