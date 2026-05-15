"""src/fugue/engine/nodes/grade.py — grade 节点 + 路由函数。"""

from typing import Any

from langchain_core.runnables import RunnableConfig

from fugue.engine.runtime import get_config
from fugue.engine.state import RAGState
from fugue.registry import grader_registry


def grade(state: RAGState, config: RunnableConfig) -> dict[str, Any]:
    """调用 grader 评分当前 documents 是否足够回答 query。

    1. 调 grader_registry.get(cfg.grade_strategy)
    2. 归档当前 documents 到 retrieval_history
    3. 返回 grade_score / grade_decision / retrieval_history
    """
    cfg = get_config(config)
    grader_fn = grader_registry.get(cfg.grade_strategy)
    docs = state.get("documents", [])
    score, decision = grader_fn(
        docs,
        state["original_query"],
        cfg.grade_threshold,
        score_normalizers=cfg.score_normalizers,
    )
    history = list(state.get("retrieval_history", []))
    history.append(list(docs))
    return {
        "grade_score": score,
        "grade_decision": decision,
        "retrieval_history": history,
    }


def route_after_grade(state: RAGState, config: RunnableConfig) -> str:
    """conditional edge 路由：
    - sufficient → 'post_process'
    - insufficient 且 fallback_chain 已耗尽（retry_count >= len）→ 'post_process'
    - insufficient 且 retry_count >= max_retries → 'post_process'
    - insufficient 且未耗尽 → 'fallback_to_query_transform'
    """
    cfg = get_config(config)
    decision = state.get("grade_decision", "insufficient")
    retry_count = state.get("retry_count", 0)

    if decision == "sufficient":
        return "post_process"

    # insufficient：检查 fallback 是否可用
    if not cfg.fallback_chain:
        return "post_process"
    if retry_count >= len(cfg.fallback_chain):
        return "post_process"
    if retry_count >= cfg.max_retries:
        return "post_process"

    return "fallback_to_query_transform"
