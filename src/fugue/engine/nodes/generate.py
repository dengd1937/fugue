"""src/fugue/engine/nodes/generate.py — 生成最终答案。"""

from typing import Any

from langchain_core.runnables import RunnableConfig

from fugue.engine.runtime import get_config
from fugue.engine.state import RAGState
from fugue.registry import generator_registry


def generate(state: RAGState, config: RunnableConfig) -> dict[str, Any]:
    """调 generator_registry.get(cfg.gen_mode)；返回 {answer}。

    传给 generator 的参数：query, docs=ranked_documents, temperature。
    """
    cfg = get_config(config)
    generator_fn = generator_registry.get(cfg.gen_mode)
    answer = generator_fn(
        state["original_query"],
        state.get("ranked_documents", []),
        cfg.temperature,
    )
    return {"answer": answer}
