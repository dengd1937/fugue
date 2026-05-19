"""src/ragline/engine/runtime.py — 从 LangGraph RunnableConfig 提取 GraphConfig。"""

from dataclasses import fields as dc_fields

from langchain_core.runnables import RunnableConfig

from ragline.config import GraphConfig

_GRAPH_CONFIG_FIELDS = {f.name for f in dc_fields(GraphConfig)}


def get_config(config: RunnableConfig) -> GraphConfig:
    """从 RunnableConfig['configurable'] 提取 GraphConfig 字段。

    自动过滤 LangGraph 框架注入的键（如 thread_id, checkpoint_ns），
    只保留 GraphConfig dataclass 字段。
    """
    raw = config.get("configurable") or {}
    filtered = {k: v for k, v in raw.items() if k in _GRAPH_CONFIG_FIELDS}
    return GraphConfig(**filtered)
