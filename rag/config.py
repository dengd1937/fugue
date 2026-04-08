"""rag/config.py — LangGraph 图配置。"""

from dataclasses import asdict, dataclass, field
from dataclasses import fields as dc_fields
from typing import Literal


@dataclass
class GraphConfig:
    """RAG 图的运行时配置，可通过 RunnableConfig["configurable"] 注入。"""

    # query_transform
    transforms: list[str | list[str]] = field(default_factory=lambda: ["rewrite"])
    n_rewrites: int = 3
    max_queries: int = 20

    # retrieve
    retrievers: list[str] = field(default_factory=lambda: ["vector", "es"])
    route_strategy: Literal["all", "intent"] = "all"
    retriever_weights: dict[str, float] = field(default_factory=dict)

    # grade
    grade_threshold: float = 0.6
    grade_strategy: Literal["llm", "score", "hybrid"] = "score"
    score_normalizers: dict[str, float] = field(default_factory=lambda: {"es": 20.0, "web": 10.0})

    # fallback
    fallback_chain: list[str] = field(default_factory=lambda: ["web"])
    # max_retries：允许的最大 fallback 次数。
    # 传 -1（默认）时，__post_init__ 会将其规整为 len(fallback_chain)，
    # 确保默认行为是走完整条 fallback 链。
    # 注意：to_configurable() 序列化的是规整后的整数值，
    # 经 GraphConfig(**d) 还原后 -1 不复存在，但行为等价。
    max_retries: int = -1

    def __post_init__(self) -> None:
        # -1 为哨兵值：自动对齐 fallback_chain 长度，走完整条链
        if self.max_retries < 0:
            self.max_retries = len(self.fallback_chain)

    # post_process
    processors: list[str] = field(default_factory=lambda: ["rerank"])
    top_k: int = 3

    # generate
    gen_mode: Literal["basic", "cot", "citation"] = "basic"
    temperature: float = 0.7

    def to_configurable(self) -> dict[str, object]:
        """转为 RunnableConfig["configurable"] 的 dict。"""
        return asdict(self)


# 暴露字段名集合，供 get_config() 过滤 LangGraph 框架注入的键
_GRAPH_CONFIG_FIELDS: frozenset[str] = frozenset(f.name for f in dc_fields(GraphConfig))
