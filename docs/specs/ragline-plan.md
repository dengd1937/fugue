# Ragline RAG 系统 — 详细实现方案

## 1. 设计哲学

> **图拓扑不可变，行为全靠配置驱动。**

整个 RAG 系统固定为 7 个节点的状态机，所有需求变更（增删检索器、调整重写数量、切换生成模式、修改质量阈值等）都通过修改 `GraphConfig` 完成，无需改动图结构。

---

## 2. 系统架构总览

```
__start__
    │
    ▼
query_transform        ← config.transforms (nested: parallel + pipeline) / config.n_rewrites
    │
    │  Command + Send() × (n_rewrites × len(retrievers))
    ▼
retrieve               ← config.retrievers / state.source
    │
    ▼
grade                  ← config.grade_threshold / config.grade_strategy
    │
    ├─ sufficient ──────► post_process  ← config.processors / config.top_k
    │                         │
    │                         ▼
    │                     generate      ← config.gen_mode / config.temperature
    │                         │
    │                         ▼
    │                     __end__
    │
    └─ insufficient ────► prepare_fallback ──► query_transform (source=web/kg)
       (retry ≤ max_retries)
```

共 7 个节点：`__start__` / `query_transform` / `retrieve` / `grade` / `post_process` / `generate` / `__end__`

---

## 3. GraphConfig 定义

`GraphConfig` 是整个系统的唯一变更入口。所有节点通过 `config = RunnableConfig` 读取配置。

```python
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class GraphConfig:
    """图拓扑不变，所有需求变更只改这里。"""

    # ── query_transform 阶段 ──
    # transforms 支持嵌套结构，结构即执行计划：
    #   str        → 单个原子 transform，独立并行执行
    #   list[str]  → 管道链，按顺序串联执行（前一个输出喂给后一个）
    #   顶层元素之间并行执行，结果合并
    #
    # 示例：
    #   ["rewrite"]                        → 只做 rewrite
    #   ["hyde", "rewrite"]                → hyde 和 rewrite 并行，结果合并
    #   [["step_back", "rewrite"]]         → step_back → rewrite 串联
    #   ["hyde", ["step_back", "rewrite"]] → hyde 并行 + step_back→rewrite 串联，结果合并
    #
    # ⚠️ 注意组合爆炸：管道链场景下，n_rewrites 对每级 transform 都生效。
    # 例如 [["step_back", "rewrite"]] + n_rewrites=5：
    #   step_back 生成 5 个 → rewrite 对每个再生成 5 个 = 25 个 queries
    #   最终 Send 数量 = (25 + 1) × len(retrievers)
    # max_queries 用于截断最终 query 总数，防止组合爆炸。
    transforms: list[str | list[str]] = field(default_factory=lambda: ["rewrite"])
    n_rewrites: int = 3
    max_queries: int = 20  # 截断上限，超过此数量的 queries 会被丢弃

    # ── retrieve 阶段 ──
    retrievers: list[str] = field(default_factory=lambda: ["vector", "es"])
    route_strategy: Literal["all", "intent"] = "all"
    #   all:    每个 query 打到所有 retrievers（笛卡尔积）
    #   intent: LLM/规则判断 query 意图，只路由到匹配的 retrievers
    retriever_weights: dict[str, float] = field(default_factory=dict)
    #   用于 RRF/weighted fusion，如 {"vector": 1.0, "es": 0.8, "kg": 0.5}
    #   空 dict 表示等权

    # ── grade 阶段 ──
    grade_threshold: float = 0.6
    grade_strategy: Literal["llm", "score", "hybrid"] = "score"
    # 归一化参数：不同检索来源的分数范围不同，需要归一化到 0~1。
    # key 为 retriever source 名称，value 为该来源原始分数的上界估计值。
    # 未列出的来源默认 max=1.0（假设已归一化）。
    score_normalizers: dict[str, float] = field(
        default_factory=lambda: {"es": 20.0, "web": 10.0}
    )

    # ── 回退策略 ──
    # fallback_chain 定义回退顺序，逐一尝试。
    # 例如 ["web", "kg"] 表示先回退到 web，web 也不够再回退到 kg。
    # 空列表 [] 表示不回退。
    fallback_chain: list[str] = field(default_factory=lambda: ["web"])
    max_retries: int = 1  # 每个 fallback 源的最大重试次数

    # ── post_process 阶段 ──
    processors: list[str] = field(default_factory=lambda: ["rerank"])
    top_k: int = 3

    # ── generate 阶段 ──
    gen_mode: Literal["basic", "cot", "citation"] = "basic"
    temperature: float = 0.7
```

### 需求变更示例对照表

| 领导的需求 | 改动 |
|---|---|
| 增加 query 重写到 5 个 | `n_rewrites = 5` |
| 多路检索加 Knowledge Graph | `retrievers = ["vector", "es", "kg"]` |
| rerank 后返回 5 条 | `top_k = 5` |
| KB 搜不到走 Web | `fallback_chain = ["web"]` (默认已支持) |
| 质量分数低于 0.7 才回退 | `grade_threshold = 0.7` |
| 换成 CoT 生成 | `gen_mode = "cot"` |
| 加 HyDE 改写策略（与 rewrite 并行） | `transforms = ["rewrite", "hyde"]` |
| 先 step_back 再 rewrite（链式） | `transforms = [["step_back", "rewrite"]]` |
| hyde 并行 + step_back→rewrite 链式 | `transforms = ["hyde", ["step_back", "rewrite"]]` |
| 关闭回退 | `fallback_chain = []` |
| 按意图路由（SQL 问题只走 SQL） | `route_strategy = "intent"` |
| 多路融合排序（RRF） | `processors = ["rrf", "rerank"]` |
| 向量权重高于 ES | `retriever_weights = {"vector": 1.0, "es": 0.6}` |
| 用 parent-child 分块检索 | `retrievers = ["vector_parent_child", "es"]` |
| 检索后压缩上下文 | `processors = ["rrf", "rerank", "compression"]` |
| 提取 metadata 过滤条件 | `transforms = ["self_query", "rewrite"]` |

---

## 4. State 定义

### 4.1 Document 类型定义

所有检索器必须返回符合 `Document` 结构的字典，确保全链路类型安全。

```python
from typing import Annotated, Any, Literal
from typing_extensions import TypedDict


class Document(TypedDict):
    """检索文档的统一类型。所有 retriever 必须返回此结构。"""

    doc_id: str
    content: str
    score: float
    source: str  # retriever 名称，如 "vector" / "es" / "kg" / "web"
    metadata: dict[str, Any]
```

### 4.2 State 与 Reducer

```python
from langgraph.graph.message import add_messages


def merge_docs(existing: list[Document], new: list[Document]) -> list[Document]:
    """自定义 reducer：按 (source, doc_id) 复合键去重合并文档列表。"""
    seen = {(d["source"], d["doc_id"]) for d in existing}
    return existing + [d for d in new if (d["source"], d["doc_id"]) not in seen]


class RAGState(TypedDict):
    """图的全局状态，在节点之间流转。"""

    # 用户原始问题
    original_query: str

    # query_transform 产出的改写列表（仅用于调试/可观测性）
    rewritten_queries: list[str]

    # 当前轮次 retrieve 产出的文档（通过 merge_docs reducer 自动去重合并）
    documents: Annotated[list[Document], merge_docs]

    # grade 产出的评分 & 路由信号
    grade_score: float
    grade_decision: Literal["sufficient", "insufficient"]

    # 当前检索来源，grade 回退时会改写此字段
    source: Literal["kb", "web", "kg"]

    # 回退计数器
    retry_count: int

    # 历史轮次的检索结果（每轮 grade 后归档，避免 reducer 污染）
    retrieval_history: list[list[Document]]

    # post_process 产出的精排文档
    ranked_documents: list[Document]

    # generate 产出的最终回答
    answer: str
```

### 关于 `Annotated[list[Document], merge_docs]`

LangGraph 的 reducer 机制允许多个并行 `Send()` 分支返回的文档自动合并。每个 `retrieve` 实例返回自己的 `documents`，reducer 负责按 `(source, doc_id)` 复合键去重拼接。

**关于 fallback 轮次的状态管理**：`grade` 节点在判断 insufficient 后，由 `query_transform` 通过 `Overwrite([])` 显式重置 `documents`（而非传入空列表），以绕过 reducer 的累加语义。同时 `grade` 会将当前文档归档到 `retrieval_history`，确保每轮检索结果互不污染。

---

## 5. Registry 模式 — 插件注册表

Registry 是实现"加一个检索器不改图结构"的关键。每个阶段维护一个注册表，节点运行时遍历 config 中激活的 handler。

```python
from typing import Callable, Any


class Registry:
    """通用插件注册表。"""

    def __init__(self):
        self._handlers: dict[str, Callable] = {}

    def register(self, name: str, fn: Callable) -> None:
        self._handlers[name] = fn

    def get(self, name: str) -> Callable:
        if name not in self._handlers:
            raise KeyError(
                f"Handler '{name}' not registered. "
                f"Available: {list(self._handlers.keys())}"
            )
        return self._handlers[name]

    def run_pipeline(self, names: list[str], **kwargs) -> Any:
        """链式执行：前一个的输出作为后一个的输入。"""
        result = kwargs
        for name in names:
            result = self.get(name)(**result) if isinstance(result, dict) else self.get(name)(result)
        return result


# ── 全局注册表实例 ──
transform_registry = Registry()
retriever_registry = Registry()
processor_registry = Registry()
grader_registry = Registry()
generator_registry = Registry()
```

### 注册具体实现

```python
# ── Query Transforms ──
# 统一签名：接收 queries: list[str]，返回 list[str]
# parallel 模式下每个 transform 只收到 [original_query]
# pipeline 模式下每个 transform 收到前一个 transform 的输出
#
# self_query 特殊处理：返回 TransformResult 包含 metadata_filter，
# 通过 Send payload 的 metadata_filter 字段显式传递给 retrieve 节点，
# 不再通过 "query||filter_json" 字符串编码。

def rewrite_fn(queries: list[str], n: int) -> list[str]:
    """LLM 改写：将每个 query 改写成 n 个语义相近但表述不同的版本。"""
    results = []
    for q in queries:
        prompt = f"请将以下问题改写成 {n} 个不同的表述：\n{q}"
        # response = llm.invoke(prompt)
        # results.extend(parse_queries(response))
    return results

def decompose_fn(queries: list[str], n: int) -> list[str]:
    """子问题分解：将每个复杂问题拆成可独立检索的子问题。"""
    results = []
    for q in queries:
        prompt = f"请将以下问题拆解为 {n} 个可独立回答的子问题：\n{q}"
        # results.extend(parse_queries(llm.invoke(prompt)))
    return results

def hyde_fn(queries: list[str], n: int) -> list[str]:
    """HyDE：对每个 query 生成假设性回答，用回答作为新的检索 query。"""
    results = []
    for q in queries:
        prompt = f"请针对以下问题写 {n} 段可能的回答（用于检索）：\n{q}"
        # results.extend(parse_queries(llm.invoke(prompt)))
    return results

def step_back_fn(queries: list[str], n: int) -> list[str]:
    """Step-Back：对每个 query 生成更抽象的上位问题。"""
    results = []
    for q in queries:
        prompt = f"请将以下问题抽象为 {n} 个更宏观的上位问题：\n{q}"
        # results.extend(parse_queries(llm.invoke(prompt)))
    return results

transform_registry.register("rewrite", rewrite_fn)
transform_registry.register("decompose", decompose_fn)
transform_registry.register("hyde", hyde_fn)
transform_registry.register("step_back", step_back_fn)


# ── Self-Query Transform ──
# 从自然语言 query 中提取结构化 metadata 过滤条件。
# 返回 TransformResult 列表，每个结果包含 query 和 metadata_filter。
# query_transform 节点会将 metadata_filter 通过 Send payload 传递给 retrieve。

@dataclass
class TransformResult:
    """transform 的结构化输出，支持携带 metadata_filter。"""
    query: str
    metadata_filter: dict[str, Any] | None = None


def self_query_fn(queries: list[str], n: int) -> list[TransformResult]:
    """
    Self-Query: 提取 metadata 过滤条件。
    输入: ["2024年Q3的销售报告"]
    输出: [TransformResult(query="销售报告", metadata_filter={"year": 2024, "quarter": "Q3"})]
    """
    results = []
    for q in queries:
        prompt = (
            f"从以下问题中提取搜索关键词和 metadata 过滤条件。\n"
            f"问题：{q}\n"
            f'返回 JSON：{{"query": "...", "filter": {{...}}}}'
        )
        # parsed = json.loads(llm.invoke(prompt))
        # results.append(TransformResult(
        #     query=parsed["query"],
        #     metadata_filter=parsed.get("filter"),
        # ))
    return results

transform_registry.register("self_query", self_query_fn)


# ── Intent Router ──
# 不是 transform（不改写 query），而是给 query 标注应该走哪些 retriever。
# 作为独立 registry 注册，供 query_transform 在 route_strategy="intent" 时调用。

def intent_router(query: str, available_retrievers: list[str]) -> list[str]:
    """
    判断 query 意图，返回该 query 应该路由到的 retriever 列表。
    示例：
        "上个月销售额是多少" → ["sql"]
        "LangGraph 怎么用"  → ["vector", "es"]
        "今天天气怎么样"     → ["web"]
    """
    prompt = (
        f"判断以下问题应该使用哪些检索方式。\n"
        f"可选：{available_retrievers}\n"
        f"问题：{query}\n"
        f"返回 JSON 数组，如 [\"vector\", \"es\"]"
    )
    # targets = json.loads(llm.invoke(prompt))
    # return [t for t in targets if t in available_retrievers]
    ...


# ── Retrievers ──
# 统一签名：(query: str, metadata_filter: dict | None) -> list[Document]
# 所有 retriever 必须接受 metadata_filter 参数（不支持的忽略即可）。

def vector_search_fn(query: str, metadata_filter: dict | None = None) -> list[Document]:
    """向量相似度检索。"""
    # return vector_store.similarity_search(query, k=20, filter=metadata_filter)
    ...

def es_search_fn(query: str, metadata_filter: dict | None = None) -> list[Document]:
    """Elasticsearch BM25 全文检索。"""
    ...

def kg_search_fn(query: str, metadata_filter: dict | None = None) -> list[Document]:
    """Knowledge Graph 图谱检索。"""
    ...

def web_search_fn(query: str, metadata_filter: dict | None = None) -> list[Document]:
    """Web 搜索回退。"""
    ...

retriever_registry.register("vector", vector_search_fn)
retriever_registry.register("es", es_search_fn)
retriever_registry.register("kg", kg_search_fn)
retriever_registry.register("web", web_search_fn)


# ── Parent-Child Chunk Retriever ──
# 检索小分块（高精度），然后扩展到父分块（完整上下文）。
# 对外接口与普通 retriever 完全一致，内部自动完成 child → parent 扩展。

def vector_parent_child_fn(query: str, metadata_filter: dict | None = None) -> list[Document]:
    """
    Small-to-Big：先检索小分块，再按 parent_id 扩展到父分块。
    需要 vector_store 存储时就建好 child → parent 的映射关系。
    """
    # 1. 检索小分块
    child_chunks = []
    # child_chunks = vector_store.similarity_search(query, k=20)

    # 2. 按 parent_id 聚合，取每组最高分作为 parent score
    parent_scores = {}
    for c in child_chunks:
        pid = c["metadata"]["parent_id"]
        parent_scores[pid] = max(parent_scores.get(pid, 0), c["score"])

    # 3. 从 docstore 取父分块
    # parents = doc_store.mget(list(parent_scores.keys()))
    parents = []
    return [
        {
            "doc_id": p["id"],
            "content": p["content"],
            "score": parent_scores[p["id"]],
            "source": "vector_parent_child",
            "metadata": p.get("metadata", {}),
        }
        for p in parents
    ]


def vector_sentence_window_fn(query: str, metadata_filter: dict | None = None) -> list[Document]:
    """
    Sentence Window：检索单句级分块，返回时自动扩展到前后 N 句窗口。
    需要分块时存储 window_start / window_end 元数据。
    """
    # 1. 检索句子级分块
    # sentence_chunks = vector_store.similarity_search(query, k=20)

    # 2. 按窗口元数据扩展上下文
    # expanded = [expand_window(c, window_size=3) for c in sentence_chunks]
    ...

retriever_registry.register("vector_parent_child", vector_parent_child_fn)
retriever_registry.register("vector_sentence_window", vector_sentence_window_fn)


# ── Post-Processors ──

def rerank_fn(docs: list[Document], query: str, top_k: int, **kwargs) -> list[Document]:
    """交叉编码器精排。"""
    # scores = cross_encoder.predict([(query, d["content"]) for d in docs])
    # return sorted(zip(docs, scores), key=lambda x: -x[1])[:top_k]
    ...

def filter_fn(docs: list[Document], **kwargs) -> list[Document]:
    """过滤低质量 / 过期文档。"""
    ...

def dedupe_fn(docs: list[Document], **kwargs) -> list[Document]:
    """基于内容指纹去重。"""
    ...

processor_registry.register("rerank", rerank_fn)
processor_registry.register("filter", filter_fn)
processor_registry.register("dedupe", dedupe_fn)


# ── Reciprocal Rank Fusion (RRF) ──
# 多路召回的核心融合算法：按来源分组排序，用 RRF 公式合并排名。
# 支持 retriever_weights 加权。

def rrf_fn(docs: list[Document], query: str, top_k: int, **kwargs) -> list[Document]:
    """
    Reciprocal Rank Fusion。
    公式：RRF_score(d) = Σ (w_source / (k + rank_in_source))
    k=60 是标准常数，rank 从 1 开始。

    使用 (source, doc_id) 复合键标识文档（与 merge_docs reducer 一致），
    避免依赖 Python 对象 id 导致浅拷贝后分数丢失。
    """
    from collections import defaultdict

    config_weights = kwargs.get("retriever_weights", {})
    k = 60

    # 按来源分组，组内按原始 score 降序排列
    groups: dict[str, list[Document]] = defaultdict(list)
    for d in docs:
        groups[d["source"]].append(d)
    for source in groups:
        groups[source].sort(key=lambda x: -x.get("score", 0))

    # 计算 RRF score，使用 (source, doc_id) 作为稳定键
    rrf_scores: dict[tuple[str, str], float] = {}
    for source, group in groups.items():
        w = config_weights.get(source, 1.0)
        for rank, doc in enumerate(group, start=1):
            doc_key = (doc["source"], doc["doc_id"])
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0) + w / (k + rank)

    # 写回 rrf_score 并排序
    for d in docs:
        d["rrf_score"] = rrf_scores.get((d["source"], d["doc_id"]), 0)
    return sorted(docs, key=lambda x: -x["rrf_score"])


def weighted_fusion_fn(docs: list[Document], query: str, top_k: int, **kwargs) -> list[Document]:
    """
    加权分数融合：normalized_score × retriever_weight。
    依赖 _normalize_score 将不同来源分数归一化到 0~1。
    """
    config_weights = kwargs.get("retriever_weights", {})
    score_normalizers = kwargs.get("score_normalizers", {})
    for d in docs:
        norm = _normalize_score(d, score_normalizers)
        w = config_weights.get(d["source"], 1.0)
        d["weighted_score"] = norm * w
    return sorted(docs, key=lambda x: -x["weighted_score"])


processor_registry.register("rrf", rrf_fn)
processor_registry.register("weighted_fusion", weighted_fusion_fn)


# ── Contextual Compression ──
# 检索后用 LLM 压缩文档，只保留与 query 相关的段落/句子。

def compression_fn(docs: list[Document], query: str, **kwargs) -> list[Document]:
    """
    LLM 文档压缩：提取每篇文档中与 query 直接相关的句子。
    大幅减少 context 长度，提升 generate 质量。
    """
    compressed = []
    for d in docs:
        prompt = (
            f"从以下文档中提取与问题直接相关的内容，去掉无关部分。\n"
            f"问题：{query}\n"
            f"文档：{d['content']}\n"
            f"只返回相关内容，不要添加解释。"
        )
        # relevant = llm.invoke(prompt)
        # compressed.append({**d, "content": relevant, "original_content": d["content"]})
    return compressed

processor_registry.register("compression", compression_fn)


# ── Graders ──

# 不同检索器的分数不在同一量纲（vector: 0~1, BM25: 0~∞, KG: 离散），
# 需要先归一化再评估。归一化参数从 config.score_normalizers 读取，避免硬编码。

def _normalize_score(doc: Document, score_normalizers: dict[str, float]) -> float:
    """
    按检索来源归一化分数到 0~1 区间。
    score_normalizers: 来自 config.score_normalizers，key 为 source，value 为分数上界。
    未注册的 source 默认 max=1.0（假设分数已归一化）。
    """
    raw = doc.get("score", 0)
    source = doc.get("source", "")
    max_score = score_normalizers.get(source, 1.0)
    if max_score <= 0:
        return 0.0
    return min(raw / max_score, 1.0)


def score_grader(
    docs: list[Document], threshold: float, **kwargs
) -> tuple[float, str]:
    """基于归一化后的平均分数判断。"""
    if not docs:
        return 0.0, "insufficient"
    normalizers = kwargs.get("score_normalizers", {})
    avg = sum(_normalize_score(d, normalizers) for d in docs) / len(docs)
    return avg, "sufficient" if avg >= threshold else "insufficient"

def llm_grader(
    docs: list[Document], query: str, threshold: float, **kwargs
) -> tuple[float, str]:
    """LLM 判断文档与 query 的相关性（不依赖检索分数）。"""
    ...

grader_registry.register("score", score_grader)
grader_registry.register("llm", llm_grader)
```

---

## 6. 节点实现

### 统一 config 注入

所有节点通过 LangGraph 的 `RunnableConfig` 获取配置，不在节点签名中直接接收 `GraphConfig`。`build_rag_graph()` 不接业务参数。

```python
from dataclasses import fields as dc_fields
from langchain_core.runnables import RunnableConfig

# GraphConfig 允许的字段名集合（启动时计算一次）
_GRAPH_CONFIG_FIELDS = {f.name for f in dc_fields(GraphConfig)}


def get_config(config: RunnableConfig) -> GraphConfig:
    """
    从 RunnableConfig 中提取业务配置。
    自动过滤 LangGraph 框架注入的键（如 thread_id、checkpoint_ns），
    只保留 GraphConfig 定义的字段，避免 TypeError。
    """
    raw = config.get("configurable", {})
    filtered = {k: v for k, v in raw.items() if k in _GRAPH_CONFIG_FIELDS}
    return GraphConfig(**filtered)
```

### 6.1 query_transform

```python
from langgraph.types import Command, Overwrite, Send


def _run_transform_branch(
    branch: str | list[str],
    queries: list[str],
    n: int,
) -> list[str | TransformResult]:
    """
    执行单个 transform 分支。

    - str       → 原子 transform，直接调用
    - list[str] → 管道链，按顺序串联（前一个输出 → 后一个输入）

    返回值可以是 str（普通 query）或 TransformResult（携带 metadata_filter）。
    """
    if isinstance(branch, str):
        fn = transform_registry.get(branch)
        return fn(queries=queries, n=n)

    # pipeline: 链式执行
    current = queries
    for name in branch:
        fn = transform_registry.get(name)
        result = fn(queries=current, n=n)
        # 如果返回 TransformResult，提取 query 继续传递给下一级
        if result and isinstance(result[0], TransformResult):
            current = [r.query for r in result]
        else:
            current = result
    # 最后一级的完整结果（可能是 TransformResult）
    return result


def query_transform(
    state: RAGState, config: RunnableConfig
) -> Command[Literal["retrieve"]]:
    """
    遍历 config.transforms 中的顶层元素，每个元素是一个独立并行的分支。
    分支内部可以是单个 transform（str）或管道链（list[str]）。
    所有分支的输出合并去重后，与 retrievers 做笛卡尔积发射 Send()。

    使用 Overwrite([]) 重置 documents 字段，绕过 merge_docs reducer 的累加语义，
    确保 fallback 轮次的文档不会与上一轮混淆。
    使用 max_queries 截断最终 query 数量，防止管道链场景下的组合爆炸。
    """
    cfg = get_config(config)
    query = state["original_query"]
    all_queries: list[str] = [query]  # 原始 query 始终保留
    # metadata_filter 映射：query → filter（来自 self_query 等 transform）
    query_filters: dict[str, dict | None] = {}

    for branch in cfg.transforms:
        results = _run_transform_branch(branch, [query], cfg.n_rewrites)
        for r in results:
            if isinstance(r, TransformResult):
                all_queries.append(r.query)
                if r.metadata_filter:
                    query_filters[r.query] = r.metadata_filter
            else:
                all_queries.append(r)

    # 去重（保持顺序）+ 截断防爆炸
    all_queries = list(dict.fromkeys(all_queries))[:cfg.max_queries]

    # 确定检索来源：首轮用 config.retrievers，fallback 轮用 fallback 源
    source = state.get("source", "kb")
    if source != "kb":
        retriever_names = [source]  # fallback 时只用指定的回退检索器
    else:
        retriever_names = cfg.retrievers

    # 构建 Send() 列表
    sends = []
    for q in all_queries:
        if cfg.route_strategy == "intent" and source == "kb":
            targets = intent_router(q, retriever_names)
        else:
            targets = retriever_names

        for retriever_name in targets:
            sends.append(
                Send(
                    "retrieve",
                    {
                        "query": q,
                        "retriever_name": retriever_name,
                        "source": source,
                        "metadata_filter": query_filters.get(q),
                    },
                )
            )

    # 通过 Command 同时更新 state 和发射 Send()
    # ⚠️ 关键：使用 Overwrite([]) 而非普通 []，绕过 merge_docs reducer
    return Command(
        update={"rewritten_queries": all_queries, "documents": Overwrite([])},
        goto=sends,
    )
```

### 6.2 retrieve

```python
class RetrieveInput(TypedDict):
    """Send() 发送给 retrieve 节点的负载类型。"""

    query: str
    retriever_name: str
    source: str
    metadata_filter: dict | None


def retrieve(state: RetrieveInput) -> dict:
    """
    单个检索任务。由 Send() 调度，每次只执行一个 (query, retriever) 组合。
    多个实例并行执行，结果通过 State reducer 自动合并。

    注意：此节点只接收 Send() 的负载（RetrieveInput），不是完整的 RAGState。
    metadata_filter 由 query_transform 通过 Send payload 显式传递，
    不再依赖 inspect.signature 探测或字符串内编码。
    """
    query = state["query"]
    retriever_name = state["retriever_name"]
    metadata_filter = state.get("metadata_filter")

    fn = retriever_registry.get(retriever_name)
    docs = fn(query=query, metadata_filter=metadata_filter)

    # 标记来源，用于 (source, doc_id) 复合去重
    for d in docs:
        d["source"] = retriever_name

    return {"documents": docs}
```

### 6.3 grade

```python
def grade(state: RAGState, config: RunnableConfig) -> dict:
    """
    评估检索结果质量，输出路由信号。
    grade_strategy 决定用哪种评分方式，grade_threshold 决定及格线。
    同时将当前轮次的文档归档到 retrieval_history，为可能的 fallback 做准备。
    """
    cfg = get_config(config)
    docs = state["documents"]
    grader_fn = grader_registry.get(cfg.grade_strategy)

    score, decision = grader_fn(
        docs=docs,
        query=state["original_query"],
        threshold=cfg.grade_threshold,
        score_normalizers=cfg.score_normalizers,
    )

    # 归档当前轮次的文档到 history
    history = state.get("retrieval_history", [])
    history = history + [docs]

    return {
        "grade_score": score,
        "grade_decision": decision,
        "retrieval_history": history,
    }
```

### 6.4 grade 条件边路由

```python
def route_after_grade(state: RAGState, config: RunnableConfig) -> str:
    """
    conditional edge 路由函数。
    - sufficient → post_process
    - insufficient 且 fallback_chain 未耗尽 → prepare_fallback → query_transform
    - insufficient 且 fallback_chain 已耗尽或 retry >= max → post_process（兜底）
    """
    cfg = get_config(config)

    if state["grade_decision"] == "sufficient":
        return "post_process"

    if not cfg.fallback_chain:
        return "post_process"

    retry_count = state.get("retry_count", 0)

    # 检查 fallback_chain 是否还有未尝试的源
    if retry_count >= len(cfg.fallback_chain):
        return "post_process"

    if retry_count >= cfg.max_retries:
        return "post_process"

    return "fallback_to_query_transform"


def prepare_fallback(state: RAGState, config: RunnableConfig) -> dict:
    """
    回退前的状态改写：从 fallback_chain 中取下一个 source，递增 retry_count。
    documents 会在 query_transform 通过 Overwrite([]) 显式清空，绕过 reducer 累加。
    """
    cfg = get_config(config)
    retry_count = state.get("retry_count", 0)
    # 按 retry_count 索引 fallback_chain，依次尝试不同的 fallback 源
    next_source = cfg.fallback_chain[retry_count]
    return {
        "source": next_source,
        "retry_count": retry_count + 1,
    }
```

### 6.5 post_process

```python
def post_process(state: RAGState, config: RunnableConfig) -> dict:
    """
    链式执行 config.processors 中激活的后处理器。
    顺序固定为：merge（内置） → config 中的处理器按声明顺序执行。

    如果经历了 fallback，会合并所有轮次的文档一起处理。
    """
    cfg = get_config(config)

    # 合并所有轮次的文档（当前轮 + 历史轮）
    docs = list(state["documents"])
    for round_docs in state.get("retrieval_history", [])[:-1]:  # 最后一轮已在 documents 里
        docs.extend(round_docs)

    # 按 (source, doc_id) 去重
    seen = set()
    unique_docs = []
    for d in docs:
        key = (d["source"], d["doc_id"])
        if key not in seen:
            seen.add(key)
            unique_docs.append(d)
    docs = unique_docs

    for proc_name in cfg.processors:
        fn = processor_registry.get(proc_name)
        docs = fn(
            docs=docs,
            query=state["original_query"],
            top_k=cfg.top_k,
            retriever_weights=cfg.retriever_weights,  # 传递给 rrf / weighted_fusion
            score_normalizers=cfg.score_normalizers,   # 传递给 weighted_fusion
        )

    # 最终截断到 top_k
    docs = docs[: cfg.top_k]

    return {"ranked_documents": docs}
```

### 6.6 generate

```python
def generate(state: RAGState, config: RunnableConfig) -> dict:
    """
    基于精排文档 + 原始问题生成最终回答。
    gen_mode 控制 prompt 模板，temperature 控制创造性。
    """
    cfg = get_config(config)
    docs = state["ranked_documents"]
    query = state["original_query"]

    # 普通模式：直接拼接
    context_plain = "\n\n".join(d["content"] for d in docs)
    # citation 模式：每段标注编号，确保 LLM 能正确引用
    context_cited = "\n\n".join(
        f"[{i + 1}] {d['content']}" for i, d in enumerate(docs)
    )

    prompts = {
        "basic": (
            f"基于以下上下文回答问题。\n\n"
            f"上下文：\n{context_plain}\n\n"
            f"问题：{query}"
        ),
        "cot": (
            f"基于以下上下文，请一步步推理后回答问题。\n\n"
            f"上下文：\n{context_plain}\n\n"
            f"问题：{query}\n\n"
            f"请先展示推理过程，再给出最终回答。"
        ),
        "citation": (
            f"基于以下上下文回答问题，每个论点必须标注来源编号 [1][2]...。\n"
            f"来源编号已在每段文档前标注。\n\n"
            f"上下文：\n{context_cited}\n\n"
            f"问题：{query}"
        ),
    }

    prompt = prompts[cfg.gen_mode]
    # answer = llm.invoke(prompt, temperature=cfg.temperature)

    return {"answer": "..."}  # answer
```

---

## 7. 图组装

```python
from langgraph.graph import StateGraph, END


def build_rag_graph() -> StateGraph:
    """
    构建 RAG 图。业务配置不在此传入，而是运行时通过 RunnableConfig 注入。
    图拓扑永远固定，所有行为差异由 config 驱动。
    """

    graph = StateGraph(RAGState)

    # ── 添加节点 ──
    graph.add_node("query_transform", query_transform)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("prepare_fallback", prepare_fallback)
    graph.add_node("post_process", post_process)
    graph.add_node("generate", generate)

    # ── 固定边 ──
    graph.set_entry_point("query_transform")
    # query_transform 通过 Command + Send() 动态发射到 retrieve，不需要显式 add_edge
    graph.add_edge("retrieve", "grade")

    # ── 条件边：grade 路由 ──
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "post_process": "post_process",
            "fallback_to_query_transform": "prepare_fallback",
        },
    )
    # fallback 路径：prepare_fallback → query_transform（重新 Send() 调度）
    graph.add_edge("prepare_fallback", "query_transform")

    # ── 后续固定边 ──
    graph.add_edge("post_process", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
```

### 调用方式

```python
# 构建图（只需一次）
graph = build_rag_graph()

# ── 今天的需求：1 路重写 + 向量 + ES ──
config_today = GraphConfig(
    transforms=["rewrite"],
    n_rewrites=1,
    retrievers=["vector", "es"],
    top_k=1,
)

# ── 明天的需求：3 路重写 + rerank top 3 ──
config_tomorrow = GraphConfig(
    transforms=["rewrite"],
    n_rewrites=3,
    retrievers=["vector", "es"],
    top_k=3,
)

# ── 后天的需求：5 路重写 + rerank top 5 ──
config_next = GraphConfig(
    transforms=["rewrite"],
    n_rewrites=5,
    retrievers=["vector", "es"],
    top_k=5,
)

# ── KB 搜不到走 Web，阈值 0.6 ──
config_fallback = GraphConfig(
    transforms=["rewrite"],
    n_rewrites=3,
    retrievers=["vector", "es"],
    grade_threshold=0.6,
    fallback_chain=["web"],
    max_retries=1,
    top_k=3,
)

# ── 链式改写：先 step_back 抽象，再 rewrite 展开 ──
config_pipeline = GraphConfig(
    transforms=[["step_back", "rewrite"]],  # 嵌套 = 串联
    n_rewrites=3,
    retrievers=["vector", "es"],
    grade_threshold=0.6,
    fallback_chain=["web"],
    top_k=5,
)

# ── 混合模式：hyde 并行 + step_back→rewrite 串联 ──
config_hybrid = GraphConfig(
    transforms=["hyde", ["step_back", "rewrite"]],  # 顶层并行，嵌套串联
    n_rewrites=3,
    retrievers=["vector", "es"],
    grade_threshold=0.6,
    fallback_chain=["web"],
    top_k=5,
)
# 执行流程：
#   branch A: hyde(["原始query"]) → ["假设性回答1", ...]
#   branch B: step_back(["原始query"]) → ["抽象问题"] → rewrite(["抽象问题"]) → ["改写1", ...]
#   合并 A + B → Send()

# 同一个 graph，不同 config
# GraphConfig 通过 configurable 字段传入，节点内部用 get_config() 解析
from dataclasses import asdict

result = graph.invoke(
    {"original_query": "LangGraph 如何实现动态路由？", "retry_count": 0, "source": "kb"},
    config={"configurable": asdict(config_fallback)},
)
```

---

## 8. 扩展指南

### 8.1 新增一个检索器（例如 SQL）

只需两步，不改图结构：

```python
# Step 1: 实现 handler
def sql_search_fn(query: str, metadata_filter: dict | None = None) -> list[Document]:
    """将自然语言转 SQL，查库返回结果。"""
    sql = text2sql(query)
    rows = db.execute(sql)
    return [
        Document(doc_id=r.id, content=str(r), score=1.0, source="sql", metadata={})
        for r in rows
    ]

# Step 2: 注册
retriever_registry.register("sql", sql_search_fn)
```

然后在 config 中激活：

```python
config = GraphConfig(retrievers=["vector", "es", "sql"])
```

### 8.2 新增一个 query 改写策略（例如 step_back）

```python
def step_back_fn(queries: list[str], n: int) -> list[str]:
    """Step-Back Prompting：对每个 query 生成更抽象的上位问题。"""
    results = []
    for q in queries:
        prompt = f"请将以下问题抽象为 {n} 个更宏观的上位问题：\n{q}"
        # results.extend(parse_queries(llm.invoke(prompt)))
    return results

transform_registry.register("step_back", step_back_fn)
```

三种用法，只改 config 一行：

```python
# 1) 与 rewrite 并行，各自独立
config = GraphConfig(transforms=["step_back", "rewrite"])

# 2) step_back → rewrite 串联
config = GraphConfig(transforms=[["step_back", "rewrite"]])

# 3) hyde 并行 + step_back→rewrite 串联，结果合并
config = GraphConfig(transforms=["hyde", ["step_back", "rewrite"]])
```

### 8.3 新增一个后处理器（例如 compression）

```python
def compression_fn(docs: list[Document], query: str, **kwargs) -> list[Document]:
    """LLM 文档压缩：只保留与 query 相关的句子。"""
    ...

processor_registry.register("compression", compression_fn)

# 激活（按声明顺序链式执行）
config = GraphConfig(processors=["rerank", "compression", "dedupe"])
```

### 8.4 切换 grade 策略

```python
def hybrid_grader(docs, query, threshold):
    """混合评分：score + LLM 双重校验。"""
    score_result = score_grader(docs, threshold)
    if score_result[1] == "sufficient":
        return llm_grader(docs, query, threshold)
    return score_result

grader_registry.register("hybrid", hybrid_grader)

config = GraphConfig(grade_strategy="hybrid", grade_threshold=0.7)
```

---

## 9. 关键设计决策 FAQ

**Q: 为什么 `retrieve` 用 `Send()` 而不是普通 `add_edge`？**

`Send()` 是 LangGraph 的动态扇出机制。它允许在运行时根据 state 决定并行发射多少个实例，每个实例有独立的输入参数。这样 `query_transform` 可以根据 `n_rewrites × len(retrievers)` 动态生成任意数量的并行检索任务，而图的结构不需要预先知道这个数量。

**Q: `Send()` 扇出后，`grade` 怎么拿到合并结果？不需要 collect 节点吗？**

不需要。LangGraph 保证：当一个节点通过 `Send()` 发射 N 个并行分支时，下游的 `add_edge("retrieve", "grade")` 会在 **所有 N 个分支执行完毕、reducer 合并完成后** 才触发 `grade`。这是框架内置的 fan-in 机制，不需要手动加 collect/join 节点。

**Q: 为什么不把每种检索器做成独立节点？**

如果向量检索和 ES 检索是独立节点，每加一种检索方式就要加一个节点 + 修改图结构。通过 Registry 模式，`retrieve` 是一个通用的 stage runner，新检索器只需 `register()` + config 激活。

**Q: 回退环路会不会死循环？**

双重保护：`max_retries` 字段 + `fallback_chain` 长度。`route_after_grade` 同时检查 `retry_count >= max_retries` 和 `retry_count >= len(fallback_chain)`，超过任一上限后强制走 `post_process`，用现有文档兜底生成回答。回退路径是 `grade → prepare_fallback → query_transform → Send(retrieve) → grade`，形成完整闭环。`prepare_fallback` 从 `fallback_chain` 按顺序取下一个回退源并递增计数器，`query_transform` 通过 `Overwrite([])` 显式重置 `documents`（绕过 `merge_docs` reducer 的累加语义）后重新 `Send()` 调度。

**Q: `post_process` 内多个处理器的执行顺序重要吗？**

重要。它们按 `config.processors` 列表的声明顺序链式执行。多路召回场景下推荐顺序：`rrf → compression → rerank`（先融合排序，再压缩上下文，最后精排）。纯向量检索场景可以简化为 `rerank`。

**Q: `transforms` 的嵌套结构怎么理解？**

把 `transforms` 想象成一个并行执行器：顶层的每个元素是一个独立的分支，所有分支并行执行，结果合并。如果某个元素是一个列表（如 `["step_back", "rewrite"]`），则该分支内部按声明顺序串联执行。这意味着你可以在同一个 config 里自由混合并行和串联：`["hyde", ["step_back", "rewrite"], "decompose"]` 表示 hyde、decompose 各自独立跑，同时 step_back→rewrite 作为一条管道跑，三路结果最后合并。

**Q: 并行检索的结果如何合并？**

通过 LangGraph 的 State reducer 机制。`documents` 字段使用自定义的 `merge_docs` reducer，多个并行 `retrieve` 实例写入的 `documents` 会自动按 `(source, doc_id)` 复合键去重合并。不同检索器返回同一篇文档时（如向量和 ES 都命中同一个 doc_id），各自保留，因为它们可能携带不同的分数和元数据。

---

## 10. 多路召回场景速查

以下是 RAG 领域常见的多路召回模式，以及对应的 config 配置方式。图拓扑始终不变，所有场景只改 `GraphConfig`。

### 10.1 经典 Vector + BM25 混合检索（RRF 融合）

```python
GraphConfig(
    transforms=["rewrite"],
    n_rewrites=3,
    retrievers=["vector", "es"],
    processors=["rrf", "rerank"],          # 先 RRF 融合，再 cross-encoder 精排
    retriever_weights={"vector": 1.0, "es": 0.8},
    top_k=5,
)
```

### 10.2 Vector + ES + Knowledge Graph 三路召回

```python
GraphConfig(
    retrievers=["vector", "es", "kg"],
    processors=["rrf", "rerank"],
    retriever_weights={"vector": 1.0, "es": 0.7, "kg": 1.2},  # KG 高权重
    top_k=5,
)
```

### 10.3 意图路由（SQL 问题只走 SQL）

```python
GraphConfig(
    retrievers=["vector", "es", "sql"],
    route_strategy="intent",               # LLM 判断 query 走哪些 retriever
    processors=["rerank"],
    top_k=5,
)
# "上季度销售额" → intent_router → ["sql"]
# "产品使用指南" → intent_router → ["vector", "es"]
```

### 10.4 Parent-Child 分块（小块检索，大块返回）

```python
GraphConfig(
    retrievers=["vector_parent_child", "es"],  # vector 侧用 parent-child 模式
    processors=["rrf", "rerank"],
    top_k=3,
)
```

### 10.5 Sentence Window 检索

```python
GraphConfig(
    retrievers=["vector_sentence_window", "es"],
    processors=["rerank"],
    top_k=5,
)
```

### 10.6 Self-Query（自动提取 metadata 过滤）

```python
GraphConfig(
    transforms=["self_query", "rewrite"],      # self_query 并行 + rewrite 并行
    retrievers=["vector", "es"],
    processors=["rrf", "rerank"],
    top_k=5,
)
# "2024年Q3财报" → self_query → TransformResult("财报", {"year":2024,"quarter":"Q3"})
#                → rewrite   → ["2024第三季度财务报告", ...]
# self_query 的 metadata_filter 通过 Send payload 传给 retrieve，无需字符串编码
```

### 10.7 HyDE + Step-Back 混合改写 + 多路检索 + 全栈后处理

```python
GraphConfig(
    transforms=["hyde", ["step_back", "rewrite"]],  # hyde 并行 + step_back→rewrite 串联
    n_rewrites=3,
    retrievers=["vector_parent_child", "es", "kg"],
    route_strategy="all",
    processors=["rrf", "compression", "rerank"],    # RRF → 压缩 → 精排
    retriever_weights={"vector_parent_child": 1.0, "es": 0.7, "kg": 1.5},
    grade_threshold=0.6,
    fallback_chain=["web"],
    max_retries=1,
    gen_mode="citation",
    top_k=5,
)
```

### 10.8 KB 优先 → Web 兜底（带质量阈值）

```python
GraphConfig(
    transforms=[["step_back", "rewrite"]],
    retrievers=["vector", "es"],
    grade_threshold=0.7,         # 分数低于 0.7 触发 fallback
    grade_strategy="llm",        # 用 LLM 判断相关性
    fallback_chain=["web"],
    max_retries=1,
    processors=["rrf", "rerank"],
    top_k=5,
)
```

---

## 11. 文件结构建议

```
rag/
├── config.py              # GraphConfig dataclass
├── types.py               # Document TypedDict, TransformResult, RetrieveInput
├── state.py               # RAGState TypedDict + reducers (merge_docs)
├── registry.py            # Registry 类 + 全局实例
├── handlers/
│   ├── transforms.py      # rewrite / decompose / hyde / step_back / self_query
│   ├── intent_router.py   # intent_router() 意图路由判断
│   ├── retrievers.py      # vector / es / kg / web / sql / parent_child / sentence_window
│   ├── processors.py      # rrf / weighted_fusion / rerank / filter / dedupe / compression
│   ├── graders.py         # score / llm / hybrid + _normalize_score
│   └── generators.py      # basic / cot / citation prompt 模板
├── nodes.py               # 6 个节点函数 + get_config() helper
├── graph.py               # build_rag_graph() 图组装
└── main.py                # 入口，加载 config，调用 graph.invoke()
```
