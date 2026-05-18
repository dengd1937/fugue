---
feature: ragline
spec: docs/specs/ragline-design.md
routing: Development Workflow
---

# Ragline — 配置驱动的通用 Python RAG 库

> **核心理念**：图拓扑不变，行为全靠配置驱动。新需求通过插件接入，永不动图结构。

---

## 1. 产品定义

### 1.1 一句话描述

Ragline 是一个 Python RAG 库，让开发者通过修改配置（而非组装代码）快速构建、迭代生产级 RAG 系统，新需求通过 Registry 插件接入。

### 1.2 目标用户

- **画像**：在自家项目中集成 RAG 能力的 Python 中级以上开发者；独立开发者与企业 AI 工程师；**对架构有洁癖、需要长期演进**的工程师群体
- **现有方案痛点**：LangChain/LlamaIndex/Haystack 抽象层级散乱、组装代码冗长、provider 切换有锁定感、迭代场景常要动图结构；RAGFlow/Dify 是带 UI 的平台产品，不适合作为库集成
- **不服务**：希望"拖拽即用、零代码"的非开发者用户（这群人应该用 Dify）；只需要跑 demo 的研究者（应该用 FlashRAG）

### 1.3 产品形态

- **主**：Python SDK，`pip install ragline` → `RAG.from_yaml("config.yaml")` → `rag.ingest()` / `rag.query()`
- **次（薄壳）**：`ragline serve --config config.yaml` 起 FastAPI REST 服务
- 配置等价：`config.yaml` ↔ `RaglineConfig` dataclass ↔ Pydantic v2 校验模型

### 1.4 核心场景

1. **首次集成**：`pip install ragline` → 写 10 行 YAML → `ingest()` 索引文档 → `query()` 拿答案
2. **迭代需求**：「加一路 BM25 检索」「换成 citation 生成」「加 reranker」—— 改 YAML 一行，无需改代码
3. **扩展插件**：自家 SQL/KG/私有 API 检索器 —— 实现一个函数 + `@retriever_registry("my_kg")` 装饰，YAML 激活即可
4. **服务化**：同一份 YAML 用 `ragline serve` 起 REST，对外提供 `/query` `/ingest` 接口

### 1.5 核心卖点（差异化）

1. **图拓扑稳定**：7 节点状态机永不变形，所有行为变化通过 `GraphConfig` 驱动
2. **配置即行为**：一份 YAML 描述完整 RAG 行为，无需写组装代码
3. **无限可插拔**：Registry + entry_points，第三方插件包 `pip install` 即用

---

## 2. 竞品分析

### 2.1 直接对手对比

| 对手 | 形态 | 配置驱动度 | 多路召回 | Query 改写 | Reranker | Fallback/Grade | 拓扑稳定承诺 | Stars |
|---|---|---|---|---|---|---|---|---|
| **LangChain + LangGraph** | SDK + Studio | 仅节点参数级，**拓扑写代码** | ✅ 需手写 | 模板里有 | 需接 | 自己实现 | ❌ | 100k+ |
| **LlamaIndex** | SDK | Python 代码组装，**无原生 YAML** | ✅ 需手写 | ✅ | ✅ | 自己实现 | ❌ | 40k+ |
| **Haystack 2.x** | SDK + YAML 导出 | YAML 序列化，但**代码先行导出** | ✅ 需连边 | ✅ | ✅ | 自己实现 | ❌ | 17k+ |
| **RAGFlow** | 平台 + UI | 后台配置，非 SDK | ✅ | ✅ | ✅ | 引擎内置 | N/A | 30k+ |
| **Dify** | 平台 + 可视化 workflow | UI 拖拽，**面向非开发者** | ✅ | ✅ | ✅ | workflow 节点 | N/A | 60k+ |
| **RAGLight** | SDK + serve + Builder | env+Builder，最像 Ragline | hybrid | reformulation | 可选 | ❌ | ❌ | 660 |

### 2.2 市场空白判定

| Ragline 卖点 | 空白现状 |
|---|---|
| **配置即行为** | 部分被占（Haystack YAML、RAGLight env+Builder），但**都没做到"一份 YAML 完整描述行为"**。空白存在但不大。 |
| **图拓扑稳定** | **真空白**。所有现有库都是"加功能=改图/写节点"，没有明确承诺"拓扑不变"。最有原创性的点。 |
| **无限可插拔** | 大家都说支持多 provider，但深度差异大。Ragline 靠 Registry + entry_points 显式化，让第三方包真正零侵入。 |

### 2.3 必须避开的反模式

| 反模式 | 谁踩过 | Ragline 怎么避 |
|---|---|---|
| 抽象层太多（Settings/ServiceContext/StorageContext 嵌套） | LlamaIndex | 只有 `RaglineConfig` 三层 + Registry，扁平 |
| YAML 仅作序列化导出 | Haystack | YAML 是**主输入**，Python `GraphConfig` 只是等价物 |
| 集成市场依赖（每 provider 一独立包） | LangChain | OpenAI 兼容协议 cover 大部分，剩下走 Registry 插件 |
| 平台化诱惑（加 UI/用户/权限） | Dify/RAGFlow | MVP 明确不做 |
| 改一个需求改图结构 | 几乎所有人 | 拓扑稳定是核心承诺，CI 强制 |
| 研究堆砌（23 算法复现但没法生产） | FlashRAG | MVP 只做 5 种 transforms，每种打磨到位 |

---

## 3. 功能优先级与 MVP 范围

### 3.1 MVP 定义

> **MVP 交付的核心价值**：开发者能用一份 YAML（或等价 `GraphConfig`）跑通"分块 → 索引 → 多路召回 → 融合 → 精排 → 生成"完整链路，并能通过 Registry 接入自定义插件，**全程不动图结构**。

### 3.2 功能清单与优先级

#### 引擎与配置层（P0）

| 功能 | 优先级 |
|---|---|
| 7 节点图引擎（基于 LangGraph，隐藏在 `engine/`） | P0 |
| Registry 插件机制（7 个：transform/retriever/processor/grader/generator/parser/chunker） | P0 |
| `GraphConfig` dataclass + Pydantic v2 校验 | P0 |
| YAML 加载器（与 Python 等价） | P0 |
| Fallback 拓扑（节点+路由，**默认禁用**） | P0 |

#### 文档处理层

| 功能 | 优先级 |
|---|---|
| Recursive Chunker | P0 |
| Markdown / Plain Text Parser | P0 |
| PDF Parser（pypdf 起步） | P0 |
| Sentence / Token Chunker | P1 |
| HTML / Word / Excel Parser | P1 |
| 增量索引 / 缓存 | P2 |

#### Provider 层

| 功能 | 优先级 |
|---|---|
| OpenAI 兼容 LLM | P0 |
| OpenAI 兼容 Embedding | P0 |
| Chroma Vector Store | P0 |
| 内置 BM25（rank_bm25） | P0 |
| BGE-Reranker 本地（FlagEmbedding） | P0 |
| Qdrant / PGVector | P1 |
| Anthropic / Gemini 原生 SDK | P2 |
| Cohere / Jina Reranker API | P2 |

#### Query Transforms（差异化亮点）

| 功能 | 优先级 |
|---|---|
| `rewrite`（基础改写） | P0 |
| `hyde`（假设文档） | P0 |
| `step_back`（抽象上位） | P0 |
| **嵌套 transforms（并行+串联混合）** | P0 |
| `decompose`（子问题分解） | P1 |
| `self_query`（提取 metadata filter） | P1 |
| Intent Router (`route_strategy="intent"`) | P1 |

#### Multi-Route 与融合

| 功能 | 优先级 |
|---|---|
| Vector + BM25 笛卡尔积召回 | P0 |
| RRF 融合 | P0 |
| BGE Reranker | P0 |
| `weighted_fusion` | P1 |
| `compression`（LLM 文档压缩） | P1 |
| `filter` / `dedupe` | P1 |
| Parent-Child / Sentence-Window Retriever | P1 |

#### Grade & Generator

| 功能 | 优先级 |
|---|---|
| `score` grader（归一化+阈值） | P0 |
| `basic` generator | P0 |
| `citation` generator | P0 |
| `llm` grader | P1 |
| `cot` generator | P1 |
| `hybrid` grader | P2 |

#### SDK & Server

| 功能 | 优先级 |
|---|---|
| `RAG.ingest(paths)` / `RAG.query(q)` SDK API | P0 |
| `ragline serve` 最小 REST（`/ingest`, `/query`, `/health`） | P0 |
| `/collections` endpoint | P1 |
| `ragline chat` CLI 向导 | P1 |
| Streaming（`/stream`、SDK 异步迭代） | P2 |
| `RAG.dry_run()` 调试方法 | P1 |

### 3.3 明确不做（MVP）

- ❌ Streaming
- ❌ 评估 (RAGAS / faithfulness / hit rate)
- ❌ 管理后台 UI
- ❌ 多租户 / 用户系统 / 权限
- ❌ Agent / 多轮对话 / 工具调用
- ❌ 内置 Web / KG 检索器实现（接口预留）

### 3.4 迭代规划

| 版本 | 范围 |
|---|---|
| V1 (MVP, 0.1.0) | 全部 P0 |
| V1.x | LLM grader / cot+decompose+self_query / intent router / 更多 parser / Qdrant+PGVector / CLI 向导 / weighted_fusion+compression+filter+dedupe / parent-child+sentence-window |
| V2 (1.0.0) | Streaming / 缓存 / 更多 reranker / 更多 LLM 原生 SDK / hybrid grader |

---

## 4. 技术设计

### 4.1 八项核心决策汇总

| 编号 | 决策 | 选定 |
|---|---|---|
| D1 | 图引擎底层 | **B. 隐藏式 LangGraph**（用户不可见，可未来替换） |
| D2 | Provider 适配 | **A. OpenAI 兼容 SDK** |
| D3 | Registry 物理 | **全局单例 + `register()` + `entry_points` 自动发现** |
| D4 | YAML 校验 | **Pydantic v2**（与 dataclass 互转） |
| D5 | SDK 风格 | **同步主**，async P1 |
| D6 | Ingest 流水线 | **A. 独立 pipeline，Registry 化** |
| D7 | Server 框架 | **FastAPI + uvicorn** |
| D8 | 项目结构 | **单包 `ragline` + extras** (`server`/`bge`/`chroma`/`pdf`) |

### 4.2 Architecture（整体架构）

#### 目录结构

```
ragline/
├── pyproject.toml              # uv 管理 + extras: server/bge/chroma/pdf
├── src/ragline/
│   ├── __init__.py             # 公开 API re-export
│   ├── api/                    # ━━ 公开 API 层 ━━
│   │   ├── rag.py              # RAG class
│   │   ├── ingest.py           # IngestPipeline class
│   │   └── types.py            # Document / QueryResult / IngestResult / Ragline*Error
│   ├── config.py               # RaglineConfig + GraphConfig + Pydantic + YAML loader
│   ├── registry.py             # Registry 类 + 7 全局单例 + entry_points
│   ├── engine/                 # ━━ 隐藏 LangGraph 层（用户不可见）━━
│   │   ├── state.py            # RAGState + merge_docs reducer
│   │   ├── runtime.py          # get_config() 配置注入
│   │   ├── graph.py            # build_rag_graph()
│   │   └── nodes/              # 6 个节点拆文件
│   │       ├── query_transform.py / retrieve.py / grade.py
│   │       ├── prepare_fallback.py / post_process.py / generate.py
│   ├── handlers/               # ━━ Registry 插件实现 ━━
│   │   ├── transforms/         # rewrite / hyde / step_back / pipeline(嵌套执行器)
│   │   ├── retrievers/         # vector / bm25
│   │   ├── processors/         # rrf / rerank
│   │   ├── graders/            # score / normalizer
│   │   ├── generators/         # basic / citation
│   │   ├── parsers/            # markdown / text / pdf
│   │   └── chunkers/           # recursive
│   ├── providers/              # ━━ 外部服务客户端（非插件）━━
│   │   ├── llm.py              # OpenAI 兼容 LLM 统一 client
│   │   ├── embedding.py        # OpenAI 兼容 Embedding 统一 client
│   │   ├── bm25.py             # rank_bm25 封装
│   │   ├── vector_store/       # base.py + chroma.py
│   │   └── reranker/           # base.py + bge.py
│   └── server/                 # ━━ extras [server]：FastAPI 薄壳 ━━
│       ├── app.py / endpoints.py / cli.py
├── tests/{unit,integration,e2e,fixtures}/
└── docs/{specs,modules,adr,plans}/
```

**约束**：每层 ≤ 8 个文件/目录；单文件 ≤ 800 行。

#### 隐藏式 LangGraph 边界

```
┌─────────────────────────────────────────────────────────┐
│                  用户世界（公开 API）                     │
│   from ragline import RAG, GraphConfig                    │
│   from ragline.registry import retriever_registry         │
│   ★ 用户代码里看不到任何 langgraph 类型 ★               │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  内部世界（engine/）                      │
│   from langgraph.graph import StateGraph, END           │
│   from langgraph.types import Command, Send, Overwrite  │
│   ★ LangGraph 是实现细节，可未来替换 ★                  │
└─────────────────────────────────────────────────────────┘
```

**强制约束**：`import-linter` 在 CI 强制执行 ——
- 仅 `engine/` 可 import `langgraph` / `langchain_core`
- `api/` / `config.py` / `registry.py` / `handlers/` / `providers/` / `server/` 禁止

#### 两条主数据流

**Query 路径**：
```
RAG.query(question)
  └─► api/rag.py
       ├─► load RaglineConfig（含 GraphConfig）
       └─► engine/graph.invoke({original_query: ...}, config={configurable: cfg})
             │
             ▼
       engine/nodes/* 通过 Registry 调用 handlers/
       handlers/ 通过 providers/ 调外部服务
             │
             ▼
       QueryResult { answer, ranked_documents, grade_score, ... }
```

**Ingest 路径（独立 pipeline）**：
```
RAG.ingest(paths)
  └─► api/ingest.py: IngestPipeline
       ├─► parser_registry → list[ParsedDocument]
       ├─► chunker_registry → list[Chunk]
       ├─► providers/embedding → list[list[float]]
       ├─► providers/vector_store.add(...)
       └─► bm25 provider 索引 update
        │
        ▼
   IngestResult { num_documents, num_chunks, duration_seconds }
```

#### Registry 与 entry_points

```python
# ragline/registry.py
class Registry(Generic[H]):
    def register(self, name: str, fn: H) -> None: ...
    def get(self, name: str) -> H: ...
    def __call__(self, name: str) -> Callable[[H], H]: ...  # 装饰器糖

# 7 个全局单例
transform_registry  = Registry[TransformFn]("transform")
retriever_registry  = Registry[RetrieverFn]("retriever")
processor_registry  = Registry[ProcessorFn]("processor")
grader_registry     = Registry[GraderFn]("grader")
generator_registry  = Registry[GeneratorFn]("generator")
parser_registry     = Registry[ParserFn]("parser")
chunker_registry    = Registry[ChunkerFn]("chunker")
```

**第三方插件包发现**：`RAG.__init__` 扫描 `importlib.metadata.entry_points(group="ragline.handlers")`，第三方包写：

```toml
[project.entry-points."ragline.handlers"]
my_kg_retriever = "my_pkg.handlers:register"
```

`pip install ragline-kg-plugin` 后用户**无需 import**，YAML 写 `retrievers: ["vector", "kg"]` 即用。

### 4.3 Components（核心组件契约）

#### RaglineConfig 三层

```python
@dataclass
class RaglineConfig:
    graph:     GraphConfig
    ingest:    IngestConfig
    providers: ProviderConfig

@dataclass
class GraphConfig:
    # 沿用 fuge_plan.md，行为单一变更入口
    transforms: list[str | list[str]] = field(default_factory=lambda: ["rewrite"])
    n_rewrites: int = 3
    max_queries: int = 20
    retrievers: list[str] = field(default_factory=lambda: ["vector", "bm25"])
    route_strategy: Literal["all", "intent"] = "all"
    retriever_weights: dict[str, float] = field(default_factory=dict)
    grade_threshold: float = 0.6
    grade_strategy: Literal["score", "llm", "hybrid"] = "score"
    score_normalizers: dict[str, float] = field(default_factory=lambda: {"bm25": 20.0})
    fallback_chain: list[str] = field(default_factory=list)  # MVP 默认空
    max_retries: int = 1
    processors: list[str] = field(default_factory=lambda: ["rrf", "rerank"])
    top_k: int = 3
    gen_mode: Literal["basic", "citation"] = "basic"   # cot 为 P1
    temperature: float = 0.7

@dataclass
class IngestConfig:
    parser:          str = "auto"     # 按扩展名分派
    chunker:         str = "recursive"
    chunk_size:      int = 512
    chunk_overlap:   int = 64
    collection_name: str = "default"
    persist_dir:     str = "./.ragline"

@dataclass
class ProviderConfig:
    # LLM
    llm_base_url:       str = "https://api.openai.com/v1"
    llm_api_key:        str | None = None  # 默认读 OPENAI_API_KEY
    llm_model:          str = "gpt-4o-mini"
    llm_timeout:        float = 60.0
    llm_max_retries:    int   = 2
    llm_concurrency:    int   = 10
    # Embedding
    embedding_base_url:    str | None = None  # None 复用 llm_base_url
    embedding_api_key:     str | None = None
    embedding_model:       str = "text-embedding-3-small"
    embedding_timeout:     float = 30.0
    embedding_max_retries: int = 2
    embedding_concurrency: int = 5
    embedding_batch_size:  int = 64
    # Reranker（本地）
    reranker_model:   str = "BAAI/bge-reranker-v2-m3"
    reranker_timeout: float = 30.0
    reranker_device:  Literal["cpu", "cuda", "auto"] = "auto"
```

**YAML 等价表示**：
```yaml
graph:
  transforms: ["rewrite", ["step_back", "rewrite"]]
  n_rewrites: 3
  retrievers: ["vector", "bm25"]
  processors: ["rrf", "rerank"]
  fallback_chain: []
ingest:
  chunker: "recursive"
  chunk_size: 512
providers:
  llm_model: "gpt-4o-mini"
  llm_api_key: "${OPENAI_API_KEY}"   # 支持 ${env_var} 展开
```

#### RAG 主入口

```python
class RAG:
    def __init__(
        self,
        config: RaglineConfig | None = None,
        *,
        collection_name: str | None = None,
        env_file: str | Path | None = None,
    ) -> None: ...

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "RAG":
        """从 YAML 实例化，overrides 可覆盖任意层级字段。"""

    def ingest(
        self, sources: str | Path | Iterable[str | Path],
        *, show_progress: bool = True,
    ) -> IngestResult: ...

    def query(
        self, question: str,
        *, graph_override: GraphConfig | None = None,
    ) -> QueryResult: ...

    def close(self) -> None: ...
    def __enter__(self) -> "RAG": ...
    def __exit__(self, *exc) -> None: ...
```

#### 公开类型

```python
class Document(TypedDict):
    doc_id:   str
    content:  str
    score:    float
    source:   str
    metadata: dict[str, Any]

@dataclass
class QueryResult:
    answer:             str
    ranked_documents:   list[Document]
    grade_score:        float
    grade_decision:     Literal["sufficient", "insufficient"]
    rewritten_queries:  list[str]
    retrieval_rounds:   int          # 1=无 fallback；2+ 走过 fallback

@dataclass
class IngestResult:
    num_documents:    int
    num_chunks:       int
    collection_name:  str
    duration_seconds: float

# 三种内部数据对象的区分
ParsedDocument  # 文件 → parser 输出
Chunk           # ParsedDocument → chunker 输出，写入 vector store
Document        # query 路径，retriever 输出（全图流转的统一类型）
```

#### Handler 签名协议

```python
TransformFn = Callable[[list[str], int], list[str | TransformResult]]
RetrieverFn = Callable[[str, dict | None], list[Document]]
ProcessorFn = Callable[..., list[Document]]
GraderFn    = Callable[..., tuple[float, Literal["sufficient", "insufficient"]]]
GeneratorFn = Callable[..., str]
ParserFn    = Callable[[Path], list[ParsedDocument]]
ChunkerFn   = Callable[..., list[Chunk]]
```

`TransformResult` 携带 `metadata_filter`，通过 Send payload 传给 retrieve（非字符串编码）。

#### Provider 协议

```python
class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...
    def similarity_search(
        self, query_embedding: list[float],
        k: int = 20, metadata_filter: dict | None = None,
    ) -> list[Document]: ...
    def delete_collection(self) -> None: ...
    def stats(self) -> dict: ...

class Reranker(Protocol):
    def rerank(
        self, query: str, documents: list[str], top_k: int | None = None,
    ) -> list[tuple[int, float]]: ...

class LLMClient:
    """OpenAI 兼容统一封装。base_url + api_key 即可切换 provider。"""
    def complete(self, prompt: str, *, temperature: float = 0.7) -> str: ...

class EmbeddingClient:
    def embed(self, texts: list[str]) -> list[list[float]]: ...
```

### 4.4 Data Flow（数据流与状态机）

#### Query 路径完整状态变化

```
Initial State:
  { original_query, source="kb", retry_count=0,
    rewritten_queries=[], documents=[], retrieval_history=[],
    grade_score=0, grade_decision="", ranked_documents=[], answer="" }
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│ query_transform                                                  │
│  • 遍历 cfg.transforms 顶层（嵌套并行+串联）→ all_queries        │
│  • 去重 + 截断 max_queries                                       │
│  • 构建 Sends: queries × retriever_names                         │
│  Command(                                                        │
│    update={ rewritten_queries, "documents": Overwrite([]) },     │
│    goto=[Send("retrieve", {query, retriever_name, source,        │
│                            metadata_filter})] × N                │
│  )                                                               │
└─────────────────────────────────────────────────────────────────┘
       │
       ├──► retrieve × N（并行实例，每个独立 Send payload）
       │     │
       │     ▼ merge_docs reducer 按 (source, doc_id) 去重合并
       │
       ▼ LangGraph fan-in 保证：所有 N 完成才进 grade
┌─────────────────────────────────────────────────────────────────┐
│ grade                                                            │
│  update: { grade_score, grade_decision,                          │
│            retrieval_history: history + [docs] }                 │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼ conditional edge: route_after_grade
       ├──► sufficient ────────────────────────► post_process
       └──► insufficient + 还有 fallback ─────► prepare_fallback
                                                  • source = next
                                                  • retry_count++
                                                  └─► query_transform (环路)
```

#### Fallback 状态机（环形路径）

```
第 1 轮（source=kb）
  query_transform → retrieve(vec/bm25) → grade
                                          ↓ insufficient
                                  prepare_fallback
                                  source="web", retry_count=1
                                          ↓
第 2 轮（source=web）
  query_transform
    • Overwrite([]) 重置 documents      ← 关键：上轮 kb 文档不混入
    • retrieval_history 保留上轮归档     ← 不被清空
  retrieve(web) → grade
                  ↓ sufficient（或 fallback 耗尽兜底）
              post_process
                • 合并 documents + retrieval_history[:-1]
                • 去重 → rrf → rerank → top_k
```

#### 三个易踩坑设计点（必须有 integration test 覆盖）

1. **`Overwrite([])` 的必要性**：`documents` 字段用 `merge_docs` reducer，直接返回 `{"documents": []}` 会被视为空增量，上轮文档保留 → 必须 `Overwrite([])` 强制清空。
2. **`retrieval_history[:-1]` 切片**：当前轮文档已在 `state["documents"]`，`retrieval_history[-1]` 是同一轮归档，要切掉避免重复合并。
3. **Send payload 与 RAGState 不同**：retrieve 节点收 `RetrieveInput`（Send payload），不是 `RAGState`，文档要强调。

#### Ingest 路径（顺序流水线）

```
RAG.ingest(["./docs/*.pdf", "./README.md"])
       │
       ▼
1. glob 展开 → list[Path]
2. parser_registry.get(ext or "auto")(path) → list[ParsedDocument]
3. chunker_registry.get(name)(parsed_docs, chunk_size, chunk_overlap) → list[Chunk]
4. embedding_client.embed([c.content for c in chunks]) → list[list[float]]
5. vector_store.add(chunks, embeddings)
6. bm25_provider.update(chunks)   ← 当 retrievers 含 "bm25"
       │
       ▼
IngestResult(num_documents, num_chunks, duration_seconds)
```

**BM25 持久化决策**：MVP 用**内存重建**（`RAG.__init__` 从 Chroma 拉所有 chunks 构建索引）；P1 优化为 pickle 持久化。

**chunk_id 生成规则**：`sha1(source_path + chunk_index + content[:128])`，Chroma upsert 接管重复 ingest 去重。

### 4.5 Error Handling（错误处理与运行时韧性）

#### 异常分类（`ragline/api/types.py`）

```python
class RaglineError(Exception): ...
class RaglineConfigError(RaglineError): ...
class RaglineRegistryError(RaglineError): ...
class RaglineLLMError(RaglineError): ...
class RaglineEmbeddingError(RaglineError): ...
class RaglineRetrieverError(RaglineError): ...   # best-effort，正常不抛
```

#### Failure 边界 8 类决策

| 失败场景 | MVP 行为 | 理由 |
|---|---|---|
| 单个 retriever 抛异常 | 节点内 try/except，返回空 `documents`，记录到 error 元数据 | best-effort：单源失败不拖垮全图 |
| 所有 retriever 失败 | reducer 合并后 documents=[]，grade insufficient 兜底 → generate 收到空上下文 → LLM 自答"没找到" | 不 fail-fast，给用户回答兜底 |
| transforms LLM 失败 | OpenAI SDK retry 耗尽后抛 `RaglineLLMError` | 不静默兜底（付费调用要让用户感知） |
| generate LLM 失败 | 抛 `RaglineLLMError`，但 `ranked_documents` 已组装好附带错误堆栈 | 帮用户定位是 LLM 还是检索问题 |
| Embedding 失败（ingest） | 分批重试；最终失败抛 `RaglineEmbeddingError`，已写入不回滚 | RAG 数据可重复 ingest，不做事务 |
| Fallback 链耗尽仍 insufficient | post_process 兜底；空也给 LLM 让其自答 | 一致行为：所有"找不到"都给回答 |
| YAML 配置错误 | Pydantic `ValidationError` 包装为 `RaglineConfigError` 含字段路径 | 启动期发现，不让运行期才崩 |
| Registry 缺 handler | `KeyError` 包装为 `RaglineRegistryError` 含可用列表 | `'kg' not registered. Available: ['vector', 'bm25']` |

#### 重试策略（分层不叠加）

| 调用类型 | 重试方 | 默认 |
|---|---|---|
| LLM | OpenAI SDK 内置（仅 429/500/503/超时） | 2 次 |
| Embedding | Ragline ingest 层（批量分块二次切分） | 2 次 |
| Reranker / Chroma | 不重试 | - |

#### 超时控制

每个外部依赖单点超时（`ProviderConfig.*_timeout`），**不做 graph 整体超时**（P1）。

#### 并发控制

- `max_queries=20` 默认值作为第一道闸门
- `LLMClient` 内置 `threading.Semaphore`，`llm_concurrency=10`
- `EmbeddingClient` 类似，`embedding_concurrency=5`
- vector/bm25 本地不限流
- 用户自定义 retriever 内调 LLM/外部 API 需自行加锁，文档给出示例

#### 启动期 fail-fast 校验

`RAG.__init__` 末尾 `_validate_config()` 一次性执行（不延迟到运行期）：

- LLM `api_key` 存在（含 env_file 加载）
- `persist_dir` 可读写
- `cfg.graph.transforms/retrievers/processors/graders/generators` 所有名字 ∈ 对应 Registry
- `cfg.ingest.parser/chunker` 已注册
- `cfg.graph.fallback_chain` 中所有源已注册（如非空）
- Pydantic 字段类型校验

#### 可观测性

- **MVP**：Python `logging`，logger name `ragline.<package>.<module>`
- **P1**：`RaglineConfig.on_event` callback，用户接 OpenTelemetry / Langfuse
- **不做**：内置 Prometheus / Datadog 依赖

#### 资源生命周期

- `RAG.close()` 显式清理：reranker 显存、Chroma 连接、httpx client、BM25 索引
- `with RAG.from_yaml(...) as rag:` 推荐用法
- Server 模式靠 FastAPI lifespan 注册清理

### 4.6 Testing（测试策略）

#### 测试分层

```
tests/
├── unit/           # 纯函数 / Registry / mock 全外部依赖
├── integration/    # 节点级 + 多组件联动（mock LLM/Embedding，真 Chroma）
├── e2e/            # 真 OpenAI 兼容 API + 真 Chroma + 真 BGE 模型
└── fixtures/       # sample docs / PDFs / mock LLM 响应
```

#### 外部依赖矩阵

| 依赖 | unit | integration | e2e |
|---|---|---|---|
| LLM API | mock httpx | mock `LLMClient.complete` | 真 key（gpt-4o-mini） |
| Embedding API | mock httpx | mock `EmbeddingClient.embed` | 真 key |
| Chroma | mock protocol | 真 Chroma（`tmp_path`） | 真 Chroma |
| BGE Reranker | mock protocol | deterministic dummy | 真模型（CI 缓存） |

#### 必测的关键场景

**Integration 层**（最重要）：

- 嵌套 transforms 扇出：`["hyde", ["step_back", "rewrite"]]` × `n_rewrites=3` × 2 retrievers → 验证 Send 数量
- `max_queries` 截断
- `self_query` metadata_filter 通过 Send payload 传递（非字符串编码）
- `intent_router` 路由（mock LLM 返回 `["sql"]`）
- `merge_docs` 多 retriever 同 doc_id 按 source 区分保留
- **Fallback 完整闭环**：insufficient → prepare_fallback → query_transform → `Overwrite([])` 重置 → 第二轮 retrieve → 验证第一轮文档已清空、retrieval_history 保留
- Fallback 链耗尽兜底
- post_process 跨轮合并 + 去重
- 空 documents 兜底（LLM 自答）
- citation generator 包含编号

#### CI 设计

| 项目 | ci.yml（PR/push） | e2e.yml（nightly） |
|---|---|---|
| 内容 | unit + integration + ruff + mypy + import-linter | e2e（需 `OPENAI_API_KEY` secret） |
| Python 矩阵 | 3.12, 3.13 | 3.12 |
| 时间预算 | < 5 min | < 10 min |
| 覆盖率门槛 | **≥ 90%（核心模块 engine/、config.py、registry.py、api/）** | 不强制 |
| 阻塞 PR | ✅ | ❌（仅通知） |

#### 工具栈

| 用途 | 工具 |
|---|---|
| 测试运行器 | `pytest` |
| 覆盖率 | `pytest-cov` + Codecov |
| Mock | `pytest-mock` |
| Prompt 快照 | **`syrupy`（MVP 必引入）** —— 防止 prompt 默改 |
| Lint / 类型 | `ruff` + `mypy`（CI 强制） |
| Import 边界守护 | **`import-linter`（CI 强制）** —— engine/ 唯一允许 import langgraph |
| 依赖管理 | `uv` |

### 4.7 关键设计约束与承诺

#### 物理约束（CI 强制）

1. **import-linter**：`engine/` 是唯一允许 import `langgraph` / `langchain_core` 的模块
2. **每层 ≤ 8 个文件/目录、单文件 ≤ 800 行**：用 `pre-commit` hook 检查
3. **覆盖率 ≥ 90%（核心模块）**：CI 阻塞低于门槛的 PR
4. **`uv add <package>`**：禁止 pip/poetry/conda（hook 强制）

#### 多线程承诺

- `RAG.query()` **线程安全**（FastAPI 多线程并发安全）
- `RAG.ingest()` **不能与 query 并发**（BM25 索引 update 阶段）
- **插件注册时机**：所有 Registry 注册（手动 `@registry("name")` + entry_points 自动发现）应在**第一次 `RAG()` 实例化完成之前**发生（`RAG.__init__` 内的 entry_points 扫描完成时即截止）。此后**不应**进行动态注册——Registry 全局单例的写入操作非线程安全

#### 版本承诺

- MVP = `0.1.0`
- `0.x` 期间任何小版本可能 breaking，README 显式标注
- `1.0.0` 稳定承诺范围：`GraphConfig` 字段名 / Registry API / `RAG.from_yaml/ingest/query` 主入口签名

#### 安全承诺

- `ragline serve` MVP **无鉴权**，文档明确"仅限 localhost 或受信内网部署"
- P1 加 token-based auth

### 4.8 端到端使用示例

```yaml
# config.yaml
graph:
  transforms: ["rewrite", "hyde"]
  n_rewrites: 3
  retrievers: ["vector", "bm25"]
  processors: ["rrf", "rerank"]
  top_k: 3
  gen_mode: "citation"
ingest:
  chunker: "recursive"
  chunk_size: 512
providers:
  llm_model: "gpt-4o-mini"
  llm_api_key: "${OPENAI_API_KEY}"
```

```python
# main.py
from ragline import RAG

with RAG.from_yaml("config.yaml") as rag:
    rag.ingest(["./docs/*.pdf", "./README.md"])
    result = rag.query("LangGraph 怎么用动态路由？")
    print(result.answer)
    for doc in result.ranked_documents:
        print(f"[{doc['source']}] {doc['score']:.3f}")
```

服务化：

```bash
ragline serve --config config.yaml --port 8000
# 仅限 localhost / 受信网络

curl -X POST localhost:8000/ingest -d '{"paths": ["./docs"]}'
curl -X POST localhost:8000/query  -d '{"question": "..."}'
```

第三方插件包：

```python
# ragline_kg_plugin/handlers.py
from ragline.registry import retriever_registry
from ragline.api.types import Document

@retriever_registry("kg")
def kg_search(query: str, metadata_filter: dict | None = None) -> list[Document]:
    ...

def register() -> None:
    pass  # 被 entry_points 触发时自动 import
```

```toml
# ragline_kg_plugin/pyproject.toml
[project.entry-points."ragline.handlers"]
kg = "ragline_kg_plugin.handlers:register"
```

用户 `pip install ragline-kg-plugin` 后，直接在 YAML 写 `retrievers: ["vector", "kg"]` 即用。

---

## 5. 路由决策

**后续工作流**：**Development Workflow**

**理由**：本 spec 已穷尽产品定义 + 技术设计 + 测试策略。MVP 无新 UI（CLI 仅基础 `ragline serve`），不触发 Design Workflow。下一步是 `writing-plans` skill 将 spec 转为分任务的实现计划（`docs/plans/ragline.md`），然后 `subagent-driven-development` 执行。

---

## 6. 参考资料

- 原始设计：`docs/specs/fuge_plan.md`
- 竞品资料：
  - [LangChain RAG Agent (LangGraph)](https://docs.langchain.com/oss/python/langgraph/agentic-rag)
  - [LlamaIndex Ingestion Pipeline](https://developers.llamaindex.ai/python/framework/module_guides/loading/ingestion_pipeline/)
  - [Haystack Pipelines](https://docs.haystack.deepset.ai/docs/pipelines)
  - [RAGLight](https://github.com/Bessouat40/RAGLight)
  - [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG)
