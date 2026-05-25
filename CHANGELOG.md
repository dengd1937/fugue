# Changelog

本文件遵循 [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) 1.1.0 规范，版本号遵循 [Semantic Versioning](https://semver.org/spec/v2.0.0.html)。

## [0.1.0] - 2026-05-24

### Added

#### 顶层 API
- `RAG` 门面类：统一 ingest / query 入口，支持同步与异步调用
- `RaglineConfig`：Pydantic v2 强类型配置，涵盖检索、生成、服务器参数
- 公共数据类型：`Document`、`QueryResult`、`IngestResult`
- 错误层次：`RaglineError` → `RaglineConfigError` / `RaglineRegistryError` / `RaglineLLMError` / `RaglineEmbeddingError` / `RaglineRetrieverError`

#### 检索引擎（基于 LangGraph）
- 6 节点有向图：`query_transform` → `retrieve` → `grade` → (`post_process` | `prepare_fallback` ↻) → `generate`
- `prepare_fallback`：grade 判定不足时进入回退循环，重新执行 query_transform
- `route_strategy`：支持 `all`（全量检索）/ `intent`（意图路由）两种策略
- 状态机基于 LangGraph `StateGraph`，节点之间通过类型化 State 传递

#### 内置 Handlers（7 类）
- **transforms**：`rewrite`（LLM 查询改写）/ `pipeline`（多步串联 transform）
- **retrievers**：`vector`（向量检索）/ `bm25`（稀疏检索）
- **processors**：`rrf`（倒数排名融合 RRF）/ `rerank`（重排序）
- **graders**：内置 `score` grader，grade_strategy 支持 `score` / `llm` / `hybrid` 三种判定策略
- **generators**：LLM 答案生成节点
- **parsers**：输出结构化解析（markdown / PDF）
- **chunkers**：`recursive`（递归字符分块，可配 `chunk_size` + `chunk_overlap`）

#### Providers
- **LLM**：OpenAI 兼容接口，支持任意 OpenAI 兼容端点（LLM provider）
- **Embedding**：OpenAI Embedding + 本地模型适配层
- **VectorStore**：ChromaDB 持久化与内存两种模式
- **BM25**：rank-bm25 封装，实现与向量检索统一接口
- **Reranker**：BGE 重排序器（FlagEmbedding，作为 `[bge]` extra）

#### HTTP Server（[server] extra）
- 基于 FastAPI + uvicorn 的 REST 服务
- 端点：`POST /query`、`POST /ingest`、`GET /health`
- CLI 入口：`ragline serve` / `ragline` 命令（`[server]` extra 安装后可用）

#### 对外测试支持（ragline.testing）
- `FakeLLM`：可编程假 LLM，用于单元测试隔离
- `FakeEmbedding`：返回固定向量的假 Embedding Provider
- `isolated_registries`：上下文管理器，测试期间隔离全局注册表副作用
- `mock_rag_providers`：上下文管理器，一键注入假 Provider 组合（FakeLLM + FakeEmbedding）
- `examples/quickstart`：可运行的最小示例，验证端到端集成

#### 工程基线
- PEP 561 `py.typed` marker，支持下游 mypy 严格类型检查
- `import-linter` 架构契约，禁止跨层循环依赖
- 260 个自动化测试，96.84% 语句覆盖率（`--cov-fail-under=96` 门控）
- 可选 extras：`[server]` / `[bge]` / `[chroma]` / `[pdf]` / `[all]`
- 懒加载设计：未安装 extra 时不导入对应重型依赖

### Notes

这是 ragline 的首个公开版本。在 0.x 系列内 API 可能调整；进入 1.0 后将遵循语义化版本承诺。
