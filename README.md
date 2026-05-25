# Ragline

[![PyPI version](https://img.shields.io/pypi/v/ragline.svg)](https://pypi.org/project/ragline/)
[![License: MIT](https://img.shields.io/pypi/l/ragline.svg)](https://github.com/dengd1937/ragline/blob/main/LICENSE)
[![Python versions](https://img.shields.io/pypi/pyversions/ragline.svg)](https://pypi.org/project/ragline/)
[![CI status](https://github.com/dengd1937/ragline/actions/workflows/ci.yml/badge.svg)](https://github.com/dengd1937/ragline/actions/workflows/ci.yml)

Configuration-driven RAG. Topology stays, behavior plugs in.

Ragline 是一个配置驱动的 Python RAG 库，基于 LangGraph 构建。图拓扑（transform → retrieve → grade → process → generate）固定不变，每个节点的行为通过 YAML 配置和插件机制动态注入。

---

## Install

```bash
# 推荐：安装所有可选组件
pip install "ragline[all]"

# 最小安装（含 ChromaDB 向量存储；无 PDF、无本地 reranker、无 server）
pip install ragline

# 按需安装 extras
pip install "ragline[server]"   # REST API 服务（FastAPI + uvicorn）
pip install "ragline[chroma]"   # ChromaDB 向量存储（已随主包默认安装；保留作向后兼容）
pip install "ragline[bge]"      # 本地 BGE reranker（FlagEmbedding）
pip install "ragline[pdf]"      # PDF 解析（pypdf）
```

---

## Quick Start

创建配置文件 `ragline.yaml`：

```yaml
graph:
  transforms:
    - rewrite
  n_rewrites: 3
  retrievers:
    - vector
    - bm25
  processors:
    - rrf
    - rerank
  top_k: 3
  gen_mode: citation

ingest:
  chunk_size: 512
  chunk_overlap: 64
  persist_dir: ./.ragline

providers:
  llm_model: gpt-4o-mini
  llm_api_key: "${OPENAI_API_KEY}"
```

编写 Python 代码：

```python
from ragline import RAG

with RAG.from_yaml("ragline.yaml") as rag:
    rag.ingest(["./docs/**/*.md"])

    result = rag.query("Ragline 的核心设计原则是什么？")
    print(result.answer)

    for doc in result.ranked_documents:
        print(f"[{doc['source']}] {doc['content'][:80]}")
```

`RAG` 支持上下文管理器（`with` 语句），退出时自动释放向量库、LLM 客户端等所有资源。也可以显式调用 `rag.close()`。

---

## 在外部项目中使用 ragline

Ragline 当前尚未发布到 PyPI，但可以通过以下三种方式在外部项目中引用：

**方式一：本地路径 editable 安装（适合本地开发调试）**

```bash
uv add --editable "/path/to/ragline[all]"
```

**方式二：通过 Git URL 安装（适合 CI/CD 或团队共享环境）**

```bash
uv add "ragline @ git+https://github.com/<you>/ragline.git@main"
```

**方式三：uv workspace（适合 monorepo 场景）**

在外部项目的 `pyproject.toml` 中配置 workspace，将 ragline 作为本地成员包引用，uv 会自动处理依赖解析。

完整可运行示例请参考 [`examples/quickstart/`](./examples/quickstart/) 目录，其中包含最小化的消费者项目结构和配置文件。

---

## 在测试中使用 ragline

Ragline 提供 `ragline.testing` 模块，包含以下测试辅助工具：

```python
from ragline.testing import FakeLLM, FakeEmbedding, isolated_registries, mock_rag_providers
```

- `FakeLLM` / `FakeEmbedding`：可配置返回值的假实现，无需真实 API Key
- `isolated_registries`：上下文管理器，在测试期间隔离全局注册表，防止测试间污染
- `mock_rag_providers`：组合 context manager，同时 patch LLM 和 Embedding 提供者

**最小 pytest 示例：**

```python
def test_my_consumer():
    with isolated_registries(), mock_rag_providers() as (llm, _):
        llm.answer = "expected answer"
        from ragline import RAG, RaglineConfig
        cfg = RaglineConfig(...)
        cfg.providers.llm_api_key = "fake"
        with RAG(cfg) as rag:
            result = rag.query("...")
            assert result.answer == "expected answer"
```

**注意**：RAG 是同进程单例，同一进程内只能存在一个实例。在测试中使用 `isolated_registries()` 可以在每个测试之间重置注册表状态，但仍建议每个测试用例独立创建和销毁 `RAG` 实例，避免状态泄漏。详见 [Status & Compatibility](#status--compatibility) 节中的「MVP 单实例限制」说明。

---

## Why Ragline?

### 配置即行为（Configuration-as-behavior）

RAG 流水线的每一个节点——query transform、retriever 组合、post-processor、grader、generator——全部通过 YAML 字段名驱动。切换检索策略、调整 rewrite 次数、启用 citation 模式，只需修改配置文件，不需要改代码、不需要重新部署。

### 图拓扑稳定（Topology stability）

基于 LangGraph 构建的有向图拓扑（transform → retrieve → grade → process → generate → fallback）在所有配置下保持不变。稳定的拓扑意味着可预期的行为、可靠的测试覆盖、以及在生产环境中一致的错误边界。

### 无限可插拔（Unlimited pluggability）

任何节点都可以被第三方插件替换或扩展。通过 `entry_points` 机制，插件包安装后自动被 Ragline 发现，无需修改框架代码。内置注册表（Registry）提供 transform、retriever、processor、grader、generator、parser、chunker 七个扩展点。

---

## Plugin Example

下面展示如何用约 30 行代码编写一个自定义 retriever，并通过 entry_points 接入 Ragline。

```python
# my_plugin/retrievers.py
from typing import Any

from ragline.registry import retriever_registry


def my_custom_retriever(
    query: str,
    top_k: int = 5,
    metadata_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """调用外部搜索 API 的自定义检索器。"""
    # 在这里接入你的检索逻辑（外部搜索 API、数据库、知识图谱等）
    return [
        {
            "doc_id": "x1",
            "content": "来自外部检索的结果",
            "score": 0.9,
            "source": "my_source",
            "metadata": {},
        }
    ]


def register() -> None:
    retriever_registry.register("my_custom", my_custom_retriever)
```

在 `pyproject.toml` 中声明 entry point：

```toml
[project.entry-points."ragline.handlers"]
my_plugin = "my_plugin.retrievers:register"
```

安装插件包后，在 `ragline.yaml` 中直接使用：

```yaml
graph:
  retrievers:
    - vector
    - my_custom
  processors:
    - rrf
```

Ragline 在 `RAG()` 初始化时自动调用 `discover_plugins()` 扫描所有已安装的 entry points。

---

## Status & Compatibility

当前版本为 **0.x（MVP）**，API 尚未稳定。在达到 1.0 之前，minor 版本升级可能包含 breaking changes，请升级前查阅 CHANGELOG。

**稳定契约（0.x 期间承诺不 breaking 变更的接口）：**

- `GraphConfig` / `IngestConfig` / `ProviderConfig` 的已有字段名（新增字段不算 breaking）
- `Registry.register(name, fn)` / `Registry.get(name)` / `Registry.has(name)` API
- `RAG.from_yaml(path)` / `RAG.ingest(sources)` / `RAG.query(question)` 方法签名

**MVP 单实例限制：**

同一进程内只支持一个 `RAG` 实例。Registry 是全局单例，创建第二个 `RAG` 实例会覆盖前一个的 handler 闭包，行为不可预测。多 RAG 实例场景请使用独立进程。

---

## Security

`ragline serve`（即 `ragline.server`）以 **单 worker、无认证** 模式运行。它不提供任何身份验证、速率限制或访问控制。

**请仅将 Ragline server 部署在可信内网环境中。** 不要将其直接暴露到公共互联网，除非在前面部署了独立的认证代理（如 nginx + auth 模块）。

---

## Thread Safety

- **`RAG.query()`** 是线程安全的，可以从多个线程并发调用。
- **`RAG.ingest()`** 不应与 `query()` 并发执行。ingest 会修改向量存储和 BM25 索引，与查询并发运行可能导致结果不一致。
- **插件注册**（`registry.register()`）必须在第一次 `RAG()` 实例化之前完成。实例化后再注册新 handler 的行为不受支持。

---

## Known Limitations (MVP)

- **BM25 中文召回为 0**：内置 BM25 使用空白字符分词，对中文文本无法有效切词，检索召回率为 0。计划在 P1 阶段集成 jieba 分词器解决此问题。英文、日文（部分）及以空格分隔的语言不受影响。
- **大语料启动较慢**：语料超过 50k chunks 时，`RAG()` 初始化阶段的 BM25 索引重建（从向量存储批量拉取已有 chunks）可能需要数十秒。
- **不支持多 worker / 多 RAG 实例**：同一进程内仅支持单个 `RAG` 实例；多进程部署时各进程的 BM25 索引互相独立，无法同步。

---

## License

MIT

---

## Links

- 设计规范：[docs/specs/ragline-design.md](docs/specs/ragline-design.md)
- 开发计划：[docs/specs/ragline-plan.md](docs/specs/ragline-plan.md)
