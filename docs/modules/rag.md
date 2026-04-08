# rag 模块文档

## 概述

基于 LangGraph 的 7 节点 RAG 系统。核心设计哲学：**图拓扑不可变，行为全靠 GraphConfig 驱动**。

## 依赖拓扑

```
types.py → config.py → state.py → registry.py → handlers/* → nodes.py → graph.py → main.py
```

## 文件结构

| 文件 | 职责 |
|------|------|
| `types.py` | `Document`、`TransformResult`、`RetrieveInput` 类型定义 |
| `config.py` | `GraphConfig` dataclass，唯一配置入口 |
| `state.py` | `RAGState` TypedDict + `merge_docs` reducer |
| `registry.py` | `Registry` 类 + 5 个全局实例 |
| `handlers/transforms.py` | rewrite / decompose / hyde / step_back / self_query |
| `handlers/intent_router.py` | 意图路由（`route_strategy="intent"` 时启用） |
| `handlers/retrievers.py` | vector / es / kg / web / sql / parent_child / sentence_window |
| `handlers/processors.py` | rrf / weighted_fusion / rerank / filter / dedupe / compression |
| `handlers/graders.py` | score / llm / hybrid grader + _normalize_score |
| `handlers/generators.py` | basic / cot / citation 生成模式 |
| `nodes.py` | 6 个 LangGraph 节点函数 + `route_after_grade` 条件边 |
| `graph.py` | `build_rag_graph()` 图组装 |
| `main.py` | CLI 入口 |

## 图拓扑

```
query_transform ──Send()──► retrieve ──► grade
                                           │
                              ┌── sufficient ──► post_process ──► generate ──► END
                              └── insufficient ──► prepare_fallback ──► query_transform (循环)
```

## 扩展方式

### 新增检索器

```python
# 1. 实现 handler
def my_retriever(query: str, metadata_filter: dict | None = None) -> list[Document]: ...

# 2. 注册
from rag.registry import retriever_registry
retriever_registry.register("my_retriever", my_retriever)

# 3. 激活
config = GraphConfig(retrievers=["vector", "my_retriever"])
```

### 新增 transform

```python
from rag.registry import transform_registry
transform_registry.register("my_transform", lambda queries, n, **kw: [...])

# 并行用法
config = GraphConfig(transforms=["rewrite", "my_transform"])
# 串联用法
config = GraphConfig(transforms=[["my_transform", "rewrite"]])
```

### 新增后处理器

```python
from rag.registry import processor_registry
processor_registry.register("my_proc", lambda docs, query, top_k, **kw: docs)
config = GraphConfig(processors=["rrf", "my_proc", "rerank"])
```

## 调用示例

```python
from dataclasses import asdict
from rag.config import GraphConfig
from rag.graph import build_rag_graph

graph = build_rag_graph()
cfg = GraphConfig(transforms=["rewrite"], n_rewrites=3, retrievers=["vector", "es"], top_k=3)

result = graph.invoke(
    {"original_query": "问题", "retry_count": 0, "source": "kb"},
    config={"configurable": asdict(cfg)},
)
print(result["answer"])
```
