# Ragline Quickstart 示例

## 如何运行

确保已安装依赖后，在项目根目录执行：

```bash
uv run python examples/quickstart/consumer_minimal.py
```

脚本使用内置 mock providers（无需真实 API key），会输出两行：

```
INGESTED: <n> chunks
ANSWER: Ragline is a config-driven RAG library.
```

其中 `INGESTED` 行显示从 `docs/` 目录读取并切分的 chunk 数量，`ANSWER` 行来自 mock LLM 的固定回答。

## 换成真实 Provider

如需对接真实 OpenAI，移除脚本中的 `mock_rag_providers`，改用 `consumer.yaml` 加载配置：

1. 设置环境变量：
   ```bash
   export OPENAI_API_KEY=sk-...
   ```

2. 将脚本中的实例化改为 `from_yaml`：
   ```python
   from pathlib import Path
   from ragline import RAG

   yaml_path = Path(__file__).parent / "consumer.yaml"
   with RAG.from_yaml(yaml_path) as rag:
       result = rag.ingest(Path(__file__).parent / "docs", show_progress=True)
       print(f"INGESTED: {result.num_chunks} chunks")
       query_result = rag.query("What is Ragline?")
       print(f"ANSWER: {query_result.answer}")
   ```

`consumer.yaml` 中 `${OPENAI_API_KEY}` 占位符会被 ragline 自动展开为环境变量值。

## 这个示例做了什么

脚本的完整流程：

1. **构建配置**：用 `RaglineConfig` dataclass 指定向量检索（无 BM25、无 reranker），将 chroma 持久化目录设为临时目录。
2. **Mock providers**：用 `mock_rag_providers()` patch 掉 `LLMClient`、`EmbeddingClient` 和 `ChromaVectorStore`，同时隔离全局 registry，避免跨进程污染。
3. **Ingest**：调用 `rag.ingest(docs/)` 读取 `docs/doc1.md`、`docs/doc2.md`、`docs/doc3.md`，经解析、切分、（fake）embedding 后写入 mock vector store。
4. **Query**：调用 `rag.query("What is Ragline?")` 走完完整 RAG 图（transform → retrieve → grade → generate），mock LLM 返回固定字符串。
5. **输出**：打印 ingest chunk 数和查询答案。
