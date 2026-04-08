"""rag/main.py — RAG 系统入口。"""

from dataclasses import asdict

from dotenv import load_dotenv

from rag.config import GraphConfig
from rag.graph import build_rag_graph


def main() -> None:
    load_dotenv()

    graph = build_rag_graph()

    cfg = GraphConfig(
        transforms=["rewrite"],
        n_rewrites=1,
        retrievers=["vector", "es"],
        top_k=3,
    )

    result = graph.invoke(
        {
            "original_query": "LangGraph 如何实现动态路由？",
            "retry_count": 0,
            "source": "kb",
        },
        config={"configurable": asdict(cfg)},
    )
    print(result.get("answer", "（无回答）"))


if __name__ == "__main__":
    main()
