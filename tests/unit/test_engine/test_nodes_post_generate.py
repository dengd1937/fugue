"""tests/unit/test_engine/test_nodes_post_generate.py — post_process + generate 测试。"""

from unittest.mock import MagicMock

import pytest

from ragline.api.types import Document
from ragline.engine.nodes.generate import generate
from ragline.engine.nodes.post_process import MAX_DOCS_BEFORE_PROCESS, post_process
from ragline.engine.state import RAGState
from ragline.registry import generator_registry, processor_registry


def _doc(source: str, doc_id: str, score: float = 0.9, content: str = "x") -> Document:
    return Document(doc_id=doc_id, content=content, score=score, source=source, metadata={})


def _state(
    *,
    docs: list[Document] | None = None,
    history: list[list[Document]] | None = None,
    ranked: list[Document] | None = None,
    query: str = "q",
) -> RAGState:
    return RAGState(
        original_query=query,
        rewritten_queries=[],
        documents=docs or [],
        grade_score=0.0,
        grade_decision="sufficient",
        source="kb",
        retry_count=0,
        retrieval_history=history or [],
        ranked_documents=ranked or [],
        answer="",
    )


@pytest.fixture
def clean_processor_registry():
    saved = {n: processor_registry.get(n) for n in processor_registry.names()}
    for n in list(processor_registry.names()):
        processor_registry.unregister(n)
    yield processor_registry
    for n in list(processor_registry.names()):
        processor_registry.unregister(n)
    for n, fn in saved.items():
        processor_registry.register(n, fn)


@pytest.fixture
def clean_generator_registry():
    saved = {n: generator_registry.get(n) for n in generator_registry.names()}
    for n in list(generator_registry.names()):
        generator_registry.unregister(n)
    yield generator_registry
    for n in list(generator_registry.names()):
        generator_registry.unregister(n)
    for n, fn in saved.items():
        generator_registry.register(n, fn)


# post_process 测试 -----------------------------------------------------


def test_post_process_cross_round_merge_dedup(clean_processor_registry) -> None:
    """state.documents=[D1] + retrieval_history=[[D2], [D1]]，
    history[:-1]=[D2] 合并后去重输出 [D1, D2]，processor 不变."""
    d1 = _doc("vector", "1")
    d2 = _doc("bm25", "2")
    # mock 处理器：identity
    identity = MagicMock(side_effect=lambda docs, q, k, **kw: docs)
    processor_registry.register("identity", identity)

    state = _state(docs=[d1], history=[[d2], [d1]])
    config = {
        "configurable": {
            "processors": ["identity"],
            "top_k": 10,
        }
    }
    result = post_process(state, config)  # type: ignore[arg-type]
    docs = result["ranked_documents"]
    # 当前轮 d1 + 历史 d2，d1 去重
    assert len(docs) == 2
    keys = {(d["source"], d["doc_id"]) for d in docs}
    assert keys == {("vector", "1"), ("bm25", "2")}


def test_post_process_chained_processors(clean_processor_registry) -> None:
    """cfg.processors=['p1', 'p2']，p1 输出喂 p2，按顺序调用。"""
    d1 = _doc("vector", "1")
    d2 = _doc("vector", "2")
    p1_out = [d1]  # p1 输出
    p2_out = [d2]  # p2 输出
    p1 = MagicMock(return_value=p1_out)
    p2 = MagicMock(return_value=p2_out)
    processor_registry.register("p1", p1)
    processor_registry.register("p2", p2)

    state = _state(docs=[d1, d2])
    config = {"configurable": {"processors": ["p1", "p2"], "top_k": 10}}
    result = post_process(state, config)  # type: ignore[arg-type]
    # p1 调用一次，传入合并后的 [d1, d2]
    p1.assert_called_once()
    # p2 调用一次，输入是 p1 的输出
    p2.assert_called_once()
    p2_args = p2.call_args
    assert p2_args.args[0] == p1_out  # p2 收到 p1 的输出
    assert result["ranked_documents"] == p2_out


def test_post_process_top_k_truncation(clean_processor_registry) -> None:
    """processor 返回 10 个 top_k=3 时截断到 3."""
    docs = [_doc("v", str(i), score=1.0 - i * 0.1) for i in range(10)]
    identity = MagicMock(side_effect=lambda d, q, k, **kw: d)
    processor_registry.register("identity", identity)

    state = _state(docs=docs)
    config = {"configurable": {"processors": ["identity"], "top_k": 3}}
    result = post_process(state, config)  # type: ignore[arg-type]
    assert len(result["ranked_documents"]) == 3


def test_post_process_empty_documents(clean_processor_registry) -> None:
    """无 documents 时返回 {ranked_documents: []}."""
    identity = MagicMock(side_effect=lambda d, q, k, **kw: d)
    processor_registry.register("identity", identity)

    state = _state(docs=[])
    config = {"configurable": {"processors": ["identity"], "top_k": 5}}
    result = post_process(state, config)  # type: ignore[arg-type]
    assert result["ranked_documents"] == []


def test_post_process_kwargs_passthrough(clean_processor_registry) -> None:
    """调用 processor 时 kwargs 含 retriever_weights/score_normalizers/top_k."""
    identity = MagicMock(side_effect=lambda d, q, k, **kw: d)
    processor_registry.register("identity", identity)

    state = _state(docs=[_doc("v", "1")])
    config = {
        "configurable": {
            "processors": ["identity"],
            "top_k": 7,
            "retriever_weights": {"vector": 1.0},
            "score_normalizers": {"bm25": 20.0},
        }
    }
    post_process(state, config)  # type: ignore[arg-type]
    call = identity.call_args
    # positional: (docs, query, top_k)
    assert call.args[2] == 7  # top_k
    assert call.kwargs["retriever_weights"] == {"vector": 1.0}
    assert call.kwargs["score_normalizers"] == {"bm25": 20.0}


def test_post_process_defensive_upper_bound(clean_processor_registry) -> None:
    """合并后 > MAX_DOCS_BEFORE_PROCESS 时按 score 降序截断。"""
    # 构造 2000 个 doc，score 递减
    docs = [_doc("v", str(i), score=1.0 - i * 0.0001) for i in range(2000)]
    captured = []

    def capture_fn(d, q, k, **kw):
        captured.append(len(d))
        return d

    processor_registry.register("capture", capture_fn)

    state = _state(docs=docs)
    config = {"configurable": {"processors": ["capture"], "top_k": 10}}
    post_process(state, config)  # type: ignore[arg-type]
    # processor 收到的 docs 应该是 MAX_DOCS_BEFORE_PROCESS 即 1000
    assert captured[0] == MAX_DOCS_BEFORE_PROCESS


# generate 测试 --------------------------------------------------------


def test_generate_basic_mode(clean_generator_registry) -> None:
    """gen_mode='basic' 时调 basic generator。"""
    mock_gen = MagicMock(return_value="basic answer")
    generator_registry.register("basic", mock_gen)

    state = _state(ranked=[_doc("v", "1")])
    config = {"configurable": {"gen_mode": "basic", "temperature": 0.7}}
    result = generate(state, config)  # type: ignore[arg-type]
    assert result == {"answer": "basic answer"}
    mock_gen.assert_called_once()


def test_generate_citation_mode(clean_generator_registry) -> None:
    """gen_mode='citation' 时调 citation generator。"""
    mock_gen = MagicMock(return_value="citation answer [1]")
    generator_registry.register("citation", mock_gen)

    state = _state(ranked=[_doc("v", "1")])
    config = {"configurable": {"gen_mode": "citation", "temperature": 0.5}}
    result = generate(state, config)  # type: ignore[arg-type]
    assert result == {"answer": "citation answer [1]"}


def test_generate_temperature_passthrough(clean_generator_registry) -> None:
    mock_gen = MagicMock(return_value="x")
    generator_registry.register("basic", mock_gen)

    state = _state(ranked=[_doc("v", "1")])
    config = {"configurable": {"gen_mode": "basic", "temperature": 0.1}}
    generate(state, config)  # type: ignore[arg-type]
    # generator 签名: (query, docs, temperature)
    call = mock_gen.call_args
    assert call.args[2] == 0.1  # temperature 是第 3 个位置参数
