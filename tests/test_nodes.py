"""tests/test_nodes.py — nodes.py 单元测试（TDD RED 阶段）。"""

from __future__ import annotations

from dataclasses import asdict

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Send

import rag.handlers  # noqa: F401 — 确保所有真实 handler 在 fixture 捕获状态前已注册
from rag.config import GraphConfig
from rag.registry import (
    generator_registry,
    grader_registry,
    processor_registry,
    retriever_registry,
    transform_registry,
)
from rag.types import Document, TransformResult

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def make_config(cfg: GraphConfig) -> RunnableConfig:
    return {"configurable": asdict(cfg)}


def make_doc(doc_id: str, source: str, score: float = 0.9) -> Document:
    return {
        "doc_id": doc_id,
        "content": f"content_{doc_id}",
        "score": score,
        "source": source,
        "metadata": {},
    }


# ---------------------------------------------------------------------------
# fixture：注册 fake handlers，测试后清理
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=False)
def fake_handlers():
    """为每个测试注册 fake handler，测试结束后清理。"""
    # 保存原始状态
    original = {
        "transform": dict(transform_registry._handlers),
        "retriever": dict(retriever_registry._handlers),
        "processor": dict(processor_registry._handlers),
        "grader": dict(grader_registry._handlers),
        "generator": dict(generator_registry._handlers),
    }

    # 注册 fake handlers
    transform_registry.register(
        "fake_rewrite", lambda queries, n, **kw: [f"rewritten_{q}" for q in queries]
    )
    transform_registry.register(
        "fake_transform_result",
        lambda queries, n, **kw: [
            TransformResult(query=f"tr_{q}", metadata_filter={"k": "v"}) for q in queries
        ],
    )
    # pipeline 步骤 A：追加 _A
    transform_registry.register(
        "pipeline_step_a", lambda queries, n, **kw: [f"{q}_A" for q in queries]
    )
    # pipeline 步骤 B：追加 _B
    transform_registry.register(
        "pipeline_step_b", lambda queries, n, **kw: [f"{q}_B" for q in queries]
    )

    retriever_registry.register(
        "fake_vector", lambda query, **kw: [make_doc("d1", "fake_vector", 0.9)]
    )
    retriever_registry.register("fake_es", lambda query, **kw: [make_doc("d2", "fake_es", 0.8)])

    processor_registry.register("fake_rerank", lambda docs, **kw: docs)

    grader_registry.register("fake_score", lambda docs, threshold, **kw: (0.8, "sufficient"))
    grader_registry.register(
        "fake_insufficient", lambda docs, threshold, **kw: (0.2, "insufficient")
    )

    generator_registry.register("fake_gen", lambda docs, query, **kw: f"answer: {query}")

    yield

    # 恢复原始状态
    transform_registry._handlers = original["transform"]
    retriever_registry._handlers = original["retriever"]
    processor_registry._handlers = original["processor"]
    grader_registry._handlers = original["grader"]
    generator_registry._handlers = original["generator"]


# ---------------------------------------------------------------------------
# 1. get_config
# ---------------------------------------------------------------------------


class TestGetConfig:
    def test_extracts_known_fields(self):
        from rag.nodes import get_config

        cfg = GraphConfig(top_k=5, temperature=0.3)
        config: RunnableConfig = {"configurable": asdict(cfg)}
        result = get_config(config)
        assert result.top_k == 5
        assert result.temperature == 0.3

    def test_filters_unknown_framework_keys(self):
        from rag.nodes import get_config

        cfg = GraphConfig(top_k=7)
        raw = asdict(cfg)
        raw["thread_id"] = "abc-123"
        raw["checkpoint_ns"] = "ns"
        config: RunnableConfig = {"configurable": raw}
        result = get_config(config)
        assert result.top_k == 7
        # 框架键不应导致 TypeError
        assert not hasattr(result, "thread_id")

    def test_empty_configurable_returns_default(self):
        from rag.nodes import get_config

        config: RunnableConfig = {"configurable": {}}
        result = get_config(config)
        assert isinstance(result, GraphConfig)
        assert result.top_k == 3  # 默认值
        assert result.temperature == 0.7


# ---------------------------------------------------------------------------
# 2. _run_transform_branch
# ---------------------------------------------------------------------------


class TestRunTransformBranch:
    def test_str_branch_calls_single_handler(self, fake_handlers):
        from rag.nodes import _run_transform_branch

        results = _run_transform_branch("fake_rewrite", ["q1", "q2"], n=2)
        assert results == ["rewritten_q1", "rewritten_q2"]

    def test_str_branch_transform_result(self, fake_handlers):
        from rag.nodes import _run_transform_branch

        results = _run_transform_branch("fake_transform_result", ["q1"], n=1)
        assert len(results) == 1
        assert isinstance(results[0], TransformResult)
        assert results[0].query == "tr_q1"
        assert results[0].metadata_filter == {"k": "v"}

    def test_list_branch_pipeline_executes_in_order(self, fake_handlers):
        from rag.nodes import _run_transform_branch

        # pipeline_step_a: q -> q_A, pipeline_step_b: q_A -> q_A_B
        results = _run_transform_branch(["pipeline_step_a", "pipeline_step_b"], ["q1"], n=1)
        assert results == ["q1_A_B"]

    def test_list_branch_single_step(self, fake_handlers):
        from rag.nodes import _run_transform_branch

        results = _run_transform_branch(["fake_rewrite"], ["hello"], n=1)
        assert results == ["rewritten_hello"]


# ---------------------------------------------------------------------------
# 3. query_transform
# ---------------------------------------------------------------------------


class TestQueryTransform:
    def test_returns_command(self, fake_handlers):
        from rag.nodes import query_transform

        cfg = GraphConfig(
            transforms=["fake_rewrite"],
            n_rewrites=1,
            retrievers=["fake_vector"],
            max_queries=10,
        )
        state = {
            "original_query": "test query",
            "source": "kb",
            "retry_count": 0,
        }
        result = query_transform(state, make_config(cfg))
        assert isinstance(result, Command)

    def test_command_update_has_rewritten_queries(self, fake_handlers):
        from rag.nodes import query_transform

        cfg = GraphConfig(
            transforms=["fake_rewrite"],
            n_rewrites=1,
            retrievers=["fake_vector"],
            max_queries=10,
        )
        state = {
            "original_query": "original",
            "source": "kb",
        }
        result = query_transform(state, make_config(cfg))
        assert "rewritten_queries" in result.update
        queries = result.update["rewritten_queries"]
        # 原始 query + rewritten
        assert "original" in queries
        assert "rewritten_original" in queries

    def test_command_goto_is_send_list(self, fake_handlers):
        from rag.nodes import query_transform

        cfg = GraphConfig(
            transforms=["fake_rewrite"],
            n_rewrites=1,
            retrievers=["fake_vector"],
            max_queries=10,
        )
        state = {"original_query": "q", "source": "kb"}
        result = query_transform(state, make_config(cfg))
        assert isinstance(result.goto, list)
        for item in result.goto:
            assert isinstance(item, Send)

    def test_send_count_equals_queries_times_retrievers(self, fake_handlers):
        from rag.nodes import query_transform

        cfg = GraphConfig(
            transforms=["fake_rewrite"],
            n_rewrites=1,
            retrievers=["fake_vector", "fake_es"],
            max_queries=10,
        )
        state = {"original_query": "q", "source": "kb"}
        result = query_transform(state, make_config(cfg))
        # queries = ["q", "rewritten_q"] (2 unique), retrievers = 2 → 4 sends
        assert len(result.goto) == 4

    def test_max_queries_truncation(self, fake_handlers):
        from rag.nodes import query_transform

        # max_queries=1 → 只保留原始 query，1 query × 2 retrievers = 2 sends
        cfg = GraphConfig(
            transforms=["fake_rewrite"],
            n_rewrites=1,
            retrievers=["fake_vector", "fake_es"],
            max_queries=1,
        )
        state = {"original_query": "q", "source": "kb"}
        result = query_transform(state, make_config(cfg))
        assert len(result.goto) == 2  # 1 query × 2 retrievers

    def test_fallback_source_uses_only_that_retriever(self, fake_handlers):
        from rag.nodes import query_transform

        cfg = GraphConfig(
            transforms=["fake_rewrite"],
            n_rewrites=1,
            retrievers=["fake_vector", "fake_es"],
            max_queries=10,
        )
        state = {"original_query": "q", "source": "fake_es"}
        result = query_transform(state, make_config(cfg))
        # source != "kb" → 只用 "fake_es" retriever
        # 2 queries × 1 retriever = 2 sends
        for send in result.goto:
            assert send.node == "retrieve"
            assert send.arg["retriever_name"] == "fake_es"

    def test_no_transforms_uses_original_query_only(self, fake_handlers):
        from rag.nodes import query_transform

        cfg = GraphConfig(
            transforms=[],
            n_rewrites=1,
            retrievers=["fake_vector"],
            max_queries=10,
        )
        state = {"original_query": "only_this", "source": "kb"}
        result = query_transform(state, make_config(cfg))
        queries = result.update["rewritten_queries"]
        assert queries == ["only_this"]

    def test_command_update_has_documents_overwrite(self, fake_handlers):
        from langgraph.types import Overwrite

        from rag.nodes import query_transform

        cfg = GraphConfig(
            transforms=["fake_rewrite"],
            n_rewrites=1,
            retrievers=["fake_vector"],
            max_queries=10,
        )
        state = {"original_query": "q", "source": "kb"}
        result = query_transform(state, make_config(cfg))
        assert "documents" in result.update
        assert isinstance(result.update["documents"], Overwrite)

    def test_deduplicates_queries(self, fake_handlers):
        from rag.nodes import query_transform

        # 注册一个返回重复 query 的 transform
        transform_registry.register(
            "dup_transform",
            lambda queries, n, **kw: list(queries),  # 返回与输入相同
        )
        cfg = GraphConfig(
            transforms=["dup_transform"],
            n_rewrites=1,
            retrievers=["fake_vector"],
            max_queries=10,
        )
        state = {"original_query": "same", "source": "kb"}
        result = query_transform(state, make_config(cfg))
        queries = result.update["rewritten_queries"]
        assert len(queries) == len(set(queries)), "queries 应去重"


# ---------------------------------------------------------------------------
# 4. retrieve
# ---------------------------------------------------------------------------


class TestRetrieve:
    def test_calls_retriever_and_returns_docs(self, fake_handlers):
        from rag.nodes import retrieve

        state = {
            "query": "test",
            "retriever_name": "fake_vector",
            "source": "kb",
            "metadata_filter": None,
        }
        result = retrieve(state)
        assert "documents" in result
        assert len(result["documents"]) == 1
        assert result["documents"][0]["doc_id"] == "d1"

    def test_sets_source_to_retriever_name(self, fake_handlers):
        from rag.nodes import retrieve

        state = {
            "query": "test",
            "retriever_name": "fake_vector",
            "source": "kb",
            "metadata_filter": None,
        }
        result = retrieve(state)
        for doc in result["documents"]:
            assert doc["source"] == "fake_vector"

    def test_metadata_filter_none_works(self, fake_handlers):
        from rag.nodes import retrieve

        state = {
            "query": "anything",
            "retriever_name": "fake_es",
            "source": "kb",
            "metadata_filter": None,
        }
        result = retrieve(state)
        assert isinstance(result["documents"], list)

    def test_metadata_filter_dict_passed_through(self, fake_handlers):
        """验证 metadata_filter 能传入，不抛异常。"""
        from rag.nodes import retrieve

        # 注册能接受 metadata_filter 的 retriever
        captured = {}
        retriever_registry.register(
            "cap_retriever",
            lambda query, metadata_filter=None, **kw: (
                captured.update({"mf": metadata_filter}) or []
            ),
        )
        state = {
            "query": "q",
            "retriever_name": "cap_retriever",
            "source": "kb",
            "metadata_filter": {"year": 2024},
        }
        retrieve(state)
        assert captured["mf"] == {"year": 2024}

    def test_missing_metadata_filter_key_defaults_to_none(self, fake_handlers):
        from rag.nodes import retrieve

        # RetrieveInput 无 metadata_filter 键时应优雅处理
        state = {
            "query": "q",
            "retriever_name": "fake_vector",
            "source": "kb",
            # metadata_filter 不存在
        }
        result = retrieve(state)
        assert "documents" in result


# ---------------------------------------------------------------------------
# 5. grade
# ---------------------------------------------------------------------------


class TestGrade:
    def test_writes_grade_score_and_decision(self, fake_handlers):
        from rag.nodes import grade

        cfg = GraphConfig(grade_strategy="fake_score", grade_threshold=0.5)
        docs = [make_doc("d1", "fake_vector")]
        state = {
            "original_query": "q",
            "documents": docs,
            "retrieval_history": [],
        }
        result = grade(state, make_config(cfg))
        assert result["grade_score"] == 0.8
        assert result["grade_decision"] == "sufficient"

    def test_appends_to_retrieval_history(self, fake_handlers):
        from rag.nodes import grade

        cfg = GraphConfig(grade_strategy="fake_score", grade_threshold=0.5)
        docs = [make_doc("d1", "fake_vector")]
        state = {
            "original_query": "q",
            "documents": docs,
            "retrieval_history": [],
        }
        result = grade(state, make_config(cfg))
        assert len(result["retrieval_history"]) == 1
        assert result["retrieval_history"][0] == docs

    def test_appends_to_existing_retrieval_history(self, fake_handlers):
        from rag.nodes import grade

        cfg = GraphConfig(grade_strategy="fake_score", grade_threshold=0.5)
        prev_docs = [make_doc("prev", "fake_es")]
        docs = [make_doc("d1", "fake_vector")]
        state = {
            "original_query": "q",
            "documents": docs,
            "retrieval_history": [prev_docs],
        }
        result = grade(state, make_config(cfg))
        assert len(result["retrieval_history"]) == 2
        assert result["retrieval_history"][0] == prev_docs
        assert result["retrieval_history"][1] == docs

    def test_grade_insufficient(self, fake_handlers):
        from rag.nodes import grade

        cfg = GraphConfig(grade_strategy="fake_insufficient", grade_threshold=0.5)
        state = {
            "original_query": "q",
            "documents": [make_doc("d1", "fake_vector")],
            "retrieval_history": [],
        }
        result = grade(state, make_config(cfg))
        assert result["grade_decision"] == "insufficient"
        assert result["grade_score"] == 0.2


# ---------------------------------------------------------------------------
# 6. route_after_grade
# ---------------------------------------------------------------------------


class TestRouteAfterGrade:
    def test_sufficient_goes_to_post_process(self):
        from rag.nodes import route_after_grade

        cfg = GraphConfig(fallback_chain=["web"], max_retries=1)
        state = {"grade_decision": "sufficient", "retry_count": 0}
        assert route_after_grade(state, make_config(cfg)) == "post_process"

    def test_insufficient_no_fallback_chain_goes_to_post_process(self):
        from rag.nodes import route_after_grade

        cfg = GraphConfig(fallback_chain=[], max_retries=1)
        state = {"grade_decision": "insufficient", "retry_count": 0}
        assert route_after_grade(state, make_config(cfg)) == "post_process"

    def test_insufficient_retry_count_exceeds_fallback_chain_goes_to_post_process(
        self,
    ):
        from rag.nodes import route_after_grade

        cfg = GraphConfig(fallback_chain=["web"], max_retries=3)
        state = {"grade_decision": "insufficient", "retry_count": 1}
        # retry_count(1) >= len(fallback_chain)(1) → post_process
        assert route_after_grade(state, make_config(cfg)) == "post_process"

    def test_insufficient_max_retries_exceeded_goes_to_post_process(self):
        from rag.nodes import route_after_grade

        cfg = GraphConfig(fallback_chain=["web", "kg", "sql"], max_retries=1)
        state = {"grade_decision": "insufficient", "retry_count": 1}
        # retry_count(1) >= max_retries(1) → post_process
        assert route_after_grade(state, make_config(cfg)) == "post_process"

    def test_insufficient_with_retries_available_goes_to_fallback(self):
        from rag.nodes import route_after_grade

        cfg = GraphConfig(fallback_chain=["web"], max_retries=1)
        state = {"grade_decision": "insufficient", "retry_count": 0}
        # retry_count(0) < len(fallback_chain)(1) AND < max_retries(1)
        assert route_after_grade(state, make_config(cfg)) == "fallback_to_query_transform"

    def test_default_retry_count_zero(self):
        from rag.nodes import route_after_grade

        cfg = GraphConfig(fallback_chain=["web"], max_retries=1)
        # state 中没有 retry_count → 默认 0
        state = {"grade_decision": "insufficient"}
        assert route_after_grade(state, make_config(cfg)) == "fallback_to_query_transform"


# ---------------------------------------------------------------------------
# 7. prepare_fallback
# ---------------------------------------------------------------------------


class TestPrepareFallback:
    def test_first_fallback_sets_source_and_increments_retry(self):
        from rag.nodes import prepare_fallback

        cfg = GraphConfig(fallback_chain=["web"])
        state = {"retry_count": 0}
        result = prepare_fallback(state, make_config(cfg))
        assert result["source"] == "web"
        assert result["retry_count"] == 1

    def test_second_fallback_uses_next_source(self):
        from rag.nodes import prepare_fallback

        cfg = GraphConfig(fallback_chain=["web", "kg"])
        state = {"retry_count": 1}
        result = prepare_fallback(state, make_config(cfg))
        assert result["source"] == "kg"
        assert result["retry_count"] == 2

    def test_default_retry_count_zero(self):
        from rag.nodes import prepare_fallback

        cfg = GraphConfig(fallback_chain=["web"])
        state = {}  # retry_count 不存在
        result = prepare_fallback(state, make_config(cfg))
        assert result["source"] == "web"
        assert result["retry_count"] == 1


# ---------------------------------------------------------------------------
# 8. post_process
# ---------------------------------------------------------------------------


class TestPostProcess:
    def test_applies_processors_and_returns_ranked_docs(self, fake_handlers):
        from rag.nodes import post_process

        cfg = GraphConfig(processors=["fake_rerank"], top_k=10)
        docs = [make_doc("d1", "fake_vector"), make_doc("d2", "fake_es")]
        state = {
            "original_query": "q",
            "documents": docs,
            "retrieval_history": [docs],
        }
        result = post_process(state, make_config(cfg))
        assert "ranked_documents" in result
        assert len(result["ranked_documents"]) == 2

    def test_top_k_truncation(self, fake_handlers):
        from rag.nodes import post_process

        cfg = GraphConfig(processors=["fake_rerank"], top_k=2)
        docs = [
            make_doc("d1", "fake_vector"),
            make_doc("d2", "fake_es"),
            make_doc("d3", "fake_vector"),
            make_doc("d4", "fake_es"),
        ]
        state = {
            "original_query": "q",
            "documents": docs,
            "retrieval_history": [docs],
        }
        result = post_process(state, make_config(cfg))
        assert len(result["ranked_documents"]) == 2

    def test_merges_retrieval_history_excluding_last_round(self, fake_handlers):
        from rag.nodes import post_process

        cfg = GraphConfig(processors=["fake_rerank"], top_k=100)
        prev_docs = [make_doc("prev1", "fake_vector", 0.7)]
        current_docs = [make_doc("curr1", "fake_es", 0.9)]
        state = {
            "original_query": "q",
            "documents": current_docs,
            "retrieval_history": [prev_docs, current_docs],
        }
        result = post_process(state, make_config(cfg))
        doc_ids = {d["doc_id"] for d in result["ranked_documents"]}
        assert "prev1" in doc_ids
        assert "curr1" in doc_ids

    def test_deduplicates_by_source_and_doc_id(self, fake_handlers):
        from rag.nodes import post_process

        cfg = GraphConfig(processors=["fake_rerank"], top_k=100)
        doc_a = make_doc("d1", "fake_vector", 0.9)
        doc_b = make_doc("d1", "fake_vector", 0.8)  # 同一 (source, doc_id)
        state = {
            "original_query": "q",
            "documents": [doc_a],
            "retrieval_history": [[doc_b], [doc_a]],
        }
        result = post_process(state, make_config(cfg))
        count = sum(
            1
            for d in result["ranked_documents"]
            if d["doc_id"] == "d1" and d["source"] == "fake_vector"
        )
        assert count == 1

    def test_empty_documents_returns_empty(self, fake_handlers):
        from rag.nodes import post_process

        cfg = GraphConfig(processors=["fake_rerank"], top_k=3)
        state = {
            "original_query": "q",
            "documents": [],
            "retrieval_history": [[]],
        }
        result = post_process(state, make_config(cfg))
        assert result["ranked_documents"] == []

    def test_no_processors_still_truncates(self, fake_handlers):
        """processors 为空列表时，仍然按 top_k 截断。"""
        from rag.nodes import post_process

        cfg = GraphConfig(processors=[], top_k=1)
        docs = [make_doc("d1", "fake_vector"), make_doc("d2", "fake_es")]
        state = {
            "original_query": "q",
            "documents": docs,
            "retrieval_history": [docs],
        }
        result = post_process(state, make_config(cfg))
        assert len(result["ranked_documents"]) == 1


# ---------------------------------------------------------------------------
# 9. generate
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_writes_answer_to_state(self, fake_handlers):
        from rag.nodes import generate

        cfg = GraphConfig(gen_mode="fake_gen")
        docs = [make_doc("d1", "fake_vector")]
        state = {
            "original_query": "what is X",
            "ranked_documents": docs,
        }
        result = generate(state, make_config(cfg))
        assert "answer" in result
        assert result["answer"] == "answer: what is X"

    def test_generate_with_empty_docs(self, fake_handlers):
        from rag.nodes import generate

        cfg = GraphConfig(gen_mode="fake_gen")
        state = {
            "original_query": "empty test",
            "ranked_documents": [],
        }
        result = generate(state, make_config(cfg))
        assert "answer" in result
        assert isinstance(result["answer"], str)

    def test_generator_receives_ranked_documents(self, fake_handlers):
        """验证 generator 收到 ranked_documents 而不是原始 documents。"""
        from rag.nodes import generate

        received_docs: list = []
        generator_registry.register(
            "capture_gen",
            lambda docs, query, **kw: received_docs.extend(docs) or "captured",
        )
        cfg = GraphConfig(gen_mode="capture_gen")
        ranked = [make_doc("r1", "fake_vector"), make_doc("r2", "fake_es")]
        state = {
            "original_query": "q",
            "ranked_documents": ranked,
        }
        generate(state, make_config(cfg))
        assert len(received_docs) == 2
        assert received_docs[0]["doc_id"] == "r1"
