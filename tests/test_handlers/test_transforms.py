"""tests/test_handlers/test_transforms.py — transforms handler 测试。"""

import json

from rag.types import TransformResult

# ---------------------------------------------------------------------------
# FakeLLM stub
# ---------------------------------------------------------------------------


class FakeLLM:
    """测试用 LLM stub，每次 invoke 返回预设响应。"""

    def __init__(self, response: str) -> None:
        self._response = response
        self.call_count = 0
        self.last_prompt: str | None = None

    def invoke(self, prompt: str, temperature: float | None = None) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        return self._response


# ---------------------------------------------------------------------------
# _parse_numbered_lines 测试
# ---------------------------------------------------------------------------


class TestParseNumberedLines:
    def test_parses_standard_numbered_list(self) -> None:
        from rag.handlers.transforms import _parse_numbered_lines

        result = _parse_numbered_lines("1. a\n2. b\n3. c")
        assert result == ["a", "b", "c"]

    def test_empty_string_returns_empty_list(self) -> None:
        from rag.handlers.transforms import _parse_numbered_lines

        result = _parse_numbered_lines("")
        assert result == []

    def test_non_numbered_lines_returns_empty_list(self) -> None:
        from rag.handlers.transforms import _parse_numbered_lines

        result = _parse_numbered_lines("- a\n- b")
        assert result == []

    def test_strips_whitespace_from_items(self) -> None:
        from rag.handlers.transforms import _parse_numbered_lines

        result = _parse_numbered_lines("1.  hello world  \n2.  foo bar  ")
        assert result == ["hello world", "foo bar"]

    def test_skips_blank_lines(self) -> None:
        from rag.handlers.transforms import _parse_numbered_lines

        result = _parse_numbered_lines("1. a\n\n2. b")
        assert result == ["a", "b"]

    def test_handles_only_whitespace(self) -> None:
        from rag.handlers.transforms import _parse_numbered_lines

        result = _parse_numbered_lines("   \n  ")
        assert result == []


# ---------------------------------------------------------------------------
# rewrite_fn 测试
# ---------------------------------------------------------------------------


class TestRewriteFn:
    def test_returns_parsed_lines_single_query(self) -> None:
        from rag.handlers.transforms import rewrite_fn

        llm = FakeLLM("1. a\n2. b\n3. c")
        result = rewrite_fn(["q"], n=3, llm_client=llm)
        assert result == ["a", "b", "c"]

    def test_multiple_queries_total_output_n_times_len(self) -> None:
        from rag.handlers.transforms import rewrite_fn

        # FakeLLM 每次返回 3 行，2 个 queries → 总输出 6 个
        llm = FakeLLM("1. x\n2. y\n3. z")
        result = rewrite_fn(["q1", "q2"], n=3, llm_client=llm)
        assert len(result) == 6
        assert llm.call_count == 2

    def test_empty_queries_returns_empty(self) -> None:
        from rag.handlers.transforms import rewrite_fn

        llm = FakeLLM("1. a")
        result = rewrite_fn([], n=3, llm_client=llm)
        assert result == []
        assert llm.call_count == 0

    def test_llm_returns_empty_produces_empty_for_that_query(self) -> None:
        from rag.handlers.transforms import rewrite_fn

        llm = FakeLLM("")
        result = rewrite_fn(["q"], n=3, llm_client=llm)
        assert result == []

    def test_prompt_contains_query(self) -> None:
        from rag.handlers.transforms import rewrite_fn

        llm = FakeLLM("1. rewritten")
        rewrite_fn(["my special query"], n=1, llm_client=llm)
        assert llm.last_prompt is not None
        assert "my special query" in llm.last_prompt


# ---------------------------------------------------------------------------
# decompose_fn 测试
# ---------------------------------------------------------------------------


class TestDecomposeFn:
    def test_returns_sub_questions(self) -> None:
        from rag.handlers.transforms import decompose_fn

        llm = FakeLLM("1. sub1\n2. sub2")
        result = decompose_fn(["complex question"], n=2, llm_client=llm)
        assert result == ["sub1", "sub2"]

    def test_multiple_queries(self) -> None:
        from rag.handlers.transforms import decompose_fn

        llm = FakeLLM("1. sub1\n2. sub2")
        result = decompose_fn(["q1", "q2"], n=2, llm_client=llm)
        assert len(result) == 4

    def test_empty_queries(self) -> None:
        from rag.handlers.transforms import decompose_fn

        llm = FakeLLM("1. sub1")
        result = decompose_fn([], n=2, llm_client=llm)
        assert result == []


# ---------------------------------------------------------------------------
# hyde_fn 测试
# ---------------------------------------------------------------------------


class TestHydeFn:
    def test_returns_hypothetical_documents(self) -> None:
        from rag.handlers.transforms import hyde_fn

        llm = FakeLLM("1. hypothetical doc 1\n2. hypothetical doc 2")
        result = hyde_fn(["what is ML?"], n=2, llm_client=llm)
        assert result == ["hypothetical doc 1", "hypothetical doc 2"]

    def test_prompt_contains_query(self) -> None:
        from rag.handlers.transforms import hyde_fn

        llm = FakeLLM("1. doc")
        hyde_fn(["machine learning basics"], n=1, llm_client=llm)
        assert llm.last_prompt is not None
        assert "machine learning basics" in llm.last_prompt


# ---------------------------------------------------------------------------
# step_back_fn 测试
# ---------------------------------------------------------------------------


class TestStepBackFn:
    def test_returns_abstract_queries(self) -> None:
        from rag.handlers.transforms import step_back_fn

        llm = FakeLLM("1. abstract query")
        result = step_back_fn(["specific detail question"], n=1, llm_client=llm)
        assert result == ["abstract query"]

    def test_prompt_contains_query(self) -> None:
        from rag.handlers.transforms import step_back_fn

        llm = FakeLLM("1. broader q")
        step_back_fn(["narrow query"], n=1, llm_client=llm)
        assert llm.last_prompt is not None
        assert "narrow query" in llm.last_prompt


# ---------------------------------------------------------------------------
# self_query_fn 测试
# ---------------------------------------------------------------------------


class TestSelfQueryFn:
    def test_returns_list_of_transform_results(self) -> None:
        from rag.handlers.transforms import self_query_fn

        payload = json.dumps({"query": "sales", "filter": {"year": 2024}})
        llm = FakeLLM(payload)
        results = self_query_fn(["revenue in 2024"], n=1, llm_client=llm)
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], TransformResult)

    def test_result_has_query_field(self) -> None:
        from rag.handlers.transforms import self_query_fn

        payload = json.dumps({"query": "sales", "filter": {"year": 2024}})
        llm = FakeLLM(payload)
        results = self_query_fn(["find sales"], n=1, llm_client=llm)
        assert results[0].query == "sales"

    def test_result_has_metadata_filter(self) -> None:
        from rag.handlers.transforms import self_query_fn

        payload = json.dumps({"query": "sales", "filter": {"year": 2024}})
        llm = FakeLLM(payload)
        results = self_query_fn(["find sales"], n=1, llm_client=llm)
        assert results[0].metadata_filter == {"year": 2024}

    def test_invalid_json_returns_empty_list(self) -> None:
        from rag.handlers.transforms import self_query_fn

        llm = FakeLLM("not valid json at all")
        results = self_query_fn(["some query"], n=1, llm_client=llm)
        assert results == []

    def test_multiple_queries_produces_multiple_results(self) -> None:
        from rag.handlers.transforms import self_query_fn

        payload = json.dumps({"query": "foo", "filter": None})
        llm = FakeLLM(payload)
        results = self_query_fn(["q1", "q2"], n=1, llm_client=llm)
        assert len(results) == 2

    def test_missing_query_key_returns_empty(self) -> None:
        from rag.handlers.transforms import self_query_fn

        payload = json.dumps({"filter": {"year": 2024}})
        llm = FakeLLM(payload)
        results = self_query_fn(["find sales"], n=1, llm_client=llm)
        assert results == []

    def test_empty_queries_returns_empty(self) -> None:
        from rag.handlers.transforms import self_query_fn

        llm = FakeLLM("{}")
        results = self_query_fn([], n=1, llm_client=llm)
        assert results == []


# ---------------------------------------------------------------------------
# 注册验证
# ---------------------------------------------------------------------------


class TestTransformRegistration:
    def test_all_transforms_registered(self) -> None:
        import rag.handlers.transforms  # noqa: F401
        from rag.registry import transform_registry

        for name in ["rewrite", "decompose", "hyde", "step_back", "self_query"]:
            assert transform_registry.has(name), f"'{name}' not registered"
