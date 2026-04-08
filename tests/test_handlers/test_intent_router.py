"""tests/test_handlers/test_intent_router.py — intent_router handler 测试。"""


# ---------------------------------------------------------------------------
# FakeLLM stub
# ---------------------------------------------------------------------------


class FakeLLM:
    def __init__(self, response: str) -> None:
        self._response = response
        self.call_count = 0
        self.last_prompt: str | None = None

    def invoke(self, prompt: str, temperature: float | None = None) -> str:
        self.call_count += 1
        self.last_prompt = prompt
        return self._response


# ---------------------------------------------------------------------------
# intent_router 测试
# ---------------------------------------------------------------------------


class TestIntentRouter:
    def test_returns_subset_of_available_retrievers(self) -> None:
        from rag.handlers.intent_router import intent_router

        llm = FakeLLM('["vector"]')
        result = intent_router("some query", ["vector", "es"], llm_client=llm)
        assert result == ["vector"]

    def test_filters_out_unavailable_retrievers(self) -> None:
        from rag.handlers.intent_router import intent_router

        llm = FakeLLM('["vector", "nonexistent"]')
        result = intent_router("some query", ["vector", "es"], llm_client=llm)
        assert result == ["vector"]

    def test_invalid_json_fallbacks_to_full_set(self) -> None:
        from rag.handlers.intent_router import intent_router

        llm = FakeLLM("not json at all")
        result = intent_router("some query", ["vector", "es"], llm_client=llm)
        assert set(result) == {"vector", "es"}

    def test_all_results_unavailable_fallbacks_to_full_set(self) -> None:
        from rag.handlers.intent_router import intent_router

        llm = FakeLLM('["nonexistent1", "nonexistent2"]')
        result = intent_router("some query", ["vector", "es"], llm_client=llm)
        assert set(result) == {"vector", "es"}

    def test_multiple_valid_retrievers(self) -> None:
        from rag.handlers.intent_router import intent_router

        llm = FakeLLM('["vector", "es"]')
        result = intent_router("some query", ["vector", "es", "kg"], llm_client=llm)
        assert set(result) == {"vector", "es"}

    def test_prompt_contains_query(self) -> None:
        from rag.handlers.intent_router import intent_router

        llm = FakeLLM('["vector"]')
        intent_router("my special query", ["vector"], llm_client=llm)
        assert llm.last_prompt is not None
        assert "my special query" in llm.last_prompt

    def test_prompt_contains_available_retrievers(self) -> None:
        from rag.handlers.intent_router import intent_router

        llm = FakeLLM('["vector"]')
        intent_router("q", ["vector", "es", "kg"], llm_client=llm)
        assert llm.last_prompt is not None
        assert "vector" in llm.last_prompt
        assert "es" in llm.last_prompt

    def test_empty_available_retrievers_returns_empty(self) -> None:
        from rag.handlers.intent_router import intent_router

        llm = FakeLLM('["vector"]')
        result = intent_router("q", [], llm_client=llm)
        assert result == []

    def test_returns_list_type(self) -> None:
        from rag.handlers.intent_router import intent_router

        llm = FakeLLM('["vector"]')
        result = intent_router("q", ["vector"], llm_client=llm)
        assert isinstance(result, list)

    def test_json_array_with_non_string_values_filtered(self) -> None:
        from rag.handlers.intent_router import intent_router

        # LLM 返回包含非字符串的 JSON 数组，过滤或 fallback
        llm = FakeLLM('[1, "vector", null]')
        result = intent_router("q", ["vector", "es"], llm_client=llm)
        # 过滤后只剩 "vector"
        assert "vector" in result
