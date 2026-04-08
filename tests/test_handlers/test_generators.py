"""tests/test_handlers/test_generators.py — generators handler 测试。"""

from typing import Any

from rag.types import Document

# ---------------------------------------------------------------------------
# FakeLLM stub
# ---------------------------------------------------------------------------


class FakeLLM:
    def __init__(self, response: str) -> None:
        self._response = response
        self.last_prompt: str | None = None

    def invoke(self, prompt: str, temperature: float | None = None) -> str:
        self.last_prompt = prompt
        return self._response


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _doc(
    doc_id: str = "d1",
    content: str = "some content here",
    score: float = 0.9,
    source: str = "vector",
    metadata: dict[str, Any] | None = None,
) -> Document:
    return Document(
        doc_id=doc_id,
        content=content,
        score=score,
        source=source,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# build_prompt 测试
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_basic_mode_contains_content(self) -> None:
        from rag.handlers.generators import build_prompt

        doc = _doc(content="important context info")
        result = build_prompt("basic", "what is this?", [doc])
        assert "important context info" in result
        assert "what is this?" in result

    def test_basic_mode_contains_query(self) -> None:
        from rag.handlers.generators import build_prompt

        doc = _doc()
        result = build_prompt("basic", "my question", [doc])
        assert "my question" in result

    def test_citation_mode_contains_numbered_refs(self) -> None:
        from rag.handlers.generators import build_prompt

        doc1 = _doc("d1", content="first doc")
        doc2 = _doc("d2", content="second doc")
        result = build_prompt("citation", "q", [doc1, doc2])
        assert "[1]" in result
        assert "[2]" in result

    def test_citation_mode_contains_all_doc_contents(self) -> None:
        from rag.handlers.generators import build_prompt

        doc1 = _doc("d1", content="alpha content")
        doc2 = _doc("d2", content="beta content")
        result = build_prompt("citation", "q", [doc1, doc2])
        assert "alpha content" in result
        assert "beta content" in result

    def test_cot_mode_contains_reasoning_hint(self) -> None:
        from rag.handlers.generators import build_prompt

        doc = _doc()
        result = build_prompt("cot", "q", [doc])
        # CoT prompt 应包含推理/思考相关文字
        assert any(hint in result for hint in ["推理", "思考", "步骤", "step", "reason", "think"])

    def test_cot_mode_contains_query(self) -> None:
        from rag.handlers.generators import build_prompt

        doc = _doc()
        result = build_prompt("cot", "explain this", [doc])
        assert "explain this" in result

    def test_unknown_mode_falls_back_to_basic(self) -> None:
        from rag.handlers.generators import build_prompt

        doc = _doc(content="ctx")
        result = build_prompt("unknown_mode", "q", [doc])
        assert "ctx" in result
        assert "q" in result

    def test_empty_docs_still_returns_string(self) -> None:
        from rag.handlers.generators import build_prompt

        result = build_prompt("basic", "q", [])
        assert isinstance(result, str)
        assert "q" in result

    def test_single_doc_citation_has_only_ref_one(self) -> None:
        from rag.handlers.generators import build_prompt

        doc = _doc("d1")
        result = build_prompt("citation", "q", [doc])
        assert "[1]" in result
        assert "[2]" not in result


# ---------------------------------------------------------------------------
# basic_generate_fn 测试
# ---------------------------------------------------------------------------


class TestBasicGenerateFn:
    def test_returns_llm_response(self) -> None:
        from rag.handlers.generators import basic_generate_fn

        llm = FakeLLM("This is the answer.")
        docs = [_doc()]
        result = basic_generate_fn(docs, query="q", temperature=0.7, llm_client=llm)
        assert result == "This is the answer."

    def test_prompt_contains_doc_content(self) -> None:
        from rag.handlers.generators import basic_generate_fn

        llm = FakeLLM("answer")
        docs = [_doc(content="unique context text")]
        basic_generate_fn(docs, query="q", temperature=0.7, llm_client=llm)
        assert llm.last_prompt is not None
        assert "unique context text" in llm.last_prompt

    def test_prompt_contains_query(self) -> None:
        from rag.handlers.generators import basic_generate_fn

        llm = FakeLLM("answer")
        docs = [_doc()]
        basic_generate_fn(docs, query="specific question", temperature=0.7, llm_client=llm)
        assert llm.last_prompt is not None
        assert "specific question" in llm.last_prompt

    def test_returns_string(self) -> None:
        from rag.handlers.generators import basic_generate_fn

        llm = FakeLLM("answer")
        result = basic_generate_fn([], query="q", temperature=0.0, llm_client=llm)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# cot_generate_fn 测试
# ---------------------------------------------------------------------------


class TestCotGenerateFn:
    def test_returns_llm_response(self) -> None:
        from rag.handlers.generators import cot_generate_fn

        llm = FakeLLM("Step 1: think. Step 2: answer.")
        result = cot_generate_fn([_doc()], query="q", temperature=0.5, llm_client=llm)
        assert result == "Step 1: think. Step 2: answer."

    def test_prompt_has_reasoning_hint(self) -> None:
        from rag.handlers.generators import cot_generate_fn

        llm = FakeLLM("answer")
        cot_generate_fn([_doc()], query="q", temperature=0.5, llm_client=llm)
        assert llm.last_prompt is not None
        assert any(
            hint in llm.last_prompt for hint in ["推理", "思考", "步骤", "step", "reason", "think"]
        )


# ---------------------------------------------------------------------------
# citation_generate_fn 测试
# ---------------------------------------------------------------------------


class TestCitationGenerateFn:
    def test_returns_llm_response(self) -> None:
        from rag.handlers.generators import citation_generate_fn

        llm = FakeLLM("The answer [1].")
        docs = [_doc()]
        result = citation_generate_fn(docs, query="q", temperature=0.0, llm_client=llm)
        assert result == "The answer [1]."

    def test_prompt_contains_numbered_refs(self) -> None:
        from rag.handlers.generators import citation_generate_fn

        llm = FakeLLM("answer")
        docs = [_doc("d1", content="doc1"), _doc("d2", content="doc2")]
        citation_generate_fn(docs, query="q", temperature=0.0, llm_client=llm)
        assert llm.last_prompt is not None
        assert "[1]" in llm.last_prompt
        assert "[2]" in llm.last_prompt

    def test_prompt_contains_query(self) -> None:
        from rag.handlers.generators import citation_generate_fn

        llm = FakeLLM("answer")
        citation_generate_fn([_doc()], query="citation test query", temperature=0.0, llm_client=llm)
        assert llm.last_prompt is not None
        assert "citation test query" in llm.last_prompt


# ---------------------------------------------------------------------------
# 注册验证
# ---------------------------------------------------------------------------


class TestGeneratorRegistration:
    def test_all_generators_registered(self) -> None:
        import rag.handlers.generators  # noqa: F401
        from rag.registry import generator_registry

        for name in ["basic", "cot", "citation"]:
            assert generator_registry.has(name), f"'{name}' not registered"
