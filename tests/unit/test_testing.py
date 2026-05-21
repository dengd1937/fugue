"""tests/unit/test_testing.py — ragline.testing 公开模块测试。"""

from unittest.mock import patch

import pytest

from ragline.testing import (
    FakeEmbedding,
    FakeLLM,
    isolated_registries,
    mock_rag_providers,
)

# ──────────────────────────────────────────────
# FakeLLM 测试
# ──────────────────────────────────────────────


class TestFakeLLM:
    def test_default_answer_and_calls_recorded(self) -> None:
        """场景1：FakeLLM().complete("q") 返回默认 "fake answer"；记录调用。"""
        fake = FakeLLM()
        result = fake.complete("q")
        assert result == "fake answer"
        assert fake.calls == [("q", {"temperature": 0.7})]

    def test_custom_temperature_keyword_only(self) -> None:
        """场景2：temperature 是 keyword-only；位置参数传入必须抛 TypeError。"""
        fake = FakeLLM(answer="hi")
        result = fake.complete("q", temperature=0.1)
        assert result == "hi"
        assert fake.calls == [("q", {"temperature": 0.1})]
        # temperature 必须是 keyword-only，位置参数调用应该失败
        with pytest.raises(TypeError):
            fake.complete("q", 0.1)  # type: ignore[call-arg]

    def test_close_count_and_idempotent(self) -> None:
        """场景3：连续调用两次 close()：close_calls == 2，不抛错。"""
        fake = FakeLLM()
        fake.close()
        fake.close()
        assert fake.close_calls == 2

    def test_answer_writable_at_runtime(self) -> None:
        """场景4：运行时可修改 .answer，complete() 读取 self.answer 而非 closure 缓存。"""
        f = FakeLLM()
        f.answer = "X"
        assert f.complete("q") == "X"


# ──────────────────────────────────────────────
# FakeEmbedding 测试
# ──────────────────────────────────────────────


class TestFakeEmbedding:
    def test_embed_dim4_and_calls_recorded(self) -> None:
        """场景5：FakeEmbedding(dim=4).embed(["a","b"]) 返回 [[0.1]*4,[0.1]*4]；记录调用。"""
        fake = FakeEmbedding(dim=4)
        result = fake.embed(["a", "b"])
        assert result == [[0.1] * 4, [0.1] * 4]
        assert fake.calls == [["a", "b"]]

    def test_embed_empty_list(self) -> None:
        """场景6：embed([]) 返回 []；仍记录调用。"""
        fake = FakeEmbedding()
        result = fake.embed([])
        assert result == []
        assert fake.calls == [[]]

    def test_close_count_and_idempotent(self) -> None:
        """场景7：连续调用两次 close()：close_calls == 2，不抛错。"""
        fake = FakeEmbedding()
        fake.close()
        fake.close()
        assert fake.close_calls == 2


# ──────────────────────────────────────────────
# isolated_registries 测试
# ──────────────────────────────────────────────


class TestIsolatedRegistries:
    def test_registries_empty_inside_and_restored_outside(self) -> None:
        """场景8：enter 后所有 7 个 registry 为空；exit 后恢复原名称集合与 callable identity。"""
        from ragline.registry import (
            chunker_registry,
            generator_registry,
            grader_registry,
            parser_registry,
            processor_registry,
            retriever_registry,
            transform_registry,
        )

        # 先注册一些内容
        def dummy_fn() -> None: ...

        transform_registry.register("test_handler", dummy_fn)

        original_transforms = set(transform_registry.names())
        original_fn_identity = transform_registry.get("test_handler")

        with isolated_registries():
            # 所有 registry 应为空
            assert transform_registry.names() == []
            assert retriever_registry.names() == []
            assert processor_registry.names() == []
            assert grader_registry.names() == []
            assert generator_registry.names() == []
            assert parser_registry.names() == []
            assert chunker_registry.names() == []

        # 恢复后名称集合与 callable identity 应该一致
        assert set(transform_registry.names()) == original_transforms
        assert transform_registry.get("test_handler") is original_fn_identity

        # 清理
        transform_registry.unregister("test_handler")

    def test_nested_isolated_registries(self) -> None:
        """场景9：嵌套调用：内层 exit 后外层 yield 时仍为空；外层 exit 后完全恢复。"""
        from ragline.registry import transform_registry

        def dummy_fn() -> None: ...

        transform_registry.register("outer_handler", dummy_fn)
        original_names = set(transform_registry.names())

        with isolated_registries():
            # 外层 enter 后应为空
            assert transform_registry.names() == []
            with isolated_registries():
                # 内层 enter 后也应为空
                assert transform_registry.names() == []
            # 内层 exit 后，外层 context 内仍为空
            assert transform_registry.names() == []

        # 外层 exit 后完全恢复
        assert set(transform_registry.names()) == original_names
        assert transform_registry.get("outer_handler") is dummy_fn

        # 清理
        transform_registry.unregister("outer_handler")


# ──────────────────────────────────────────────
# mock_rag_providers 测试
# ──────────────────────────────────────────────


class TestMockRagProviders:
    def test_rag_uses_fake_providers(self) -> None:
        """场景10：mock_rag_providers 让 RAG() 用 fake；守护无真实网络调用。"""
        from ragline.api.rag import RAG

        with (
            patch(
                "ragline.providers.llm.OpenAI",
                side_effect=AssertionError("network!"),
            ),
            mock_rag_providers() as (llm, embedding),
        ):
            rag = RAG()
            rag.query("test question")
            assert len(llm.calls) > 0

    def test_custom_fake_injection(self) -> None:
        """场景11：mock_rag_providers(llm=FakeLLM(answer="X")) 自定义注入。"""
        from ragline.api.rag import RAG

        custom_llm = FakeLLM(answer="X")
        with mock_rag_providers(llm=custom_llm) as (llm, embedding):
            rag = RAG()
            result = rag.query("test question")
            assert result.answer == "X"
