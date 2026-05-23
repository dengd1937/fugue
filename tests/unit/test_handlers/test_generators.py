"""tests/unit/test_handlers/test_generators.py — generators 处理器单元测试。"""

from unittest.mock import MagicMock

import pytest

from ragline.api.types import Document
from ragline.registry import generator_registry

# ===== Fixtures =====


@pytest.fixture
def mock_llm():
    mock = MagicMock()
    mock.complete.return_value = "AI answer"
    return mock


def make_doc(doc_id: str, content: str, score: float = 0.9, source: str = "vector") -> Document:
    return Document(doc_id=doc_id, content=content, score=score, source=source, metadata={})


# ===== 测试 1: basic prompt 内容 =====


def test_basic_generator_prompt_content(mock_llm):
    from ragline.handlers.generators.basic import make_basic_generator

    gen = make_basic_generator(mock_llm)
    docs = [make_doc("d1", "文档内容 A")]
    result = gen("我的问题", docs, temperature=0.7)

    mock_llm.complete.assert_called_once()
    call_args = mock_llm.complete.call_args
    prompt = call_args[0][0]

    assert "基于以下上下文回答问题" in prompt
    assert "文档内容 A" in prompt
    assert "我的问题" in prompt
    assert call_args[1]["temperature"] == 0.7
    assert result == "AI answer"


# ===== 测试 2: citation prompt 含编号 =====


def test_citation_generator_prompt_has_numbers(mock_llm):
    from ragline.handlers.generators.citation import make_citation_generator

    gen = make_citation_generator(mock_llm)
    docs = [make_doc("d1", "A"), make_doc("d2", "B"), make_doc("d3", "C")]
    gen("q", docs, 0.7)

    call_args = mock_llm.complete.call_args
    prompt = call_args[0][0]

    assert "[1] A" in prompt
    assert "[2] B" in prompt
    assert "[3] C" in prompt


# ===== 测试 3: basic 空 docs =====


def test_basic_generator_empty_docs(mock_llm):
    from ragline.handlers.generators.basic import make_basic_generator

    gen = make_basic_generator(mock_llm)
    gen("q", [], temperature=0.7)

    call_args = mock_llm.complete.call_args
    prompt = call_args[0][0]

    assert "上下文：\n\n\n问题：q" in prompt


# ===== 测试 4: citation 空 docs =====


def test_citation_generator_empty_docs(mock_llm):
    from ragline.handlers.generators.citation import make_citation_generator

    gen = make_citation_generator(mock_llm)
    gen("q", [], 0.7)

    mock_llm.complete.assert_called_once()
    call_args = mock_llm.complete.call_args
    prompt = call_args[0][0]

    # 空 docs 时，上下文区域（"上下文：\n" 之后，"问题：" 之前）不含文档编号
    # 模板本身含 "[1][2]..." 仅在描述说明中，context 填充部分为空字符串
    assert "上下文：\n\n\n问题：q" in prompt


# ===== 测试 5: temperature 透传 basic =====


def test_basic_generator_temperature_passthrough(mock_llm):
    from ragline.handlers.generators.basic import make_basic_generator

    gen = make_basic_generator(mock_llm)
    docs = [make_doc("d1", "content")]
    gen("q", docs, temperature=0.1)

    call_args = mock_llm.complete.call_args
    assert call_args[1]["temperature"] == 0.1


# ===== 测试 6: temperature 透传 citation =====


def test_citation_generator_temperature_passthrough(mock_llm):
    from ragline.handlers.generators.citation import make_citation_generator

    gen = make_citation_generator(mock_llm)
    docs = [make_doc("d1", "content")]
    gen("q", docs, 0.2)

    call_args = mock_llm.complete.call_args
    assert call_args[1]["temperature"] == 0.2


# ===== 测试 7: register_generators 注册 =====


def test_register_generators_registers_both(mock_llm, isolated_registries_fx):
    from ragline.handlers.generators import register_generators

    register_generators(mock_llm)

    assert generator_registry.has("basic")
    assert generator_registry.has("citation")


# ===== 测试 8: 多 doc 合并 context（basic）=====


def test_basic_generator_multiple_docs_context(mock_llm):
    from ragline.handlers.generators.basic import make_basic_generator

    gen = make_basic_generator(mock_llm)
    docs = [make_doc("d1", "A"), make_doc("d2", "B"), make_doc("d3", "C")]
    gen("q", docs, temperature=0.7)

    call_args = mock_llm.complete.call_args
    prompt = call_args[0][0]

    assert "A\n\nB\n\nC" in prompt


# ===== 测试 9: syrupy 快照 BASIC_PROMPT_TEMPLATE =====


def test_basic_prompt_template_snapshot(snapshot):
    from ragline.handlers.generators.basic import BASIC_PROMPT_TEMPLATE

    rendered = BASIC_PROMPT_TEMPLATE.format(context="测试上下文", query="测试查询")
    assert rendered == snapshot


# ===== 测试 10: syrupy 快照 CITATION_PROMPT_TEMPLATE =====


def test_citation_prompt_template_snapshot(snapshot):
    from ragline.handlers.generators.citation import CITATION_PROMPT_TEMPLATE

    rendered = CITATION_PROMPT_TEMPLATE.format(context="[1] 测试", query="测试查询")
    assert rendered == snapshot
