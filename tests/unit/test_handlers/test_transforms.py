"""tests/unit/test_handlers/test_transforms.py — transforms 处理器单元测试。"""

from unittest.mock import MagicMock

import pytest

from ragline.api.types import TransformResult

# ===== Fixtures =====


@pytest.fixture
def mock_llm():
    """返回一个 MagicMock LLMClient。"""
    return MagicMock()


@pytest.fixture
def clean_transform_registry():
    """清空再 yield，结束时再清空（防止污染其他测试）。"""
    from ragline.registry import transform_registry

    saved = {n: transform_registry.get(n) for n in transform_registry.names()}
    for n in list(transform_registry.names()):
        transform_registry.unregister(n)
    yield transform_registry
    for n in list(transform_registry.names()):
        transform_registry.unregister(n)
    for n, fn in saved.items():
        transform_registry.register(n, fn)


# ===== 测试 1: rewrite_fn 解析换行 =====


def test_rewrite_fn_parses_lines(mock_llm):
    from ragline.handlers.transforms.atoms import rewrite_fn

    mock_llm.complete.return_value = "Q1\nQ2\nQ3"
    result = rewrite_fn(["原问题"], 3, mock_llm)
    assert result == ["Q1", "Q2", "Q3"]
    mock_llm.complete.assert_called_once()


# ===== 测试 2: hyde_fn 段落解析 =====


def test_hyde_fn_parses_paragraphs(mock_llm):
    from ragline.handlers.transforms.atoms import hyde_fn

    mock_llm.complete.return_value = "A1 段第一行\nA1 段第二行\n\nA2 段"
    result = hyde_fn(["q"], 2, mock_llm)
    assert result == ["A1 段第一行\nA1 段第二行", "A2 段"]
    assert len(result) == 2


# ===== 测试 3: step_back_fn 行解析 =====


def test_step_back_fn_parses_lines(mock_llm):
    from ragline.handlers.transforms.atoms import step_back_fn

    mock_llm.complete.return_value = "宏观Q1\n宏观Q2\n宏观Q3"
    result = step_back_fn(["原问题"], 3, mock_llm)
    assert result == ["宏观Q1", "宏观Q2", "宏观Q3"]


# ===== 测试 4: 空 queries =====


def test_empty_queries_returns_empty(mock_llm):
    from ragline.handlers.transforms.atoms import hyde_fn, rewrite_fn, step_back_fn

    assert rewrite_fn([], 3, mock_llm) == []
    assert hyde_fn([], 3, mock_llm) == []
    assert step_back_fn([], 3, mock_llm) == []
    mock_llm.complete.assert_not_called()


# ===== 测试 5: register_transforms 注册 =====


def test_register_transforms_registers_all(mock_llm, clean_transform_registry):
    from ragline.handlers.transforms import register_transforms

    register_transforms(mock_llm)
    assert clean_transform_registry.has("rewrite")
    assert clean_transform_registry.has("hyde")
    assert clean_transform_registry.has("step_back")


# ===== 测试 6: run_transform_branch（原子）=====


def test_run_transform_branch_atom():
    from ragline.handlers.transforms.pipeline import run_transform_branch

    mock_registry = MagicMock()
    mock_registry.get.return_value = lambda q, n: ["改写Q1"]
    result = run_transform_branch("rewrite", ["原"], 3, mock_registry)
    assert result == ["改写Q1"]
    mock_registry.get.assert_called_once_with("rewrite")


# ===== 测试 7: run_transform_branch（管道 2 阶段）=====


def test_run_transform_branch_pipeline_two_stages():
    from ragline.handlers.transforms.pipeline import run_transform_branch

    mock_registry = MagicMock()

    def mock_step_back(queries, n):
        return ["抽象Q1"]

    def mock_rewrite(queries, n):
        assert queries == ["抽象Q1"], f"期望 ['抽象Q1']，实际 {queries}"
        return ["改写Q1.1", "改写Q1.2"]

    def side_effect(name):
        if name == "step_back":
            return mock_step_back
        elif name == "rewrite":
            return mock_rewrite
        raise KeyError(name)

    mock_registry.get.side_effect = side_effect

    result = run_transform_branch(["step_back", "rewrite"], ["原"], 3, mock_registry)
    assert result == ["改写Q1.1", "改写Q1.2"]


# ===== 测试 8: 管道中 TransformResult 传递 =====


def test_run_transform_branch_pipeline_with_transform_result():
    from ragline.handlers.transforms.pipeline import run_transform_branch

    mock_registry = MagicMock()

    def mock_self_query(queries, n):
        return [TransformResult(query="过滤Q1", metadata_filter={"year": 2024})]

    def mock_rewrite(queries, n):
        assert queries == ["过滤Q1"], f"期望 ['过滤Q1']，实际 {queries}"
        return ["最终Q1"]

    def side_effect(name):
        if name == "self_query":
            return mock_self_query
        elif name == "rewrite":
            return mock_rewrite
        raise KeyError(name)

    mock_registry.get.side_effect = side_effect

    result = run_transform_branch(["self_query", "rewrite"], ["原"], 3, mock_registry)
    assert result == ["最终Q1"]


# ===== 测试 9: syrupy 快照 prompt 模板 =====


def test_rewrite_prompt_snapshot(snapshot):
    from ragline.handlers.transforms.atoms import REWRITE_PROMPT

    rendered = REWRITE_PROMPT.format(n=3, query="测试查询")
    assert rendered == snapshot


def test_hyde_prompt_snapshot(snapshot):
    from ragline.handlers.transforms.atoms import HYDE_PROMPT

    rendered = HYDE_PROMPT.format(n=3, query="测试查询")
    assert rendered == snapshot


def test_step_back_prompt_snapshot(snapshot):
    from ragline.handlers.transforms.atoms import STEP_BACK_PROMPT

    rendered = STEP_BACK_PROMPT.format(n=3, query="测试查询")
    assert rendered == snapshot
