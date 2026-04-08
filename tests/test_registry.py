"""tests/test_registry.py — rag.registry 单元测试（TDD RED 阶段先写）"""

import pytest

from rag.registry import (
    Registry,
    generator_registry,
    grader_registry,
    processor_registry,
    retriever_registry,
    transform_registry,
)

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def double(x: int, **_: object) -> int:
    return x * 2


def add_one(x: int, **_: object) -> int:
    return x + 1


def square(x: int, **_: object) -> int:
    return x * x


# ---------------------------------------------------------------------------
# register / get
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_register_and_get_returns_function() -> None:
    reg = Registry()
    reg.register("double", double)
    assert reg.get("double") is double


@pytest.mark.unit
def test_get_unregistered_raises_key_error() -> None:
    reg = Registry()
    with pytest.raises(KeyError):
        reg.get("nonexistent")


@pytest.mark.unit
def test_get_error_message_contains_available_names() -> None:
    reg = Registry()
    reg.register("alpha", double)
    reg.register("beta", add_one)
    with pytest.raises(KeyError, match="alpha"):
        reg.get("nonexistent")


@pytest.mark.unit
def test_register_overwrites_existing() -> None:
    reg = Registry()
    reg.register("fn", double)
    reg.register("fn", add_one)
    assert reg.get("fn") is add_one


# ---------------------------------------------------------------------------
# has
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_has_returns_true_for_registered() -> None:
    reg = Registry()
    reg.register("fn", double)
    assert reg.has("fn") is True


@pytest.mark.unit
def test_has_returns_false_for_unregistered() -> None:
    reg = Registry()
    assert reg.has("missing") is False


# ---------------------------------------------------------------------------
# names
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_names_returns_empty_list_initially() -> None:
    reg = Registry()
    assert reg.names() == []


@pytest.mark.unit
def test_names_returns_registered_names() -> None:
    reg = Registry()
    reg.register("alpha", double)
    reg.register("beta", add_one)
    assert set(reg.names()) == {"alpha", "beta"}


@pytest.mark.unit
def test_names_returns_list_type() -> None:
    reg = Registry()
    reg.register("fn", double)
    assert isinstance(reg.names(), list)


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_pipeline_single_step() -> None:
    reg = Registry()
    reg.register("double", double)
    result = reg.run_pipeline(["double"], 5)
    assert result == 10


@pytest.mark.unit
def test_run_pipeline_two_steps_double_then_add_one() -> None:
    reg = Registry()
    reg.register("double", double)
    reg.register("add_one", add_one)
    result = reg.run_pipeline(["double", "add_one"], 5)
    assert result == 11  # 5*2=10, 10+1=11


@pytest.mark.unit
def test_run_pipeline_order_matters() -> None:
    reg = Registry()
    reg.register("double", double)
    reg.register("add_one", add_one)
    # add_one first: 5+1=6, 6*2=12
    result = reg.run_pipeline(["add_one", "double"], 5)
    assert result == 12


@pytest.mark.unit
def test_run_pipeline_empty_returns_initial() -> None:
    reg = Registry()
    result = reg.run_pipeline([], 42)
    assert result == 42


@pytest.mark.unit
def test_run_pipeline_three_steps() -> None:
    reg = Registry()
    reg.register("double", double)
    reg.register("add_one", add_one)
    reg.register("square", square)
    # 3 -> double=6 -> add_one=7 -> square=49
    result = reg.run_pipeline(["double", "add_one", "square"], 3)
    assert result == 49


@pytest.mark.unit
def test_run_pipeline_raises_key_error_for_missing_step() -> None:
    reg = Registry()
    with pytest.raises(KeyError):
        reg.run_pipeline(["nonexistent"], 5)


@pytest.mark.unit
def test_run_pipeline_passes_kwargs_to_handlers() -> None:
    def greet(name: str, *, prefix: str = "Hello") -> str:
        return f"{prefix}, {name}!"

    reg = Registry()
    reg.register("greet", greet)
    result = reg.run_pipeline(["greet"], "World", prefix="Hi")
    assert result == "Hi, World!"


# ---------------------------------------------------------------------------
# 全局实例独立性
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_global_registries_are_independent() -> None:
    """注册到一个全局实例不影响其他实例。"""
    transform_registry.register("_test_fn", double)
    assert not retriever_registry.has("_test_fn")
    assert not processor_registry.has("_test_fn")
    assert not grader_registry.has("_test_fn")
    assert not generator_registry.has("_test_fn")


@pytest.mark.unit
def test_five_global_instances_are_distinct_objects() -> None:
    registries = [
        transform_registry,
        retriever_registry,
        processor_registry,
        grader_registry,
        generator_registry,
    ]
    # 所有实例两两不同
    for i, r1 in enumerate(registries):
        for j, r2 in enumerate(registries):
            if i != j:
                assert r1 is not r2


@pytest.mark.unit
def test_global_registry_instances_are_registry_type() -> None:
    for reg in (
        transform_registry,
        retriever_registry,
        processor_registry,
        grader_registry,
        generator_registry,
    ):
        assert isinstance(reg, Registry)


# ---------------------------------------------------------------------------
# 独立性（Registry 实例间不共享状态）
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_two_registry_instances_do_not_share_state() -> None:
    reg1 = Registry()
    reg2 = Registry()
    reg1.register("fn", double)
    assert not reg2.has("fn")
