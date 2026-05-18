"""tests/unit/test_registry.py — Registry 单元测试。"""

import logging
from unittest.mock import MagicMock, patch

import pytest

from ragline.api.types import FugueRegistryError
from ragline.registry import (
    Registry,
    chunker_registry,
    discover_plugins,
    generator_registry,
    grader_registry,
    parser_registry,
    processor_registry,
    retriever_registry,
    transform_registry,
)


# 测试 1：register + get，identity check
def test_register_and_get() -> None:
    reg: Registry = Registry("test")

    def my_fn(x: int) -> int:
        return x * 2

    reg.register("my_fn", my_fn)
    assert reg.get("my_fn") is my_fn


# 测试 2：__call__ 装饰器糖
def test_decorator_sugar() -> None:
    reg: Registry = Registry("test_deco")

    def original_fn(x: int) -> int:
        return x + 1

    decorated = reg("foo")(original_fn)

    # 装饰后返回原函数
    assert reg.get("foo") is original_fn
    # decorated_fn 仍是原函数，能正常调用
    assert decorated is original_fn
    assert decorated(5) == 6


# 测试 3：get 不存在的 handler 抛 FugueRegistryError
def test_get_nonexistent_raises() -> None:
    reg: Registry = Registry("test_get")
    reg.register("alpha", lambda: None)
    reg.register("beta", lambda: None)

    with pytest.raises(FugueRegistryError) as exc_info:
        reg.get("gamma")

    msg = str(exc_info.value)
    assert "Available:" in msg
    assert "alpha" in msg
    assert "beta" in msg


# 测试 4：unregister
def test_unregister() -> None:
    reg: Registry = Registry("test_unreg")
    reg.register("foo", lambda: None)
    assert reg.has("foo")

    reg.unregister("foo")
    assert not reg.has("foo")

    # unregister 不存在的名字不抛错
    reg.unregister("nonexistent")  # should not raise


# 测试 5：names 返回排序后的列表
def test_names_sorted() -> None:
    reg: Registry = Registry("test_names")
    reg.register("zebra", lambda: None)
    reg.register("apple", lambda: None)
    reg.register("mango", lambda: None)

    assert reg.names() == ["apple", "mango", "zebra"]


# 测试 6：重复 register 覆盖 + warning
def test_register_overwrite_warns(caplog: pytest.LogCaptureFixture) -> None:
    reg: Registry = Registry("test_overwrite")

    def fn_v1() -> str:
        return "v1"

    def fn_v2() -> str:
        return "v2"

    reg.register("handler", fn_v1)

    with caplog.at_level(logging.WARNING, logger="ragline.registry"):
        reg.register("handler", fn_v2)

    # 覆盖成功
    assert reg.get("handler") is fn_v2

    # warning 记录包含 "already registered"
    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("already registered" in msg for msg in warning_messages)


# 测试 7：discover_plugins
def test_discover_plugins(caplog: pytest.LogCaptureFixture) -> None:
    register_fn1 = MagicMock()
    register_fn3 = MagicMock()

    ep1 = MagicMock()
    ep1.name = "plugin1"
    ep1.load.return_value = register_fn1

    ep2 = MagicMock()
    ep2.name = "plugin2"
    ep2.load.side_effect = ImportError("missing dep")

    ep3 = MagicMock()
    ep3.name = "plugin3"
    ep3.load.return_value = register_fn3

    with (
        patch("ragline.registry.entry_points", return_value=[ep1, ep2, ep3]),
        caplog.at_level(logging.WARNING, logger="ragline.registry"),
    ):
        discover_plugins()

    # ep1 和 ep3 的 register_fn 被调用
    register_fn1.assert_called_once()
    register_fn3.assert_called_once()

    # ep2 失败时记录 warning
    warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("plugin2" in msg for msg in warning_messages)


# 测试 8：7 个全局单例都存在且类型正确
def test_global_singletons_exist() -> None:
    for reg in [
        transform_registry,
        retriever_registry,
        processor_registry,
        grader_registry,
        generator_registry,
        parser_registry,
        chunker_registry,
    ]:
        assert isinstance(reg, Registry)
