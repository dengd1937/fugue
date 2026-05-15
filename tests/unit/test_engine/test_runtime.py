"""tests/unit/test_engine/test_runtime.py — get_config 测试。"""

from fugue.config import GraphConfig
from fugue.engine.runtime import get_config


def test_get_config_extracts_known_fields() -> None:
    config = {"configurable": {"n_rewrites": 5, "thread_id": "xyz", "top_k": 7}}
    result = get_config(config)  # type: ignore[arg-type]
    assert isinstance(result, GraphConfig)
    assert result.n_rewrites == 5
    assert result.top_k == 7


def test_get_config_empty_configurable() -> None:
    result = get_config({"configurable": {}})  # type: ignore[arg-type]
    default = GraphConfig()
    assert result.n_rewrites == default.n_rewrites
    assert result.top_k == default.top_k


def test_get_config_missing_configurable() -> None:
    result = get_config({})  # type: ignore[arg-type]
    assert isinstance(result, GraphConfig)


def test_get_config_unknown_fields_filtered() -> None:
    """框架注入的 thread_id / checkpoint_ns 等被过滤。"""
    config = {
        "configurable": {
            "thread_id": "session-1",
            "checkpoint_ns": "ns",
            "n_rewrites": 3,
        }
    }
    result = get_config(config)  # type: ignore[arg-type]
    assert result.n_rewrites == 3


def test_get_config_no_configurable_key() -> None:
    """缺 configurable 键也能 fallback 到默认。"""
    result = get_config({"tags": []})  # type: ignore[arg-type]
    assert isinstance(result, GraphConfig)
