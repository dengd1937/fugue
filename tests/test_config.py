"""tests/test_config.py — rag.config 单元测试（TDD RED 阶段先写）"""

import pytest

from rag.config import _GRAPH_CONFIG_FIELDS, GraphConfig

# ---------------------------------------------------------------------------
# 默认值
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_transforms() -> None:
    cfg = GraphConfig()
    assert cfg.transforms == ["rewrite"]


@pytest.mark.unit
def test_default_n_rewrites() -> None:
    cfg = GraphConfig()
    assert cfg.n_rewrites == 3


@pytest.mark.unit
def test_default_max_queries() -> None:
    cfg = GraphConfig()
    assert cfg.max_queries == 20


@pytest.mark.unit
def test_default_retrievers() -> None:
    cfg = GraphConfig()
    assert cfg.retrievers == ["vector", "es"]


@pytest.mark.unit
def test_default_route_strategy() -> None:
    cfg = GraphConfig()
    assert cfg.route_strategy == "all"


@pytest.mark.unit
def test_default_retriever_weights_is_empty_dict() -> None:
    cfg = GraphConfig()
    assert cfg.retriever_weights == {}


@pytest.mark.unit
def test_default_grade_threshold() -> None:
    cfg = GraphConfig()
    assert cfg.grade_threshold == 0.6


@pytest.mark.unit
def test_default_grade_strategy() -> None:
    cfg = GraphConfig()
    assert cfg.grade_strategy == "score"


@pytest.mark.unit
def test_default_score_normalizers() -> None:
    cfg = GraphConfig()
    assert cfg.score_normalizers == {"es": 20.0, "web": 10.0}


@pytest.mark.unit
def test_default_fallback_chain() -> None:
    cfg = GraphConfig()
    assert cfg.fallback_chain == ["web"]


@pytest.mark.unit
def test_default_max_retries() -> None:
    # 默认 fallback_chain=["web"]（长度 1），__post_init__ 规整后为 1
    cfg = GraphConfig()
    assert cfg.max_retries == 1


@pytest.mark.unit
def test_max_retries_auto_sized_to_fallback_chain_length() -> None:
    # fallback_chain 有 2 个元素时，max_retries 应自动规整为 2
    cfg = GraphConfig(fallback_chain=["web", "kg"])
    assert cfg.max_retries == 2


@pytest.mark.unit
def test_max_retries_explicit_value_respected() -> None:
    # 显式传入 max_retries 时不受 fallback_chain 长度覆盖
    cfg = GraphConfig(fallback_chain=["web", "kg"], max_retries=1)
    assert cfg.max_retries == 1


@pytest.mark.unit
def test_max_retries_empty_fallback_chain() -> None:
    # fallback_chain 为空时，max_retries 规整为 0
    cfg = GraphConfig(fallback_chain=[])
    assert cfg.max_retries == 0


@pytest.mark.unit
def test_default_processors() -> None:
    cfg = GraphConfig()
    assert cfg.processors == ["rerank"]


@pytest.mark.unit
def test_default_top_k() -> None:
    cfg = GraphConfig()
    assert cfg.top_k == 3


@pytest.mark.unit
def test_default_gen_mode() -> None:
    cfg = GraphConfig()
    assert cfg.gen_mode == "basic"


@pytest.mark.unit
def test_default_temperature() -> None:
    cfg = GraphConfig()
    assert cfg.temperature == 0.7


# ---------------------------------------------------------------------------
# 可变默认值隔离（不同实例不共享同一列表/字典）
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mutable_defaults_are_independent() -> None:
    cfg1 = GraphConfig()
    cfg2 = GraphConfig()
    cfg1.transforms.append("decompose")
    assert cfg2.transforms == ["rewrite"], "不同实例的 transforms 不应共享"


@pytest.mark.unit
def test_mutable_retriever_weights_independent() -> None:
    cfg1 = GraphConfig()
    cfg2 = GraphConfig()
    cfg1.retriever_weights["vector"] = 0.8
    assert cfg2.retriever_weights == {}


# ---------------------------------------------------------------------------
# to_configurable()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_to_configurable_returns_dict() -> None:
    cfg = GraphConfig()
    result = cfg.to_configurable()
    assert isinstance(result, dict)


@pytest.mark.unit
def test_to_configurable_contains_all_fields() -> None:
    cfg = GraphConfig()
    result = cfg.to_configurable()
    for field_name in _GRAPH_CONFIG_FIELDS:
        assert field_name in result, f"缺少字段: {field_name}"


@pytest.mark.unit
def test_to_configurable_roundtrip() -> None:
    original = GraphConfig(
        n_rewrites=5,
        top_k=10,
        temperature=0.3,
        gen_mode="cot",
        route_strategy="intent",
    )
    d = original.to_configurable()
    restored = GraphConfig(**d)
    assert restored.n_rewrites == 5
    assert restored.top_k == 10
    assert restored.temperature == 0.3
    assert restored.gen_mode == "cot"
    assert restored.route_strategy == "intent"


@pytest.mark.unit
def test_to_configurable_custom_values_preserved() -> None:
    cfg = GraphConfig(retriever_weights={"vector": 0.9, "es": 0.5})
    d = cfg.to_configurable()
    assert d["retriever_weights"] == {"vector": 0.9, "es": 0.5}


# ---------------------------------------------------------------------------
# _GRAPH_CONFIG_FIELDS
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_graph_config_fields_is_frozenset() -> None:
    assert isinstance(_GRAPH_CONFIG_FIELDS, frozenset)


@pytest.mark.unit
def test_graph_config_fields_contains_expected_names() -> None:
    expected = {
        "transforms",
        "n_rewrites",
        "max_queries",
        "retrievers",
        "route_strategy",
        "retriever_weights",
        "grade_threshold",
        "grade_strategy",
        "score_normalizers",
        "fallback_chain",
        "max_retries",
        "processors",
        "top_k",
        "gen_mode",
        "temperature",
    }
    assert expected.issubset(_GRAPH_CONFIG_FIELDS)


@pytest.mark.unit
def test_graph_config_fields_count_matches_dataclass() -> None:
    cfg = GraphConfig()
    assert len(_GRAPH_CONFIG_FIELDS) == len(cfg.to_configurable())
