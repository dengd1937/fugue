"""tests/unit/test_config.py — FugueConfig + YAML loader 单元测试。"""

import logging
from pathlib import Path

import pytest

from ragline.api.types import FugueConfigError


# ---------------------------------------------------------------------------
# 1. 默认实例化
# ---------------------------------------------------------------------------
def test_default_instantiation() -> None:
    from ragline.config import FugueConfig, GraphConfig, IngestConfig

    cfg = FugueConfig()
    assert cfg.graph is not None
    assert cfg.ingest is not None
    assert cfg.providers is not None

    assert GraphConfig().fallback_chain == []
    assert GraphConfig().processors == ["rrf", "rerank"]
    assert IngestConfig().chunker == "recursive"


# ---------------------------------------------------------------------------
# 2. YAML 往返
# ---------------------------------------------------------------------------
def test_yaml_roundtrip(tmp_path: Path) -> None:
    from ragline.config import FugueConfig, GraphConfig, IngestConfig, ProviderConfig, dump_yaml, load_yaml

    custom = FugueConfig(
        graph=GraphConfig(top_k=5, temperature=0.3),
        ingest=IngestConfig(chunk_size=256, collection_name="my_col"),
        providers=ProviderConfig(llm_model="gpt-4o"),
    )
    yaml_file = tmp_path / "config.yaml"
    dump_yaml(custom, yaml_file)

    loaded = load_yaml(yaml_file)
    assert loaded.graph.top_k == 5
    assert loaded.graph.temperature == 0.3
    assert loaded.ingest.chunk_size == 256
    assert loaded.ingest.collection_name == "my_col"
    assert loaded.providers.llm_model == "gpt-4o"


# ---------------------------------------------------------------------------
# 3. 嵌套 transforms YAML
# ---------------------------------------------------------------------------
def test_nested_transforms_yaml(tmp_path: Path) -> None:
    from ragline.config import load_yaml

    content = """
graph:
  transforms:
    - hyde
    - - step_back
      - rewrite
"""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(content, encoding="utf-8")

    cfg = load_yaml(yaml_file)
    assert cfg.graph.transforms == ["hyde", ["step_back", "rewrite"]]


# ---------------------------------------------------------------------------
# 4. 环境变量展开
# ---------------------------------------------------------------------------
def test_env_var_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ragline.config import load_yaml

    monkeypatch.setenv("TEST_KEY", "abc")
    content = """
providers:
  llm_api_key: "${TEST_KEY}"
"""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(content, encoding="utf-8")

    cfg = load_yaml(yaml_file)
    assert cfg.providers.llm_api_key == "abc"


# ---------------------------------------------------------------------------
# 5. 未定义环境变量保留占位符 + 警告
# ---------------------------------------------------------------------------
def test_undefined_env_var(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from ragline.config import load_yaml

    content = """
providers:
  llm_api_key: "${UNDEFINED_X}"
"""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(content, encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="ragline.config"):
        cfg = load_yaml(yaml_file)

    assert cfg.providers.llm_api_key == "${UNDEFINED_X}"
    assert any("UNDEFINED_X" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# 6. 未知字段被拒绝
# ---------------------------------------------------------------------------
def test_unknown_field_rejected(tmp_path: Path) -> None:
    from ragline.config import load_yaml

    content = """
graph:
  unknown_field: 1
"""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(content, encoding="utf-8")

    with pytest.raises(FugueConfigError) as exc_info:
        load_yaml(yaml_file)

    assert "unknown_field" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 7. 类型不匹配被拒绝
# ---------------------------------------------------------------------------
def test_type_mismatch_rejected(tmp_path: Path) -> None:
    from ragline.config import load_yaml

    content = """
graph:
  n_rewrites: "three"
"""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(content, encoding="utf-8")

    with pytest.raises(FugueConfigError) as exc_info:
        load_yaml(yaml_file)

    assert "n_rewrites" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 8. 边界值被拒绝
# ---------------------------------------------------------------------------
def test_boundary_value_rejected(tmp_path: Path) -> None:
    from ragline.config import load_yaml

    # n_rewrites: 0 (ge=1 → 应该拒绝)
    content_low = """
graph:
  n_rewrites: 0
"""
    yaml_file = tmp_path / "config_low.yaml"
    yaml_file.write_text(content_low, encoding="utf-8")

    with pytest.raises(FugueConfigError) as exc_info:
        load_yaml(yaml_file)
    assert "n_rewrites" in str(exc_info.value)

    # n_rewrites: 21 (le=20 → 应该拒绝)
    content_high = """
graph:
  n_rewrites: 21
"""
    yaml_file2 = tmp_path / "config_high.yaml"
    yaml_file2.write_text(content_high, encoding="utf-8")

    with pytest.raises(FugueConfigError) as exc_info2:
        load_yaml(yaml_file2)
    assert "n_rewrites" in str(exc_info2.value)


# ---------------------------------------------------------------------------
# 9. 空 YAML 文件等同默认值
# ---------------------------------------------------------------------------
def test_empty_yaml(tmp_path: Path) -> None:
    from ragline.config import FugueConfig, load_yaml

    yaml_file = tmp_path / "empty.yaml"
    yaml_file.write_text("", encoding="utf-8")

    cfg = load_yaml(yaml_file)
    default = FugueConfig()

    assert cfg.graph.top_k == default.graph.top_k
    assert cfg.ingest.chunker == default.ingest.chunker
    assert cfg.providers.llm_model == default.providers.llm_model


# ---------------------------------------------------------------------------
# 10. YAML 语法错误
# ---------------------------------------------------------------------------
def test_invalid_yaml_syntax(tmp_path: Path) -> None:
    from ragline.config import load_yaml

    content = """
graph:
  transforms: [
  broken yaml here: {{{
"""
    yaml_file = tmp_path / "bad.yaml"
    yaml_file.write_text(content, encoding="utf-8")

    with pytest.raises(FugueConfigError) as exc_info:
        load_yaml(yaml_file)

    assert "Invalid YAML" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 11. 不存在的文件路径
# ---------------------------------------------------------------------------
def test_load_yaml_nonexistent_file() -> None:
    from ragline.config import load_yaml

    with pytest.raises(FugueConfigError) as exc_info:
        load_yaml("/nonexistent/path/config.yaml")

    assert "Failed to read" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 12. 顶层不是 dict
# ---------------------------------------------------------------------------
def test_top_level_not_dict(tmp_path: Path) -> None:
    from ragline.config import load_yaml

    content = "- a\n- b\n"
    yaml_file = tmp_path / "list.yaml"
    yaml_file.write_text(content, encoding="utf-8")

    with pytest.raises(FugueConfigError) as exc_info:
        load_yaml(yaml_file)

    assert "mapping" in str(exc_info.value).lower() or "list" in str(exc_info.value).lower()
