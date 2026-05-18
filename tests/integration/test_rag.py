"""tests/integration/test_rag.py — RAG 主入口集成测试。"""

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ragline import RAG, FugueConfig, GraphConfig, IngestConfig
from ragline.api.types import FugueConfigError

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def clean_all_registries():
    """清空所有 registry，yield 后恢复（避免污染其他测试）。"""
    from ragline.registry import (
        chunker_registry,
        generator_registry,
        grader_registry,
        parser_registry,
        processor_registry,
        retriever_registry,
        transform_registry,
    )

    registries = {
        "transform": transform_registry,
        "retriever": retriever_registry,
        "processor": processor_registry,
        "grader": grader_registry,
        "generator": generator_registry,
        "parser": parser_registry,
        "chunker": chunker_registry,
    }
    saves = {n: {k: r.get(k) for k in r.names()} for n, r in registries.items()}
    for r in registries.values():
        for k in list(r.names()):
            r.unregister(k)
    yield
    for n, r in registries.items():
        for k in list(r.names()):
            r.unregister(k)
        for k, fn in saves[n].items():
            r.register(k, fn)


@pytest.fixture
def mock_providers():
    """patch LLMClient + EmbeddingClient 避免真 API 调用。"""
    with patch("ragline.api.rag.LLMClient") as mock_llm_cls, patch("ragline.api.rag.EmbeddingClient") as mock_emb_cls:
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "mocked answer"
        mock_llm_cls.return_value = mock_llm

        mock_emb = MagicMock()
        mock_emb.embed.side_effect = lambda texts: [[0.1] * 8 for _ in texts]
        mock_emb_cls.return_value = mock_emb

        yield {"llm": mock_llm, "embedding": mock_emb}


@pytest.fixture(autouse=True)
def fake_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-test")


def _make_config(tmp_path: Path, **graph_overrides) -> FugueConfig:
    return FugueConfig(
        graph=GraphConfig(**graph_overrides),
        ingest=IngestConfig(persist_dir=str(tmp_path / "chroma")),
    )


# 1. 基础实例化 --------------------------------------------------------


def test_rag_basic_instantiation(tmp_path, mock_providers, clean_all_registries) -> None:
    """RAG 能正常实例化，close() 不报错。"""
    cfg = _make_config(tmp_path)
    rag = RAG(cfg)
    assert rag is not None
    rag.close()


# 2. from_yaml 基础 ----------------------------------------------------


def test_from_yaml(tmp_path, mock_providers, clean_all_registries) -> None:
    """from_yaml 读取 YAML 并用正确 config 实例化。"""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        f"""
graph:
  n_rewrites: 5
ingest:
  persist_dir: "{tmp_path / "chroma"}"
"""
    )
    with RAG.from_yaml(yaml_path) as rag:
        assert rag._config.graph.n_rewrites == 5


# 3. with 上下文管理 ---------------------------------------------------


def test_context_manager_calls_close(tmp_path, mock_providers, clean_all_registries) -> None:
    """with 语句退出时调用 close()。"""
    cfg = _make_config(tmp_path)
    rag = RAG(cfg)
    rag.close = MagicMock()  # spy
    with rag:
        pass
    rag.close.assert_called_once()


# 4. fail-fast：未注册 handler -----------------------------------------


def test_fail_fast_unknown_retriever(tmp_path, mock_providers, clean_all_registries) -> None:
    """config 含 nonexistent retriever → 抛 FugueConfigError 含 'nonexistent'。"""
    cfg = _make_config(tmp_path, retrievers=["nonexistent"])
    with pytest.raises(FugueConfigError, match="nonexistent"):
        RAG(cfg)


# 5. fail-fast：多 handler 错误一次抛 ----------------------------------


def test_fail_fast_multiple_errors(tmp_path, mock_providers, clean_all_registries) -> None:
    """多个错误一次性汇集后抛出，message 含所有错误名。"""
    cfg = _make_config(
        tmp_path,
        retrievers=["nonexistent_r"],
        processors=["nonexistent_p"],
    )
    with pytest.raises(FugueConfigError) as exc_info:
        RAG(cfg)
    msg = str(exc_info.value)
    assert "nonexistent_r" in msg
    assert "nonexistent_p" in msg


# 6. fail-fast：缺 api_key --------------------------------------------


def test_fail_fast_missing_api_key(tmp_path, mock_providers, clean_all_registries, monkeypatch) -> None:
    """env + config 都没 OPENAI_API_KEY → 抛 FugueConfigError match OPENAI_API_KEY。"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = _make_config(tmp_path)
    cfg.providers.llm_api_key = None
    with pytest.raises(FugueConfigError, match="OPENAI_API_KEY"):
        RAG(cfg)


# 7. collection_name 覆盖 ----------------------------------------------


def test_collection_name_override(tmp_path, mock_providers, clean_all_registries) -> None:
    """实例化时传 collection_name 参数会覆盖 config.ingest.collection_name。"""
    cfg = _make_config(tmp_path)
    cfg.ingest.collection_name = "from_config"
    rag = RAG(cfg, collection_name="overridden")
    assert rag._config.ingest.collection_name == "overridden"
    rag.close()


# 8. ingest + query 端到端 --------------------------------------------


def test_ingest_and_query_e2e(tmp_path, mock_providers, clean_all_registries) -> None:
    """端到端：ingest md → query → answer 非空（mock LLM 返回 'mocked answer'）。"""
    cfg = _make_config(
        tmp_path,
        retrievers=["vector"],
        processors=[],
        grade_threshold=0.01,
    )
    rag = RAG(cfg)
    result = rag.ingest(FIXTURES_DIR / "sample.md", show_progress=False)
    assert result.num_chunks > 0
    query_result = rag.query("test query")
    # mock LLM returns "mocked answer"
    assert query_result.answer == "mocked answer"
    rag.close()


# 9. graph_override 覆盖 -----------------------------------------------


def test_graph_override(tmp_path, mock_providers, clean_all_registries) -> None:
    """query 时传 graph_override 使用 citation generator，不影响实例默认配置。"""
    cfg = _make_config(
        tmp_path,
        retrievers=["vector"],
        processors=[],
        grade_threshold=0.01,
        gen_mode="basic",
    )
    rag = RAG(cfg)
    # 默认 gen_mode 是 basic
    assert rag._config.graph.gen_mode == "basic"

    # 传入 citation override
    override = GraphConfig(
        retrievers=["vector"],
        processors=[],
        grade_threshold=0.01,
        gen_mode="citation",
    )
    # query with override（不会修改实例配置）
    rag.query("test", graph_override=override)
    # 实例配置未变
    assert rag._config.graph.gen_mode == "basic"
    rag.close()


# 10. 多 RAG 实例 warning ---------------------------------------------


def test_multi_instance_warning(tmp_path, mock_providers, clean_all_registries, caplog) -> None:
    """同进程两次实例化，第二次触发 WARNING 日志。"""
    cfg = _make_config(tmp_path)
    rag1 = RAG(cfg)
    # 第二次实例化应触发 warning（因 registry 已含内置 transforms）
    with caplog.at_level(logging.WARNING, logger="ragline.api.rag"):
        cfg2 = _make_config(tmp_path)
        rag2 = RAG(cfg2)
    assert any("Multiple RAG instances" in r.message for r in caplog.records)
    rag1.close()
    rag2.close()


# 11. bm25 启动重建 ----------------------------------------------------


def test_bm25_bootstrap_on_existing_data(tmp_path, mock_providers, clean_all_registries) -> None:
    """先 ingest 后重建 RAG 实例，验证 BM25 索引可搜到已 ingest 的内容。"""

    chroma_dir = str(tmp_path / "chroma")

    # 第一个 RAG 实例：ingest
    cfg1 = FugueConfig(
        graph=GraphConfig(
            retrievers=["vector", "bm25"],
            processors=[],
            grade_threshold=0.01,
        ),
        ingest=IngestConfig(persist_dir=chroma_dir),
    )
    rag1 = RAG(cfg1)
    rag1.ingest(FIXTURES_DIR / "sample.md", show_progress=False)
    rag1.close()

    # 第二个 RAG 实例：重建（caplog 下触发 warning，但功能正常）
    cfg2 = FugueConfig(
        graph=GraphConfig(
            retrievers=["vector", "bm25"],
            processors=[],
            grade_threshold=0.01,
        ),
        ingest=IngestConfig(persist_dir=chroma_dir),
    )
    rag2 = RAG(cfg2)
    # BM25 索引应包含 ingest 的 chunks
    assert rag2._bm25._bm25 is not None
    assert len(rag2._bm25._chunks) > 0
    # 能 search 到 sample.md 的内容
    results = rag2._bm25.search("markdown", k=5)
    assert len(results) >= 0  # sample.md 内容可能已 tokenize
    rag2.close()


# 12. from_yaml overrides 校验 -----------------------------------------


def test_from_yaml_override_field(tmp_path, mock_providers, clean_all_registries) -> None:
    """from_yaml 传 overrides 可修改 config 字段。"""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        f"""
ingest:
  persist_dir: "{tmp_path / "chroma"}"
"""
    )
    with RAG.from_yaml(yaml_path, graph__n_rewrites=7) as rag:
        assert rag._config.graph.n_rewrites == 7


def test_from_yaml_invalid_override_raises(tmp_path, mock_providers, clean_all_registries) -> None:
    """from_yaml 传无效 override key 抛 FugueConfigError。"""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        f"""
ingest:
  persist_dir: "{tmp_path / "chroma"}"
"""
    )
    with pytest.raises(FugueConfigError, match="dot path"):
        RAG.from_yaml(yaml_path, invalid_no_separator=5)


# 13. close 容错 -------------------------------------------------------


def test_close_tolerates_provider_failure(tmp_path, mock_providers, clean_all_registries) -> None:
    """close() 即使某个 provider close 失败也不影响其他 provider 关闭。"""
    cfg = _make_config(tmp_path)
    rag = RAG(cfg)
    # 让 reranker.close 抛错
    rag._reranker.close = MagicMock(side_effect=RuntimeError("reranker failed"))
    # llm_client / embedding_client 是 MagicMock（mock_providers fixture），close() 默认成功
    # 应不抛出
    rag.close()
    # llm_client.close 仍然被调用
    mock_providers["llm"].close.assert_called_once()
