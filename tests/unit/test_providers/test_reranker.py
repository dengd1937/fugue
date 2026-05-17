"""tests/unit/test_providers/test_reranker.py — BGEReranker 单元测试（全 mock）。"""

from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# 测试 1: 基础调用 + 降序排列
# ---------------------------------------------------------------------------
def test_basic_rerank_descending_order():
    """compute_score 返回 [0.3, 0.8, 0.1]，结果按分数降序排列。"""
    with (
        patch("fugue.providers.reranker.bge.require") as mock_require,
        patch("fugue.providers.reranker.bge._resolve_device", return_value="cpu"),
    ):
        mock_cls = mock_require.return_value.FlagReranker
        mock_inst = MagicMock()
        mock_inst.compute_score.return_value = [0.3, 0.8, 0.1]
        mock_cls.return_value = mock_inst

        from fugue.providers.reranker.bge import BGEReranker

        reranker = BGEReranker(device="cpu")
        result = reranker.rerank("q", ["a", "b", "c"])

        assert result == [(1, 0.8), (0, 0.3), (2, 0.1)]
        mock_inst.compute_score.assert_called_once()


# ---------------------------------------------------------------------------
# 测试 2: top_k 截断
# ---------------------------------------------------------------------------
def test_rerank_top_k_truncation():
    """top_k=2 时只返回分数最高的 2 个。"""
    with (
        patch("fugue.providers.reranker.bge.require") as mock_require,
        patch("fugue.providers.reranker.bge._resolve_device", return_value="cpu"),
    ):
        mock_cls = mock_require.return_value.FlagReranker
        mock_inst = MagicMock()
        mock_inst.compute_score.return_value = [0.1, 0.5, 0.9, 0.3, 0.7]
        mock_cls.return_value = mock_inst

        from fugue.providers.reranker.bge import BGEReranker

        reranker = BGEReranker(device="cpu")
        result = reranker.rerank("q", ["a", "b", "c", "d", "e"], top_k=2)

        assert result == [(2, 0.9), (4, 0.7)]


# ---------------------------------------------------------------------------
# 测试 3: 空 documents
# ---------------------------------------------------------------------------
def test_rerank_empty_documents():
    """空文档列表应立即返回 []，不调用 compute_score。"""
    with (
        patch("fugue.providers.reranker.bge.require") as mock_require,
        patch("fugue.providers.reranker.bge._resolve_device", return_value="cpu"),
    ):
        mock_cls = mock_require.return_value.FlagReranker
        mock_inst = MagicMock()
        mock_cls.return_value = mock_inst

        from fugue.providers.reranker.bge import BGEReranker

        reranker = BGEReranker(device="cpu")
        result = reranker.rerank("q", [])

        assert result == []
        mock_inst.compute_score.assert_not_called()


# ---------------------------------------------------------------------------
# 测试 4: device="auto" + CUDA 可用 → cuda
# ---------------------------------------------------------------------------
def test_device_auto_cuda_available():
    """device='auto' 且 CUDA 可用时，应以 devices=['cuda'] + use_fp16=True 初始化。"""
    with (
        patch("fugue.providers.reranker.bge.require") as mock_require,
        patch("torch.cuda.is_available", return_value=True),
    ):
        mock_cls = mock_require.return_value.FlagReranker
        mock_inst = MagicMock()
        mock_cls.return_value = mock_inst

        from fugue.providers.reranker.bge import BGEReranker

        BGEReranker(device="auto")

        mock_cls.assert_called_once()
        _, kwargs = mock_cls.call_args
        assert kwargs["devices"] == ["cuda"]
        assert kwargs["use_fp16"] is True


# ---------------------------------------------------------------------------
# 测试 5: device="auto" + CUDA 不可用 → cpu
# ---------------------------------------------------------------------------
def test_device_auto_cuda_unavailable():
    """device='auto' 且 CUDA 不可用时，应以 devices=['cpu'] + use_fp16=False 初始化。"""
    with (
        patch("fugue.providers.reranker.bge.require") as mock_require,
        patch("torch.cuda.is_available", return_value=False),
    ):
        mock_cls = mock_require.return_value.FlagReranker
        mock_inst = MagicMock()
        mock_cls.return_value = mock_inst

        from fugue.providers.reranker.bge import BGEReranker

        BGEReranker(device="auto")

        _, kwargs = mock_cls.call_args
        assert kwargs["devices"] == ["cpu"]
        assert kwargs["use_fp16"] is False


# ---------------------------------------------------------------------------
# 测试 6: close 释放（幂等）
# ---------------------------------------------------------------------------
def test_close_idempotent():
    """close() 应释放 _reranker 属性，多次调用不抛错。"""
    with (
        patch("fugue.providers.reranker.bge.require") as mock_require,
        patch("fugue.providers.reranker.bge._resolve_device", return_value="cpu"),
    ):
        mock_cls = mock_require.return_value.FlagReranker
        mock_inst = MagicMock()
        mock_cls.return_value = mock_inst

        from fugue.providers.reranker.bge import BGEReranker

        reranker = BGEReranker(device="cpu")
        assert hasattr(reranker, "_reranker")

        reranker.close()
        assert not hasattr(reranker, "_reranker")

        # 第二次 close 不应抛错
        reranker.close()


# ---------------------------------------------------------------------------
# 测试 7: device 显式指定 "cpu"
# ---------------------------------------------------------------------------
def test_device_explicit_cpu():
    """显式 device='cpu' 时，不应调用 torch.cuda.is_available。"""
    with patch("fugue.providers.reranker.bge.require") as mock_require, patch("torch.cuda.is_available") as mock_cuda:
        mock_cls = mock_require.return_value.FlagReranker
        mock_inst = MagicMock()
        mock_cls.return_value = mock_inst

        from fugue.providers.reranker.bge import BGEReranker

        BGEReranker(device="cpu")

        mock_cuda.assert_not_called()
        _, kwargs = mock_cls.call_args
        assert kwargs["devices"] == ["cpu"]
        assert kwargs["use_fp16"] is False


# ---------------------------------------------------------------------------
# 测试 8: compute_score 返回单个 float（边界情况）
# ---------------------------------------------------------------------------
def test_single_document_float_score():
    """单文档时 compute_score 返回 float，应正确适配为 [(0, score)]。"""
    with (
        patch("fugue.providers.reranker.bge.require") as mock_require,
        patch("fugue.providers.reranker.bge._resolve_device", return_value="cpu"),
    ):
        mock_cls = mock_require.return_value.FlagReranker
        mock_inst = MagicMock()
        mock_inst.compute_score.return_value = 0.75  # 单个 float
        mock_cls.return_value = mock_inst

        from fugue.providers.reranker.bge import BGEReranker

        reranker = BGEReranker(device="cpu")
        result = reranker.rerank("q", ["only doc"])

        assert result == [(0, 0.75)]


# ---------------------------------------------------------------------------
# 测试 9: compute_score 返回 numpy 标量（旧版 FlagEmbedding 行为）
# ---------------------------------------------------------------------------
def test_compute_score_returns_numpy_scalar() -> None:
    """compute_score 返回 numpy 标量时（旧版 FlagEmbedding 行为）应正确处理。"""
    try:
        import numpy as np

        scalar_value = np.float32(0.42)
    except ImportError:
        return  # numpy 不可用时跳过

    with (
        patch("fugue.providers.reranker.bge.require") as mock_require,
        patch("fugue.providers.reranker.bge._resolve_device", return_value="cpu"),
    ):
        mock_cls = mock_require.return_value.FlagReranker
        mock_inst = MagicMock()
        mock_inst.compute_score.return_value = scalar_value
        mock_cls.return_value = mock_inst

        from fugue.providers.reranker.bge import BGEReranker

        reranker = BGEReranker(device="cpu")
        result = reranker.rerank("q", ["only doc"])

        assert len(result) == 1
        assert result[0][0] == 0
        assert abs(result[0][1] - 0.42) < 1e-5


# ---------------------------------------------------------------------------
# 测试 Protocol 结构
# ---------------------------------------------------------------------------
def test_bge_reranker_implements_protocol():
    """BGEReranker 应满足 Reranker Protocol（runtime_checkable）。"""
    with (
        patch("fugue.providers.reranker.bge.require") as mock_require,
        patch("fugue.providers.reranker.bge._resolve_device", return_value="cpu"),
    ):
        mock_cls = mock_require.return_value.FlagReranker
        mock_inst = MagicMock()
        mock_cls.return_value = mock_inst

        from fugue.providers.reranker import BGEReranker, Reranker

        reranker = BGEReranker(device="cpu")
        assert isinstance(reranker, Reranker)


# ---------------------------------------------------------------------------
# 新增用例 A: FlagEmbedding 缺失时抛 ImportError
# ---------------------------------------------------------------------------
def test_missing_flagembedding_raises_import_error():
    """FlagEmbedding 未安装时构造 BGEReranker 应抛 ImportError，消息包含安装提示。"""
    import pytest

    with (
        patch(
            "fugue.providers.reranker.bge.require",
            side_effect=ImportError("可选依赖 'FlagEmbedding' 未安装。请运行: pip install 'fugue[bge]'"),
        ),
        patch("fugue.providers.reranker.bge._resolve_device", return_value="cpu"),
    ):
        from fugue.providers.reranker.bge import BGEReranker

        with pytest.raises(ImportError) as exc_info:
            BGEReranker(model_name="x", device="cpu")

        assert "pip install 'fugue[bge]'" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 新增用例 B: 模块顶层不含 FlagReranker 名字（懒加载验证）
# ---------------------------------------------------------------------------
def test_bge_module_no_toplevel_flagreranker():
    """改造后 bge 模块顶层不应暴露 FlagReranker 名字。"""
    import fugue.providers.reranker.bge as bgemod

    assert not hasattr(bgemod, "FlagReranker"), "bge 模块顶层不应有 FlagReranker 属性（懒加载）"

    # 两种导入路径均不抛异常
    from fugue.providers.reranker import BGEReranker as _BGEReranker2  # noqa: F401
    from fugue.providers.reranker.bge import BGEReranker as _BGEReranker1  # noqa: F401
