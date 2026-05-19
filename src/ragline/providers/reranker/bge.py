"""src/ragline/providers/reranker/bge.py — BGE Reranker 实现。"""

from typing import Literal

from ragline._optional import require


def _resolve_device(device: Literal["cpu", "cuda", "auto"]) -> str:
    """device="auto" → 检测 torch.cuda 可用性 → cuda 或 cpu。"""
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class BGEReranker:
    """BAAI/bge-reranker 系列模型封装。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: Literal["cpu", "cuda", "auto"] = "auto",
        timeout: float = 30.0,
    ) -> None:
        """构造 BGE Reranker。

        Args:
            model_name: HuggingFace 模型名（默认 BAAI/bge-reranker-v2-m3）
            device: 设备选择（cpu / cuda / auto）
            timeout: 单次 rerank 超时秒数。**MVP 阶段为预留参数，
                FlagReranker.compute_score 未原生支持 timeout；
                P1 将通过 ThreadPoolExecutor 包装实现。当前值仅存储不强制。**
        """
        _flagembedding = require("FlagEmbedding", extra="bge")
        flag_reranker_cls = _flagembedding.FlagReranker
        resolved = _resolve_device(device)
        self._reranker = flag_reranker_cls(
            model_name,
            use_fp16=(resolved == "cuda"),  # cpu 不支持 fp16
            devices=[resolved],
        )
        self._timeout = timeout

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """调用 FlagReranker.compute_score，返回 (idx, score) 降序列表。"""
        if not documents:
            return []
        pairs = [[query, doc] for doc in documents]
        scores = self._reranker.compute_score(pairs, normalize=True)
        # compute_score 单个 pair 返回标量（float 或 numpy.float32），多个返回 list；统一为 list
        if not hasattr(scores, "__len__"):
            scores_list: list[float] = [float(scores)]
        else:
            scores_list = [float(s) for s in scores]
        indexed = sorted(
            enumerate(scores_list),
            key=lambda x: x[1],
            reverse=True,
        )
        if top_k is not None:
            indexed = indexed[:top_k]
        return indexed

    def close(self) -> None:
        """释放模型占用资源。"""
        if hasattr(self, "_reranker"):
            del self._reranker
