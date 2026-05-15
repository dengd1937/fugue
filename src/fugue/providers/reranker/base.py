"""src/fugue/providers/reranker/base.py — Reranker Protocol。"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Reranker(Protocol):
    """重排序抽象。"""

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """返回 [(原始索引, 新分数), ...] 按分数降序。

        Args:
            query: 查询字符串
            documents: 待重排文档列表
            top_k: 截断保留前 K 个；None 表示返回全部
        """
        ...

    def close(self) -> None:
        """显式释放模型资源（显存/RAM）。"""
        ...
