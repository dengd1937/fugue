"""src/ragline/providers/embedding.py — OpenAI 兼容 Embedding 客户端。"""

import threading
from types import TracebackType

from openai import OpenAI

from ragline.api.types import RaglineEmbeddingError


class EmbeddingClient:
    """OpenAI 兼容 Embedding 统一封装。

    支持按 batch_size 分批调用；单批失败时拆成两半重试一次（避免坏 input 整批挂）。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 30.0,
        max_retries: int = 2,
        max_concurrent: int = 5,
        batch_size: int = 64,
    ) -> None:
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._model = model
        self._batch_size = batch_size
        self._sem = threading.Semaphore(max_concurrent)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量 embed；按 batch_size 分批；失败时该 batch 拆两半重试一次。"""
        if not texts:
            return []
        result: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            result.extend(self._embed_batch(batch))
        return result

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """单批 embed，失败时拆两半重试一次（仅一次切分）。"""
        try:
            return self._embed_raw(batch)
        except Exception as first_exc:
            if len(batch) <= 1:
                raise RaglineEmbeddingError(f"Embedding failed for single text: {first_exc}") from first_exc
            mid = len(batch) // 2
            left, right = batch[:mid], batch[mid:]
            try:
                left_result = self._embed_raw(left)
                right_result = self._embed_raw(right)
                return left_result + right_result
            except Exception as second_exc:
                raise RaglineEmbeddingError(f"Embedding failed after split retry: {second_exc}") from second_exc

    def _embed_raw(self, batch: list[str]) -> list[list[float]]:
        """实际 API 调用，受 Semaphore 限流。"""
        with self._sem:
            resp = self._client.embeddings.create(model=self._model, input=batch)
            return [item.embedding for item in resp.data]

    def close(self) -> None:
        """关闭底层 httpx client。"""
        self._client.close()

    def __enter__(self) -> "EmbeddingClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
