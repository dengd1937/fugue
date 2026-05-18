"""tests/unit/test_providers/test_embedding.py — EmbeddingClient 单元测试。"""

from unittest.mock import MagicMock, call, patch

import pytest

from ragline.api.types import FugueEmbeddingError


def make_embedding_response(embeddings: list[list[float]]) -> MagicMock:
    """构造 mock embeddings.create 的返回值。"""
    resp = MagicMock()
    resp.data = [MagicMock(embedding=e) for e in embeddings]
    return resp


@pytest.fixture
def mock_openai():
    """patch OpenAI 构造函数，返回 mock client 实例。"""
    with patch("ragline.providers.embedding.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        yield mock_cls, mock_client


class TestEmbeddingClientBasic:
    """测试 1: 基础调用 — 验证结构、参数透传。"""

    def test_basic_embed_returns_list_of_float_lists(self, mock_openai):
        from ragline.providers.embedding import EmbeddingClient

        mock_cls, mock_client = mock_openai
        mock_client.embeddings.create.return_value = make_embedding_response([[0.1, 0.2], [0.3, 0.4]])

        client = EmbeddingClient(
            base_url="http://localhost:11434/v1",
            api_key="test-key",
            model="nomic-embed",
        )
        result = client.embed(["a", "b"])

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == [0.1, 0.2]
        assert result[1] == [0.3, 0.4]

        mock_client.embeddings.create.assert_called_once_with(model="nomic-embed", input=["a", "b"])

    def test_openai_constructed_with_correct_params(self, mock_openai):
        from ragline.providers.embedding import EmbeddingClient

        mock_cls, mock_client = mock_openai
        mock_client.embeddings.create.return_value = make_embedding_response([[0.1]])

        EmbeddingClient(
            base_url="http://host/v1",
            api_key="key-123",
            model="model-x",
            timeout=60.0,
            max_retries=3,
        )

        mock_cls.assert_called_once_with(
            base_url="http://host/v1",
            api_key="key-123",
            timeout=60.0,
            max_retries=3,
        )


class TestEmbeddingClientBatching:
    """测试 2: 分批 — batch_size=2，5 个文本 → create 调用 3 次。"""

    def test_batching_splits_correctly(self, mock_openai):
        from ragline.providers.embedding import EmbeddingClient

        _, mock_client = mock_openai
        mock_client.embeddings.create.side_effect = [
            make_embedding_response([[0.1], [0.2]]),
            make_embedding_response([[0.3], [0.4]]),
            make_embedding_response([[0.5]]),
        ]

        client = EmbeddingClient(
            base_url="http://localhost/v1",
            api_key="key",
            model="m",
            batch_size=2,
        )
        result = client.embed(["t0", "t1", "t2", "t3", "t4"])

        assert mock_client.embeddings.create.call_count == 3
        calls = mock_client.embeddings.create.call_args_list
        assert calls[0] == call(model="m", input=["t0", "t1"])
        assert calls[1] == call(model="m", input=["t2", "t3"])
        assert calls[2] == call(model="m", input=["t4"])
        assert result == [[0.1], [0.2], [0.3], [0.4], [0.5]]


class TestEmbeddingClientRetry:
    """测试 3: 批失败 → 拆两半成功。"""

    def test_split_retry_on_batch_failure(self, mock_openai):
        from ragline.providers.embedding import EmbeddingClient

        _, mock_client = mock_openai
        e0, e1, e2, e3 = [0.1], [0.2], [0.3], [0.4]
        mock_client.embeddings.create.side_effect = [
            Exception("api error"),  # 整批失败
            make_embedding_response([e0, e1]),  # 左半成功
            make_embedding_response([e2, e3]),  # 右半成功
        ]

        client = EmbeddingClient(
            base_url="http://localhost/v1",
            api_key="key",
            model="m",
            batch_size=4,
        )
        result = client.embed(["t0", "t1", "t2", "t3"])

        assert result == [e0, e1, e2, e3]
        assert mock_client.embeddings.create.call_count == 3


class TestEmbeddingClientDoubleFailure:
    """测试 4: 二次失败 → 抛 FugueEmbeddingError。"""

    def test_raises_ragline_embedding_error_after_split_retry(self, mock_openai):
        from ragline.providers.embedding import EmbeddingClient

        _, mock_client = mock_openai
        mock_client.embeddings.create.side_effect = Exception("always fails")

        client = EmbeddingClient(
            base_url="http://localhost/v1",
            api_key="key",
            model="m",
            batch_size=4,
        )
        with pytest.raises(FugueEmbeddingError, match="Embedding failed after split retry"):
            client.embed(["t0", "t1", "t2", "t3"])

        # 1 整批 + 1 左半（失败即停，因为 second_exc 包装了所有）
        assert mock_client.embeddings.create.call_count >= 2


class TestEmbeddingClientEmpty:
    """测试 5: 空输入 → 直接返回 []，不调用 API。"""

    def test_empty_input_short_circuits(self, mock_openai):
        from ragline.providers.embedding import EmbeddingClient

        _, mock_client = mock_openai

        client = EmbeddingClient(
            base_url="http://localhost/v1",
            api_key="key",
            model="m",
        )
        result = client.embed([])

        assert result == []
        mock_client.embeddings.create.assert_not_called()


class TestEmbeddingClientSingleText:
    """测试 6: 单个 text 失败 → match 'single text'。"""

    def test_single_text_failure_raises_single_text_error(self, mock_openai):
        from ragline.providers.embedding import EmbeddingClient

        _, mock_client = mock_openai
        mock_client.embeddings.create.side_effect = Exception("broken")

        client = EmbeddingClient(
            base_url="http://localhost/v1",
            api_key="key",
            model="m",
            batch_size=1,
        )
        with pytest.raises(FugueEmbeddingError, match="single text"):
            client.embed(["only_one"])


class TestEmbeddingClientContextManager:
    """测试 7: context manager → close() 被调用。"""

    def test_context_manager_calls_close(self, mock_openai):
        from ragline.providers.embedding import EmbeddingClient

        _, mock_client = mock_openai

        with EmbeddingClient(
            base_url="http://localhost/v1",
            api_key="key",
            model="m",
        ):
            pass

        mock_client.close.assert_called_once()


class TestEmbeddingClientOrdering:
    """测试 8: 顺序保持 — batch_size=2，4 个文本，验证 embedding 对应顺序。"""

    def test_order_preserved_across_batches(self, mock_openai):
        from ragline.providers.embedding import EmbeddingClient

        _, mock_client = mock_openai
        batch1_emb = [[1.0, 0.0], [0.0, 1.0]]
        batch2_emb = [[2.0, 0.0], [0.0, 2.0]]
        mock_client.embeddings.create.side_effect = [
            make_embedding_response(batch1_emb),
            make_embedding_response(batch2_emb),
        ]

        client = EmbeddingClient(
            base_url="http://localhost/v1",
            api_key="key",
            model="m",
            batch_size=2,
        )
        result = client.embed(["a", "b", "c", "d"])

        assert result == batch1_emb + batch2_emb
        assert result[0] == [1.0, 0.0]
        assert result[2] == [2.0, 0.0]
