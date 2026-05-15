"""tests/unit/test_providers/test_llm.py — LLMClient 单元测试。"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from fugue.api.types import FugueLLMError
from fugue.providers.llm import LLMClient

# ---------------------------------------------------------------------------
# 测试 1：基础调用
# ---------------------------------------------------------------------------


def test_complete_basic_call() -> None:
    """complete() 正确调用 chat.completions.create 并返回内容。"""
    with patch("fugue.providers.llm.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "hello world"
        mock_client.chat.completions.create.return_value = mock_resp

        llm = LLMClient(base_url="http://localhost", api_key="test-key", model="test-model")
        result = llm.complete("hello")

        assert result == "hello world"
        mock_client.chat.completions.create.assert_called_once_with(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.7,
        )


# ---------------------------------------------------------------------------
# 测试 2：失败包装
# ---------------------------------------------------------------------------


def test_complete_wraps_exception_as_llm_error() -> None:
    """chat.completions.create 抛异常时，complete() 包装为 FugueLLMError。"""
    with patch("fugue.providers.llm.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        original_exc = Exception("network error")
        mock_client.chat.completions.create.side_effect = original_exc

        llm = LLMClient(base_url="http://localhost", api_key="test-key", model="test-model")

        with pytest.raises(FugueLLMError, match="LLM call failed"):
            llm.complete("x")

        # 验证原始异常通过 __cause__ 链可访问
        try:
            llm.complete("x")
        except FugueLLMError as exc:
            assert exc.__cause__ is original_exc


# ---------------------------------------------------------------------------
# 测试 3：Semaphore 并发控制
# ---------------------------------------------------------------------------


def test_semaphore_concurrency_limit() -> None:
    """max_concurrent=2 时，最多 2 个线程同时执行 LLM 调用。"""
    entered_count = 0
    entered_lock = threading.Lock()
    entered = threading.Semaphore(0)  # 用于通知主线程"有 worker 进入了 mock"
    release = threading.Event()

    def mock_create(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal entered_count
        with entered_lock:
            entered_count += 1
        entered.release()  # 通知主线程
        release.wait(timeout=5.0)  # 等主线程信号才返回
        resp = MagicMock()
        resp.choices[0].message.content = "ok"
        return resp

    with patch("fugue.providers.llm.OpenAI") as mock_openai_cls:
        mock_client_inst = MagicMock()
        mock_client_inst.chat.completions.create.side_effect = mock_create
        mock_openai_cls.return_value = mock_client_inst

        client = LLMClient(base_url="http://x", api_key="k", model="m", max_concurrent=2)

        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = [ex.submit(client.complete, f"q{i}") for i in range(3)]

            # 等待恰好 2 个进入（Semaphore 限制）
            for _ in range(2):
                acquired = entered.acquire(timeout=2.0)
                assert acquired, "expected 2 workers to enter within timeout"

            # 此时第 3 个 worker 应该被 Semaphore 阻塞
            # 短暂等待确认第 3 个未进入（反向证明，≤100ms）
            time.sleep(0.05)
            assert entered_count == 2, f"expected only 2 workers inside, got {entered_count}"

            # 释放所有阻塞的 worker
            release.set()

            results = [f.result(timeout=5.0) for f in futures]

        assert all(r == "ok" for r in results)
        assert entered_count == 3, "expected all 3 workers to eventually run"


# ---------------------------------------------------------------------------
# 测试 4：temperature 透传
# ---------------------------------------------------------------------------


def test_complete_temperature_passthrough() -> None:
    """complete() 将 temperature 参数正确透传给 chat.completions.create。"""
    with patch("fugue.providers.llm.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = "answer"
        mock_client.chat.completions.create.return_value = mock_resp

        llm = LLMClient(base_url="http://localhost", api_key="test-key", model="gpt-4")
        llm.complete("x", temperature=0.1)

        _, call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs["temperature"] == 0.1


# ---------------------------------------------------------------------------
# 测试 5：close
# ---------------------------------------------------------------------------


def test_close_delegates_to_underlying_client() -> None:
    """close() 调用底层 OpenAI client 的 close()。"""
    with patch("fugue.providers.llm.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        llm = LLMClient(base_url="http://localhost", api_key="test-key", model="gpt-4")
        llm.close()

        mock_client.close.assert_called_once()


# ---------------------------------------------------------------------------
# 测试 6：context manager 协议
# ---------------------------------------------------------------------------


def test_context_manager_closes_client() -> None:
    """with 语句退出时，__exit__ 应调用底层 client 的 close()。"""
    with patch("fugue.providers.llm.OpenAI") as mock_openai_cls:
        mock_client_inst = MagicMock()
        mock_openai_cls.return_value = mock_client_inst

        with LLMClient(base_url="http://x", api_key="k", model="m") as client:
            assert client is not None

        mock_client_inst.close.assert_called_once()
