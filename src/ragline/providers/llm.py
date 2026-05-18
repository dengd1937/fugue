"""src/ragline/providers/llm.py — OpenAI 兼容 LLM 客户端。"""

import threading

from openai import OpenAI

from ragline.api.types import FugueLLMError


class LLMClient:
    """OpenAI 兼容 LLM 统一封装。base_url + api_key 即可切换 provider。

    支持 GPT/DeepSeek/Kimi/Qwen/vLLM/Ollama 等所有 OpenAI 兼容 endpoint。
    通过 threading.Semaphore 控制最大并发数。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 60.0,
        max_retries: int = 2,
        max_concurrent: int = 10,
    ) -> None:
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._model = model
        self._sem = threading.Semaphore(max_concurrent)

    def complete(self, prompt: str, *, temperature: float = 0.7) -> str:
        """同步调用 chat.completions.create；失败抛 FugueLLMError。"""
        with self._sem:
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                raise FugueLLMError(f"LLM call failed: {e}") from e

    def close(self) -> None:
        """关闭底层 httpx client。"""
        self._client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()
