"""src/ragline/handlers/generators/basic.py — Basic Generator 工厂。"""

from collections.abc import Callable
from typing import Any

from ragline.api.types import Document
from ragline.providers.llm import LLMClient

GeneratorFn = Callable[..., str]


BASIC_PROMPT_TEMPLATE = "基于以下上下文回答问题。\n\n上下文：\n{context}\n\n问题：{query}"


def make_basic_generator(llm: LLMClient) -> GeneratorFn:
    """返回 basic generator 闭包，闭包绑定 LLM client。"""

    def basic_fn(
        query: str,
        docs: list[Document],
        temperature: float,
        **kwargs: Any,  # noqa: ARG001
    ) -> str:
        context = "\n\n".join(d["content"] for d in docs)
        prompt = BASIC_PROMPT_TEMPLATE.format(context=context, query=query)
        return llm.complete(prompt, temperature=temperature)

    return basic_fn
