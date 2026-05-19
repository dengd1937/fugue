"""src/ragline/handlers/generators/citation.py — Citation Generator 工厂。"""

from collections.abc import Callable
from typing import Any

from ragline.api.types import Document
from ragline.providers.llm import LLMClient

GeneratorFn = Callable[..., str]


CITATION_PROMPT_TEMPLATE = (
    "基于以下上下文回答问题，每个论点必须标注来源编号 [1][2]...。\n"
    "来源编号已在每段文档前标注。\n\n"
    "上下文：\n{context}\n\n"
    "问题：{query}"
)


def make_citation_generator(llm: LLMClient) -> GeneratorFn:
    """返回 citation generator 闭包，闭包绑定 LLM client。"""

    def citation_fn(
        query: str,
        docs: list[Document],
        temperature: float,
        **kwargs: Any,  # noqa: ARG001
    ) -> str:
        context = "\n\n".join(f"[{i + 1}] {d['content']}" for i, d in enumerate(docs))
        prompt = CITATION_PROMPT_TEMPLATE.format(context=context, query=query)
        return llm.complete(prompt, temperature=temperature)

    return citation_fn
