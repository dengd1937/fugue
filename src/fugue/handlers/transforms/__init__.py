"""src/fugue/handlers/transforms/__init__.py — 注册函数 + re-export。"""

from fugue.handlers.transforms.atoms import hyde_fn, rewrite_fn, step_back_fn
from fugue.handlers.transforms.pipeline import run_transform_branch
from fugue.providers.llm import LLMClient
from fugue.registry import transform_registry


def register_transforms(llm_client: LLMClient) -> None:
    """注册 rewrite / hyde / step_back 到 transform_registry。

    LLM 通过闭包注入，让 transform 函数签名保持 (queries, n) -> list[str]。
    """
    transform_registry.register(
        "rewrite",
        lambda queries, n: rewrite_fn(queries, n, llm_client),
    )
    transform_registry.register(
        "hyde",
        lambda queries, n: hyde_fn(queries, n, llm_client),
    )
    transform_registry.register(
        "step_back",
        lambda queries, n: step_back_fn(queries, n, llm_client),
    )


__all__ = [
    "hyde_fn",
    "register_transforms",
    "rewrite_fn",
    "run_transform_branch",
    "step_back_fn",
]
