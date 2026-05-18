"""src/ragline/handlers/generators/__init__.py — 注册函数 + re-export。"""

from ragline.handlers.generators.basic import (
    BASIC_PROMPT_TEMPLATE,
    GeneratorFn,
    make_basic_generator,
)
from ragline.handlers.generators.citation import (
    CITATION_PROMPT_TEMPLATE,
    make_citation_generator,
)
from ragline.providers.llm import LLMClient
from ragline.registry import generator_registry


def register_generators(llm: LLMClient) -> None:
    """注册 basic + citation generator 到 generator_registry。"""
    generator_registry.register("basic", make_basic_generator(llm))
    generator_registry.register("citation", make_citation_generator(llm))


__all__ = [
    "BASIC_PROMPT_TEMPLATE",
    "CITATION_PROMPT_TEMPLATE",
    "GeneratorFn",
    "make_basic_generator",
    "make_citation_generator",
    "register_generators",
]
