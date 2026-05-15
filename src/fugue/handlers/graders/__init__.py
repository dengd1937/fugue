"""src/fugue/handlers/graders/__init__.py — 注册函数 + re-export。"""

from fugue.handlers.graders.normalizer import normalize_score
from fugue.handlers.graders.score import score_grader
from fugue.registry import grader_registry


def register_graders() -> None:
    """注册 score grader 到 grader_registry。无依赖（不需要 LLM/provider）。"""
    grader_registry.register("score", score_grader)


# 模块 import 即注册（无依赖时直接执行副作用）
register_graders()


__all__ = ["normalize_score", "register_graders", "score_grader"]
