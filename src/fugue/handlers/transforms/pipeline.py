"""src/fugue/handlers/transforms/pipeline.py — 嵌套 transform 执行器。"""

from collections.abc import Callable
from typing import TYPE_CHECKING

from fugue.api.types import TransformResult

if TYPE_CHECKING:
    from fugue.registry import Registry


def _extract_query(item: str | TransformResult) -> str:
    """如果 item 是 TransformResult，返回 .query；否则返回 item 本身。"""
    if isinstance(item, TransformResult):
        return item.query
    return item


def run_transform_branch(
    branch: str | list[str],
    queries: list[str],
    n: int,
    registry: "Registry[Callable[..., list[str | TransformResult]]]",
) -> list[str | TransformResult]:
    """执行单个 transform 分支。

    - str: 原子 transform，调用 registry.get(branch)(queries, n)
    - list[str]: 管道链，按顺序串联，前一个的输出作为后一个的输入
      若中间结果是 TransformResult，提取 .query 继续传递
    """
    if isinstance(branch, str):
        transform_fn = registry.get(branch)
        return list(transform_fn(queries, n))

    # 管道：list[str]
    current: list[str] = list(queries)
    last_result: list[str | TransformResult] = []
    for step_name in branch:
        transform_fn = registry.get(step_name)
        last_result = list(transform_fn(current, n))
        # 提取 .query 给下一步用作输入
        current = [_extract_query(r) for r in last_result]
    return last_result
