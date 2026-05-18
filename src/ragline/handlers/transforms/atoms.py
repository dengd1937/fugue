"""src/ragline/handlers/transforms/atoms.py — Query Transform 原子函数。"""

from ragline.providers.llm import LLMClient

REWRITE_PROMPT = "请将以下问题改写成 {n} 个不同的表述，每行一个，不要编号，不要解释：\n{query}"

HYDE_PROMPT = "针对以下问题生成 {n} 段假设性的简短回答，每段用空行分隔，用于检索相关文档：\n{query}"

STEP_BACK_PROMPT = "请将以下问题抽象为 {n} 个更宏观的上位问题，每行一个，不要编号，不要解释：\n{query}"


def _split_lines(text: str) -> list[str]:
    """按换行符切分并去空白。"""
    return [line.strip() for line in text.splitlines() if line.strip()]


def _split_paragraphs(text: str) -> list[str]:
    """按空行切分（用于 hyde 多段回答）。"""
    paragraphs = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append("\n".join(current).strip())
                current = []
        else:
            current.append(stripped)
    if current:
        paragraphs.append("\n".join(current).strip())
    return [p for p in paragraphs if p]


def rewrite_fn(queries: list[str], n: int, llm: LLMClient) -> list[str]:
    """对每个 query 改写成 n 个不同表述。返回所有改写结果。"""
    if not queries:
        return []
    results: list[str] = []
    for q in queries:
        prompt = REWRITE_PROMPT.format(n=n, query=q)
        text = llm.complete(prompt)
        lines = _split_lines(text)
        results.extend(lines[:n])
    return results


def hyde_fn(queries: list[str], n: int, llm: LLMClient) -> list[str]:
    """对每个 query 生成 n 段假设性回答。返回所有段落。"""
    if not queries:
        return []
    results: list[str] = []
    for q in queries:
        prompt = HYDE_PROMPT.format(n=n, query=q)
        text = llm.complete(prompt)
        paragraphs = _split_paragraphs(text)
        results.extend(paragraphs[:n])
    return results


def step_back_fn(queries: list[str], n: int, llm: LLMClient) -> list[str]:
    """对每个 query 抽象为 n 个上位问题。返回所有上位问题。"""
    if not queries:
        return []
    results: list[str] = []
    for q in queries:
        prompt = STEP_BACK_PROMPT.format(n=n, query=q)
        text = llm.complete(prompt)
        lines = _split_lines(text)
        results.extend(lines[:n])
    return results
