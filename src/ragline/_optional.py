"""src/ragline/_optional.py — 可选依赖懒加载 helper。"""

import importlib
from typing import Any


def require(module_name: str, *, extra: str) -> Any:
    """尝试懒加载可选依赖模块，缺失时抛出带安装提示的 ImportError。

    返回类型刻意声明为 Any 而非 ModuleType：可选依赖（如 pypdf、FlagEmbedding）
    无类型 stub，调用方需对返回值做动态属性访问（如 mod.PdfReader、mod.FlagReranker）；
    若返回 ModuleType，mypy 会报 attr-defined 错误。Any 是动态可选导入 shim 的惯例设计。
    """
    try:
        return importlib.import_module(module_name)
    except (ModuleNotFoundError, ImportError) as e:
        raise ImportError(f"可选依赖 '{module_name}' 未安装。请运行: pip install 'ragline[{extra}]'") from e
