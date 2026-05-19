"""src/ragline/handlers/__init__.py — 触发各子目录注册副作用。

graders/parsers/chunkers 在子目录 __init__.py import 时即注册（无依赖）；
transforms/retrievers/processors/generators 需要 RAG.__init__ 显式调用
register_*(client/provider) 完成注册（带闭包注入）。
"""

from ragline.handlers import (  # noqa: F401  # triggers side-effect registration
    chunkers,
    generators,
    graders,
    parsers,
    processors,
    retrievers,
    transforms,
)
