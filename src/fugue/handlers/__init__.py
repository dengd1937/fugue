"""src/fugue/handlers/__init__.py — 顶层 import 触发子模块注册副作用。

各子模块的 __init__.py 在 import 时执行 registry 注册（部分需要 LLM
注入的延迟到 register_*(client) 调用）。
"""
