"""src/fugue/server/app.py — FastAPI application factory。"""

import atexit
from pathlib import Path

from fastapi import FastAPI

from fugue.api.rag import RAG
from fugue.server.endpoints import create_endpoints


def create_app(config_path: str | Path) -> FastAPI:
    """从 YAML 配置创建 Fugue FastAPI app。

    MVP 采用同步实例化 + atexit 清理（不用 lifespan）：
    lifespan 在多 worker 下行为复杂，且 CLI 强制 workers=1，简化方案足够。
    """
    rag = RAG.from_yaml(config_path)
    app = FastAPI(title="Fugue", version="0.1.0")
    create_endpoints(app, rag)
    atexit.register(rag.close)
    return app
