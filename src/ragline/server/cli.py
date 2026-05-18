"""src/ragline/server/cli.py — Ragline CLI entry point。"""

import argparse
import logging

import uvicorn

from ragline.server.app import create_app

logger = logging.getLogger("ragline.server")


def _build_parser() -> argparse.ArgumentParser:
    """构建 argparse parser（独立函数便于测试）。"""
    parser = argparse.ArgumentParser(prog="ragline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Start Ragline REST server")
    serve.add_argument("--config", required=True, help="Path to YAML config")
    serve.add_argument("--host", default="127.0.0.1", help="Bind host")
    serve.add_argument("--port", type=int, default=8000, help="Bind port")
    # 不暴露 --workers：MVP 强制 workers=1
    # 理由: 多 worker 下 Chroma 本地文件锁争用 + BM25 重复重建

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point。

    pyproject.toml [project.scripts] 配置：
        ragline = "ragline.server.cli:main"
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "serve":
        app = create_app(args.config)
        logger.warning(
            "⚠️ Ragline 0.x: no authentication, single-worker only. Deploy to trusted networks only. workers=1 enforced."
        )
        uvicorn.run(app, host=args.host, port=args.port, workers=1)
