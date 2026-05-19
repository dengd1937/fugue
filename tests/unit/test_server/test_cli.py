"""tests/unit/test_server/test_cli.py — ragline CLI 单元测试。"""

from unittest.mock import MagicMock, patch

import pytest

from ragline.server.cli import _build_parser, main

# argparse 解析测试 -----------------------------------------------


def test_parser_serve_basic() -> None:
    """ragline serve --config x.yaml --port 9000 解析正确。"""
    parser = _build_parser()
    args = parser.parse_args(["serve", "--config", "x.yaml", "--port", "9000"])
    assert args.cmd == "serve"
    assert args.config == "x.yaml"
    assert args.port == 9000
    assert args.host == "127.0.0.1"  # 默认


def test_parser_custom_host() -> None:
    parser = _build_parser()
    args = parser.parse_args(["serve", "--config", "x.yaml", "--host", "0.0.0.0"])
    assert args.host == "0.0.0.0"


def test_parser_missing_config() -> None:
    """缺 --config 应触发 argparse SystemExit。"""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["serve"])


def test_parser_unknown_subcommand() -> None:
    """未知子命令应 SystemExit。"""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["unknown"])


def test_parser_missing_subcommand() -> None:
    """缺子命令应 SystemExit（required=True）。"""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_invalid_port_type() -> None:
    """--port 非整数应 SystemExit。"""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "--config", "x.yaml", "--port", "not-a-number"])


# main 函数测试（mock uvicorn + create_app） -----------------------


def test_main_invokes_uvicorn_with_app() -> None:
    """main 调用 create_app 加载配置，然后 uvicorn.run(app, host, port, workers=1)。"""
    mock_app = MagicMock()
    with (
        patch("ragline.server.cli.create_app", return_value=mock_app) as mock_create,
        patch("ragline.server.cli.uvicorn") as mock_uvicorn,
    ):
        main(["serve", "--config", "test.yaml", "--port", "9001", "--host", "0.0.0.0"])
        mock_create.assert_called_once_with("test.yaml")
        mock_uvicorn.run.assert_called_once()
        call_args = mock_uvicorn.run.call_args
        # 第一个 positional 参数是 app
        assert call_args.args[0] is mock_app
        # workers 必须强制为 1
        assert call_args.kwargs["workers"] == 1
        assert call_args.kwargs["host"] == "0.0.0.0"
        assert call_args.kwargs["port"] == 9001


def test_main_warns_about_no_auth(caplog) -> None:
    """main 启动时应记录 WARNING 提示无认证。"""
    import logging

    mock_app = MagicMock()
    with (
        patch("ragline.server.cli.create_app", return_value=mock_app),
        patch("ragline.server.cli.uvicorn"),
        caplog.at_level(logging.WARNING, logger="ragline.server"),
    ):
        main(["serve", "--config", "x.yaml"])
    assert any("no authentication" in r.message for r in caplog.records)
    assert any("workers=1" in r.message for r in caplog.records)


def test_main_no_args_uses_sys_argv(monkeypatch) -> None:
    """main 不传 argv 时从 sys.argv 读取。"""
    import sys

    mock_app = MagicMock()
    monkeypatch.setattr(sys, "argv", ["ragline", "serve", "--config", "/tmp/x.yaml"])
    with (
        patch("ragline.server.cli.create_app", return_value=mock_app),
        patch("ragline.server.cli.uvicorn") as mock_uvicorn,
    ):
        main()
        mock_uvicorn.run.assert_called_once()
