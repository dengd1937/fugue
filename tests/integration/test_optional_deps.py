"""tests/integration/test_optional_deps.py — 跨模块缺失可选依赖回归测试。

使用子进程隔离验证：pypdf / FlagEmbedding 缺失时，fugue 模块仍可正常导入，
只在实际调用相关功能时才抛出带安装提示的 ImportError。
"""

import subprocess
import sys
import textwrap

import pytest

# ---------------------------------------------------------------------------
# 子进程脚本：注入缺失哨兵后逐条验证
# ---------------------------------------------------------------------------

_SCRIPT = textwrap.dedent("""\
    import sys

    # 1. 注入哨兵：模拟 pypdf / FlagEmbedding 未安装
    sys.modules["pypdf"] = None          # type: ignore[assignment]
    sys.modules["FlagEmbedding"] = None  # type: ignore[assignment]

    # -----------------------------------------------------------------------
    # 验证 1：注入哨兵后 import fugue 成功（不抛 ModuleNotFoundError）
    # -----------------------------------------------------------------------
    try:
        import fugue
    except Exception as e:
        print(f"[FAIL] import fugue raised: {e}", file=sys.stderr)
        raise SystemExit(1)

    # -----------------------------------------------------------------------
    # 验证 2：register_parsers() 成功（pdf_parser 本身不调用 require）
    # -----------------------------------------------------------------------
    try:
        from fugue.handlers.parsers import register_parsers
        register_parsers()
    except Exception as e:
        print(f"[FAIL] register_parsers() raised: {e}", file=sys.stderr)
        raise SystemExit(1)

    # -----------------------------------------------------------------------
    # 验证 3：processors 模块可导入（不构造 reranker）；pdf 已注册
    # -----------------------------------------------------------------------
    try:
        from fugue.handlers.processors import register_processors  # noqa: F401
    except Exception as e:
        print(f"[FAIL] import register_processors raised: {e}", file=sys.stderr)
        raise SystemExit(1)

    from fugue.registry import parser_registry
    if not parser_registry.has("pdf"):
        print("[FAIL] parser_registry does not have 'pdf'", file=sys.stderr)
        raise SystemExit(1)

    # -----------------------------------------------------------------------
    # 验证 4：调用 pdf_fn(path) 抛 ImportError 且消息含安装提示
    # -----------------------------------------------------------------------
    import os
    import tempfile
    from pathlib import Path

    pdf_fn = parser_registry.get("pdf")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        pdf_fn(Path(tmp_path))
        print("[FAIL] pdf_fn should have raised ImportError but did not", file=sys.stderr)
        raise SystemExit(1)
    except ImportError as e:
        msg = str(e)
        if "pip install \'fugue[pdf]\'" not in msg:
            print(
                f"[FAIL] ImportError message does not contain install hint: {msg!r}",
                file=sys.stderr,
            )
            raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as e:
        print(f"[FAIL] pdf_fn raised unexpected exception type {type(e).__name__}: {e}", file=sys.stderr)
        raise SystemExit(1)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # -----------------------------------------------------------------------
    # 验证 5：BGEReranker 构造抛 ImportError 且消息含安装提示
    # -----------------------------------------------------------------------
    try:
        from fugue.providers.reranker.bge import BGEReranker
        BGEReranker(model_name="x", device="cpu")
        print("[FAIL] BGEReranker() should have raised ImportError but did not", file=sys.stderr)
        raise SystemExit(1)
    except ImportError as e:
        msg = str(e)
        if "pip install \'fugue[bge]\'" not in msg:
            print(
                f"[FAIL] ImportError message does not contain install hint: {msg!r}",
                file=sys.stderr,
            )
            raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as e:
        print(f"[FAIL] BGEReranker raised unexpected exception type {type(e).__name__}: {e}", file=sys.stderr)
        raise SystemExit(1)

    # 全部通过 → 正常 exit 0
    print("[PASS] all checks passed", file=sys.stdout)
""")


# ---------------------------------------------------------------------------
# 父进程测试函数
# ---------------------------------------------------------------------------


def test_import_fugue_without_optional_deps() -> None:
    """在子进程中注入缺失哨兵，验证 fugue 可正常导入且缺依赖时抛出正确 ImportError。"""
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(result.stdout + result.stderr)
