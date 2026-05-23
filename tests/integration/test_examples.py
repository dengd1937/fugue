"""tests/integration/test_examples.py — examples/ 目录的 subprocess 回归测试。"""

import re
import subprocess
import sys
from pathlib import Path

import pytest


def test_consumer_minimal_runs_clean() -> None:
    """验证 examples/quickstart/consumer_minimal.py 能正常运行并输出预期内容。"""
    script = Path(__file__).resolve().parents[2] / "examples" / "quickstart" / "consumer_minimal.py"

    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=60.0,
    )

    if result.returncode != 0:
        pytest.fail(result.stdout + "\n---STDERR---\n" + result.stderr)

    assert result.returncode == 0
    assert "INGESTED:" in result.stdout
    assert "ANSWER:" in result.stdout
    assert "Ragline is a config-driven RAG library." in result.stdout

    match = re.search(r"INGESTED: (\d+) chunks", result.stdout)
    assert match is not None, f"INGESTED 行未找到，stdout: {result.stdout}"
    assert int(match.group(1)) > 0, f"INGESTED chunks 应 > 0，实际 stdout: {result.stdout}"

    assert "Traceback" not in result.stderr, f"stderr 中发现 Traceback：\n{result.stderr}"
