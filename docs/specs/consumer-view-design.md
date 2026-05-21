---
feature: consumer-view
spec: docs/specs/consumer-view-design.md
routing: Development Workflow
---

# Consumer View — Feature Spec

## 一句话定义

在 ragline 仓库内同时建立"对外的消费者支持"与"对内的契约守护"——让外部项目能轻松引用并测试 ragline，同时让 ragline 自身的迭代不会偷偷破坏对外可用性。

---

## 背景与动机

ragline 当前已具备完整的对外 API（顶层 re-export 完整、错误类型已 export、wheel 配置正确），但仍有三处缺口：

1. **缺 `py.typed`** —— 仓库内 mypy strict 是给自己用的，外部项目即便 import ragline，mypy 也只能拿到 `Any`。
2. **缺面向消费者的测试工具** —— 现有 8+ 个测试文件分别手抄 fake LLM / fake Embedding / `clean_*_registry` fixture（约 150 行重复代码），外部用户想做同样的事必须从零手写。
3. **缺"外部使用 ragline"的可运行示例与回归保护** —— 现有 e2e 全部要 `OPENAI_API_KEY` 才能跑；没有一个"不依赖真实 API、能验证 ragline 对外契约"的入口。一旦顶层 API 重命名 / 签名变动，没人会立刻发现。

本 spec 的产出同时解决这三处缺口，并通过 dogfooding 把对外契约绑到内部回归测试上。

---

## 目标用户与场景

### 两层用户（同等重要）

| 用户 | 痛点 | 本 spec 提供 |
|---|---|---|
| **外部下游工程师** | 想在自己项目里 `import ragline` 并写测试，但不知道怎么 mock provider、不知道怎么避免 registry 污染、缺少现成 demo | `ragline.testing` 公开模块 + `examples/quickstart/` 可运行示例 + README 章节 |
| **ragline 维护者** | 改 API 时可能偷偷破坏对外契约，没有自动闸门 | `tests/integration/test_examples.py` subprocess 回归 + 内部测试直接消费 `ragline.testing`（dogfooding） |

### 核心场景

1. **外部测试**：外部用户在自己的 pytest 套件里 `from ragline.testing import FakeLLM, isolated_registries, mock_rag_providers`，3–5 行代码就能跑完 RAG ingest+query 流程，不打任何真实 API。
2. **外部上手**：新用户照抄 `examples/quickstart/`，无需 API key 就能跑出第一个 demo，然后顺势替换为真实 provider。
3. **内部回归**：ragline 维护者改 API 后跑 `pytest tests/integration/test_examples.py`，一旦 examples 跑不通就 fail —— 物理保证对外契约。
4. **内部一致**：ragline 自己的测试套件用 `ragline.testing` 提供的同一套 fixture，确保对外 API 真正可用。

---

## MVP 范围（全部 P0）

| ID | 功能 | 验收 |
|---|---|---|
| F1 | `src/ragline/testing.py` 公开模块：`FakeLLM` / `FakeEmbedding` 类 + `isolated_registries` / `mock_rag_providers` 上下文管理器 | `from ragline.testing import FakeLLM, FakeEmbedding, isolated_registries, mock_rag_providers` 能用；类型可被外部 mypy 检查 |
| F2 | `examples/quickstart/`：`consumer_minimal.py` + `consumer.yaml` + 几份示例 md + 短 README | 直接 `python examples/quickstart/consumer_minimal.py` 退出码 0，stdout 含 `INGESTED:` 与固定 answer |
| F3 | `tests/integration/test_examples.py`：subprocess 跑 examples | 新增测试通过，断言 returncode + stdout 关键字 |
| F4 | `src/ragline/py.typed` + hatch wheel `force-include` + README 加"在外部项目中使用 ragline" / "在测试中使用 ragline"两节 | `python -m zipfile -l dist/*.whl \| grep py.typed` 命中 |
| F5 | 顶层 `tests/conftest.py` + 重构 11 个测试文件去除 `clean_*_registry` 重复 | `pytest` 全绿、覆盖率 ≥ 90%、`tests/` 净减 ≥ 100 行 |

---

## 明确不做的边界

- 不发布到 PyPI（保持本地 / git 安装路径）
- 不做 wheel-install smoke（打 wheel → 临时 venv → 装 → 跑），保留为 Future enhancement
- 不做 uv workspace 改造
- 不在 `examples/` 下放需要真实 API key 的示例（这归 e2e 管）
- 不改 `RAG.__init__` 加 `provider_overrides=` 参数（独立 brainstorm 范围，留 Future enhancement）
- 不动 e2e/integration 现有测试的语义（仅 Task 2 阶段重构 fixture 复用，行为不变）

---

## 技术设计

### 架构总览

```
src/ragline/
  testing.py            ← 新增（F1）。public API：fake provider + registry 隔离
  py.typed              ← 新增（F4）。PEP 561 类型标记

examples/
  quickstart/           ← 新增（F2）
    consumer_minimal.py
    consumer.yaml
    docs/
      doc1.md, doc2.md, doc3.md
    README.md

tests/
  conftest.py           ← 新增（F5）。顶层共享 fixture
  unit/test_testing.py  ← 新增（F1）。ragline.testing 自身单测
  integration/
    test_examples.py    ← 新增（F3）。subprocess 跑 examples

pyproject.toml          ← 修改（F4）。hatch.build.targets.wheel.force-include
README.md               ← 修改（F4）。新增两节
```

**架构约束遵守**：

- `ragline.testing` 不依赖 `ragline.engine`（与 `.importlinter` 约束一致），只调顶层 / `ragline.registry` / `ragline.api.*`
- 顶层 re-export（`src/ragline/__init__.py`）暂不动；外部用户走 `from ragline.testing import ...` 是显式独立入口，符合"`ragline.testing` 不是核心运行时"的语义

### 组件接口

**`src/ragline/testing.py`**（草案 API，签名锁定，实现细节可调）：

```python
class FakeLLM:
    """LLMClient-compatible fake. Returns fixed answer for any complete() call.

    Records each complete() call into self.calls and each close() call into
    self.close_calls (int counter) — both for downstream test assertions.
    Signature MUST exactly match ragline.providers.llm.LLMClient.complete
    (temperature is keyword-only).
    """
    def __init__(self, answer: str = "fake answer") -> None: ...
    def complete(self, prompt: str, *, temperature: float = 0.7) -> str: ...
    def close(self) -> None: ...

    answer: str
    calls: list[tuple[str, dict[str, Any]]]
    close_calls: int


class FakeEmbedding:
    """EmbeddingClient-compatible fake. Returns deterministic dim-vectors.

    Records each embed() call into self.calls and each close() call into
    self.close_calls (int counter).
    """
    def __init__(self, dim: int = 8) -> None: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def close(self) -> None: ...

    dim: int
    calls: list[list[str]]
    close_calls: int


@contextmanager
def isolated_registries() -> Iterator[None]:
    """Snapshot all 7 registries → yield → restore.

    Wrap any test that instantiates RAG() to prevent handler-closure
    overwrite (ragline's registries are global singletons).
    """


@contextmanager
def mock_rag_providers(
    *, llm: FakeLLM | None = None, embedding: FakeEmbedding | None = None,
) -> Iterator[tuple[FakeLLM, FakeEmbedding]]:
    """Patch ragline.api.rag.LLMClient & EmbeddingClient with fakes.

    Yields the fake instances so tests can assert against their .calls history.
    Composes with isolated_registries() — use both for full RAG() isolation.
    """
```

**`examples/quickstart/consumer_minimal.py`**（最小可运行 demo 骨架）：

```python
"""消费者示例：完全不依赖真实 API key。"""
import tempfile
from pathlib import Path

from ragline import RAG, GraphConfig, IngestConfig, RaglineConfig
from ragline.testing import isolated_registries, mock_rag_providers


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp, \
         isolated_registries(), \
         mock_rag_providers() as (llm, _):
        llm.answer = "Ragline is a config-driven RAG library."
        cfg = RaglineConfig(
            graph=GraphConfig(retrievers=["vector"], processors=[], grade_threshold=0.01),
            ingest=IngestConfig(persist_dir=str(Path(tmp) / "chroma")),
        )
        cfg.providers.llm_api_key = "fake-key-for-demo"
        with RAG(cfg) as rag:
            result = rag.ingest(Path(__file__).parent / "docs", show_progress=False)
            print(f"INGESTED: {result.num_chunks} chunks")
            qr = rag.query("What is ragline?")
            print(f"ANSWER: {qr.answer}")


if __name__ == "__main__":
    main()
```

**`tests/integration/test_examples.py`**（subprocess 回归骨架）：

```python
"""subprocess 跑 examples/，作为对外契约回归。"""
import re
import subprocess
import sys
from pathlib import Path

EXAMPLES = Path(__file__).parent.parent.parent / "examples"


def test_consumer_minimal_runs_clean() -> None:
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / "quickstart" / "consumer_minimal.py")],
        capture_output=True, text=True, check=False, timeout=60.0,
    )
    if result.returncode != 0:
        import pytest
        pytest.fail(result.stdout + "\n---STDERR---\n" + result.stderr)
    assert "INGESTED:" in result.stdout
    assert "ANSWER:" in result.stdout
    assert "Ragline is a config-driven RAG library." in result.stdout
    m = re.search(r"INGESTED: (\d+) chunks", result.stdout)
    assert m and int(m.group(1)) > 0
    assert "Traceback" not in result.stderr
```

**`pyproject.toml` 改动**：

```toml
[tool.hatch.build.targets.wheel.force-include]
"src/ragline/py.typed" = "ragline/py.typed"
```

### 数据流

**外部用户测试自己的代码**：
```
user_test.py
  → import ragline + ragline.testing
  → with isolated_registries(), mock_rag_providers() as (llm, emb):
        rag = RAG(cfg)
        result = rag.query(...)
        assert "..." in result.answer
        assert len(llm.calls) >= 1
```
全程不发起任何外部网络请求。

**ragline 内部回归保护**：
```
pytest
  → tests/integration/test_examples.py::test_consumer_minimal_runs_clean
  → subprocess.run([python, examples/quickstart/consumer_minimal.py])
       └─ 子进程：import ragline + ragline.testing → ingest+query → print stdout
  → assert returncode == 0
  → assert "INGESTED:" / fixed answer in stdout
```
**关键点**：subprocess 隔离子进程的 import，模拟"外部 Python 进程"的真实启动状态（复用 `tests/integration/test_optional_deps.py:123-131` 的成熟模式）。

**Task 2 dogfooding 流向**：
```
tests/conftest.py
  → from ragline.testing import isolated_registries, mock_rag_providers
  → 注册顶层共享 fixture
tests/unit/test_*.py, tests/integration/test_*.py
  → 删除本地 clean_*_registry
  → 改用 isolated_registries() 上下文 / 顶层 fixture
```

### 错误处理

| 场景 | 处理 |
|---|---|
| `examples/quickstart/consumer_minimal.py` 抛异常 | subprocess 退出码非 0 → `pytest.fail(stdout + stderr)`（复用 `test_optional_deps.py:130-131` 模式） |
| `FakeLLM.complete` 被传入异常类型 | 不做防御，trust input；类型注解约束 |
| `isolated_registries()` 恢复阶段抛错 | `try/finally` 保证 restore；`register` 抛 `RaglineRegistryError` 时吞掉（已 unregister 完毕，状态可接受） |
| `mock_rag_providers` 在用户已自行 inject client 的场景失效 | 文档说明：mock 走 patch 构造器；如已 inject 直接传 `FakeLLM()` 实例即可 |
| `py.typed` 未进 wheel | Task 1 验收：`python -m zipfile -l dist/*.whl \| grep py.typed` 必须命中 |
| 外部用户不调 `isolated_registries()` 多次实例化 RAG | 既有 `Multiple RAG instances` warning 保留；README 显式说明这是 known limitation |

### 测试策略

**Task 1 新增测试**：

1. `tests/unit/test_testing.py`（新增）—— `ragline.testing` 自身的单测：
   - `FakeLLM.complete` 返回固定 answer、call 被记录到 `.calls`
   - `FakeLLM.close` 不抛错
   - `FakeEmbedding.embed` 返回正确维度、对空输入 `embed([])` 返回 `[]`
   - `isolated_registries` 进入清空、yield 中所有 registry 为空、退出恢复原 handler
   - `mock_rag_providers` 让 `RAG()` 实例使用 fake；验证 fake 的 `.calls` 在 `query()` 后非空；验证全程不发起真实网络请求（patch 验证）
2. `tests/integration/test_examples.py`（新增）—— subprocess 跑 `examples/quickstart/consumer_minimal.py`

**Task 2 重构后的验证**：
1. 11 个测试文件改用 `ragline.testing` 后，**整体 pytest 通过数等于 Task 1 完成后的基数**（重构不增不减）+ **覆盖率 ≥ 90%**
2. `git diff --stat tests/` 验证净减 ≥ 100 行
3. `grep -rE 'def clean_.*_registry' tests/` 0 命中（本地 fixture 定义完全消除）

**回归门**：
- 既有 e2e 测试（4 个）一行不动，跑通即说明 public API 没破
- `.importlinter` 检查 `ragline.testing` 不引入对 `ragline.engine` 的 import

---

## 实施 Task 拆分

按 L1 铁律，每个 task 独立走 TDD → 实现 → 审查闭环。

### Task 1 — 对外契约

**范围**：F1 + F2 + F3 + F4

**交付物**：
- `src/ragline/testing.py` + 单测
- `src/ragline/py.typed`
- `examples/quickstart/` 全套
- `tests/integration/test_examples.py`
- `pyproject.toml` 加 force-include
- `README.md` 加两节

**验收**：
- `pytest` 全绿（253 + N passed）
- 覆盖率 ≥ 90%
- `python -m zipfile -l dist/*.whl | grep py.typed` 命中
- `.importlinter` 通过
- `python examples/quickstart/consumer_minimal.py` 独立可运行

### Task 2 — 内部去重（dogfooding）

**依赖**：Task 1 完成且 merge 到主线

**范围**：F5

**交付物**：
- `tests/conftest.py`（顶层共享 fixture）
- 重构 `tests/integration/test_rag.py`、`tests/integration/test_engine_graph.py`、`tests/unit/test_handlers/test_*.py`（含 chunkers/generators/graders/parsers/processors/retrievers/transforms 共 7 个）、`tests/unit/test_engine/test_nodes_post_generate.py`、`tests/unit/test_engine/test_nodes_qt_retrieve.py` —— **共 11 个文件**

**验收**：
- `pytest` 通过数与 Task 1 完成后基数一致（重构不增不减）
- 覆盖率 ≥ 90%（不退化）
- `git diff --stat tests/` 净减 ≥ 100 行
- `grep -rE 'def clean_.*_registry' tests/` 0 命中

---

## Future Enhancements（不在本 spec 范围）

| 项目 | 触发条件 |
|---|---|
| Wheel-install smoke：打 wheel → 临时 venv → 装 wheel → 跑 consumer_minimal | 准备首次 PyPI 发布前 |
| `RAG.__init__(provider_overrides=)` 让用户直接 inject client 实例 | 独立 brainstorm；当多个用户反馈"patch 模式不够清爽"时再启动 |
| `ragline.testing` 暴露 `FakeReranker` / `FakeVectorStore` | 当外部用户开始测试自定义 processor / retriever 时再加 |
| 顶层 re-export 加入 `create_endpoints`（来自 `ragline.server.endpoints`） | 当外部用户自定义 FastAPI app 装配 ragline endpoints 的需求出现时 |

---

## 路由决策

**Development Workflow**（无 UI 改动；纯库代码 + 测试 + 文档）。

理由：
- 全部产出物均为后端 Python 代码、配置、Markdown 文档；无任何前端 / UI / 视觉元素
- 直接进入 `writing-plans` skill 生成 task-level 实施计划
- Task 1 / Task 2 已在本 spec 中显式拆分，writing-plans 可直接消费
