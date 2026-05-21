# 实现计划：Consumer View

## 执行方式

本计划通过 `/subagent-driven-development` skill 执行。以下任务描述是 skill 的输入规格，不是直接执行指令。

## 概述

在 ragline 仓库内增加"消费者视角"能力：新增 `ragline.testing` 公开测试工具模块、`examples/quickstart/` 可运行示例、subprocess 回归测试、`py.typed` 类型标记；并把内部 11 个测试文件的本地 registry-隔离 fixture 替换为 `ragline.testing` 的实现以 dogfood 对外 API。

## 需求

依据 `docs/specs/consumer-view-design.md`：

- F1：`src/ragline/testing.py` 公开 `FakeLLM` / `FakeEmbedding` / `isolated_registries` / `mock_rag_providers`
- F2：`examples/quickstart/` 完整可运行示例，无 API key
- F3：`tests/integration/test_examples.py` subprocess 跑 examples 作为对外契约回归
- F4：`src/ragline/py.typed` + hatch wheel force-include + 顶层 README 两节
- F5：顶层 `tests/conftest.py` + 重构 11 个内部测试文件去除本地 `clean_*_registry`

## 架构变更

**新建**：
- `src/ragline/testing.py` —— 公开测试工具模块（Task 1.1）
- `src/ragline/py.typed` —— PEP 561 标记（Task 1.2）
- `tests/unit/test_testing.py` —— `ragline.testing` 自身单测（Task 1.1）
- `tests/integration/test_examples.py` —— subprocess 回归（Task 1.4）
- `tests/conftest.py` —— 顶层共享 fixture（Task 2.1）
- `examples/quickstart/consumer_minimal.py` + `consumer.yaml` + `docs/doc{1,2,3}.md` + `README.md` —— 可运行示例（Task 1.3）

**修改**：
- `pyproject.toml` —— hatch wheel `force-include`（Task 1.2）
- `README.md` —— 新增两节（Task 1.5）
- 11 个测试文件 —— 删除本地 `clean_*_registry`，改用 `ragline.testing` / 顶层 fixture（Task 2.2-2.5）

## 环境前置（Environment Prerequisites）

- **uv（包管理 + 虚拟环境）**
  - 验证：`uv --version`
  - 修复：参见 https://docs.astral.sh/uv/getting-started/installation/
- **pytest（测试运行器，dev extras）**
  - 验证：`uv run --extra dev pytest --version`
  - 修复：`uv sync --extra dev --extra all`（一并安装 server/bge/chroma/pdf 以便跑全套）
- **hatchling（构建后端，>=1.0 支持 force-include）**
  - 验证：`uv build --help 2>&1 | head -3`
  - 修复：项目 `[build-system]` 已声明 `requires = ["hatchling"]`，无需额外操作

无外部服务（DB / docker / 网络）。所有测试用 mock + `tmp_path`。

## 实现步骤

### 阶段 1（Task 1）：对外契约

5 个子任务，按依赖顺序串行。Task 1 完成且 merge 后才能启动 Task 2。

---

### 任务 1.1：实现 `ragline.testing` 公开模块

**文件：**
- 新建：`src/ragline/testing.py`
- 新建：`tests/unit/test_testing.py`

**测试规格**（TDD RED 阶段先写）：

`tests/unit/test_testing.py` 需覆盖以下场景与预期：

| # | 场景 | 预期 |
|---|---|---|
| 1 | `FakeLLM().complete("q")` 返回默认 `"fake answer"` | 返回值匹配；`fake.calls == [("q", {"temperature": 0.7})]` |
| 2 | `FakeLLM(answer="hi").complete("q", temperature=0.1)` keyword-only 透传 | `fake.calls == [("q", {"temperature": 0.1})]`；以位置参形式 `complete("q", 0.1)` 必须抛 `TypeError`（验证 `temperature` 是 keyword-only） |
| 3 | `FakeLLM.close()` 计数与幂等 | 连续调用两次 `close()`：`fake.close_calls == 2`，过程不抛错 |
| 4 | `FakeLLM` 运行时 `.answer` 可写并即时生效 | `f = FakeLLM(); f.answer = "X"; f.complete("q") == "X"`（消费者代码可在进入 `mock_rag_providers` 后改 answer，必须确保 `complete()` 读 `self.answer` 而非 closure 缓存） |
| 5 | `FakeEmbedding(dim=4).embed(["a", "b"])` | 返回 `[[0.1]*4, [0.1]*4]`；`fake.calls == [["a", "b"]]` |
| 6 | `FakeEmbedding().embed([])` | 返回 `[]`；`fake.calls == [[]]`（仍记录调用，保持与 LLM 行为一致） |
| 7 | `FakeEmbedding.close()` 计数与幂等 | 连续调用两次 `close()`：`fake.close_calls == 2`，不抛错 |
| 8 | `isolated_registries()` 上下文 | enter 后 7 个 registry 全空（`reg.names() == []`）；exit 后恢复原 handler 名称集合与 callable identity |
| 9 | `isolated_registries()` 嵌套调用 | 内层 enter/exit 后外层 yield 时 registry 仍为空（语义一致）；外层 exit 后完全恢复（验证 try/finally 工作） |
| 10 | `mock_rag_providers()` 让 `RAG()` 用 fake | `rag.query(...)` 后 `llm.calls` 非空；同时用 `patch("ragline.providers.llm.OpenAI", side_effect=AssertionError("network!"))` 守护——patch 未生效就会抛 AssertionError，间接验证无真实网络调用 |
| 11 | `mock_rag_providers(llm=FakeLLM(answer="X"))` 自定义注入 | `rag.query(...).answer == "X"` |

**已删除**（reviewer 发现：测试无法自然触发）：原"isolated_registries restore 阶段 register 抛错"用例 —— `Registry.register()` 永不抛 `RaglineRegistryError`（参 `src/ragline/registry/__init__.py:18-26`，已存在时只 warning 覆盖）。该用例需 mock 强制抛错才能触发，价值低，删除。

**实现要点**：
- `FakeLLM.complete` 签名**必须**与 `src/ragline/providers/llm.py:36` 一致：`def complete(self, prompt: str, *, temperature: float = 0.7) -> str` —— `temperature` 是 keyword-only
- `FakeLLM.complete` 实现里返回 `self.answer`（不在 `__init__` 里把 answer 绑死到 closure），允许测试在运行时改 `.answer` 属性
- `FakeLLM` / `FakeEmbedding` 各自暴露 `close_calls: int = 0` 计数器；`close()` 每次调用 `self.close_calls += 1` 然后立即返回
- `FakeEmbedding.embed` 签名与 `src/ragline/providers/embedding.py:38` 一致：`def embed(self, texts: list[str]) -> list[list[float]]`
- `isolated_registries` 用 `try/finally` 保证 restore；snapshot 7 个 registry：`transform_registry / retriever_registry / processor_registry / grader_registry / generator_registry / parser_registry / chunker_registry`
- `mock_rag_providers` 用 `unittest.mock.patch` 同时 patch `ragline.api.rag.LLMClient` 与 `ragline.api.rag.EmbeddingClient`，让构造器返回传入的 fake；yield `(fake_llm, fake_embedding)` 元组
- 类型注解严格（spec routing=Development Workflow，`ragline.testing` 是 public API，类型必须可外部 mypy 检查）
- 顶层暴露：`__all__ = ["FakeLLM", "FakeEmbedding", "isolated_registries", "mock_rag_providers"]`
- 不 import `ragline.engine`、`langgraph`、`langchain_core`
- 文件级 docstring 说明"为外部消费者提供"

**另需修改 `.importlinter`**（**subagent 在 Task 1.1 实施阶段执行此修改**——本 plan 文档描述任务规格，不是已完成的变更；`.importlinter` 文件在 writing-plans 阶段保持原样）：当前 `engine-only-langgraph` 契约的 `forbidden_modules` 是 `langgraph` / `langchain_core`，不包含 `ragline.engine`。即便把 `ragline.testing` 加到该契约的 `source_modules`，也只能禁止 `ragline.testing → langgraph/langchain_core`，**无法**拦截 `ragline.testing → ragline.engine`。

实施时在 `.importlinter` 中追加一个**独立的**新 forbidden 契约（与 `engine-only-langgraph` 并列）：

```ini
[importlinter:contract:testing-no-engine]
name = ragline.testing must not import ragline.engine
type = forbidden
source_modules =
    ragline.testing
forbidden_modules =
    ragline.engine
```

同时把 `ragline.testing` 加入 `engine-only-langgraph` 的 `source_modules`（顺手禁止 `ragline.testing → langgraph/langchain_core`，与其他公开模块一致）。两个契约共同保证 `ragline.testing` 的隔离边界。

**验证标准**：
- `uv run --extra dev --extra all pytest tests/unit/test_testing.py -v` 全绿（11 用例 passed）
- 整体 `pytest` 仍全绿
- 覆盖率不退化
- `uv run --extra dev --extra all python -c "from ragline.testing import FakeLLM, FakeEmbedding, isolated_registries, mock_rag_providers; print('ok')"` 输出 `ok`
- `uv run --extra dev lint-imports` 通过（验证 `.importlinter` 新增的 `ragline.testing` 源模块下的禁止规则生效）
- `grep -rE '^(from|import) (ragline\.engine|langgraph|langchain_core)' src/ragline/testing.py` 0 命中

**审查要求**：python-reviewer + code-quality-reviewer

---

### 任务 1.2：添加 `py.typed` 与 wheel `force-include`

**文件：**
- 新建：`src/ragline/py.typed`（空文件）
- 修改：`pyproject.toml`（追加 `[tool.hatch.build.targets.wheel.force-include]`）

**测试规格**（验证型，非 pytest）：

无 pytest 用例。验证通过命令：

1. `uv build --wheel` 成功（无 error/warning）
2. `python -m zipfile -l dist/ragline-*.whl | grep -E 'ragline/py\.typed'` 必须命中
3. 检查 wheel 不引入任何意外多余文件（diff 与 Task 1.1 前的 `python -m zipfile -l dist/*.whl` 输出，新增条目只应是 `ragline/py.typed` 与 `ragline/testing.py`）

**实现要点**：
- `src/ragline/py.typed` 是 PEP 561 协议要求的空文件
- `pyproject.toml` 加入：
  ```toml
  [tool.hatch.build.targets.wheel.force-include]
  "src/ragline/py.typed" = "ragline/py.typed"
  ```
- 不修改 `packages = ["src/ragline"]`（保持现状）

**验证标准**：上述 3 项验证命令全通过；整体 `pytest` 仍全绿。

**审查要求**：code-quality-reviewer

---

### 任务 1.3：建立 `examples/quickstart/` 可运行示例

**文件**（共 6 个新建，超出"1-3 files"原则——理由：本任务是"建立单个 quickstart 目录"的整体单元，3 份 `docs/*.md` 是同质的纯文本数据 fixture（视为 1 个数据 chunk），不可被拆解到其他任务；`consumer.yaml` / `consumer_minimal.py` / `examples/quickstart/README.md` 各承担配置 / 代码 / 文档的不同职责但必须一起到位，否则 demo 跑不通）：
- 新建：`examples/quickstart/consumer_minimal.py`
- 新建：`examples/quickstart/consumer.yaml`
- 新建：`examples/quickstart/docs/doc1.md` / `doc2.md` / `doc3.md`（≥50 字的简短示例 markdown）
- 新建：`examples/quickstart/README.md`

**测试规格**（手动验证型，subprocess 测试在 Task 1.4 实现）：

1. `uv run python examples/quickstart/consumer_minimal.py` 退出码 0
2. stdout 必须包含字符串 `INGESTED:`（来自 `print(f"INGESTED: {result.num_chunks} chunks")`）
3. stdout 必须包含 `Ragline is a config-driven RAG library.`（来自 mock 的固定 answer）

**实现要点**：

`consumer_minimal.py` 必须满足：
- 只 import `ragline` 顶层 + `ragline.testing`（不碰 `ragline.engine` / `ragline.api.*` 子模块）
- 用 `tempfile.TemporaryDirectory` 提供 chroma 持久化目录
- 用 `with isolated_registries(), mock_rag_providers() as (llm, _):` 包裹 RAG 实例化
- 在进入上下文后设 `llm.answer = "Ragline is a config-driven RAG library."`
- ingest `examples/quickstart/docs/` 整目录
- `cfg.providers.llm_api_key = "fake-key-for-demo"`（避开 `RAGLINE_ENV` / `OPENAI_API_KEY` 校验，因为 `mock_rag_providers` 已 patch 构造器，fake-key 不会触发真实调用）
- `print` 两行：`INGESTED: <n> chunks` 和 `ANSWER: <answer>`

`consumer.yaml`：合法、最小的 yaml 配置，演示 `from_yaml` 用法；标注"OPENAI_API_KEY 走环境变量"（不在脚本里用，仅文档示意）。

`docs/doc{1,2,3}.md`：内容主题相关，避免空文件（≥50 字英文，便于 chunker 切出多 chunk）。建议主题：ragline 的设计原则、插件机制、典型用法 —— 与 README 一致。

`examples/quickstart/README.md`：3 节即可：
1. 「如何运行」（`uv run python examples/quickstart/consumer_minimal.py`）
2. 「换成真实 provider」（去掉 `mock_rag_providers`，设 `OPENAI_API_KEY`，参考 `consumer.yaml`）
3. 「这个示例做了什么」

**验证标准**：3 项手动命令全通过；`.importlinter` 检查仍通过（如果 importlinter 包含 examples/ 目录则需排除，否则跳过）。

**审查要求**：code-quality-reviewer

---

### 任务 1.4：subprocess 回归测试

**文件：** 新建 `tests/integration/test_examples.py`

**依赖**：Task 1.3 必须先完成

**测试规格**（TDD RED 阶段先写）：

1. `test_consumer_minimal_runs_clean()`：subprocess 跑 `examples/quickstart/consumer_minimal.py`
   - `result.returncode == 0`
   - **stdout 断言**（覆盖完整对外契约，不仅 happy path）：
     - `"INGESTED:" in result.stdout`
     - `"ANSWER:" in result.stdout`（防止 print 前缀被改动后失去断言）
     - `"Ragline is a config-driven RAG library." in result.stdout`
     - 解析 `INGESTED: <n> chunks` 行后断言 `n > 0`（ingest 真正切出 chunk，非 0 chunks 静默通过）
   - **stderr 断言**：`"Traceback" not in result.stderr`（守护非零 exit 之外的静默错误）
   - 失败时 `pytest.fail(result.stdout + "\n---STDERR---\n" + result.stderr)`，便于排查
2. 子进程超时保护：`subprocess.run(..., timeout=60.0)` —— **代码骨架必须显式写出 timeout 参数**，遗漏会导致 CI 卡死

**实现要点**：
- 复用 `tests/integration/test_optional_deps.py:123-131` 的 subprocess 模式
- 路径用 `pathlib.Path` 相对计算（不硬编码绝对路径）
- 不设 `OPENAI_API_KEY` env（验证 examples 真的不依赖 key）
- 子进程继承当前 venv 的 PYTHONPATH（subprocess 默认即可，不需要 env=os.environ copy）
- 解析 `INGESTED:` 行用 `re.search(r"INGESTED: (\d+) chunks", result.stdout)`，断言 `int(match.group(1)) > 0`

**验证标准**：
- `uv run --extra dev --extra all pytest tests/integration/test_examples.py -v` 全绿
- 整体 `pytest` 全绿
- 覆盖率 ≥ 90%

**审查要求**：code-quality-reviewer

---

### 任务 1.5：顶层 README 加两节

**文件：** 修改 `README.md`

**测试规格**：无 pytest 用例（纯文档）。

**实现要点**：
在 `## Quick Start` 之后、`## Why Ragline?` 之前插入两节：

1. **「在外部项目中使用 ragline」**：
   - 当前未发布 PyPI，说明三种引用方式（uv 路径 editable / uv git / uv workspace）
   - 给出 `uv add --editable "/path/to/ragline[all]"` 与 `uv add "ragline @ git+https://github.com/<you>/ragline.git@main"` 两条命令
   - 链到 `examples/quickstart/`

2. **「在测试中使用 ragline」**：
   - 介绍 `from ragline.testing import FakeLLM, FakeEmbedding, isolated_registries, mock_rag_providers`
   - 给一段 8-12 行的最小 pytest 示例（用 `with isolated_registries(), mock_rag_providers() as (llm, _): ...`）
   - 强调"同进程 RAG 单例"限制（链到既有的 Status & Compatibility 节）

**验证标准**：
- `test -f examples/quickstart/consumer_minimal.py && test -f examples/quickstart/README.md` 退出 0（确保被链接的目标存在）
- README 新增两节中所有相对路径在仓库内可解析（如 `./examples/quickstart/` 必须是真实目录）
- 文件无 markdown 语法错误（人工预览 + `grep -nE '^\s*```' README.md | wc -l` 应为偶数，验证 fenced block 配对）

**审查要求**：code-quality-reviewer（轻量）

---

### 阶段 2（Task 2）：内部去重（dogfooding）

**依赖**：Task 1 全部完成且 merge 到主线后再启动。

5 个子任务，串行。每个子任务跑完后整体 `pytest` 必须不退化。

---

### 任务 2.1：建立顶层 `tests/conftest.py`

**文件：** 新建 `tests/conftest.py`

**测试规格**（验证型）：

无新增 pytest 用例。验证：
- 创建 `tests/conftest.py` 后，整体 `pytest` 全绿（既有 253+ passed 数量不变）
- 顶层 fixture 在 `pytest --fixtures tests/integration/test_rag.py` 输出中能看到

**实现要点**：
- 导出**两个** fixture，全部基于 `ragline.testing`：
  ```python
  @pytest.fixture
  def isolated_registries_fx():
      from ragline.testing import isolated_registries
      with isolated_registries():
          yield

  @pytest.fixture
  def mock_rag_providers_fx():
      from ragline.testing import mock_rag_providers
      with mock_rag_providers() as fakes:
          yield fakes
  ```
- 不 autouse（避免污染所有测试）
- 不 import 任何具体测试逻辑

**验证标准**：整体 `pytest` 全绿；`pytest --collect-only` 不报新增 collection error。

**审查要求**：code-quality-reviewer

---

### 任务 2.2：重构 `test_rag.py` + `test_engine_graph.py`

**文件：**
- 修改：`tests/integration/test_rag.py`
- 修改：`tests/integration/test_engine_graph.py`

**测试规格**（验证型）：

无新增用例。两个文件重构后：
- `tests/integration/test_rag.py` pytest 通过用例数与改造前一致
- `tests/integration/test_engine_graph.py` 同上
- 整体覆盖率不退化

**实现要点**：

`test_rag.py`（行号参考）：
- 删除 `clean_all_registries` fixture（line 15-46）
- 删除 `mock_providers` fixture（line 49-61）
- 用 `isolated_registries_fx` + `mock_rag_providers_fx` 替换：每个 test 函数签名把 `mock_providers, clean_all_registries` 改成 `isolated_registries_fx, mock_rag_providers_fx`
- **断言改写（迁移规则）**：
  - `mock_providers["llm"]` → `mock_rag_providers_fx[0]`
  - `mock_providers["embedding"]` → `mock_rag_providers_fx[1]`
- **特殊处理两个会因迁移而失败的断言（reviewer Critical 发现）**：
  1. `test_close_tolerates_provider_failure`（test_rag.py:317）：原代码 `mock_providers["llm"].close.assert_called_once()` 假设 `.close` 是 `MagicMock`。`FakeLLM.close` 是真实方法，**没有** `.assert_called_once()`。改写为：`assert mock_rag_providers_fx[0].close_calls == 1`（依赖 Task 1.1 中 `FakeLLM` 暴露的 `close_calls: int` 计数器）。
  2. `test_ingest_and_query_e2e`（test_rag.py:185）：原代码 `assert query_result.answer == "mocked answer"` 假设 mock 返回此字符串；`FakeLLM` 默认 `"fake answer"`。改写为：在测试函数体内、`RAG()` 实例化**之前**插入一行 `mock_rag_providers_fx[0].answer = "mocked answer"`（依赖 Task 1.1 中"`complete()` 读取 `self.answer`、运行时可写"的特性）。
- `fake_api_key` autouse fixture（line 64-66）保留不动 —— 与本计划无关
- 不调整 `_make_config` 函数（line 69-73）—— 与本计划无关

`test_engine_graph.py`（行号参考）：
- 删除 `clean_registries` fixture（line 24-44）
- 用 `isolated_registries_fx` 替换：把所有 `def test_xxx(clean_registries)` 改成 `def test_xxx(isolated_registries_fx)`
- `_register_minimal(clean_registries)` 改为 `_register_minimal()`（fixture 只起到隔离作用，不传入函数内部）
- `_register_minimal` 函数定义（line 47）当前签名 `def _register_minimal(clean_registries: Any) -> None:` 改为无参 `def _register_minimal() -> None:`

**验证标准**：
- `uv run --extra dev --extra all pytest tests/integration/test_rag.py tests/integration/test_engine_graph.py -v` 全绿
- 整体 `pytest` 全绿，通过数不变
- 覆盖率 ≥ 90%
- `git diff --stat tests/integration/test_rag.py tests/integration/test_engine_graph.py` 显示净减 ≥ 40 行

**审查要求**：python-reviewer + code-quality-reviewer

---

### 任务 2.3：重构 `test_handlers/{chunkers,generators,graders}.py`

**文件：**
- 修改：`tests/unit/test_handlers/test_chunkers.py`
- 修改：`tests/unit/test_handlers/test_generators.py`
- 修改：`tests/unit/test_handlers/test_graders.py`

**测试规格**（验证型）：

3 个文件重构后通过用例数与改造前一致。

**实现要点**：

每个文件操作相同：
- 删除本地 `clean_<name>_registry` fixture 定义
- 用 `isolated_registries_fx` 替换所有引用
- 注意：`isolated_registries_fx` 是空 yield 不返回值；如果测试代码原本 `(clean_chunker_registry)` 拿来用作 `chunker_registry` 的别名（如 `test_chunkers.py:237` 的 `clean_chunker_registry.has("recursive")`），需要改为直接 `from ragline.registry import chunker_registry; chunker_registry.has("recursive")`
- 同样处理 `test_generators.py` 内 `clean_generator_registry.has("basic")` / `clean_generator_registry.has("citation")` 等访问点
- 同样处理 `test_graders.py` 内 `clean_grader_registry.has("score")` 等访问点

**验证标准**：
- `uv run --extra dev --extra all pytest tests/unit/test_handlers/test_{chunkers,generators,graders}.py -v` 全绿
- 整体 `pytest` 全绿
- 覆盖率不退化
- 3 文件合计净减 ≥ 35 行

**审查要求**：python-reviewer + code-quality-reviewer

---

### 任务 2.4：重构 `test_handlers/{parsers,processors,retrievers,transforms}.py`

**文件：**
- 修改：`tests/unit/test_handlers/test_parsers.py`
- 修改：`tests/unit/test_handlers/test_processors.py`
- 修改：`tests/unit/test_handlers/test_retrievers.py`
- 修改：`tests/unit/test_handlers/test_transforms.py`

注：本任务 4 文件略超"1-3 files"边界，但 4 个文件改动同质（统一删除本地 `clean_*_registry` fixture），且接口压力一致；按"群组重构"处理可减少 1 个 subagent dispatch overhead。

**测试规格**（验证型）：

4 个文件重构后通过用例数与改造前一致。

**实现要点**：

每个文件操作相同：
- 删除本地 `clean_<name>_registry` fixture 定义（`clean_parser_registry` / `clean_processor_registry` / `clean_retriever_registry` / `clean_transform_registry`）
- 用 `isolated_registries_fx` 替换所有引用
- 如果测试代码原本拿 `clean_xxx_registry` 当 registry 别名调用方法（如 `clean_processor_registry.has("rrf")`），需要直接 `from ragline.registry import processor_registry; processor_registry.has("rrf")`
- 注意：`test_parsers.py` 还有一个独立的 `sample_pdf` session fixture（line 23-32），与本计划无关，保留不动
- 注意：`test_retrievers.py` 还有 `mock_vector_store` / `mock_embedding` / `mock_bm25` 三个 MagicMock fixture（line 12-27），与本计划无关，保留不动

**验证标准**：
- `uv run --extra dev --extra all pytest tests/unit/test_handlers/test_{parsers,processors,retrievers,transforms}.py -v` 全绿
- 整体 `pytest` 全绿
- 覆盖率不退化
- 4 文件合计净减 ≥ 45 行

**审查要求**：python-reviewer + code-quality-reviewer

---

### 任务 2.5：重构 `test_engine/{test_nodes_post_generate,test_nodes_qt_retrieve}.py`

**文件：**
- 修改：`tests/unit/test_engine/test_nodes_post_generate.py`
- 修改：`tests/unit/test_engine/test_nodes_qt_retrieve.py`

**测试规格**（验证型）：

2 个文件重构后通过用例数与改造前一致。

**实现要点**：

- `test_nodes_post_generate.py` 有 `clean_processor_registry`（line 39-48）+ `clean_generator_registry`（line 51-60）。两者都删除，用 `isolated_registries_fx` 单一 fixture 替换。注意原代码同时引用了两个 fixture，调整后只需一个。
- `test_nodes_qt_retrieve.py` 有 `clean_transform_registry`（line 20-29）+ `clean_retriever_registry`（line 32-41）。同上处理。

**验证标准**：
- `uv run --extra dev --extra all pytest tests/unit/test_engine/test_nodes_post_generate.py tests/unit/test_engine/test_nodes_qt_retrieve.py -v` 全绿
- 整体 `pytest` 全绿
- 覆盖率 ≥ 90%
- 全计划完成后 `git diff --stat tests/` 净减 ≥ 100 行（spec 验收门槛）
- `grep -rE 'def clean_(all_)?[a-z_]*_?registr' tests/` 0 命中（除 `tests/conftest.py` 内的新定义不计）

**审查要求**：python-reviewer + code-quality-reviewer

---

## 测试策略

| 层级 | 文件 | 范围 |
|---|---|---|
| 单元 | `tests/unit/test_testing.py`（新） | `ragline.testing` 4 个 API 的语义与边界 |
| 集成 | `tests/integration/test_examples.py`（新） | subprocess 跑 examples 验证对外契约 |
| 集成 | 既有 e2e + integration 全部 | 不动语义，仅 Task 2 阶段重构 fixture 复用 |
| 验证型 | wheel 内容 | `python -m zipfile -l dist/*.whl \| grep py.typed` |

**回归门**：
- 既有 4 个 e2e 测试一行不动，跑通即说明 public API 未破
- `.importlinter` 检查 `ragline.testing` 不引入 `ragline.engine` 依赖

## E2E 稳定性要求

本计划无 E2E 测试新增。既有 4 个 e2e 不在范围。

## 风险与缓解

- **风险 1**：`FakeLLM.complete` 签名与 `LLMClient.complete` 不一致（keyword-only `temperature`）
  - 缓解：Task 1.1 实现要点显式锁定签名；TDD 测试用例 #2 验证 keyword-only 透传
- **风险 2**：`py.typed` 未进 wheel
  - 缓解：Task 1.2 验收命令包含 `zipfile -l` grep 验证
- **风险 3**：`mock_rag_providers` 的 patch 路径在 ragline 内部重构后失效
  - 缓解：Task 1.1 单测 #9 验证完整流向（实例化 → query → fake 被调用）
- **风险 4**：Task 2 重构期间覆盖率掉到 90% 以下
  - 缓解：每个子任务完成后必须跑 `pytest --cov` 验证
- **风险 5**：`examples/quickstart/consumer_minimal.py` 在 CI / 不同环境（无外网）失败
  - 缓解：`mock_rag_providers()` 全程 patch OpenAI 构造器；`fake-key-for-demo` 不触发真实调用；显式 `tempfile.TemporaryDirectory` 不依赖任何文件系统状态
- **风险 6**：`isolated_registries_fx` 与既有 `fake_api_key` autouse fixture 在 `test_rag.py` 中冲突
  - 缓解：Task 2.2 显式保留 `fake_api_key` autouse；`isolated_registries_fx` 是函数级非 autouse，不会重叠
- **风险 7**：重构 `_register_minimal(clean_registries)` 改无参后，函数内引用未更新导致 NameError
  - 缓解：Task 2.2 实现要点已显式说明 `_register_minimal` 改无参版本；TDD 跑测试会即时暴露

## 验收标准

**Task 1（对外契约）完成后**：
- [ ] `uv run --extra dev --extra all pytest` 全绿（≥ 253 + 11 新用例 passed）
- [ ] 整体覆盖率 ≥ 90%（pyproject `--cov-fail-under=90` 通过）
- [ ] `uv build --wheel` 成功；`python -m zipfile -l dist/ragline-*.whl | grep ragline/py.typed` 命中
- [ ] `uv run python examples/quickstart/consumer_minimal.py` 退出码 0 且 stdout 含两个关键字
- [ ] `.importlinter` 通过（包括 `ragline.testing` 不依赖 `ragline.engine`）
- [ ] `from ragline.testing import FakeLLM, FakeEmbedding, isolated_registries, mock_rag_providers` 可用
- [ ] README 含两节新增内容

**Task 2（内部去重）完成后**：
- [ ] 整体 `pytest` 通过数等于 Task 1 完成后的基数（重构不增不减）
- [ ] 覆盖率 ≥ 90%
- [ ] `git diff --stat tests/` 显示净减 ≥ 100 行（相对 Task 1 完成时的快照）
- [ ] `grep -rE 'def clean_(all_)?[a-z_]*_?registr' tests/` 0 命中（仅 `tests/conftest.py` 中的新 fixture 定义不计，且其命名为 `isolated_registries_fx` / `mock_rag_providers_fx`，不匹配该 grep）

---

## 已审定决策

本段记录 dual-review 多轮中确认的有意保留 / 驳回 / 边界澄清。每条记录"决策 + 审定结论 + 有意理由"，不作免责描述。

- **决策点**：`.importlinter` 文件实体不在 writing-plans 阶段修改，由 subagent 在 Task 1.1 实施阶段执行。
  - 审定结论：round 3 reviewer 1 误把"plan 描述的实施动作"理解为"writing-plans 阶段应已完成的变更"，据此标 N1 为 BLOCKING。该判定属概念误解，已驳回。
  - 有意理由：plan 是 implementation specification，描述的是 subagent 在 TDD 循环中要执行的变更；writing-plans 阶段直接修改 `.importlinter` 会绕过 Task 1.1 的 RED→GREEN→IMPROVE→quality-gate→review 四道门控，违反 L1 铁律。Plan Task 1.1 已显式加注"subagent 在实施阶段执行此修改"消除歧义。

- **决策点**：Plan Task 2.2 标注的 `test_rag.py:185` / `:317` 行号保持不变。
  - 审定结论：round 2 reviewer 1 提出行号偏差 13/10 行，经 `grep -n 'assert query_result.answer == "mocked answer"' tests/integration/test_rag.py` 与 `grep -n 'mock_providers\["llm"\]\.close\.assert_called_once' tests/integration/test_rag.py` 验证，输出分别为 `185` 与 `317`，与 plan 一致。属 reviewer 误读，已驳回。
  - 有意理由：行号经过实际工具验证，无需修改。
