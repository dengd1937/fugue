# 实现计划：可选依赖懒加载（lazy-optional-imports）

## 执行方式

本计划通过 `/subagent-driven-development` skill 执行。以下任务描述是 skill 的输入规格，不是直接执行指令。

## 概述

修复 `import fugue` 在未安装 `[pdf]`/`[bge]` 可选依赖时崩溃的问题。新增共享
`_optional.require()` helper，将 `pdf.py`、`bge.py` 的可选依赖 import 改为
函数/方法内懒加载，缺失时抛含 `pip install 'fugue[xxx]'` 的清晰 ImportError。

## 需求

- 未装 `[pdf]`/`[bge]` 时 `import fugue`、`register_parsers()`、`register_processors()` 成功
- 真正调用 PDF 解析 / 构造 BGE reranker 时才抛清晰可操作 `ImportError`
- 装了 `[pdf]`/`[bge]` 时行为与改动前完全一致（零回归）
- 仅覆盖 pypdf(`[pdf]`) + FlagEmbedding(`[bge]`)；不碰 chromadb/server/PyPI 改名

## 架构变更

- 新增 `src/fugue/_optional.py`：`require(module_name, *, extra)` helper
- 修改 `src/fugue/handlers/parsers/pdf.py`：删顶层 `import pypdf`，改函数内 `require`
- 修改 `src/fugue/providers/reranker/bge.py`：删顶层 `from FlagEmbedding import FlagReranker`，改 `__init__` 内 `require`
- 新增 `tests/integration/test_optional_deps.py`：缺失依赖回归测试
- 不动 `parsers/__init__.py`、`reranker/__init__.py`、`rerank.py`、`_LazyReranker`（追踪已确认无需改）

## 环境前置（Environment Prerequisites）

- **pytest（测试运行器）**：项目唯一测试依赖，已随 `[dev]` extra 装在 `.venv`
  - 验证：`.venv/bin/pytest --version`
  - 修复：`uv sync --extra dev`
- **可选依赖 pypdf / FlagEmbedding**：本地 `.venv` 已装 `[all]`，故 import 拦截
  用于模拟"缺失"；无需卸载真实包
  - 验证：`.venv/bin/python -c "import pypdf, FlagEmbedding"`
  - 修复：`uv sync --extra all`

## 实现步骤

### 阶段 1：共享 helper

新增懒加载入口，后续两个改造任务依赖它。

### 阶段 2：两处可选依赖改造

`pdf.py`、`bge.py` 分别接入 helper。互不依赖，均依赖阶段 1。

### 阶段 3：跨模块回归

模拟缺失依赖验证 `import fugue` 全链路不崩。

## 任务

### 任务 1：新增 `_optional.require()` 共享 helper

**文件：**
- 新建 `src/fugue/_optional.py`
- 新建 `tests/unit/test_optional.py`

**实现规格：**
- 签名 `require(module_name: str, *, extra: str) -> Any`
- 返回类型**刻意为 `typing.Any`**（不是 `ModuleType`）：可选依赖
  pypdf/FlagEmbedding 无类型 stub，调用方需在返回值上做动态属性访问
  （`pypdf.PdfReader`、`mod.FlagReranker`）；`ModuleType` 会导致
  `attr-defined` mypy 报错。`Any` 是动态可选导入 shim 的惯例设计，
  docstring 须注明此理由
- 行为：`importlib.import_module(module_name)` 成功则返回模块对象
- 失败（捕获 `ModuleNotFoundError` / `ImportError`）：抛
  `ImportError`，消息格式固定为
  `可选依赖 '<module_name>' 未安装。请运行: pip install 'fugue[<extra>]'`，
  并 `raise ... from e` 保留原始异常链
- 模块仅依赖标准库 `importlib` + `typing.Any`，无 fugue 内部 import（避免循环）

**测试规格：**
- `require("pathlib", extra="x")` 返回 `pathlib` 模块对象（断言 `.Path` 存在）
- `require("definitely_absent_pkg_xyz", extra="pdf")` 抛 `ImportError`，
  消息含 `pip install 'fugue[pdf]'` 且 `__cause__` 非 None
- 参数化覆盖 extra 名出现在消息中（如 `extra="bge"` → 消息含 `fugue[bge]`）

**验证标准：** 上述 3 组断言全部通过；`mypy src` 零错误（含 `Any` 返回类型，
调用方无 `attr-defined`）

**审查要求：** code-quality-reviewer-prompt + python-reviewer-prompt

---

### 任务 2：`pdf.py` 改为懒加载 pypdf

**文件：**
- 修改 `src/fugue/handlers/parsers/pdf.py`
- 修改 `tests/unit/test_handlers/test_parsers.py`（仅追加新用例；既有用例与
  顶层 `import pypdf` fixture 不动——见下方测试策略说明）

**实现规格：**
- 在 `pdf.py` 顶部 `from fugue._optional import require`，删除模块顶层 `import pypdf`
- `pdf_parser(path)` 函数体首行：`pypdf = require("pypdf", extra="pdf")`
- 函数其余逻辑不变（`pypdf.PdfReader(...)` 等照常；`pypdf` 现为 `Any`，
  动态属性访问不触发 mypy）
- 模块顶层不再引用 pypdf 任何符号

**测试规格：**
- 既有 `test_parsers.py` 正向用例 + 顶层 `import pypdf`（fixture 用
  `pypdf.PdfWriter()` 生成测试 PDF）保持原样、保持通过——本任务**不得**
  用全局 `sys.modules` 注入模拟缺失（会破坏该文件自身顶层 import）
- 新增用例**通过 patch 接缝模拟缺失**：
  `patch("fugue.handlers.parsers.pdf.require", side_effect=ImportError("可选依赖 'pypdf' 未安装。请运行: pip install 'fugue[pdf]'"))`，
  断言调用 `pdf_parser(any_path)` 抛 `ImportError` 且消息含
  `pip install 'fugue[pdf]'`
- 新增用例：在**不 patch、pypdf 已装**前提下，断言
  `import fugue.handlers.parsers.pdf` 后该模块的 `sys.modules` 命名空间
  顶层无 `pypdf` 属性（证明顶层 eager import 已移除）

**验证标准：** 既有 parser 测试零回归 + 新增 2 用例通过

**审查要求：** code-quality-reviewer-prompt + python-reviewer-prompt

---

### 任务 3：`bge.py` 改为懒加载 FlagEmbedding

**文件：**
- 修改 `src/fugue/providers/reranker/bge.py`
- 修改 `tests/unit/test_providers/test_reranker.py`（**必须改写既有用例的
  mock 接缝**——见测试规格；这是本任务的核心工作量，非"仅追加"）

**实现规格：**
- 在 `bge.py` 顶部 `from fugue._optional import require`，删除模块顶层
  `from FlagEmbedding import FlagReranker`
- 在 `BGEReranker.__init__` 内，构造 `self._reranker` 之前：
  ```python
  _flagembedding = require("FlagEmbedding", extra="bge")
  FlagReranker = getattr(_flagembedding, "FlagReranker")
  ```
  用 `getattr`（而非 `.FlagReranker` 属性直取）：返回 `Any` 规避
  mypy `attr-defined`，且对 FlagEmbedding 未来 `__init__` 重构更鲁棒
- `_resolve_device` 内既有的函数内 `import torch`（bge.py:13）保持不变
- `__init__` 其余逻辑（`use_fp16`、`devices`、`self._timeout`）不变

**测试规格（关键——既有 mock 接缝失效，必须迁移）：**
- 背景：`test_reranker.py` 现有 **10 处** `patch("fugue.providers.reranker.bge.FlagReranker")`
  （分布在 9 个测试函数 + 1 个 numpy 标量用例）。改造后
  `fugue.providers.reranker.bge` 模块顶层不再有 `FlagReranker` 名字，
  这些 patch 会抛 `AttributeError: module ... has no attribute 'FlagReranker'`，
  导致既有 reranker 测试**全部失败**
- 迁移规则（机械、逐处套用，不得遗漏任一处）：将每处
  `patch("fugue.providers.reranker.bge.FlagReranker") as mock_cls`
  改写为
  `patch("fugue.providers.reranker.bge.require") as mock_require`，
  并在进入 `with` 后设
  `mock_cls = mock_require.return_value.FlagReranker`
  （即 require 返回的 fake 模块对象，其 `.FlagReranker` 即原 `mock_cls`）；
  后续对 `mock_cls` 的断言/配置全部不变
  - `patch("fugue.providers.reranker.bge._resolve_device", ...)` 与
    `patch("torch.cuda.is_available", ...)` 等其他 patch 点**保持不变**
- 既有 9 个用例迁移后断言语义与原先完全等价（FlagEmbedding 已装时行为不变）
- 新增用例 A：`patch("fugue.providers.reranker.bge.require",
  side_effect=ImportError("可选依赖 'FlagEmbedding' 未安装。请运行: pip install 'fugue[bge]'"))`，
  断言构造 `BGEReranker(model_name="x", device="cpu")` 抛 `ImportError`
  且消息含 `pip install 'fugue[bge]'`
- 新增用例 B：在 FlagEmbedding 已装、不 patch 前提下，断言
  `from fugue.providers.reranker.bge import BGEReranker` 与
  `from fugue.providers.reranker import BGEReranker` 均不抛异常，
  且 `fugue.providers.reranker.bge` 模块顶层 namespace 无 `FlagReranker` 属性

**验证标准：** 既有 9 用例迁移后全绿（零行为回归）+ 新增 2 用例通过；
`test_reranker.py` 中不再出现 `patch("...bge.FlagReranker")` 字样

**审查要求：** code-quality-reviewer-prompt + python-reviewer-prompt

---

### 任务 4：跨模块缺失依赖回归测试

**文件：**
- 新建 `tests/integration/test_optional_deps.py`

**实现规格（仅测试，无生产代码改动）——子进程隔离，禁止进程内 reload：**
- **不得**在 pytest 进程内删除/`importlib.reload` `fugue.*` 模块：
  既有 `test_parsers.py`/`test_reranker.py` 在收集期已 module-level
  import `fugue.*`，进程内卸载会留下陈旧函数引用、污染全量运行
- 改为**子进程隔离**：用
  `subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True)`
  在干净子进程内验证。`SCRIPT` 首行先注入缺失哨兵：
  ```python
  import sys
  sys.modules["pypdf"] = None
  sys.modules["FlagEmbedding"] = None
  ```
  随后执行各断言，失败时 `raise SystemExit(<非零>)` 或打印标记
- 测试函数断言子进程 `returncode == 0`（失败时附 `result.stderr` 便于定位）

**测试规格（子进程脚本内逐条验证，全部通过则子进程 exit 0）：**
- 注入缺失后 `import fugue` 成功（无 `ModuleNotFoundError`）
- `from fugue.handlers.parsers import register_parsers; register_parsers()` 成功
- `from fugue.handlers.processors import register_processors` 成功导入
  （`register_processors(reranker)` 需 reranker 入参，此处只断言**导入成功**
  且 `parser_registry.has("pdf")` 为真，不实际构造 reranker、不触发 BGE 加载）
- 调用注册的 `pdf` parser（`parser_registry.get("pdf")(some_tmp_path)`）
  抛含 `pip install 'fugue[pdf]'` 的 `ImportError`
- 构造 `BGEReranker` 路径：在子进程内 `from fugue.providers.reranker.bge
  import BGEReranker; BGEReranker(model_name="x", device="cpu")` 抛含
  `pip install 'fugue[bge]'` 的 `ImportError`
- 父进程测试函数本身不触碰全局 `sys.modules`，故对同进程其他测试零副作用

**验证标准：** 子进程 `returncode == 0`（5 条子断言全过）；该测试文件
**独立运行与全量 `pytest`（非 e2e）运行结果一致**（无跨测试污染）——
通过子进程隔离天然保证

**审查要求：** code-quality-reviewer-prompt + python-reviewer-prompt

## 测试策略

- 单元测试：`tests/unit/test_optional.py`（helper）、
  `tests/unit/test_handlers/test_parsers.py`（pdf 懒加载）、
  `tests/unit/test_providers/test_reranker.py`（bge 懒加载）
- 集成测试：`tests/integration/test_optional_deps.py`（`import fugue` 全链路缺失回归）
- 端到端测试：不涉及（无 UI / 无外部服务新增）

## 风险与缓解

- **风险（CRITICAL，已在 Task 3 消解）**：`test_reranker.py` 现有 10 处
  `patch("fugue.providers.reranker.bge.FlagReranker")`，改函数内 import 后
  patch 目标消失 → 既有 reranker 测试全部 `AttributeError`
  - 缓解：Task 3 已删除"不改既有"约束，给出**逐处机械迁移规则**
    （patch 点改为 `...bge.require`、`mock_cls = mock_require.return_value.FlagReranker`），
    并把"不再出现 `patch("...bge.FlagReranker")`"列为验证标准
- **风险（CRITICAL，已在 Task 4 消解）**：进程内删除/reload `fugue.*`
  会污染已收集的 `test_parsers.py`/`test_reranker.py` module-level import
  - 缓解：Task 4 改为**子进程隔离**（`subprocess.run([sys.executable,"-c",...])`），
    父进程零 `sys.modules` 操作，跨测试污染天然不可能
- **风险（已在 Task 1/2/3 消解）**：`require()` 返回 `ModuleType` 时
  `.PdfReader`/`.FlagReranker` 触发 mypy `attr-defined`
  - 缓解：`require() -> Any`（设计性，docstring 注明）；bge 侧用 `getattr`
- **风险**：`test_parsers.py:5` 顶层 `import pypdf`（fixture 生成 PDF），
  全局 `sys.modules` 注入会破坏该文件自身 import
  - 缓解：Task 2 单元测试改用 `patch("...parsers.pdf.require")` 接缝，
    不碰全局 `sys.modules`；缺失场景的全链路验证交由 Task 4 子进程
- **风险**：覆盖率门控 90%（`cov-fail-under=90`），新增 `_optional.py` 必须被覆盖
  - 缓解：任务 1 测试规格覆盖 require 成功/失败两分支；既有 reranker
    9 用例迁移后仍贡献原有覆盖（不致掉档）

## 验收标准

- [ ] 隔离环境裸装 wheel（无 `[pdf]`/`[bge]`）后 `import fugue` 成功
- [ ] 该环境 `register_parsers()` / `register_processors()` 导入成功
- [ ] 该环境调用 PDF 解析 / 构造 BGE 抛含 `pip install 'fugue[...]'` 的 ImportError
- [ ] 装 `[pdf]`/`[bge]` 后既有测试套件全绿（零回归）
- [ ] 新增单元 + 集成回归测试全部通过
- [ ] mypy src 零错误；覆盖率 ≥ 90% 门控通过
