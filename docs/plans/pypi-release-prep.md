# 实现计划：pypi-release-prep

## 执行方式

本计划通过 `/subagent-driven-development` skill 执行。以下任务描述是 skill 的输入规格，不是直接执行指令。

## 概述

让 ragline 具备 PyPI 首次发布（0.1.0）所需的元数据、许可证、依赖纯净度条件。通过 4 个独立任务（T1-T4）依次完成 pyproject 元数据补全、MIT LICENSE 文件、transformers 依赖迁移、CHANGELOG.md 起稿，每任务独立走 TDD → 实现 → 审查闭环。

## 需求

- `pip install ragline` 最小安装不再拉 `transformers` / `torch` / `safetensors`（节省约 200-500MB）
- `pip install ragline[bge]` 仍然拉到 transformers（与 FlagEmbedding 1.4 兼容的 `>=4.44.2,<5` pin）
- PyPI 项目页面展示 author / license / Homepage / Repository / Issues / Changelog 链接，trove classifiers 完整
- LICENSE 文件存在，下游可在 MIT 范围内合法使用
- CHANGELOG.md 记录 0.1.0 完整功能清单（7 个子模块）
- 现有 260 tests 全部通过，coverage ≥ 96.84%（不降）
- `uvx twine check dist/*` PASSED
- 不修改任何 Python 源码 / API 契约

## 架构变更

- `pyproject.toml`（修改）：
  - `[project]` 表新增 `authors` / `license` / `license-files` / `urls` / `keywords` / `classifiers` 字段
  - `[project].dependencies` 移除 `transformers>=4.44.2,<5`
  - `[project.optional-dependencies].bge` 新增 `transformers>=4.44.2,<5`
- `LICENSE`（新建）：MIT 全文，版权行 `Copyright (c) 2026 dengdi`
- `CHANGELOG.md`（新建）：Keep-a-Changelog 1.1.0 格式，记录 0.1.0 七个子模块
- `uv.lock`（重新生成）：由 `uv lock` 自动产出，反映 transformers 从 core 移除后的依赖图

## 环境前置（Environment Prerequisites）

- **uv（依赖管理 + 测试运行）**：
  - 验证：`uv --version`（要求 ≥ 0.4，本地实测 0.7.6）
  - 修复：重新安装 uv（`brew reinstall uv` 或 `curl -LsSf https://astral.sh/uv/install.sh | sh`）
- **hatch（wheel 构建）**：
  - 验证：`uvx hatch --version` 能跑通
  - 修复：`uv tool install hatch`
- **twine（元数据合规检查）**：
  - 验证：`uvx twine --version` 能跑通
  - 修复：无需预装，`uvx twine check` 直接拉用

## 实现步骤

每任务由 `subagent-driven-development` skill 调度 implementer subagent 完成。任务顺序固定为 T1 → T2 → T3 → T4，强制串行：

- T1 必先（建立 `tests/unit/test_packaging.py` 文件 + 更新 pyproject 元数据 + 上调 cov-fail-under）
- T2 依赖 T1（在已有 test_packaging.py 中追加 LICENSE 场景）
- T3 依赖 T1（修改 pyproject.toml dependencies，与 T1 串行）
- T4 依赖 T3（在已有 test_packaging.py 中追加 CHANGELOG 场景；同时 T3 完成后 pyproject 与 lockfile 才是发布前最终态）

### 任务 T1：补 pyproject `[project]` 元数据 + 上调 cov-fail-under

**文件：** 修改 `pyproject.toml`（新增 `[project]` 字段；不改既有 dependencies / extras / scripts；`[tool.pytest.ini_options]` 中 `--cov-fail-under` 上调到 96）；修改 `.github/workflows/ci.yml`（line 48 的显式 CLI `--cov-fail-under=90` 同步上调到 96）；新建 `tests/unit/test_packaging.py`

**关于 ci.yml 的说明**：pytest 处理 `addopts` 在前，CLI 参数在后，CLI 覆盖 addopts。如果仅改 `addopts` 不动 ci.yml，CI 跑 pytest 时仍按 `--cov-fail-under=90` 门控，本任务声明的"≥ 96 由 pytest 硬性保证"在 CI 上失效。必须双侧同步。

**测试规格：** 需覆盖以下场景，由 implementer subagent 编写 `tests/unit/test_packaging.py` 中的测试用例（如该文件不存在则新建；如已存在则追加）。

- 场景 1：`tomllib.load()` 读取 pyproject.toml，`project.authors` 含一个元素，name == "dengdi"，email == "dengdi1803@gmail.com"
- 场景 2：`project.license` == "MIT"
- 场景 3：`project.license-files` == ["LICENSE"]
- 场景 4：`project.keywords` 列表恰好等于 `["rag", "llm", "langgraph", "retrieval", "embedding", "chromadb", "bm25"]`
- 场景 5：`project.classifiers` 列表包含以下 10 条全部（顺序不限）：
  - `Development Status :: 4 - Beta`
  - `Intended Audience :: Developers`
  - `License :: OSI Approved :: MIT License`
  - `Operating System :: OS Independent`
  - `Programming Language :: Python :: 3`
  - `Programming Language :: Python :: 3.12`
  - `Programming Language :: Python :: 3.13`
  - `Topic :: Scientific/Engineering :: Artificial Intelligence`
  - `Topic :: Software Development :: Libraries :: Python Modules`
  - `Typing :: Typed`
- 场景 6：`project.urls` 表含 `Homepage` / `Repository` / `Issues` / `Changelog` 四个键，值分别等于 spec D2 中给定 URL
- 场景 7：未破坏既有非依赖字段（`name == "ragline"`，`version == "0.1.0"`，`requires-python == ">=3.12"`）。**不**检查 `dependencies` 列表内容（避免与 T3 场景 1 冲突；dependencies 完整性由 T3 自身的场景 5 保证）
- 场景 8：`[tool.pytest.ini_options].addopts` 字符串包含 `--cov-fail-under=96`（替换原有的 `--cov-fail-under=90`，使自动化门控与 plan 要求 ≥ 96.84% 对齐）
- 场景 9：`.github/workflows/ci.yml` 内容包含子串 `--cov-fail-under=96`，且**不**包含 `--cov-fail-under=90`（CI 与 pyproject addopts 同步，避免 CLI 覆盖 addopts 导致 CI 门控失效）

**验证标准（GREEN 条件）：**
- 上述 9 个场景全通过
- `pytest tests/unit/test_packaging.py` 0 失败
- `pytest`（全量）coverage ≥ 96%（由 pyproject `--cov-fail-under=96` 硬性保证；CI 同步生效因 ci.yml 也已上调），260 tests 全过
- `ruff check pyproject.toml` 不报错（虽然 ruff 不直接 lint toml，但 ruff 自身 config 应在 pyproject 内不被破坏）

**显式不在 T1 GREEN 内**：`hatch build` 与 `uvx twine check dist/*` 属于 finishing 阶段（spec 整体验收第 4-5 项），不作为 T1 任务的 RED→GREEN 条件。spec 4.10 任务表中曾把 twine check 列在 T1 的 TDD 列里，那是 spec 的描述误差，本 plan 以此处为准。

**审查要求：**
- `code-quality-reviewer-prompt.md`（必选）
- `spec-reviewer-prompt.md`（spec 对齐：D1-D6 + D2-D3 + D4-D6 字段）
- `python-reviewer-prompt.md`（涉及 `tests/unit/test_packaging.py` 的 Python 代码）

依赖：无（独立任务）。

### 任务 T2：新建 MIT LICENSE

**文件：** 新建 `LICENSE`（项目根目录）

**测试规格：** 由 implementer subagent 在 `tests/unit/test_packaging.py` 中追加以下场景：

- 场景 1：项目根目录存在文件名为 `LICENSE`（无扩展名）的文件
- 场景 2：LICENSE 第一行为 `MIT License`
- 场景 3：LICENSE 文件含子串 `Copyright (c) 2026 dengdi`
- 场景 4：LICENSE 文件含完整的 MIT 许可条款关键短语，至少包含：
  - `Permission is hereby granted, free of charge`
  - `WITHOUT WARRANTY OF ANY KIND`
  - `INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY`
- 场景 5：行数在 21-23 之间（标准 MIT 模板行数，允许末尾空行差异）

**验证标准（GREEN 条件）：**
- 上述 5 个场景全通过
- LICENSE 内容与 spec 4.7 节给出的完整文本字符级一致（除尾部可能的空行差异）
- `pytest`（全量）0 失败

**审查要求：**
- `code-quality-reviewer-prompt.md`（必选）
- `spec-reviewer-prompt.md`（spec 对齐：D7 + spec 4.7 节）
- `python-reviewer-prompt.md`（如本任务确实修改了 `tests/unit/test_packaging.py`）

依赖：T1（需要 `tests/unit/test_packaging.py` 文件已由 T1 建立；本任务在该文件中追加场景而非新建）。

### 任务 T3：transformers 从 core 移到 `[bge]` extra + uv lock 重新生成

**文件：** 修改 `pyproject.toml`、`uv.lock`（lockfile 由 `uv lock` 自动重新生成，禁止手动编辑）

**测试规格：** 在 `tests/unit/test_packaging.py` 中追加以下场景：

- 场景 1：`tomllib.load()` 读取 pyproject.toml，`project.dependencies` 列表中**不**包含任何以 `transformers` 开头的字符串
- 场景 2：`project.optional-dependencies.bge` 列表包含 `transformers>=4.44.2,<5` 字符串字面量
- 场景 3：`project.optional-dependencies.bge` 列表同时包含 `FlagEmbedding>=1.2`（不能误删）
- 场景 4：`project.optional-dependencies.all` 仍然引用 `ragline[server,bge,chroma,pdf]`（不变）
- 场景 5：现有所有依赖（除 transformers 外）仍在 `project.dependencies` 中（langgraph / langchain-core / openai / pydantic / pyyaml / rank-bm25 / chromadb 不变）
- 场景 6：`uv lock --check` 命令退出码为 0（lockfile 与 pyproject 已同步；不要用 mtime 比较——uv 不变更内容时不会重写文件）

**验证标准（GREEN 条件）：**
- 上述 6 个场景全通过
- `uv lock --check` 0 退出码（lockfile 与 pyproject 已同步）
- `uv sync --extra dev --extra bge` 不报错（bge 必须一起装；理由见下方"transformers 迁移后的测试环境要求"）
- `pytest`（全量）260 tests 全过，coverage 不降
- 没有运行期测试因 transformers 找不到而失败

**transformers 迁移后的测试环境要求：**

`tests/unit/test_providers/test_reranker.py`（L80 / L103 / L148）使用绝对路径 `patch("torch.cuda.is_available")`。`torch` 是 `FlagEmbedding` 的传递依赖，仅在 `[bge]` extra 中存在。若只装 `[dev]`，这 3 个测试在 `with patch(...)` 进入时会 `ModuleNotFoundError: No module named 'torch'`，**不是** transformers 找不到，而是更深的 torch 找不到。

**修复方式（implementer 二选一）：**

- 方式 A（推荐，最小改动）：在 T3 中保持测试代码不动，将 GREEN 条件的 sync 命令改为 `uv sync --extra dev --extra bge`，CI 与本地都装 bge。本任务采用此方式。
- 方式 B：在 T3 中同时修改 `test_reranker.py`，把 `patch("torch.cuda.is_available")` 改为 `patch("ragline.providers.reranker.bge.torch")` + 在 bge.py 中把 torch 提到模块级 lazy 引用。改动面更大，不推荐。

**"最小安装不拉 transformers" 的验证不在本任务内**：放到 finishing 阶段手动跑（在干净 venv 里 `pip install dist/ragline-*.whl` 不带 extras，确认 `pip list` 不含 transformers）。

**审查要求：**
- `code-quality-reviewer-prompt.md`（必选）
- `spec-reviewer-prompt.md`（spec 对齐：D8）
- `python-reviewer-prompt.md`（涉及 `tests/unit/test_packaging.py` 的修改）

依赖：T1（T1 完成后 `pyproject.toml` 是最新状态，T3 在此基础上做 dependencies 移动）。

### 任务 T4：新建 CHANGELOG.md 记录 0.1.0

**文件：** 新建 `CHANGELOG.md`（项目根目录）

**测试规格：** 在 `tests/unit/test_packaging.py` 中追加以下场景：

- 场景 1：项目根目录存在 `CHANGELOG.md`
- 场景 2：第一行以 `# Changelog` 开头
- 场景 3：文件含子串 `Keep a Changelog`（引用规范名）
- 场景 4：文件含子串 `Semantic Versioning`（引用规范名）
- 场景 5：文件含 `## [0.1.0] - 2026-05-24` 节标题
- 场景 6：`## [0.1.0]` 节下含 `### Added` 段标题
- 场景 7：`### Added` 段下含 7 个 `####` 子模块标题，覆盖 spec 4.9 节列出的 7 个主题（关键词检查，宽松匹配）：
  - 顶层 API（关键词：`RAG` / `RaglineConfig`）
  - 检索引擎（关键词：`LangGraph` / `query_transform`）
  - 内置 Handlers（关键词：`transforms` / `retrievers` / `rrf`）
  - Providers（关键词：`LLM` / `Embedding` / `ChromaDB`）
  - HTTP Server（关键词：`FastAPI` / `[server]`）
  - 对外测试支持（关键词：`FakeLLM` / `ragline.testing`）
  - 工程基线（关键词：`py.typed` / `coverage` 或 `import-linter`）
- 场景 8：含 `### Notes` 段，说明 0.x API 可能调整

**验证标准（GREEN 条件）：**
- 上述 8 个场景全通过
- 文件结构符合 Keep-a-Changelog 1.1.0 标准（`# Changelog` → 导言 → `## [version] - date` → `### Added/Changed/...` → 项目列表）
- `pytest`（全量）0 失败

**审查要求：**
- `code-quality-reviewer-prompt.md`（必选）
- `spec-reviewer-prompt.md`（spec 对齐：D9 + spec 4.9 节）
- `python-reviewer-prompt.md`（如本任务确实修改了 `tests/unit/test_packaging.py`）

依赖：T3（需要 T3 完成后的 `tests/unit/test_packaging.py` 已包含 T1+T2+T3 场景；T4 在末尾追加自己的场景）。

## 测试策略

- **单元测试**：`tests/unit/test_packaging.py`（新建）—— 所有 4 个任务的 GREEN 条件都通过该文件断言（tomllib 读取 + 文件内容 / 行数检查）
- **集成测试**：本 feature 无新增集成测试。但**现有所有 260 tests 必须 100% 通过且 coverage ≥ 96.84%**，这是 T3 transformers 迁移不破坏现有测试的保证
- **E2E 测试**：不涉及（无 UI、无 HTTP 端点变更）
- **构建验证**（在 finishing 阶段手动跑，不写为 pytest）：
  - `uv run hatch build` 生成 `dist/ragline-0.1.0.tar.gz` + `dist/ragline-0.1.0-py3-none-any.whl`
  - `uvx twine check dist/*` 输出 `PASSED`

## E2E 稳定性要求

不适用（本 feature 不引入 E2E 测试）。

## 风险与缓解

- **风险 1**：hatchling 版本不足支持 PEP 639 SPDX license 表达式（`license = "MIT"` 而非 `license = { file = "LICENSE" }`）
  - 影响：构建时报错 `license must be a table containing "file" or "text" key`
  - 缓解：`pyproject.toml` 当前 `[build-system].requires = ["hatchling"]` 未 pin 版本，pip/uv 会拉最新版（≥ 1.27 已 GA 数月）。如确实出错，在 T1 任务中显式 pin 为 `["hatchling>=1.27"]`
- **风险 2**：T3 transformers 迁移后，`test_reranker.py` L80/L103/L148 用绝对路径 `patch("torch.cuda.is_available")`，torch 是 FlagEmbedding 的传递依赖，仅在 `[bge]` extra 中存在。`uv sync --extra dev`（不带 bge）后这 3 个测试会因 torch ModuleNotFoundError 失败
  - 缓解：T3 GREEN 条件已固定为 `uv sync --extra dev --extra bge`；本地与 CI 一并装 bge 才跑测试。这是有意决策（方式 A），保留代码不动；如未来要 enforce "测试不依赖 bge 也能跑" 是单独议题，不在本 PR 范围
- **风险 3**：uv.lock 在不同机器上生成结果有差异（因 Python 版本 / 平台不同导致依赖图差异）
  - 缓解：T3 在当前 worktree 内由当前 uv 实例生成；提交后 CI 跑 `uv sync --extra dev --extra bge` 必须无报错
- **风险 4**：CHANGELOG 内容与实际功能不符（implementer 可能漏列或加入不存在的功能）
  - 缓解：spec 4.9 节列出 7 个子模块的关键词，T4 GREEN 条件用关键词匹配硬性检查；spec-reviewer 会比对 spec 4.9 节
- **风险 5**：pyproject.toml 当前 `--cov-fail-under=90` 与 plan 验收要求 `coverage ≥ 96.84%`（不降）不匹配。如新增测试稀释覆盖率到 91-96%，pytest 仍通过但隐式违反 plan 要求；且 `.github/workflows/ci.yml` line 48 显式 CLI `--cov-fail-under=90` 覆盖 pyproject addopts，仅改 pyproject 在 CI 上失效
  - 缓解：T1 任务双侧同步上调到 96——pyproject `[tool.pytest.ini_options].addopts` + `.github/workflows/ci.yml` line 48 均改为 `--cov-fail-under=96`。这样本地 pytest 与 CI 均按 96 门控

## 验收标准

- [ ] T1-T4 任务全部通过 SDD per-task 三道门控（TDD / Quality Gate / Code Review）
- [ ] `pytest`（全量）260 tests 全过，coverage ≥ 96.84%
- [ ] `uv lock --check` 0 退出码
- [ ] `uv run hatch build` 生成 wheel + sdist 成功
- [ ] `uvx twine check dist/*` 输出 PASSED
- [ ] LICENSE 文件存在，内容与 SPDX MIT 一致
- [ ] CHANGELOG.md 包含 0.1.0 节及 7 个子模块
- [ ] pyproject.toml 元数据完整（authors / license / urls / keywords / classifiers）
- [ ] minimal 安装（`pip install .` 不带 extras）依赖图不含 transformers（在 finishing 阶段手动验证）
- [ ] `[bge]` 安装仍然拉到 transformers 且 BGE rerank 单元测试通过（在 finishing 阶段手动验证）

## 已审定决策

（来源：spec `docs/specs/pypi-release-prep-design.md` 中的"已审定决策"段；本节由 writing-plans 在 plan-review 阶段如确认决策为有意保留时填入。下列条目均来自 spec D1-D10，已与用户确认。）

- **决策点 D1**：Development Status classifier 选 `4 - Beta`
  - 审定结论：260 tests + 96.84% coverage + 完整外部契约支持 Beta 级
  - 有意理由：Alpha 信号过保守，会暗示接口随时破带动；Beta 更准确反映成熟度
- **决策点 D2**：Repository URL 用 `https://github.com/dengd1937/ragline`
  - 审定结论：当前 origin remote 一致
  - 有意理由：发布前不迁仓
- **决策点 D4**：使用 PEP 639 SPDX 表达式 `license = "MIT"` + `license-files = ["LICENSE"]`，而非 PEP 621 旧格式 `license = { file = "LICENSE" }`
  - 审定结论：现代格式，符合 PyPI 推荐
  - 有意理由：hatchling 1.27+ 已 GA 数月，未 pin 即可拉最新；旧格式将在未来版本被弃用
- **决策点 D5**：keywords 用 `["rag", "llm", "langgraph", "retrieval", "embedding", "chromadb", "bm25"]`
  - 审定结论：覆盖主要检索场景，PyPI 搜索可被命中
  - 有意理由：精炼到 7 个核心关键词，避免 SEO 式堆砌
- **决策点 D8**：transformers 从 core 移到 `[bge]` extra，而非保留 core 并接受 200-500MB 传递依赖
  - 审定结论：transformers 在 ragline 源码中 0 命中，只服务于 BGE reranker
  - 有意理由：最小安装体验 > 单次安装便利；BGE 用户显式 `pip install ragline[bge]` 仍可拿到 pin
- **决策点 D9**：CHANGELOG.md 与 P0 元数据合并在同一个 PR
  - 审定结论：版本号 bump 与变更记录天然耦合
  - 有意理由：单 PR 闭环，避免 CHANGELOG 比代码晚 merge 导致版本号悬空
- **决策点 D10**：chromadb 保持在 core deps，不挪到 extras
  - 审定结论：ChromaVectorStore 是当前唯一 vector store 实现，默认 `GraphConfig.retrievers = ["vector", "bm25"]` 直接走它
  - 有意理由：裸 `pip install ragline` 必须开箱即用；若 chromadb 推回 extra，最小安装的默认配置立即 ImportError
