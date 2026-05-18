# 实现计划：rename-to-ragline

## 执行方式

本计划通过 `/subagent-driven-development` skill 执行。以下任务描述是 skill 的输入规格，不是直接执行指令。

## 概述

将本库从 `fugue` 全量重命名为 `ragline`（解除 PyPI 包名冲突 + 统一品牌）。纯机械标识符重命名，零业务逻辑变更，靠现有 248 非 e2e 测试套件零回归 + mypy 零错误兜底正确性。

## 需求

- `fugue` → `ragline` 全量：包目录、import 路径、8 个公开 `Fugue*` 类、pyproject 元数据、CLI 命令、entry_points 组、e2e 环境变量、文档
- 每个任务完成后 `uv run pytest`（默认 `-m 'not e2e'`，248 测试）必须全绿、`mypy src` 零错误
- 不改业务逻辑/行为，不动 7 节点拓扑，不借机重构
- GitHub 仓库改名为手动收尾项，不在代码 diff 内

## 架构变更

- `src/fugue/` → `src/ragline/`（目录 `git mv`，保留子结构 config/providers/server/registry/api/engine/handlers）
- `pyproject.toml`：`name`、`[project.scripts]`、`[tool.hatch.build.targets.wheel] packages`、`[tool.pytest] --cov`、`[all]` extra 自引用
- `src/ragline/registry/__init__.py`：`entry_points(group=...)` 组名 + docstring
- `tests/e2e/conftest.py`：5 个环境变量名 + docstring 串
- `README.md` + `docs/specs/` 三个历史 spec + `fugue-design.md` 改名

### 关键排序决策（spec 切片 1+3 合并的理由）

spec 列出 6 个有序切片，但切片 1（目录重命名）与切片 3（pyproject）**无法各自独立通过标准测试门控**：

- `git mv src/fugue src/ragline` 后，editable 安装的包映射仍指向旧路径 + `packages = ["src/fugue"]`，`import ragline` 无法解析 → 必须同步改 pyproject `packages` 并 `uv sync` 重装
- `--cov=fugue` 在包改名后令覆盖率目标失效，`--cov-fail-under=90` 在 0% 覆盖率下失败 → 必须同步改 `--cov=ragline`

故合并为单一原子任务（任务 1），使每个任务结束时 `uv run pytest` 标准命令真正全绿。这是 writing-plans 对「如何实现」的精化，不改变 spec「有序语义切片、每片测试绿」的意图。

### 任务粒度说明

任务 1、2 是跨数十文件的**单一原子机械重命名**。"每任务 1-3 文件"约束针对的是多关注点混合任务；机械重命名是单一逻辑变更，拆分到文件级只会产生中间破损状态（严格更差）。因此任务 1、2 按"一次机械 sweep + 全量套件零回归验证"作为正确原子单元，文件数不作上限约束。

## 环境前置（Environment Prerequisites）

- **uv + pytest 测试运行器**：项目唯一测试依赖；非 e2e 测试无外部服务依赖
  - 验证：`uv run pytest -m 'not e2e' --collect-only -q`（退出码 0 即就绪）
  - 修复：`uv sync`
- **e2e 测试外部依赖（仅任务 4 相关，默认门控不跑）**：e2e 测试由 `-m 'not e2e'` 默认排除，需 OpenAI 兼容 API key + 模型，不在每任务门控内
  - 验证：不适用（默认门控不执行 e2e）
  - 修复：不适用

## 实现步骤

### 阶段 1：代码标识符重命名

#### 任务 1：包目录重命名 + 全量小写 `fugue` token + pyproject/工具链配置 + uv 重装

**文件：** `git mv src/fugue/ → src/ragline/`（整目录）；修改 src 全部 `.py`、tests 全部 `.py`（含 `tests/e2e/*.py`，不止 conftest）、`pyproject.toml`、`uv.lock`（`uv lock` 重生成）、`.importlinter`、`.pre-commit-config.yaml`、`.github/workflows/ci.yml`

**测试规格：**
- 替换范围 = **全部小写 `fugue` token**（**不动** `Fugue*` 大写类名→任务 2，**不动** `FUGUE_E2E_*` 环境变量→任务 4；本任务后类名仍为旧名但位于新包路径下，套件依旧绿）：
  - src 内所有 `from fugue.X import Y` → `from ragline.X import Y`、`import fugue` → `import ragline`
  - tests（含 `tests/e2e/*.py`：conftest.py / test_e2e_basic.py / test_e2e_server.py / test_e2e_multipath.py 等均含 `from fugue import …`）内所有 `from fugue.X`、`import fugue`、以及 `patch("fugue.…")` / `monkeypatch` / mocker 等**字符串形式模块路径** `"fugue...."` → `"ragline...."`
  - **运行时用户可见默认值**：`src/fugue/config/__init__.py` 中 `IngestConfig` 与 `IngestConfigSchema` 的 `persist_dir: str = "./.fugue"`（两处，约 63 / 134 行）→ `"./.ragline"`（磁盘实际创建目录，品牌一致）
  - **运行时用户可见字符串**（关键，否则提示用户安装不存在的包）：`src/fugue/_optional.py` 的 `f"pip install 'fugue[{extra}]'"` → `'ragline[{extra}]'`；及其断言该消息的测试 `tests/unit/test_optional.py`（含 `"pip install 'fugue[pdf]'"` 字面量与 `f"…'fugue[{extra}]'"`）、`tests/unit/test_handlers/test_parsers.py`（`"fugue[pdf]"`）、`tests/unit/test_providers/test_reranker.py`（`"fugue[bge]"`）、`tests/integration/test_optional_deps.py`（`_SCRIPT` 子进程串内 `"pip install 'fugue[pdf]'"` / `'fugue[bge]'`）
  - **硬编码 logger 名 + CLI prog**（非 import，spec 易漏；测试用 `caplog(logger="fugue.xxx")` 断言，源/测试须同步否则 logger 层级错配致断言失败）：`src/fugue/server/cli.py` `logging.getLogger("fugue.server")`、`src/fugue/engine/nodes/retrieve.py` `logging.getLogger("fugue.engine.nodes.retrieve")` → `ragline.*`；`cli.py` `argparse` 的 `prog="fugue"` → `prog="ragline"`；`tests/unit/test_server/test_cli.py` `sys.argv = ["fugue", "serve", …]` → `["ragline", …]`，及各 test 内 `caplog.at_level(..., logger="fugue.*")` 串
  - `pyproject.toml`：`name = "ragline"`；`[project.scripts]` → `ragline = "ragline.server.cli:main"`；`[tool.hatch.build.targets.wheel] packages = ["src/ragline"]`；`addopts` 中 `--cov=fugue` → `--cov=ragline`；`[all]` 自引用 `"fugue[server,bge,chroma,pdf]"` → `"ragline[...]"`；如有 `[tool.coverage]`/`[tool.importlinter]` 内联段一并改
  - `.importlinter`：`root_package = fugue` → `ragline`，全部 `source_modules`（`fugue.api`/`fugue.config`/`fugue.registry`/`fugue.handlers`/`fugue.providers`/`fugue.server` 等）与 `ignore_imports`（`fugue.api.rag -> fugue.engine.graph` 等）的 `fugue.` → `ragline.`（**必须与目录重命名原子**，`lint-imports` 是 CI 门控）
  - `.pre-commit-config.yaml`：本地 mypy hook 的 `files: ^src/fugue/` 与 `args: ["src/fugue"]` → `ragline`（否则 hook 静默不触发，commit 门控失效）
  - `.github/workflows/ci.yml`：`uv run mypy src/fugue` → `src/ragline`、`--cov=fugue` → `--cov=ragline`
  - `uv lock` 重新生成 `uv.lock`；`uv sync` 重装使 `import ragline` 可解析
- 不改任何函数体逻辑、签名、行为；仅路径/名称/字符串替换
- 不新增功能测试（纯机械重命名）；现有套件即回归网

**验证标准：**
- `import ragline` 成功；`src/fugue/` 不存在、`src/ragline/` 子结构完整
- `uv run pytest`（默认 `-m 'not e2e'`，248 测试）全绿；`uv run pytest -m e2e --collect-only -q` 成功收集（验证 e2e import 路径已修复）
- `mypy src` 零错误；`uv run lint-imports --config .importlinter` 通过
- `grep -rIn 'fugue' . --exclude-dir=.git --exclude-dir=docs/plans --exclude-dir=.venv --exclude=README.md | grep -v -e 'Fugue' -e 'FUGUE_'` 允许残留仅 `docs/specs/`（历史 spec 留任务 5）、`README.md`（留任务 5）、本 feature 自身 spec——src/tests/配置文件区零小写 `fugue` 残留
- `uv lock --check` 一致

**审查要求：** code-quality-reviewer（必选）+ python-reviewer（diff 含 .py）

---

#### 任务 2：公开类标识符 `Fugue*` → `Ragline*`（src + tests 同步，先长后短）

**文件：** src 全部含 `Fugue*` 的 `.py`；tests 全部含 `Fugue*` 的 `.py`（任务 1 后 import 路径已是 `ragline`，此任务只动大写类标识符）

**测试规格：**
- 8 个公开类标识符全量重命名，按**前缀安全顺序**（同一前缀族内更长的先替，避免 `FugueConfig` 把 `FugueConfigError` 截断、`FugueError` 把 `Fugue*Error` 子类截断）：
  1. `FugueConfigError` → `RaglineConfigError`
  2. `FugueConfigSchema` → `RaglineConfigSchema`
  3. `FugueConfig` → `RaglineConfig`（必须晚于 1、2，因是其前缀）
  4. `FugueEmbeddingError` → `RaglineEmbeddingError`
  5. `FugueLLMError` → `RaglineLLMError`
  6. `FugueRegistryError` → `RaglineRegistryError`
  7. `FugueRetrieverError` → `RaglineRetrieverError`
  8. `FugueError` → `RaglineError`（必须最后，是所有 `Fugue*Error` 前缀）
- 范围：类定义、所有引用、`__all__` 导出、异常继承链、tests 中的引用与断言（含 `pytest.raises(FugueXxx)`）、docstring 中作为 API 名出现的 `Fugue*`
- **品牌名 `Fugue` 字面量**（非标识符，但 `grep -n 'Fugue'` 会命中，且均为「指代本库」须改）：`src/ragline/server/app.py` `FastAPI(title="Fugue", …)` → `title="Ragline"`；`src/ragline/server/cli.py` 警告串 `"⚠️ Fugue 0.x: …"` → `"⚠️ Ragline 0.x: …"`；`tests/e2e/test_e2e_server.py` `FastAPI(title="Fugue E2E")` → `title="Ragline E2E"`；各模块 docstring 中的 `"""Fugue …"""`（如 `Fugue 异常基类。`/`Fugue 主入口。`）→ `Ragline …`
- 不改类的行为、继承结构、方法签名；仅标识符/品牌名替换
- 不新增功能测试；现有套件即回归网

**验证标准：**
- `uv run pytest`（248 测试）全绿
- `mypy src` 零错误
- `grep -rn 'Fugue' src tests` 无残留（含 docstring API 名与品牌名字面量）
- 8 个 `Ragline*` 类均可从原模块路径正常 import，继承链不变（如 `RaglineConfigError` 仍继承 `RaglineError`）

**审查要求：** code-quality-reviewer（必选）+ python-reviewer（diff 含 .py）

---

#### 任务 3：entry_points 组 `fugue.handlers` → `ragline.handlers`

**文件：** `src/ragline/registry/__init__.py`；如有测试断言该组名则同步（`tests/` 下 registry 相关测试）

**测试规格：**
- `entry_points(group="fugue.handlers")` → `entry_points(group="ragline.handlers")`（约第 67 行，以当时文件为准定位）
- 同文件 docstring 中描述该 group 的字符串 `'fugue.handlers'` → `'ragline.handlers'`（约第 64 行）
- 若 `tests/` 中存在断言 entry_points group 名的测试，同步更新断言字符串
- 行为不变：仅第三方插件发现契约的 group 字符串变更

**验证标准：**
- `uv run pytest`（248 测试）全绿
- `mypy src` 零错误
- `grep -rn 'fugue\.handlers\|group="fugue\|group='"'"'fugue' src/ragline/registry` 无残留
- registry 加载逻辑行为等价（内置 handler 注册路径不受影响，仅外部插件 group 名变更）

**审查要求：** code-quality-reviewer（必选）+ python-reviewer（diff 含 .py）

---

#### 任务 4：e2e conftest 环境变量 `FUGUE_E2E_*` → `RAGLINE_E2E_*`

**文件：** `tests/e2e/conftest.py`

**测试规格：**
- 5 个环境变量名替换：`FUGUE_E2E_BASE_URL` → `RAGLINE_E2E_BASE_URL`；`FUGUE_E2E_LLM_MODEL` → `RAGLINE_E2E_LLM_MODEL`；`FUGUE_E2E_EMBEDDING_MODEL` → `RAGLINE_E2E_EMBEDDING_MODEL`；`FUGUE_E2E_RERANKER_MODEL` → `RAGLINE_E2E_RERANKER_MODEL`；以及该文件内任何其余 `FUGUE_E2E_*`
- `os.environ.get(...)` 调用与 docstring 中列出的变量名串同步更新
- 不改 fixture 逻辑、默认值常量值；仅环境变量键名与文档串
- e2e 测试由默认 `-m 'not e2e'` 排除，不在 248 门控内；本任务验证靠 e2e 测试可正常收集（collect）且 conftest 可正常 import

**验证标准：**
- `uv run pytest`（默认 248 测试）全绿（不受影响）
- `uv run pytest -m e2e --collect-only -q` 成功收集、conftest 无 import 错误
- `mypy src` 零错误
- `grep -rn 'FUGUE_E2E\|FUGUE_' tests/e2e/conftest.py` 无残留

**审查要求：** code-quality-reviewer（必选）+ python-reviewer（diff 含 .py）

---

### 阶段 2：文档

#### 任务 5：文档全量更新（README + 历史 spec + spec 改名）

**文件：** `README.md`；`docs/specs/lazy-optional-imports-design.md`；`docs/specs/chromadb-core-dependency-design.md`；`git mv docs/specs/fugue-design.md docs/specs/ragline-design.md`；`tests/fixtures/e2e/doc1.md`、`doc2.md`、`doc3.md`、`doc_chinese.md`（语料 fixture 含品牌名 "Fugue"/"Fugue 是…"）

**测试规格：**
- `README.md`（约 23 处 `fugue`/`Fugue`/`FUGUE`，含标题 `# Fugue`、`Why Fugue?`、`Fugue 是…`）：所有「指代本库 / 安装命令 / import 示例 / CLI 命令 / 品牌标题」的 `fugue`/`Fugue` → `ragline`/`Ragline`（含 `pip install fugue`、`import fugue`、`fugue serve`、`fugue[...]` extras、标题）；YAML 配置示例中的 `persist_dir: ./.fugue` 改为 `./.ragline`（示例路径，品牌一致；非保留项）
- `docs/specs/lazy-optional-imports-design.md`、`docs/specs/chromadb-core-dependency-design.md`：
  - 「指代本库」语境的 `fugue` → `ragline`（如代码路径 `fugue.handlers.parsers.pdf`、`pip install 'fugue[pdf]'`、`import fugue`）
  - 「叙述 PyPI 包名 fugue 被占用 / 改名缘由」语境：**不机械替换**，改写为「原名 fugue」「旧包名 fugue」等表述以保留语义真实性
- `docs/specs/fugue-design.md` → `git mv` 为 `docs/specs/ragline-design.md`；更新其文件头 metadata `feature:` 锚点（`fugue` → `ragline`，spec 路径同步）；正文按上述指代/叙述区分原则替换
- `tests/fixtures/e2e/doc{1,2,3}.md` / `doc_chinese.md`：语料中作为「指代本库」的 `Fugue`/`Fugue 是…` → `Ragline`（这些是 e2e 检索语料，品牌一致即可，无断言耦合其具体措辞）
- `docs/specs/rename-to-ragline-design.md`（本 feature 自身 spec）**不改**——其叙述对象就是本次改名，`fugue` 出现是必要历史指代
- 不改 `docs/plans/`（计划文件由 finishing 清理）

**验证标准：**
- `grep -rn 'fugue\|Fugue' README.md docs/specs/ragline-design.md docs/specs/lazy-optional-imports-design.md docs/specs/chromadb-core-dependency-design.md tests/fixtures/e2e/` 仅余有意保留的历史叙述（「原名 fugue」类），无「指代本库」/品牌标题/`./.fugue`/语料品牌名残留
- `docs/specs/fugue-design.md` 不存在、`docs/specs/ragline-design.md` 存在且 `feature: ragline`、`spec:` 路径已更新
- README 安装/使用示例命令与改名后的真实包名/CLI 一致（`pip install ragline`、`import ragline`、`ragline serve`）
- 无测试代码变更，248 测试不受影响（保持全绿）

**审查要求：** code-quality-reviewer（必选）

## 测试策略

- 单元/集成测试：不新增。本计划是纯机械标识符重命名，行为完全不变；为重命名写新断言无工程价值。正确性验证 = 现有 248 非 e2e 测试套件每任务后零回归 + `mypy src` 零错误 + `grep` 残留收敛 + 隔离环境 `pip install <wheel>` 后 `import ragline` 成功。
- 端到端测试：e2e 由 `-m 'not e2e'` 默认排除，不在每任务门控内；任务 4 后验证 e2e 可正常收集（conftest import 无误）。

## E2E 稳定性要求

本计划不新增/不修改 e2e 测试用例逻辑（任务 4 仅改环境变量键名）。现有 e2e 套件稳定性要求维持不变，不在本计划范围内调整。

## 风险与缓解

- **风险**：mass replace 误伤子串（如注释、URL、第三方名中的 `fugue`/`Fugue`）
  - 缓解：`fugue` 为高辨识度 token，子串误伤低；`Fugue*` 用先长后短词边界感知替换；每任务后全套件零回归 + grep 收敛双重校验
- **风险**：任务 1 后未 `uv sync` 重装导致 `import ragline` 解析失败、套件红
  - 缓解：任务 1 测试规格显式包含 `uv sync` 重装步骤与 `import ragline` 验证项
- **风险**：`--cov=fugue` 未同步改导致覆盖率门控 0% 失败（被误判为测试回归）
  - 缓解：`--cov=ragline` 合并进任务 1，与包改名原子提交
- **风险**：文档「叙述历史包名」语境被机械替换导致语义失真
  - 缓解：任务 5 测试规格显式要求人工区分「指代本库」vs「叙述旧名」，后者改写为「原名 fugue」
- **风险**：`.importlinter` / `.pre-commit-config.yaml` / `ci.yml` 等工具链配置遗漏，per-task gate 假绿、合并后 CI 红或 pre-commit 静默失效
  - 缓解：三个配置文件已显式纳入任务 1 文件清单与替换范围；任务 1 验证含 `lint-imports` 通过；`.importlinter` 与目录重命名原子提交
- **风险**：`_optional.py` 用户可见安装提示串未改，运行时提示用户 `pip install 'fugue[...]'`（不存在的包）
  - 缓解：任务 1 替换范围显式列入该运行时字符串及其全部断言测试
- **风险**：硬编码 `getLogger("fugue.*")` 未随 import 同步，caplog 断言 logger 层级错配致测试红
  - 缓解：任务 1 显式列举 cli.py / retrieve.py 两处硬编码 logger 名 + prog
- **风险**：GitHub 仓库改名窗口期旧名失效影响 clone/CI
  - 缓解：仓库改名为手动收尾项，在代码合并后由用户在 GitHub 端协调执行，不阻塞实现

## 验收标准

- [ ] `src/ragline/` 存在、`src/fugue/` 不存在，子结构完整
- [ ] 全仓 `grep -rn 'fugue\|Fugue\|FUGUE' . --exclude-dir=.git --exclude-dir=.venv`（排除 `docs/plans/`、`docs/specs/rename-to-ragline-design.md`、有意保留的「原名 fugue」叙述）收敛为零
- [ ] 8 个 `Ragline*` 类标识符全替换，src + tests 一致，继承链不变；品牌名字面量（FastAPI title、CLI 警告、docstring）已改
- [ ] pyproject `name = "ragline"`、CLI `ragline`、`packages = ["src/ragline"]`、`--cov=ragline`、`[all]` 自引用 `ragline[...]`
- [ ] `.importlinter` `root_package`/`source_modules`/`ignore_imports`、`.pre-commit-config.yaml` mypy hook 路径、`ci.yml` mypy/cov 全部为 ragline；`lint-imports` 通过
- [ ] `src/ragline/_optional.py` 运行时安装提示串为 `ragline[...]`
- [ ] entry_points 组为 `ragline.handlers`
- [ ] `tests/e2e/conftest.py` 为 `RAGLINE_E2E_*`，e2e 可正常收集
- [ ] `uv lock --check` 一致；`uv.lock` 本包名为 ragline
- [ ] 每任务后 248 非 e2e 测试零回归；`mypy src` 零错误
- [ ] 隔离环境 `pip install <wheel>` 后 `import ragline` 成功、CLI `ragline --help` 可用（finishing 阶段收尾验证项，不分配给单个任务）
- [ ] `docs/specs/ragline-design.md` 存在且 `feature:` 锚点已更新；README + 历史 spec 无误导
- [ ] GitHub 仓库改名（手动收尾项，文档记录，不阻塞合并）
