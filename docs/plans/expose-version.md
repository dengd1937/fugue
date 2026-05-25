# 实现计划：expose-version

## 执行方式

本计划通过 `/subagent-driven-development` skill 执行。以下任务描述是 skill 的输入规格，不是直接执行指令。

## 概述

为 ragline 暴露标准的 `__version__` 公开属性（动态从 `importlib.metadata` 读取，单一真相源 pyproject.toml）+ README 顶部加 4 个 shields.io badges。单任务合打，文件数 3 个（`src/ragline/__init__.py`、`tests/unit/test_packaging.py`、`README.md`），符合 SDD 任务粒度。

## 需求

- `import ragline; print(ragline.__version__)` 输出 `"0.1.0"`（与 pyproject `project.version` 同步）
- `ragline.__version__` 是 str 类型，可被外部代码安全使用
- `__version__` 在 `ragline.__all__` 中
- README 顶部含 4 个 badges：PyPI version / License / Python versions / CI status
- 现有 288 tests 全过，coverage ≥ 96%
- ruff format / check / mypy strict (src/ragline) PASS
- 不引入新依赖（`importlib.metadata` 是 Python 3.8+ stdlib）
- 不修改 pyproject.toml 的 `version` 字段（仍为 `"0.1.0"`，发布前不再 bump）

## 架构变更

- `src/ragline/__init__.py`（修改）：
  - 顶部新增 `from importlib.metadata import version as _pkg_version` import
  - 在现有 import 之后赋值 `__version__ = _pkg_version("ragline")`
  - `__all__` 列表追加 `"__version__"`（按字母序合理位置）
- `tests/unit/test_packaging.py`（修改）：末尾追加 2 个测试场景（场景 29 + 30）
- `README.md`（修改）：在文档顶部 H1 标题与第一段说明之间插入 4 个 badges markdown

## 环境前置（Environment Prerequisites）

- **uv**：依赖管理 + 测试运行
  - 验证：`uv --version`（要求 ≥ 0.4，本地实测 0.7.6）
  - 修复：`brew reinstall uv` 或重新安装

无其他环境需求（`importlib.metadata` 是 Python stdlib）。

## 实现步骤

### 任务 T1：暴露 `__version__` + README badges

**文件：** 修改 `src/ragline/__init__.py`、`tests/unit/test_packaging.py`、`README.md`

**测试规格：** 在 `tests/unit/test_packaging.py` 末尾追加以下场景：

- 场景 29：`import ragline`；`ragline.__version__` 存在且是 `str` 类型；匹配 SemVer 正则 `^\d+\.\d+\.\d+(?:[.-].+)?$`（支持 0.1.0、0.1.0-rc1、0.1.0.post1 等）
- 场景 30：`ragline.__version__` 字符串等于 `pyproject["project"]["version"]`（验证单一真相源对齐）。**必须复用** test_packaging.py 既有的 `pyproject` fixture（用 `with PYPROJECT_PATH.open("rb") as f: tomllib.load(f)` 加载，避免 ruff SIM115 触发裸 `open()` 警告）

不在 test_packaging.py 中追加 README badges 验证（README 内容由 spec-reviewer 在 review 阶段对照检查）。

**验证标准（GREEN 条件）：**

- 上述 2 个场景全过（pytest tests/unit/test_packaging.py 30 场景全过）
- `pytest`（全量）≥ 290 passed（288 原有 + 2 新增），coverage ≥ 96%（仅追加测试不降覆盖率）
- `import ragline; print(ragline.__version__)` 输出 `"0.1.0"`
- `ragline.__all__` 包含 `"__version__"`
- README 顶部 H1 标题下方含 4 个 badges markdown（PyPI / License / Python / CI）
- ruff format / check 通过
- mypy strict 在 `src/ragline` 上 PASS

**README badges 内容（implementer 直接使用，spec D3 给定）：**

```markdown
[![PyPI version](https://img.shields.io/pypi/v/ragline.svg)](https://pypi.org/project/ragline/)
[![License: MIT](https://img.shields.io/pypi/l/ragline.svg)](https://github.com/dengd1937/ragline/blob/main/LICENSE)
[![Python versions](https://img.shields.io/pypi/pyversions/ragline.svg)](https://pypi.org/project/ragline/)
[![CI status](https://github.com/dengd1937/ragline/actions/workflows/ci.yml/badge.svg)](https://github.com/dengd1937/ragline/actions/workflows/ci.yml)
```

按 PyPI / License / Python / CI 顺序排列。

**README 插入位置说明：** 当前 README 顶部结构：

```
# Ragline

Configuration-driven RAG. Topology stays, behavior plugs in.

Ragline 是一个配置驱动的 Python RAG 库...
```

implementer 应在 `# Ragline` 标题与第一段英文 tagline 之间插入 badges（即 badges 紧贴 H1 标题下方，常见 Python 库 README 习惯）。

**审查要求：**

- `code-quality-reviewer-prompt.md`（必选，含 OWASP）
- `spec-reviewer-prompt.md`（spec 对齐：D1-D5）
- `python-reviewer-prompt.md`（涉及 src/ragline/__init__.py + tests/unit/test_packaging.py 的 Python 代码）

依赖：无（独立任务）。

## 测试策略

- **单元测试**：`tests/unit/test_packaging.py` 末尾追加 2 个场景断言 `__version__` 存在性与同步性
- **集成测试**：本任务无新增集成测试。但**现有 288 tests 必须 100% 通过**，因为 `src/ragline/__init__.py` 的修改会影响所有 import ragline 的测试
- **E2E 测试**：不涉及
- **手动验证**（finishing 阶段可选）：在 worktree 里 `uv run python -c "import ragline; print(ragline.__version__)"` 确认输出

## 风险与缓解

- **风险 1**：`importlib.metadata.version("ragline")` 在 editable install 未生效时 raises `PackageNotFoundError`，会导致整个 `import ragline` 失败（影响所有现有测试）
  - 缓解：项目使用 `uv sync` 创建 editable install，正常情况下 ragline 始终在 metadata 中。如果出错，立即暴露安装问题（fail-loud 是有意决策，见 spec D1）。implementer 在写 import 时不要捕获 `PackageNotFoundError`。
- **风险 2**：`__version__` 在 `import` 时即查询 metadata，可能影响 import 性能（毫秒级，可忽略）
  - 缓解：实测影响极小（importlib.metadata 有 cache）；不预先优化
- **风险 3**：README badges URL 中 `dengd1937/ragline` 仓库名硬编码，未来仓库重命名时需同步更新
  - 缓解：与 pyproject.toml `[project.urls]` 的 Repository URL 保持一致（同一来源约束，已在 D3 锁定）；未来重命名时一并更新

## 验收标准

- [ ] T1 通过 SDD per-task 三道门控（TDD / Quality Gate / Code Review）
- [ ] `pytest`（全量）≥ 290 passed，coverage ≥ 96%
- [ ] `import ragline; print(ragline.__version__)` 输出 `"0.1.0"`
- [ ] `"__version__" in ragline.__all__`
- [ ] README 顶部含 4 个 badges
- [ ] ruff / mypy strict (src/ragline) PASS

## 已审定决策

（来源：spec `docs/specs/expose-version-design.md` 中的 D1-D7；plan-review 阶段如确认有意保留则填入此段。）

- **决策点 D1**：`__version__` 实现策略 = `importlib.metadata.version("ragline")` 动态查询
  - 审定结论：单一真相源，stdlib 工具，fail-loud
  - 有意理由：避免双向维护漂移；包未安装时立即暴露（不静默 fallback）
- **决策点 D3**：4 个 badges（PyPI / License / Python / CI），不含 coverage
  - 审定结论：Coverage 需 Codecov 单独配置，超出本 PR 微型 scope
  - 有意理由：单一 PR 聚焦版本号 + 基础 badges，后续 PR 加 coverage
- **决策点 D7**：单任务合打（不拆 #5 + #7）
  - 审定结论：两者都是 PyPI 首发外观打磨，scope 关联紧密
  - 有意理由：单次 review 高效；3 文件仍在 SDD 任务粒度内
