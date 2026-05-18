# 实现计划：chromadb 提为核心依赖（chromadb-core-dependency）

## 执行方式

本计划通过 `/subagent-driven-development` skill 执行。以下任务描述是 skill 的输入规格，不是直接执行指令。

## 概述

将 chromadb 从 `[project.optional-dependencies].chroma` 可选 extra 移进
`[project].dependencies` 主依赖（chromadb 是默认 ChromaVectorStore 的强制
运行时依赖）。`chroma` extra 保留为空 `[]` 向后兼容，`[all]` 保留 chroma。
同步修正 README 安装说明。零源码 / 业务逻辑改动。

## 需求

- 裸装 `pip install fugue`（无任何 extra）即自带 chromadb，`import fugue` + `RAG()` 可用
- `pip install 'fugue[chroma]'` 不报 "no extra named chroma"（空 extra 保留）
- `pip install 'fugue[all]'` 不破坏
- 既有测试套件零回归；mypy src 零错误
- README 不再把 chromadb 描述为需 `[chroma]` extra 才装

## 架构变更

- `pyproject.toml`：`[project].dependencies` 增加 `chromadb>=0.5`；
  `[project.optional-dependencies].chroma` 由 `["chromadb>=0.5"]` 改为 `[]`；
  `[all]` 的 `fugue[server,bge,chroma,pdf]` 不变（chroma 引用空 extra，合法无害）
- `uv.lock`：`uv lock` 机械重生成
- `README.md`：第 20 行 `pip install "fugue[chroma]"   # ChromaDB 向量存储` 措辞修正

## 环境前置（Environment Prerequisites）

- **uv（依赖管理 / 锁文件）**：本项目用 `.venv + uv`
  - 验证：`uv --version`
  - 修复：见 https://docs.astral.sh/uv/ 安装；项目已有 `.venv`
- **pytest（测试运行器）**：随 `[dev]` extra 装在 `.venv`
  - 验证：`.venv/bin/pytest --version`
  - 修复：`uv sync --extra dev`

## 测试策略

本变更为依赖清单 + 文档元数据调整，**无 Python 源码 / 业务逻辑改动**，
因此**不新增单元/集成测试**——为 pyproject 元数据写断言无工程价值
（writing-plans「No Placeholders」不适用于"无意义测试"，此处显式声明）。

验证改为基于命令的回归与隔离安装核验：

- 单元/集成：复用既有全量套件（`pytest -m 'not e2e'`）确认零回归
- 锁一致性：`uv lock --check`
- 隔离安装核验：构建 wheel → 全新 venv `pip install <wheel>`（**不带任何 extra**）
  → 断言 `python -c "import chromadb; import fugue"` 成功
- 向后兼容核验：全新 venv `pip install '<wheel>[chroma]'` 与
  `pip install '<wheel>[all]'` 均不报错（两者都必测，对齐验收标准）

## 任务

### 任务 1：pyproject.toml 依赖重分类 + uv.lock 重生成

**文件：** 修改 `pyproject.toml` + `uv.lock`（无测试文件——见测试策略）

**实现规格：**
- `pyproject.toml` `[project].dependencies` 数组内增加一行 `"chromadb>=0.5"`
  （位置跟随现有松散顺序即可，可读为先；不强制字母序——现有列表本身
  非严格字母序，如 `transformers` 在末尾）
- `pyproject.toml` `[project.optional-dependencies].chroma` 由
  ```toml
  chroma = [
      "chromadb>=0.5",
  ]
  ```
  改为
  ```toml
  chroma = []
  ```
  （空数组，保留 key 作向后兼容占位）
- `[all]` 的 `all = ["fugue[server,bge,chroma,pdf]"]` **保持不变**
- 执行 `uv lock` 重生成 `uv.lock`（必须用 uv；禁止手改锁文件）
- 不改任何 `.py` 文件、不改 `[server]`/`[bge]`/`[pdf]`/`[dev]` extra

**测试规格（验证命令，非新增测试）：**
- `uv lock --check` 退出 0（锁与 pyproject 一致）
- `uv.lock` 中 fugue 的 `requires-dist` 含 `chromadb`（作为主依赖项，
  非仅 `extra == 'chroma'` 标注）
- 全量套件 `.venv/bin/pytest -q -p no:cacheprovider`（非 e2e）零回归
- `.venv/bin/mypy src` 零错误
- 隔离核验：`python -m build` 出 wheel → 新建临时 venv
  `pip install <wheel>`（无 extra）→ `python -c "import chromadb, fugue"` 成功；
  再 `pip install '<wheel>[chroma]'` 与 `pip install '<wheel>[all]'`
  均不报 "no extra named ..."（两者都必测，对齐验收标准）

**验证标准：** 上述全部命令通过；pyproject 三处改动精确（增主依赖、
chroma 空、all 不变）；uv.lock 由 uv 机械重生成无手改痕迹

**审查要求：** code-quality-reviewer-prompt（无 .py 改动，不派 python-reviewer）

---

### 任务 2：README 安装说明修正

**文件：** 修改 `README.md`（仅文档）

**实现规格：**
- 定位安装章节（约第 13-22 行）`pip install "fugue[chroma]"   # ChromaDB 向量存储`
- 修正为准确反映现状：chromadb 现为**默认安装**（随 `pip install fugue` 自带），
  `[chroma]` extra 保留仅为向后兼容（no-op）。具体措辞由 implementer 拟，
  须满足：① 不再暗示需 `[chroma]` 才装 chromadb ② 说明裸装即含向量存储
  ③ 提及 `[chroma]` 仍可用但为空（向后兼容），不误导
- 不改 README 其它章节；不动 `[server]`/`[bge]`/`[pdf]` 行（它们仍是真可选）

**测试规格（验证，非新增测试）：**
- 人工/grep 核验：README 安装段不再出现"`[chroma]` 才装 chromadb"语义；
  `grep -n chroma README.md` 结果与新措辞一致
- 不破坏 Markdown 结构（标题/代码块完整）

**验证标准：** README 安装说明与任务 1 后的实际行为一致、不误导；
其余章节零改动

**审查要求：** code-quality-reviewer-prompt（文档改动）

## 风险与缓解

- **风险**：`uv lock` 可能顺带升级其它包版本，引入非预期变更
  - 缓解：任务 1 后 `git diff uv.lock` 审查——预期仅 fugue 自身
    requires-dist 中 chromadb 的归类变化（从 `extra=='chroma'` 标注转主依赖），
    不应有无关包版本跳动；若有，报告 DONE_WITH_CONCERNS 并附 diff
- **风险**：空 `chroma = []` extra 在某些旧 pip/setuptools 上行为差异
  - 缓解：隔离核验显式测 `pip install '<wheel>[chroma]'` 不报错；
    本项目用 uv/现代 pip，空 extra 是合法 PEP 621 写法
- **风险**：误改 `[all]` 导致 `fugue[all]` 解析失败
  - 缓解：规格明确 `[all]` 保持不变；隔离核验**必测**
    `pip install '<wheel>[all]'` 不报错（已在测试策略与 Task 1 验证中固化）

## 验收标准

- [ ] `pyproject.toml` `[project].dependencies` 含 `chromadb>=0.5`
- [ ] `[project.optional-dependencies].chroma == []`，`[all]` 仍 `fugue[server,bge,chroma,pdf]`
- [ ] `uv lock --check` 一致；uv.lock 反映 chromadb 为 fugue 主依赖
- [ ] 隔离裸装（无 extra）后 `import chromadb, fugue` 成功
- [ ] `pip install 'fugue[chroma]'` / `'fugue[all]'` 不报错
- [ ] 既有全量测试套件零回归（非 e2e）；mypy src 零错误
- [ ] README 安装说明不再误导，其余章节零改动
