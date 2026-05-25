---
feature: expose-version
spec: docs/specs/expose-version-design.md
routing: Development Workflow
---

# Expose Version

## Product Definition

**一句话描述**：为 ragline 暴露标准的 `__version__` 公开属性，并改善 README 顶部布局，让 PyPI 首发用户能从仓库首屏理解如何安装与信任该项目。

**目标用户**：
- 通过 `pip install ragline` 后想检查版本的开发者（`import ragline; print(ragline.__version__)`）
- 浏览 GitHub 首页评估技术选型的访客（看 badge 即时了解项目状态）
- 在 issue 报告中需要附带版本号的用户

**产品形态**：Python 公开 API 微扩展（新增 1 个模块级属性）+ README 文档调整。无运行期行为变化。

**核心场景**：
1. `import ragline; print(ragline.__version__)` 输出 `"0.1.0"`（PyPI 发布后随版本号同步）
2. 用户访问 README 顶部立刻看到 4 个 shields.io badges（PyPI version / License / Python versions / CI status）
3. README "Install" 段落以 `pip install ragline` 为首选，明确告知用户从 PyPI 安装是推荐方式
4. issue 报告者能轻松引用版本号（`ragline.__version__`）

**明确不做**：
- 不打 git tag `v0.1.0`（属于 P1 发布动作）
- 不发布到 PyPI（属于 P1）
- 不加 publish CI workflow（属于 P2）
- 不加 coverage badge（需 Codecov 单独配置，超本 PR 范围）
- 不改其他文档（CHANGELOG / spec 等）
- 不引入新依赖

## Feature List

| 功能 | 优先级 | 描述 | 依赖 |
|------|--------|------|------|
| 暴露 `ragline.__version__` | P0 | 用户预期惯例；issue 报告需要 | 无 |
| README 顶部 4 个 badges | P0 | PyPI 页面之外，GitHub 是用户第一接触点 | 无 |
| README "Install" 段落改写以 PyPI 为首选 | P0 | 当前 README 是源码安装风格，发布前必须改 | 无 |

## MVP Scope

所有 P0 功能合并入同一 PR：

- `ragline.__version__` 暴露（`importlib.metadata.version("ragline")` 动态查询）
- `__version__` 加入 `__all__`
- README 顶部插入 4 个 shields.io badges
- README Install 段落确认 PyPI 安装为首选

两者都属于"PyPI 首发外观打磨"，scope 关联紧密，单次 review 高效。

## Competitive Analysis

`__version__` 是 PEP 8 推荐的模块级公开属性；shields.io badges 是 Python 库 README 的行业惯例。跳过竞品对比。

参考资料：
- `importlib.metadata.version()` — Python 3.8+ stdlib
- shields.io — https://shields.io
- PyPA 公认的 Python 包 `__version__` 暴露惯例

## Technical Design

### Selected Approach

使用 `importlib.metadata.version("ragline")` 动态查询版本号，以 `pyproject.toml` 为单一真相源，通过测试验证对齐。README 顶部插入 4 个 badges，Install 段落确认已以 PyPI 安装为首选。

### Alternatives Considered

#### 方案 A — `importlib.metadata.version()`（已选）

- **描述**：在 `src/ragline/__init__.py` 顶部调用 `importlib.metadata.version("ragline")` 赋值给 `__version__`
- **优点**：单一真相源（`pyproject.toml`），升级时只改一处；Python 3.12+ stdlib 完全支持；包未安装时 fail-loud，便于早期发现安装问题
- **缺点**：包未安装（仅源码 clone）时抛出 `PackageNotFoundError`
- **影响范围**：`src/ragline/__init__.py`

#### 方案 B — 硬编码字符串

- **描述**：直接在 `__init__.py` 写 `__version__ = "0.1.0"`
- **优点**：简单直接，无依赖
- **缺点**：两处维护（`__init__.py` + `pyproject.toml`），容易漂移，release 流程更复杂
- **影响范围**：`src/ragline/__init__.py`，每次发布都需手动同步

#### 方案 C — `__about__.py` 间接层

- **描述**：新增 `src/ragline/__about__.py` 存储版本，`__init__.py` 从中导入
- **优点**：关注点分离，适合大型项目
- **缺点**：当前规模不需要这层间接，增加文件数量
- **影响范围**：新增文件 + `src/ragline/__init__.py`

### Architecture

单点变更，无架构层变化：

```
src/ragline/__init__.py
  └── from importlib.metadata import version as _pkg_version
      __version__ = _pkg_version("ragline")
      __all__ = [..., "__version__"]

tests/unit/test_packaging.py
  └── 场景 29：__version__ 存在且匹配 SemVer 正则
      场景 30：__version__ 与 pyproject.toml version 对齐

README.md
  └── 顶部插入 4 个 badges
```

### Data Model

无数据模型变更。

### API Contract

新增模块级公开属性：

```python
ragline.__version__: str  # SemVer 格式，如 "0.1.0"
```

Badge 地址（shields.io）：

```markdown
[![PyPI version](https://img.shields.io/pypi/v/ragline.svg)](https://pypi.org/project/ragline/)
[![License: MIT](https://img.shields.io/pypi/l/ragline.svg)](https://github.com/dengd1937/ragline/blob/main/LICENSE)
[![Python versions](https://img.shields.io/pypi/pyversions/ragline.svg)](https://pypi.org/project/ragline/)
[![CI status](https://github.com/dengd1937/ragline/actions/workflows/ci.yml/badge.svg)](https://github.com/dengd1937/ragline/actions/workflows/ci.yml)
```

### Error Handling

- 包未以可编辑或正式安装方式安装时，`importlib.metadata.version("ragline")` 抛出 `PackageNotFoundError`
- 这是预期的 fail-loud 行为，无需捕获
- 开发环境需 `uv pip install -e .` 或等价安装确保 metadata 存在

## Design Constraints

- Python 3.12+，`importlib.metadata` stdlib 完全支持
- 不引入任何新运行期依赖
- `__all__` 顺序与现有风格一致（字母序，具体位置由 implementer 按既有风格决定）
- README badges 按 PyPI / License / Python / CI 顺序排列

## Technical Constraints & Risks

- 需确保开发环境以可编辑安装（`pip install -e .`）运行，否则 `PackageNotFoundError` 会在 import 时抛出
- 测试场景 30 读取 `pyproject.toml`，需确保测试从项目根目录运行（pytest 标准行为）

## Success Metrics

- `import ragline; print(ragline.__version__)` 输出 `"0.1.0"`
- `pytest` 全量通过（≥ 290 passed，新增场景 29 + 30）
- coverage ≥ 96%
- `ruff format / check` 通过，`mypy --strict src/ragline` 通过
- README 顶部含 4 个 badges
- `__version__` 在 `__all__` 中

## Routing Decision

后续工作流：**Development Workflow（writing-plans → subagent-driven-development → code-review → finishing）**
理由：微型 Python API 扩展 + 文档调整；单文件单任务；不涉及 UI / 设计。

### 已审定决策

1. D1：方案 A `importlib.metadata.version("ragline")`（已与用户确认）
2. D2：`__all__` 包含 `__version__`（按惯例）
3. D3：4 个 badges（PyPI / License / Python / CI）；不含 coverage（已与用户确认）
4. D4：README Install 段当前已合理，仅需添加 badges
5. D5：测试场景 29 + 30 追加到 `tests/unit/test_packaging.py`
6. D6：`__all__` 字母序由 implementer 按现有风格决定
7. D7：单任务合打（已与用户确认）
