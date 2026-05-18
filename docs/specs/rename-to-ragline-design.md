---
feature: rename-to-ragline
spec: docs/specs/rename-to-ragline-design.md
routing: Development Workflow
---

# Rename to Ragline

## Product Definition

**一句话描述**：将本库从 `fugue` 全量重命名为 `ragline`，解除 PyPI 包名冲突，实现 import 名、公开标识符、CLI、entry_points 契约、环境变量、文档品牌标识的完全统一。

**目标用户**：本库的未来使用者与插件开发者。

**产品形态**：Python 库（PyPI 包）。

**核心场景**：
1. 用户执行 `pip install ragline`，随后 `import ragline` 成功，无任何残留 `fugue` 标识符暴露在公开 API 中。
2. 第三方插件开发者通过 `entry_points(group="ragline.handlers")` 注册处理器，品牌一致、无歧义。

## Feature List

| 功能 | 优先级 | 描述 | 依赖 |
|------|--------|------|------|
| 包目录重命名 + 内部 import 路径替换 | P0 | `src/fugue/` → `src/ragline/`；src 内所有 `from fugue.X import Y` 更新为 `ragline` 路径 | 无 |
| 公开类标识符重命名（8 个 `Fugue*` → `Ragline*`） | P0 | FugueConfig / FugueConfigError / FugueConfigSchema / FugueEmbeddingError / FugueError / FugueLLMError / FugueRegistryError / FugueRetrieverError；src + tests 同步替换 | 切片 1 完成 |
| pyproject.toml 更新 + uv lock 重生成 | P0 | `name`、`[project.scripts]` CLI 命令名、`packages`、`--cov` 参数、`[all]` 自引用全部更新为 `ragline`；执行 `uv lock` 重新生成 | 切片 1–2 完成 |
| entry_points 组重命名 | P0 | `fugue.handlers` → `ragline.handlers`；registry 加载串同步更新 | 切片 1–3 完成 |
| 环境变量重命名 | P0 | e2e conftest 中 5 个 `FUGUE_E2E_*` → `RAGLINE_E2E_*` | 切片 1–4 完成 |
| 文档更新 | P0 | README（15 处）+ 3 个 spec 正文 + `docs/specs/fugue-design.md` 重命名为 `ragline-design.md` 并更新 `feature:` 锚点；区分「叙述历史包名」与「指代本库」两类语境 | 切片 1–5 完成 |
| GitHub 仓库改名 | P1 | 手动收尾项，在 GitHub 端执行，不在代码 diff 内 | 代码切片全部完成 |

## MVP Scope

P0 功能全量实现：包目录、类标识符、pyproject + uv.lock、entry_points、环境变量、文档六个切片均完成后方为 MVP 交付。

## Competitive Analysis

本任务为内部重命名，不涉及竞品对比，此节不适用。

## Technical Design

### Selected Approach

**方案 A：语义切片有序重命名 + 测试兜底**。按「可独立验证的原子语义单元」将工作拆成 6 个有序切片，每片完成后现有测试套件（248 非 e2e 测试）必须保持全绿，并以 `grep -rn 'fugue\|Fugue\|FUGUE'`（排除有意保留的历史叙述）收敛作为完成判据。

### Alternatives Considered

#### 方案 B — 一次性全局 sed 批量替换

- **描述**：用单条 sed/awk 命令对全仓所有文本文件做全局 `fugue → ragline` 替换，一次提交完成。
- **优点**：操作步骤少，速度快。
- **缺点**：无法按语义单元分类验证；误伤难定位（如历史叙述语境中「PyPI 包名 fugue 已被占用」会被错误改写）；一旦出错影响范围不可控，不符合逐切片门控要求。
- **影响范围**：全仓文本文件。

### Architecture

**六切片有序执行**，每切片为独立可验证单元：

1. **包目录 + 内部 import**：`git mv src/fugue src/ragline`，替换 src 内 188 处 `from fugue.X import Y` → `from ragline.X import Y`，确保 `import ragline` 可加载。
2. **公开类标识符**：8 个 `Fugue*` → `Ragline*`，src（~89 处）与 tests（~89 处）同步替换，采用「先长后短」顺序避免截断（如先替 `FugueConfigError` 再替 `FugueConfig`）。
3. **pyproject.toml + uv lock**：更新 `name = "ragline"`、`[project.scripts] ragline = "ragline.server.cli:main"`、`packages = ["src/ragline"]`、`--cov=ragline`、`[all]` 自引用 `ragline[...]`；执行 `uv lock` 重新生成 `uv.lock`。
4. **entry_points 组**：`src/ragline/registry/__init__.py` 第 67 行 `entry_points(group="fugue.handlers")` → `entry_points(group="ragline.handlers")`。
5. **环境变量**：`tests/e2e/conftest.py` 中 5 个 `FUGUE_E2E_*` → `RAGLINE_E2E_*`，conftest 文档串同步更新。
6. **文档**：README（15 处）+ `docs/specs/lazy-optional-imports-design.md`、`docs/specs/chromadb-core-dependency-design.md`、`docs/specs/fugue-design.md` 正文中「指代本库」的 `fugue` 替换为 `ragline`；「叙述 PyPI 包名冲突」语境保留语义，改写为「原名 fugue」等表述；`docs/specs/fugue-design.md` 文件重命名为 `ragline-design.md` 并更新其 `feature:` 锚点。

### Data Model

本任务不涉及数据模型变更。

### API Contract

**entry_points 组契约变更（breaking for third-party plugins）**：`fugue.handlers` → `ragline.handlers`。pre-release 阶段无外部插件依赖此契约，变更安全。

其余公开 Python API（类名、方法签名）：8 个 `Fugue*` 类标识符机械替换为 `Ragline*`，无签名或行为变更。

### Error Handling

- **过度替换风险**：`fugue` 是高辨识度 token，子串误伤风险低；`Fugue*` 长标识符用词边界感知替换；先长后短顺序执行，避免 `FugueConfigError` 被 `Fugue → Ragline` 提前截断为错误形式。
- **文档语义失真风险**：docs 切片区分「指代本库」与「叙述历史包名」两类语境，前者机械替换，后者人工改写为「原名 fugue」等表述，不机械替换。
- **GitHub 仓库改名**：不在代码 diff 内，作为收尾手动步骤记录，不阻塞代码切片实现与合并。

## Design Constraints

- 不改任何业务逻辑或行为；不动固定 7 节点拓扑；不借机重构。
- 仅机械标识符替换，可通过现有测试套件零回归验证正确性。

## Technical Constraints & Risks

- 替换范围大（src ~277 处 + tests ~327 处），需严格按切片执行，避免局部遗漏导致 import 错误。
- `uv.lock` 必须在 pyproject 更新后重新生成，否则 `uv lock --check` 失败。
- e2e 测试依赖外部 API key，环境变量改名后须确保 CI/本地 e2e 参数化配置同步更新，不在每切片门控内但需在最终交付前验证。
- GitHub 仓库改名为手动操作，存在窗口期内旧名失效的风险，需在代码发布前协调执行。

## Success Metrics

- `src/ragline/` 存在，无 `src/fugue/`
- 全仓 `grep -rn 'fugue\|Fugue\|FUGUE'` 仅余有意保留的历史叙述（「原名 fugue」类）
- 8 个 `Ragline*` 类标识符全替换，src + tests 一致
- pyproject `name = "ragline"`、CLI 命令 `ragline`、`packages = ["src/ragline"]`、`--cov=ragline`、`[all]` 自引用为 `ragline[...]`
- entry_points 组为 `ragline.handlers`
- `RAGLINE_E2E_*` 环境变量；conftest 文档串同步
- `uv lock --check` 一致；`uv.lock` 中本包名为 ragline
- 248 非 e2e 测试零回归；mypy src 零错误
- 隔离环境 `pip install <wheel>` 后 `import ragline` 成功、CLI `ragline --help` 可用
- README + 3 spec 文档无误导；`docs/specs/ragline-design.md` 存在且 `feature:` 锚点已更新
- GitHub 仓库改名（手动收尾项，文档记录）

## Routing Decision

后续工作流：Development Workflow
理由：纯后端标识符重命名 + 文档，无 UI，无业务逻辑变更。
