---
feature: pypi-release-prep
spec: docs/specs/pypi-release-prep-design.md
routing: Development Workflow
---

# PyPI Release Prep

## Product Definition

**一句话描述**：让 ragline 具备 PyPI 首次发布（0.1.0）所需的元数据、许可证、依赖纯净度条件。

**目标用户**：
- 即将通过 `pip install ragline` 安装的下游 Python 开发者
- 浏览 PyPI 项目页面评估技术选型的潜在用户
- 阅读 CHANGELOG 决定升级时机的现有用户

**产品形态**：仓库元数据 + 包元数据 + 文档。无运行期行为变化。

**核心场景**：
1. `pip install ragline` 最小安装不再拉 `transformers` / `torch` / `safetensors` 等 BGE 专属依赖（节省约 200-500MB）
2. PyPI 项目页面展示 author / license / Homepage / Repository / Issues / Changelog 链接，trove classifiers 完整
3. 下游可在 MIT 许可证范围内合法使用、修改、再发布
4. 用户能从 `CHANGELOG.md` 看到 0.1.0 版本包含的功能清单

**明确不做**：
- 不发布到 PyPI（属于后续 P1-P2 工作）
- 不打 git tag `v0.1.0`（属于发布动作本身）
- 不加 GitHub Actions publish workflow（属于 P2 自动化）
- 不改任何运行期行为 / API 契约 / 公开接口

## Feature List

| 功能 | 优先级 | 描述 | 依赖 |
|------|--------|------|------|
| 加 MIT LICENSE 文件 | P0 | 法律阻塞：无 license = "all rights reserved"，下游不能合法使用 | 无 |
| 补 pyproject `[project]` 元数据 | P0 | PyPI 页面会很裸；trove classifiers 影响搜索发现 | LICENSE 决定 `license` 字段值 |
| transformers 移到 `[bge]` extra | P0 | 最小安装白白拉 200-500MB 传递依赖 | 无 |
| 起 CHANGELOG.md | P1 | 与版本号 bump 同步做更自然；用户升级有依据 | 元数据决定 version |

## MVP Scope

P0 三项（LICENSE + pyproject 元数据 + transformers 迁移）合并入同一 PR，CHANGELOG.md 同 PR 起稿，整体作为 PyPI 首次发布的"硬阻塞"清零。

## Competitive Analysis

跳过。PyPI 包元数据（PEP 621 / trove classifiers）与 Keep-a-Changelog 都是行业既定标准模式，无需竞品调研。

参考资料：
- PEP 621 — Storing project metadata in pyproject.toml
- PyPI trove classifiers — https://pypi.org/classifiers/
- Keep a Changelog 1.1.0 — https://keepachangelog.com/

## Technical Design

### Selected Approach

纯配置 + 文档变更，不涉及任何运行期代码修改。四个独立任务逐一实施，每个任务涉及文件不超过 2 个，符合 SDD 任务粒度要求。

### Alternatives Considered

#### 方案 A — 一次性全量修改

- **描述**：在单次提交中同时修改 pyproject.toml、新建 LICENSE、新建 CHANGELOG.md，不拆任务
- **优点**：操作简单，步骤少
- **缺点**：出错时难以定位；无法逐任务走 TDD → 实现 → 审查闭环
- **影响范围**：所有 4 个文件

#### 方案 B — 拆为 4 个独立任务（选定方案）

- **描述**：T1（pyproject 元数据）/ T2（LICENSE）/ T3（transformers 迁移 + uv lock）/ T4（CHANGELOG.md）分别走 TDD 验证
- **优点**：每任务有明确验收脚本，风险隔离，符合 SDD 流程
- **缺点**：任务数略多
- **影响范围**：同上，但按任务隔离

### Architecture

无架构变更。本 feature 仅涉及：

- `pyproject.toml`：`[project]` 表补充 authors / license / license-files / urls / keywords / classifiers；`[project.dependencies]` 移除 transformers；`[project.optional-dependencies]` 的 `bge` extra 中添加 transformers
- `LICENSE`：新建，MIT 全文
- `CHANGELOG.md`：新建，Keep-a-Changelog 1.1.0 格式，记录 0.1.0 功能清单
- `uv.lock`：由 `uv lock` 重新生成，不手动编辑

### Data Model

无数据模型变更。

### API Contract

无 API 变更。

### Error Handling

无运行期行为变化，不涉及错误处理。

---

### 已审定决策

| 编号 | 决策 |
|------|------|
| D1 | Development Status = `4 - Beta`（260 tests / 96.84% coverage，接口可能调但主体可用） |
| D2 | Repository URL = `https://github.com/dengd1937/ragline` |
| D3 | authors = `{ name = "dengdi", email = "dengdi1803@gmail.com" }` |
| D4 | license = `"MIT"`，license-files = `["LICENSE"]`（PEP 639 SPDX 表达式，hatchling 1.27+） |
| D5 | keywords = `["rag", "llm", "langgraph", "retrieval", "embedding", "chromadb", "bm25"]` |
| D6 | classifiers 见下方完整列表 |
| D7 | LICENSE 版权行：`Copyright (c) 2026 dengdi` |
| D8 | transformers 从 core deps 移到 `[bge]` extra，移除后 `uv lock` 重新生成 |
| D9 | CHANGELOG.md 与 P0 元数据合并在同一个 PR |
| D10 | chromadb 保持在 core deps，不挪到 extras |

### pyproject.toml — 元数据变更（T1 + T3 合并视图）

```toml
[project]
authors = [
    { name = "dengdi", email = "dengdi1803@gmail.com" }
]
license = "MIT"
license-files = ["LICENSE"]
keywords = ["rag", "llm", "langgraph", "retrieval", "embedding", "chromadb", "bm25"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Typing :: Typed",
]

[project.urls]
Homepage = "https://github.com/dengd1937/ragline"
Repository = "https://github.com/dengd1937/ragline"
Issues = "https://github.com/dengd1937/ragline/issues"
Changelog = "https://github.com/dengd1937/ragline/blob/main/CHANGELOG.md"

# transformers 从 dependencies 移除，加入 [bge] extra
[project.optional-dependencies]
bge = [
    "FlagEmbedding>=1.2",
    "transformers>=4.44.2,<5",
]
```

### LICENSE 文件内容（T2）

```
MIT License

Copyright (c) 2026 dengdi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### CHANGELOG.md 内容（T4，Keep-a-Changelog 1.1.0 格式）

整体结构：

```
# Changelog
（标准导言段，引用 Keep-a-Changelog 1.1.0 + Semantic Versioning）

## [0.1.0] - 2026-05-24

### Added
（按下方七个子模块组织，每个子模块用三级标题 #### 分隔）

### Notes
（说明这是首个公开版本，0.x 内 API 可能调整）
```

`### Added` 必须包含以下七个子模块（每个子模块用 `####` 标题，下面用 bullet 列出项目）：

1. **顶层 API**：`RAG` 类（`ingest()` / `query()`）；`RaglineConfig` + `GraphConfig` / `IngestConfig` / `ProviderConfig`；数据类型 `Document` / `Chunk` / `ParsedDocument` / `IngestResult` / `QueryResult` / `TransformResult`；6 个特定错误类（Config / Embedding / LLM / Registry / Retriever + 基类）
2. **检索引擎（基于 LangGraph）**：6 节点图 `query_transform → retrieve → grade → (post_process | prepare_fallback ↻) → generate`；`fallback_chain` + `max_retries`；`route_strategy: all | intent` 路由
3. **内置 Handlers（7 类）**：transforms（rewrite + pipeline）/ retrievers（vector + bm25）/ processors（rrf + rerank）/ graders（score / llm / hybrid + 归一化）/ generators（basic + citation）/ parsers（markdown_text + pdf）/ chunkers（recursive）
4. **Providers**：LLM（OpenAI 兼容）/ Embedding / VectorStore（ChromaDB 持久化）/ BM25（rank-bm25）/ Reranker（BGE）
5. **HTTP Server（`[server]` extra）**：FastAPI + uvicorn；`POST /query` / `POST /ingest` / `GET /health`；CLI 入口 `ragline`
6. **对外测试支持（`ragline.testing`）**：`FakeLLM` / `FakeEmbedding` 零网络替身；`isolated_registries()` / `mock_rag_providers()` 上下文管理器；配套 `examples/quickstart/`
7. **工程基线**：PEP 561 `py.typed`；import-linter 契约（engine 只许依赖 langgraph / testing 不依赖 engine）；260 tests + 96.84% coverage；可选 extras（`[server]` / `[bge]` / `[pdf]` / `[all]`）；可选依赖懒加载

`### Notes` 内容：

> 这是 ragline 的首个公开版本。在 0.x 系列内 API 可能调整；进入 1.0 后将遵循语义化版本承诺。

implementer 在 RED→GREEN 中自行决定每条 bullet 的具体措辞与排版细节，但必须覆盖以上 7 个子模块的全部要点。

### 任务拆分

| Task | 内容 | 涉及文件 | TDD 验证手段 |
|------|------|----------|-------------|
| T1 | 补 pyproject `[project]` 元数据（authors / license / license-files / urls / keywords / classifiers） | `pyproject.toml` | `uvx twine check dist/*` after `hatch build`；`python -c "from importlib.metadata import metadata; m=metadata('ragline'); assert m['License-Expression']=='MIT' or m['License']"` |
| T2 | 新建 MIT `LICENSE` 文件 | `LICENSE` | 文件存在 + 包含"MIT License"标题 + 版权行 + 完整条款（与 SPDX MIT 一致） |
| T3 | transformers 从 core 移到 `[bge]` extra；`uv lock` 重新生成 | `pyproject.toml`、`uv.lock` | `python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb')); assert not any('transformers' in dep for dep in d['project']['dependencies']); assert any('transformers' in dep for dep in d['project']['optional-dependencies']['bge'])"；pytest`（260 tests 全过） |
| T4 | 新建 `CHANGELOG.md` 记录 0.1.0 | `CHANGELOG.md` | 文件存在；第一行是 `# Changelog`；包含 `## [0.1.0] - 2026-05-24` 节；`### Added` 段下含 7 个 `####` 子模块（顶层 API / 检索引擎 / 内置 Handlers / Providers / HTTP Server / 测试支持 / 工程基线）；包含 `### Notes` 段说明 0.x API 可能调整 |

### 整体验收

- `pytest`：260 passed（或更多）
- Coverage ≥ 96%（不降）
- `hatch build`：生成 `dist/ragline-0.1.0.tar.gz` + `dist/ragline-0.1.0-py3-none-any.whl`
- `uvx twine check dist/*`：PASSED（元数据合规）
- 手动验证：在干净 `.venv` 里 `pip install dist/ragline-0.1.0-py3-none-any.whl`，确认 `transformers` 不在依赖图里；再装 `[bge]` 确认 BGE 可用
- `examples/quickstart/consumer_minimal.py` 仍能跑通（subprocess test 已覆盖）

### Environment Prerequisites

| 工具 | 验证 | 失败修复建议 |
|------|------|-------------|
| `uv` | `uv --version` 输出 ≥ 0.4 | 重装 uv |
| `hatch`（用于 build） | `uvx hatch --version` 能跑 | `uv tool install hatch` |
| `twine`（用于元数据检查） | `uvx twine --version` 能跑 | 无需预装，`uvx twine check` 直接用 |

## Design Constraints

- 所有变更仅限配置文件与文档，不得修改任何 Python 源码
- pyproject.toml 使用 PEP 639 SPDX license 表达式（要求 hatchling 1.27+）
- CHANGELOG.md 严格遵循 Keep-a-Changelog 1.1.0 格式
- uv.lock 必须由 `uv lock` 自动生成，禁止手动编辑

## Technical Constraints & Risks

- hatchling 版本需 ≥ 1.27 才支持 `license = "MIT"` SPDX 表达式（PEP 639）；当前 `[build-system].requires = ["hatchling"]` 未 pin 版本，pip/uv 会拉最新版（≥ 1.27）；如出错可显式 pin 为 `hatchling>=1.27`
- transformers 迁移后需确认所有 260 个测试仍通过，排查是否有测试直接导入 transformers 而非通过 `[bge]` 路径

## Success Metrics

- `uvx twine check dist/*` 输出：PASSED
- 最小安装依赖图不包含 `transformers`：验证通过
- PyPI 页面（模拟）：author / license / URLs / classifiers 全部展示正确

## Routing Decision

后续工作流：Development Workflow（writing-plans → subagent-driven-development → code-review → finishing-a-development-branch）
理由：纯包元数据 + 文档变更，无运行期行为变化；4 任务每个不超过 2 个文件，符合 SDD 任务粒度。

## Implementation Deviations

### 2026-05-25 — T1 预防性 pin `hatchling>=1.27`

**偏差章节**：Technical Design / Constraints

**原方案**：spec Technical Constraints & Risks 段写"`[build-system].requires = ["hatchling"]` 未 pin 版本，pip/uv 会拉最新版（≥ 1.27）；**如出错可**显式 pin 为 `hatchling>=1.27`"——条件触发。

**实际实现**：T1 实施时**预防性**直接 pin 为 `["hatchling>=1.27"]`（pyproject.toml line 2），未等待出错。

**原因**：风险 1 的缓解路径明确，预先 pin 不增加成本但消除潜在的不可重复构建风险；与 plan 风险 1 段精神一致。

### 2026-05-25 — T1 同步上调 `.github/workflows/ci.yml` 的 `--cov-fail-under`

**偏差章节**：Technical Design / Task Decomposition

**原方案**：spec 4.10 任务表 T1 列出文件仅 `pyproject.toml`，未列 `.github/workflows/ci.yml`。

**实际实现**：T1 同时修改 ci.yml line 48，把显式 CLI `--cov-fail-under=90` 同步上调为 96。

**原因**：plan dual-review pass 2 发现 ci.yml 的 CLI 参数会覆盖 pyproject `addopts`，若不同步则 CI 仍按 90 门控，本任务声明的"≥ 96 由 pytest 硬性保证"在 CI 上失效。spec 是事前文档，未预见到 CLI/addopts 覆盖关系；plan dual-review 补入这一点，作为 spec 的事实更正。

### 2026-05-25 — T4 CHANGELOG 内容在 pass 2 修复 6 处事实错误

**偏差章节**：Technical Design / Content

**原方案**：spec 4.9 节列出 CHANGELOG 必须覆盖的 7 个子模块要点（关键词级宽松约束）。

**实际实现**：T4 implementer 首次实现时在 7 个子模块下生成了与实际源码不符的虚构内容：

- 检索引擎节点列表写为 `query_transform → retrieve → grade → generate → parse → route`（`parse` / `route` 节点不存在；遗漏 `prepare_fallback` / `post_process`）
- `route_strategy` 枚举写为 `dense-only / sparse-only / hybrid`（实际仅 `all` / `intent`）
- 错误类层次写为 `ConfigError / ProviderError / RetrievalError`（实际类名带 `Ragline` 前缀，且有 5 个子类）
- `chunkers` 描述为"固定窗口 / 语义分块"（实际仅 `recursive`）
- `mock_rag_providers` 描述为"pytest fixture"（实际是 `@contextmanager` 装饰的上下文管理器）
- HTTP Server 列了"支持 CORS 中间件与请求/响应结构化日志"（源码 `src/ragline/server/` 没有这些）

dual-review（code-quality + spec）同时 BLOCK，pass 2 修复全部 6 处，使 CHANGELOG 内容与源码字符级对齐。

**原因**：spec 4.9 节的关键词级宽松约束（场景 27 使用 `any(kw in block_text for kw in keywords)` 匹配）允许测试通过即使内容虚构。本次问题揭示了"关键词宽松匹配"作为唯一验证机制的不足；后续 CHANGELOG 维护需要人工或额外校验工具来保证内容真实性。spec 验证策略保留，但需在 reviewer 阶段补充事实校验。
