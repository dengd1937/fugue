# 实现计划：Fugue — 配置驱动的通用 Python RAG 库

## 执行方式

本计划通过 `/rune:subagent-driven-development` skill 执行。以下任务描述是 skill 的输入规格，不是直接执行指令。每个任务由独立 implementer subagent 完成 TDD（RED→GREEN→IMPROVE）→ Quality Gate → Code Review 三道门控。

## 概述

实现 `fuge_plan.md` + `docs/specs/fugue-design.md` 定义的 Fugue MVP（0.1.0）：基于 LangGraph 的 7 节点固定图引擎，通过 Registry + GraphConfig 实现"配置即行为"，支持 OpenAI 兼容 LLM/Embedding、Chroma、内置 BM25、BGE-Reranker，对外暴露 SDK + `fugue serve` REST 薄壳。

## 需求

引用自 `docs/specs/fugue-design.md` 第 3 章 MVP 范围，全部 P0 项均覆盖：

- 7 节点固定拓扑（query_transform/retrieve/grade/prepare_fallback/post_process/generate）+ Fallback 拓扑（默认禁用）
- 7 个 Registry（transform/retriever/processor/grader/generator/parser/chunker）+ entry_points 自动发现
- FugueConfig 三层（GraphConfig/IngestConfig/ProviderConfig）+ YAML loader + Pydantic v2 校验 + `${env_var}` 展开
- Providers：OpenAI 兼容 LLM/Embedding（含 Semaphore 并发控制）+ Chroma + 内置 BM25（启动重建）+ BGE-Reranker（FlagEmbedding）
- Handlers：transforms（rewrite/hyde/step_back + 嵌套并行+串联执行器）+ retrievers（vector/bm25）+ processors（rrf/rerank）+ graders（score）+ generators（basic/citation）+ parsers（markdown/text/pdf）+ chunkers（recursive）
- API：`RAG.from_yaml() / ingest() / query() / close() / 上下文管理器`，含启动期 fail-fast 校验
- Server：FastAPI 薄壳 + `fugue serve` CLI
- 物理约束：`engine/` 是唯一允许 import `langgraph` 的模块（import-linter CI 强制）；核心模块覆盖率 ≥ 90%
- 测试：unit + integration + e2e 三层，syrupy 做 prompt 快照

## 架构变更

新建项目（git 中之前实现已删除）。文件结构遵循 spec 4.2 章节，每层 ≤ 8 文件、单文件 ≤ 800 行。

- 项目根：`pyproject.toml` / `.python-version` / `.gitignore` / `README.md` / `.pre-commit-config.yaml` / `.importlinter` / `.github/workflows/`
- 源码：`src/fugue/` 下 8 顶层（`__init__.py` / `api/` / `config.py` / `registry.py` / `engine/` / `handlers/` / `providers/` / `server/`）
- 测试：`tests/{unit,integration,e2e,fixtures}/`
- 文档：保留 `docs/specs/` + `docs/plans/`

## 环境前置（Environment Prerequisites）

- **Python 3.12+**
  - 验证：`python --version | grep -E "3\.(1[2-9]|[2-9][0-9])"`
  - 修复：`uv python install 3.12 && uv python pin 3.12`

- **uv**（包管理）
  - 验证：`uv --version`
  - 修复：`curl -LsSf https://astral.sh/uv/install.sh | sh`

- **测试运行器 pytest**（项目依赖）
  - 验证：`uv run pytest --version`
  - 修复：`uv sync --extra dev`（pyproject 已声明 dev extra）

- **Chroma**（integration test 用，本地无服务）
  - 验证：`uv run python -c "import chromadb; chromadb.PersistentClient(path='/tmp/_fugue_check')"`
  - 修复：`uv sync --extra chroma`

- **BGE Reranker 模型**（integration/e2e 用，首次下载）
  - 验证：`uv run python -c "from FlagEmbedding import FlagReranker; FlagReranker('BAAI/bge-reranker-v2-m3')"`
  - 修复：`uv sync --extra bge`（首次运行自动下载约 600MB 模型权重）

- **OPENAI_API_KEY**（仅 e2e 测试需要，可跳过）
  - 验证：`[ -n "$OPENAI_API_KEY" ]`
  - 修复：`export OPENAI_API_KEY=sk-...` 或 `pytest -m 'not e2e'` 跳过

---

## 实现步骤

### 阶段 1：项目骨架与基础设施

#### 任务 1：项目脚手架与依赖管理

**文件：**
- 新建 `pyproject.toml`
- 新建 `.python-version`（内容 `3.12`）
- 新建 `.gitignore`
- 新建 `README.md`（雏形，含"0.x 版本可能 breaking"声明）
- 新建 `src/fugue/__init__.py`（空，后续 re-export）
- 新建 `tests/__init__.py`

**pyproject.toml 必含字段：**
- `[project]` name=`fugue`, version=`0.1.0`, requires-python=`>=3.12`
- `dependencies`：核心运行时仅 `langgraph>=0.3,<0.4`, `langchain-core>=0.3`, `openai>=1.40`, `pydantic>=2.6`, `pyyaml>=6.0`
- `[project.optional-dependencies]`：
  - `dev` = pytest, pytest-cov, pytest-mock, syrupy, ruff, mypy, import-linter, pre-commit
  - `server` = fastapi, uvicorn
  - `bge` = FlagEmbedding
  - `chroma` = chromadb
  - `pdf` = pypdf
  - `all` = server + bge + chroma + pdf
- `[project.scripts]` `fugue = "fugue.server.cli:main"`
- `[tool.uv]`、`[tool.ruff]`（line-length 120, target-version py312, select 含 E/F/W/I/N/UP/B/SIM）、`[tool.mypy]`（strict, python_version 3.12）
- `[tool.pytest.ini_options]` `addopts = "-m 'not e2e' --cov=fugue --cov-report=term-missing --cov-fail-under=90"`, `markers = ["e2e: marks tests as end-to-end (requires API keys)"]`

**测试规格：** 不直接测 pyproject.toml；通过后续任务的 `uv sync` 间接验证。

**验证标准：**
- `uv sync --extra dev` 成功执行，无依赖冲突
- `uv run python -c "import fugue"` 成功（即使 `__init__.py` 空）
- `uv run pytest --collect-only` 不报 import 错（即使无测试）

**审查要求：** code-quality-reviewer（Python 依赖规范、版本范围合理性）

**依赖：** 无
**风险：** 低

---

#### 任务 2：CI 与 Quality Gate 配置

**文件：**
- 新建 `.github/workflows/ci.yml`（PR/push 触发：unit + integration + ruff + mypy + import-linter）
- 新建 `.github/workflows/e2e.yml`（nightly + workflow_dispatch：e2e）
- 新建 `.importlinter`（强制 `engine/` 是唯一允许 import `langgraph` / `langchain_core` 的模块）
- 新建 `.pre-commit-config.yaml`（ruff format/lint + mypy + 大文件检查）

**`.importlinter` 必含规则：**
```ini
[importlinter]
root_package = fugue

[importlinter:contract:engine-only-langgraph]
name = engine/ is the only module allowed to import langgraph
type = forbidden
source_modules =
    fugue.api
    fugue.config
    fugue.registry
    fugue.handlers
    fugue.providers
    fugue.server
forbidden_modules =
    langgraph
    langchain_core
```

**ci.yml 必含步骤：**
- matrix: python 3.12, 3.13；os: ubuntu-latest
- `uv sync --extra dev --extra all`
- `uv run ruff check src tests`
- `uv run ruff format --check src tests`
- `uv run mypy src/fugue`
- `uv run lint-imports --config .importlinter`
- `uv run pytest -m 'not e2e' --cov=fugue --cov-fail-under=90`
- 上传 Codecov

**e2e.yml 必含步骤：**
- `schedule: cron '0 3 * * *'`（每日 03:00 UTC）
- `workflow_dispatch`
- `env: OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}`
- `uv run pytest -m e2e`
- 失败仅通知（不阻塞）

**测试规格：** 通过 CI 实际跑通验证；本地 `uv run pre-commit run --all-files` 验证 hooks 工作（任务 1 完成后即可跑）。

**验证标准：**
- `uv run pre-commit run --all-files` 通过（在仅有任务 1 产出的项目上）
- `uv run lint-imports --config .importlinter` 通过（无 langgraph 引入即默认通过）
- CI workflow YAML 经 `actionlint`（可选）或 `yamllint` 语法验证

**审查要求：** code-quality-reviewer（CI 配置完整性、import-linter 规则正确性）

**依赖：** 任务 1
**风险：** 低（仅配置）

---

#### 任务 3：公开类型与异常

**文件：**
- 新建 `src/fugue/api/__init__.py`（re-export 公开类型）
- 新建 `src/fugue/api/types.py`
- 新建 `tests/unit/test_types.py`

**`api/types.py` 必含定义：**

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

class Document(TypedDict):
    doc_id: str
    content: str
    score: float
    source: str
    metadata: dict[str, Any]

@dataclass(frozen=True)
class ParsedDocument:
    source_path: Path
    content: str
    metadata: dict[str, Any]

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    parent_id: str | None
    content: str
    metadata: dict[str, Any]

@dataclass(frozen=True)
class TransformResult:
    query: str
    metadata_filter: dict[str, Any] | None = None

@dataclass(frozen=True)
class QueryResult:
    answer: str
    ranked_documents: list[Document]
    grade_score: float
    grade_decision: Literal["sufficient", "insufficient"]
    rewritten_queries: list[str]
    retrieval_rounds: int

@dataclass(frozen=True)
class IngestResult:
    num_documents: int
    num_chunks: int
    collection_name: str
    duration_seconds: float

# 异常体系
class FugueError(Exception):
    """Fugue 异常基类。"""

class FugueConfigError(FugueError): ...
class FugueRegistryError(FugueError): ...
class FugueLLMError(FugueError): ...
class FugueEmbeddingError(FugueError): ...
class FugueRetrieverError(FugueError): ...
```

**`api/__init__.py` 必须 re-export**：`Document, ParsedDocument, Chunk, TransformResult, QueryResult, IngestResult, FugueError, FugueConfigError, FugueRegistryError, FugueLLMError, FugueEmbeddingError, FugueRetrieverError`。

**`src/fugue/__init__.py`（更新）re-export**：`from fugue.api import *`（或显式列出主公开符号 + `RAG` 等后续添加）。

**测试规格（tests/unit/test_types.py）：**
- `Document` TypedDict 字段齐全（结构性测试，调用 `Document(doc_id=..., ...)` 不报 TypeError）
- `QueryResult` 是 frozen dataclass：实例化后修改字段抛 `FrozenInstanceError`
- 异常继承关系：`FugueLLMError` 是 `FugueError` 子类（`issubclass` 断言）
- 异常可携带 message：`raise FugueRegistryError("kg not found")` 后 `str(e) == "kg not found"`

**验证标准：**
- `uv run pytest tests/unit/test_types.py -v` 全绿
- `uv run mypy src/fugue/api/types.py` 0 错误
- `uv run python -c "from fugue.api.types import Document, FugueError; print(Document.__annotations__)"` 正常

**审查要求：** code-quality-reviewer

**依赖：** 任务 1
**风险：** 低

---

#### 任务 4：Registry 实现

**文件：**
- 新建 `src/fugue/registry.py`
- 新建 `tests/unit/test_registry.py`

**`registry.py` 必含定义：**

```python
from collections.abc import Callable
from typing import Generic, TypeVar
from importlib.metadata import entry_points

from fugue.api.types import FugueRegistryError

H = TypeVar("H", bound=Callable[..., object])

class Registry(Generic[H]):
    def __init__(self, name: str) -> None: ...
    def register(self, name: str, fn: H) -> None: ...
    def unregister(self, name: str) -> None: ...
    def get(self, name: str) -> H: ...
    def has(self, name: str) -> bool: ...
    def names(self) -> list[str]: ...
    def __call__(self, name: str) -> Callable[[H], H]: ...  # 装饰器糖

# 7 个全局单例（类型参数与签名详见任务 6-17 各 handler 协议）
transform_registry: Registry = Registry("transform")
retriever_registry: Registry = Registry("retriever")
processor_registry: Registry = Registry("processor")
grader_registry:    Registry = Registry("grader")
generator_registry: Registry = Registry("generator")
parser_registry:    Registry = Registry("parser")
chunker_registry:   Registry = Registry("chunker")

def discover_plugins() -> None:
    """扫描 entry_points group='fugue.handlers' 触发第三方插件注册。
    每个 entry_point 指向一个 register() 函数；该函数被调用以执行注册副作用。
    """
    for ep in entry_points(group="fugue.handlers"):
        register_fn = ep.load()
        register_fn()
```

**关键行为约定：**
- `register(name, fn)`：名字已存在时**覆盖**（不抛错），warning 日志 `fugue.registry`
- `get(name)` 找不到时抛 `FugueRegistryError("Handler '{name}' not registered in {registry_name}. Available: {sorted_names}")`
- `__call__` 返回的装饰器**返回原函数**（不包装）
- `discover_plugins()` 失败时（如 entry_point 模块导入失败），warning 日志但不抛错（避免单插件挂掉全部）

**测试规格（tests/unit/test_registry.py）：**
- `register + get`：注册函数后 get 返回同一对象
- `__call__ 装饰器糖`：`@reg("foo")` 装饰后 `reg.get("foo")` 返回原函数
- `get 不存在的 handler`：抛 `FugueRegistryError`，消息含 "Available:" 和已注册名字列表
- `unregister`：删除后 `has` 返回 False
- `names`：返回排序后的注册名字列表
- 重复 register：覆盖+发 warning（用 `caplog` 验证）
- `discover_plugins`：mock `entry_points`，验证 register_fn 被调用；其中一个 raise 时其他仍被调用

**验证标准：**
- `uv run pytest tests/unit/test_registry.py -v` 全绿，覆盖率 ≥ 95%
- `uv run mypy src/fugue/registry.py` 0 错误
- `uv run lint-imports --config .importlinter` 通过

**审查要求：** code-quality-reviewer

**依赖：** 任务 3（依赖 FugueRegistryError）
**风险：** 低

---

#### 任务 5：FugueConfig + YAML loader

**文件：**
- 新建 `src/fugue/config.py`
- 新建 `tests/unit/test_config.py`

**`config.py` 必含定义：**

1. **三个 dataclass（沿用 spec 4.3）**：
   - `GraphConfig`（含所有 fuge_plan 字段；`fallback_chain` 默认 `field(default_factory=list)`，`gen_mode` Literal["basic", "citation"]，**不含 cot/llm grader/hybrid grader/decompose/self_query/intent_router/weighted_fusion/compression/filter/dedupe/parent_child/sentence_window**——它们是 P1）
   - `IngestConfig`（parser="auto", chunker="recursive", chunk_size=512, chunk_overlap=64, collection_name="default", persist_dir="./.fugue"）
   - `ProviderConfig`（含 LLM/Embedding/Reranker 全部字段，默认值见 spec 4.3）
   - `FugueConfig`（聚合上述三层）

2. **Pydantic v2 校验镜像**：每个 dataclass 对应一个 `*Schema` Pydantic 模型，仅用于 YAML 解析期校验：
   ```python
   class GraphConfigSchema(BaseModel):
       model_config = ConfigDict(extra="forbid")  # 拒绝未知字段
       transforms: list[str | list[str]] = ["rewrite"]
       n_rewrites: int = Field(default=3, ge=1, le=20)
       max_queries: int = Field(default=20, ge=1, le=100)
       # ... 其余字段，含 ge/le 边界校验
   ```
   提供 `to_dataclass(self) -> GraphConfig` 方法把 Schema 转 dataclass。

3. **YAML loader 函数**：
   ```python
   def load_yaml(path: str | Path) -> FugueConfig:
       """读 YAML 文件 → ${env_var} 展开 → Pydantic 校验 → dataclass。
       失败时抛 FugueConfigError，message 含字段路径和原因。"""

   def expand_env_vars(text: str) -> str:
       """${VAR_NAME} → os.environ['VAR_NAME']；未定义时保留原字符串并 warning。"""
   ```

4. **dataclass → YAML 序列化**（双向等价）：
   ```python
   def dump_yaml(config: FugueConfig, path: str | Path) -> None: ...
   ```

**关键行为约定：**
- `${VAR}` 展开发生在 YAML 解析后、Pydantic 校验前
- Pydantic `extra="forbid"`：拒绝未知字段，抛 `FugueConfigError` 含字段路径
- 类型不匹配（如 `n_rewrites: "three"`）抛 `FugueConfigError` 含字段路径与期望类型
- 嵌套 transforms `[["step_back", "rewrite"]]` 与 `["rewrite", "hyde"]` 都合法
- 不在此任务校验"handler 名是否已注册"（那是 RAG.__init__ 的工作，见任务 25）

**测试规格（tests/unit/test_config.py）：**
- **默认实例化**：`FugueConfig()` 不抛错；`GraphConfig().fallback_chain == []`；`GraphConfig().processors == ["rrf", "rerank"]`
- **YAML round-trip**：dump_yaml(config) → load_yaml(path) 后 dataclass 字段全等
- **嵌套 transforms 解析**：YAML `transforms: ["hyde", ["step_back", "rewrite"]]` 正确解析为 `["hyde", ["step_back", "rewrite"]]`（保留嵌套结构）
- **`${env_var}` 展开**：YAML `llm_api_key: "${TEST_KEY}"` + 设置 `TEST_KEY=abc` 后值为 `"abc"`
- **未定义 env var**：YAML `${UNDEFINED_VAR}` 保留原字符串，捕获 warning log
- **未知字段**：YAML 含 `unknown_field: 1` 抛 `FugueConfigError`，message 含 `unknown_field`
- **类型错误**：YAML `n_rewrites: "three"` 抛 `FugueConfigError`，message 含 `n_rewrites` 和期望类型
- **边界值**：`n_rewrites: 0` 或 `n_rewrites: 21` 抛 `FugueConfigError`（Pydantic ge/le）

**验证标准：**
- `uv run pytest tests/unit/test_config.py -v` 全绿，覆盖率 ≥ 95%
- `uv run mypy src/fugue/config.py` 0 错误

**审查要求：** code-quality-reviewer

**依赖：** 任务 3（FugueConfigError）
**风险：** 中（Pydantic v2 与 dataclass 互转易出错；嵌套 transforms 类型签名复杂）

---

### 阶段 2：Provider 层

#### 任务 6：LLMClient（OpenAI 兼容）

**文件：**
- 新建 `src/fugue/providers/__init__.py`（空）
- 新建 `src/fugue/providers/llm.py`
- 新建 `tests/unit/test_providers/__init__.py`（空）
- 新建 `tests/unit/test_providers/test_llm.py`

**`providers/llm.py` 必含定义：**

```python
import threading
from openai import OpenAI

from fugue.api.types import FugueLLMError

class LLMClient:
    """OpenAI 兼容 LLM 统一封装。base_url + api_key 即可切换 provider。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 60.0,
        max_retries: int = 2,
        max_concurrent: int = 10,
    ) -> None:
        self._client = OpenAI(
            base_url=base_url, api_key=api_key,
            timeout=timeout, max_retries=max_retries,
        )
        self._model = model
        self._sem = threading.Semaphore(max_concurrent)

    def complete(self, prompt: str, *, temperature: float = 0.7) -> str:
        """同步调用 chat.completions.create；失败抛 FugueLLMError。"""
        with self._sem:
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                raise FugueLLMError(f"LLM call failed: {e}") from e

    def close(self) -> None:
        """关闭底层 httpx client。"""
        self._client.close()
```

**测试规格（tests/unit/test_providers/test_llm.py）：**
- **基础调用**：mock `openai.OpenAI`，验证 `complete("hello")` 调用 `chat.completions.create` 一次，参数含 model/messages/temperature
- **失败包装**：mock 抛 `openai.APIError`，验证 `complete` 抛 `FugueLLMError` 含原始异常
- **Semaphore 并发控制**：`max_concurrent=2`，使用 `threading.Barrier(3)` 让 3 个线程在 mock LLM 内同时等待，验证 Semaphore 释放前活跃计数 ≤ 2。**不得用 `time.sleep`**（CI runner 调度抖动会 flaky）；用 `threading.Event` + `concurrent.futures.ThreadPoolExecutor` 精确追踪进入顺序
- **temperature 透传**：`complete("x", temperature=0.1)` 调用时 kwargs 含 `temperature=0.1`
- **close**：调用后底层 client 的 close 被调用

**验证标准：**
- `uv run pytest tests/unit/test_providers/test_llm.py -v` 全绿，覆盖率 ≥ 95%
- `uv run mypy src/fugue/providers/llm.py` 0 错误

**审查要求：** code-quality-reviewer

**依赖：** 任务 3（FugueLLMError）
**风险：** 中（Semaphore 并发测试用 Barrier+Event 同步避免 flaky）

---

#### 任务 7：EmbeddingClient（OpenAI 兼容）

**文件：**
- 新建 `src/fugue/providers/embedding.py`
- 新建 `tests/unit/test_providers/test_embedding.py`

**`providers/embedding.py` 必含定义：**

```python
import threading
from openai import OpenAI

from fugue.api.types import FugueEmbeddingError

class EmbeddingClient:
    """OpenAI 兼容 Embedding 统一封装。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 30.0,
        max_retries: int = 2,
        max_concurrent: int = 5,
        batch_size: int = 64,
    ) -> None: ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量 embed；按 batch_size 分批；失败时该 batch 分块二次切分重试一次；
        最终失败抛 FugueEmbeddingError。"""

    def close(self) -> None: ...
```

**关键行为约定：**
- 输入 `texts` 长度 > `batch_size` 时分批调用，结果按原顺序拼接
- 单 batch 失败时：分两半重试一次（避免 1 个坏 input 整批挂）；二次失败抛 `FugueEmbeddingError`
- Semaphore 控制并发批数

**测试规格（tests/unit/test_providers/test_embedding.py）：**
- **基础调用**：mock OpenAI，验证 `embed(["a", "b"])` 返回 list[list[float]]
- **分批**：`batch_size=2` + 输入 5 个 texts，验证调用 3 次 API（2+2+1），结果顺序保持
- **批失败重试切分**：`batch_size=4` + 输入 4 个，mock 第一次抛错、第二次成功；验证最终成功且分两半各重试一次
- **彻底失败**：两次都抛错，最终抛 `FugueEmbeddingError`
- **空输入**：`embed([])` 返回 `[]` 且不调用 API

**验证标准：**
- `uv run pytest tests/unit/test_providers/test_embedding.py -v` 全绿，覆盖率 ≥ 95%
- `uv run mypy src/fugue/providers/embedding.py` 0 错误

**审查要求：** code-quality-reviewer

**依赖：** 任务 3（FugueEmbeddingError）
**风险：** 中（批量分块重试逻辑较易写错）

---

#### 任务 8：VectorStore（Chroma 实现）

**文件：**
- 新建 `src/fugue/providers/vector_store/__init__.py`（re-export）
- 新建 `src/fugue/providers/vector_store/base.py`（Protocol）
- 新建 `src/fugue/providers/vector_store/chroma.py`（实现）
- 新建 `tests/integration/__init__.py`（空）
- 新建 `tests/integration/test_chroma.py`

**`base.py` 必含 Protocol：**

```python
from typing import Protocol
from fugue.api.types import Chunk, Document

class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...
    def similarity_search(
        self,
        query_embedding: list[float],
        k: int = 20,
        metadata_filter: dict | None = None,
    ) -> list[Document]: ...
    def delete_collection(self) -> None: ...
    def stats(self) -> dict: ...
    def close(self) -> None: ...
    def iter_chunks(self, batch_size: int = 1000) -> Iterator[list[Chunk]]:
        """分批迭代所有 chunks（避免大语料 OOM），供 BM25 启动重建。"""
```

**`chroma.py` 必含实现：**

```python
import chromadb

class ChromaVectorStore:
    def __init__(self, persist_dir: str, collection_name: str = "default") -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(collection_name)
        self._name = collection_name

    def add(self, chunks, embeddings) -> None:
        """chroma upsert：ids = chunk_ids，避免重复 ingest 报错。"""

    def similarity_search(self, query_embedding, k=20, metadata_filter=None) -> list[Document]:
        """调用 self._collection.query，结果转 Document。
        Document.score 用 1 - distance（cosine）归一化到 0~1。
        source 字段填 "vector"（retriever 调用方覆写）。"""

    def iter_chunks(self, batch_size: int = 1000) -> Iterator[list[Chunk]]:
        """分批迭代所有 chunks（含 metadata + parent_id），供 BM25 启动重建。
        用 Chroma get(limit=batch_size, offset=...) 分批拉取，避免大语料 OOM。"""
```

**测试规格（tests/integration/test_chroma.py）：用真 Chroma（tmp_path）**
- **add + similarity_search**：写入 5 个 chunks + embeddings，查询返回 top_k 文档，score 在 [0, 1]
- **upsert 行为**：同 chunk_id 再次 add，count 不增加
- **metadata_filter**：filter `{"year": 2024}` 只返回匹配文档
- **stats**：返回含 `num_chunks` 字段，等于已写入数量
- **iter_chunks 分批**：写入 2500 个 chunks 后，`iter_chunks(batch_size=1000)` 返回 3 个 batch（1000+1000+500），每 batch 是 list[Chunk]
- **iter_chunks 元数据**：每个 Chunk 含 parent_id（若有）
- **空查询**：`similarity_search` 在空 collection 上返回 `[]`
- **delete_collection**：删除后 stats num_chunks=0

**验证标准：**
- `uv run pytest tests/integration/test_chroma.py -v` 全绿，覆盖率 ≥ 90%
- `uv run mypy src/fugue/providers/vector_store/` 0 错误
- `uv run lint-imports` 通过（chroma.py 不允许 import langgraph）

**审查要求：** code-quality-reviewer

**依赖：** 任务 3（Chunk, Document）
**风险：** 中（Chroma API 版本变化；score 归一化策略）

---

#### 任务 9：Reranker（BGE 实现）

**文件：**
- 新建 `src/fugue/providers/reranker/__init__.py`
- 新建 `src/fugue/providers/reranker/base.py`（Protocol）
- 新建 `src/fugue/providers/reranker/bge.py`
- 新建 `tests/unit/test_providers/test_reranker.py`

**`base.py` 必含 Protocol：**

```python
from typing import Protocol

class Reranker(Protocol):
    def rerank(
        self, query: str, documents: list[str], top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """返回 [(原始索引, 新分数), ...] 按分数降序。"""
    def close(self) -> None: ...
```

**`bge.py` 必含实现：**

```python
from FlagEmbedding import FlagReranker

class BGEReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "auto",  # cpu / cuda / auto
        timeout: float = 30.0,
    ) -> None:
        # device="auto" → 检测 CUDA 可用性
        self._reranker = FlagReranker(model_name, use_fp16=True, devices=...)

    def rerank(self, query, documents, top_k=None) -> list[tuple[int, float]]:
        """调用 FlagReranker.compute_score，返回 (idx, score) 降序。
        documents 为空时返回 []。"""

    def close(self) -> None:
        """显式释放显存 / 大 RAM 占用。"""
        del self._reranker
```

**测试规格（tests/unit/test_providers/test_reranker.py）：mock FlagReranker**
- **基础调用**：mock `FlagReranker.compute_score` 返回 `[0.3, 0.8, 0.1]`，验证 `rerank("q", ["a", "b", "c"])` 返回 `[(1, 0.8), (0, 0.3), (2, 0.1)]`
- **top_k 截断**：`top_k=2` 时返回前 2 个
- **空 documents**：`rerank("q", [])` 返回 `[]` 且不调用模型
- **device="auto" 检测**：mock torch.cuda.is_available 验证分支
- **close**：调用后内部 `_reranker` 被 del

**验证标准：**
- `uv run pytest tests/unit/test_providers/test_reranker.py -v` 全绿，覆盖率 ≥ 90%
- `uv run mypy src/fugue/providers/reranker/` 0 错误

**审查要求：** code-quality-reviewer

**依赖：** 任务 3
**风险：** 中（FlagEmbedding API 版本差异；mock 复杂）

---

#### 任务 10：BM25 Provider

**文件：**
- 新建 `src/fugue/providers/bm25.py`
- 新建 `tests/unit/test_providers/test_bm25.py`

**`bm25.py` 必含定义：**

```python
import threading
from rank_bm25 import BM25Okapi

from fugue.api.types import Chunk, Document

class BM25Provider:
    """内存中维护的 BM25 索引。启动时从 vector store 重建。"""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._lock = threading.RLock()

    def rebuild(self, chunks: list[Chunk]) -> None:
        """全量重建索引。tokenize 用简单空白分割（中英文 MVP 都接受）。"""

    def update(self, new_chunks: list[Chunk]) -> None:
        """增量追加 chunks 并重建（rank_bm25 不支持增量，必须全量重建）。"""

    def search(self, query: str, k: int = 20) -> list[Document]:
        """返回 top-k Document，score 是 BM25 原始分数（未归一化），source="bm25"。"""

    def close(self) -> None: ...
```

**关键行为约定：**
- `tokenize` MVP 用 `text.lower().split()`（中英文都先按空白分；中文未来 P1 加 jieba）
- `search` 时若索引为空（`_bm25 is None`），返回 `[]`（不抛错）
- 所有写操作（rebuild/update）持锁；读操作（search）也持锁——MVP 简化，性能 P1 优化

**测试规格（tests/unit/test_providers/test_bm25.py）：**
- **rebuild + search**：5 个 chunks → search "lang" → 返回含 "lang" 的 chunk 排第一
- **空 search**：未 rebuild 时 search 返回 `[]`
- **update 触发全量重建**：rebuild(3 chunks) → update(2 new chunks) → search 能找到所有 5 个内容
- **score 字段**：Document.score 是 float，source="bm25"
- **k 截断**：search k=2 时返回最多 2 个
- **并发 search**：3 线程同时 search 不报错（验证读锁工作）
- **性能基线**：rebuild 10,000 chunks 在 5 秒内完成（标记 `@pytest.mark.slow`，CI 含此 marker）
- **中文分词已知失效**：用 5 个全中文 chunks rebuild，查询中文词 search 返回 `[]`（验证已知限制）；该用例标记 `@pytest.mark.xfail(reason="MVP 用空白分词，P1 加 jieba")` 以便 P1 修复后自动变 XPASS 提醒

**验证标准：**
- `uv run pytest tests/unit/test_providers/test_bm25.py -v` 全绿，覆盖率 ≥ 95%
- `uv run mypy src/fugue/providers/bm25.py` 0 错误

**审查要求：** code-quality-reviewer

**依赖：** 任务 3
**风险：** 低（rank_bm25 稳定）

---

### 阶段 3：Handlers — Query 路径

#### 任务 11：Transforms（rewrite + hyde + step_back + 嵌套执行器）

**文件：**
- 新建 `src/fugue/handlers/__init__.py`（顶层 import 子模块触发注册）
- 新建 `src/fugue/handlers/transforms/__init__.py`（import atoms + pipeline，注册到 transform_registry）
- 新建 `src/fugue/handlers/transforms/atoms.py`（rewrite_fn + hyde_fn + step_back_fn）
- 新建 `src/fugue/handlers/transforms/pipeline.py`（_run_transform_branch 嵌套执行器）
- 新建 `tests/unit/test_handlers/__init__.py`
- 新建 `tests/unit/test_handlers/test_transforms.py`

**`atoms.py` 必含 3 个函数（统一签名 `(queries: list[str], n: int, llm: LLMClient) -> list[str]`）：**

```python
def rewrite_fn(queries: list[str], n: int, llm) -> list[str]:
    """对每个 query 改写成 n 个不同表述。
    Prompt: '请将以下问题改写成 n 个不同的表述，每行一个：\n{q}'
    返回所有改写结果（不含原 query）。"""

def hyde_fn(queries: list[str], n: int, llm) -> list[str]:
    """对每个 query 生成 n 段假设性回答（用于检索）。"""

def step_back_fn(queries: list[str], n: int, llm) -> list[str]:
    """对每个 query 抽象为 n 个上位问题。"""
```

**LLMClient 注入约定：**

由于 transform 签名在 spec 中是 `Callable[[list[str], int], list[str | TransformResult]]`，但实际需要 LLM。**采用闭包注入**：

- `transforms/__init__.py` 定义 `register_transforms(llm_client: LLMClient) -> None`
- 该函数闭包绑定 LLM，包装 atom_fn 为 `Callable[[list[str], int], list[str]]` 后 register
- `RAG.__init__` 调用 `register_transforms(self._llm_client)`

```python
# transforms/__init__.py
def register_transforms(llm_client) -> None:
    from fugue.registry import transform_registry
    from fugue.handlers.transforms.atoms import rewrite_fn, hyde_fn, step_back_fn

    transform_registry.register("rewrite",  lambda q, n: rewrite_fn(q, n, llm_client))
    transform_registry.register("hyde",     lambda q, n: hyde_fn(q, n, llm_client))
    transform_registry.register("step_back", lambda q, n: step_back_fn(q, n, llm_client))
```

**`pipeline.py` 必含函数：**

```python
def run_transform_branch(
    branch: str | list[str],
    queries: list[str],
    n: int,
    registry,  # transform_registry
) -> list[str | TransformResult]:
    """执行单个 transform 分支：
    - str: 原子 transform，registry.get(branch)(queries, n)
    - list[str]: 管道链，按顺序串联（前一个输出 → 后一个输入）
      若中间结果是 TransformResult，提取 .query 继续传递"""
```

**测试规格（tests/unit/test_handlers/test_transforms.py）：mock LLM**
- **rewrite_fn**：mock LLM 返回 "Q1\nQ2\nQ3"，验证 `rewrite_fn(["原问题"], 3, mock_llm)` 返回 `["Q1", "Q2", "Q3"]`（解析换行）
- **hyde_fn**：mock LLM 返回 "A1\n\nA2"，验证返回 `["A1", "A2"]`
- **step_back_fn**：mock LLM 返回 "更宏观Q1\n更宏观Q2\n更宏观Q3"，验证返回 3 个 query
- **register_transforms**：调用后 transform_registry 中 "rewrite"/"hyde"/"step_back" 都已注册
- **run_transform_branch（原子）**：branch="rewrite"，registry mock 返回 `["Q1"]`，验证调用一次
- **run_transform_branch（管道）**：branch=["step_back", "rewrite"]，step_back 返回 ["抽象Q1"]，rewrite 收到 ["抽象Q1"] 返回 ["改写Q1.1", "改写Q1.2"]，验证最终输出
- **管道中 TransformResult 传递**：branch=["self_query", "rewrite"] mock，验证 self_query 输出 TransformResult 后 rewrite 收到的是 .query 字符串
- **空输入**：queries=[] 时所有 fn 返回 []

**关键约定：syrupy 快照测试 prompt 模板**——`rewrite_fn` 调用 LLM 时的 prompt 字符串用 syrupy 快照，任何 prompt 修改要更新快照才能 commit。

**验证标准：**
- `uv run pytest tests/unit/test_handlers/test_transforms.py -v` 全绿，覆盖率 ≥ 95%
- `uv run mypy src/fugue/handlers/transforms/` 0 错误
- syrupy 快照存在于 `tests/unit/test_handlers/__snapshots__/test_transforms.ambr`

**审查要求：** code-quality-reviewer

**依赖：** 任务 3（TransformResult）、任务 4（registry）、任务 6（LLMClient）
**风险：** 中（LLM 响应解析的鲁棒性）

---

#### 任务 12：Retrievers（vector + bm25）

**文件：**
- 新建 `src/fugue/handlers/retrievers/__init__.py`
- 新建 `src/fugue/handlers/retrievers/atoms.py`
- 新建 `tests/unit/test_handlers/test_retrievers.py`

**`atoms.py` 必含定义：**

```python
def make_vector_search(vector_store, embedding_client):
    """返回 retriever 闭包：query + metadata_filter → list[Document]。"""
    def vector_search(query: str, metadata_filter: dict | None = None) -> list[Document]:
        emb = embedding_client.embed([query])[0]
        docs = vector_store.similarity_search(emb, k=20, metadata_filter=metadata_filter)
        for d in docs:
            d["source"] = "vector"
        return docs
    return vector_search

def make_bm25_search(bm25_provider):
    def bm25_search(query: str, metadata_filter: dict | None = None) -> list[Document]:
        # bm25 MVP 不支持 metadata_filter，忽略参数
        docs = bm25_provider.search(query, k=20)
        for d in docs:
            d["source"] = "bm25"
        return docs
    return bm25_search
```

**`__init__.py` 提供 `register_retrievers(vector_store, embedding_client, bm25_provider)` 注册到 retriever_registry。**

**测试规格（tests/unit/test_handlers/test_retrievers.py）：mock provider**
- **vector_search**：mock vector_store + embedding_client，验证调用链与返回 Document 的 source="vector"
- **bm25_search**：mock bm25_provider，验证 source="bm25"
- **metadata_filter 透传**：vector_search 调用 vector_store 时 metadata_filter 参数被透传
- **bm25 忽略 metadata_filter**：传入 filter 不报错且不影响结果
- **register_retrievers**：注册后 retriever_registry 含 "vector" 和 "bm25"

**验证标准：**
- `uv run pytest tests/unit/test_handlers/test_retrievers.py -v` 全绿，覆盖率 ≥ 95%
- `uv run mypy src/fugue/handlers/retrievers/` 0 错误

**审查要求：** code-quality-reviewer

**依赖：** 任务 3, 4, 7（embedding）, 8（vector store）, 10（bm25）
**风险：** 低

---

#### 任务 13：Processors（rrf + rerank）

**文件：**
- 新建 `src/fugue/handlers/processors/__init__.py`
- 新建 `src/fugue/handlers/processors/rrf.py`
- 新建 `src/fugue/handlers/processors/rerank.py`
- 新建 `tests/unit/test_handlers/test_processors.py`

**`rrf.py` 必含函数（沿用 fuge_plan §5）：**

```python
def rrf_fn(
    docs: list[Document],
    query: str,
    top_k: int,
    *,
    retriever_weights: dict[str, float] | None = None,
    **kwargs,
) -> list[Document]:
    """Reciprocal Rank Fusion: RRF_score = Σ (w_source / (60 + rank))
    使用 (source, doc_id) 复合键标识，按 source 分组组内按原始 score 降序排 rank。
    写回 d["rrf_score"] 并按 rrf_score 降序排序。"""
```

**`rerank.py` 必含函数：**

```python
def make_rerank(reranker):
    """返回 processor 闭包，闭包绑定 reranker。"""
    def rerank_fn(
        docs: list[Document], query: str, top_k: int, **kwargs,
    ) -> list[Document]:
        if not docs:
            return []
        contents = [d["content"] for d in docs]
        scored = reranker.rerank(query, contents, top_k=top_k)
        # scored = [(原始 idx, 新 score), ...] 已降序
        result = []
        for idx, score in scored:
            d = dict(docs[idx])
            d["rerank_score"] = score
            result.append(d)
        return result
    return rerank_fn
```

**`__init__.py` 提供 `register_processors(reranker)`。**

**测试规格（tests/unit/test_handlers/test_processors.py）：**
- **rrf 基础**：3 docs 跨 2 source，权重均 1.0，验证 rrf_score 计算正确（手算对比）
- **rrf 加权**：retriever_weights={"vector": 1.0, "bm25": 0.5}，验证 bm25 文档分数低
- **rrf 跨 source 同 doc_id**：保留两条记录（按 (source, doc_id) 复合键）
- **rrf 空输入**：返回 `[]`
- **rerank 基础**：mock reranker 返回 `[(2, 0.9), (0, 0.5)]`，验证返回顺序与 rerank_score 字段
- **rerank 空输入**：返回 `[]`
- **rerank top_k 截断**：top_k=2 时只返回 2 个

**验证标准：**
- `uv run pytest tests/unit/test_handlers/test_processors.py -v` 全绿，覆盖率 ≥ 95%
- `uv run mypy src/fugue/handlers/processors/` 0 错误

**审查要求：** code-quality-reviewer

**依赖：** 任务 3, 4, 9（reranker）
**风险：** 低（RRF 是确定性算法）

---

#### 任务 14：Graders（score + normalizer）

**文件：**
- 新建 `src/fugue/handlers/graders/__init__.py`
- 新建 `src/fugue/handlers/graders/score.py`
- 新建 `src/fugue/handlers/graders/normalizer.py`
- 新建 `tests/unit/test_handlers/test_graders.py`

**`normalizer.py` 必含函数：**

```python
def normalize_score(doc: Document, score_normalizers: dict[str, float]) -> float:
    """按 source 归一化分数到 0~1。
    score_normalizers={"bm25": 20.0} 表示 bm25 原始分数上界 20，未列出的 source 默认 1.0。"""
    raw = doc.get("score", 0)
    source = doc.get("source", "")
    max_score = score_normalizers.get(source, 1.0)
    if max_score <= 0:
        return 0.0
    return min(raw / max_score, 1.0)
```

**`score.py` 必含函数：**

```python
def score_grader(
    docs: list[Document],
    query: str,
    threshold: float,
    *,
    score_normalizers: dict[str, float] | None = None,
    **kwargs,
) -> tuple[float, Literal["sufficient", "insufficient"]]:
    """基于归一化后的平均分数判断。docs=[] 时返回 (0.0, 'insufficient')。"""
```

**`__init__.py` 提供 `register_graders()`（无依赖，直接 register `score_grader`）。**

**测试规格（tests/unit/test_handlers/test_graders.py）：**
- **normalize_score**：source="bm25"，raw=15.0，max=20 → 0.75；source="vector"，raw=0.9 → 0.9（默认 max=1）
- **normalize_score 边界**：raw=25 max=20 → 1.0（截断）；max=0 → 0.0
- **score_grader 充分**：3 docs 归一化均分 0.7，threshold=0.6 → ("0.7", "sufficient")
- **score_grader 不足**：均分 0.4 → "insufficient"
- **score_grader 空 docs**：→ (0.0, "insufficient")
- **score_grader 跨 source 混合**：vector docs + bm25 docs 各自归一化后平均

**验证标准：**
- `uv run pytest tests/unit/test_handlers/test_graders.py -v` 全绿，覆盖率 ≥ 95%
- `uv run mypy src/fugue/handlers/graders/` 0 错误

**审查要求：** code-quality-reviewer

**依赖：** 任务 3, 4
**风险：** 低

---

#### 任务 15：Generators（basic + citation）

**文件：**
- 新建 `src/fugue/handlers/generators/__init__.py`
- 新建 `src/fugue/handlers/generators/basic.py`
- 新建 `src/fugue/handlers/generators/citation.py`
- 新建 `tests/unit/test_handlers/test_generators.py`

**`basic.py` 必含定义：**

```python
def make_basic_generator(llm):
    def basic_fn(
        query: str, docs: list[Document], temperature: float, **kwargs,
    ) -> str:
        context = "\n\n".join(d["content"] for d in docs)
        prompt = (
            f"基于以下上下文回答问题。\n\n"
            f"上下文：\n{context}\n\n"
            f"问题：{query}"
        )
        return llm.complete(prompt, temperature=temperature)
    return basic_fn
```

**`citation.py` 必含定义：**

```python
def make_citation_generator(llm):
    def citation_fn(
        query: str, docs: list[Document], temperature: float, **kwargs,
    ) -> str:
        context = "\n\n".join(
            f"[{i + 1}] {d['content']}" for i, d in enumerate(docs)
        )
        prompt = (
            f"基于以下上下文回答问题，每个论点必须标注来源编号 [1][2]...。\n"
            f"来源编号已在每段文档前标注。\n\n"
            f"上下文：\n{context}\n\n"
            f"问题：{query}"
        )
        return llm.complete(prompt, temperature=temperature)
    return citation_fn
```

**`__init__.py` 提供 `register_generators(llm)`。**

**测试规格（tests/unit/test_handlers/test_generators.py）：mock LLM**
- **basic prompt 内容**：调用后 LLM 收到的 prompt 含 "基于以下上下文回答问题"、上下文文本、原 query
- **citation prompt 含编号**：3 docs 时 prompt 含 `[1]` `[2]` `[3]`
- **空 docs**：basic 不抛错，prompt 中上下文为空字符串
- **temperature 透传**：调用 `llm.complete` 时 kwargs 含 `temperature`
- **syrupy 快照**：两个 prompt 模板的快照（防止默改）

**验证标准：**
- `uv run pytest tests/unit/test_handlers/test_generators.py -v` 全绿，覆盖率 ≥ 95%
- `uv run mypy src/fugue/handlers/generators/` 0 错误
- syrupy 快照存在

**审查要求：** code-quality-reviewer

**依赖：** 任务 3, 4, 6
**风险：** 低

---

### 阶段 4：Handlers — Ingest 路径

#### 任务 16：Parsers（markdown/text + pdf）

**文件：**
- 新建 `src/fugue/handlers/parsers/__init__.py`
- 新建 `src/fugue/handlers/parsers/markdown_text.py`
- 新建 `src/fugue/handlers/parsers/pdf.py`
- 新建 `tests/unit/test_handlers/test_parsers.py`
- 新建 `tests/fixtures/sample.md`（5 行 Markdown）
- 新建 `tests/fixtures/sample.txt`（5 行纯文本）
- 新建 `tests/fixtures/sample.pdf`（最小 PDF，用 pypdf 生成或 git 直接提交一份）

**`markdown_text.py` 必含定义：**

```python
def markdown_parser(path: Path) -> list[ParsedDocument]:
    """读取 .md 文件，返回单个 ParsedDocument（content 是原文，metadata={"format":"markdown"}）。"""

def text_parser(path: Path) -> list[ParsedDocument]:
    """读取 .txt 文件，metadata={"format":"text"}。"""

def auto_parser(path: Path) -> list[ParsedDocument]:
    """按扩展名分派：.md → markdown_parser, .txt → text_parser, .pdf → pdf_parser。
    未知扩展名抛 ValueError。"""
```

**`pdf.py` 必含定义：**

```python
import pypdf

def pdf_parser(path: Path) -> list[ParsedDocument]:
    """用 pypdf 解析，每页一个 ParsedDocument，metadata={"format":"pdf","page":1}。"""
```

**`__init__.py` 注册 `markdown` / `text` / `pdf` / `auto` 四个 parser 到 parser_registry。**

**测试规格（tests/unit/test_handlers/test_parsers.py）：用 fixtures**
- **markdown_parser**：解析 `sample.md`，返回 1 个 ParsedDocument，content 非空
- **text_parser**：解析 `sample.txt`，同上
- **pdf_parser**：解析 `sample.pdf`，返回 ≥1 个 ParsedDocument，每个含 page metadata
- **auto_parser**：分别用 .md/.txt/.pdf 路径调用，返回类型正确
- **auto_parser 未知扩展**：`.xyz` 抛 `ValueError` 含 "unsupported extension"
- **空文件**：sample_empty.md（空内容）解析成功，content=""

**验证标准：**
- `uv run pytest tests/unit/test_handlers/test_parsers.py -v` 全绿，覆盖率 ≥ 90%
- `uv run mypy src/fugue/handlers/parsers/` 0 错误

**审查要求：** code-quality-reviewer

**依赖：** 任务 3, 4
**风险：** 低（pypdf 稳定）

---

#### 任务 17：Chunkers（recursive）

**文件：**
- 新建 `src/fugue/handlers/chunkers/__init__.py`
- 新建 `src/fugue/handlers/chunkers/recursive.py`
- 新建 `tests/unit/test_handlers/test_chunkers.py`

**`recursive.py` 必含定义：**

```python
import hashlib

def recursive_chunker(
    parsed_docs: list[ParsedDocument],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """递归按分隔符切分：先按 "\n\n" 切，块仍超长则按 "\n" 切，再按 ". " 切，最后按字符。
    生成 chunk_id = sha1(source_path + chunk_index + content[:128]).hexdigest()[:16]。
    parent_id = sha1(source_path).hexdigest()[:16]（同一文档共享）。
    metadata 继承 ParsedDocument.metadata + chunk_index + source_path。"""
```

**`__init__.py` 注册 `recursive` 到 chunker_registry。**

**测试规格（tests/unit/test_handlers/test_chunkers.py）：**
- **chunk_size 切分**：1000 字符文档 + chunk_size=200，验证产出 ≥5 个 chunks
- **chunk_overlap**：连续 chunks 有约 64 字符重叠（递归切分场景下不严格要求"末尾=下一个开头"，但相邻 chunks 的 content 必有 ≥1 字符的子串重叠）
- **chunk_id 稳定**：同一内容多次 chunk，chunk_id 相同（用于 upsert 去重）
- **chunk_id 唯一**：跨文档/跨索引保持唯一（无碰撞，验证 16 chars 足够）
- **parent_id 共享**：同一 source_path 的所有 chunks 共享 parent_id
- **metadata 继承**：parent metadata `{"format": "pdf"}` 传递到 chunk metadata
- **空文档**：parsed_docs=[] 返回 `[]`
- **超短文档**：100 字符 + chunk_size=512 → 1 个 chunk（不切）

**验证标准：**
- `uv run pytest tests/unit/test_handlers/test_chunkers.py -v` 全绿，覆盖率 ≥ 95%
- `uv run mypy src/fugue/handlers/chunkers/` 0 错误

**审查要求：** code-quality-reviewer

**依赖：** 任务 3, 4
**风险：** 中（递归切分边界情况多）

---

### 阶段 5：Engine — 隐藏 LangGraph 层

#### 任务 18：RAGState + reducer + runtime

**文件：**
- 新建 `src/fugue/engine/__init__.py`
- 新建 `src/fugue/engine/state.py`
- 新建 `src/fugue/engine/runtime.py`
- 新建 `tests/unit/test_engine/__init__.py`
- 新建 `tests/unit/test_engine/test_state.py`
- 新建 `tests/unit/test_engine/test_runtime.py`

**`state.py` 必含定义：**

```python
from typing import Annotated, Literal, TypedDict
from fugue.api.types import Document

def merge_docs(existing: list[Document], new: list[Document]) -> list[Document]:
    """按 (source, doc_id) 复合键去重合并。"""
    seen = {(d["source"], d["doc_id"]) for d in existing}
    return existing + [d for d in new if (d["source"], d["doc_id"]) not in seen]

class RAGState(TypedDict):
    original_query: str
    rewritten_queries: list[str]
    documents: Annotated[list[Document], merge_docs]
    grade_score: float
    grade_decision: Literal["sufficient", "insufficient"]
    source: str  # "kb" / "web" / 任意自定义 fallback 源
    retry_count: int
    retrieval_history: list[list[Document]]
    ranked_documents: list[Document]
    answer: str

class RetrieveInput(TypedDict):
    """Send() 发送给 retrieve 节点的负载。"""
    query: str
    retriever_name: str
    source: str
    metadata_filter: dict | None
```

**`runtime.py` 必含定义：**

```python
from dataclasses import fields as dc_fields
from langchain_core.runnables import RunnableConfig

from fugue.config import GraphConfig

_GRAPH_CONFIG_FIELDS = {f.name for f in dc_fields(GraphConfig)}

def get_config(config: RunnableConfig) -> GraphConfig:
    """从 RunnableConfig.configurable 提取业务配置。
    自动过滤 LangGraph 框架注入的键，只保留 GraphConfig 字段。"""
    raw = config.get("configurable", {})
    filtered = {k: v for k, v in raw.items() if k in _GRAPH_CONFIG_FIELDS}
    return GraphConfig(**filtered)
```

**测试规格：**
- **test_state.py**：
  - merge_docs 基础去重：(source="v", doc_id="1") 重复时只保留一个
  - merge_docs 跨 source 同 doc_id：两条都保留
  - merge_docs 顺序保持：existing 在前，new 中未重复的按顺序追加
- **test_runtime.py**：
  - get_config 提取 configurable：`{"configurable": {"n_rewrites": 5, "thread_id": "xyz"}}` 返回 `GraphConfig(n_rewrites=5)`（thread_id 被过滤）
  - get_config 空 configurable：返回默认 GraphConfig
  - get_config 未知字段：被过滤，不抛错

**验证标准：**
- `uv run pytest tests/unit/test_engine/ -v` 全绿，覆盖率 ≥ 95%
- `uv run lint-imports` 通过（runtime.py 允许 import langchain_core，state.py 不需要）

**审查要求：** code-quality-reviewer

**依赖：** 任务 3, 5
**风险：** 低

---

#### 任务 19：节点 query_transform + retrieve

**文件：**
- 新建 `src/fugue/engine/nodes/__init__.py`
- 新建 `src/fugue/engine/nodes/query_transform.py`
- 新建 `src/fugue/engine/nodes/retrieve.py`
- 新建 `tests/unit/test_engine/test_nodes_qt_retrieve.py`

**`query_transform.py` 必含函数：**

```python
from langgraph.types import Command, Overwrite, Send
from langchain_core.runnables import RunnableConfig

from fugue.api.types import TransformResult
from fugue.engine.state import RAGState
from fugue.engine.runtime import get_config
from fugue.registry import transform_registry
from fugue.handlers.transforms.pipeline import run_transform_branch

def query_transform(
    state: RAGState, config: RunnableConfig,
) -> Command:
    """1. `all_queries = [state['original_query']]`  ← **始终将原始 query 作为第 0 个**
       2. 遍历 cfg.transforms 顶层分支（并行+串联），调用 run_transform_branch；
       3. **TransformResult 提取在本节点进行**：
          if isinstance(r, TransformResult):
              all_queries.append(r.query)
              if r.metadata_filter: query_filters[r.query] = r.metadata_filter
          else:
              all_queries.append(r)  # str
       4. 去重（保持顺序）+ 截断 cfg.max_queries；
       5. 按 source 决定 retriever_names：fallback 时单源 [state['source']]，kb 时 cfg.retrievers；
       6. 构建 Send 列表（笛卡尔积 queries × retriever_names），payload 含 metadata_filter；
       7. Command(update={'rewritten_queries': all_queries, 'documents': Overwrite([])}, goto=sends)"""
```

**`retrieve.py` 必含函数：**

```python
import logging
from fugue.engine.state import RetrieveInput
from fugue.registry import retriever_registry

logger = logging.getLogger("fugue.engine.nodes.retrieve")

def retrieve(state: RetrieveInput) -> dict:
    """单个检索任务。best-effort：捕获异常返回空 documents 并记录 log。"""
    try:
        fn = retriever_registry.get(state["retriever_name"])
        docs = fn(query=state["query"], metadata_filter=state.get("metadata_filter"))
        for d in docs:
            d["source"] = state["retriever_name"]
        return {"documents": docs}
    except Exception as e:
        logger.error(
            "Retriever '%s' failed for query '%s': %s",
            state["retriever_name"], state["query"][:50], e,
        )
        return {"documents": []}
```

**测试规格（tests/unit/test_engine/test_nodes_qt_retrieve.py）：**

**query_transform**：
- **基础扇出**：transforms=["rewrite"] + n_rewrites=2 + retrievers=["vector"]，mock rewrite 返回 ["q1","q2"]，验证 Command.goto 含 3 个 Send（[原始 query, q1, q2] × 1 retriever）——**原始 query 始终在第 0 位**
- **嵌套并行+串联**：transforms=["hyde", ["step_back", "rewrite"]] + retrievers=["vector","bm25"]，mock 各 transform，验证最终 Send 数 = (n_hyde + n_step_back×n_rewrite + 1) × 2，被 max_queries 截断
- **去重**：transforms 产出含重复 query，all_queries 去重后扇出
- **fallback 单源**：state.source="web"，验证 retriever_names = ["web"]
- **TransformResult 含 metadata_filter**：query_filters 中记录 filter，Send payload 含 metadata_filter
- **Overwrite([])**：Command.update["documents"] 是 Overwrite 实例

**retrieve**：
- **正常返回**：mock retriever 返回 2 docs，验证 source 字段被填 retriever_name
- **异常 best-effort**：mock retriever 抛 Exception，验证返回 `{"documents": []}` 且 log.error 被调用
- **metadata_filter 透传**：调用 retriever 时 kwargs 含正确 metadata_filter

**验证标准：**
- `uv run pytest tests/unit/test_engine/test_nodes_qt_retrieve.py -v` 全绿，覆盖率 ≥ 95%
- `uv run lint-imports` 通过（query_transform.py 允许 import langgraph）

**审查要求：** code-quality-reviewer

**依赖：** 任务 3, 4, 5, 11, 18
**风险：** 中（Send 扇出逻辑测试复杂）

---

#### 任务 20：节点 grade + prepare_fallback

**文件：**
- 新建 `src/fugue/engine/nodes/grade.py`（含 `grade` 与 `route_after_grade`）
- 新建 `src/fugue/engine/nodes/prepare_fallback.py`
- 新建 `tests/unit/test_engine/test_nodes_grade_fallback.py`

**`grade.py` 必含函数：**

```python
def grade(state: RAGState, config: RunnableConfig) -> dict:
    """调 grader_registry.get(cfg.grade_strategy)；归档当前 documents 到 retrieval_history。"""

def route_after_grade(state: RAGState, config: RunnableConfig) -> str:
    """conditional edge 路由：
    - sufficient → 'post_process'
    - insufficient 且 fallback_chain 已耗尽（retry_count >= len）或 retry >= max_retries → 'post_process'
    - insufficient 且未耗尽 → 'fallback_to_query_transform'。"""
```

**`prepare_fallback.py` 必含函数：**

```python
def prepare_fallback(state: RAGState, config: RunnableConfig) -> dict:
    """从 fallback_chain[retry_count] 取下一个 source，retry_count++。
    不重置 documents（query_transform 用 Overwrite([]) 处理）。"""
```

**测试规格（tests/unit/test_engine/test_nodes_grade_fallback.py）：**

**grade**：
- 调用 grader_registry.get("score")，验证返回 dict 含 grade_score / grade_decision / retrieval_history
- retrieval_history 在原 history 后 append 当前 documents
- 空 documents：grade_decision="insufficient"，history 含一项空 list

**route_after_grade**：
- sufficient → "post_process"
- insufficient + fallback_chain=[] → "post_process"
- insufficient + fallback_chain=["web"] + retry_count=0 + max_retries=1 → "fallback_to_query_transform"
- insufficient + retry_count=1 (已尝试过) + max_retries=1 → "post_process"
- insufficient + retry_count=1 + fallback_chain 长度=1 → "post_process"

**prepare_fallback**：
- retry_count=0 + fallback_chain=["web", "kg"] → {"source": "web", "retry_count": 1}
- retry_count=1 + fallback_chain=["web", "kg"] → {"source": "kg", "retry_count": 2}

**验证标准：**
- `uv run pytest tests/unit/test_engine/test_nodes_grade_fallback.py -v` 全绿，覆盖率 ≥ 95%

**审查要求：** code-quality-reviewer

**依赖：** 任务 3, 4, 5, 14, 18
**风险：** 中（路由逻辑边界条件多）

---

#### 任务 21：节点 post_process + generate

**文件：**
- 新建 `src/fugue/engine/nodes/post_process.py`
- 新建 `src/fugue/engine/nodes/generate.py`
- 新建 `tests/unit/test_engine/test_nodes_post_generate.py`

**`post_process.py` 必含函数：**

```python
def post_process(state: RAGState, config: RunnableConfig) -> dict:
    """1. 合并 state.documents + retrieval_history[:-1]（最后一项是当前轮，已在 documents）；
       2. 按 (source, doc_id) 去重；
       3. **防御性上界**：合并后若超过 1000 docs（常量 MAX_DOCS_BEFORE_PROCESS），
          按原始 score 降序截断到 1000，避免长 fallback 链下 O(n²) 处理爆炸；
       4. 链式 processors（按 cfg.processors 顺序）；
       5. 截断 cfg.top_k；
       6. 返回 {ranked_documents}。
       传给 processor 的 kwargs 含 retriever_weights / score_normalizers / top_k。"""
```

**`generate.py` 必含函数：**

```python
def generate(state: RAGState, config: RunnableConfig) -> dict:
    """调 generator_registry.get(cfg.gen_mode)；返回 {answer}。"""
```

**测试规格（tests/unit/test_engine/test_nodes_post_generate.py）：**

**post_process**：
- **跨轮合并去重**：state.documents=[D1] + retrieval_history=[[D2], [D1]]，验证 history[:-1]=[D2] 合并后去重输出
- **链式 processors**：cfg.processors=["rrf", "rerank"]，mock 两个 processor，验证按顺序调用，前一个输出喂后一个
- **top_k 截断**：processor 输出 10 个，top_k=3，最终返回 3 个
- **空 documents**：返回 {"ranked_documents": []}
- **kwargs 透传**：调用 processor 时 kwargs 含 retriever_weights/score_normalizers/top_k

**generate**：
- 调 generator_registry.get("basic")，验证返回 {"answer": ...}
- gen_mode="citation"：调 citation generator
- temperature 透传

**验证标准：**
- `uv run pytest tests/unit/test_engine/test_nodes_post_generate.py -v` 全绿，覆盖率 ≥ 95%

**审查要求：** code-quality-reviewer

**依赖：** 任务 3, 4, 5, 13, 15, 18
**风险：** 中（跨轮合并去重逻辑易写错）

---

#### 任务 22：图组装 build_rag_graph

**文件：**
- 新建 `src/fugue/engine/graph.py`
- 新建 `tests/integration/test_engine_graph.py`

**`graph.py` 必含函数：**

```python
from langgraph.graph import StateGraph, END

def build_rag_graph():
    """组装 7 节点图：
    - 节点: query_transform / retrieve / grade / prepare_fallback / post_process / generate
    - 边: entry → query_transform；retrieve → grade；
          grade conditional → {post_process, prepare_fallback}；
          prepare_fallback → query_transform；
          post_process → generate → END
    返回 compiled graph。"""
```

**测试规格（tests/integration/test_engine_graph.py）：用 mock LLM/embed/retrievers**

- **完整跑通**：构建图 + 注册 mock handlers + invoke({original_query: "..."}, configurable: GraphConfig().__dict__)，验证返回 state 含 answer 非空
- **嵌套 transforms 端到端**：transforms=["hyde", ["step_back", "rewrite"]] mock 调用，验证 retrieve 被调用 N 次（手算 N）
- **fallback 闭环**：grade mock 第一次返回 insufficient，prepare_fallback 切到 source="web"，retrieve mock for web 返回新 docs，第二次 grade sufficient → post_process → generate；验证 retrieval_rounds=2
- **多 retriever 合并**：retrievers=["vector", "bm25"] mock 各返回 docs，验证 grade 收到的 documents 合并后含 (source="vector", doc_id=*) 和 (source="bm25", doc_id=*) 两组
- **Overwrite([]) 正向**：fallback 第二轮 retrieve 后 documents 不含第一轮的（用 doc_id 区分）
- **Overwrite([]) 反面对比**：注入一个直接返回 `{"documents": []}`（不用 Overwrite）的伪 query_transform，验证第二轮 documents 错误地累加了第一轮（说明 reducer 行为符合预期；这一测试用于锁定 reducer 语义，未来 LangGraph 版本变更时若该测试不再触发错误累加，说明 reducer 行为变了需重新评估）

**验证标准：**
- `uv run pytest tests/integration/test_engine_graph.py -v` 全绿
- `uv run lint-imports` 通过

**审查要求：** code-quality-reviewer

**依赖：** 任务 18, 19, 20, 21
**风险：** 高（整图集成测试容易暴露各节点 bug）

---

### 阶段 6：API 层

#### 任务 23：IngestPipeline

**文件：**
- 新建 `src/fugue/api/ingest.py`
- 新建 `tests/integration/test_ingest_pipeline.py`

**`api/ingest.py` 必含定义：**

```python
import time
from pathlib import Path
from glob import glob

class IngestPipeline:
    def __init__(
        self,
        config,           # FugueConfig
        embedding_client,
        vector_store,
        bm25_provider,
    ) -> None: ...

    def run(
        self,
        sources: str | Path | Iterable[str | Path],
        *,
        show_progress: bool = True,
    ) -> IngestResult:
        """1. glob 展开（支持 './docs/*.pdf' 等模式）→ list[Path]
           2. 对每个 file：parser_registry.get(suffix or 'auto')(path) → list[ParsedDocument]
           3. chunker_registry.get(cfg.chunker)(parsed_docs, chunk_size, overlap) → list[Chunk]
           4. embedding_client.embed([c.content for c in chunks])
           5. vector_store.add(chunks, embeddings)
           6. bm25_provider.update(chunks)  ← 仅当 retrievers 包含 'bm25'
           返回 IngestResult。"""
```

**测试规格（tests/integration/test_ingest_pipeline.py）：mock embedding，真 chroma + bm25**

- **端到端 ingest**：3 个 fixtures 文件（md + txt + pdf），run() 后 IngestResult.num_chunks > 0，Chroma stats num_chunks 相符
- **glob 展开**：sources="tests/fixtures/*.md" 展开正确
- **bm25 update**：retrievers=["bm25"] 时 BM25 索引 search 能找到内容
- **bm25 skip**：retrievers=["vector"] 时 bm25_provider.update 不被调用（用 spy 验证）
- **chunk_id 稳定**：同一文件二次 ingest 后 num_chunks 不变（upsert 去重）
- **空 sources**：返回 IngestResult(num_documents=0, num_chunks=0)
- **未知扩展名**：抛 ValueError（auto_parser 错误透传）

**验证标准：**
- `uv run pytest tests/integration/test_ingest_pipeline.py -v` 全绿

**审查要求：** code-quality-reviewer

**依赖：** 任务 3, 4, 5, 7, 8, 10, 16, 17
**风险：** 中（端到端 ingest 集成 bug 风险）

---

#### 任务 24：RAG 主入口 class

**文件：**
- 新建 `src/fugue/api/rag.py`
- 修改 `src/fugue/__init__.py`（re-export RAG, FugueConfig 等）
- 修改 `src/fugue/handlers/__init__.py`（显式 import 各子目录触发注册副作用）
- 新建 `tests/integration/test_rag.py`

**`api/rag.py` 必含定义：**

```python
from contextlib import AbstractContextManager
from pathlib import Path

from fugue.config import FugueConfig, GraphConfig, load_yaml
from fugue.api.types import IngestResult, QueryResult, FugueConfigError, FugueRegistryError
from fugue.api.ingest import IngestPipeline
from fugue.engine.graph import build_rag_graph
from fugue.providers.llm import LLMClient
from fugue.providers.embedding import EmbeddingClient
from fugue.providers.vector_store.chroma import ChromaVectorStore
from fugue.providers.reranker.bge import BGEReranker
from fugue.providers.bm25 import BM25Provider
from fugue.registry import (
    discover_plugins, transform_registry, retriever_registry,
    processor_registry, grader_registry, generator_registry,
    parser_registry, chunker_registry,
)
from fugue.handlers.transforms import register_transforms
from fugue.handlers.retrievers import register_retrievers
from fugue.handlers.processors import register_processors
from fugue.handlers.generators import register_generators
# graders / parsers / chunkers 在各自 __init__.py import 时即自动注册（无依赖）

class RAG(AbstractContextManager):
    def __init__(
        self,
        config: FugueConfig | None = None,
        *,
        collection_name: str | None = None,
        env_file: str | Path | None = None,
    ) -> None:
        """1. 加载 .env（若 env_file 存在或 ./.env 存在）
           2. config 默认 FugueConfig()，collection_name 覆盖 cfg.ingest.collection_name
           3. 初始化 providers：LLMClient / EmbeddingClient / ChromaVectorStore /
              BGEReranker（lazy，首次用时加载）/ BM25Provider
           4. 注册内置 handlers：register_transforms/retrievers/processors/generators
           5. 调 discover_plugins() 扫描 entry_points
           6. _validate_config()：检查所有 config 引用的 handler 名 ∈ 对应 registry
           7. _bootstrap_bm25()：若 retrievers 含 "bm25"，分批从 vector_store.iter_chunks(batch_size=1000)
              拉取并 update BM25 索引（避免一次性 OOM）
           8. 构建 graph: self._graph = build_rag_graph()"""

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "RAG":
        """读 YAML → FugueConfig；overrides 可覆盖任意层级（dot path 如 graph.n_rewrites=5）。"""

    def ingest(self, sources, *, show_progress=True) -> IngestResult:
        """委托给 IngestPipeline.run。"""

    def query(self, question: str, *, graph_override=None) -> QueryResult:
        """构建 configurable（GraphConfig 字典） → graph.invoke({original_query: question, source: 'kb', retry_count: 0}) → 解析最终 state 为 QueryResult。
        retrieval_rounds = len(state['retrieval_history'])。"""

    def close(self) -> None:
        """显式释放 reranker / vector_store / llm_client / embedding_client / bm25。"""

    def __enter__(self) -> "RAG":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
```

**多 RAG 实例限制（0.1.0 已知限制，必须 docstring 警告 + README 标注）**：

由于 Registry 是全局单例 + 内置 handlers 在 `RAG.__init__` 时通过闭包绑定当前实例的 `LLMClient`/`EmbeddingClient`/`Reranker` 注册，**在同一进程实例化第二个 RAG（不同 provider 配置）会静默覆盖第一个实例的 transforms/generators 闭包**。导致第一个 RAG 实例继续使用第二个 LLM 客户端，结果不可预测。

MVP 处理：
- `RAG.__init__` 检测全局 Registry 中已存在内置 handler（被前一个 RAG 注册过），发 warning：`"⚠️ Multiple RAG instances in same process will share Registry singleton. Previous handlers will be overwritten."`
- README + RAG class docstring 明确"MVP 仅支持单 RAG 实例 per process；多实例场景请使用独立进程"
- P1 改造为 per-instance handler bag（fix 此限制）

**`_validate_config()` 必查项（spec 4.5 启动 fail-fast）：**
- LLM api_key 存在（env 或 config）：缺则抛 `FugueConfigError("OPENAI_API_KEY missing")`
- persist_dir 可写：`os.access(parent, os.W_OK)`
- 所有 `cfg.graph.transforms / retrievers / processors / grade_strategy / gen_mode` 引用的名字 ∈ 对应 registry
- 所有 `cfg.ingest.parser / chunker` 引用的名字 ∈ 对应 registry
- 所有 `cfg.graph.fallback_chain` 中源 ∈ retriever_registry（若非空）
- 失败时**一次性收集所有问题**，抛单个 `FugueConfigError` 含全部问题列表

**`handlers/__init__.py` 必须**：
```python
from fugue.handlers import transforms, retrievers, processors, graders, generators, parsers, chunkers
# 触发各子目录 __init__.py 副作用（graders/parsers/chunkers 在导入时即注册）
```

**测试规格（tests/integration/test_rag.py）：mock LLM/Embedding，真 Chroma + 真 BM25**
- **from_yaml 基础**：写一个临时 yaml，调 from_yaml，验证 RAG 实例化成功且 config 正确
- **with 上下文管理**：`with RAG() as rag: ...` 退出后 close 被调用（spy）
- **ingest + query 端到端**（mock LLM/embed）：ingest 3 个 docs → query "..." → 返回 QueryResult.answer 非空，ranked_documents 非空
- **fail-fast 校验：未注册 handler**：config.graph.retrievers=["nonexistent"] → from_yaml 抛 `FugueConfigError` 含 "nonexistent" 和可用列表
- **fail-fast 校验：多 handler 错**：多个未注册项，验证抛单个异常含所有问题
- **fail-fast 校验：缺 api_key**：env 无 OPENAI_API_KEY + config 无 → 抛 FugueConfigError
- **graph_override 覆盖**：query("...", graph_override=GraphConfig(gen_mode="citation"))，验证 citation generator 被调用
- **collection_name 覆盖**：实例化时传入参数覆盖 yaml 中的值
- **bm25 启动重建**：retrievers 含 "bm25" 时 RAG.__init__ 后 BM25 索引含已 ingest 的 chunks（先 ingest 再重建实例验证）

**验证标准：**
- `uv run pytest tests/integration/test_rag.py -v` 全绿，覆盖率 ≥ 90%
- `uv run lint-imports` 通过

**审查要求：** code-quality-reviewer + security-reviewer（涉及 API key 读取 / 文件路径，需要 security review）

**依赖：** 任务 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22, 23
**风险：** 高（最大集成点，可能暴露前面任务的 bug）

---

### 阶段 7：Server 层

#### 任务 25：FastAPI app + endpoints

**文件：**
- 新建 `src/fugue/server/__init__.py`
- 新建 `src/fugue/server/app.py`
- 新建 `src/fugue/server/endpoints.py`
- 新建 `tests/integration/test_server.py`

**`endpoints.py` 必含定义：**

```python
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fugue import RAG

class QueryRequest(BaseModel):
    question: str

class IngestRequest(BaseModel):
    paths: list[str]

def create_endpoints(app: FastAPI, rag: RAG) -> None:
    @app.post("/query")
    def query(req: QueryRequest):
        try:
            result = rag.query(req.question)
            return result.__dict__  # dataclass → dict
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ingest")
    def ingest(req: IngestRequest):
        try:
            result = rag.ingest(req.paths)
            return result.__dict__
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    def health():
        return {"status": "ok"}
```

**`app.py` 必含定义**（MVP 采用同步实例化 + `atexit` 清理，不用 FastAPI lifespan：lifespan 在多 worker 下行为复杂，且 MVP 强制单 worker，简化方案足够）：

```python
import atexit
from pathlib import Path
from fastapi import FastAPI

from fugue import RAG
from fugue.server.endpoints import create_endpoints

def create_app(config_path: str | Path) -> FastAPI:
    rag = RAG.from_yaml(config_path)
    app = FastAPI(title="Fugue", version="0.1.0")
    create_endpoints(app, rag)
    atexit.register(rag.close)
    return app
```

**测试规格（tests/integration/test_server.py）：用 fastapi.testclient + mock RAG**
- **/query**：POST 含 question，返回 200 + answer 字段
- **/ingest**：POST 含 paths，返回 200 + num_chunks 字段
- **/health**：GET 返回 `{"status": "ok"}`
- **/query 异常**：mock RAG.query 抛 FugueError，返回 500 + detail
- **请求体校验**：POST /query 空 body 返回 422（FastAPI/Pydantic 自动）

**验证标准：**
- `uv run pytest tests/integration/test_server.py -v` 全绿
- `uv run mypy src/fugue/server/` 0 错误

**审查要求：** code-quality-reviewer + security-reviewer（HTTP endpoint，需检查 detail 是否泄露内部信息）

**依赖：** 任务 24
**风险：** 中

---

#### 任务 26：CLI `fugue serve`

**文件：**
- 新建 `src/fugue/server/cli.py`
- 新建 `tests/unit/test_server/test_cli.py`

**`cli.py` 必含定义：**

```python
import argparse
import logging
import uvicorn

logger = logging.getLogger("fugue.server")

def main() -> None:
    parser = argparse.ArgumentParser(prog="fugue")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Start Fugue REST server")
    serve.add_argument("--config", required=True, help="Path to YAML config")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    # 不暴露 --workers：MVP 强制 workers=1（多 worker 下 Chroma 本地文件锁会争用，BM25 重复重建）

    args = parser.parse_args()

    if args.cmd == "serve":
        from fugue.server.app import create_app
        app = create_app(args.config)
        # 启动期 WARNING：MVP 无 auth + 单 worker 限制
        logger.warning(
            "⚠️ Fugue 0.x: no authentication, single-worker only. "
            "Deploy to trusted networks only. workers=1 enforced."
        )
        uvicorn.run(app, host=args.host, port=args.port, workers=1)
```

**`pyproject.toml` 已配置 `[project.scripts] fugue = "fugue.server.cli:main"`（任务 1）。**

**测试规格（tests/unit/test_server/test_cli.py）：**
- **argparse 解析**：模拟 `fugue serve --config x.yaml --port 9000`，验证 args.config="x.yaml" 等
- **缺 --config**：argparse 抛 SystemExit
- **未知子命令**：抛 SystemExit
- **mock uvicorn.run 验证**：argparse 解析后调用 uvicorn.run(app, host=, port=)

**验证标准：**
- `uv run pytest tests/unit/test_server/test_cli.py -v` 全绿
- `uv run fugue serve --help` 输出帮助文档

**审查要求：** code-quality-reviewer

**依赖：** 任务 25
**风险：** 低

---

### 阶段 8：E2E 与文档

#### 任务 27：E2E 测试套件

**文件：**
- 新建 `tests/e2e/__init__.py`（空）
- 新建 `tests/e2e/conftest.py`（含 `openai_key` fixture 自动 skip）
- 新建 `tests/e2e/test_e2e_basic.py`（基础 ingest + query）
- 新建 `tests/e2e/test_e2e_multipath.py`（多路 + rrf + rerank）
- 新建 `tests/e2e/test_e2e_server.py`（fugue serve REST）
- 新建 `tests/fixtures/e2e/`（3-5 篇 markdown + 1 PDF，**至少 1 篇中文 markdown**——用于验证 BM25 中文限制的端到端可见性，且匹配 README Quick Start 的中文示例）

**`conftest.py` 必含：**

```python
import os
import pytest

pytest_plugins = []

def pytest_collection_modifyitems(config, items):
    """所有 tests/e2e/ 下的测试自动加 e2e marker。"""
    for item in items:
        if "tests/e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)

@pytest.fixture(scope="session")
def openai_key():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not set")
    return key
```

**E2E 场景（每个 < $0.02/run）：**

- **test_e2e_basic.py**：
  - ingest 3 个 markdown，query 一个内容相关问题，验证 answer 含 fixtures 中的关键词，ranked_documents 数量 = top_k=3
- **test_e2e_multipath.py**：
  - transforms=["rewrite"] + retrievers=["vector","bm25"] + processors=["rrf","rerank"] + gen_mode="citation"
  - 验证 answer 含 `[1]` 编号引用
  - 验证 rewritten_queries 长度 > 1
  - **中文场景**：ingest 1 篇中文 markdown + 用中文 query，验证 answer 非空（即使 BM25 召回为零，向量路径仍能工作，验证端到端不破）
- **test_e2e_server.py**：
  - 用 `httpx.Client` + `TestClient` 启动真 server
  - POST /ingest → 200 + num_chunks > 0
  - POST /query → 200 + answer 非空
  - GET /health → `{"status":"ok"}`

**测试规格：上述场景。**

**验证标准：**
- 设置 OPENAI_API_KEY 后 `uv run pytest tests/e2e/ -m e2e -v` 全绿
- 未设置 OPENAI_API_KEY 时 `uv run pytest -m 'not e2e'` 跳过 e2e（默认行为）
- 单次完整 e2e 跑成本 < $0.05（用 gpt-4o-mini）

**审查要求：** code-quality-reviewer

**依赖：** 任务 24, 25, 26
**风险：** 中（e2e 依赖外部 API 稳定性）

---

#### 任务 28：README 完整版（MVP 唯一必交付文档）

**文件：**
- 修改 `README.md`（替换任务 1 的雏形为完整版）

**说明：** `docs/getting-started.md` 与 `docs/extending.md` 推迟到 V1.x。MVP 唯一必交付文档是 README，已涵盖核心信息。

**README.md 必含章节：**
1. **Tagline**：「Configuration-driven RAG. Topology stays, behavior plugs in.」
2. **Install**：`pip install fugue[all]`
3. **Quick start**：spec 4.8 中的端到端示例（YAML + Python，**含中文示例**）
4. **Why Fugue?**：三个差异化卖点（配置即行为 / 图拓扑稳定 / 无限可插拔）
5. **Plugin example**：30 行写一个自定义 retriever + entry_points pyproject.toml 片段
6. **Status & Compatibility**：「0.x — APIs may change. Stable contracts: `GraphConfig` field names / Registry API / `RAG.from_yaml`/`ingest`/`query`. **MVP supports single RAG instance per process; multi-instance会静默覆盖 Registry，P1 修复**。」
7. **Security**：「`fugue serve` MVP has no authentication, single-worker only. Deploy to trusted networks only.」
8. **Thread safety**：「`RAG.query()` 线程安全；`RAG.ingest()` 不与 query 并发；插件注册截止于第一次 `RAG()` 实例化。」
9. **Known limitations (MVP)**：
   - BM25 用空白分词，中文召回为 0（P1 加 jieba）
   - 大语料（>50k chunks）启动 BM25 重建可能较慢
   - 多 worker / 多 RAG 实例不支持
10. **License**：MIT
11. **Links**：spec / fuge_plan / 竞品对比

**测试规格：**
- README 中所有代码示例**实际可运行**（用 doctest 或手动验证）
- 文档中 YAML 示例符合 Pydantic schema（用 `load_yaml` 加载不报错）

**验证标准：**
- 文档中所有 YAML 用 `uv run python -c "from fugue.config import load_yaml; load_yaml('...')"` 解析成功
- README 第一行的 install 命令实际能跑通

**审查要求：** code-quality-reviewer（文档准确性）

**依赖：** 任务 27（功能完整后写文档）
**风险：** 低

---

## 测试策略

### 单元测试
- 覆盖：types / registry / config / providers / handlers / engine 节点 / runtime
- mock 所有外部依赖（LLM API、Embedding API、文件系统的真实读写除外）
- 目标覆盖率：核心模块（engine/、config.py、registry.py、api/）≥ 90%；其余 ≥ 85%

### 集成测试
- 覆盖：Chroma 真实交互、Ingest 完整流水线、整图跑通（mock LLM/Embedding）、RAG class 主入口、Server endpoints
- 用 `tmp_path` 隔离 Chroma；用 syrupy 快照 prompt 模板

### 端到端测试（nightly only）
- 真 OpenAI 兼容 API（gpt-4o-mini）+ 真 Chroma + 真 BGE 模型
- 覆盖：基础 ingest+query / 多路融合+citation / fugue serve REST
- 单次跑成本 < $0.05

## E2E 稳定性要求
- 所有 E2E 测试用 `tmp_path` 创建独立 Chroma collection，测试结束清理
- 禁止依赖其他测试的残留 chunks
- 不稳定测试用 `pytest.mark.flaky(retries=2)`（仅 e2e）
- 用 `httpx.Client` 而非 `requests`（与 FastAPI 同栈）
- 显式 `httpx.Timeout(connect=5, read=30)`

## 风险与缓解

- **风险**：LangGraph 0.3.x API 变化导致 engine/ 失败
  - 缓解：pyproject.toml `langgraph>=0.3,<0.4`；CI 矩阵跑两个 patch 版本；import-linter 隔离影响范围

- **风险**：嵌套 transforms（并行+串联）实现 bug 影响 fan-out
  - 缓解：任务 11 + 任务 19 + 任务 22 三层验证；syrupy 快照 prompt；专门 integration test 覆盖嵌套场景

- **风险**：BM25 启动重建在大语料下启动慢
  - 缓解：MVP 不优化，文档标注万级 chunks 内可控；P1 加 pickle 持久化

- **风险**：覆盖率 90% 门槛过严，部分边界 case 难覆盖
  - 缓解：核心模块严格 90%；其余模块 85%；用 `# pragma: no cover` 标记防御性分支

- **风险**：BGE-Reranker 模型下载在 CI 慢（600MB）
  - 缓解：CI 使用 `actions/cache` 缓存 `~/.cache/huggingface`；e2e 在 nightly 跑不影响 PR

- **风险**：Server 无 auth 被误用于公网
  - 缓解：README/Quick Start 明确警告；启动日志打 WARNING `⚠️ No authentication. Trusted networks only.`

- **风险**：多 RAG 实例在同进程导致 Registry 闭包覆盖（Reviewer F1）
  - 缓解：`RAG.__init__` 检测已注册时发 warning；README 标注限制；P1 改 per-instance handler bag

- **风险**：BM25 全量重建在大语料下启动慢/OOM（Reviewer F3）
  - 缓解：`iter_chunks(batch_size=1000)` 分批拉取避免 OOM；性能基线测试覆盖 10k chunks；超大语料标注 P1 优化（pickle 持久化）

- **风险**：LangGraph `Overwrite` API 在 0.3.x patch 版本间不稳定（Reviewer F2）
  - 缓解：版本约束 `>=0.3,<0.4` + 反面对比测试（任务 22）锁定 reducer 语义；CI 跑两个 patch 版本

- **风险**：uvicorn 多 worker 与 Chroma PersistentClient 不兼容（Reviewer F5）
  - 缓解：CLI 强制 workers=1（不暴露 `--workers` 参数）；README 明确多 worker 部署需独立进程

- **风险**：BM25 空白分词对中文召回率为零（Reviewer F6）
  - 缓解：xfail 测试用例可见；E2E 含中文 fixture；README "Known limitations" 节明确标注；P1 加 jieba

- **风险**：post_process 跨多轮 fallback 后文档量爆炸（Reviewer F7）
  - 缓解：`MAX_DOCS_BEFORE_PROCESS=1000` 防御性上界，超过则按 score 降序截断

## 验收标准

- [ ] 所有 28 个任务通过 TDD + Quality Gate + Code Review 三道门控
- [ ] `uv run pytest -m 'not e2e' --cov-fail-under=90` 通过
- [ ] `uv run ruff check src tests` 0 错误
- [ ] `uv run mypy src/fugue` 0 错误
- [ ] `uv run lint-imports --config .importlinter` 通过
- [ ] `uv run pre-commit run --all-files` 通过
- [ ] 设置 OPENAI_API_KEY 后 `uv run pytest -m e2e` 通过
- [ ] `pip install -e .[all] && fugue serve --config examples/config.yaml` 启动成功
- [ ] README 的 Quick Start 代码块可直接运行通过
- [ ] spec 中所有 P0 功能均有对应代码 + 测试

---

## 任务依赖图（简版）

```
任务 1 (脚手架)
  ├─► 任务 2 (CI)
  ├─► 任务 3 (types)
  │    ├─► 任务 4 (Registry)
  │    └─► 任务 5 (Config)
  │
  ├─► 任务 6 (LLMClient) ┐
  ├─► 任务 7 (Embedding) ┤
  ├─► 任务 8 (Chroma)    ┤
  ├─► 任务 9 (Reranker)  ┤
  └─► 任务 10 (BM25)     ┘
        │
        ├─► 任务 11 (Transforms)  ─┐
        ├─► 任务 12 (Retrievers)  ─┤
        ├─► 任务 13 (Processors)  ─┤
        ├─► 任务 14 (Graders)     ─┤
        ├─► 任务 15 (Generators)  ─┤
        ├─► 任务 16 (Parsers)     ─┤
        └─► 任务 17 (Chunkers)    ─┘
              │
              ├─► 任务 18 (State+Runtime)
              │     ├─► 任务 19 (qt/retrieve)
              │     ├─► 任务 20 (grade/fallback)
              │     └─► 任务 21 (post/generate)
              │           │
              │           └─► 任务 22 (build_rag_graph)
              │
              └─► 任务 23 (IngestPipeline)
                    │
                    └─► 任务 24 (RAG class) ← 集成最大点
                          │
                          ├─► 任务 25 (FastAPI)
                          │     └─► 任务 26 (CLI)
                          │           └─► 任务 27 (E2E)
                          │                 └─► 任务 28 (Docs)
```

可并行：
- 任务 6-10 可全并行（providers 互相独立）
- 任务 11-17 可全并行（handlers 互相独立）
- 任务 19-21 可全并行（engine nodes 互相独立）
