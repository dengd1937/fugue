---
feature: lazy-optional-imports
spec: docs/specs/lazy-optional-imports-design.md
routing: Development Workflow
---

# 可选依赖懒加载（lazy-optional-imports）

## 1. 问题陈述

Ragline 的导入副作用注册链在模块顶层 eager import 可选依赖，导致
README 承诺的"最小 / 分级 `pip install`"实际不可用。

崩溃链路（实测）：

- **PDF**：`import ragline` → `api/rag.py` → `api/ingest.py:9` →
  `handlers/__init__.py:8` → `handlers/parsers/__init__.py:7`
  `from ragline.handlers.parsers.pdf import pdf_parser` →
  `parsers/pdf.py:5` `import pypdf` → `ModuleNotFoundError`（未装 `[pdf]`）
- **BGE**：`processors/__init__.py:3` → `rerank.py:7`
  `from ragline.providers.reranker.base import Reranker` →
  触发 `providers/reranker/__init__.py:4`
  `from ragline.providers.reranker.bge import BGEReranker` →
  `reranker/bge.py:5` `from FlagEmbedding import FlagReranker` →
  `ModuleNotFoundError`（未装 `[bge]`）

净结果：裸装 `pip install ragline` 后 `import ragline` 直接失败，
与 README 的分级安装承诺不符。

## 2. 目标与范围

**目标**：未安装 `[pdf]` / `[bge]` 时：

- `import ragline` 成功
- `register_parsers()` / `register_processors()` 注册成功（handler 可注册）
- 仅当**真正调用** PDF 解析或构造 BGE reranker 时，抛出清晰可操作的
  `ImportError`，提示 `pip install 'ragline[xxx]'`

**范围**：仅 `pdf`（pypdf）+ `bge`（FlagEmbedding）两个可选依赖点。

**明确不做**：

- `chromadb`（实为强制依赖，归属独立的"问题 3"，本次不动）
- `server` 的 fastapi/uvicorn（仅 `ragline serve` 时加载，本就不在
  `import ragline` 链中，无需改）
- PyPI 包名冲突（独立的"问题 1"，需单独的产品决策）

## 3. 技术设计（方案 A，L1 轻量）

### 3.1 关键认知

追踪确认实际只需改 **2 个叶子文件**：

- `pdf.py` 移除顶层 `import pypdf` 后，`parsers/__init__.py:7`
  导入 `pdf_parser` 变廉价 → `parsers/__init__.py` **无需改**
- `bge.py` 移除顶层 `from FlagEmbedding import` 后，
  `reranker/__init__.py:4` 的 re-export 变廉价 →
  `reranker/__init__.py`、`rerank.py`、`_LazyReranker` **均无需改**

唯一真正构造 `BGEReranker` 的是 `api/rag.py` 的
`_LazyReranker._ensure_loaded()`（首次 `rerank()` 调用时），
与"首次调用招错"语义天然吻合。

### 3.2 共享 helper

新增 `src/ragline/_optional.py`：

```python
def require(module_name: str, *, extra: str) -> Any:
    """导入可选依赖模块；缺失时抛清晰可操作的 ImportError。

    返回类型为 Any（非 ModuleType）：可选依赖无类型 stub，
    调用方需在返回值上做动态属性访问，ModuleType 会触发
    mypy attr-defined。
    """
```

- 成功：返回已导入模块对象（静态类型 `Any`）
- 失败（`ModuleNotFoundError` / `ImportError`）：重抛
  `ImportError`，消息含模块名 + `pip install 'ragline[<extra>]'`，
  并 `from e` 保留原始 traceback

### 3.3 改动点

| 文件 | 改动 |
|---|---|
| `src/ragline/_optional.py` | **新增** `require()` helper |
| `src/ragline/handlers/parsers/pdf.py` | 删顶层 `import pypdf`；`pdf_parser()` 首行 `pypdf = require("pypdf", extra="pdf")` |
| `src/ragline/providers/reranker/bge.py` | 删顶层 `from FlagEmbedding import FlagReranker`；`BGEReranker.__init__()` 内 `FlagReranker = getattr(require("FlagEmbedding", extra="bge"), "FlagReranker")` |

`bge.py:13` 既有的 `import torch`（函数内懒加载）保持不变，作为同类先例。

### 3.4 错误语义

- 错误类型：内置 `ImportError`（生态惯例；用户对缺失可选依赖即预期 ImportError）
- 招错时机：首次调用（`pdf_parser()` 调用时 / `BGEReranker()` 构造时，
  后者经 `_LazyReranker` 即首次 `rerank()`）
- 消息示例：
  `PDF 解析需要可选依赖 'pypdf'。请运行: pip install 'ragline[pdf]'`

### Implementation Deviations

#### 2026-05-17 — bge.py 取 FlagReranker 方式
**偏差章节**：3.3 改动点
**原方案**：`FlagReranker = getattr(require("FlagEmbedding", extra="bge"), "FlagReranker")`
**实际实现**：`_flagembedding = require("FlagEmbedding", extra="bge"); flag_reranker_cls = _flagembedding.FlagReranker`（普通属性访问，snake_case 局部名）
**原因**：`getattr(obj, "常量字面量")` 触发 ruff B009 需 `# noqa` 抑制；而
`require()` 已返回 `Any`，普通属性访问对 mypy 同样干净（无 attr-defined），
运行时行为与 `getattr` 完全等价。改用属性访问消除不必要的 lint 抑制，
变量名用 snake_case `flag_reranker_cls` 规避 N806。getattr 在此零收益。

- **helper 单测** `tests/unit/test_optional.py`：`require()` 成功返回 +
  缺失抛带 extra 提示的 `ImportError`
- **pdf/bge 单测**（追加到既有 `test_parsers.py`/`test_reranker.py`）：
  通过 `patch("<module>.require")` 接缝模拟缺失，**不碰全局 sys.modules**
  （避免破坏测试文件自身顶层 import）。既有 `test_reranker.py` 的 10 处
  `patch("...bge.FlagReranker")` 须机械迁移到 `patch("...bge.require")`
- **跨模块回归** `tests/integration/test_optional_deps.py`：用
  **子进程隔离**（`subprocess.run([sys.executable,"-c",...])`，脚本内
  `sys.modules["pypdf"]=None`、`sys.modules["FlagEmbedding"]=None` 后
  `import ragline`），断言子进程 returncode==0；父进程零 sys.modules 操作，
  杜绝进程内 reload 对已收集测试的污染

不依赖真实卸载 pypdf/FlagEmbedding（本地/CI 装了 `[all]`），
用 import 拦截 / 子进程哨兵模拟缺失。

## 5. 验收标准

- [ ] 隔离环境裸装 wheel（无 `[pdf]`/`[bge]`）后 `import ragline` 成功
- [ ] 该环境 `register_parsers()` / `register_processors()` 成功
- [ ] 该环境调用 PDF 解析 / 构造 BGE 抛含 `pip install 'ragline[...]'` 的 ImportError
- [ ] 装 `[pdf]`/`[bge]` 后行为与改动前完全一致（无回归）
- [ ] 新回归测试通过；既有测试套件全绿
- [ ] mypy src 零错误

## 6. 路由决策

后续工作流：**Development Workflow**
理由：纯后端代码重构 + 测试，无 UI。
