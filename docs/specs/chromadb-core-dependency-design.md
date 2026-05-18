---
feature: chromadb-core-dependency
spec: docs/specs/chromadb-core-dependency-design.md
routing: Development Workflow
---

# chromadb 提为核心依赖（chromadb-core-dependency）

## 1. 问题陈述

`RAG.__init__`（`src/fugue/api/rag.py:155`）无条件构造
`ChromaVectorStore`，chromadb 是**默认且唯一**的向量存储实现，属强制
运行时依赖。但它被放在 `[project.optional-dependencies].chroma`
可选 extra 里：

```toml
[project.optional-dependencies]
chroma = ["chromadb>=0.5"]
all = ["fugue[server,bge,chroma,pdf]"]
```

裸装 `pip install fugue`（不带任何 extra）缺 chromadb → 即使
lazy-optional-imports（问题 2）修复后 `import fugue` 不崩，
`RAG()` 仍会因 chromadb 缺失而失败。"默认路径却需可选 extra"
是定位矛盾。

**与问题 2 的区别**：pdf/bge 是真·可选（非默认路径，已懒加载）；
chromadb 是事实强制（默认路径必用），正解是提为主依赖，**不**懒加载。

## 2. 目标与范围

**目标**：chromadb 进 `[project].dependencies`，任何安装方式都自带；
裸装 `pip install fugue` 即可正常 `RAG()`。

**范围**：仅依赖清单与文档调整。

**明确不做**：

- 不改任何 Python 源码 / 业务逻辑（向量存储行为不变）
- 不做向量存储可插拔抽象（超范围，未来另议）
- 不动 pdf/bge 懒加载（问题 2，已在独立 PR）
- 不改 PyPI 包名（问题 1，独立决策）

## 3. 技术设计（L1 轻量）

唯一方案——依赖重分类，无需多方案比选。

### 3.1 改动点

| 文件 | 改动 |
|---|---|
| `pyproject.toml` | `[project].dependencies` 增加 `chromadb>=0.5`；`[project.optional-dependencies].chroma` 由 `["chromadb>=0.5"]` 改为 `[]`（空，向后兼容占位）；`[all]` 保留 `chroma`（引用空 extra，无害） |
| `uv.lock` | `uv lock` 机械重生成（chromadb 在 fugue 的依赖归类从可选转主依赖；包版本不变，已在锁内） |
| `README.md` | 修正 `pip install "fugue[chroma]"` 措辞：chromadb 现为默认安装，`[chroma]` extra 保留仅为向后兼容（no-op） |

### 3.2 关键设计决策

- **chroma extra 保留为空**：`chroma = []` 而非删除。已有用户/脚本/CI
  的 `pip install 'fugue[chroma]'` 仍合法、不报错，只是不再额外装包。
  零破坏性（用户已确认选此方案）。
- **不懒加载 chromadb**：与 pdf/bge 相反——chromadb 是默认路径必经，
  懒加载只会把"缺失"从 import 期推迟到 RAG() 期，不解决根本问题。
  提为主依赖才是正解。

### 3.3 Error Handling / Testing

- 无业务逻辑、错误路径变更（向量存储行为完全不变）
- 验证 = 全量测试套件绿 + `uv lock --check` 一致 + 隔离环境裸装
  （无 extra）确认 chromadb 随主依赖装上、`import fugue` + 构造
  `RAG()`（不触发网络的最小构造）可用
- 不新增功能测试：这是依赖清单元数据变更，行为不变；为 pyproject
  元数据写断言无工程价值

## 4. 验收标准

- [ ] `pyproject.toml` `[project].dependencies` 含 `chromadb>=0.5`
- [ ] `[project.optional-dependencies].chroma == []`，`[all]` 仍含 `chroma`
- [ ] `uv lock --check` 一致；`uv.lock` 中 fugue 的 requires-dist 反映 chromadb 为主依赖
- [ ] 隔离环境 `pip install <wheel>`（无 extra）后 chromadb 已安装、
      `import fugue` 成功
- [ ] `pip install 'fugue[chroma]'` 不报 "no extra named chroma"
- [ ] 既有全量测试套件零回归（非 e2e）；mypy src 零错误
- [ ] README 安装说明不再误导

## 5. 路由决策

后续工作流：**Development Workflow**
理由：纯后端依赖配置 + 文档，无 UI。
