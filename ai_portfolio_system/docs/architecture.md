# 架构设计文档

## 1. 项目范围

本次实现一个可本地运行的「AI 求职项目组合」演示系统，包含三个核心 Agent：

- **RAG 求职知识库助手**：回答求职相关问题，并返回引用来源。
- **SQL 数据分析 Agent**：根据自然语言生成安全 SQL，展示结果并解释。
- **简历/JD 匹配助手**：输入简历与 JD，输出技能缺口、补课优先级、项目改写建议。

并附加：

- **Evaluator Agent**：对任意 Agent 输出进行多维评分，给出改进建议。
- **Self-Attention 迭代器**：基于 Evaluator 反馈，最多迭代 3 轮优化 Prompt 与检索参数。

## 2. 分层设计

### 2.1 Core 层（基础设施）

职责：提供 LLM、Embeddings、基础工具、配置。

| 模块 | 文件 | 说明 |
| --- | --- | --- |
| 配置 | `src/core/config.py` | 读取 `config.yaml`；环境变量覆盖；LLM 模式选择。 |
| LLM | `src/core/llm.py` | 统一接口 `call()`；支持 OpenAI、Ollama、Mock。 |
| Embeddings | `src/core/embeddings.py` | 优先 sentence-transformers；无 torch 时自动回退到 scikit-learn TF-IDF。 |
| 工具 | `src/core/tools.py` | 注册 SQL、Web 搜索、文件读取等工具。 |
| 日志 | `src/core/logging.py` | 结构化日志，方便观察每轮迭代。 |

LLM 接口设计：

```python
class LLM(ABC):
    def chat(self, messages: list[dict], response_format: BaseModel | None = None) -> str | BaseModel:
        ...
```

### 2.2 Retrieval 层（多路召回 + 约束）

职责：把不同来源的信息召回并过滤，保证质量与可控。

| 模块 | 文件 | 说明 |
| --- | --- | --- |
| 向量检索 | `src/retrieval/vector_store.py` | 默认 SQLite + numpy 持久化；设置 `VECTOR_STORE_BACKEND=chroma` 可切换 Chroma。 |
| 向量检索 | `src/retrieval/vector_search.py` | 语义检索 + 重排序 + 返回带引用的 chunk。 |
| Web 检索 | `src/retrieval/web_search.py` | DuckDuckGo 搜索；返回标题/摘要/链接/时间。 |
| 约束引擎 | `src/retrieval/constraints.py` | 域名白名单/黑名单、时间窗、相关度阈值、最大数量。 |

约束规则：

- 白名单：只允许 `github.com`, `zhihu.com`, `juejin.cn`, `cnblogs.com` 等。
- 黑名单：屏蔽广告、低质站点。
- 时间窗：默认只取最近 2 年内的内容。
- 相关度阈值：余弦相似度 ≥ 0.5。
- 最大结果数：每路最多 5 条，合并后最多 8 条。

默认向量存储使用 SQLite + numpy，便于本地无 GPU 快速运行。如需使用 Chroma，安装 `requirements-full.txt` 并设置环境变量 `VECTOR_STORE_BACKEND=chroma`。

### 2.3 Agent 层

职责：面向用户任务的垂直 Agent。

| Agent | 文件 | 能力 |
| --- | --- | --- |
| BaseAgent | `src/agents/base.py` | 共享：LLM 调用、检索器引用、结果打包。 |
| RAGAgent | `src/agents/rag_agent.py` | 接收问题 → 多路检索 → 回答并引用来源。 |
| SQLAgent | `src/agents/sql_agent.py` | 接收问题 → 生成 SQL → 安全执行 → 解释结果。 |
| MatchAgent | `src/agents/match_agent.py` | 解析简历/JD → 结构化输出 → 匹配建议。 |
| EvaluatorAgent | `src/agents/evaluator_agent.py` | 对任意输出评分（准确性、完整性、安全性、引用质量）。 |

### 2.4 Orchestrator 层（多 Agent 路由）

职责：识别用户意图，调度 Agent，组合结果，管理对话状态。

- 意图分类：
  - `knowledge` → RAGAgent
  - `sql` → SQLAgent
  - `match` → MatchAgent
  - `evaluate` → EvaluatorAgent（显式要求评估时）
- 结果组合：如果用户同时请求「查库 + 匹配」，Orchestrator 串行调用多个 Agent 并合并输出。
- 状态管理：对话历史保存在 SQLite 表中（`conversations`）。

### 2.5 Tuning 层（Self-Attention 迭代）

职责：让 Agent 在本地工程约束下「自我优化」。

```
输入: query + 初始回答
  ↓
Evaluator 评分
  ↓
如果分数 < threshold 且轮次 < 3:
  调整 prompt 模板 / 检索 top_k / 重排序策略
  重新生成回答
  ↓
Evaluator 评分
  ↓
输出: 最佳回答 + 迭代历史
```

可调参数：

- `prompt_template`：在模板中插入上一轮评估反馈。
- `retrieval_top_k`：逐步增加召回数量。
- `temperature`：根据问题类型调整创造性/确定性。
- `system_role`：根据评估强化角色设定。

这不是真正的模型权重微调，而是工程上可落地的「提示词与检索策略自迭代」，符合项目难度要求。

### 2.6 UI 层（Codex 风格弹窗）

职责：提供类似参考图的终端交互界面。

- 使用 `textual` 构建 TUI。
- 组件：Header、聊天记录区、输入框、模型信息面板、日志/状态条。
- 支持 `/agent`, `/model`, `/evaluate`, `/iter` 等命令。

## 3. 数据流

以用户输入「我想找一份 AI 产品经理实习，需要补哪些技能？」为例：

1. UI 接收输入，交给 Orchestrator。
2. Orchestrator 识别意图为 `knowledge`。
3. RAGAgent 执行：
   - 对 query 做嵌入，向量检索 Chroma 中的 JD、面试笔记、知识点。
   - 同时调用 WebRetriever，检索公开求职经验，经约束引擎过滤。
   - 合并检索结果，按相关度重排序。
   - 使用 Prompt 模板生成带引用回答。
4. EvaluatorAgent 对回答评分。
5. 如果评分不足，Tuning 层启动 Self-Attention 迭代，最多 3 轮。
6. Orchestrator 返回最终回答 + 元信息（引用、评分、迭代次数）给 UI。
7. UI 渲染。

## 4. AI 检查 AI 机制

EvaluatorAgent 的评分维度：

- 准确性（Accuracy）：事实是否正确。
- 完整性（Completeness）：是否覆盖关键要点。
- 引用质量（Citation）：来源是否相关、可追溯。
- 安全性（Safety）：SQL 是否安全、是否泄露隐私、是否有害内容。
- 一致性（Consistency）：同一问题多次回答是否一致。

输出结构：

```json
{
  "score": 0.82,
  "dimensions": {
    "accuracy": 0.85,
    "completeness": 0.80,
    "citation": 0.90,
    "safety": 1.0,
    "consistency": 0.75
  },
  "issues": ["缺少量化指标", "第二个引用未说明日期"],
  "suggestions": ["补充具体技能清单", "为 Web 来源添加时间戳"],
  "better_answer": "..."
}
```

## 5. 安全与约束

- SQL Agent：只读连接；拦截 `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`。
- 表白名单：只允许查询已注册的业务表。
- Web 检索：经约束引擎过滤后才能进入上下文。
- 文件系统：Agent 只能读取 `data/` 目录下文件。

## 6. 运行方式

```bash
# 1. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. 安装最小依赖（无需 GPU / API key，使用 SQLite 向量库 + TF-IDF）
pip install -r requirements.txt

# 2b. （可选）安装完整生产依赖：Chroma + sentence-transformers
pip install -r requirements-full.txt

# 3. 准备数据
python -m src.scripts.ingest

# 4. 启动 Codex 风格弹窗
python -m src.main tui

# 5. 命令行交互模式
python -m src.main chat
```

可选环境变量：

- `LLM_PROVIDER=mock`（默认，无需 key）
- `LLM_PROVIDER=openai` + `OPENAI_API_KEY=...`
- `LLM_PROVIDER=ollama` + `OLLAMA_MODEL=qwen2.5`
- `VECTOR_STORE_BACKEND=chroma` 切换向量存储后端
