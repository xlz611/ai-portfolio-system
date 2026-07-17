# AI 求职项目组合系统

复刻参考图「RAG + SQL Agent + 简历/JD 匹配」项目组合，并包装成可运行的 Codex 风格终端弹窗。

## 目标

1. 运行出类似 OpenAI Codex 的命令行弹窗。
2. 覆盖技术栈：Prompt Engineering、RAG、RAG 检索增强、向量数据库、Function Calling、Agent、SQL Agent、LangChain、Fine-tuning / Self-Attention、Python。
3. 以多 Agent 体系构建：RAG 求职知识库助手、SQL 数据分析 Agent、简历/JD 匹配助手、AI 评估器。
4. 检索采用多路召回（向量库 + Web），并附加检索约束（白名单、黑名单、时间窗、相关度阈值、最大结果数）。
5. 内置微调/迭代机制：Self-Attention 闭环（最多 3 轮），让 Agent 基于评估反馈自我修正 Prompt 与检索策略。
6. 先定框架，再分层实现，每步可验证。
7. 以 AI 检查 AI：评估器 Agent 对输出自动打分、找缺陷、给修正建议。

## 技术栈

| 需求 | 实现 |
| --- | --- |
| Prompt / 提示词工程 | `src/prompts/*.yaml` + 模板引擎 + 动态优化 |
| RAG / RAG 检索增强 | SQLite + numpy 向量检索（可选 Chroma）+ 引用溯源 |
| 向量数据库 | SQLite 向量存储（默认）；可切换 Chroma |
| Function Calling | 结构化输出（Pydantic）+ 工具绑定 |
| Agent | 多 Agent 路由 + ReAct 风格工具调用 |
| SQL Agent | SQLite + SQLAlchemy + 只读安全模式 + 危险 SQL 拦截 |
| LangChain | 核心编排与工具抽象 |
| Fine-tuning / Self-Attention | 自评-迭代闭环（3 轮），修正 prompt 与检索参数 |
| Python | 3.13（managed） |
| TUI 弹窗 | textual |

## 架构概览

```
┌─────────────────────────────────────────────┐
│  UI 层 (textual) — Codex 风格终端弹窗         │
├─────────────────────────────────────────────┤
│  Orchestrator 层 — Agent 路由 / 状态管理       │
├─────────────────────────────────────────┬───┤
│  Agent 层                              │   │
│  ├── RAGAgent 求职知识库助手            │   │
│  ├── SQLAgent 数据分析 Agent            │ E │
│  ├── MatchAgent 简历/JD 匹配助手        │ v │
│  └── EvaluatorAgent AI 检查 AI          │ a │
├─────────────────────────────────────────┤ l │
│  Retrieval 层 — 多路召回 + 约束过滤       │ u │
│  ├── VectorRetriever (SQLite / Chroma)  │ a │
│  └── WebRetriever (DuckDuckGo)          │ t │
│  └── ConstraintEngine                   │ o │
├─────────────────────────────────────────┤ r │
│  Core 层 — LLM / Embeddings / Tools       │   │
├─────────────────────────────────────────┴───┤
│  Tuning 层 — Self-Attention 迭代优化        │
└─────────────────────────────────────────────┘
```

## 设计原则

- **分层隔离**：Core / Retrieval / Agent / Orchestrator / UI 各层只通过明确接口交互。
- **可替换 LLM**：支持 OpenAI 兼容 API / Ollama 本地模型 / Mock LLM，无 API key 也能跑。
- **检索约束**：任何 Web 结果必须经过约束引擎（域名白名单、时间、相关度）才能进入上下文。
- **安全**：SQL Agent 只读连接、危险操作关键词拦截、白名单表限制。
- **可验证**：每步输出都有元信息（来源、评分、迭代轮次），方便调试。

## 快速开始

```bash
# 1. 安装最小依赖（无需 GPU / API key 也能运行）
pip install -r requirements.txt

# 2. （可选）安装完整生产依赖：Chroma + sentence-transformers + torch
pip install -r requirements-full.txt

# 3. 准备数据
python -m src.scripts.ingest

# 4. 启动 Codex 风格弹窗
python -m src.main tui

# 5. 或者使用命令行交互模式
python -m src.main chat
```

环境变量：
- `LLM_PROVIDER=mock`（默认，离线）
- `LLM_PROVIDER=openai` + `OPENAI_API_KEY=...`
- `LLM_PROVIDER=ollama` + `OLLAMA_MODEL=qwen2.5`
- `VECTOR_STORE_BACKEND=chroma` 使用 Chroma 替换 SQLite 向量存储。
