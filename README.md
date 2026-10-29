# PDF-RAG：智能 PDF 学习助手

基于 [hello-agents](https://pypi.org/project/hello-agents/) 的 `RAGTool` 与 `MemoryTool` 构建的 PDF 文档问答助手：上传 PDF、基于检索回答、记录学习笔记与学习历程，并提供 RAG 消融/路由评测脚本。

## 功能

- **PDF 加载与索引**：解析 PDF、按 600 / 120 的块大小与重叠切分，写入 Qdrant 向量库；重复加载同一文档会自动跳过。
- **文档问答**：检索 Top-5 片段后由 LLM 生成回答；可选开启高级检索（MQE / HyDE）。
- **比较类问题**：自动识别「A 和 B 有什么区别」这类句式，分别检索双方证据后合并生成对比答案。
- **记忆系统**：工作记忆、情景记忆、语义记忆三类记录，支持按语义回顾学习历程。
- **智能路由**：关键词路由（`smart_ask`）与 LLM 路由（`smart_ask_llm`）两种方式，决定问题走 RAG 还是 Memory。
- **学习统计与报告**：会话时长、加载文档数、提问次数、笔记数，可导出学习报告 JSON。
- **Web 界面**：Gradio 多标签页应用，包含文档加载、智能问答、学习笔记、学习回顾、统计与报告。

## 目录结构

```
pdf-rag/
├── app.py                  # Gradio Web 界面
├── assistant.py            # PDFLearningAssistant：RAG / Memory / 路由 / 报告
├── agent_demo.py           # 观察 Agent 何时调用 rag / memory 工具的测试
├── eval/
│   ├── evaluate.py         # 检索问答评测（LLM 打分 0-100）
│   ├── evaluation_dataset.json
│   ├── test_routing.py     # 关键词路由 vs LLM 路由对比评测
│   ├── routing_test_set.json
│   └── results*.json       # 历史评测结果
├── docs/
│   └── RAG知识库问答系统实验总报告（七项实验整合版）.docx
├── requirements.txt
├── .env.example            # 环境变量模板
└── knowledge_base/         # 本地文档目录
```

## 快速开始

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env      # 填入自己的密钥
python app.py             # 打开终端输出的本地地址
```

### 环境变量

| 变量 | 说明 |
| --- | --- |
| `QDRANT_URL` / `QDRANT_API_KEY` | Qdrant 向量库地址与密钥 |
| `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` | Neo4j 图数据库连接信息 |
| `EMBED_MODEL_TYPE` / `EMBED_MODEL_NAME` / `EMBED_API_KEY` / `EMBED_BASE_URL` | 嵌入模型配置 |
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL_ID` | 对话模型配置 |
| `RERANK_ENABLED` / `RERANKER_MODEL` / `RERANK_API_KEY` | 重排模型配置 |
| `EVAL_API_KEY` / `EVAL_BASE_URL` / `EVAL_MODEL` | 评测模型（可选，缺省回退到 `LLM_*`） |

> `.env` 已在 `.gitignore` 中排除，请勿把真实密钥提交到仓库。

## 评测

```bash
python eval/evaluate.py       # 问答质量评测，结果写入 eval/results_llm_<时间戳>.json
python eval/test_routing.py   # 路由准确率评测，结果写入 eval/routing_results.json
python agent_demo.py          # 观察 Agent 的工具选择行为
```

`eval/evaluate.py` 让评测模型对照标准答案与参考关键词，对回答给出 0-100 分并列出命中/遗漏关键词；`eval/test_routing.py` 对比关键词路由与 LLM 路由在路由测试集上的表现。

## 实验报告

完整实验记录见 [RAG 知识库问答系统实验总报告（七项实验整合版）](<docs/RAG知识库问答系统实验总报告（七项实验整合版）.docx>)。该报告把 7 份专项实验整合为九章，覆盖 A–G 检索参数对照、20 题端到端评估、LLM 评估员修正、智能路由测试、性能与成本核算、分块策略对照、Rerank 重排对照，并统一了 chunk_size 以字符为单位的计量口径。

## 说明

- 向量检索依赖 Qdrant 云服务，记忆模块依赖 Neo4j，运行前请先准备好对应实例。
- 仓库中的 `eval/results*.json` 为历史实验快照，可直接用于对比不同配置的效果。
