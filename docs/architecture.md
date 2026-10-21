# Architecture

本文说明当前企业文档问答项目的架构选择、目录职责和核心模块边界。

## 为什么采用当前架构

项目采用分层架构，而不是把 API、检索、模型调用和日志写在同一个模块里。这样做有几个目的：

- API 层只负责 HTTP 输入输出和错误处理。
- Service 层承接业务用例，隔离 FastAPI 与底层实现。
- LangGraph 明确表达问答流程，便于展示 AI workflow。
- Agent 层作为 LangGraph 节点外壳，保持节点职责清晰。
- RAG 层集中处理文档解析、切分、向量库、检索和生成辅助逻辑。
- Schema 层集中定义对外请求响应模型，降低 API 变更风险。
- Legacy 层保留旧入口，避免重构后破坏历史调用。

这种结构比单文件脚本更适合面试讲解和后续维护，也比过度拆分更容易阅读。

## 目录职责

```text
app/
├── agents/    LangGraph 节点外壳
├── api/       FastAPI 路由
├── graphs/    LangGraph 状态和工作流
├── rag/       RAG 基础能力
├── schemas/   Pydantic 模型
├── services/  业务服务
├── tasks/     Celery 文档入库任务
├── tools/     legacy 工具兼容
├── utils/     通用工具
├── celery_app.py Celery 应用配置
├── config.py  配置加载
├── main.py    FastAPI 应用入口
└── prompts.py Prompt 常量
```

## FastAPI

FastAPI 位于 `app/api`，负责：

- 暴露健康检查、同步/异步文档管理、任务状态查询、问答、日志和 legacy task 接口。
- 使用 Pydantic response model 保持对外 Schema 稳定。
- 通过统一异常处理返回结构化错误。

API 层不直接实现 RAG 算法，也不直接操作 Chroma；这些职责下沉到 Service 和 RAG 层。

## Service

Service 层位于 `app/services`，负责业务用例编排：

- `document_qa_service.py`：文档上传、重新入库、问答、摘要、删除和日志查询。
- `document_storage_service.py`：SQLite 文档元数据持久化。
- `ingestion_task_service.py`：SQLite 入库任务状态、状态流转、重复任务保护和文档操作锁。
- `llm_service.py`：LLM 调用和重试策略。
- `qa_log_service.py`：问答日志写入和读取。
- `legacy_task_service.py`：旧 task 接口兼容。
- `task_service.py`：历史 `TaskService` 名称兼容入口。

## Celery 与异步入库

异步上传与重新入库复用同步路径中的 `DocumentQAService`，没有维护第二套解析、切分或向量写入实现：

```text
FastAPI
  -> 保存上传文件和文档元数据
  -> SQLite 创建 PENDING 任务
  -> Redis 发布 Celery 消息
  -> 独立 Worker 标记 PROCESSING
  -> DocumentQAService 解析、切分、Embedding、写入 Chroma
  -> SQLite 记录 SUCCESS 或 FAILURE
```

SQLite 的 `task_id` 同时作为 Celery task ID。活跃任务唯一约束防止同一文档重复提交；稳定向量 ID 与 upsert 支持重复执行。SQLite 与 Chroma 不具备跨系统原子事务，Worker 重投递也不是 exactly-once 保证。

## LangGraph

LangGraph 位于 `app/graphs`：

- `state.py` 定义工作流状态。
- `document_qa_graph.py` 定义节点、边和错误流转。

当前节点名称和流程保持稳定：

```text
input_node -> router_node -> document_node -> retrieval_node
-> answer_node -> validator_node -> log_node -> output_node
```

异常路径进入 `error_node` 后继续写日志并输出。

## Agent

Agent 位于 `app/agents`，作为 LangGraph 节点外壳：

- `routing.py`：任务路由。
- `document_agent.py`：文档解析、切分、摘要和向量入库编排。
- `retrieval_agent.py`：调用 RAG 检索实现。
- `answer_agent.py`：构建上下文并生成回答。
- `validation.py`：答案校验和拒答判断。
- `log_agent.py`：将 workflow state 写入 QA 日志。

Agent 不直接承担全部业务服务职责，避免 LangGraph 节点变成大型服务对象。

## RAG

RAG 位于 `app/rag`：

- `ingestion.py`：文件读取、文档解析和文本切分。
- `retrieval.py`：Chroma 检索、fallback 检索和规则型 rerank。
- `generation.py`：Prompt 构造、摘要辅助和无检索内容提示。
- `vector_store.py`：Chroma 持久化、embedding 和向量操作。

`vector_store.py` 独立保留，因为它封装了外部存储和 embedding 相关细节。

## Schema

Schema 位于 `app/schemas`：

- `documents.py`：文档上传、列表、删除、摘要等模型。
- `qa.py`：文档问答请求和响应模型。
- `logs.py`：日志查询响应模型。
- `common.py`：通用响应和 chunk 模型。
- `tasks.py`：legacy task/tool/request/response 模型。
- `response.py`：健康检查和错误响应模型。
- `legacy.py`：旧 import 路径兼容 shim。

Schema 文件保持边界清晰，不与业务服务互相依赖。
