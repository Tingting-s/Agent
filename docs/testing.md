# Testing

本文说明项目测试命令、覆盖范围和当前稳定基线。

## 运行全部测试

```bash
python -m pytest -q
```

当前稳定基线：

```text
186 passed, 0 failed, 1352 warnings
```

## 常用分组测试

Agent 测试：

```bash
python -m pytest tests/agents -q
```

LangGraph 测试：

```bash
python -m pytest tests/graphs -q
```

RAG 测试：

```bash
python -m pytest tests/rag -q
```

API 测试：

```bash
python -m pytest tests/api -q
```

服务层测试：

```bash
python -m pytest tests/services -q
```

Celery 任务测试：

```bash
python -m pytest tests/tasks -q
```

## QA 评估集

```bash
python -m pytest tests/rag/test_qa_eval_cases.py -v
```

评估数据：

```text
tests/fixtures/qa_eval_cases.json
```

## 测试数据与隔离

`tests/fixtures/` 保存文本、Markdown 和 JSON 测试资产，包括设备手册、培训资料、FAQ、QA 评估集和人工验收问题。这些文件属于可公开的测试输入，应保留在仓库中。

`tests/conftest.py` 会把以下运行路径重定向到 pytest 临时目录：

```text
DOCUMENT_DB_PATH -> 临时 SQLite 数据库
UPLOAD_DIR       -> 临时上传目录
CHROMA_DIR       -> 临时向量目录
LOG_DIR          -> 临时日志目录
```

测试还使用 `tmp_path`、`monkeypatch`、mock vector store、mock LLM 和 Celery eager/Mock 模式，避免读写项目中的正式数据。

## 覆盖范围

测试覆盖：

- 文档上传、列表、删除、批量删除和重新入库。
- 文档解析、文本切分和向量入库。
- Chroma 检索、fallback 检索和规则型 rerank。
- 问答、拒答和来源片段返回。
- LangGraph 工作流。
- QA 日志字段和旧日志兼容。
- legacy task/tool 兼容路径。
- Streamlit 调用后端 API 的关键逻辑。
- SQLite 入库任务状态、并发重复保护和状态恢复。
- Celery eager/Mock 模式下的重试、幂等入库和任务投递。
- 异步上传、状态查询、异步 reindex 及同步操作冲突。
- 共享 Chroma 目录的跨实例写互斥。

真实 Redis、独立 Worker、容器重启和数据持久化不由普通 `pytest` 代替，应在 Docker daemon 可用时按 `docs/deployment.md` 进行端到端验收。

最近一次真实 Docker 验收已覆盖 Redis 投递、独立 Worker 执行、异步入库、SQLite 状态、Chroma 检索、来源追踪、异步 reindex 和容器重启后的持久化。远程 Qwen 请求因权限类 `403` 未验证成功；本地检索与抽取式 fallback 已验证。Worker 强制退出后的即时状态恢复不视为已验证能力。

## Warning 说明

当前 warnings 来自第三方依赖：

- Chroma 内部访问 Pydantic deprecated API。
- LangGraph / LangChain pending deprecation。

项目自身代码当前没有 pytest warning。不要用全局 `filterwarnings` 粗暴隐藏这些 warning；后续升级 Chroma、LangGraph 或 Pydantic 时应重新审计。

## 常见问题

- Chroma 或 sentence-transformers 出现 fallback 日志不等于 pytest 失败，应以测试退出码和断言为准。
- Docker 端到端测试应使用 `UPLOADS_HOST_PATH`、`DATA_HOST_PATH`、`CHROMA_HOST_PATH` 和 `LOGS_HOST_PATH` 指向隔离目录。
- 清理演示数据前应停止 API、Worker 和 Streamlit；`scripts/clear_demo_data.py` 需要人工确认。
- 不要用普通单元测试结果替代真实 Redis、独立 Worker 和容器持久化验收。
