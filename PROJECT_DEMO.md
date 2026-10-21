# 企业文档问答系统演示说明

## 1. 项目名称

企业文档问答系统

## 2. 项目定位

这是一个面向企业内部资料查询场景的文档问答原型系统，支持同步/异步文档上传、解析切分、语义向量检索、答案生成、来源片段追踪、无答案拒答、任务状态持久化、失败重试、日志记录和 Streamlit 演示页面。

项目重点展示 FastAPI 后端、LangGraph 流程编排、多 Agent 模块、Chroma 向量检索、SQLite 元数据与任务状态、Celery + Redis 后台入库和 pytest 测试覆盖。Streamlit 页面用于演示和验证，不是正式前端。

## 3. 启动方式

后端启动：

```bash
python -m uvicorn app.main:app --reload
```

前端演示页面启动：

```bash
streamlit run web/streamlit_app.py
```

默认访问地址：

```text
FastAPI:   http://127.0.0.1:8000
Streamlit: http://localhost:8501
```

## 4. Docker 演示方式

准备环境变量：

```bash
cp .env.example .env
```

校验配置并启动四个服务：

```bash
docker compose config --quiet
docker compose up --build -d
docker compose ps
```

打开页面：

```text
http://127.0.0.1:8501
```

Compose 包含：

- `api`：FastAPI，同步接口和异步任务提交入口。
- `redis`：Celery broker/result backend，不暴露宿主机端口。
- `worker`：独立 Celery Worker，执行文档解析、切分、Embedding 和 Chroma 入库。
- `web`：Streamlit，通过 `API_BASE_URL=http://api:8000` 访问 FastAPI。

API 与 Worker 共享以下宿主机目录：

- `uploads/`
- `logs/`
- `chroma_db/`
- `data/`

Embedding 缓存与 Redis AOF 使用 Docker 命名 volume。结束后使用 `docker compose down` 停止服务；需要保留缓存和 Redis 数据时不要添加 `-v`。

当前 Docker 配置用于本地演示和项目展示，不是生产级部署。

## 5. 可选：初始化演示数据

如果希望快速进入演示状态，可以先导入测试样本文档：

```bash
python scripts/init_demo_data.py
```

脚本会尝试导入：

- `tests/fixtures/sample_device_manual.txt`
- `tests/fixtures/sample_training_material.md`
- `tests/fixtures/sample_faq.txt`

导入完成后会输出每个文档的 `document_id`、原始文件名、`chunk_count` 和 `vector_indexed`。

标准 RAG 页面演示建议优先上传：

- `tests/fixtures/rag_device_manual.txt`
- `tests/fixtures/rag_training_manual.md`
- `tests/fixtures/rag_faq_manual.txt`

这三份文档是纯文本或 Markdown，适合验证向量检索、来源片段、规则型 Rerank 和无答案拒答。Word 表格文档主要用于解析兼容性测试，不建议作为主要 RAG 知识库演示数据。

## 6. 演示流程

### 6.1 异步上传 PDF

准备一个可提取文本的 PDF，并调用：

```bash
curl -X POST http://127.0.0.1:8000/api/documents/upload/async \
  -F "file=@YOUR_DOCUMENT.pdf"
```

预期 HTTP 状态为 `202`，响应包含 `task_id`、`document_id` 和 `PENDING`。请求只负责保存文件、创建元数据和发布消息，不等待完整向量入库。

### 6.2 查询后台任务

```bash
curl http://127.0.0.1:8000/api/documents/tasks/YOUR_TASK_ID
docker compose logs -f worker
```

观察 SQLite 状态从 `PENDING`、`PROCESSING` 进入 `SUCCESS`；成功记录应包含 `chunk_count > 0` 和 `vector_indexed=true`。Worker 日志用于确认任务经过 Redis 并由独立进程执行，而不是 eager 模式。

### 6.3 问答与来源追踪

使用返回的 `document_id` 调用 `/api/qa/ask`，提问 PDF 中有明确依据的问题。检查：

- `retrieval_source=chroma`。
- `source_chunks` 来自刚上传文档。
- 文档中没有依据的问题触发拒答。
- 未配置可用远程 LLM 时，结果可能来自抽取式 fallback；不要将其描述为远程 Qwen 生成成功。

### 6.4 异步重新入库

```bash
curl -X POST http://127.0.0.1:8000/api/documents/YOUR_DOCUMENT_ID/reindex/async
```

记录新的 `task_id`，再次查询至终态，并确认 chunk 数量和问答仍正常。同一文档已有活跃任务时返回原任务；同步 reindex 或删除与活动任务冲突时返回 `409`。

### 6.5 失败处理

- 不支持的格式在 API 校验阶段返回 `400`。
- Redis 投递失败返回 `503`，SQLite 任务进入 `FAILURE`，不会伪装为提交成功。
- Redis 恢复后可通过异步 reindex 创建新任务。
- Worker 使用有限重试，但强制退出后的即时恢复和 exactly-once 不作保证。

## 7. 推荐演示问题

有答案问题：

- 设备启动前需要检查什么？
- 设备报警后应该如何处理？
- 设备日常维护需要注意什么？
- 培训考核标准是什么？

无答案问题：

- 设备采购价格是多少？
- 公司年终奖怎么算？
- 是否支持海外售后？

## 8. 演示重点

- Chroma 向量检索：上传后的文档 chunk 会写入本地 Chroma 索引。
- `retrieval_source`：用于说明当前回答依据来自 Chroma 检索还是 fallback。
- `source_chunks`：表示回答依据，可用于核对答案是否来自文档。
- 无答案拒答：检索不到可靠依据时，不编造价格、采购、售后等信息。
- 阶段耗时日志：日志中记录检索、重排、Prompt 构造、LLM、校验等耗时。
- 重新入库：`reindex` 可用于文档已上传但向量入库失败、切换 embedding 模型或旧索引异常后的重建。
- 异步入库：API 返回 `task_id`，Redis 分发消息，Celery Worker 执行耗时处理。
- 持久任务状态：SQLite 保存状态、重试次数、chunk 数量和失败原因，API 重启后仍可查询。
- 并发保护：活动任务阻止同一文档的同步 reindex/delete，Chroma 写操作使用跨进程锁。
- 日志追踪：`logs/qa_logs.jsonl` 和 `/api/logs` 可用于排查检索失败、生成超时和异常问题。

## 9. 辅助脚本

项目自检：

```bash
python scripts/check_project_status.py
```

初始化演示数据：

```bash
python scripts/init_demo_data.py
```

清理演示数据：

```bash
python scripts/clear_demo_data.py
```

清理脚本需要手动输入 `YES` 才会删除运行数据。

## 10. 当前限制

- Streamlit 是演示端，不是正式前端。
- 当前无用户登录和权限管理。
- 当前不支持 OCR，不支持扫描 PDF 或图片文档识别。
- 当前不支持多知识库和文档版本管理。
- 当前 Rerank 为规则型重排，不是模型级 Rerank。
- 当前使用本地 Chroma、SQLite 和文件目录，适合本地开发、演示和项目说明，不适合作为生产环境直接使用。
- SQLite 与 Chroma 不存在跨系统原子事务，任务重投递也不是 exactly-once。
- Worker 强制退出后的任务状态不保证立即自动恢复。
- 真实 Docker 验收中的远程 Qwen 请求曾返回权限类 `403`，已验证的是 Chroma 检索、来源追踪和抽取式 fallback。

## 11. 数据目录和清理说明

运行时会生成以下目录或文件：

- `uploads/`：原始上传文件。
- `logs/`：问答日志。
- `chroma_db/`：Chroma 本地向量索引。
- `data/documents.db`：SQLite 文档元数据。

不要把这些运行数据提交到 Git。如果需要清空演示数据，请先停止 FastAPI 和 Streamlit 服务，再运行：

```bash
python scripts/clear_demo_data.py
```

也可以手动删除：

```text
uploads/
logs/
chroma_db/
data/documents.db
```

这些目录和数据库文件会在下次运行服务或接口调用时重新生成。

## 12. 验收命令

```bash
python -m pytest -q
python -m pytest tests/rag/test_rag_baseline_retrieval.py -v
python -m pytest tests/rag/test_qa_eval_cases.py -v
python -m pytest tests/rag/test_vector_store.py -v
```

当前重构后稳定基线：

- 全量 `python -m pytest -q`：186 passed, 0 failed, 1352 warnings

真实 Docker Compose 验收已覆盖异步上传、Redis 投递、独立 Worker、SQLite 状态、Chroma 入库、来源追踪、异步重新入库和容器重启持久化。演示时仍应展示本次实际响应和容器日志，不预先声称未执行的故障注入成功。

## 13. 简历项目描述参考

- 基于 FastAPI、LangGraph 和 Chroma 实现企业文档问答原型，支持文档上传、解析切分、向量检索、问答生成、来源片段追踪和无答案拒答。
- 使用 Celery + Redis 构建异步文档入库链路，API 返回 `task_id`，独立 Worker 执行解析、Embedding 与 Chroma 入库。
- 使用 SQLite 持久化文档元数据和任务状态，支持有限重试、重复任务保护、重新入库以及服务重启后的历史状态查询。
- 使用 pytest 构建 API、RAG、LangGraph 和评估集测试，覆盖无答案拒答、Chroma 检索、规则型 Rerank、日志兼容和向量清理等场景。
