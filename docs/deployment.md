# Deployment

本文说明本地开发、Windows、Linux、Conda 和 Docker 的启动方式。

## 前置条件

- Python 3.12
- Conda 或 venv
- 可选：Docker 和 Docker Compose
- 可选：真实 LLM API Key

## Windows + Conda

创建环境：

```powershell
conda create -n enterprise-doc-qa python=3.12 -y
conda activate enterprise-doc-qa
```

安装依赖：

```powershell
pip install -r requirements.txt
```

如需复现实测版本：

```powershell
pip install -r requirements-lock.txt
```

配置环境变量：

```powershell
copy .env.example .env
```

启动 FastAPI：

```powershell
python -m uvicorn app.main:app --reload
```

启动 Streamlit：

```powershell
streamlit run web/streamlit_app.py
```

## Linux + Conda

创建环境：

```bash
conda create -n enterprise-doc-qa python=3.12 -y
conda activate enterprise-doc-qa
```

安装依赖：

```bash
pip install -r requirements.txt
```

配置环境变量：

```bash
cp .env.example .env
```

启动服务：

```bash
python -m uvicorn app.main:app --reload
```

启动演示页面：

```bash
streamlit run web/streamlit_app.py
```

## 访问地址

```text
FastAPI:   http://127.0.0.1:8000
Swagger:   http://127.0.0.1:8000/docs
Health:    http://127.0.0.1:8000/health
Streamlit: http://localhost:8501
```

## Docker Compose

准备环境变量：

```bash
cp .env.example .env
```

静态校验并启动：

```bash
docker compose config --quiet
docker compose up --build -d
docker compose ps
```

Compose 包含四个服务：

- `api`：FastAPI，同步接口和异步任务提交入口。
- `redis`：Celery broker/result backend，启用 AOF，不暴露宿主机端口。
- `worker`：独立 Celery Worker，初始 `solo` pool、并发 1。
- `web`：Streamlit 演示端，通过 `http://api:8000` 调用后端。

API 不依赖 Redis 才能启动，Redis 故障时旧同步接口仍可使用；Worker 只等待 Redis 健康，不依赖 API 进程。

API 和 Worker 必须共享 `/app/uploads`、`/app/data`、`/app/chroma_db`、`/app/logs`。默认映射到项目同名目录；端到端测试时可在 `.env` 改为隔离路径：

```text
UPLOADS_HOST_PATH=./pytest-cache-files-docker-e2e/uploads
DATA_HOST_PATH=./pytest-cache-files-docker-e2e/data
CHROMA_HOST_PATH=./pytest-cache-files-docker-e2e/chroma_db
LOGS_HOST_PATH=./pytest-cache-files-docker-e2e/logs
```

Embedding 模型缓存使用 `embedding_model_cache` 命名 volume。首次联网加载后 API 和 Worker 复用同一缓存。离线运行设置 `DOCKER_EMBEDDING_LOCAL_FILES_ONLY=true`。

查看任务注册与日志：

```bash
docker compose exec worker celery -A app.celery_app:celery_app inspect registered
docker compose logs -f worker
docker compose logs -f api
```

容器重启但保留数据：

```bash
docker compose stop
docker compose start
```

不要在需要保留 Redis 任务结果和模型缓存时执行 `docker compose down -v`。

停止：

```bash
docker compose down
```

## 已验证范围

真实 Docker Compose 环境已验证 API、Redis、独立 Celery Worker 和 Streamlit 能共同启动，并使用隔离宿主机目录完成异步上传、Redis 消息分发、Worker 入库、SQLite 状态查询、Chroma 检索、来源追踪、异步重新入库以及容器重启后的数据持久化。

该验收不代表所有故障都能自动恢复。Worker 强制退出后的即时恢复、所有可能的跨进程竞态和 exactly-once 执行均不作保证。远程 Qwen 调用曾返回权限类 `403`，因此已验证的是本地 Chroma 检索、来源追踪和抽取式 fallback，不应描述为远程模型生成已成功验收。

## 运行时目录

以下目录用于本地运行，不应提交到 Git：

```text
uploads/
logs/
chroma_db/
data/documents.db
```

需要清理演示数据时，先停止服务，再执行：

```bash
python scripts/clear_demo_data.py
```

该脚本需要手动输入确认，不会删除源码、测试、README 或 `.env.example`。

## 一致性边界

- SQLite 活跃任务唯一索引保证同一文档只有一个 `PENDING/PROCESSING` 入库任务。
- API 的同步 reindex/delete 使用文档操作锁；活动异步任务期间返回 `409`。
- Chroma 初始化、upsert、delete 使用 `chroma_db/.chroma-write-lock.sqlite3` 的 SQLite 写事务锁，覆盖 API 与 Worker 两个进程。
- Worker 重复投递依靠稳定向量 ID 和 upsert 保持幂等；reindex 先删除该文档旧 chunks。
- SQLite 与 Chroma 没有跨系统原子事务。若 Chroma 成功后 SQLite 状态写入失败，Celery 重试会重新执行幂等入库并再次记录状态。
- `acks_late` 与 Redis visibility timeout 能改善 Worker 异常退出后的重投递，但不是 exactly-once 保证。
- 重新入库会先删除旧 chunks，再写入新 chunks；如果中途永久失败，旧向量可能不可用，需要再次提交 reindex 恢复。

## 生产化提醒

当前项目是企业文档问答原型，不是生产部署模板。生产化仍需要补充：

- 用户认证和权限控制。
- 文档级访问控制。
- 集中式日志和监控。
- 远程对象存储。
- 可运维的向量数据库部署。
