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

启动：

```bash
docker compose up --build
```

停止：

```bash
docker compose down
```

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

## 生产化提醒

当前项目是企业文档问答原型，不是生产部署模板。生产化仍需要补充：

- 用户认证和权限控制。
- 文档级访问控制。
- 异步任务队列。
- 集中式日志和监控。
- 远程对象存储。
- 可运维的向量数据库部署。
