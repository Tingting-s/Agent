# Testing

本文说明项目测试命令、覆盖范围和当前稳定基线。

## 运行全部测试

```bash
python -m pytest -q
```

当前稳定基线：

```text
142 passed, 0 failed, 1352 warnings
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

## QA 评估集

```bash
python -m pytest tests/rag/test_qa_eval_cases.py -v
```

评估数据：

```text
tests/fixtures/qa_eval_cases.json
```

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

## Warning 说明

当前 warnings 来自第三方依赖：

- Chroma 内部访问 Pydantic deprecated API。
- LangGraph / LangChain pending deprecation。

项目自身代码当前没有 pytest warning。不要用全局 `filterwarnings` 粗暴隐藏这些 warning；后续升级 Chroma、LangGraph 或 Pydantic 时应重新审计。
