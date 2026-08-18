from __future__ import annotations

from html import escape
import logging
import os
import re
from typing import Any
from uuid import UUID

import requests
import streamlit as st


DEFAULT_API_BASE_URL = (
    os.getenv("API_BASE_URL")
    or os.getenv("FASTAPI_BASE_URL")
    or os.getenv("VITE_API_BASE_URL")
    or "http://127.0.0.1:8000"
).rstrip("/")
REQUEST_TIMEOUT = 20
UPLOAD_TIMEOUT = 300
QA_TIMEOUT = 60
SUPPORTED_FILE_TYPES = ["txt", "md", "docx", "pdf"]
STORED_FILE_PREFIX_RE = re.compile(r"^[0-9a-fA-F][0-9a-fA-F-]{19,}_")
LOGGER = logging.getLogger(__name__)
NAVIGATION_PAGES = ["系统概览", "文档管理", "文档摘要", "知识库问答", "问答日志", "系统说明"]
EXAMPLE_QUESTIONS = [
    "设备启动前需要检查什么？",
    "设备报警后应该如何处理？",
    "培训考核标准是什么？",
    "文档中是否说明异常处理流程？",
]


def inject_custom_css() -> None:
    st.markdown(
        """
<style>
    :root {
        --primary: #2563eb;
        --primary-dark: #1d4ed8;
        --bg: #f3f6fb;
        --card-bg: #ffffff;
        --border: #e5eaf3;
        --text: #172033;
        --muted: #68758a;
        --success: #0f9f6e;
        --warning: #d97706;
        --danger: #dc2626;
        --shadow: 0 10px 28px rgba(15, 23, 42, 0.07);
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    .block-container {
        max-width: 1280px;
        padding-top: 1.35rem;
        padding-bottom: 2rem;
    }

    section[data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    section[data-testid="stSidebar"] * {
        color: rgba(255, 255, 255, 0.9);
    }

    section[data-testid="stSidebar"] input {
        color: #111827 !important;
        background: #ffffff !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 0.45rem 0.55rem;
        border-radius: 10px;
        margin-bottom: 0.15rem;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.08);
    }

    .app-sidebar-title {
        margin: 0.15rem 0 0.15rem;
        color: #ffffff;
        font-size: 1.22rem;
        font-weight: 800;
        letter-spacing: 0;
    }

    .app-sidebar-footer {
        color: rgba(255, 255, 255, 0.58);
        font-size: 0.78rem;
        line-height: 1.5;
        margin-top: 1.25rem;
        padding-top: 0.8rem;
        border-top: 1px solid rgba(255, 255, 255, 0.12);
    }

    .page-header {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 48%, #4f8dfd 100%);
        border-radius: 18px;
        padding: 1.05rem 1.35rem;
        color: #ffffff;
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
    }

    .page-title {
        margin: 0;
        font-size: 1.62rem;
        font-weight: 800;
        letter-spacing: 0;
    }

    .page-description {
        margin: 0.35rem 0 0;
        color: rgba(255, 255, 255, 0.86);
        line-height: 1.5;
    }

    .card {
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 16px;
        box-shadow: var(--shadow);
        padding: 1.15rem 1.2rem;
        margin-bottom: 1rem;
    }

    .card-title {
        margin: 0 0 0.35rem;
        color: var(--text);
        font-size: 1.05rem;
        font-weight: 750;
    }

    .card-desc {
        margin: 0;
        color: var(--muted);
        line-height: 1.6;
        font-size: 0.92rem;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 16px;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
        padding: 1rem 1rem;
        min-height: 112px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.82rem;
        margin-bottom: 0.55rem;
    }

    .metric-value {
        color: var(--text);
        font-size: 1.45rem;
        font-weight: 800;
        line-height: 1.2;
    }

    .metric-help {
        color: #8a95a8;
        font-size: 0.78rem;
        margin-top: 0.35rem;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 0.22rem 0.58rem;
        font-size: 0.8rem;
        font-weight: 650;
        border: 1px solid transparent;
    }

    .status-success {
        color: #047857;
        background: #dcfce7;
        border-color: #bbf7d0;
    }

    .status-error {
        color: #b91c1c;
        background: #fee2e2;
        border-color: #fecaca;
    }

    .status-muted {
        color: #475569;
        background: #eef2f7;
        border-color: #dde5ef;
    }

    .capability-card {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem;
        min-height: 122px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.045);
    }

    .capability-icon {
        width: 32px;
        height: 32px;
        display: grid;
        place-items: center;
        border-radius: 10px;
        background: #eaf2ff;
        color: var(--primary);
        font-size: 1rem;
        margin-bottom: 0.55rem;
    }

    .process-flow {
        display: grid;
        grid-template-columns: repeat(6, minmax(0, 1fr));
        gap: 0.65rem;
        margin-bottom: 1rem;
    }

    .process-step {
        position: relative;
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.9rem 0.75rem;
        text-align: center;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.045);
    }

    .process-step:not(:last-child)::after {
        content: "→";
        position: absolute;
        right: -0.55rem;
        top: 50%;
        transform: translateY(-50%);
        color: #8aa4ce;
        font-weight: 800;
        z-index: 2;
    }

    .process-step-number {
        display: inline-grid;
        place-items: center;
        width: 26px;
        height: 26px;
        border-radius: 999px;
        background: #eaf2ff;
        color: var(--primary);
        font-weight: 800;
        margin-bottom: 0.45rem;
    }

    .process-step-title {
        color: var(--text);
        font-weight: 750;
        font-size: 0.92rem;
    }

    .record-item {
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.78rem 0.9rem;
        margin-bottom: 0.55rem;
        background: #fbfdff;
    }

    .record-title {
        color: var(--text);
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .record-meta {
        color: var(--muted);
        font-size: 0.82rem;
    }

    .doc-current {
        border-left: 4px solid var(--primary);
        background: #f8fbff;
        border-radius: 14px;
        padding: 0.9rem 1rem;
        border-top: 1px solid var(--border);
        border-right: 1px solid var(--border);
        border-bottom: 1px solid var(--border);
    }

    .thin-divider {
        height: 1px;
        background: var(--border);
        margin: 1rem 0;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.045);
    }

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #cbd7ea;
        font-weight: 650;
    }

    .stButton > button[kind="primary"] {
        border: 0;
        background: var(--primary);
    }

    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"] {
        border-radius: 10px;
    }
</style>
""",
        unsafe_allow_html=True,
    )


def get_api_base_url() -> str:
    value = st.sidebar.text_input(
        "后端地址",
        value=st.session_state.get("api_base_url", DEFAULT_API_BASE_URL),
        help="默认地址为 http://127.0.0.1:8000",
    )
    api_base_url = value.strip().rstrip("/") or DEFAULT_API_BASE_URL
    st.session_state["api_base_url"] = api_base_url
    return api_base_url


def _request_json(
    method: str,
    path: str,
    *,
    api_base_url: str,
    timeout: int = REQUEST_TIMEOUT,
    **kwargs: Any,
) -> tuple[dict[str, Any] | None, str]:
    url = f"{api_base_url}{path}"
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        if path == "/api/qa/ask":
            return None, "问答请求超时，请检查后端日志或稍后重试。"
        if path == "/api/documents/upload":
            return None, (
                "请求超时：/api/documents/upload。"
                "文档上传接口会同步解析、切分并写入 Chroma 向量库；"
                "首次加载 sentence-transformers embedding 模型或向量入库耗时较长时，请稍后刷新文档列表确认结果。"
            )
        return None, f"请求超时：{path}"
    except requests.exceptions.ConnectionError:
        return None, f"后端连接失败，请确认 FastAPI 已启动：{api_base_url}"
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        return None, f"接口返回 HTTP 错误：{status_code}"
    except requests.exceptions.RequestException as exc:
        return None, f"接口请求失败：{exc}"

    try:
        payload = response.json()
    except ValueError:
        return None, "接口返回内容不是合法 JSON。"
    if not isinstance(payload, dict):
        return None, "接口返回格式不符合预期。"
    return payload, ""


def api_get(path: str, *, api_base_url: str, timeout: int = REQUEST_TIMEOUT) -> tuple[dict[str, Any] | None, str]:
    return _request_json("GET", path, api_base_url=api_base_url, timeout=timeout)


def api_post(
    path: str,
    *,
    api_base_url: str,
    timeout: int = REQUEST_TIMEOUT,
    **kwargs: Any,
) -> tuple[dict[str, Any] | None, str]:
    return _request_json("POST", path, api_base_url=api_base_url, timeout=timeout, **kwargs)


def check_health(api_base_url: str) -> tuple[dict[str, Any] | None, str]:
    return api_get("/health", api_base_url=api_base_url)


def fetch_logs(api_base_url: str, limit: int = 50) -> tuple[list[dict[str, Any]], str]:
    payload, error = api_get(f"/api/logs?limit={limit}", api_base_url=api_base_url)
    if error:
        return [], error
    if not payload:
        return [], ""
    items = payload.get("items", [])
    if not isinstance(items, list):
        return [], "日志接口返回格式不符合预期。"
    return [item for item in items if isinstance(item, dict)], ""


def upload_document(api_base_url: str, uploaded_file: Any) -> tuple[dict[str, Any] | None, str]:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    return api_post(
        "/api/documents/upload",
        api_base_url=api_base_url,
        timeout=UPLOAD_TIMEOUT,
        files=files,
    )


def fetch_documents(api_base_url: str) -> tuple[list[dict[str, Any]], str]:
    payload, error = api_get("/api/documents", api_base_url=api_base_url)
    if error:
        return [], error
    if not payload:
        return [], ""
    items = payload.get("items", [])
    if not isinstance(items, list):
        return [], "文档列表接口返回格式不符合预期。"
    return [item for item in items if isinstance(item, dict)], ""


def delete_document(api_base_url: str, document_id: str) -> tuple[dict[str, Any] | None, str]:
    return _request_json("DELETE", f"/api/documents/{document_id}", api_base_url=api_base_url)


def reindex_document(api_base_url: str, document_id: str) -> tuple[dict[str, Any] | None, str]:
    return api_post(
        f"/api/documents/{document_id}/reindex",
        api_base_url=api_base_url,
        timeout=UPLOAD_TIMEOUT,
    )


def batch_delete_documents(api_base_url: str, document_ids: list[str]) -> tuple[dict[str, Any] | None, str]:
    return _request_json(
        "DELETE",
        "/api/documents/batch",
        api_base_url=api_base_url,
        json={"document_ids": document_ids},
    )


def summarize_document(api_base_url: str, document_id: str) -> tuple[dict[str, Any] | None, str]:
    return api_post(
        "/api/documents/summary",
        api_base_url=api_base_url,
        json={"document_id": document_id},
    )


def ask_question(api_base_url: str, document_id: str, question: str) -> tuple[dict[str, Any] | None, str]:
    request_body = {"document_id": str(document_id or "").strip(), "question": question}
    LOGGER.info("QA request body document_id=%s", request_body["document_id"])
    return api_post(
        "/api/qa/ask",
        api_base_url=api_base_url,
        json=request_body,
        timeout=QA_TIMEOUT,
    )


def get_retrieval_source(payload: dict[str, Any]) -> str:
    retrieval_source = str(payload.get("retrieval_source") or "").strip()
    if retrieval_source:
        return retrieval_source

    source_chunks = payload.get("source_chunks")
    if isinstance(source_chunks, list):
        for chunk in source_chunks:
            if not isinstance(chunk, dict):
                continue
            chunk_source = str(chunk.get("retrieval_source") or "").strip()
            if chunk_source:
                return chunk_source
            metadata = chunk.get("metadata")
            if isinstance(metadata, dict):
                metadata_source = str(metadata.get("retrieval_source") or "").strip()
                if metadata_source:
                    return metadata_source
    return ""


def retrieval_source_label(retrieval_source: str) -> str:
    labels = {
        "chroma": "检索来源：Chroma 向量检索",
        "fallback": "检索来源：基础检索 fallback",
        "none": "未检索到相关内容",
    }
    return labels.get(retrieval_source, "检索来源：未返回")


def render_page_header(title: str, description: str) -> None:
    st.markdown(
        f"""
<div class="page-header">
    <h1 class="page-title">{escape(title)}</h1>
    <p class="page-description">{escape(description)}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str | int | float, help_text: str = "") -> None:
    st.markdown(
        f"""
<div class="metric-card">
    <div class="metric-label">{escape(str(label))}</div>
    <div class="metric-value">{escape(str(value))}</div>
    <div class="metric-help">{escape(help_text)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_status_badge(status: str) -> None:
    normalized = str(status or "unknown").lower()
    css_class = "status-success" if normalized in {"ok", "success", "运行中"} else (
        "status-error" if normalized in {"error", "failed", "不可用", "异常"} else "status-muted"
    )
    label = status or "unknown"
    st.markdown(
        f'<span class="status-badge {css_class}">{escape(str(label))}</span>',
        unsafe_allow_html=True,
    )


def render_card(title: str, description: str = "") -> None:
    st.markdown(
        f"""
<div class="card">
    <div class="card-title">{escape(title)}</div>
    <p class="card-desc">{escape(description)}</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_process_flow() -> None:
    steps = ["文档上传", "文档解析", "知识检索", "答案生成", "答案校验", "日志记录"]
    items = "".join(
        f"""
<div class="process-step">
    <div class="process-step-number">{index}</div>
    <div class="process-step-title">{escape(title)}</div>
</div>
"""
        for index, title in enumerate(steps, start=1)
    )
    st.markdown(f'<div class="process-flow">{items}</div>', unsafe_allow_html=True)


def render_capability_cards() -> None:
    capabilities = [
        ("📄", "文档上传与解析", "上传企业文档并提取可用于问答的正文内容。"),
        ("✂️", "文本切分与检索", "将长文档切分为片段，并根据问题查找相关内容。"),
        ("💬", "问答生成", "基于检索到的文档内容生成简洁回答。"),
        ("✅", "答案质量校验", "检查回答是否为空、是否具备文档依据。"),
        ("🔎", "来源片段追踪", "展示支撑回答的文档片段，便于核对依据。"),
        ("🧾", "问答日志记录", "记录问题、回答、状态、耗时和异常信息。"),
    ]
    for row_start in range(0, len(capabilities), 3):
        cols = st.columns(3)
        for col, (icon, title, desc) in zip(cols, capabilities[row_start : row_start + 3], strict=False):
            with col:
                st.markdown(
                    f"""
<div class="capability-card">
    <div class="capability-icon">{icon}</div>
    <div class="card-title">{escape(title)}</div>
    <p class="card-desc">{escape(desc)}</p>
</div>
""",
                    unsafe_allow_html=True,
                )


def _init_state() -> None:
    st.session_state.setdefault("uploaded_documents", [])
    st.session_state.setdefault("current_document_id", "")
    st.session_state.setdefault("qa_question", "")
    st.session_state.setdefault("last_upload_result", None)
    st.session_state.setdefault("last_reindex_result", None)
    st.session_state.setdefault("last_batch_delete_result", None)
    st.session_state.setdefault("last_summary_result", None)
    st.session_state.setdefault("last_qa_result", None)
    st.session_state.setdefault("documents_loaded", False)
    st.session_state.setdefault("documents_api_base_url", "")


def _document_options() -> list[dict[str, Any]]:
    docs = st.session_state.get("uploaded_documents", [])
    return [_normalize_document_entry(doc) for doc in docs if isinstance(doc, dict) and doc.get("document_id")]


def _stored_file_name(document: dict[str, Any]) -> str:
    return str(document.get("stored_file_name") or document.get("file_name") or "").strip()


def _strip_document_prefix(file_name: str, document_id: str = "") -> str:
    normalized_file_name = str(file_name or "").strip()
    if not normalized_file_name:
        return "未命名文档"

    normalized_document_id = str(document_id or "").strip()
    if normalized_document_id:
        document_prefix = f"{normalized_document_id}_"
        if normalized_file_name.startswith(document_prefix):
            display_name = normalized_file_name[len(document_prefix) :].strip()
            return display_name or normalized_file_name

    prefix_match = STORED_FILE_PREFIX_RE.match(normalized_file_name)
    if prefix_match:
        display_name = normalized_file_name[prefix_match.end() :].strip()
        return display_name or normalized_file_name

    return normalized_file_name


def _display_name(document: dict[str, Any]) -> str:
    explicit_display_name = str(document.get("display_name") or "").strip()
    if explicit_display_name:
        return explicit_display_name
    original_file_name = str(document.get("original_file_name") or "").strip()
    if original_file_name:
        return original_file_name
    stored_file_name = _stored_file_name(document)
    return _strip_document_prefix(stored_file_name, str(document.get("document_id") or ""))


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _safe_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _normalize_document_entry(document: dict[str, Any]) -> dict[str, Any]:
    stored_file_name = _stored_file_name(document)
    display_name = _display_name(document)
    return {
        "document_id": str(document.get("document_id", "")),
        "original_file_name": str(document.get("original_file_name") or display_name),
        "display_name": display_name,
        "stored_file_name": stored_file_name,
        "file_name": stored_file_name,
        "upload_time": str(document.get("upload_time", "")),
        "status": str(document.get("status", "")),
        "chunk_count": _safe_int(document.get("chunk_count")),
        "vector_indexed": _safe_bool(document.get("vector_indexed", False)),
        "message": str(document.get("message", "")),
        "error_message": str(document.get("error_message", "")),
        "updated_time": str(document.get("updated_time", "")),
    }


def _document_label(document: dict[str, Any]) -> str:
    return _display_name(document)


def _is_uuid(value: str) -> bool:
    try:
        UUID(str(value or "").strip())
    except (TypeError, ValueError, AttributeError):
        return False
    return True


def _resolve_qa_document_id(selected_document_id: str) -> tuple[str, dict[str, Any]]:
    selected_value = str(selected_document_id or "").strip()
    documents = _document_options()
    for document in documents:
        document_id = str(document.get("document_id") or "").strip()
        if document_id == selected_value:
            return document_id, document

    for document in documents:
        document_id = str(document.get("document_id") or "").strip()
        document_names = {
            str(document.get("display_name") or "").strip(),
            str(document.get("original_file_name") or "").strip(),
            str(document.get("stored_file_name") or "").strip(),
            str(document.get("file_name") or "").strip(),
        }
        if selected_value and selected_value in document_names and _is_uuid(document_id):
            return document_id, document

    return selected_value, {}


def _log_qa_submit(selected_display_name: str, selected_document_id: str, request_document_id: str) -> None:
    message = (
        "QA submit "
        f"selected display_name={selected_display_name} "
        f"selected document_id={selected_document_id} "
        f"request body document_id={request_document_id}"
    )
    LOGGER.info(message)
    if request_document_id and not _is_uuid(request_document_id):
        LOGGER.warning("QA request body document_id is not a UUID: %s", request_document_id)


def _load_persisted_documents(api_base_url: str, *, force: bool = False, show_error: bool = False) -> None:
    already_loaded = bool(st.session_state.get("documents_loaded"))
    loaded_api_base_url = str(st.session_state.get("documents_api_base_url") or "")
    if already_loaded and loaded_api_base_url == api_base_url and not force:
        return

    documents, error = fetch_documents(api_base_url)
    if error:
        if show_error:
            st.warning(f"文档列表加载失败：{error}")
        return

    normalized_documents = [_normalize_document_entry(document) for document in documents]
    st.session_state["uploaded_documents"] = normalized_documents
    st.session_state["documents_loaded"] = True
    st.session_state["documents_api_base_url"] = api_base_url

    current_document_id = str(st.session_state.get("current_document_id") or "")
    if current_document_id and not any(
        document.get("document_id") == current_document_id for document in normalized_documents
    ):
        st.session_state["current_document_id"] = ""


def _select_document(label: str = "选择文档") -> str:
    documents = _document_options()
    if not documents:
        st.warning("请先上传或选择文档。")
        return ""

    current_document_id = st.session_state.get("current_document_id", "")
    current_index = 0
    for index, document in enumerate(documents):
        if document.get("document_id") == current_document_id:
            current_index = index
            break

    selected = st.selectbox(
        label,
        options=documents,
        index=current_index,
        format_func=_document_label,
    )
    selected_document_id = str(selected.get("document_id", ""))
    st.session_state["current_document_id"] = selected_document_id
    return selected_document_id


def _find_document(document_id: str) -> dict[str, Any]:
    return next((doc for doc in _document_options() if doc.get("document_id") == document_id), {})


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _render_status_summary(payload: dict[str, Any]) -> None:
    status = str(payload.get("status", "unknown"))
    cols = st.columns([1, 1, 2])
    with cols[0]:
        st.caption("状态")
        render_status_badge(status)
    cols[1].metric("耗时", f"{_safe_float(payload.get('elapsed_time')):.3f}s")
    error_message = str(payload.get("error_message") or "")
    cols[2].metric("错误信息", "无" if not error_message else error_message[:28])


def _summary_log_rows(logs: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    return [
        {
            "问题": item.get("question", ""),
            "状态": item.get("status", ""),
            "响应时间": f"{_safe_float(item.get('elapsed_time')):.3f}s",
            "时间": item.get("timestamp", ""),
        }
        for item in logs[:limit]
    ]


def render_overview_page(api_base_url: str, health: dict[str, Any] | None = None, health_error: str = "") -> None:
    render_page_header(
        "企业文档问答系统",
        "支持企业文档上传、知识检索、答案生成、来源追踪和日志记录。",
    )

    logs, logs_error = fetch_logs(api_base_url, limit=100)
    backend_ok = bool(health and health.get("status") == "ok")
    status_label = "运行中" if backend_ok else "不可用"
    success_count = sum(1 for item in logs if item.get("status") == "success")
    error_count = sum(1 for item in logs if item.get("status") != "success")
    elapsed_values = [_safe_float(item.get("elapsed_time")) for item in logs if item.get("elapsed_time") is not None]
    avg_elapsed = sum(elapsed_values) / len(elapsed_values) if elapsed_values else 0.0

    cols = st.columns(5)
    metric_data = [
        ("后端状态", status_label, "当前问答服务连接状态"),
        ("累计问答次数", len(logs), "已记录的问答请求总数"),
        ("成功问答次数", success_count, "正常完成的问答请求"),
        ("异常次数", error_count, "处理失败或异常记录"),
        ("平均响应时间", f"{avg_elapsed:.3f}s", "最近问答平均耗时"),
    ]
    for col, (label, value, help_text) in zip(cols, metric_data, strict=False):
        with col:
            render_metric_card(label, value, help_text)

    if backend_ok:
        st.info("后端服务连接正常。")
    else:
        st.error(health_error or "后端服务连接失败，请检查 FastAPI 是否启动。")
    if logs_error:
        st.warning(f"日志接口不可用：{logs_error}")

    st.subheader("系统流程概览")
    render_process_flow()

    st.subheader("系统能力概览")
    render_capability_cards()

    st.subheader("最近问答记录")
    if logs:
        st.dataframe(_summary_log_rows(logs, limit=5), use_container_width=True, hide_index=True)
    else:
        st.info("暂无问答记录，请先上传文档并进行问答测试。")


def render_document_page(api_base_url: str) -> None:
    render_page_header(
        "文档管理",
        "上传企业文档，保存 document_id，并在当前会话中选择后续摘要和问答使用的文档。",
    )

    render_card("上传区域", "支持 txt、md、docx、pdf。文本型 PDF 可尝试解析，扫描件 OCR 暂不支持。")
    uploaded_file = st.file_uploader("选择文档", type=SUPPORTED_FILE_TYPES)
    if st.button("上传文档", type="primary", disabled=uploaded_file is None, use_container_width=True):
        if uploaded_file is None:
            st.warning("请选择需要上传的文档。")
        else:
            with st.spinner("正在上传并向量化文档，请稍候..."):
                payload, error = upload_document(api_base_url, uploaded_file)
            if error:
                st.error(error)
            elif payload:
                upload_payload = {**payload, "original_file_name": uploaded_file.name}
                st.session_state["last_upload_result"] = upload_payload
                if upload_payload.get("status") == "success" and upload_payload.get("document_id"):
                    document = _normalize_document_entry(upload_payload)
                    documents = st.session_state["uploaded_documents"]
                    existing_document = next(
                        (item for item in documents if item.get("document_id") == document["document_id"]),
                        None,
                    )
                    if isinstance(existing_document, dict):
                        existing_document.update(document)
                    else:
                        documents.append(document)
                    st.session_state["current_document_id"] = document["document_id"]
                    _load_persisted_documents(api_base_url, force=True)
                    st.success("上传成功，文档已加入当前会话。")
                else:
                    st.error(str(upload_payload.get("error_message") or upload_payload.get("message") or "上传失败。"))

    last_upload = st.session_state.get("last_upload_result")
    if isinstance(last_upload, dict):
        last_upload_document = _normalize_document_entry(last_upload)
        st.subheader("最近上传结果")
        _render_status_summary(last_upload)
        st.markdown(
            f"""
<div class="doc-current">
    <strong>文档名称：</strong>{escape(str(last_upload_document.get("display_name", "")))}<br/>
    <strong>文档编号：</strong>{escape(str(last_upload_document.get("document_id", "")))}<br/>
    <strong>存储文件名：</strong>{escape(str(last_upload_document.get("stored_file_name", "")))}<br/>
    <strong>status：</strong>{escape(str(last_upload_document.get("status", "")))}<br/>
    <strong>message：</strong>{escape(str(last_upload_document.get("message", "")))}
</div>
""",
            unsafe_allow_html=True,
        )
        with st.expander("查看上传接口返回"):
            st.json(last_upload)

    refresh_col, _ = st.columns([1, 5])
    with refresh_col:
        if st.button("刷新文档列表", use_container_width=True):
            _load_persisted_documents(api_base_url, force=True, show_error=True)

    documents = _document_options()
    st.subheader("已上传文档")
    if not documents:
        st.info("暂无已上传文档。请先上传 txt、md、docx 或文本型 PDF。")
        return

    st.dataframe(
        [
            {
                "文档名称": doc.get("display_name", ""),
                "document_id": doc.get("document_id", ""),
                "存储文件名": doc.get("stored_file_name", ""),
                "上传时间": doc.get("upload_time", ""),
                "status": doc.get("status", ""),
                "chunk_count": doc.get("chunk_count", 0),
                "vector_indexed": doc.get("vector_indexed", False),
                "error_message": doc.get("error_message", ""),
            }
            for doc in documents
        ],
        use_container_width=True,
        hide_index=True,
    )

    selected_document_id = _select_document("选择当前文档")
    last_batch_delete = st.session_state.get("last_batch_delete_result")
    if isinstance(last_batch_delete, dict):
        failed_count = _safe_int(last_batch_delete.get("failed_count"))
        deleted_count = _safe_int(last_batch_delete.get("deleted_count"))
        if failed_count:
            st.warning(f"批量删除完成：成功 {deleted_count} 个，失败 {failed_count} 个。")
            failed_results = [
                {
                    "document_id": item.get("document_id", ""),
                    "status": item.get("status", ""),
                    "message": item.get("message", ""),
                }
                for item in last_batch_delete.get("results", [])
                if isinstance(item, dict) and item.get("status") != "success"
            ]
            if failed_results:
                st.dataframe(failed_results, use_container_width=True, hide_index=True)
        elif last_batch_delete.get("status") == "success":
            st.success(f"批量删除成功，共删除 {deleted_count} 个文档。")
        elif last_batch_delete.get("error_message"):
            st.error(str(last_batch_delete.get("error_message")))

    st.markdown("#### 批量删除文档")
    document_by_id = {str(doc.get("document_id")): doc for doc in documents if doc.get("document_id")}
    selected_batch_ids = st.multiselect(
        "选择要批量删除的文档",
        options=list(document_by_id),
        format_func=lambda document_id: _document_label(document_by_id.get(document_id, {})),
    )
    if not selected_batch_ids:
        st.caption("请先选择要删除的文档。")
    batch_confirmed = st.checkbox(
        f"确认批量删除选中的 {len(selected_batch_ids)} 个文档",
        key="confirm-batch-delete",
        disabled=not selected_batch_ids,
    )
    if st.button(
        "批量删除选中文档",
        disabled=not selected_batch_ids or not batch_confirmed,
        use_container_width=True,
    ):
        with st.spinner("正在批量删除文档..."):
            payload, error = batch_delete_documents(api_base_url, selected_batch_ids)
        if error:
            st.session_state["last_batch_delete_result"] = {
                "status": "error",
                "deleted_count": 0,
                "failed_count": len(selected_batch_ids),
                "results": [],
                "error_message": error,
            }
        elif payload:
            st.session_state["last_batch_delete_result"] = payload
            if st.session_state.get("current_document_id") in selected_batch_ids:
                st.session_state["current_document_id"] = ""
                st.session_state["last_summary_result"] = None
                st.session_state["last_qa_result"] = None
            _load_persisted_documents(api_base_url, force=True, show_error=True)
        else:
            st.session_state["last_batch_delete_result"] = {
                "status": "error",
                "deleted_count": 0,
                "failed_count": len(selected_batch_ids),
                "results": [],
                "error_message": "批量删除接口没有返回数据。",
            }
        st.rerun()

    selected = _find_document(selected_document_id)
    if selected:
        last_reindex = st.session_state.get("last_reindex_result")
        if isinstance(last_reindex, dict) and str(last_reindex.get("document_id") or "") == selected_document_id:
            if last_reindex.get("status") == "success":
                st.success(str(last_reindex.get("message") or "文档重新入库成功"))
            else:
                st.error(str(last_reindex.get("error_message") or last_reindex.get("message") or "重新入库失败。"))
            with st.expander("查看重新入库接口返回"):
                st.json(last_reindex)

        st.markdown(
            f"""
<div class="doc-current">
    <strong>文档名称：</strong>{escape(str(selected.get("display_name", "")))}<br/>
    <strong>文档编号：</strong>{escape(str(selected.get("document_id", "")))}<br/>
    <strong>上传时间：</strong>{escape(str(selected.get("upload_time", "")))}<br/>
    <strong>状态：</strong>{escape(str(selected.get("status", "")))}<br/>
    <strong>chunk_count：</strong>{escape(str(selected.get("chunk_count", 0)))}<br/>
    <strong>vector_indexed：</strong>{escape(str(selected.get("vector_indexed", False)))}<br/>
    <strong>error_message：</strong>{escape(str(selected.get("error_message", "")))}<br/>
    <strong>存储文件名：</strong>{escape(str(selected.get("stored_file_name", "")))}
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("#### 重新入库")
        if st.button("重新入库", disabled=not selected_document_id, use_container_width=True):
            with st.spinner("正在重新解析并向量化文档，请稍候..."):
                payload, error = reindex_document(api_base_url, selected_document_id)
            if error:
                st.session_state["last_reindex_result"] = {
                    "status": "error",
                    "document_id": selected_document_id,
                    "chunk_count": selected.get("chunk_count", 0),
                    "vector_indexed": selected.get("vector_indexed", False),
                    "error_message": error,
                }
            elif payload:
                st.session_state["last_reindex_result"] = payload
                _load_persisted_documents(api_base_url, force=True, show_error=True)
            else:
                st.session_state["last_reindex_result"] = {
                    "status": "error",
                    "document_id": selected_document_id,
                    "chunk_count": 0,
                    "vector_indexed": False,
                    "error_message": "重新入库接口没有返回数据。",
                }
            st.rerun()

        st.markdown("#### 删除文档")
        confirm_key = f"confirm-delete-{selected_document_id}"
        confirmed = st.checkbox(
            f"确认删除文档：{selected.get('display_name', '')}",
            key=confirm_key,
        )
        if st.button("删除文档", disabled=not confirmed, use_container_width=True):
            payload, error = delete_document(api_base_url, selected_document_id)
            if error:
                st.error(error)
                return
            if payload and payload.get("status") == "success":
                st.success("文档已删除。")
                if st.session_state.get("current_document_id") == selected_document_id:
                    st.session_state["current_document_id"] = ""
                    st.session_state["last_summary_result"] = None
                    st.session_state["last_qa_result"] = None
                _load_persisted_documents(api_base_url, force=True, show_error=True)
                st.rerun()
            else:
                message = ""
                if isinstance(payload, dict):
                    message = str(payload.get("error_message") or payload.get("message") or "")
                st.error(message or "删除文档失败。")


def render_summary_page(api_base_url: str) -> None:
    render_page_header("文档摘要", "选择已上传文档，调用摘要接口生成可读的文档摘要。")
    document_id = _select_document()
    if not document_id:
        st.warning("请先上传或选择文档。")
        return

    if st.button("生成摘要", type="primary", use_container_width=True):
        with st.spinner("正在生成摘要..."):
            payload, error = summarize_document(api_base_url, document_id)
        if error:
            st.error(error)
            return
        if not payload:
            st.warning("摘要接口没有返回数据。")
            return
        st.session_state["last_summary_result"] = payload

    payload = st.session_state.get("last_summary_result")
    if not isinstance(payload, dict):
        st.info("请选择文档并点击“生成摘要”。")
        return

    st.subheader("摘要结果")
    _render_status_summary(payload)
    error_message = str(payload.get("error_message") or "")
    if error_message:
        st.error(error_message)
    summary = str(payload.get("summary") or "")
    st.text_area("摘要内容", value=summary or "暂无摘要内容。", height=260)
    with st.expander("查看摘要接口返回"):
        st.json(payload)


def render_qa_page(api_base_url: str) -> None:
    render_page_header("知识库问答", "基于当前选中文档进行提问，展示答案、状态信息和来源片段。")

    left, right = st.columns([0.95, 1.35], gap="large")
    with left:
        render_card("提问设置", "请选择当前文档，输入问题，或点击示例问题快速填充。")
        document_id = _select_document("当前文档")
        if not document_id:
            st.warning("没有选择文档时无法提问，请先上传文档。")

        st.markdown("**示例问题**")
        for question in EXAMPLE_QUESTIONS:
            if st.button(question, key=f"example-{question}", use_container_width=True):
                st.session_state["qa_question"] = question

        question = st.text_area(
            "问题",
            key="qa_question",
            height=132,
            placeholder="例如：设备启动前需要检查什么？",
        )
        can_submit = bool(document_id and question.strip())
        if st.button("提交问题", type="primary", disabled=not can_submit, use_container_width=True):
            request_document_id, selected_document = _resolve_qa_document_id(document_id)
            selected_display_name = _document_label(selected_document) if selected_document else ""
            _log_qa_submit(selected_display_name, document_id, request_document_id)
            with st.spinner("正在检索文档并生成答案..."):
                payload, error = ask_question(api_base_url, request_document_id, question.strip())
            if error:
                st.session_state["last_qa_result"] = {"status": "error", "error_message": error, "answer": ""}
            elif payload:
                st.session_state["last_qa_result"] = payload
            else:
                st.session_state["last_qa_result"] = {
                    "status": "error",
                    "error_message": "问答接口没有返回数据。",
                    "answer": "",
                }

    with right:
        st.subheader("回答结果")
        payload = st.session_state.get("last_qa_result")
        if not isinstance(payload, dict):
            st.info("提交问题后，这里会展示回答、状态和来源片段。")
            return

        _render_status_summary(payload)
        request_id = str(payload.get("request_id") or "")
        if request_id:
            st.caption(f"request_id：{request_id}")
        st.caption(retrieval_source_label(get_retrieval_source(payload)))

        status = str(payload.get("status") or "")
        answer = str(payload.get("answer") or "")
        error_message = str(payload.get("error_message") or "")
        if status == "success":
            st.markdown(
                f"""
<div class="card">
    <div class="card-title">回答</div>
    <p class="card-desc">{escape(answer or "暂无回答内容。")}</p>
</div>
""",
                unsafe_allow_html=True,
            )
        elif error_message:
            st.error(error_message)
            if answer:
                st.warning(answer)
        else:
            st.warning(answer or "当前问题未得到可用答案。")

        source_chunks = payload.get("source_chunks", [])
        st.markdown("#### 来源片段")
        if isinstance(source_chunks, list) and source_chunks:
            for index, chunk in enumerate(source_chunks, start=1):
                if not isinstance(chunk, dict):
                    continue
                chunk_id = str(chunk.get("chunk_id") or f"chunk-{index}")
                score = chunk.get("score", "")
                title = f"片段 {index} · {chunk_id}"
                if score != "":
                    title += f" · score={score}"
                with st.expander(title, expanded=index == 1):
                    st.markdown(str(chunk.get("text") or ""))
                    metadata = {key: value for key, value in chunk.items() if key != "text"}
                    if metadata:
                        st.caption("片段元数据")
                        st.write(metadata)
        else:
            st.info("本次回答没有返回来源片段。")

        with st.expander("查看问答接口返回"):
            st.json(payload)


def render_logs_page(api_base_url: str) -> None:
    render_page_header("问答日志", "查看最近问答请求，按状态筛选，并展开单条记录查看详情。")
    top_cols = st.columns([1, 1, 4])
    limit = top_cols[0].selectbox("日志条数", [10, 20, 50, 100, 200], index=2)
    if top_cols[1].button("刷新日志", use_container_width=True):
        st.rerun()

    logs, error = fetch_logs(api_base_url, limit=int(limit))
    if error:
        st.error(error)
        return
    if not logs:
        st.info("暂无日志记录。")
        return

    statuses = ["全部"] + sorted({str(item.get("status", "")) for item in logs if item.get("status")})
    selected_status = st.selectbox("按状态筛选", statuses)
    filtered_logs = logs if selected_status == "全部" else [
        item for item in logs if item.get("status") == selected_status
    ]

    table_rows = [
        {
            "request_id": item.get("request_id", ""),
            "document_id": item.get("document_id", ""),
            "question": item.get("question", ""),
            "status": item.get("status", ""),
            "retrieval_source": item.get("retrieval_source", ""),
            "elapsed_time": item.get("elapsed_time", 0),
            "error_message": item.get("error_message", ""),
            "timestamp": item.get("timestamp", ""),
        }
        for item in filtered_logs
    ]
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    st.subheader("日志详情")
    for index, item in enumerate(filtered_logs, start=1):
        question = str(item.get("question") or "未记录问题")
        status = str(item.get("status") or "unknown")
        with st.expander(f"{index}. {status} · {question}"):
            st.json(item)


def render_about_page() -> None:
    render_page_header("企业文档问答系统", "了解系统定位、技术架构、当前支持范围和后续优化方向。")

    sections = [
        (
            "项目定位",
            [
                "本项目面向企业内部资料查询场景，支持文档上传、内容解析、知识检索、答案生成、来源追踪和日志记录。",
                "当前为后端原型配套演示页面，用于功能验证与项目展示。",
            ],
        ),
        (
            "技术架构",
            [
                "系统后端采用 LangGraph 组织文档处理、知识检索、答案生成、答案校验和日志记录流程。",
                "不同处理环节拆分为独立 Agent 模块，包括 DocumentRouterAgent、DocumentAgent、RetrievalAgent、AnswerAgent、DocumentValidatorAgent 和 LogAgent。",
            ],
        ),
        (
            "当前支持",
            [
                "txt、md、docx、文本型 PDF。",
                "文档上传、文档摘要、文档问答。",
                "来源片段展示和问答日志记录。",
            ],
        ),
        (
            "当前限制",
            [
                "暂不支持扫描件 OCR。",
                "当前检索为基础关键词或简单相似度检索。",
                "暂未接入生产级向量数据库。",
                "暂未实现登录、权限管理和多用户管理。",
                "当前页面是演示端，不是生产级前端。",
            ],
        ),
        (
            "后续优化方向",
            [
                "接入 bge-m3 向量化。",
                "接入 Chroma 或 FAISS。",
                "加入 Rerank。",
                "优化 Prompt。",
                "接入外部 App。",
                "增加权限管理和文档版本管理。",
            ],
        ),
    ]

    for row_start in range(0, len(sections), 2):
        cols = st.columns(2)
        for col, (title, items) in zip(cols, sections[row_start : row_start + 2], strict=False):
            with col:
                body = "".join(f"<li>{escape(item)}</li>" for item in items)
                st.markdown(
                    f"""
<div class="card">
    <div class="card-title">{escape(title)}</div>
    <ul class="card-desc">{body}</ul>
</div>
""",
                    unsafe_allow_html=True,
                )


def render_sidebar() -> tuple[str, str, dict[str, Any] | None, str]:
    st.sidebar.markdown('<div class="app-sidebar-title">企业文档问答系统</div>', unsafe_allow_html=True)
    api_base_url = get_api_base_url()
    health, health_error = check_health(api_base_url)
    page = st.sidebar.radio("功能导航", NAVIGATION_PAGES)
    st.sidebar.divider()
    st.sidebar.caption("后端地址")
    st.sidebar.code(api_base_url, language=None)
    if health and health.get("status") == "ok":
        st.sidebar.success("后端服务正常")
    else:
        st.sidebar.error("后端服务连接失败，请检查 FastAPI 是否启动")
        if health_error:
            st.sidebar.caption(health_error)
    st.sidebar.markdown(
        '<div class="app-sidebar-footer">演示版本：用于功能验证与项目展示</div>',
        unsafe_allow_html=True,
    )
    return page, api_base_url, health, health_error


def main() -> None:
    st.set_page_config(
        page_title="企业文档问答系统",
        page_icon="📚",
        layout="wide",
    )
    inject_custom_css()
    _init_state()

    page, api_base_url, health, health_error = render_sidebar()
    _load_persisted_documents(api_base_url)

    if page == "系统概览":
        render_overview_page(api_base_url, health, health_error)
    elif page == "文档管理":
        render_document_page(api_base_url)
    elif page == "文档摘要":
        render_summary_page(api_base_url)
    elif page == "知识库问答":
        render_qa_page(api_base_url)
    elif page == "问答日志":
        render_logs_page(api_base_url)
    else:
        render_about_page()


if __name__ == "__main__":
    main()
