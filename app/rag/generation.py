from __future__ import annotations

import logging
from typing import Any

from app.services.llm_service import LLMServiceError, get_qwen_llm_service


NO_RETRIEVAL_CONTENT_MESSAGE = "当前文档中未检索到与问题相关的内容。"
MAX_PROMPT_CHUNKS = 3
MAX_PROMPT_CHUNK_CHARS = 800
LOGGER = logging.getLogger(__name__)


def build_document_qa_prompt(*, question: str, retrieved_chunks: list[dict[str, Any]]) -> str:
    if not retrieved_chunks:
        return NO_RETRIEVAL_CONTENT_MESSAGE

    context_blocks = []
    for index, chunk in enumerate(retrieved_chunks[:MAX_PROMPT_CHUNKS], start=1):
        file_name = str(chunk.get("file_name") or "")
        chunk_id = str(chunk.get("chunk_id") or f"chunk-{index}")
        text = str(chunk.get("text") or "").strip()[:MAX_PROMPT_CHUNK_CHARS]
        if not text:
            continue
        context_blocks.append(f"[{index}] file={file_name} chunk={chunk_id}\n{text}")

    if not context_blocks:
        LOGGER.info("Prompt build skipped: final_chunk_count=0 reason=empty_chunk_text")
        return NO_RETRIEVAL_CONTENT_MESSAGE

    LOGGER.info(
        "Prompt built: final_chunk_count=%s chunk_ids=%s question=%s",
        len(context_blocks),
        [
            str(chunk.get("chunk_id") or "")
            for chunk in retrieved_chunks[:MAX_PROMPT_CHUNKS]
            if str(chunk.get("text") or "").strip()
        ],
        question,
    )
    context = "\n\n".join(context_blocks)
    return (
        "你是企业文档问答助手。必须严格遵守以下规则：\n"
        "1. 只能根据【检索片段】回答用户问题。\n"
        "2. 只回答用户提出的问题，不要总结全文。\n"
        "3. 不要输出与问题无关的文档内容。\n"
        f"4. 如果检索片段中没有明确依据，请回答：“{NO_RETRIEVAL_CONTENT_MESSAGE}”。\n"
        "5. 禁止编造企业制度、设备参数、价格、售后政策、培训要求等信息。\n"
        "6. 回答控制在 2～5 条要点以内；如果一句话足够，可以只回答一句。\n"
        "7. 不要把检索片段逐段复述出来。\n"
        "8. 如果问题只问一个点，只回答这个点；如果问题涉及多个方面，再分点回答。\n\n"
        f"用户问题：{question}\n\n"
        f"【检索片段】\n{context}\n\n"
        "请基于上述检索片段给出答案。"
    )


def summarize_document_text(text: str, max_length: int | None = None) -> dict[str, Any]:
    normalized = text.strip()
    if not normalized:
        return {"summary": "", "key_points": [], "risks": []}

    try:
        return get_qwen_llm_service().summarize_document(normalized)
    except LLMServiceError:
        return _fallback_summary(normalized, max_length=max_length)


def _fallback_summary(content: str, *, max_length: int | None = None) -> dict[str, Any]:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    flattened = " ".join(lines)
    summary_length = max_length if max_length is not None else 300
    summary = flattened[:summary_length] if flattened else "Empty document."
    key_points = lines[:5]
    risks = [
        line
        for line in lines
        if any(keyword in line.lower() for keyword in ("risk", "issue", "blocker", "warning"))
    ][:5]
    return {
        "summary": summary,
        "key_points": key_points,
        "risks": risks,
    }
