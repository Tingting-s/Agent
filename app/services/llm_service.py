from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from typing import Any

from openai import OpenAI
from pydantic import ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from app.config import get_settings
from app.schemas.tasks import DocumentSummaryResult


LOGGER = logging.getLogger(__name__)
LLM_TIMEOUT_MESSAGE = "答案生成超时，请稍后重试。"


def default_retry_decorator(attempts: int = 3, wait_seconds: int = 1, retry_on: type[Exception] = Exception):
    return retry(
        stop=stop_after_attempt(attempts),
        wait=wait_fixed(wait_seconds),
        retry=retry_if_exception_type(retry_on),
        reraise=True,
    )


class LLMServiceError(RuntimeError):
    """Base exception for LLM service failures."""


class LLMRequestError(LLMServiceError):
    """Raised when the remote LLM request fails."""


class LLMTimeoutError(LLMServiceError):
    """Raised when the remote LLM request times out."""


class LLMResponseParseError(LLMServiceError):
    """Raised when the model response cannot be parsed or validated."""


class QwenLLMService:
    def __init__(self, client: Any | None = None) -> None:
        self.settings = get_settings()
        self.model_name = self.settings.QWEN_MODEL
        self.timeout = self.settings.LLM_TIMEOUT
        self._client = client

    def summarize_document(self, content: str) -> dict[str, Any]:
        payload = self._request_json(
            system_prompt=(
                "You are an office assistant that summarizes documents. "
                "Return JSON only. "
                'Required top-level fields: "summary", "key_points", "risks". '
                'The "key_points" field must be a list of concise strings. '
                'The "risks" field must be a list of concise strings.'
            ),
            user_prompt=(
                "Summarize the following document and return valid JSON only:\n"
                f"{content}"
            ),
        )
        return self._validate_payload(DocumentSummaryResult, payload, "Document summary")

    def answer_document_question(self, *, question: str, context: str, summary: str = "") -> dict[str, Any]:
        payload = self._request_json(
            system_prompt=(
                "You are an enterprise document QA assistant. Return JSON only. "
                "Answer strictly from the provided retrieved snippets. "
                "Only answer the user's question; do not summarize the whole document. "
                "Do not include unrelated document content. "
                "Do not invent enterprise policies, device parameters, prices, after-sales policies, "
                "or training requirements. "
                "Keep the answer concise, preferably 2-5 bullets. "
                "Do not restate snippets paragraph by paragraph. "
                'Required fields: "answer", "confidence". '
                'If the snippets do not contain clear evidence, answer exactly: '
                '"当前文档中未检索到与问题相关的内容。".'
            ),
            user_prompt=(
                f"Question: {question}\n"
                f"Retrieved snippets:\n{context}\n"
                "Return valid JSON only."
            ),
        )
        answer = str(payload.get("answer") or "").strip()
        confidence = payload.get("confidence", 0.0)
        try:
            normalized_confidence = float(confidence)
        except (TypeError, ValueError):
            normalized_confidence = 0.0
        return {
            "answer": answer,
            "confidence": max(0.0, min(1.0, normalized_confidence)),
            "source": "llm",
        }

    def _get_client(self) -> Any:
        if self._client is None:
            api_key = self.settings.DASHSCOPE_API_KEY.strip()
            if not api_key:
                raise LLMServiceError(
                    "DASHSCOPE_API_KEY is not configured. Please update your .env file before using text tools."
                )
            self._client = OpenAI(
                api_key=api_key,
                base_url=self.settings.DASHSCOPE_BASE_URL,
                timeout=self.timeout,
            )
        return self._client

    @default_retry_decorator(attempts=3, wait_seconds=1, retry_on=LLMRequestError)
    def _create_chat_completion(self, messages: list[dict[str, str]]) -> str:
        try:
            response = self._get_client().chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.2,
                timeout=self.timeout,
            )
        except LLMServiceError:
            raise
        except Exception as exc:
            if self._is_timeout_exception(exc):
                LOGGER.warning(
                    "LLM request timed out: model=%s timeout=%s error_type=%s error=%s",
                    self.model_name,
                    self.timeout,
                    type(exc).__name__,
                    exc,
                )
                raise LLMTimeoutError(LLM_TIMEOUT_MESSAGE) from exc
            raise LLMRequestError(
                f"Failed to call Qwen model '{self.model_name}' via DashScope: {exc}"
            ) from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise LLMRequestError("The Qwen API response did not include a valid message payload.") from exc

        if not isinstance(content, str) or not content.strip():
            raise LLMRequestError("The Qwen model returned an empty response.")
        return content

    def _request_json(self, *, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        raw_content = self._create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return self._parse_json_response(raw_content)

    def _parse_json_response(self, raw_content: str) -> dict[str, Any]:
        cleaned_content = self._strip_markdown_code_fence(raw_content).strip()
        candidates = [cleaned_content]

        start_index = cleaned_content.find("{")
        end_index = cleaned_content.rfind("}")
        if start_index != -1 and end_index != -1 and end_index > start_index:
            extracted = cleaned_content[start_index : end_index + 1]
            if extracted != cleaned_content:
                candidates.append(extracted)

        for candidate in candidates:
            if not candidate:
                continue
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload

        raise LLMResponseParseError(
            "The Qwen model returned content that could not be parsed as JSON. "
            "Please try again with clearer input."
        )

    def _strip_markdown_code_fence(self, raw_content: str) -> str:
        match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", raw_content.strip(), flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1)
        return raw_content

    def _is_timeout_exception(self, exc: Exception) -> bool:
        if isinstance(exc, TimeoutError):
            return True
        name = type(exc).__name__.lower()
        message = str(exc).lower()
        return "timeout" in name or "timed out" in message or "read timed out" in message

    def _validate_payload(
        self,
        schema_model: type[DocumentSummaryResult],
        payload: dict[str, Any],
        label: str,
    ) -> dict[str, Any]:
        try:
            validated = schema_model.model_validate(payload)
        except ValidationError as exc:
            missing_fields = sorted(
                {
                    ".".join(str(part) for part in error["loc"])
                    for error in exc.errors()
                    if error.get("loc")
                }
            )
            raise LLMResponseParseError(
                f"{label} response is missing required JSON fields or contains invalid values: "
                f"{', '.join(missing_fields)}."
            ) from exc
        return validated.model_dump()


@lru_cache
def get_qwen_llm_service() -> QwenLLMService:
    return QwenLLMService()
