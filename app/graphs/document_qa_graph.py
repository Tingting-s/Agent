from __future__ import annotations

import logging
from time import perf_counter
from typing import Any
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from app.agents.answer_agent import AnswerAgent
from app.agents.answer_agent import AnswerGenerationTimeout
from app.agents.document_agent import DocumentAgent
from app.agents.routing import DocumentRouterAgent
from app.agents.validation import DocumentValidatorAgent
from app.agents.log_agent import LogAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.graphs.state import DocumentQAState
from app.services.qa_log_service import QALogService
from app.services.llm_service import LLM_TIMEOUT_MESSAGE


LOGGER = logging.getLogger(__name__)


class DocumentQAGraph:
    def __init__(
        self,
        *,
        router_agent: DocumentRouterAgent | None = None,
        document_agent: DocumentAgent | None = None,
        retrieval_agent: RetrievalAgent | None = None,
        answer_agent: AnswerAgent | None = None,
        validator_agent: DocumentValidatorAgent | None = None,
        log_agent: LogAgent | None = None,
        log_service: QALogService | None = None,
    ) -> None:
        self.router_agent = router_agent or DocumentRouterAgent()
        self.document_agent = document_agent or DocumentAgent()
        self.retrieval_agent = retrieval_agent or RetrievalAgent()
        self.answer_agent = answer_agent or AnswerAgent()
        self.validator_agent = validator_agent or DocumentValidatorAgent()
        self.log_agent = log_agent or LogAgent(log_service=log_service)
        self.graph = self._build_graph()

    def invoke(self, state: DocumentQAState) -> DocumentQAState:
        initial_state = dict(state)
        initial_state.setdefault("request_id", str(uuid4()))
        initial_state.setdefault("retry_count", 0)
        initial_state.setdefault("elapsed_time", 0.0)
        initial_state.setdefault("error_message", "")
        initial_state.setdefault("status", "received")
        initial_state.setdefault("chunks", [])
        initial_state.setdefault("retrieved_chunks", [])
        initial_state.setdefault("source_chunks", [])
        initial_state.setdefault("metadata", {})
        self._log_stage("received_question", "started", initial_state)
        result = self.graph.invoke(initial_state)
        self._log_stage("qa_total", "completed", result, elapsed_time=self._elapsed(result))
        return result

    def _build_graph(self):  # type: ignore[no-untyped-def]
        graph = StateGraph(DocumentQAState)
        graph.add_node("input_node", self._input_node)
        graph.add_node("router_node", self._router_node)
        graph.add_node("document_node", self._document_node)
        graph.add_node("retrieval_node", self._retrieval_node)
        graph.add_node("answer_node", self._answer_node)
        graph.add_node("validator_node", self._validator_node)
        graph.add_node("log_node", self._log_node)
        graph.add_node("output_node", self._output_node)
        graph.add_node("error_node", self._error_node)

        graph.add_edge(START, "input_node")
        graph.add_edge("input_node", "router_node")
        graph.add_conditional_edges(
            "router_node",
            self._after_router,
            {"document_node": "document_node", "error_node": "error_node"},
        )
        graph.add_conditional_edges(
            "document_node",
            self._after_document,
            {"retrieval_node": "retrieval_node", "answer_node": "answer_node", "error_node": "error_node"},
        )
        graph.add_conditional_edges(
            "retrieval_node",
            self._after_retrieval,
            {"answer_node": "answer_node", "error_node": "error_node"},
        )
        graph.add_conditional_edges(
            "answer_node",
            self._after_answer,
            {"validator_node": "validator_node", "error_node": "error_node"},
        )
        graph.add_edge("validator_node", "log_node")
        graph.add_edge("error_node", "log_node")
        graph.add_edge("log_node", "output_node")
        graph.add_edge("output_node", END)
        return graph.compile()

    def _input_node(self, state: DocumentQAState) -> dict[str, Any]:
        try:
            return {
                "request_id": str(state.get("request_id") or uuid4()),
                "question": str(state.get("question") or "").strip(),
                "file_path": str(state.get("file_path") or "").strip(),
                "document_id": state.get("document_id"),
                "chunk_size": int(state.get("chunk_size") or 800),
                "chunk_overlap": int(state.get("chunk_overlap") or 100),
                "top_k": int(state.get("top_k") or 5),
                "retry_count": int(state.get("retry_count") or 0),
                "start_time": float(state.get("start_time") or perf_counter()),
                "status": "input_received",
            }
        except Exception as exc:
            return self._node_error("input_node", exc)

    def _router_node(self, state: DocumentQAState) -> dict[str, Any]:
        try:
            decision = self.router_agent.route_document_request(
                question=state.get("question"),
                file_path=state.get("file_path"),
                document_id=state.get("document_id"),
                task_type=state.get("task_type"),
            )
            updates: dict[str, Any] = {
                "task_type": decision.task_type,
                "status": "routed",
                "metadata": {**dict(state.get("metadata") or {}), "route_reason": decision.reason},
            }
            if decision.task_type == "invalid_request":
                updates["status"] = "error"
                updates["error_message"] = decision.reason
            return updates
        except Exception as exc:
            return self._node_error("router_node", exc)

    def _document_node(self, state: DocumentQAState) -> dict[str, Any]:
        try:
            result = self.document_agent.parse(
                file_path=str(state.get("file_path") or ""),
                document_id=state.get("document_id"),
                chunk_size=int(state.get("chunk_size") or 800),
                chunk_overlap=int(state.get("chunk_overlap") or 100),
            )
            if result.get("status") == "error":
                return {
                    "status": "error",
                    "error_message": str(result.get("error_message") or "Document parsing failed."),
                    "file_name": result.get("file_name", ""),
                }
            return {
                "file_path": str(result.get("file_path") or state.get("file_path") or ""),
                "file_name": str(result.get("file_name") or ""),
                "parsed_text": str(result.get("parsed_text") or ""),
                "chunks": list(result.get("chunks") or []),
                "summary": str(result.get("summary") or ""),
                "key_points": list(result.get("key_points") or []),
                "risks": list(result.get("risks") or []),
                "status": "document_parsed",
                "metadata": {**dict(state.get("metadata") or {}), **dict(result.get("metadata") or {})},
            }
        except Exception as exc:
            return self._node_error("document_node", exc)

    def _retrieval_node(self, state: DocumentQAState) -> dict[str, Any]:
        try:
            retrieval_start = perf_counter()
            retrieval_kwargs = {
                "question": str(state.get("question") or ""),
                "chunks": list(state.get("chunks") or []),
                "top_k": int(state.get("top_k") or 5),
            }
            try:
                retrieval_result = self.retrieval_agent.retrieve_with_source(
                    **retrieval_kwargs,
                    request_id=str(state.get("request_id") or ""),
                )
            except TypeError as exc:
                if "request_id" not in str(exc):
                    raise
                retrieval_result = self.retrieval_agent.retrieve_with_source(**retrieval_kwargs)
            retrieved_chunks = list(retrieval_result.get("chunks") or [])
            retrieval_source = str(retrieval_result.get("retrieval_source") or "fallback")
            fallback_reason = str(retrieval_result.get("fallback_reason") or "")
            retrieval_elapsed = float(
                retrieval_result.get("retrieval_elapsed")
                or round(perf_counter() - retrieval_start, 6)
            )
            return {
                "retrieved_chunks": retrieved_chunks,
                "source_chunks": retrieved_chunks,
                "status": "retrieved",
                "metadata": {
                    **dict(state.get("metadata") or {}),
                    "retrieval_source": retrieval_source,
                    "fallback_reason": fallback_reason,
                    "retrieval_elapsed": retrieval_elapsed,
                    "chroma_elapsed": retrieval_result.get("chroma_elapsed"),
                    "rerank_elapsed": retrieval_result.get("rerank_elapsed"),
                    "rerank_enabled": retrieval_result.get("rerank_enabled"),
                    "rerank_top_n": retrieval_result.get("rerank_top_n"),
                    "embedding_model": retrieval_result.get("embedding_model"),
                },
            }
        except Exception as exc:
            return self._node_error("retrieval_node", exc)

    def _answer_node(self, state: DocumentQAState) -> dict[str, Any]:
        try:
            if state.get("task_type") == "document_summary":
                answer_payload = self.answer_agent.generate_summary_answer(
                    summary=str(state.get("summary") or ""),
                    key_points=list(state.get("key_points") or []),
                    risks=list(state.get("risks") or []),
                )
                prompt = "Summarize the parsed enterprise document."
            else:
                prompt_start = perf_counter()
                self._log_stage("prompt_build", "started", state)
                prompt = self.answer_agent.build_prompt(
                    question=str(state.get("question") or ""),
                    context_chunks=list(state.get("retrieved_chunks") or []),
                    summary=str(state.get("summary") or ""),
                )
                prompt_build_elapsed = round(perf_counter() - prompt_start, 6)
                self._log_stage("prompt_build", "completed", state, elapsed_time=prompt_build_elapsed)
                llm_start = perf_counter()
                self._log_stage("llm_generate", "started", state)
                answer_payload = self.answer_agent.generate_answer(
                    question=str(state.get("question") or ""),
                    context_chunks=list(state.get("retrieved_chunks") or []),
                    summary=str(state.get("summary") or ""),
                    request_id=str(state.get("request_id") or ""),
                    document_id=str(state.get("document_id") or ""),
                    retrieval_source=str(dict(state.get("metadata") or {}).get("retrieval_source") or ""),
                )
                llm_elapsed = round(perf_counter() - llm_start, 6)
                self._log_stage("llm_generate", "completed", state, elapsed_time=llm_elapsed)

            return {
                "prompt": prompt,
                "answer": str(answer_payload.get("answer") or ""),
                "status": "answered",
                "metadata": {
                    **dict(state.get("metadata") or {}),
                    "answer_source": answer_payload.get("source", "unknown"),
                    "answer_confidence": answer_payload.get("confidence", 0.0),
                    "prompt_build_elapsed": locals().get("prompt_build_elapsed"),
                    "llm_elapsed": locals().get("llm_elapsed"),
                },
            }
        except AnswerGenerationTimeout as exc:
            llm_elapsed = round(perf_counter() - llm_start, 6) if "llm_start" in locals() else None
            self._log_stage("llm_generate", "timeout", state, elapsed_time=llm_elapsed or self._elapsed(state))
            updates = self._node_error("answer_node", exc)
            updates["metadata"] = {
                **dict(state.get("metadata") or {}),
                **dict(updates.get("metadata") or {}),
                "prompt_build_elapsed": locals().get("prompt_build_elapsed"),
                "llm_elapsed": llm_elapsed,
            }
            return updates
        except Exception as exc:
            return self._node_error("answer_node", exc)

    def _validator_node(self, state: DocumentQAState) -> dict[str, Any]:
        try:
            validation_start = perf_counter()
            validation_result = self.validator_agent.validate_document_answer(
                answer=str(state.get("answer") or ""),
                retrieved_chunks=list(state.get("retrieved_chunks") or []),
                task_type=str(state.get("task_type") or ""),
                retrieval_source=str(dict(state.get("metadata") or {}).get("retrieval_source") or ""),
                relevance_sufficient=bool(list(state.get("retrieved_chunks") or [])),
            )
            status = "success" if validation_result["is_valid"] else "error"
            validation_message = str(validation_result.get("error_message") or "")
            return {
                "validation_result": validation_result,
                "status": status,
                "error_message": "" if validation_result["is_valid"] else validation_message,
                "metadata": {
                    **dict(state.get("metadata") or {}),
                    "validation_elapsed": round(perf_counter() - validation_start, 6),
                    "validation_status": status,
                    "validation_message": validation_message,
                },
            }
        except Exception as exc:
            return self._node_error("validator_node", exc)

    def _log_node(self, state: DocumentQAState) -> dict[str, Any]:
        try:
            log_start = perf_counter()
            self._log_stage("log_record", "started", state)
            elapsed_time = self._elapsed(state)
            log_state = dict(state)
            log_state["elapsed_time"] = elapsed_time
            log_state["metadata"] = {
                **dict(state.get("metadata") or {}),
                "total_elapsed": elapsed_time,
            }
            result = self.log_agent.record(state=log_state)
            self._log_stage("log_record", "completed", state, elapsed_time=round(perf_counter() - log_start, 6))
            return {
                "log_id": result.get("log_id"),
                "elapsed_time": elapsed_time,
            }
        except Exception:
            return {
                "elapsed_time": self._elapsed(state),
            }

    def _output_node(self, state: DocumentQAState) -> dict[str, Any]:
        try:
            validation = dict(
                state.get("validation_result")
                or {
                    "is_valid": False,
                    "confidence": 0.0,
                    "issues": [state.get("error_message") or "unknown_error"],
                    "error_message": state.get("error_message") or "unknown_error",
                }
            )
            status = str(state.get("status") or "error")
            response = {
                "document_id": state.get("document_id"),
                "file_name": state.get("file_name") or "",
                "file_path": state.get("file_path") or "",
                "status": status,
                "message": self._response_message(state),
                "question": state.get("question") or "",
                "answer": state.get("answer") or "",
                "summary": state.get("summary") or "",
                "sources": list(state.get("source_chunks") or state.get("retrieved_chunks") or []),
                "validation": validation,
                "log_id": state.get("log_id"),
                "metadata": {
                    **dict(state.get("metadata") or {}),
                    "request_id": state.get("request_id"),
                    "task_type": state.get("task_type"),
                    "elapsed_time": state.get("elapsed_time", 0.0),
                    "error_message": state.get("error_message") or "",
                    "retry_count": state.get("retry_count", 0),
                },
            }
            return {"response": response}
        except Exception as exc:
            return {
                "response": {
                    "document_id": state.get("document_id"),
                    "file_name": state.get("file_name") or "",
                    "file_path": state.get("file_path") or "",
                    "status": "error",
                    "message": "Document QA output packaging failed.",
                    "question": state.get("question") or "",
                    "answer": "",
                    "summary": state.get("summary") or "",
                    "sources": [],
                    "validation": {
                        "is_valid": False,
                        "confidence": 0.0,
                        "issues": ["output_node_failed"],
                    },
                    "log_id": state.get("log_id"),
                    "metadata": {"error_message": self._merge_error(state, "output_node", exc)},
                }
            }

    def _error_node(self, state: DocumentQAState) -> dict[str, Any]:
        return {
            "status": "timeout" if state.get("status") == "timeout" else "error",
            "answer": state.get("answer") or "",
            "source_chunks": list(state.get("source_chunks") or []),
            "validation_result": {
                "is_valid": False,
                "confidence": 0.0,
                "issues": [state.get("error_message") or "unknown_error"],
                "error_message": state.get("error_message") or "unknown_error",
            },
            "elapsed_time": self._elapsed(state),
        }

    def _after_router(self, state: DocumentQAState) -> str:
        return "error_node" if state.get("status") == "error" else "document_node"

    def _after_document(self, state: DocumentQAState) -> str:
        if state.get("status") == "error":
            return "error_node"
        if state.get("task_type") == "document_summary":
            return "answer_node"
        return "retrieval_node"

    def _after_retrieval(self, state: DocumentQAState) -> str:
        return "error_node" if state.get("status") == "error" else "answer_node"

    def _after_answer(self, state: DocumentQAState) -> str:
        return "error_node" if state.get("status") in {"error", "timeout"} else "validator_node"

    def _node_error(self, node_name: str, exc: Exception) -> dict[str, Any]:
        if node_name == "answer_node" and self._is_timeout_error(exc):
            return {
                "status": "timeout",
                "answer": LLM_TIMEOUT_MESSAGE,
                "error_message": LLM_TIMEOUT_MESSAGE,
                "metadata": {"answer_source": "timeout", "failed_stage": "llm_generate"},
            }
        return {
            "status": "error",
            "error_message": self._public_error(node_name, exc),
            "metadata": {"failed_stage": node_name},
        }

    def _merge_error(self, state: DocumentQAState, node_name: str, exc: Exception) -> str:
        existing = str(state.get("error_message") or "").strip()
        current = self._public_error(node_name, exc)
        return f"{existing}; {current}" if existing else current

    def _elapsed(self, state: DocumentQAState) -> float:
        start_time = float(state.get("start_time") or perf_counter())
        return round(perf_counter() - start_time, 6)

    def _response_message(self, state: DocumentQAState) -> str:
        if state.get("status") == "success":
            if state.get("task_type") == "document_summary":
                return "Document summary generated successfully."
            return "Document question answered successfully."
        error_message = str(state.get("error_message") or "").strip()
        return error_message or "Document QA request failed."

    def _public_error(self, node_name: str, exc: Exception) -> str:
        if node_name == "document_node":
            return str(exc) or "Document parsing failed."
        if node_name == "retrieval_node":
            return "Document retrieval failed."
        if node_name == "answer_node":
            if self._is_timeout_error(exc):
                return LLM_TIMEOUT_MESSAGE
            return "Answer generation failed."
        if node_name == "validator_node":
            return "Answer validation failed."
        if node_name == "router_node":
            return "Invalid document QA request."
        return "Document QA request failed."

    def _is_timeout_error(self, exc: Exception) -> bool:
        if isinstance(exc, (TimeoutError, AnswerGenerationTimeout)):
            return True
        name = type(exc).__name__.lower()
        message = str(exc).lower()
        return "timeout" in name or "timed out" in message or LLM_TIMEOUT_MESSAGE in str(exc)

    def _log_stage(
        self,
        stage_name: str,
        event: str,
        state: dict[str, Any],
        *,
        elapsed_time: float = 0.0,
    ) -> None:
        metadata = dict(state.get("metadata") or {})
        retrieval_source = str(metadata.get("retrieval_source") or "")
        source_chunks_count = len(list(state.get("source_chunks") or state.get("retrieved_chunks") or []))
        LOGGER.info(
            "QA stage %s: request_id=%s document_id=%s question=%s stage_name=%s elapsed_time=%s retrieval_source=%s source_chunks_count=%s",
            event,
            state.get("request_id") or "",
            state.get("document_id") or "",
            state.get("question") or "",
            stage_name,
            elapsed_time,
            retrieval_source,
            source_chunks_count,
        )
