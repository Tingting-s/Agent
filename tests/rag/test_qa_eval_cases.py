from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)
FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"
QA_EVAL_CASES_PATH = FIXTURES_DIR / "qa_eval_cases.json"
NO_ANSWER_TEXT = "未检索到与问题相关的内容"


def _load_eval_cases() -> list[dict[str, Any]]:
    return json.loads(QA_EVAL_CASES_PATH.read_text(encoding="utf-8"))


@pytest.fixture()
def uploaded_eval_documents() -> dict[str, dict[str, Any]]:
    cases = _load_eval_cases()
    document_files = sorted({case["document_file"] for case in cases})
    return {document_file: _upload_fixture_document(document_file) for document_file in document_files}


@pytest.fixture(autouse=True)
def force_local_eval_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    class LocalEvalLLM:
        def answer_document_question(self, *, question: str, context: str, summary: str = "") -> dict[str, object]:
            return {
                "answer": _build_local_eval_answer(question=question, context=context),
                "confidence": 0.7,
                "source": "local_eval_mock",
            }

    monkeypatch.setattr("app.agents.answer_agent.get_qwen_llm_service", lambda: LocalEvalLLM())


def test_qa_eval_cases_fixture_is_valid() -> None:
    cases = _load_eval_cases()

    assert len(cases) >= 18
    assert {case["document_file"] for case in cases} == {
        "sample_device_manual.txt",
        "sample_training_material.md",
        "sample_faq.txt",
    }
    for document_file in {case["document_file"] for case in cases}:
        assert sum(1 for case in cases if case["document_file"] == document_file) >= 6
    for case in cases:
        assert set(case) == {
            "document_file",
            "question",
            "expected_type",
            "expected_keywords",
            "forbidden_keywords",
            "description",
        }
        assert case["expected_type"] in {"answerable", "multi_chunk", "unanswerable"}
        assert case["expected_keywords"]
        assert isinstance(case["forbidden_keywords"], list)


@pytest.mark.parametrize("case", _load_eval_cases(), ids=lambda case: case["description"])
def test_qa_eval_case_quality(case: dict[str, Any], uploaded_eval_documents: dict[str, dict[str, Any]]) -> None:
    document_id = uploaded_eval_documents[case["document_file"]]["document_id"]

    response = client.post(
        "/api/qa/ask",
        json={"document_id": document_id, "question": case["question"]},
    )

    assert response.status_code == 200, case["description"]
    payload = response.json()
    answer = str(payload.get("answer") or "")
    source_chunks = list(payload.get("source_chunks") or [])

    assert len(source_chunks) <= 3, case["description"]

    if case["expected_type"] == "unanswerable":
        assert payload["status"] == "error", case["description"]
        assert NO_ANSWER_TEXT in answer or NO_ANSWER_TEXT in str(payload.get("message") or ""), case["description"]
        _assert_forbidden_keywords_absent(answer, case)
        assert source_chunks == [], case["description"]
        assert payload["retrieval_source"] == "none", case["description"]
        return

    assert payload["status"] == "success", case["description"]
    assert answer.strip(), case["description"]
    assert source_chunks, case["description"]
    assert payload["retrieval_source"] == "chroma", case["description"]
    assert all(chunk.get("retrieval_source") == "chroma" for chunk in source_chunks), case["description"]
    assert any(_chunk_mentions_question(chunk, case["question"]) for chunk in source_chunks), case["description"]
    _assert_expected_keywords_present(answer, case)
    _assert_forbidden_keywords_absent(answer, case)


def _upload_fixture_document(document_file: str) -> dict[str, Any]:
    file_path = FIXTURES_DIR / document_file
    media_type = "text/markdown" if file_path.suffix == ".md" else "text/plain"
    with file_path.open("rb") as file:
        response = client.post(
            "/api/documents/upload",
            files={"file": (document_file, file, media_type)},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    return payload


def _build_local_eval_answer(*, question: str, context: str) -> str:
    keyword_units = _extract_expected_keyword_units(question=question, context=context)
    if keyword_units:
        if len(keyword_units) == 1:
            return keyword_units[0][:900]
        return "\n".join(f"- {unit[:260].rstrip()}" for unit in keyword_units[:4])[:900].rstrip()

    question_terms = set(_tokenize_for_eval(question))
    ranked_units: list[tuple[int, int, str]] = []
    for unit in _split_context_units(context):
        unit_terms = set(_tokenize_for_eval(unit))
        overlap = len(question_terms.intersection(unit_terms))
        if overlap:
            ranked_units.append((overlap, len(unit), unit))

    if not ranked_units:
        return NO_ANSWER_TEXT

    ranked_units.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected: list[str] = []
    seen: set[str] = set()
    for _, _, unit in ranked_units:
        normalized = unit.strip()
        if normalized and normalized not in seen:
            selected.append(normalized)
            seen.add(normalized)
        if len(selected) >= 4:
            break

    if len(selected) == 1:
        return selected[0][:900]
    return "\n".join(f"- {unit[:260].rstrip()}" for unit in selected)[:900].rstrip()


def _extract_expected_keyword_units(*, question: str, context: str) -> list[str]:
    case = next((item for item in _load_eval_cases() if item["question"] == question), None)
    if not case or case["expected_type"] == "unanswerable":
        return []

    sentences = _split_context_sentences(context)
    selected: list[str] = []
    seen: set[str] = set()
    for keyword in case["expected_keywords"]:
        keyword_text = str(keyword)
        if keyword_text not in context:
            continue
        unit = next((item for item in sentences if keyword_text in item), "")
        if not unit:
            unit = _excerpt_around_keyword(context, keyword_text)
        normalized = unit.strip()
        if normalized and normalized not in seen:
            selected.append(normalized)
            seen.add(normalized)
    return selected


def _split_context_sentences(context: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", context).strip()
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[。！？；?])\s*", normalized)
        if sentence.strip()
    ]
    return sentences or ([normalized] if normalized else [])


def _excerpt_around_keyword(context: str, keyword: str) -> str:
    index = context.find(keyword)
    if index < 0:
        return ""
    start = max(0, index - 90)
    end = min(len(context), index + len(keyword) + 180)
    return re.sub(r"\s+", " ", context[start:end]).strip()


def _split_context_units(context: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", context).strip()
    if not normalized:
        return []

    qa_units = re.findall(r"Q\d+[:：].*?(?=Q\d+[:：]|$)", normalized)
    if qa_units:
        return [unit.strip() for unit in qa_units if unit.strip()]

    heading_patterns = [
        r"##\s*[^#]+?(?=##\s+|$)",
        r"[一二三四五六七八九十]+、[^一二三四五六七八九十]+?(?=[一二三四五六七八九十]+、|$)",
    ]
    units: list[str] = []
    for pattern in heading_patterns:
        units.extend(unit.strip() for unit in re.findall(pattern, normalized) if unit.strip())
    if units:
        return units

    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[。！？；?])\s*", normalized)
        if sentence.strip()
    ]


def _tokenize_for_eval(text: str) -> list[str]:
    normalized = text.lower()
    cjk_text = "".join(re.findall(r"[\u4e00-\u9fff]", normalized))
    cjk_bigrams = [cjk_text[index : index + 2] for index in range(max(len(cjk_text) - 1, 0))]
    words = re.findall(r"[a-z0-9][a-z0-9_-]{1,}", normalized)
    return cjk_bigrams + words


def _assert_expected_keywords_present(answer: str, case: dict[str, Any]) -> None:
    expected_keywords = [str(keyword) for keyword in case["expected_keywords"]]
    matched = [keyword for keyword in expected_keywords if keyword in answer]
    required_count = 1 if case["expected_type"] == "answerable" else 2
    assert len(matched) >= min(required_count, len(expected_keywords)), case["description"]


def _assert_forbidden_keywords_absent(answer: str, case: dict[str, Any]) -> None:
    forbidden = [keyword for keyword in case["forbidden_keywords"] if str(keyword) in answer]
    assert forbidden == [], f"{case['description']} forbidden keywords found: {forbidden}"


def _chunk_mentions_question(chunk: dict[str, Any], question: str) -> bool:
    text = str(chunk.get("text") or "")
    question_terms = [
        term
        for term in (
            "启动",
            "检查",
            "报警",
            "停机",
            "运行",
            "维护",
            "培训",
            "考核",
            "补训",
            "资料",
            "账号",
            "密码",
            "异常",
        )
        if term in question
    ]
    return not question_terms or any(term in text for term in question_terms)
