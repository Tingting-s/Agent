from app.agents.routing import DocumentRouterAgent


def test_document_router_routes_question_to_document_qa() -> None:
    decision = DocumentRouterAgent().route_document_request(
        question="设备启动前需要检查什么？",
        file_path="tests/fixtures/sample_device_manual.txt",
    )

    assert decision.task_type == "document_qa"
    assert decision.steps == ("document_qa",)


def test_document_router_routes_empty_question_to_summary() -> None:
    decision = DocumentRouterAgent().route_document_request(
        question="",
        file_path="tests/fixtures/sample_device_manual.txt",
    )

    assert decision.task_type == "document_summary"
    assert decision.steps == ("document_summary",)


def test_document_router_rejects_missing_document_reference() -> None:
    decision = DocumentRouterAgent().route_document_request(
        question="设备启动前需要检查什么？",
        file_path="",
        document_id=None,
    )

    assert decision.task_type == "invalid_request"
