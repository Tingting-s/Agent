from app.agents.routing import RouterAgent
from app.schemas.tasks import TaskRequest


def test_router_routes_document_summary_from_keywords() -> None:
    decision = RouterAgent().route(TaskRequest(user_input="Please summarize tests/data/project_notes.txt"))

    assert decision.task_type == "document_summary"
    assert decision.steps == ("document_summary",)


def test_router_routes_document_qa_from_question_input() -> None:
    decision = RouterAgent().route(
        TaskRequest(
            user_input="What is the project status?",
            additional_inputs={"question": "What is the project status?"},
        )
    )

    assert decision.task_type == "document_qa"
    assert decision.steps == ("document_qa",)


def test_router_marks_weather_as_deprecated() -> None:
    decision = RouterAgent().route(TaskRequest(user_input="Shanghai weather", task_type="weather_query"))

    assert decision.task_type == "weather_query"
    assert decision.steps == ("deprecated",)
