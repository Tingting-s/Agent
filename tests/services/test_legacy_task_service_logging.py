import logging

from app.services.task_service import TaskService
from app.schemas.tasks import TaskRequest


def test_task_service_emits_basic_logs(caplog) -> None:
    service = TaskService()

    with caplog.at_level(logging.INFO):
        service.handle_request(
            TaskRequest(
                user_input="tests/data/project_notes.txt",
                task_type="document_summary",
            )
        )

    messages = [record.getMessage() for record in caplog.records]
    assert any("Handling legacy task request as document workflow." in message for message in messages)
