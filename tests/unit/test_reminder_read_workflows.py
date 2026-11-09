from datetime import UTC, datetime
from typing import cast
from uuid import uuid4

from sqlalchemy.orm import Session

from reminder_agent.application.contracts import ApplicationRequest, ApplicationResult
from reminder_agent.application.reminder_read_workflows import (
    list_reminders_from_interpretation,
    summarize_reminders_from_interpretation,
)
from reminder_agent.features.intent.contracts import (
    Interpretation,
    ListRemindersIntent,
    ReminderSummaryIntent,
)
from reminder_agent.infrastructure.db.models import User

NOW = datetime(2026, 11, 8, 12, tzinfo=UTC)


def _request() -> ApplicationRequest:
    return cast(ApplicationRequest, object())


def _user() -> User:
    user = User(name="Ada", timezone="UTC", onboarding_status="ACTIVE")
    user.id = uuid4()
    return user


def test_list_workflow_delegates_channel_neutral_command(monkeypatch) -> None:
    user = _user()
    monkeypatch.setattr(
        "reminder_agent.features.reminders.listing.service.authorized_user_for_request",
        lambda *_args, **_kwargs: user,
    )
    captured = {}

    def operation(_db, *, request, command, now):
        captured.update(request=request, command=command, now=now)
        return ApplicationResult(content="listed", response_kind="REMINDER_LIST")

    result = list_reminders_from_interpretation(
        cast(Session, None),
        request=_request(),
        interpretation=Interpretation(
            intent=ListRemindersIntent(timezone_name="UTC", task_query="bank")
        ),
        now=NOW,
        list_operation=operation,
    )

    assert result == ApplicationResult(content="listed", response_kind="REMINDER_LIST")
    assert captured["command"].task_query == "bank"
    assert captured["now"] == NOW


def test_summary_workflow_rejects_mixed_intents_without_querying() -> None:
    called = False

    def operation(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("mixed read-only intents must not query")

    result = summarize_reminders_from_interpretation(
        cast(Session, None),
        request=None,
        authenticated_user=_user(),
        interpretation=Interpretation(
            intents=(
                ReminderSummaryIntent(metric="PENDING_COUNT", timezone_name="UTC"),
                ListRemindersIntent(timezone_name="UTC"),
            )
        ),
        now=NOW,
        summary_operation=operation,
    )

    assert result is not None
    assert result.content == "Please ask one reminder progress question at a time."
    assert called is False
