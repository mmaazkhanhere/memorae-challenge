from datetime import UTC, datetime
from typing import cast

from sqlalchemy.orm import Session

from reminder_agent.channels.telegram.provider import TelegramUpdate
from reminder_agent.features.intent import (
    Interpretation,
    ListRemindersIntent,
    ReminderSummaryIntent,
)
from reminder_agent.features.reminders.summary.telegram_adapter import (
    handle_reminder_summary_request,
)
from reminder_agent.infrastructure.db.models import User


def _update() -> TelegramUpdate:
    return TelegramUpdate.model_validate(
        {
            "update_id": 1201,
            "message": {
                "message_id": 1201,
                "from": {"id": 1001, "first_name": "Ada"},
                "chat": {"id": 2001, "type": "private"},
                "text": "How many reminders are pending?",
            },
        }
    )


def _user(*, timezone_name: str = "UTC") -> User:
    return User(name="Ada", timezone=timezone_name, onboarding_status="ACTIVE")


def test_summary_rejects_mixed_commands_without_querying() -> None:
    called = False

    def summary_operation(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("mixed summary commands must not query")

    interpretation = Interpretation(
        intents=(
            ReminderSummaryIntent(metric="PENDING_COUNT", timezone_name="UTC"),
            ListRemindersIntent(timezone_name="UTC"),
        )
    )
    reply = handle_reminder_summary_request(
        cast(Session, None),
        _update(),
        user=_user(),
        interpretation=interpretation,
        now=datetime(2026, 11, 8, 12, tzinfo=UTC),
        summary_operation=summary_operation,
    )

    assert reply is not None
    assert reply.text == "Please ask one reminder progress question at a time."
    assert called is False


def test_summary_rejects_a_timezone_mismatch_without_querying() -> None:
    called = False

    def summary_operation(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("timezone mismatch must not query")

    interpretation = Interpretation(
        intent=ReminderSummaryIntent(metric="PENDING_COUNT", timezone_name="Asia/Karachi")
    )
    reply = handle_reminder_summary_request(
        cast(Session, None),
        _update(),
        user=_user(),
        interpretation=interpretation,
        now=datetime(2026, 11, 8, 12, tzinfo=UTC),
        summary_operation=summary_operation,
    )

    assert reply is not None
    assert reply.text == "I couldn't verify your reminder time zone. Please try again."
    assert called is False
