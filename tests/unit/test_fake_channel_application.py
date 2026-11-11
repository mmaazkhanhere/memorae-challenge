from __future__ import annotations

from datetime import timedelta
from typing import Any

from sqlalchemy import select

from reminder_agent.application import (
    ApplicationRequest,
    AuthenticatedUserContext,
    IncomingMessage,
    ListRemindersCommand,
    OutboundResponse,
    OutgoingResponse,
    RequestSource,
    VerifiedChannelIdentity,
    list_reminders_for_request,
)
from reminder_agent.features.intent.contracts import (
    CreateOneTimeReminderIntent,
    Interpretation,
    ReminderSummaryIntent,
)
from reminder_agent.features.memory.messages import MEMORY_SAVE_MESSAGES
from reminder_agent.features.memory.saving import run_memory_save_workflow
from reminder_agent.features.reminders.create.service import (
    run_reminder_creation_workflow,
)
from reminder_agent.features.reminders.listing.service import (
    summarize_reminders_from_interpretation,
)
from reminder_agent.infrastructure.database.models import (
    ChannelIdentity,
    MemoryFact,
    Notification,
    Reminder,
    ReminderOccurrence,
    User,
)


def _fake_confirmation(
    intent: CreateOneTimeReminderIntent, *, local_display: str, timezone_name: str
) -> str:
    return f"{intent.task} at {local_display} ({timezone_name})"


class FakeChannelAdapter:
    """A transport-free response port used to characterize application reuse."""

    def __init__(self) -> None:
        self.responses: list[OutgoingResponse] = []

    def existing(self) -> None:
        return None

    def recover_from_race(self) -> None:
        return None

    def direct(self, text: str) -> OutgoingResponse:
        response = OutgoingResponse(content=text, response_kind="direct")
        self.responses.append(response)
        return response

    def stage(
        self,
        *,
        user_id: Any,
        text: str | None = None,
        durable_response: Any = None,
        response_kind: str,
        operation: str | None = None,
    ) -> OutgoingResponse:
        del user_id, operation
        content = durable_response.content if durable_response is not None else text
        response = OutgoingResponse(content=content or "", response_kind=response_kind)
        self.responses.append(response)
        return response


def test_fake_channel_reuses_memory_and_reminder_workflows_without_telegram(
    session_factory: Any, fixed_now: Any
) -> None:
    with session_factory.begin() as db:
        user = User(name="Ada", timezone="UTC", onboarding_status="ACTIVE")
        db.add(user)
        db.flush()
        identity = ChannelIdentity(
            user_id=user.id,
            channel_type="PHONE",
            external_id="fake-channel:ada",
            status="LINKED",
        )
        db.add(identity)
        db.flush()

    context = AuthenticatedUserContext(
        source=RequestSource.WEB_API,
        canonical_user_id=user.id,
        verified_identity=VerifiedChannelIdentity(
            identity_id=identity.id,
            channel_type=identity.channel_type,
        ),
        external_session_id="fake-session:ada",
    )
    request = ApplicationRequest(
        context=context,
        message_text="Remember my passport is in the safe",
        idempotency_key="fake-message-1",
    )
    assert request.incoming_message == IncomingMessage(
        source=RequestSource.WEB_API,
        text=request.message_text,
        external_message_id="fake-message-1",
        external_session_id="fake-session:ada",
        idempotency_key="fake-message-1",
    )

    adapter = FakeChannelAdapter()
    with session_factory() as db:
        memory_response = run_memory_save_workflow(
            db,
            request=request,
            user_id=user.id,
            pending=None,
            text=request.message_text,
            now=fixed_now,
            session_ttl_seconds=900,
            responses=adapter,
            upsert_clarification_session=lambda **kwargs: None,
        )
        assert memory_response.content in MEMORY_SAVE_MESSAGES["confirmation"]

        intent = CreateOneTimeReminderIntent(
            task="take medicine",
            due_at=fixed_now + timedelta(hours=2),
            timezone_name="UTC",
        )
        with db.begin():
            creation = run_reminder_creation_workflow(
                db,
                intents=[intent],
                user=user,
                identity=identity,
                locked_session=None,
                timezone_name="UTC",
                now=fixed_now,
                reference=fixed_now,
                source_message_id="fake-message-2",
                stage_response=lambda **kwargs: adapter.stage(
                    user_id=user.id,
                    **kwargs,
                ),
                confirmation_formatter=_fake_confirmation,
            )

        assert creation.failure is None
        assert creation.response is not None
        assert creation.response.response_kind == "confirmation"
        assert db.scalar(select(MemoryFact).where(MemoryFact.user_id == user.id)).content == (
            "my passport is in the safe"
        )
        assert (
            db.scalar(select(Reminder).where(Reminder.user_id == user.id)).task == "take medicine"
        )
        assert db.scalar(select(Notification)) is not None

    assert [response.response_kind for response in adapter.responses] == [
        "memory_confirmation",
        "confirmation",
    ]


def test_fake_channel_reuses_read_only_reminder_workflows_without_telegram(
    session_factory: Any, fixed_now: Any
) -> None:
    with session_factory.begin() as db:
        user = User(name="Ada", timezone="UTC", onboarding_status="ACTIVE")
        db.add(user)
        db.flush()
        identity = ChannelIdentity(
            user_id=user.id,
            channel_type="PHONE",
            external_id="fake-channel:read-only",
            status="LINKED",
        )
        db.add(identity)
        db.flush()
        reminder = Reminder(
            user_id=user.id,
            task="call the bank",
            schedule_type="ONE_TIME",
            start_local="2026-11-08T13:00:00",
            timezone="UTC",
            status="ACTIVE",
        )
        db.add(reminder)
        db.flush()
        db.add(
            ReminderOccurrence(
                reminder_id=reminder.id,
                scheduled_at=fixed_now + timedelta(hours=1),
                next_due_at=fixed_now + timedelta(hours=1),
                status="PENDING",
            )
        )

    request = ApplicationRequest(
        context=AuthenticatedUserContext(
            source=RequestSource.WEB_API,
            canonical_user_id=user.id,
            verified_identity=VerifiedChannelIdentity(
                identity_id=identity.id,
                channel_type="PHONE",
            ),
            external_session_id="fake-session:read-only",
        ),
        message_text="show my reminders",
        idempotency_key="fake-read-1",
    )

    with session_factory() as db:
        listed = list_reminders_for_request(
            db,
            request=request,
            command=ListRemindersCommand(
                start_date=None,
                end_date=None,
                task_query="bank",
                timezone_name="UTC",
            ),
            now=fixed_now,
        )
        summary = summarize_reminders_from_interpretation(
            db,
            request=request,
            interpretation=Interpretation(
                intent=ReminderSummaryIntent(metric="TODAY_REMAINING", timezone_name="UTC")
            ),
            now=fixed_now,
        )

    assert listed.response_kind == "REMINDER_LIST"
    assert isinstance(listed, OutboundResponse)
    assert "call the bank" in listed.content
    assert summary is not None
    assert summary.response_kind == "REMINDER_SUMMARY"
    assert "call the bank" in summary.content
