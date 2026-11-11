from __future__ import annotations

from datetime import UTC, date, datetime

from sqlalchemy import func, select

from reminder_agent.application import (
    ApplicationRequest,
    AuthenticatedRequestContext,
    RequestSource,
    VerifiedChannelIdentity,
    process_authenticated_conversation,
)
from reminder_agent.features.intent import IntentRoute, Interpretation, ListRemindersIntent
from reminder_agent.features.intent.deterministic.routing import ConversationRoute
from reminder_agent.infrastructure.database.models import (
    ChannelIdentity,
    IdempotencyRecord,
    MemoryFact,
    Reminder,
    User,
)


class RecordingInterpreter:
    def __init__(self) -> None:
        self.calls: list[IntentRoute] = []

    def interpret(self, _text: str, *, selected_route: IntentRoute, **_kwargs) -> Interpretation:
        self.calls.append(selected_route)
        return Interpretation(
            intent=ListRemindersIntent(
                start_date=date(2026, 11, 7),
                end_date=date(2026, 11, 7),
                timezone_name="UTC",
            )
        )


class ExplodingIntentRouter:
    def classify(self, *_args, **_kwargs):  # pragma: no cover - assertion protects this path
        raise AssertionError("a deterministic list route must not invoke the LLM router")


def _request_for(user, identity, *, text: str) -> ApplicationRequest:
    return ApplicationRequest(
        context=AuthenticatedRequestContext(
            source=RequestSource.TELEGRAM,
            canonical_user_id=user.id,
            verified_identity=VerifiedChannelIdentity(
                identity_id=identity.id,
                channel_type=identity.channel_type,
            ),
            external_session_id="2001",
        ),
        message_text=text,
        idempotency_key="conversation-entry-test",
    )


def _active_telegram_context(db):
    user = User(name="Ada", timezone="UTC", onboarding_status="ACTIVE")
    db.add(user)
    db.flush()
    identity = ChannelIdentity(
        user_id=user.id,
        channel_type="TELEGRAM",
        external_id="1001",
        telegram_chat_id=2001,
        status="LINKED",
    )
    db.add(identity)
    db.flush()
    return user, identity


def test_entry_processes_a_deterministic_list_without_telegram_update(
    session_factory, fixed_now
) -> None:
    with session_factory.begin() as db:
        user, identity = _active_telegram_context(db)
    interpreter = RecordingInterpreter()

    with session_factory() as db:
        result = process_authenticated_conversation(
            db,
            request=_request_for(user, identity, text="Show my reminders today"),
            interpreter=interpreter,
            intent_router=ExplodingIntentRouter(),
            has_pending_clarification=False,
            now=fixed_now,
        )

    assert result.route is ConversationRoute.LIST_REQUEST
    assert result.response is not None
    assert result.response.response_kind == "REMINDER_LIST"
    assert result.response.content == "No reminders for November 7, 2026. Want to add something?"
    assert interpreter.calls == [IntentRoute.LIST_REQUEST]


def test_entry_revalidates_context_before_interpretation_or_read(
    session_factory, fixed_now
) -> None:
    with session_factory.begin() as db:
        user, identity = _active_telegram_context(db)
    interpreter = RecordingInterpreter()
    forged = ApplicationRequest(
        context=AuthenticatedRequestContext(
            source=RequestSource.TELEGRAM,
            canonical_user_id=user.id,
            verified_identity=VerifiedChannelIdentity(
                identity_id=identity.id,
                channel_type="PHONE",
            ),
            external_session_id="2001",
        ),
        message_text="Show my reminders today",
        idempotency_key="forged-context",
    )

    with session_factory() as db:
        result = process_authenticated_conversation(
            db,
            request=forged,
            interpreter=interpreter,
            has_pending_clarification=False,
            now=fixed_now,
        )
        assert db.scalar(select(func.count()).select_from(Reminder)) == 0
        assert db.scalar(select(func.count()).select_from(MemoryFact)) == 0
        assert db.scalar(select(func.count()).select_from(IdempotencyRecord)) == 0

    assert result.response is not None
    assert result.response.response_kind == "AUTHORIZATION_FAILURE"
    assert interpreter.calls == []


def test_entry_leaves_unsupported_model_fallback_to_compatibility_adapter(
    session_factory, fixed_now
) -> None:
    with session_factory.begin() as db:
        user, identity = _active_telegram_context(db)
    interpreter = RecordingInterpreter()

    with session_factory() as db:
        result = process_authenticated_conversation(
            db,
            request=_request_for(user, identity, text="Write a poem about rain"),
            interpreter=interpreter,
            intent_router=ExplodingIntentRouter(),
            has_pending_clarification=False,
            now=fixed_now,
        )
        assert db.scalar(select(func.count()).select_from(Reminder)) == 0
        assert db.scalar(select(func.count()).select_from(MemoryFact)) == 0
        assert db.scalar(select(func.count()).select_from(IdempotencyRecord)) == 0

    assert result.route is ConversationRoute.MODEL_FALLBACK
    assert result.response is None
    assert interpreter.calls == []


def test_entry_defers_list_requests_with_pending_clarification(session_factory, fixed_now) -> None:
    with session_factory.begin() as db:
        user, identity = _active_telegram_context(db)
    interpreter = RecordingInterpreter()

    with session_factory() as db:
        result = process_authenticated_conversation(
            db,
            request=_request_for(user, identity, text="Show my reminders today"),
            interpreter=interpreter,
            has_pending_clarification=True,
            now=fixed_now,
        )

    assert result.route is ConversationRoute.LIST_REQUEST
    assert result.response is None
    assert interpreter.calls == []


def test_entry_uses_the_latest_list_selection_for_ordinal_follow_up(
    session_factory, fixed_now
) -> None:
    with session_factory.begin() as db:
        user, identity = _active_telegram_context(db)
    interpreter = RecordingInterpreter()

    with session_factory() as db:
        result = process_authenticated_conversation(
            db,
            request=_request_for(user, identity, text="which on is first"),
            interpreter=interpreter,
            has_pending_clarification=False,
            now=fixed_now,
            selection_context=(
                {
                    "reminder_id": "00000000-0000-0000-0000-000000000001",
                    "task": "pay rent",
                    "due_at": datetime(2026, 11, 7, 10, tzinfo=UTC).isoformat(),
                },
                {
                    "reminder_id": "00000000-0000-0000-0000-000000000002",
                    "task": "call Alice",
                    "due_at": datetime(2026, 11, 7, 12, tzinfo=UTC).isoformat(),
                },
            ),
        )

    assert result.response is not None
    assert result.response.response_kind == "REMINDER_SELECTION"
    assert "pay rent" in result.response.content
    assert interpreter.calls == []


def test_entry_uses_list_selection_before_an_unfinished_clarification(
    session_factory, fixed_now
) -> None:
    with session_factory.begin() as db:
        user, identity = _active_telegram_context(db)
    interpreter = RecordingInterpreter()

    with session_factory() as db:
        result = process_authenticated_conversation(
            db,
            request=_request_for(user, identity, text="which one is first?"),
            interpreter=interpreter,
            has_pending_clarification=True,
            now=fixed_now,
            selection_context=(
                {
                    "reminder_id": "00000000-0000-0000-0000-000000000001",
                    "task": "review my slides",
                    "due_at": datetime(2026, 11, 12, 3, 30, tzinfo=UTC).isoformat(),
                },
            ),
        )

    assert result.response is not None
    assert result.response.response_kind == "REMINDER_SELECTION"
    assert "review my slides" in result.response.content
    assert interpreter.calls == []
