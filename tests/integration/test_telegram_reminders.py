from datetime import date, datetime, timedelta, timezone
import hashlib

import pytest
from sqlalchemy import func, select

from reminder_agent.db.models import (
    ChannelIdentity,
    ConversationSession,
    IdempotencyRecord,
    Notification,
    Reminder,
    ReminderOccurrence,
    User,
)
from reminder_agent.intent import (
    CreateOneTimeReminderIntent,
    IntentRoute,
    IntentRouteSuggestion,
    Interpretation,
    NeedsClarification,
    PendingReminderFieldUpdate,
)
from reminder_agent.main import create_app
from reminder_agent import reminders as reminder_service
from reminder_agent.reminders import handle_reminder_text
from reminder_agent.settings import Settings
from reminder_agent.telegram import TelegramUpdate


HEADERS = {"X-Telegram-Bot-Api-Secret-Token": "test-secret"}


def utc_timestamp(value: datetime) -> datetime:
    """Normalize SQLite-naive and PostgreSQL-aware timestamps for assertions."""
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def post_update(client, update: dict):
    return client.post("/webhook/telegram", json=update, headers=HEADERS)


def onboard(api_client, update_factory, *, user_id=1001, chat_id=2001) -> None:
    assert (
        post_update(api_client, update_factory(100, user_id=user_id, chat_id=chat_id)).status_code
        == 200
    )
    assert (
        post_update(
            api_client,
            update_factory(
                101,
                user_id=user_id,
                chat_id=chat_id,
                text="Asia/Karachi",
            ),
        ).status_code
        == 200
    )


def test_one_time_relative_reminder_persists_all_rows_and_confirms_local_time(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)
    update = update_factory(
        102,
        text="Remind me in five minutes to call the bank.",
    )

    response = post_update(api_client, update)

    assert response.status_code == 200
    assert "2026-10-30 17:05" in telegram_client.messages[-1][1]
    assert "Telegram" in telegram_client.messages[-1][1]
    reminder = db.scalar(select(Reminder))
    occurrence = db.scalar(select(ReminderOccurrence))
    notification = db.scalar(select(Notification))
    assert reminder is not None
    owner = db.get(User, reminder.user_id)
    assert owner is not None
    assert owner.name == "Ada"
    assert reminder.task == "call the bank"
    assert reminder.schedule_type == "ONE_TIME"
    assert reminder.status == "ACTIVE"
    assert occurrence is not None and occurrence.reminder_id == reminder.id
    assert utc_timestamp(occurrence.scheduled_at) == datetime(
        2026, 10, 30, 12, 5, tzinfo=timezone.utc
    )
    assert notification is not None and notification.occurrence_id == occurrence.id
    assert notification.status == "PENDING"
    assert utc_timestamp(notification.due_at) == datetime(
        2026, 10, 30, 12, 5, tzinfo=timezone.utc
    )


def test_tomorrow_schedule_is_stored_as_utc_and_displayed_in_local_time(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)

    response = post_update(
        api_client,
        update_factory(103, text="Remind me tomorrow at 9 AM to renew the domain."),
    )

    assert response.status_code == 200
    assert "2026-10-31 09:00" in telegram_client.messages[-1][1]
    occurrence = db.scalar(select(ReminderOccurrence))
    assert occurrence is not None
    assert utc_timestamp(occurrence.scheduled_at) == datetime(
        2026, 10, 31, 4, 0, tzinfo=timezone.utc
    )


def test_transient_confirmation_failure_retries_and_records_sent(
    session_factory, fixed_now, update_factory
) -> None:
    from fastapi.testclient import TestClient
    from reminder_agent.telegram import TelegramAPIError

    class FailOnceTelegramClient:
        def __init__(self) -> None:
            self.attempts = 0
            self.messages: list[tuple[int, str]] = []

        def send_message(self, chat_id: int, text: str, *, reply_markup=None) -> None:
            self.attempts += 1
            if self.attempts == 3:
                raise TelegramAPIError(
                    "rate limited", status_code=429, failure_kind="http_error"
                )
            self.messages.append((chat_id, text))

    telegram_client = FailOnceTelegramClient()
    app = create_app(
        Settings(
            database_url="sqlite://",
            telegram_webhook_secret="test-secret",
            telegram_confirmation_retry_delay_seconds=0,
        ),
        telegram_client=telegram_client,
        session_factory=session_factory,
        clock=lambda: fixed_now,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        assert post_update(client, update_factory(110)).status_code == 200
        assert post_update(client, update_factory(111, text="Asia/Karachi")).status_code == 200
        response = post_update(
            client,
            update_factory(112, text="Remind me in five minutes to call the bank."),
        )

    assert response.status_code == 200
    assert telegram_client.attempts == 4
    assert len(telegram_client.messages) == 3
    assert "I'll remind you" in telegram_client.messages[-1][1]
    with session_factory() as db:
        delivery = db.scalar(
            select(IdempotencyRecord).where(
                IdempotencyRecord.idempotency_key == "112",
                IdempotencyRecord.source == "TELEGRAM",
            )
        )
        assert delivery is not None
        assert delivery.delivery_status == "SENT"


def test_replaying_update_does_not_duplicate_rows_or_confirmation(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)
    update = update_factory(104, text="Remind me in five minutes to stretch.")

    assert post_update(api_client, update).status_code == 200
    assert post_update(api_client, update).status_code == 200

    assert db.scalar(select(func.count()).select_from(Reminder)) == 1
    assert db.scalar(select(func.count()).select_from(ReminderOccurrence)) == 1
    assert db.scalar(select(func.count()).select_from(Notification)) == 1
    assert db.scalar(select(func.count()).select_from(IdempotencyRecord)) == 3
    assert len(telegram_client.messages) == 3  # onboarding start, timezone, reminder


def test_idempotency_unique_race_recovery_returns_duplicate(
    session_factory, fixed_now, update_factory
) -> None:
    """A loser of the unique-key race reloads the committed winner."""
    from reminder_agent.reminders import _recover_from_idempotency_race

    update = TelegramUpdate.model_validate(
        update_factory(109, text="Remind me in five minutes to call the bank.")
    )
    request_hash = hashlib.sha256(
        update.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8")
    ).hexdigest()

    with session_factory.begin() as db:
        user = User(name="Ada", timezone="UTC", onboarding_status="ACTIVE")
        db.add(user)
        db.flush()
        db.add(
            IdempotencyRecord(
                source="TELEGRAM",
                idempotency_key=str(update.update_id),
                user_id=user.id,
                operation="CREATE_REMINDER",
                request_hash=request_hash,
                status="SUCCEEDED",
                delivery_status="PENDING",
                response={
                    "chat_id": 2001,
                    "text": "The reminder was already processed.",
                    "reply_markup": None,
                },
                created_at=fixed_now,
                updated_at=fixed_now,
            )
        )

    with session_factory() as db:
        result = _recover_from_idempotency_race(db, update)

    assert result is not None
    assert result.is_duplicate is True
    assert result.should_send is False
    assert result.text == "The reminder was already processed."


def test_invalid_schedule_is_acknowledged_without_creating_reminder(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)

    response = post_update(
        api_client,
        update_factory(105, text="Remind me in zero minutes to call the bank."),
    )

    assert response.status_code == 200
    assert "future" in telegram_client.messages[-1][1]
    assert db.scalar(select(func.count()).select_from(Reminder)) == 0
    assert db.scalar(select(func.count()).select_from(ReminderOccurrence)) == 0
    assert db.scalar(select(func.count()).select_from(Notification)) == 0


def test_past_typed_schedule_is_rejected_without_creating_reminder(
    session_factory, fixed_now, update_factory
) -> None:
    with session_factory.begin() as db:
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

    class PastInterpreter:
        def interpret(self, text, *, timezone_name, now):
            return Interpretation(
                intent=CreateOneTimeReminderIntent(
                    task="call the bank",
                    due_at=fixed_now - timedelta(minutes=1),
                    timezone_name="UTC",
                )
            )

    with session_factory() as db:
        result = handle_reminder_text(
            db,
            TelegramUpdate.model_validate(
                update_factory(108, text="Remind me at a time that has passed")
            ),
            user_id=user.id,
            identity_id=identity.id,
            telegram_user_id=1001,
            interpreter=PastInterpreter(),
            now=fixed_now,
        )

        assert "future" in result.text
        assert db.scalar(select(func.count()).select_from(Reminder)) == 0
        assert db.scalar(select(func.count()).select_from(ReminderOccurrence)) == 0
        assert db.scalar(select(func.count()).select_from(Notification)) == 0


def test_intent_timezone_mismatch_is_rejected_without_creating_reminder(
    session_factory, fixed_now, update_factory
) -> None:
    with session_factory.begin() as db:
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

    class WrongTimezoneInterpreter:
        def interpret(self, text, *, timezone_name, now):
            return Interpretation(
                intent=CreateOneTimeReminderIntent(
                    task="call the bank",
                    due_at=fixed_now + timedelta(minutes=5),
                    timezone_name="Asia/Karachi",
                )
            )

    with session_factory() as db:
        result = handle_reminder_text(
            db,
            TelegramUpdate.model_validate(
                update_factory(113, text="Remind me in five minutes to call the bank")
            ),
            user_id=user.id,
            identity_id=identity.id,
            telegram_user_id=1001,
            interpreter=WrongTimezoneInterpreter(),
            now=fixed_now,
        )

        assert "time zone" in result.text
        assert db.scalar(select(func.count()).select_from(Reminder)) == 0
        assert db.scalar(select(func.count()).select_from(ReminderOccurrence)) == 0
        assert db.scalar(select(func.count()).select_from(Notification)) == 0


def test_cross_user_identity_cannot_be_used_for_reminder_creation(
    session_factory, fixed_now, update_factory
) -> None:
    from reminder_agent.db.models import ChannelIdentity, User

    with session_factory.begin() as db:
        first = User(name="Ada", timezone="UTC", onboarding_status="ACTIVE")
        second = User(name="Bob", timezone="UTC", onboarding_status="ACTIVE")
        db.add_all([first, second])
        db.flush()
        first_identity = ChannelIdentity(
            user_id=first.id,
            channel_type="TELEGRAM",
            external_id="1001",
            telegram_chat_id=2001,
            status="LINKED",
        )
        second_identity = ChannelIdentity(
            user_id=second.id,
            channel_type="TELEGRAM",
            external_id="2002",
            telegram_chat_id=3002,
            status="LINKED",
        )
        db.add_all([first_identity, second_identity])

    class FakeInterpreter:
        def interpret(self, text, *, timezone_name, now):
            return Interpretation(
                intent=CreateOneTimeReminderIntent(
                    task="private task",
                    due_at=fixed_now + timedelta(minutes=5),
                    timezone_name="UTC",
                )
            )

    with session_factory() as db:
        result = handle_reminder_text(
            db,
            TelegramUpdate.model_validate(
                update_factory(106, user_id=1001, chat_id=2001, text="Remind me in five minutes")
            ),
            user_id=second.id,
            identity_id=second_identity.id,
            telegram_user_id=1001,
            interpreter=FakeInterpreter(),
            now=fixed_now,
        )
        assert "couldn't verify" in result.text
        assert db.scalar(select(func.count()).select_from(Reminder)) == 0


def test_database_failure_rolls_back_reminder_graph_without_confirmation(
    session_factory, telegram_client, fixed_now, update_factory
) -> None:
    from fastapi.testclient import TestClient
    from sqlalchemy.orm import Session as SQLAlchemySession

    with session_factory.begin() as db:
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

    class FailingSession(SQLAlchemySession):
        def flush(self, *args, **kwargs) -> None:
            raise RuntimeError("database write failed")

    failing_factory = lambda: FailingSession(bind=session_factory.kw["bind"])
    app = create_app(
        Settings(database_url="sqlite://", telegram_webhook_secret="test-secret"),
        telegram_client=telegram_client,
        session_factory=failing_factory,
        clock=lambda: fixed_now,
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = post_update(
            client,
            update_factory(107, text="Remind me in five minutes to call the bank."),
        )

    assert response.status_code == 500
    assert telegram_client.messages == []
    with session_factory() as db:
        assert db.scalar(select(func.count()).select_from(Reminder)) == 0
        assert db.scalar(select(func.count()).select_from(ReminderOccurrence)) == 0
        assert db.scalar(select(func.count()).select_from(Notification)) == 0


def test_ambiguous_time_creates_pending_session_without_reminder(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)

    response = post_update(
        api_client,
        update_factory(120, text="Remind me to call the bank at 7"),
    )

    assert response.status_code == 200
    assert "7 PM today" in telegram_client.messages[-1][1]
    assert "7 AM tomorrow" in telegram_client.messages[-1][1]
    assert db.scalar(select(func.count()).select_from(Reminder)) == 0
    session = db.scalar(select(ConversationSession))
    assert session is not None
    assert session.status == "AWAITING_CLARIFICATION"
    assert session.pending_clarification["task"] == "call the bank"
    assert session.pending_clarification["meridiem"] is None


def test_clarification_follow_up_creates_one_reminder_and_ends_session(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(121, text="Remind me to call the bank at 7"),
    ).status_code == 200

    response = post_update(api_client, update_factory(122, text="7 PM"))

    assert response.status_code == 200
    assert "2026-10-30 19:00" in telegram_client.messages[-1][1]
    assert db.scalar(select(func.count()).select_from(Reminder)) == 1
    assert db.scalar(select(func.count()).select_from(ReminderOccurrence)) == 1
    assert db.scalar(select(func.count()).select_from(Notification)) == 1
    session = db.scalar(select(ConversationSession))
    assert session is not None
    assert session.status == "ENDED"
    assert session.pending_clarification is None


def test_standalone_meridiem_completes_dentist_reminder(
    api_client, db, fixed_now, update_factory
) -> None:
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(160, text="Remind me tomorrow at 8 to visit the dentist"),
    ).status_code == 200

    user = db.scalar(select(User).where(User.name == "Ada"))
    identity = db.scalar(select(ChannelIdentity).where(ChannelIdentity.external_id == "1001"))
    assert user is not None and identity is not None

    class UpdateRouter:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(operation=IntentRoute.UPDATE_PENDING)

    class MeridiemExtractor:
        def interpret(self, text, *, timezone_name, now, context=None):
            assert context is not None
            return Interpretation(field_updates=PendingReminderFieldUpdate(meridiem="PM"))

    handle_reminder_text(
        db,
        TelegramUpdate.model_validate(update_factory(161, text="PM")),
        user_id=user.id,
        identity_id=identity.id,
        telegram_user_id=1001,
        interpreter=MeridiemExtractor(),
        intent_router=UpdateRouter(),
        now=fixed_now,
    )

    reminder = db.scalar(select(Reminder))
    occurrence = db.scalar(select(ReminderOccurrence))
    session = db.scalar(select(ConversationSession))
    assert reminder is not None and occurrence is not None and session is not None
    assert reminder.task == "visit the dentist"
    assert utc_timestamp(occurrence.scheduled_at) == datetime(2026, 10, 31, 15, 0, tzinfo=timezone.utc)
    assert session.status == "ENDED"
    assert session.pending_clarification is None


@pytest.mark.parametrize(
    ("correction_text", "replacement_task"),
    [
        ("Actually, make that email Bilal tomorrow.", "email Bilal"),
        ("Please use send the proposal tomorrow instead.", "send the proposal"),
    ],
)
def test_classifier_selected_task_update_replaces_pending_task_without_phrase_matching(
    api_client, db, fixed_now, monkeypatch, update_factory, correction_text, replacement_task
) -> None:
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(162, text="Remind me tomorrow to email Nadia"),
    ).status_code == 200

    user = db.scalar(select(User).where(User.name == "Ada"))
    identity = db.scalar(select(ChannelIdentity).where(ChannelIdentity.external_id == "1001"))
    assert user is not None and identity is not None
    task_trace: list[tuple[str | None, str | None]] = []
    original_restrict = reminder_service.restrict_to_explicit_field_updates

    def capture_task_update(*args, **kwargs):
        proposed = args[1]
        accepted = original_restrict(*args, **kwargs)
        task_trace.append((proposed.task, accepted.task))
        return accepted

    monkeypatch.setattr(reminder_service, "restrict_to_explicit_field_updates", capture_task_update)

    class UpdateRouter:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(operation=IntentRoute.UPDATE_PENDING)

    class TaskAndTimeExtractor:
        def interpret(self, text, *, timezone_name, now, context=None):
            assert context is not None
            if text == correction_text:
                return Interpretation(
                    field_updates=PendingReminderFieldUpdate(
                        task=replacement_task,
                        certainty="CLEAR",
                        evidence={"task": correction_text},
                    )
                )
            return Interpretation(
                field_updates=PendingReminderFieldUpdate(
                    hour=8, minute=0, meridiem="PM"
                )
            )

    handle_reminder_text(
        db,
        TelegramUpdate.model_validate(update_factory(163, text=correction_text)),
        user_id=user.id,
        identity_id=identity.id,
        telegram_user_id=1001,
        interpreter=TaskAndTimeExtractor(),
        intent_router=UpdateRouter(),
        now=fixed_now,
    )

    # LLM extraction -> accepted update -> merged and persisted pending draft.
    assert task_trace == [(replacement_task, replacement_task)]
    pending_session = db.scalar(select(ConversationSession))
    assert pending_session is not None
    assert pending_session.status == "AWAITING_CLARIFICATION"
    assert pending_session.pending_clarification["task"] == replacement_task

    handle_reminder_text(
        db,
        TelegramUpdate.model_validate(update_factory(164, text="at 8 PM")),
        user_id=user.id,
        identity_id=identity.id,
        telegram_user_id=1001,
        interpreter=TaskAndTimeExtractor(),
        intent_router=UpdateRouter(),
        now=fixed_now,
    )

    reminder = db.scalar(select(Reminder))
    occurrence = db.scalar(select(ReminderOccurrence))
    session = db.scalar(select(ConversationSession))
    assert reminder is not None and occurrence is not None and session is not None
    assert reminder.task == replacement_task
    assert utc_timestamp(occurrence.scheduled_at) == datetime(2026, 10, 31, 15, 0, tzinfo=timezone.utc)
    assert session.status == "ENDED"
    assert session.pending_clarification is None


@pytest.mark.parametrize("typo", ["tomorow", "tmorrow", "tommorow"])
def test_classifier_selected_clear_date_typo_completes_pending_reminder(
    api_client, db, fixed_now, update_factory, typo
) -> None:
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(165, text="Remind me to call Ayesha"),
    ).status_code == 200

    user = db.scalar(select(User).where(User.name == "Ada"))
    identity = db.scalar(select(ChannelIdentity).where(ChannelIdentity.external_id == "1001"))
    session = db.scalar(select(ConversationSession))
    assert user is not None and identity is not None and session is not None
    assert session.pending_clarification["task"] == "call Ayesha"
    assert session.pending_clarification["date_local"] is None

    class UpdateRouter:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(operation=IntentRoute.UPDATE_PENDING)

    class TypoAwareExtractor:
        def interpret(self, text, *, timezone_name, now, context=None):
            assert context is not None
            assert text == f"{typo} at 8 PM"
            return Interpretation(
                field_updates=PendingReminderFieldUpdate(
                    date_local=date(2026, 10, 31),
                    hour=8,
                    minute=0,
                    meridiem="PM",
                    date_explicit=True,
                    certainty="CLEAR",
                    evidence={
                        "date_local": typo,
                        "date_explicit": typo,
                        "hour": "8 PM",
                        "minute": "8 PM",
                        "meridiem": "8 PM",
                    },
                )
            )

    handle_reminder_text(
        db,
        TelegramUpdate.model_validate(update_factory(166, text=f"{typo} at 8 PM")),
        user_id=user.id,
        identity_id=identity.id,
        telegram_user_id=1001,
        interpreter=TypoAwareExtractor(),
        intent_router=UpdateRouter(),
        now=fixed_now,
    )

    reminder = db.scalar(select(Reminder))
    occurrence = db.scalar(select(ReminderOccurrence))
    session = db.scalar(select(ConversationSession))
    assert reminder is not None and occurrence is not None and session is not None
    assert reminder.task == "call Ayesha"
    assert utc_timestamp(occurrence.scheduled_at) == datetime(2026, 10, 31, 15, 0, tzinfo=timezone.utc)
    assert session.status == "ENDED"
    assert session.pending_clarification is None


def test_provider_field_update_is_merged_with_the_locked_pending_session(
    api_client, db, fixed_now, update_factory
) -> None:
    onboard(api_client, update_factory)
    assert post_update(
        api_client,
        update_factory(141, text="Remind me tomorrow to call the bank"),
    ).status_code == 200
    pending_session = db.scalar(select(ConversationSession))
    assert pending_session is not None
    assert pending_session.pending_clarification["task"] == "call the bank"
    assert pending_session.pending_clarification["date_local"] == "2026-10-31"
    assert pending_session.pending_clarification["hour"] is None
    user = db.scalar(select(User).where(User.name == "Ada"))
    identity = db.scalar(select(ChannelIdentity).where(ChannelIdentity.external_id == "1001"))
    assert user is not None and identity is not None

    class FieldUpdateOnlyInterpreter:
        def interpret(self, text, *, timezone_name, now, context=None):
            assert context is not None
            assert context.task == "call the bank"
            return Interpretation(
                field_updates=PendingReminderFieldUpdate(
                    # These two values were never supplied by the user and
                    # must not overwrite the pending conversation state.
                    task="book a holiday",
                    date_local=date(2026, 11, 1),
                    hour=8,
                    minute=0,
                    meridiem="AM",
                )
            )

    reply = handle_reminder_text(
        db,
        TelegramUpdate.model_validate(update_factory(142, text="8 AM")),
        user_id=user.id,
        identity_id=identity.id,
        telegram_user_id=1001,
        interpreter=FieldUpdateOnlyInterpreter(),
        now=fixed_now,
    )

    assert "call the bank" in reply.text
    reminder = db.scalar(select(Reminder))
    assert reminder is not None
    assert reminder.task == "call the bank"
    occurrence = db.scalar(select(ReminderOccurrence))
    assert occurrence is not None
    assert utc_timestamp(occurrence.scheduled_at) == datetime(
        2026, 10, 31, 3, 0, tzinfo=timezone.utc
    )
    assert db.scalar(select(ConversationSession)).pending_clarification is None


def test_stale_clarification_interpretation_is_retried_with_latest_session_version(
    api_client, db, fixed_now, session_factory, update_factory
) -> None:
    onboard(api_client, update_factory)
    post_update(api_client, update_factory(145, text="Remind me Friday to call the bank"))
    user = db.scalar(select(User).where(User.name == "Ada"))
    identity = db.scalar(select(ChannelIdentity).where(ChannelIdentity.external_id == "1001"))
    assert user is not None and identity is not None

    class RacingInterpreter:
        def __init__(self) -> None:
            self.context_tasks: list[str | None] = []

        def interpret(self, text, *, timezone_name, now, context=None):
            assert context is not None
            self.context_tasks.append(context.task)
            if len(self.context_tasks) == 1:
                # Simulate another Telegram update committing while this
                # message is being interpreted outside the write transaction.
                with session_factory.begin() as concurrent_db:
                    latest = concurrent_db.scalar(select(ConversationSession))
                    assert latest is not None
                    latest_pending = dict(latest.pending_clarification)
                    latest_pending["task"] = "email Bob"
                    latest.pending_clarification = latest_pending
                    latest.version += 1
            return Interpretation(
                clarification=NeedsClarification(
                    pending=context.model_copy(update={"hour": 8, "meridiem": "am"}),
                    question="What date should I use?",
                )
            )

    interpreter = RacingInterpreter()
    handle_reminder_text(
        db,
        TelegramUpdate.model_validate(update_factory(146, text="8 AM")),
        user_id=user.id,
        identity_id=identity.id,
        telegram_user_id=1001,
        interpreter=interpreter,
        now=fixed_now,
    )

    assert interpreter.context_tasks == ["call the bank", "email Bob"]
    db.expire_all()
    session = db.scalar(select(ConversationSession))
    assert session is not None
    assert session.pending_clarification["task"] == "email Bob"
    assert session.pending_clarification["hour"] == 8
    assert session.version == 3


def test_replayed_clarification_follow_up_does_not_create_duplicate(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)
    post_update(api_client, update_factory(123, text="Remind me Friday to renew the domain"))
    follow_up = update_factory(124, text="8 PM")

    assert post_update(api_client, follow_up).status_code == 200
    assert post_update(api_client, follow_up).status_code == 200

    assert db.scalar(select(func.count()).select_from(Reminder)) == 1
    assert db.scalar(select(func.count()).select_from(ReminderOccurrence)) == 1
    assert db.scalar(select(func.count()).select_from(Notification)) == 1
    # onboarding start, timezone, clarification, and one confirmation only
    assert len(telegram_client.messages) == 4


def test_unrelated_follow_up_keeps_session_and_does_not_create_reminder(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)
    post_update(api_client, update_factory(125, text="Remind me to call the bank at 7"))

    response = post_update(api_client, update_factory(126, text="thanks"))

    assert response.status_code == 200
    assert db.scalar(select(func.count()).select_from(Reminder)) == 0
    session = db.scalar(select(ConversationSession))
    assert session is not None
    assert session.status == "AWAITING_CLARIFICATION"
    assert session.pending_clarification["task"] == "call the bank"


def test_unrelated_follow_up_does_not_reach_the_interpreter(
    api_client, db, fixed_now, update_factory
) -> None:
    onboard(api_client, update_factory)
    post_update(api_client, update_factory(143, text="Remind me to call the bank at 7"))
    user = db.scalar(select(User).where(User.name == "Ada"))
    identity = db.scalar(select(ChannelIdentity).where(ChannelIdentity.external_id == "1001"))
    assert user is not None and identity is not None

    class RecordingInterpreter:
        called = False

        def interpret(self, *args, **kwargs):
            self.called = True
            raise AssertionError("unrelated text must be routed before parsing")

    interpreter = RecordingInterpreter()
    reply = handle_reminder_text(
        db,
        TelegramUpdate.model_validate(update_factory(144, text="thanks")),
        user_id=user.id,
        identity_id=identity.id,
        telegram_user_id=1001,
        interpreter=interpreter,
        now=fixed_now,
    )

    assert interpreter.called is False
    assert reply.text == "You're welcome."
    assert db.scalar(select(func.count()).select_from(Reminder)) == 0


def test_cancel_follow_up_clears_pending_session_without_creating_reminder(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)
    post_update(api_client, update_factory(137, text="Remind me to call the bank at 7"))

    response = post_update(api_client, update_factory(138, text="Never mind"))

    assert response.status_code == 200
    assert "won't create" in telegram_client.messages[-1][1]
    assert db.scalar(select(func.count()).select_from(Reminder)) == 0
    session = db.scalar(select(ConversationSession))
    assert session is not None
    assert session.status == "ENDED"
    assert session.pending_clarification is None


def test_missing_task_never_completes_from_time_only_follow_up(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)
    post_update(api_client, update_factory(129, text="Remind me at 7"))

    response = post_update(api_client, update_factory(130, text="7 PM"))

    assert response.status_code == 200
    assert db.scalar(select(func.count()).select_from(Reminder)) == 0
    session = db.scalar(select(ConversationSession))
    assert session is not None
    assert session.status == "AWAITING_CLARIFICATION"
    assert session.pending_clarification["task"] is None


def test_multiple_people_in_one_task_create_one_reminder(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)
    post_update(
        api_client,
        update_factory(131, text="Remind me to call Alice and Bob tomorrow at 9 AM"),
    )

    assert db.scalar(select(func.count()).select_from(Reminder)) == 1
    assert db.scalar(select(func.count()).select_from(ReminderOccurrence)) == 1
    assert "call Alice and Bob" in telegram_client.messages[-1][1]


def test_multiple_tasks_without_repeated_prefix_wait_for_clarification(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)
    post_update(
        api_client,
        update_factory(132, text="Remind me to call Alice and email Bob tomorrow at 9 AM"),
    )

    assert db.scalar(select(func.count()).select_from(Reminder)) == 0
    session = db.scalar(select(ConversationSession))
    assert session is not None
    assert session.status == "AWAITING_CLARIFICATION"
    assert "more than one" in telegram_client.messages[-1][1]

    response = post_update(
        api_client,
        update_factory(139, text="Remind me to call Alice tomorrow at 9 AM"),
    )

    assert response.status_code == 200
    assert db.scalar(select(func.count()).select_from(Reminder)) == 1
    db.refresh(session)
    assert session.status == "ENDED"


def test_explicit_multiple_reminder_requests_create_two_reminders(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)

    response = post_update(
        api_client,
        update_factory(
            136,
            text=(
                "Remind me to call Alice tomorrow at 9 AM and remind me to email Bob "
                "tomorrow at 10 AM"
            ),
        ),
    )

    assert response.status_code == 200
    assert db.scalar(select(func.count()).select_from(Reminder)) == 2
    assert db.scalar(select(func.count()).select_from(ReminderOccurrence)) == 2
    assert db.scalar(select(func.count()).select_from(Notification)) == 2
    assert "call Alice" in telegram_client.messages[-1][1]
    assert "email Bob" in telegram_client.messages[-1][1]
    assert "2026-10-31 09:00" in telegram_client.messages[-1][1]
    assert "2026-10-31 10:00" in telegram_client.messages[-1][1]

    assert post_update(
        api_client,
        update_factory(
            136,
            text=(
                "Remind me to call Alice tomorrow at 9 AM and remind me to email Bob "
                "tomorrow at 10 AM"
            ),
        ),
    ).status_code == 200
    assert db.scalar(select(func.count()).select_from(Reminder)) == 2
    assert db.scalar(select(func.count()).select_from(ReminderOccurrence)) == 2
    assert db.scalar(select(func.count()).select_from(Notification)) == 2


def test_follow_up_with_minutes_and_day_completes_pending_reminder(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)
    post_update(api_client, update_factory(133, text="Remind me to call John at 6:55"))

    response = post_update(api_client, update_factory(134, text="6:55 pm today"))

    assert response.status_code == 200
    assert "2026-10-30 18:55" in telegram_client.messages[-1][1]
    assert db.scalar(select(func.count()).select_from(Reminder)) == 1


def test_bare_time_is_not_treated_as_a_new_clarification_command(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory)

    response = post_update(api_client, update_factory(135, text="7 pm"))

    assert response.status_code == 200
    assert "didn't find a reminder request" in telegram_client.messages[-1][1]
    assert db.scalar(select(func.count()).select_from(Reminder)) == 0
    assert db.scalar(select(func.count()).select_from(ConversationSession)) == 0


def test_stale_clarification_expires_without_reusing_partial_data(
    api_client, telegram_client, db, update_factory, fixed_now
) -> None:
    onboard(api_client, update_factory)
    post_update(api_client, update_factory(127, text="Remind me to call the bank at 7"))
    session = db.scalar(select(ConversationSession))
    assert session is not None
    session.updated_at = fixed_now - timedelta(hours=1)
    db.commit()

    response = post_update(api_client, update_factory(128, text="7 PM"))

    assert response.status_code == 200
    assert db.scalar(select(func.count()).select_from(Reminder)) == 0
    db.refresh(session)
    assert session.status == "ENDED"
    assert session.pending_clarification is None
