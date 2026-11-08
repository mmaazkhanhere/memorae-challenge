import os
from datetime import UTC, datetime
from threading import Barrier, Thread

import pytest
from sqlalchemy import event, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from reminder_agent.channels.telegram.callbacks import callback_data
from reminder_agent.channels.telegram.provider import TelegramUpdate
from reminder_agent.features.intent import IntentRoute, IntentRouteSuggestion
from reminder_agent.infrastructure.db.models import (
    ChannelIdentity,
    ConversationSession,
    IdempotencyRecord,
    Notification,
    Reminder,
    ReminderOccurrence,
    User,
)

HEADERS = {"X-Telegram-Bot-Api-Secret-Token": "test-secret"}
requires_postgres = pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL", "").startswith("postgresql"),
    reason="requires the dedicated PostgreSQL integration database",
)


class StubIntentRouter:
    def __init__(self, operation: IntentRoute) -> None:
        self.operation = operation
        self.calls: list[str] = []

    def classify(self, text, *, pending, recent_messages):
        self.calls.append(text)
        return IntentRouteSuggestion(operation=self.operation)


def _post(client, payload):
    return client.post("/webhook/telegram", json=payload, headers=HEADERS)


def _onboard(api_client, db, update_factory, start_id=1000):
    _post(api_client, update_factory(start_id))
    _post(api_client, update_factory(start_id + 1, text="UTC"))
    identity = db.scalar(select(ChannelIdentity).where(ChannelIdentity.external_id == "1001"))
    assert identity is not None
    return identity


def _seed_delivered(db, identity, *, task="call the bank", recurring=False, provider="777"):
    reminder = Reminder(
        user_id=identity.user_id,
        task=task,
        schedule_type="RECURRING" if recurring else "ONE_TIME",
        start_local="2026-11-03T12:00:00",
        timezone="UTC",
        recurrence_rule="FREQ=WEEKLY;BYDAY=TU;BYHOUR=12;BYMINUTE=00" if recurring else None,
    )
    db.add(reminder)
    db.flush()
    occurrence = ReminderOccurrence(
        reminder_id=reminder.id,
        scheduled_at=datetime.now(UTC),
        next_due_at=datetime.now(UTC),
        status="NOTIFIED",
    )
    db.add(occurrence)
    db.flush()
    db.add(
        Notification(
            occurrence_id=occurrence.id,
            destination_identity_id=identity.id,
            sequence=1,
            due_at=occurrence.scheduled_at,
            status="SENT",
            provider_message_id=provider,
            sent_at=occurrence.scheduled_at,
        )
    )
    db.commit()
    return reminder, occurrence


def test_reply_done_thanks_completes_and_confirms_the_delivered_occurrence(
    api_client, telegram_client, db, update_factory
):
    _post(api_client, update_factory(100))
    _post(api_client, update_factory(101, text="UTC"))
    identity = db.scalar(select(ChannelIdentity).where(ChannelIdentity.external_id == "1001"))
    assert identity is not None
    reminder = Reminder(
        user_id=identity.user_id,
        task="call the bank",
        schedule_type="ONE_TIME",
        start_local="2026-11-03T12:00:00",
        timezone="UTC",
    )
    db.add(reminder)
    db.flush()
    occurrence = ReminderOccurrence(
        reminder_id=reminder.id,
        scheduled_at=datetime.now(UTC),
        next_due_at=datetime.now(UTC),
        status="NOTIFIED",
    )
    db.add(occurrence)
    db.flush()
    db.add(
        Notification(
            occurrence_id=occurrence.id,
            destination_identity_id=identity.id,
            sequence=1,
            due_at=occurrence.scheduled_at,
            status="SENT",
            provider_message_id="777",
            sent_at=occurrence.scheduled_at,
        )
    )
    db.commit()

    payload = update_factory(102, text="Done, thanks")
    payload["message"]["reply_to_message"] = {
        "message_id": 777,
        "from": {"id": 999},
        "chat": {"id": 2001, "type": "private"},
        "text": "Reminder: call the bank",
    }
    response = _post(api_client, payload)
    assert response.status_code == 200
    assert "Marked as done" in telegram_client.messages[-1][1]
    db.refresh(occurrence)
    db.refresh(reminder)
    assert occurrence.status == "COMPLETED"
    assert reminder.status == "COMPLETED"


def test_inline_done_callback_completes_only_owned_occurrence(api_client, db, update_factory):
    _post(api_client, update_factory(200))
    _post(api_client, update_factory(201, text="UTC"))
    identity = db.scalar(select(ChannelIdentity).where(ChannelIdentity.external_id == "1001"))
    assert identity is not None
    reminder = Reminder(
        user_id=identity.user_id,
        task="send contract",
        schedule_type="ONE_TIME",
        start_local="2026-11-03T12:00:00",
        timezone="UTC",
    )
    db.add(reminder)
    db.flush()
    occurrence = ReminderOccurrence(
        reminder_id=reminder.id,
        scheduled_at=datetime.now(UTC),
        next_due_at=datetime.now(UTC),
        status="NOTIFIED",
    )
    db.add(occurrence)
    db.flush()
    db.add(
        Notification(
            occurrence_id=occurrence.id,
            destination_identity_id=identity.id,
            sequence=1,
            due_at=occurrence.scheduled_at,
            status="SENT",
            provider_message_id="778",
            sent_at=occurrence.scheduled_at,
        )
    )
    db.commit()
    payload = {
        "update_id": 202,
        "callback_query": {
            "id": "callback-1",
            "from": {"id": 1001},
            "data": callback_data(occurrence.id, "test-secret"),
            "message": {
                "message_id": 778,
                "from": {"id": 999},
                "chat": {"id": 2001, "type": "private"},
                "text": "Reminder: send contract",
            },
        },
    }
    response = _post(api_client, payload)
    assert response.status_code == 200
    db.refresh(occurrence)
    assert occurrence.status == "COMPLETED"


def test_inline_callback_is_acknowledged(telegram_client, api_client, db, update_factory):
    identity = _onboard(api_client, db, update_factory, 300)
    _, occurrence = _seed_delivered(db, identity, provider="778")
    payload = {
        "update_id": 302,
        "callback_query": {
            "id": "callback-ack",
            "from": {"id": 1001},
            "data": callback_data(occurrence.id, "test-secret"),
            "message": {
                "message_id": 778,
                "from": {"id": 999},
                "chat": {"id": 2001, "type": "private"},
                "text": "Reminder: call the bank",
            },
        },
    }
    assert _post(api_client, payload).status_code == 200
    first_reply_count = len(telegram_client.messages)
    assert _post(api_client, payload).status_code == 200
    assert telegram_client.callback_acknowledgements == ["callback-ack", "callback-ack"]
    assert len(telegram_client.messages) == first_reply_count


def test_forged_callback_does_not_mutate(telegram_client, api_client, db, update_factory):
    identity = _onboard(api_client, db, update_factory, 400)
    _, occurrence = _seed_delivered(db, identity, provider="779")
    valid_callback = callback_data(occurrence.id, "test-secret")
    forged = valid_callback[:-1] + ("0" if valid_callback[-1] != "0" else "1")
    payload = {
        "update_id": 402,
        "callback_query": {
            "id": "callback-forged",
            "from": {"id": 1001},
            "data": forged,
            "message": {
                "message_id": 779,
                "from": {"id": 999},
                "chat": {"id": 2001, "type": "private"},
                "text": "Reminder",
            },
        },
    }
    assert _post(api_client, payload).status_code == 200
    db.refresh(occurrence)
    assert occurrence.status == "NOTIFIED"
    assert telegram_client.callback_acknowledgements == []


def test_callback_from_another_telegram_user_does_not_mutate(
    telegram_client, api_client, db, update_factory
):
    identity = _onboard(api_client, db, update_factory, 450)
    _, occurrence = _seed_delivered(db, identity, provider="7791")
    payload = {
        "update_id": 452,
        "callback_query": {
            "id": "callback-other-user",
            "from": {"id": 9999},
            "data": callback_data(occurrence.id, "test-secret"),
            "message": {
                "message_id": 7791,
                "from": {"id": 999},
                "chat": {"id": 2001, "type": "private"},
                "text": "Reminder",
            },
        },
    }
    assert _post(api_client, payload).status_code == 200
    db.refresh(occurrence)
    assert occurrence.status == "NOTIFIED"
    assert telegram_client.callback_acknowledgements == []


def test_bare_done_requires_one_unambiguous_delivered_occurrence(
    telegram_client, api_client, db, update_factory
):
    identity = _onboard(api_client, db, update_factory, 500)
    _, first = _seed_delivered(db, identity, task="first", provider="780")
    _, second = _seed_delivered(db, identity, task="second", provider="781")
    response = _post(api_client, update_factory(502, text="Done"))
    assert response.status_code == 200
    assert "Which delivered reminder" in telegram_client.messages[-1][1]
    db.refresh(first)
    db.refresh(second)
    assert first.status == "NOTIFIED"
    assert second.status == "NOTIFIED"


def test_ambiguous_done_follow_up_matches_delivered_task_and_completes_only_that_occurrence(
    telegram_client, api_client, db, update_factory
):
    identity = _onboard(api_client, db, update_factory, 505)
    _, first = _seed_delivered(db, identity, task="call Jack", provider="7831")
    _, second = _seed_delivered(db, identity, task="call James", provider="7832")

    assert _post(api_client, update_factory(507, text="Done thanks")).status_code == 200
    assert "Which delivered reminder" in telegram_client.messages[-1][1]
    pending = db.scalar(select(ConversationSession))
    assert pending is not None
    assert pending.pending_clarification["operation"] == "COMPLETE_REMINDER"

    assert _post(api_client, update_factory(508, text="Calling James")).status_code == 200
    assert telegram_client.messages[-1][1].startswith("Marked as done")
    db.refresh(first)
    db.refresh(second)
    assert first.status == "NOTIFIED"
    assert second.status == "COMPLETED"
    db.refresh(pending)
    assert pending.pending_clarification is None


def test_ambiguous_completion_follow_up_stays_unresolved_without_mutation(
    telegram_client, api_client, db, update_factory
):
    identity = _onboard(api_client, db, update_factory, 515)
    _, first = _seed_delivered(db, identity, task="call Jack", provider="7833")
    _, second = _seed_delivered(db, identity, task="call James", provider="7834")

    assert _post(api_client, update_factory(517, text="Done")).status_code == 200
    assert _post(api_client, update_factory(518, text="I finished it")).status_code == 200
    assert "Which delivered reminder" in telegram_client.messages[-1][1]
    db.refresh(first)
    db.refresh(second)
    assert first.status == "NOTIFIED"
    assert second.status == "NOTIFIED"


def test_llm_completion_route_uses_runtime_router_and_existing_completion_service(
    api_client, telegram_client, db, update_factory
):
    router = StubIntentRouter(IntentRoute.COMPLETE_REQUEST)
    api_client.app.state.intent_router = router
    identity = _onboard(api_client, db, update_factory, 510)
    _, occurrence = _seed_delivered(db, identity, provider="784")
    payload = update_factory(512, text="I've taken care of it")
    payload["message"]["reply_to_message"] = {
        "message_id": 784,
        "from": {"id": 999},
        "chat": {"id": 2001, "type": "private"},
        "text": "Reminder: call the bank",
    }

    assert _post(api_client, payload).status_code == 200
    assert router.calls == ["I've taken care of it"]
    assert telegram_client.messages[-1][1].startswith("Marked as done")
    db.refresh(occurrence)
    assert occurrence.status == "COMPLETED"


def test_llm_completion_route_requires_clarification_for_multiple_delivered_reminders(
    api_client, telegram_client, db, update_factory
):
    router = StubIntentRouter(IntentRoute.COMPLETE_REQUEST)
    api_client.app.state.intent_router = router
    identity = _onboard(api_client, db, update_factory, 520)
    _, first = _seed_delivered(db, identity, task="first", provider="785")
    _, second = _seed_delivered(db, identity, task="second", provider="786")

    assert _post(api_client, update_factory(522, text="I finished it")).status_code == 200
    assert router.calls == ["I finished it"]
    assert "Which delivered reminder" in telegram_client.messages[-1][1]
    db.refresh(first)
    db.refresh(second)
    assert first.status == "NOTIFIED"
    assert second.status == "NOTIFIED"


def test_uncertain_llm_route_does_not_mutate_delivered_reminder(
    api_client, telegram_client, db, update_factory
):
    router = StubIntentRouter(IntentRoute.UNCERTAIN)
    api_client.app.state.intent_router = router
    identity = _onboard(api_client, db, update_factory, 530)
    _, occurrence = _seed_delivered(db, identity, provider="787")

    assert _post(api_client, update_factory(532, text="Please handle that")).status_code == 200
    assert router.calls == ["Please handle that"]
    db.refresh(occurrence)
    assert occurrence.status == "NOTIFIED"
    assert "not sure" in telegram_client.messages[-1][1].lower()


def test_deterministic_done_still_bypasses_llm_router(
    api_client, telegram_client, db, update_factory
):
    class FailingRouter:
        def classify(self, *args, **kwargs):  # pragma: no cover - deterministic route must win
            raise AssertionError("deterministic Done must not call the LLM router")

    api_client.app.state.intent_router = FailingRouter()
    identity = _onboard(api_client, db, update_factory, 540)
    _, occurrence = _seed_delivered(db, identity, provider="788")
    payload = update_factory(542, text="Done")
    payload["message"]["reply_to_message"] = {
        "message_id": 788,
        "from": {"id": 999},
        "chat": {"id": 2001, "type": "private"},
        "text": "Reminder: call the bank",
    }

    assert _post(api_client, payload).status_code == 200
    assert telegram_client.messages[-1][1].startswith("Marked as done")
    db.refresh(occurrence)
    assert occurrence.status == "COMPLETED"


def test_reply_to_unknown_notification_does_not_fallback_to_latest(
    telegram_client, api_client, db, update_factory
):
    identity = _onboard(api_client, db, update_factory, 600)
    _, occurrence = _seed_delivered(db, identity, provider="782")
    payload = update_factory(602, text="Done")
    payload["message"]["reply_to_message"] = {
        "message_id": 9999,
        "from": {"id": 999},
        "chat": {"id": 2001, "type": "private"},
        "text": "Another message",
    }
    assert _post(api_client, payload).status_code == 200
    assert "couldn't find a delivered reminder" in telegram_client.messages[-1][1]
    db.refresh(occurrence)
    assert occurrence.status == "NOTIFIED"


def test_replayed_done_update_does_not_send_or_complete_twice(
    telegram_client, api_client, db, update_factory
):
    identity = _onboard(api_client, db, update_factory, 700)
    _, occurrence = _seed_delivered(db, identity, provider="783")
    payload = update_factory(702, text="Done")
    payload["message"]["reply_to_message"] = {
        "message_id": 783,
        "from": {"id": 999},
        "chat": {"id": 2001, "type": "private"},
        "text": "Reminder",
    }
    assert _post(api_client, payload).status_code == 200
    sent_count = len(telegram_client.messages)
    assert _post(api_client, payload).status_code == 200
    assert len(telegram_client.messages) == sent_count
    db.refresh(occurrence)
    assert occurrence.status == "COMPLETED"
    record = db.scalar(
        select(IdempotencyRecord).where(
            IdempotencyRecord.source == "TELEGRAM",
            IdempotencyRecord.idempotency_key == "702",
        )
    )
    assert record is not None
    assert record.operation == "COMPLETE_REMINDER"

    repeated = update_factory(703, text="Done")
    repeated["message"]["reply_to_message"] = payload["message"]["reply_to_message"]
    assert _post(api_client, repeated).status_code == 200
    assert "already marked as done" in telegram_client.messages[-1][1]


@requires_postgres
def test_postgres_pending_completion_commits_mutation_session_and_response(
    telegram_client, api_client, db, update_factory
):
    identity = _onboard(api_client, db, update_factory, 800)
    _, first = _seed_delivered(db, identity, task="call Jack", provider="880")
    _, second = _seed_delivered(db, identity, task="call James", provider="881")

    assert _post(api_client, update_factory(802, text="Done")).status_code == 200
    follow_up = update_factory(803, text="Calling James")
    assert _post(api_client, follow_up).status_code == 200
    assert telegram_client.messages[-1][1] == "Marked as done: call James."

    db.expire_all()
    db.refresh(first)
    db.refresh(second)
    session = db.scalar(select(ConversationSession))
    record = db.scalar(
        select(IdempotencyRecord).where(
            IdempotencyRecord.source == "TELEGRAM",
            IdempotencyRecord.idempotency_key == "803",
        )
    )
    assert first.status == "NOTIFIED"
    assert second.status == "COMPLETED"
    assert session is not None and session.pending_clarification is None
    assert record is not None and record.operation == "COMPLETE_REMINDER"


@requires_postgres
def test_postgres_pending_completion_duplicate_update_replays_without_second_response(
    telegram_client, api_client, db, update_factory
):
    identity = _onboard(api_client, db, update_factory, 810)
    _, first = _seed_delivered(db, identity, task="call Jack", provider="890")
    _, second = _seed_delivered(db, identity, task="call James", provider="891")

    assert _post(api_client, update_factory(812, text="Done")).status_code == 200
    follow_up = update_factory(813, text="Calling James")
    assert _post(api_client, follow_up).status_code == 200
    sent_count = len(telegram_client.messages)
    assert _post(api_client, follow_up).status_code == 200

    db.expire_all()
    db.refresh(first)
    db.refresh(second)
    assert len(telegram_client.messages) == sent_count
    assert first.status == "NOTIFIED"
    assert second.status == "COMPLETED"
    assert (
        db.scalar(
            select(IdempotencyRecord).where(
                IdempotencyRecord.source == "TELEGRAM",
                IdempotencyRecord.idempotency_key == "813",
            )
        )
        is not None
    )


@requires_postgres
def test_postgres_pending_completion_integrity_error_replays_committed_winner(
    telegram_client, api_client, db, update_factory, fixed_now, monkeypatch
):
    identity = _onboard(api_client, db, update_factory, 815)
    _, first = _seed_delivered(db, identity, task="call Jack", provider="895")
    _, second = _seed_delivered(db, identity, task="call James", provider="896")

    assert _post(api_client, update_factory(817, text="Done")).status_code == 200
    follow_up_payload = update_factory(818, text="Calling James")
    follow_up = TelegramUpdate.model_validate(follow_up_payload)
    user = db.get(User, identity.user_id)
    assert user is not None
    pending_session = db.scalar(select(ConversationSession))
    assert pending_session is not None
    pending_before = pending_session.pending_clarification
    messages_before = list(telegram_client.messages)

    # Seed the already-committed winner. Calling the pending handler directly
    # bypasses the outer duplicate preflight so its own response insert hits
    # PostgreSQL's unique idempotency constraint inside the real transaction.
    from reminder_agent.application import conversation_workflow as reminders

    db.rollback()
    with db.begin():
        winner = reminders._store_response(
            db,
            update=follow_up,
            user_id=user.id,
            text="Marked as done: call James.",
            response_kind="completion",
            operation="COMPLETE_REMINDER",
        )

    recovery_calls: list[int] = []
    recover = reminders._recover_from_idempotency_race

    def observe_recovery(session, update):
        recovery_calls.append(update.update_id)
        return recover(session, update)

    monkeypatch.setattr(reminders, "_recover_from_idempotency_race", observe_recovery)
    reply = reminders._handle_completion_follow_up(
        db,
        follow_up,
        user=user,
        identity=identity,
        now=fixed_now,
        session_ttl_seconds=900,
    )

    assert reply is not None and reply.is_duplicate is True
    assert reply.text == winner.text
    assert recovery_calls == [818]
    assert telegram_client.messages == messages_before

    db.expire_all()
    db.refresh(first)
    db.refresh(second)
    session = db.scalar(select(ConversationSession))
    records = db.scalars(
        select(IdempotencyRecord).where(
            IdempotencyRecord.source == "TELEGRAM",
            IdempotencyRecord.idempotency_key == "818",
        )
    ).all()
    assert first.status == "NOTIFIED"
    assert second.status == "NOTIFIED"
    assert session is not None
    assert session.user_id == identity.user_id
    assert session.pending_clarification == pending_before
    assert len(records) == 1
    assert records[0].user_id == identity.user_id


@requires_postgres
def test_postgres_concurrent_pending_completion_attempts_mutate_once(
    db, api_client, fixed_now, session_factory, update_factory
):
    identity = _onboard(api_client, db, update_factory, 820)
    _, first = _seed_delivered(db, identity, task="call Jack", provider="900")
    _, second = _seed_delivered(db, identity, task="call James", provider="901")
    assert _post(api_client, update_factory(822, text="Done")).status_code == 200

    barrier = Barrier(2)
    outcomes: list[str] = []
    errors: list[BaseException] = []

    def complete_from_separate_session(update_id: int) -> None:
        from reminder_agent.channels.telegram.provider import TelegramUpdate
        from reminder_agent.application.conversation_workflow import handle_reminder_text
        from reminder_agent.features.intent import RuleBasedIntentInterpreter

        session = session_factory()
        try:
            barrier.wait(timeout=5)
            reply = handle_reminder_text(
                session,
                TelegramUpdate.model_validate(update_factory(update_id, text="Calling James")),
                user_id=identity.user_id,
                identity_id=identity.id,
                telegram_user_id=1001,
                interpreter=RuleBasedIntentInterpreter(),
                now=fixed_now,
            )
            outcomes.append(reply.text)
        except BaseException as exc:  # pragma: no cover - asserted by the main thread
            errors.append(exc)
        finally:
            session.close()

    first_thread = Thread(target=complete_from_separate_session, args=(823,))
    second_thread = Thread(target=complete_from_separate_session, args=(824,))
    first_thread.start()
    second_thread.start()
    first_thread.join(timeout=10)
    second_thread.join(timeout=10)

    assert not first_thread.is_alive() and not second_thread.is_alive()
    assert not errors
    db.expire_all()
    db.refresh(first)
    db.refresh(second)
    session = db.scalar(select(ConversationSession))
    assert sorted(outcomes) == [
        "Marked as done: call James.",
        "Please tell me the reminder task and a future time.",
    ]
    assert first.status == "NOTIFIED"
    assert second.status == "COMPLETED"
    assert session is not None and session.pending_clarification is None


@requires_postgres
def test_postgres_pending_completion_commit_failure_rolls_back_and_retry_replays(
    telegram_client, api_client, db, update_factory
):
    identity = _onboard(api_client, db, update_factory, 830)
    _, first = _seed_delivered(db, identity, task="call Jack", provider="910")
    _, second = _seed_delivered(db, identity, task="call James", provider="911")
    assert _post(api_client, update_factory(832, text="Done")).status_code == 200
    follow_up = update_factory(833, text="Calling James")
    messages_before_failure = len(telegram_client.messages)

    def fail_completion_commit(session: Session) -> None:
        if any(
            isinstance(item, ReminderOccurrence) and item.status == "COMPLETED"
            for item in session.dirty
        ):
            raise OperationalError("COMMIT", {}, RuntimeError("forced completion commit failure"))

    event.listen(Session, "before_commit", fail_completion_commit)
    try:
        failed = _post(api_client, follow_up)
    finally:
        event.remove(Session, "before_commit", fail_completion_commit)

    assert failed.status_code == 500
    assert len(telegram_client.messages) == messages_before_failure
    db.expire_all()
    db.refresh(first)
    db.refresh(second)
    session = db.scalar(select(ConversationSession))
    record = db.scalar(
        select(IdempotencyRecord).where(
            IdempotencyRecord.source == "TELEGRAM",
            IdempotencyRecord.idempotency_key == "833",
        )
    )
    assert first.status == "NOTIFIED"
    assert second.status == "NOTIFIED"
    assert session is not None and session.pending_clarification is not None
    assert record is None

    retried = _post(api_client, follow_up)
    assert retried.status_code == 200
    assert telegram_client.messages[-1][1] == "Marked as done: call James."
    db.expire_all()
    db.refresh(second)
    assert second.status == "COMPLETED"
