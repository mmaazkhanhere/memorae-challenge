from datetime import UTC, datetime, timedelta

import httpx

from reminder_agent.application.conversation_coordinator import _route_and_interpret
from reminder_agent.features.intent import (
    IntentRoute,
    IntentRouteSuggestion,
    Interpretation,
    OpenAICompatibleIntentInterpreter,
)
from reminder_agent.features.reminders.snooze import (
    parse_snooze_duration,
    snooze_occurrence,
    snooze_target_text,
)
from reminder_agent.infrastructure.db.models import (
    ChannelIdentity,
    Notification,
    Reminder,
    ReminderOccurrence,
    User,
)

NOW = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)


def _seed(db, *, recurring: bool = False):
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
    reminder = Reminder(
        user_id=user.id,
        task="call the bank",
        schedule_type="RECURRING" if recurring else "ONE_TIME",
        start_local="2026-11-03T12:00:00",
        timezone="UTC",
        recurrence_rule="FREQ=WEEKLY;BYDAY=TU;BYHOUR=12;BYMINUTE=00" if recurring else None,
    )
    db.add(reminder)
    db.flush()
    occurrence = ReminderOccurrence(
        reminder_id=reminder.id,
        scheduled_at=NOW - timedelta(minutes=1),
        next_due_at=NOW - timedelta(minutes=1),
        status="NOTIFIED",
    )
    db.add(occurrence)
    db.flush()
    sent = Notification(
        occurrence_id=occurrence.id,
        destination_identity_id=identity.id,
        sequence=1,
        due_at=occurrence.scheduled_at,
        status="SENT",
        sent_at=occurrence.scheduled_at,
        provider_message_id="777",
    )
    db.add(sent)
    db.commit()
    return user, identity, reminder, occurrence, sent


def test_parse_snooze_duration_accepts_explicit_positive_relative_durations():
    assert parse_snooze_duration("10 more minutes").duration_seconds == 600
    assert parse_snooze_duration("snooze it for 1 hour").duration_seconds == 3600
    assert parse_snooze_duration("for 2 days").duration_seconds == 172800
    assert parse_snooze_duration("snooze the bank for 15 mins").duration_seconds == 900
    assert snooze_target_text("snooze the bank for 15 mins") == "bank"


def test_parse_snooze_duration_rejects_non_positive_or_ambiguous_values():
    assert parse_snooze_duration("0 minutes") is None
    assert parse_snooze_duration("-10 minutes") is None
    assert parse_snooze_duration("snooze") is None
    assert parse_snooze_duration("tomorrow at 10") is None


def test_snooze_stages_one_new_notification_without_creating_occurrence(db):
    user, identity, reminder, occurrence, sent = _seed(db)

    result = snooze_occurrence(
        db,
        user_id=user.id,
        identity_id=identity.id,
        chat_id=2001,
        occurrence_id=occurrence.id,
        duration_seconds=600,
        provider_message_id=sent.provider_message_id,
        now=NOW,
    )

    assert result.outcome == "SNOOZED"
    assert result.due_at == NOW + timedelta(minutes=10)
    db.flush()
    db.refresh(occurrence)
    db.refresh(sent)
    notifications = (
        db.query(Notification)
        .filter_by(occurrence_id=occurrence.id)
        .order_by(Notification.sequence)
        .all()
    )
    assert occurrence.status == "PENDING"
    assert occurrence.next_due_at.replace(tzinfo=UTC) == NOW + timedelta(minutes=10)
    assert reminder.status == "ACTIVE"
    assert len(notifications) == 2
    assert notifications[0].status == "SENT"
    assert notifications[0].provider_message_id == "777"
    assert notifications[1].status == "PENDING"
    assert notifications[1].sequence == 2
    assert notifications[1].due_at.replace(tzinfo=UTC) == NOW + timedelta(minutes=10)


def test_snooze_recurring_occurrence_does_not_change_series_schedule(db):
    user, identity, reminder, occurrence, sent = _seed(db, recurring=True)
    original_rule = reminder.recurrence_rule
    original_start = reminder.start_local

    result = snooze_occurrence(
        db,
        user_id=user.id,
        identity_id=identity.id,
        chat_id=2001,
        occurrence_id=occurrence.id,
        duration_seconds=600,
        provider_message_id=sent.provider_message_id,
        now=NOW,
    )

    assert result.outcome == "SNOOZED"
    assert reminder.recurrence_rule == original_rule
    assert reminder.start_local == original_start
    assert db.query(ReminderOccurrence).filter_by(reminder_id=reminder.id).count() == 1


def test_snooze_rejects_wrong_destination_and_non_delivered_state(db):
    user, identity, _reminder, occurrence, sent = _seed(db)

    wrong_chat = snooze_occurrence(
        db,
        user_id=user.id,
        identity_id=identity.id,
        chat_id=9999,
        occurrence_id=occurrence.id,
        duration_seconds=600,
        provider_message_id=sent.provider_message_id,
        now=NOW,
    )
    assert wrong_chat.outcome == "NOT_FOUND"

    wrong_message = snooze_occurrence(
        db,
        user_id=user.id,
        identity_id=identity.id,
        chat_id=2001,
        occurrence_id=occurrence.id,
        duration_seconds=600,
        provider_message_id="not-the-delivered-message",
        now=NOW,
    )
    assert wrong_message.outcome == "NOT_DELIVERED"

    occurrence.status = "COMPLETED"
    db.commit()
    completed = snooze_occurrence(
        db,
        user_id=user.id,
        identity_id=identity.id,
        chat_id=2001,
        occurrence_id=occurrence.id,
        duration_seconds=600,
        provider_message_id=sent.provider_message_id,
        now=NOW,
    )
    assert completed.outcome == "NOT_ELIGIBLE"


def test_snooze_rejects_invalid_duration_without_mutation(db):
    user, identity, _reminder, occurrence, sent = _seed(db)
    result = snooze_occurrence(
        db,
        user_id=user.id,
        identity_id=identity.id,
        chat_id=2001,
        occurrence_id=occurrence.id,
        duration_seconds=0,
        provider_message_id=sent.provider_message_id,
        now=NOW,
    )
    assert result.outcome == "INVALID_DURATION"
    db.refresh(occurrence)
    assert occurrence.status == "NOTIFIED"
    assert db.query(Notification).filter_by(occurrence_id=occurrence.id).count() == 1


def test_deterministic_snooze_duration_does_not_call_classifier_or_external_extractor():
    class FailingClassifier:
        def classify(self, *args, **kwargs):  # pragma: no cover - assertion below
            raise AssertionError("deterministic snooze must not call the classifier")

    class FailingExtractor:
        def interpret(self, *args, **kwargs):  # pragma: no cover - assertion below
            raise AssertionError("complete snooze duration must not call the extractor")

    from reminder_agent.features.intent import ReminderIntentInterpreter

    route, result = _route_and_interpret(
        ReminderIntentInterpreter(FailingExtractor()),
        "10 more minutes",
        timezone_name="UTC",
        now=NOW,
        pending=None,
        user_name="Ada",
        intent_router=FailingClassifier(),
    )

    assert route.value == "SNOOZE_REQUEST"
    assert result.snooze_duration_seconds == 600


def test_classifier_snooze_route_can_propose_only_a_validated_duration():
    class SnoozeClassifier:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(operation=IntentRoute.SNOOZE_REQUEST)

    class SnoozeExtractor:
        def interpret(self, text, **kwargs):
            return Interpretation(snooze_duration_seconds=900, fallback_allowed=False)

    route, result = _route_and_interpret(
        SnoozeExtractor(),
        "postpone that reminder",
        timezone_name="UTC",
        now=NOW,
        pending=None,
        user_name="Ada",
        intent_router=SnoozeClassifier(),
    )

    assert route.value == "SNOOZE_REQUEST"
    assert result.snooze_duration_seconds == 900


def test_model_snooze_proposal_rejects_target_authority_and_invalid_shape(monkeypatch):
    def fake_post(*args, **kwargs):
        return httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"operation":"SNOOZE_OCCURRENCE",'
                                '"duration_seconds":600,"occurrence_id":"forged"}'
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = OpenAICompatibleIntentInterpreter(
        "https://llm.example/v1", "secret", "provider-model"
    ).interpret(
        "postpone the bank reminder",
        selected_route=IntentRoute.SNOOZE_REQUEST,
        timezone_name="UTC",
        now=NOW,
    )

    assert result.snooze_duration_seconds is None
    assert result.response_text is not None
