from sqlalchemy import func, select

from reminder_agent.features.intent import (
    IntentRoute,
    IntentRouteSuggestion,
    Interpretation,
    ListRemindersIntent,
)
from reminder_agent.features.reminders.listing.filters import REMINDER_SUMMARY_MESSAGES
from reminder_agent.infrastructure.db.models import (
    ConversationSession,
    IdempotencyRecord,
    Notification,
    Reminder,
    ReminderOccurrence,
)

HEADERS = {"X-Telegram-Bot-Api-Secret-Token": "test-secret"}


def post_update(client, update: dict):
    return client.post("/webhook/telegram", json=update, headers=HEADERS)


def onboard(api_client, update_factory) -> None:
    assert post_update(api_client, update_factory(500)).status_code == 200
    assert post_update(api_client, update_factory(501, text="Asia/Karachi")).status_code == 200


def test_telegram_listing_returns_one_time_and_recurring_items_without_writes(
    api_client, telegram_client, db, update_factory
):
    onboard(api_client, update_factory)
    assert (
        post_update(
            api_client,
            update_factory(502, text="Remind me tomorrow at 9 AM to renew the domain."),
        ).status_code
        == 200
    )
    assert (
        post_update(
            api_client,
            update_factory(503, text="Remind me every Saturday at 10 AM to review the report."),
        ).status_code
        == 200
    )

    before = {
        "reminders": db.scalar(select(func.count()).select_from(Reminder)),
        "occurrences": db.scalar(select(func.count()).select_from(ReminderOccurrence)),
        "notifications": db.scalar(select(func.count()).select_from(Notification)),
        "sessions": db.scalar(select(func.count()).select_from(ConversationSession)),
        "idempotency": db.scalar(select(func.count()).select_from(IdempotencyRecord)),
    }
    response = post_update(api_client, update_factory(504, text="What do I have tomorrow?"))

    assert response.status_code == 200
    text = telegram_client.messages[-1][1]
    assert "renew the domain" in text
    assert "review the report" in text
    assert "tomorrow:" in text
    assert "09:00 PKT" in text
    assert "10:00 PKT" in text
    after = {
        "reminders": db.scalar(select(func.count()).select_from(Reminder)),
        "occurrences": db.scalar(select(func.count()).select_from(ReminderOccurrence)),
        "notifications": db.scalar(select(func.count()).select_from(Notification)),
        "sessions": db.scalar(select(func.count()).select_from(ConversationSession)),
        "idempotency": db.scalar(select(func.count()).select_from(IdempotencyRecord)),
    }
    assert after == before


def test_telegram_listing_is_owner_scoped_and_empty_is_explicit(
    api_client, telegram_client, db, update_factory
):
    onboard(api_client, update_factory)
    response = post_update(api_client, update_factory(505, text="What do I have on 2026-11-03?"))

    assert response.status_code == 200
    assert telegram_client.messages[-1][1] == "No reminders for Tuesday. Want to add something?"


def test_telegram_progress_questions_are_read_only_and_owner_scoped(
    api_client, telegram_client, db, update_factory
):
    onboard(api_client, update_factory)
    assert (
        post_update(
            api_client,
            update_factory(508, text="Remind me today at 6 PM to submit the report"),
        ).status_code
        == 200
    )
    assert (
        post_update(
            api_client,
            update_factory(509, text="Remind me tomorrow at 9 AM to call the bank"),
        ).status_code
        == 200
    )

    response = post_update(api_client, update_factory(519, text="How many reminders are pending?"))
    assert response.status_code == 200
    assert telegram_client.messages[-1][1] in {
        message.format(count=2, reminder_word="reminders")
        for message in REMINDER_SUMMARY_MESSAGES["pending"]
    }

    response = post_update(
        api_client, update_factory(523, text="What do I still have to do today?")
    )
    assert response.status_code == 200
    assert "You still have 1 task to do today:" in telegram_client.messages[-1][1]
    assert "submit the report" in telegram_client.messages[-1][1]
    assert "call the bank" not in telegram_client.messages[-1][1]


def test_listing_does_not_discard_an_unfinished_reminder_clarification(
    api_client, telegram_client, db, update_factory
):
    onboard(api_client, update_factory)
    assert (
        post_update(api_client, update_factory(506, text="Remind me to call the bank")).status_code
        == 200
    )
    session_before = db.scalar(select(ConversationSession))
    assert session_before is not None and session_before.status == "AWAITING_CLARIFICATION"

    response = post_update(api_client, update_factory(507, text="What do I have tomorrow?"))

    assert response.status_code == 200
    session_after = db.scalar(select(ConversationSession))
    assert session_after is not None
    assert session_after.status == "AWAITING_CLARIFICATION"
    assert session_after.pending_clarification == session_before.pending_clarification
    assert "nothing on your reminder list for tomorrow" in telegram_client.messages[-1][1].lower()


def test_task_specific_listing_filters_owned_rows_and_combines_with_tomorrow(
    api_client, telegram_client, db, update_factory
):
    onboard(api_client, update_factory)
    for update_id, text in (
        (510, "Remind me tomorrow at 9 AM to doctor appointment"),
        (511, "Remind me tomorrow at 10 AM to Call John"),
        (512, "Remind me tomorrow at 11 AM to gym workout"),
    ):
        assert post_update(api_client, update_factory(update_id, text=text)).status_code == 200

    assert (
        post_update(
            api_client, update_factory(513, text="When are my doctor's appointments?")
        ).status_code
        == 200
    )
    text = telegram_client.messages[-1][1]
    assert "doctor appointment" in text and "Call John" not in text and "gym workout" not in text

    assert (
        post_update(
            api_client, update_factory(514, text="When do I have to call John?")
        ).status_code
        == 200
    )
    text = telegram_client.messages[-1][1]
    assert "Call John" in text and "doctor appointment" not in text and "gym workout" not in text

    assert (
        post_update(api_client, update_factory(515, text="Show my gym reminders.")).status_code
        == 200
    )
    assert "gym workout" in telegram_client.messages[-1][1]

    assert (
        post_update(
            api_client, update_factory(516, text="When are my doctor's appointments tomorrow?")
        ).status_code
        == 200
    )
    text = telegram_client.messages[-1][1]
    assert "doctor appointment" in text and "Call John" not in text

    assert (
        post_update(api_client, update_factory(517, text="What reminders do I have?")).status_code
        == 200
    )
    text = telegram_client.messages[-1][1]
    assert all(task in text for task in ("doctor appointment", "Call John", "gym workout"))

    assert (
        post_update(api_client, update_factory(518, text="Show my dentist reminders.")).status_code
        == 200
    )
    assert telegram_client.messages[-1][1] == 'Nothing matching "dentist" on your upcoming reminder list.'
    assert db.scalar(select(func.count()).select_from(Reminder)) == 3


def test_ambiguous_listing_uses_model_task_filter_without_trusting_model_records(
    api_client, telegram_client, db, update_factory
):
    class ListRouter:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(operation=IntentRoute.LIST_REQUEST)

    class ListExtractor:
        called = False

        def interpret(self, text, *, selected_route, timezone_name, now):
            self.called = True
            assert selected_route is IntentRoute.LIST_REQUEST
            return Interpretation(
                intent=ListRemindersIntent(task_query="gym", timezone_name=timezone_name)
            )

    onboard(api_client, update_factory)
    assert (
        post_update(
            api_client, update_factory(520, text="Remind me tomorrow at 9 AM to gym workout")
        ).status_code
        == 200
    )
    assert (
        post_update(
            api_client, update_factory(521, text="Remind me tomorrow at 10 AM to Call John")
        ).status_code
        == 200
    )
    extractor = ListExtractor()
    api_client.app.state.intent_router = ListRouter()
    api_client.app.state.reminder_interpreter = extractor

    assert (
        post_update(
            api_client, update_factory(522, text="Can you pull up my fitness things?")
        ).status_code
        == 200
    )
    assert extractor.called is True
    text = telegram_client.messages[-1][1]
    assert "gym workout" in text and "Call John" not in text
