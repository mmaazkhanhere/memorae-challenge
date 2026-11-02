from datetime import date

from reminder_agent.conversation_router import ConversationRoute, route_message, social_response
from reminder_agent.intent import PendingReminder


def _pending(*, task: str | None = "call the bank") -> PendingReminder:
    return PendingReminder(
        task=task,
        date_local=date(2026, 10, 31),
        hour=7,
        timezone_name="UTC",
        date_explicit=True,
    )


def test_router_classifies_pending_conversation_before_parsing() -> None:
    pending = _pending()

    assert route_message("Remind me to email Sam tomorrow at 9", pending=pending).route is (
        ConversationRoute.NEW_REQUEST
    )
    assert route_message("8 AM", pending=pending).route is ConversationRoute.MODEL_FALLBACK
    assert route_message("Never mind", pending=pending).route is ConversationRoute.CANCELLATION
    assert route_message("yes", pending=pending).route is ConversationRoute.CLARIFICATION_REPLY
    assert route_message("thanks", pending=pending).route is ConversationRoute.SOCIAL


def test_router_recognizes_read_only_listing_with_or_without_pending_draft() -> None:
    assert route_message("What do I have tomorrow?", pending=None).route is (
        ConversationRoute.LIST_REQUEST
    )


def test_router_recognizes_recurring_reminder_list_requests() -> None:
    assert route_message("can you show me all the recurring reminders?", pending=None).route == ConversationRoute.LIST_REQUEST
    assert route_message("what do i have on every Tuesday?", pending=None).route == ConversationRoute.LIST_REQUEST
    assert route_message("Show my reminders this week", pending=_pending()).route is (
        ConversationRoute.LIST_REQUEST
    )


def test_router_does_not_start_a_reminder_from_unrelated_text() -> None:
    assert route_message("7 PM", pending=None).route is ConversationRoute.UNRELATED
    assert route_message("I have five tasks today", pending=None).route is (
        ConversationRoute.MODEL_FALLBACK
    )


def test_router_allows_free_text_only_when_the_task_is_missing() -> None:
    pending_without_task = _pending(task=None)

    assert route_message("call the doctor", pending=pending_without_task).route is (
        ConversationRoute.MODEL_FALLBACK
    )
    assert route_message("what time is it?", pending=pending_without_task).route is (
        ConversationRoute.MODEL_FALLBACK
    )


def test_router_keeps_greetings_and_thanks_deterministic_and_personalized() -> None:
    assert route_message("Hi!", pending=None).route is ConversationRoute.SOCIAL
    assert route_message("Thanks", pending=None).route is ConversationRoute.SOCIAL
    assert social_response("Hi!", user_name="Ada Lovelace") == (
        "Hi Ada! What would you like me to remind you about?"
    )
    assert social_response("Thanks", user_name="Ada Lovelace") == "You're welcome."
