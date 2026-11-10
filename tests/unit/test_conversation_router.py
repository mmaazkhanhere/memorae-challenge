from datetime import date

import pytest

from reminder_agent.features.intent import PendingReminder
from reminder_agent.features.intent.deterministic.routing import (
    ConversationRoute,
    is_completion_command,
    is_list_request_command,
    is_memory_lookup_command,
    is_memory_save_command,
    is_reminder_status_question,
    is_summary_request_command,
    route_message,
    social_response,
)


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


def test_router_recognizes_also_remind_me_as_a_new_request() -> None:
    assert route_message(
        "Also remind me tomorrow at 2 to email the client", pending=None
    ).route is ConversationRoute.NEW_REQUEST


def test_router_recognizes_read_only_listing_with_or_without_pending_draft() -> None:
    for text in ("What do I have tomorrow?", "What reminders do I have tomorrow?"):
        assert route_message(text, pending=None).route is ConversationRoute.LIST_REQUEST
        assert route_message(text, pending=_pending()).route is ConversationRoute.LIST_REQUEST


def test_router_recognizes_recurring_reminder_list_requests() -> None:
    assert (
        route_message("can you show me all the recurring reminders?", pending=None).route
        == ConversationRoute.LIST_REQUEST
    )
    assert (
        route_message("what do i have on every Tuesday?", pending=None).route
        == ConversationRoute.LIST_REQUEST
    )
    assert route_message("Show my reminders this week", pending=_pending()).route is (
        ConversationRoute.LIST_REQUEST
    )


def test_router_recognizes_schedule_list_wording() -> None:
    for text in ("What's on my schedule tomorrow?", "Anything coming up this week?"):
        assert is_list_request_command(text) is True
        assert route_message(text, pending=None).route is ConversationRoute.LIST_REQUEST
        assert route_message(text, pending=_pending()).route is ConversationRoute.LIST_REQUEST


def test_router_recognizes_progress_questions() -> None:
    for text in (
        "How many reminders are pending?",
        "What do I still have to do today?",
        "How many tasks have I completed?",
        "How many reminders are overdue?",
    ):
        assert is_summary_request_command(text) is True
        assert route_message(text, pending=None).route is ConversationRoute.SUMMARY_REQUEST
        assert route_message(text, pending=_pending()).route is ConversationRoute.SUMMARY_REQUEST

    assert route_message('"How many reminders do I have pending?"', pending=None).route is (
        ConversationRoute.SUMMARY_REQUEST
    )


def test_router_recognizes_explicit_cancellation_requests() -> None:
    assert (
        route_message("Cancel the standup", pending=None).route is ConversationRoute.CANCEL_REQUEST
    )
    assert route_message("Cancel only tomorrow's standup", pending=_pending()).route is (
        ConversationRoute.CANCEL_REQUEST
    )


def test_router_recognizes_completion_without_invoking_reminder_creation() -> None:
    for text in ("Done", "Done, thanks", "finished it", "mark the reminder as complete"):
        assert is_completion_command(text) is True
        assert route_message(text, pending=None).route is ConversationRoute.COMPLETE_REQUEST
    assert is_completion_command("Remind me to mark the task done") is False


def test_router_recognizes_replied_reminder_status_question() -> None:
    assert is_reminder_status_question("Will you still remind me next Monday?") is True
    assert is_reminder_status_question("remind me next Monday") is False


def test_router_does_not_start_a_reminder_from_unrelated_text() -> None:
    assert route_message("7 PM", pending=None).route is ConversationRoute.UNRELATED
    assert route_message("I have five tasks today", pending=None).route is (
        ConversationRoute.MODEL_FALLBACK
    )


def test_router_recognizes_explicit_memory_save_deterministically() -> None:
    for text in (
        "Remember my gate code is 4512",
        "Make a note that my laptop model is ThinkPad T14",
        "Take note of my backup email",
    ):
        assert is_memory_save_command(text) is True
        assert route_message(text, pending=None).route is (ConversationRoute.SAVE_MEMORY_REQUEST)


def test_router_recognizes_memory_lookup_without_model_fallback() -> None:
    for text in ("What's my gate code?", "What is my Wi-Fi password?"):
        assert is_memory_lookup_command(text) is True
        assert route_message(text, pending=None).route is ConversationRoute.RETRIEVE_MEMORY_REQUEST


def test_ambiguous_temporal_personal_question_uses_model_fallback() -> None:
    assert route_message("When is my birthday?", pending=None).route is (
        ConversationRoute.MODEL_FALLBACK
    )


@pytest.mark.parametrize(
    "text",
    [
        "What's my reminder for tomorrow?",
        "What reminders do I have tomorrow?",
        "What's my next reminder?",
        "What is my upcoming reminder?",
    ],
)
def test_router_prefers_reminder_listing_over_memory_lookup_collision(text: str) -> None:
    assert route_message(text, pending=None).route is ConversationRoute.LIST_REQUEST


@pytest.mark.parametrize(
    "text",
    [
        "Any reminders for Friday?",
        "What's my next reminder to do?",
        "Do I have any payment reminders?",
        "Show reminders after 6 PM.",
        "What do I have scheduled for next month?",
        "Do I have any reminders on weekends?",
    ],
)
def test_router_recognizes_rich_listing_questions(text: str) -> None:
    assert route_message(text, pending=None).route is ConversationRoute.LIST_REQUEST


def test_next_reminder_is_not_treated_as_a_memory_lookup() -> None:
    assert is_memory_lookup_command("What's my next reminder?") is False


def test_router_distinguishes_unambiguous_reminders_from_memory_facts() -> None:
    assert (
        route_message("Remember to call Ali tomorrow at 9 AM", pending=None).route
        is ConversationRoute.NEW_REQUEST
    )
    assert route_message("Remember to call Ali", pending=None).route is (
        ConversationRoute.NEW_REQUEST
    )
    assert route_message("Save a reminder to call Ali tomorrow at 9", pending=None).route is (
        ConversationRoute.NEW_REQUEST
    )
    assert route_message("Remember my gate code is 4512", pending=None).route is (
        ConversationRoute.SAVE_MEMORY_REQUEST
    )
    assert route_message("Remember that my birthday is January 10", pending=None).route is (
        ConversationRoute.SAVE_MEMORY_REQUEST
    )


def test_router_defers_overlapping_or_ambiguous_creation_language() -> None:
    for text in (
        "Remember to keep my gate code in mind",
        "Don't forget Ali's birthday",
        "My gate code is 4512",
    ):
        assert route_message(text, pending=None).route is ConversationRoute.MODEL_FALLBACK


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
