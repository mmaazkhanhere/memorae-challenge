"""Route incoming reminder messages before any deterministic or model parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from reminder_agent.intent import PendingReminder
from reminder_agent.cancellation import is_cancel_command
from reminder_agent.snooze import is_snooze_command


class ConversationRoute(StrEnum):
    NEW_REQUEST = "NEW_REQUEST"
    LIST_REQUEST = "LIST_REQUEST"
    SUMMARY_REQUEST = "SUMMARY_REQUEST"
    RESCHEDULE_REQUEST = "RESCHEDULE_REQUEST"
    CANCEL_REQUEST = "CANCEL_REQUEST"
    COMPLETE_REQUEST = "COMPLETE_REQUEST"
    SNOOZE_REQUEST = "SNOOZE_REQUEST"
    CLARIFICATION_REPLY = "CLARIFICATION_REPLY"
    CANCELLATION = "CANCELLATION"
    SOCIAL = "SOCIAL"
    MODEL_FALLBACK = "MODEL_FALLBACK"
    UNRELATED = "UNRELATED"


@dataclass(frozen=True)
class RouteDecision:
    route: ConversationRoute


_NEW_REMINDER = re.compile(
    r"^\s*(?:please\s+)?(?:remind\s+me|set\s+(?:a\s+)?reminder)(?:\s+|$)",
    re.IGNORECASE,
)
_LIST_REMINDERS = re.compile(
    r"^\s*(?:can\s+you\s+)?(?:what\s+(?:do\s+i\s+have|are\s+my\s+reminders?|"
    r"reminders?\s+do\s+i\s+have)|"
    r"(?:list|show)(?:\s+(?:my|me))?\s+(?:all\s+(?:the\s+)?)?(?:recurring\s+)?reminders?)\b",
    re.IGNORECASE,
)
_SCHEDULE_LIST_REQUEST = re.compile(
    r"^\s*(?:what(?:'s|\s+is)\s+(?:(?:on|in)\s+)?(?:my\s+)?schedule|"
    r"what\s+is\s+scheduled|anything\s+coming\s+up)\b",
    re.IGNORECASE,
)
_TASK_LIST_REQUEST = re.compile(
    r"^\s*(?:when\s+(?:are|is)\s+(?:my\s+)?\S+|when\s+do\s+i\s+have\s+to\s+\S+|"
    r"(?:show|list)\s+(?:my\s+)?\S.+\s+reminders?)\b",
    re.IGNORECASE,
)
_SUMMARY_REQUEST = re.compile(
    r"^\s*(?:(?:how\s+many|what\s+is\s+the\s+number\s+of).*(?:pending|outstanding|unfinished|overdue|completed|done|finished)(?:.*(?:today|this\s+day))?"
    r"|(?:what|which|show|list).*(?:still\s+have\s+to\s+do|have\s+left|remaining).*(?:today|this\s+day)"
    r"|(?:what|which|show|list).*(?:today|this\s+day).*(?:still\s+have\s+to\s+do|have\s+left|remaining)"
    r"|(?:what|which|show|list).*(?:completed|done|finished).*(?:today|this\s+day))\s*[?!.]*\s*$",
    re.IGNORECASE,
)
_CLOCK_WITH_AT = re.compile(
    r"\bat\s+\d{1,2}(?::\d{2})?\s*(?:a\.m\.|p\.m\.|am|pm)?\b",
    re.IGNORECASE,
)
_BARE_CLOCK = re.compile(
    r"\s*\d{1,2}(?::\d{2})?\s*(?:a\.m\.|p\.m\.|am|pm)?\s*[,.]?\s*",
    re.IGNORECASE,
)
_DAY = re.compile(
    r"\b(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_TASK_CORRECTION = re.compile(
    r"^\s*(?:change|update|replace)\s+(?:the\s+)?(?:task|reminder|it)\s+"
    r"(?:to|with)\s+\S+",
    re.IGNORECASE,
)
_CANCELLATIONS = {"cancel", "cancel it", "never mind", "nevermind", "not now"}
_COMPLETION = re.compile(
    r"^\s*(?:done(?:\s*,?\s*thanks?)?|completed|finished(?:\s+it)?|"
    r"mark(?:ed)?\s+(?:it|that)\s+done|"
    r"mark\s+(?:the\s+)?(?:reminder|task)\s+(?:as\s+)?complete)\s*[.!?]*\s*$",
    re.IGNORECASE,
)


def is_completion_command(text: str) -> bool:
    return _COMPLETION.fullmatch(text.strip()) is not None
_ACKNOWLEDGEMENTS = {
    "yes",
    "yeah",
    "okay",
    "ok",
    "sure",
    "no",
    "nope",
    "thanks",
    "thank you",
}
_CONFIRMATION_WORDS = {"yes", "yeah", "okay", "ok", "sure", "no", "nope"}
_GREETINGS = {
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
}
_QUESTION = re.compile(r"^(?:what|when|where|who|why|how|can|could|do|does|is|are)\b", re.IGNORECASE)


def is_new_reminder_command(text: str) -> bool:
    """Return whether text explicitly starts a separate reminder request."""
    return _NEW_REMINDER.match(text.strip()) is not None


def is_list_request_command(text: str) -> bool:
    """Return whether text explicitly asks to view reminders or a schedule."""
    raw = text.strip()
    return (
        _LIST_REMINDERS.match(raw) is not None
        or _SCHEDULE_LIST_REQUEST.match(raw) is not None
        or _TASK_LIST_REQUEST.match(raw) is not None
    )


def is_summary_request_command(text: str) -> bool:
    """Return whether text asks for reminder progress/count information."""
    return _SUMMARY_REQUEST.fullmatch(text.strip()) is not None


def route_message(text: str, *, pending: PendingReminder | None) -> RouteDecision:
    """Choose a safe deterministic path or the model fallback.

    This deliberately does not attempt to determine whether free-form text is
    a reminder.  Apart from short social messages and a bare clock with no
    pending conversation, that judgment belongs to the model.
    """
    raw = text.strip()
    normalized = re.sub(r"[.!?]+$", "", raw.lower()).strip()

    if is_completion_command(raw):
        return RouteDecision(ConversationRoute.COMPLETE_REQUEST)
    if is_snooze_command(raw):
        return RouteDecision(ConversationRoute.SNOOZE_REQUEST)

    # An acknowledgement is conversational only when there is no active
    # clarification.  With pending context it must be interpreted as an answer
    # to the outstanding question, or preserve that state and ask again.
    if pending is None and (normalized in _ACKNOWLEDGEMENTS or normalized in _GREETINGS):
        return RouteDecision(ConversationRoute.SOCIAL)

    if pending is None:
        if is_new_reminder_command(raw):
            return RouteDecision(ConversationRoute.NEW_REQUEST)
        if is_summary_request_command(raw):
            return RouteDecision(ConversationRoute.SUMMARY_REQUEST)
        if is_list_request_command(raw):
            return RouteDecision(ConversationRoute.LIST_REQUEST)
        if is_cancel_command(raw):
            return RouteDecision(ConversationRoute.CANCEL_REQUEST)
        if _BARE_CLOCK.fullmatch(raw):
            return RouteDecision(ConversationRoute.UNRELATED)
        return RouteDecision(ConversationRoute.MODEL_FALLBACK)

    if normalized in _CANCELLATIONS:
        return RouteDecision(ConversationRoute.CANCELLATION)
    if is_cancel_command(raw):
        return RouteDecision(ConversationRoute.CANCEL_REQUEST)
    if normalized in _CONFIRMATION_WORDS:
        return RouteDecision(ConversationRoute.CLARIFICATION_REPLY)
    if normalized in _ACKNOWLEDGEMENTS:
        return RouteDecision(ConversationRoute.SOCIAL)
    if is_list_request_command(raw):
        return RouteDecision(ConversationRoute.LIST_REQUEST)
    if is_summary_request_command(raw):
        return RouteDecision(ConversationRoute.SUMMARY_REQUEST)
    if is_new_reminder_command(raw):
        # A complete new command intentionally supersedes a stale draft rather
        # than being parsed as an edit to its task or schedule.
        return RouteDecision(ConversationRoute.NEW_REQUEST)
    # Only explicit new-reminder commands are classified before considering
    # the active conversation.  A message containing a date/time may still be
    # a new reminder, an answer to the pending question, or something else;
    # preserve context and let the interpreter decide semantically.
    if _TASK_CORRECTION.match(raw):
        return RouteDecision(ConversationRoute.CLARIFICATION_REPLY)

    # A follow-up that cannot be verified as a simple field is context-aware
    # interpretation and therefore belongs to the model.  The service still
    # permits only explicit, validated field updates from that model response.
    return RouteDecision(ConversationRoute.MODEL_FALLBACK)


def social_response(text: str, *, user_name: str | None = None) -> str:
    """Return a small, deterministic response without invoking the model."""
    normalized = re.sub(r"[.!?]+$", "", text.strip().lower()).strip()
    name = user_name.strip().split(maxsplit=1)[0] if user_name and user_name.strip() else None
    if normalized in _GREETINGS:
        greeting = f"Hi {name}!" if name else "Hi!"
        return f"{greeting} What would you like me to remind you about?"
    if normalized in _ACKNOWLEDGEMENTS:
        return "You're welcome."
    return "I can help create reminders. Tell me what to remember and when."


def unrelated_response(text: str, *, pending: PendingReminder | None) -> str:
    """Return a non-mutating response for text outside the reminder flow."""
    normalized = re.sub(r"[.!?]+$", "", text.strip().lower()).strip()
    if normalized in _ACKNOWLEDGEMENTS or normalized in _GREETINGS:
        return social_response(text)
    if pending is not None:
        return "Your unfinished reminder is still saved for when you're ready."
    return "I didn't find a reminder request in that message."
