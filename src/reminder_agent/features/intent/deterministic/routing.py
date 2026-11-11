"""Route incoming reminder messages before any deterministic or model parsing."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from reminder_agent.features.intent.contracts import PendingReminder
from reminder_agent.features.intent.messages import general_message
from reminder_agent.features.reminders.cancel.decisions import is_cancel_command
from reminder_agent.features.reminders.snooze.operations import is_snooze_command
from reminder_agent.shared.creation_routing import (
    ExplicitCreateRoute,
    explicit_create_route,
    is_unsafe_creation_command,
)
from reminder_agent.shared.memory_text import extract_memory_lookup_query


class ConversationRoute(StrEnum):
    NEW_REQUEST = "NEW_REQUEST"
    LIST_REQUEST = "LIST_REQUEST"
    SUMMARY_REQUEST = "SUMMARY_REQUEST"
    RESCHEDULE_REQUEST = "RESCHEDULE_REQUEST"
    CANCEL_REQUEST = "CANCEL_REQUEST"
    COMPLETE_REQUEST = "COMPLETE_REQUEST"
    SNOOZE_REQUEST = "SNOOZE_REQUEST"
    SAVE_MEMORY_REQUEST = "SAVE_MEMORY_REQUEST"
    RETRIEVE_MEMORY_REQUEST = "RETRIEVE_MEMORY_REQUEST"
    CLARIFICATION_REPLY = "CLARIFICATION_REPLY"
    CANCELLATION = "CANCELLATION"
    SOCIAL = "SOCIAL"
    CAPABILITIES = "CAPABILITIES"
    MODEL_FALLBACK = "MODEL_FALLBACK"
    UNRELATED = "UNRELATED"


@dataclass(frozen=True)
class RouteDecision:
    route: ConversationRoute


_LIST_REMINDERS = re.compile(
    r"^\s*(?:can\s+you\s+)?(?:what\s+(?:do\s+i\s+have|are\s+my\s+reminders?|"
    r"reminders?\s+do\s+i\s+have)|"
    r"(?:any|do\s+i\s+have|are\s+there)\s+(?:any\s+)?reminders?(?:\s+(?:for|on)\s+\S+)?|"
    r"what(?:['Ã¢â‚¬â„¢]s|\s+is)\s+(?:my\s+)?reminders?(?:\s+(?:for|on|tomorrow|today))?|"
    r"(?:list|show)(?:\s+(?:my|me))?\s+(?:all\s+(?:the\s+)?)?(?:recurring\s+)?reminders?)\b",
    re.IGNORECASE,
)
_SCHEDULE_LIST_REQUEST = re.compile(
    r"^\s*(?:what(?:'s|\s+is)\s+(?:(?:on|in)\s+)?(?:my\s+)?schedule|"
    r"what\s+is\s+scheduled|anything\s+coming\s+up)\b",
    re.IGNORECASE,
)
_TASK_LIST_REQUEST = re.compile(
    r"^\s*(?:when\s+are\s+(?:my\s+)?\S+|when\s+do\s+i\s+have\s+to\s+\S+|"
    r"(?:show|list)\s+(?:my\s+)?\S.+\s+reminders?|"
    r"(?:do\s+i\s+have|are\s+there)\s+(?:any\s+)?\S.+\s+reminders?)\b",
    re.IGNORECASE,
)
_NEXT_REMINDER_REQUEST = re.compile(
    r"^\s*what(?:'s|\s+is)\s+(?:my\s+)?(?:next|upcoming)\s+reminders?(?:\s+to\s+do)?\s*[?!.]*\s*$",
    re.IGNORECASE,
)
_RICH_LIST_REQUEST = re.compile(
    r"^\s*(?:(?:hey|hi)[,\s]+)?(?:and\s+)?(?:"
    r"(?:what|which)\s+(?:are\s+)?(?:my\s+)?(?:weekly\s+)?(?:recurring\s+)?"
    r"reminders?\s+(?:have\s+i\s+got|do\s+i\s+have)|"
    r"(?:what(?:'s|\s+is)\s+)?(?:on\s+)?(?:my\s+)?(?:reminder\s+list|agenda)\s+"
    r"(?:for|on|this\s+week|next\s+week|the\s+weekend|tomorrow|today)\b|"
    r"(?:can\s+you\s+)?summari[sz]e\s+.*(?:this\s+week|weekly|weekend|recurring|agenda)|"
    r"(?:anything[,\s]+.*(?:forgot|take\s+care\s+of|need\s+to\s+remember))|"
    r"(?:and\s+)?on\s+(?:monday|tuesday|wednesday|thursday|friday|firday|saturday|sunday)\b|"
    r"(?:show|list)\s+(?:(?:my|me)\s+)?all\s+(?:(?:my|me)\s+)?\w+reminders?)"
    r".*[?!.]*\s*$",
    re.IGNORECASE,
)
_SELECTION_FOLLOW_UP = re.compile(
    r"^\s*(?:which|what)\s+(?:one|on|item|reminder)\s+"
    r"(?:is\s+)?(?P<ordinal>first|second|third|1|2|3)\b.*$",
    re.IGNORECASE,
)
_SUMMARY_REQUEST = re.compile(
    r"^\s*(?:(?:how\s+many|what\s+is\s+the\s+number\s+of).*(?:pending|outstanding|unfinished|overdue|completed|done|finished)(?:.*(?:today|this\s+day))?.*"
    r"|anything[,\s]+.*(?:forgot|take\s+care\s+of|need\s+to\s+remember)"
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
_REMINDER_STATUS_QUESTION = re.compile(
    r"^\s*will\s+you\s+still\s+remind(?:\s+me)?\b.*[?!.]*\s*$",
    re.IGNORECASE,
)
_DIRECT_CANCEL = re.compile(
    r"^\s*(?:please\s+)?(?:can\s+you\s+(?:please\s+)?|could\s+you\s+(?:please\s+)?|"
    r"would\s+you\s+(?:please\s+)?|)"
    r"(?:cancel|delete|remove|stop)\s+.+?\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_EXPLICIT_CANCEL_INTENT = re.compile(
    r"^\s*(?:i|we)\s+(?:want|need|would\s+like)\s+to\s+"
    r"(?:cancel|delete|remove|stop)\s+.+?\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_CONTEXTUAL_CANCEL = re.compile(
    r"^\s*(?:please\s+|can\s+you\s+|could\s+you\s+)?"
    r"(?:cancel|delete|remove|stop)\s+"
    r"(?:it|this|that|the\s+above|above|previous|prior)\s*[.!?]*\s*$",
    re.IGNORECASE,
)


def is_completion_command(text: str) -> bool:
    return _COMPLETION.fullmatch(text.strip()) is not None


def is_reminder_status_question(text: str) -> bool:
    """Return whether text asks if a delivered reminder will recur."""
    return _REMINDER_STATUS_QUESTION.fullmatch(text.strip()) is not None


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
_QUESTION = re.compile(
    r"^(?:what|when|where|who|why|how|can|could|do|does|is|are)\b", re.IGNORECASE
)
_CAPABILITIES = re.compile(
    r"^\s*(?:what\s+(?:do\s+you\s+do|can\s+you\s+do)|"
    r"what\s+can\s+you\s+help(?:\s+me)?\s+with|"
    r"how\s+can\s+you\s+help(?:\s+me)?|"
    r"what\s+are\s+you\s+(?:for|able\s+to\s+do)|"
    r"tell\s+me\s+what\s+you\s+can\s+do)\s*[?!.]*\s*$",
    re.IGNORECASE,
)


def is_new_reminder_command(text: str) -> bool:
    """Return whether text explicitly starts a separate reminder request."""
    return explicit_create_route(text) is ExplicitCreateRoute.NEW_REMINDER


def is_memory_save_command(text: str) -> bool:
    """Return whether text explicitly asks to save information."""
    return explicit_create_route(text) is ExplicitCreateRoute.SAVE_MEMORY


def is_memory_lookup_command(text: str) -> bool:
    """Return whether text is a high-precision saved-fact lookup question."""
    # Reminder-shaped questions can match the broad "what's my ..." memory
    # pattern. Keep the semantic boundary explicit for direct callers too.
    return not is_list_request_command(text) and extract_memory_lookup_query(text) is not None


def is_list_request_command(text: str) -> bool:
    """Return whether text explicitly asks to view reminders or a schedule."""
    raw = text.strip()
    return (
        _LIST_REMINDERS.match(raw) is not None
        or _RICH_LIST_REQUEST.match(raw) is not None
        or _SCHEDULE_LIST_REQUEST.match(raw) is not None
        or _TASK_LIST_REQUEST.match(raw) is not None
        or _NEXT_REMINDER_REQUEST.match(raw) is not None
    )


def selection_index(text: str) -> int | None:
    """Return the zero-based ordinal requested from the latest list."""

    match = _SELECTION_FOLLOW_UP.fullmatch(text.strip())
    if match is None:
        return None
    return {"first": 0, "1": 0, "second": 1, "2": 1, "third": 2, "3": 2}[
        match.group("ordinal").lower()
    ]


def is_summary_request_command(text: str) -> bool:
    """Return whether text asks for reminder progress/count information."""
    normalized = text.strip().strip('"“”')
    return _SUMMARY_REQUEST.fullmatch(normalized) is not None


def is_clear_cancel_command(text: str) -> bool:
    """Return true only for an explicit cancellation operation.

    Indirect negations (for example, "I do not want you to save this") are
    intentionally left to the model because they do not establish that an
    existing reminder should be canceled.
    """
    return (
        _DIRECT_CANCEL.fullmatch(text.strip()) is not None
        or _EXPLICIT_CANCEL_INTENT.fullmatch(text.strip()) is not None
    ) and is_cancel_command(text)


def is_contextual_cancel_command(text: str) -> bool:
    """Return whether cancellation names only an unresolved pronoun/reference."""
    return _CONTEXTUAL_CANCEL.fullmatch(text.strip()) is not None


def route_message(
    text: str,
    *,
    pending: PendingReminder | None,
    selection_context: Sequence[dict[str, str]] = (),
) -> RouteDecision:
    """Choose a safe deterministic path or the model fallback.

    This deliberately does not attempt to determine whether free-form text is
    a reminder.  Apart from short social messages and a bare clock with no
    pending conversation, that judgment belongs to the model.
    """
    raw = text.strip()
    normalized = re.sub(r"[.!?]+$", "", raw.lower()).strip()

    creation_route = explicit_create_route(raw)
    if selection_context and selection_index(raw) is not None:
        return RouteDecision(ConversationRoute.LIST_REQUEST)
    # Creation wording can look explicit while still being a compound,
    # referential, or negated request.  Do not let unrelated deterministic
    # routes turn that ambiguity into a read or write operation.
    if (
        is_unsafe_creation_command(raw)
        or (
            creation_route is ExplicitCreateRoute.MODEL_FALLBACK
            and not (pending is not None and normalized in _CANCELLATIONS)
        )
    ):
        return RouteDecision(ConversationRoute.MODEL_FALLBACK)

    if is_completion_command(raw):
        return RouteDecision(ConversationRoute.COMPLETE_REQUEST)
    if is_snooze_command(raw):
        return RouteDecision(ConversationRoute.SNOOZE_REQUEST)
    # Reminder/date wording wins over a broad question-shaped memory lookup.
    # For example, "What's my reminder for tomorrow?" is a schedule query,
    # while "What's my gate code?" remains an exact memory lookup.
    # Summary questions are a narrower kind of read request than listings,
    # but can share "what/list/show" wording with memory lookups.
    if is_summary_request_command(raw):
        return RouteDecision(ConversationRoute.SUMMARY_REQUEST)
    if is_list_request_command(raw):
        return RouteDecision(ConversationRoute.LIST_REQUEST)
    if is_memory_lookup_command(raw):
        return RouteDecision(ConversationRoute.RETRIEVE_MEMORY_REQUEST)
    if creation_route is ExplicitCreateRoute.SAVE_MEMORY:
        return RouteDecision(ConversationRoute.SAVE_MEMORY_REQUEST)
    if creation_route is ExplicitCreateRoute.NEW_REMINDER:
        return RouteDecision(ConversationRoute.NEW_REQUEST)
    if _CAPABILITIES.fullmatch(raw):
        return RouteDecision(ConversationRoute.CAPABILITIES)
    # An acknowledgement is conversational only when there is no active
    # clarification.  With pending context it must be interpreted as an answer
    # to the outstanding question, or preserve that state and ask again.
    if pending is None and (normalized in _ACKNOWLEDGEMENTS or normalized in _GREETINGS):
        return RouteDecision(ConversationRoute.SOCIAL)

    if pending is None:
        if is_contextual_cancel_command(raw):
            return RouteDecision(ConversationRoute.MODEL_FALLBACK)
        if is_clear_cancel_command(raw):
            return RouteDecision(ConversationRoute.CANCEL_REQUEST)
        if _BARE_CLOCK.fullmatch(raw):
            return RouteDecision(ConversationRoute.UNRELATED)
        return RouteDecision(ConversationRoute.MODEL_FALLBACK)

    if normalized in _CANCELLATIONS:
        return RouteDecision(ConversationRoute.CANCELLATION)
    if is_contextual_cancel_command(raw):
        return RouteDecision(ConversationRoute.MODEL_FALLBACK)
    if is_clear_cancel_command(raw):
        return RouteDecision(ConversationRoute.CANCEL_REQUEST)
    if normalized in _CONFIRMATION_WORDS:
        return RouteDecision(ConversationRoute.CLARIFICATION_REPLY)
    if normalized in _ACKNOWLEDGEMENTS:
        return RouteDecision(ConversationRoute.SOCIAL)
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
    """Return a small conversational response without invoking the model."""
    normalized = re.sub(r"[.!?]+$", "", text.strip().lower()).strip()
    if normalized in _GREETINGS:
        if user_name and user_name.strip():
            first_name = user_name.strip().split(maxsplit=1)[0]
            return f"Hi {first_name}! What would you like me to remind you about?"
        response = general_message("greeting")
        return response
    if normalized in _ACKNOWLEDGEMENTS:
        if user_name and user_name.strip():
            return "You're welcome."
        return general_message("thanks")
    return general_message("capabilities")


def unrelated_response(text: str, *, pending: PendingReminder | None) -> str:
    """Return a non-mutating response for text outside the reminder flow."""
    normalized = re.sub(r"[.!?]+$", "", text.strip().lower()).strip()
    if normalized in _ACKNOWLEDGEMENTS or normalized in _GREETINGS:
        return social_response(text)
    if pending is not None:
        return "Your unfinished reminder is still saved for when you're ready."
    return general_message("unrelated")
