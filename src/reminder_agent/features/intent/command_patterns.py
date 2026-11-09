"""Compiled patterns and small lexical rules for deterministic intent parsing."""

from __future__ import annotations

import re

COMMON_TASK_VERBS = {
    "buy",
    "call",
    "complete",
    "email",
    "finish",
    "pay",
    "renew",
    "review",
    "send",
    "submit",
    "text",
    "write",
}

ACKNOWLEDGEMENTS = {
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
CANCEL_FOLLOW_UPS = {"cancel", "cancel it", "never mind", "nevermind", "not now"}
TASK_CORRECTION = re.compile(
    r"^\s*(?:change|update|replace)\s+(?:the\s+)?(?:task|reminder|it)\s+"
    r"(?:to|with)\s+(?P<task>.+?)\s*$",
    re.IGNORECASE,
)
REQUIRES_INTERPRETATION = re.compile(
    r"\b(?:after|around|before|each|every|later|monthly|next|recurring|sometime|"
    r"this|tonight|weekly)\b",
    re.IGNORECASE,
)
COMPOUND_REQUEST = re.compile(
    r"\b(?:and|then)\s+(?:(?:please\s+)?remind\s+me\b|(?:to\s+)?"
    r"(?:buy|call|complete|email|finish|pay|renew|review|send|submit|text|write)\b)",
    re.IGNORECASE,
)

RELATIVE = re.compile(
    r"\bin\s+(?P<amount>\d+|[a-z]+)\s+(?P<unit>minutes?|hours?)\b",
    re.IGNORECASE,
)
TOMORROW = re.compile(
    r"\btomorrow\s+at\s+(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*"
    r"(?P<meridiem>a\.m\.|p\.m\.|am|pm)?\b",
    re.IGNORECASE,
)
CLOCK = re.compile(
    r"\bat\s+(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*"
    r"(?P<meridiem>a\.m\.|p\.m\.|am|pm)?\b",
    re.IGNORECASE,
)
FOLLOW_UP_CLOCK = re.compile(
    r"(?<!\w)(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*"
    r"(?P<meridiem>a\.m\.|p\.m\.|am|pm)?(?=\s|$|[,.])",
    re.IGNORECASE,
)
DAY = re.compile(
    r"\b(?P<day>today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
DAY_PART = re.compile(
    r"\b(?:in\s+the\s+)?(?:morning|afternoon|evening|night)\b",
    re.IGNORECASE,
)
RECURRENCE = re.compile(
    r"\bevery\s+(?P<days>(?:day|monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"(?:\s*(?:,|and)\s*(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))*)"
    r"\s+at\s+(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*"
    r"(?P<meridiem>a\.m\.|p\.m\.|am|pm)?"
    r"(?:\s*,?\s*(?:(?:to|for)\s+)?(?P<task>.+))?$",
    re.IGNORECASE,
)
RECURRENCE_AFTER_TASK = re.compile(
    r"(?P<task>.+?)\s+every\s+"
    r"(?P<days>(?:day|monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    r"(?:\s*(?:,|and)\s*(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))*)"
    r"\s+at\s+(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*"
    r"(?P<meridiem>a\.m\.|p\.m\.|am|pm)?$",
    re.IGNORECASE,
)
MONTHLY_RECURRENCE = re.compile(
    r"(?:(?P<before>.+?)\s+)?(?:every\s+month|monthly)\s+on(?:\s+the)?\s+"
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+at\s+(?P<hour>\d{1,2})"
    r"(?::(?P<minute>\d{2}))?\s*(?P<meridiem>a\.m\.|p\.m\.|am|pm)?"
    r"(?:\s+(?:(?:to|for)\s+)?(?P<after>.+))?$",
    re.IGNORECASE,
)
YEARLY_RECURRENCE = re.compile(
    r"(?:(?P<before>.+?)\s+)?(?:every\s+year|yearly)\s+on\s+"
    r"(?P<month>january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+at\s+"
    r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*"
    r"(?P<meridiem>a\.m\.|p\.m\.|am|pm)?(?:\s+(?:(?:to|for)\s+)?(?P<after>.+))?$",
    re.IGNORECASE,
)
MONTHLY_ORDINAL_RECURRENCE = re.compile(
    r"(?:(?P<before>.+?)\s+)?on\s+(?:the\s+)?(?P<ordinal>first|second|third|fourth|fifth|"
    r"\d{1,2}(?:st|nd|rd|th)?)\s*(?:day\s+)?of\s+(?:every|each)\s+month"
    r"(?:\s+at\s+(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*"
    r"(?P<meridiem>a\.m\.|p\.m\.|am|pm))?"
    r"(?:\s+(?:(?:to|for)\s+)?(?P<after>.+))?$",
    re.IGNORECASE,
)
ORDINAL_NUMBERS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
MULTIPLE_REQUEST_SEPARATOR = re.compile(
    r"\s+and\s+(?=(?:please\s+)?(?:remind\s+me|set\s+(?:a\s+)?reminder)\b)",
    re.IGNORECASE,
)
LIST_REQUEST = re.compile(
    r"^\s*(?:can\s+you\s+)?(?:what\s+(?:do\s+i\s+have|are\s+my\s+reminders?|"
    r"reminders?\s+do\s+i\s+have)|"
    r"what(?:'s|\s+is)\s+(?:my\s+)?reminders?(?:\s+(?:for|on|tomorrow|today))?|"
    r"(?:list|show)(?:\s+(?:my|me))?\s+(?:all\s+(?:the\s+)?)?(?:recurring\s+)?reminders?|"
    r"what(?:'s|\s+is)\s+(?:(?:on|in)\s+)?(?:my\s+)?schedule|"
    r"what\s+is\s+scheduled|anything\s+coming\s+up)\b",
    re.IGNORECASE,
)
SUMMARY_REQUEST = re.compile(
    r"^\s*(?:(?:how\s+many|what\s+is\s+the\s+number\s+of).*(?:pending|outstanding|unfinished|overdue|completed|done|finished)(?:.*(?:today|this\s+day))?"
    r"|(?:what|which|show|list).*(?:still\s+have\s+to\s+do|have\s+left|remaining).*(?:today|this\s+day)"
    r"|(?:what|which|show|list).*(?:today|this\s+day).*(?:still\s+have\s+to\s+do|have\s+left|remaining)"
    r"|(?:what|which|show|list).*(?:completed|done|finished).*(?:today|this\s+day))\s*[?!.]*\s*$",
    re.IGNORECASE,
)
LIST_RANGE = re.compile(
    r"\b(?:from|between)\s+(?P<start>today|tomorrow|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday|\d{4}-\d{2}-\d{2})\s+"
    r"(?:to|and)\s+(?P<end>today|tomorrow|monday|tuesday|wednesday|thursday|"
    r"friday|saturday|sunday|\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)


def contains_multiple_tasks(task: str) -> bool:
    """Detect a second action, while allowing people in one task."""

    parts = re.split(r"\s+(?:and|then)\s+", task, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return False
    second = re.sub(r"^(?:to|please)\s+", "", parts[1].strip(), flags=re.IGNORECASE)
    first_word = second.split(maxsplit=1)[0].lower() if second else ""
    return first_word in COMMON_TASK_VERBS


__all__ = [
    "ACKNOWLEDGEMENTS",
    "CANCEL_FOLLOW_UPS",
    "CLOCK",
    "COMMON_TASK_VERBS",
    "COMPOUND_REQUEST",
    "DAY",
    "DAY_PART",
    "FOLLOW_UP_CLOCK",
    "LIST_RANGE",
    "LIST_REQUEST",
    "MONTHLY_ORDINAL_RECURRENCE",
    "MONTHLY_RECURRENCE",
    "MONTHS",
    "MULTIPLE_REQUEST_SEPARATOR",
    "ORDINAL_NUMBERS",
    "RECURRENCE",
    "RECURRENCE_AFTER_TASK",
    "RELATIVE",
    "REQUIRES_INTERPRETATION",
    "SUMMARY_REQUEST",
    "TASK_CORRECTION",
    "TOMORROW",
    "YEARLY_RECURRENCE",
    "contains_multiple_tasks",
]
