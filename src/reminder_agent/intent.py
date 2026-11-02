from __future__ import annotations

import json
import logging
import inspect
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import StrEnum
from typing import Any, Literal, Protocol, Sequence
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from reminder_agent.observability import get_logger, log_event
from reminder_agent.recurrence import WeeklyRecurrence
from reminder_agent.request_policy import OUT_OF_SCOPE_RESPONSE

logger = get_logger()


class IntentRoute(StrEnum):
    NEW_REQUEST = "NEW_REQUEST"
    LIST_REQUEST = "LIST_REQUEST"
    RESCHEDULE_REQUEST = "RESCHEDULE_REQUEST"
    UPDATE_PENDING = "UPDATE_PENDING"
    CANCEL_PENDING = "CANCEL_PENDING"
    SOCIAL = "SOCIAL"
    UNRELATED = "UNRELATED"
    UNCERTAIN = "UNCERTAIN"


class IntentRouteSuggestion(BaseModel):
    """A non-authoritative route proposed by the intent-classification model."""

    model_config = ConfigDict(extra="forbid")

    operation: IntentRoute


class IntentRouter(Protocol):
    def classify(
        self,
        text: str,
        *,
        pending: "PendingReminder | None",
        recent_messages: Sequence[str],
    ) -> IntentRouteSuggestion: ...


class OpenAICompatibleIntentRouter:
    """Classify an ambiguous turn without extracting or modifying reminder data."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout_seconds: float = 10.0) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_seconds

    def classify(
        self,
        text: str,
        *,
        pending: "PendingReminder | None",
        recent_messages: Sequence[str],
    ) -> IntentRouteSuggestion:
        relevant_history = list(recent_messages)[-6:]
        log_event(
            logger,
            logging.INFO,
            "intent_classification_started",
            model=self._model,
            has_pending=pending is not None,
            recent_message_count=len(relevant_history),
        )
        prompt = (
            """
            Classify the latest message into one operation:

            - NEW_REQUEST: Separate reminder, even if one is pending.
            - LIST_REQUEST: Retrieve reminders.
            - RESCHEDULE_REQUEST: Move an existing reminder to a new time.
            - UPDATE_PENDING: Clarify or modify the pending reminder.
            - CANCEL_PENDING: Cancel the pending reminder.
            - SOCIAL: Casual chat.
            - UNRELATED: Non-reminder request.
            - UNCERTAIN: Ambiguous intent.

            Use pending state and conversation context. Distinguish new requests from pending updates by meaning, not keywords. If unclear, return UNCERTAIN.

            Do not extract fields or execute actions.
            Return JSON only: {"operation": "<OPERATION>"}.
            """
        )
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "latest_message": text,
                            "active_pending": pending.model_dump(mode="json") if pending else None,
                            "missing_fields": _missing_pending_fields(pending),
                            "last_clarification_question": _last_clarification_question(
                                recent_messages
                            ),
                            "recent_messages": relevant_history,
                        }
                    ),
                },
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        try:
            response = httpx.post(
                self._url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            data = json.loads(content) if isinstance(content, str) else content
            suggestion = IntentRouteSuggestion.model_validate(data)
            log_event(
                logger,
                logging.INFO,
                "intent_classification_completed",
                model=self._model,
                operation=suggestion.operation,
            )
            return suggestion
        except (httpx.HTTPError, KeyError, TypeError, ValueError, ValidationError, json.JSONDecodeError):
            log_event(
                logger,
                logging.WARNING,
                "intent_router_failed",
                model=self._model,
                reason="provider_or_schema_error",
            )
            return IntentRouteSuggestion(operation=IntentRoute.UNCERTAIN)


class CreateOneTimeReminderIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal["CREATE_ONE_TIME_REMINDER"] = "CREATE_ONE_TIME_REMINDER"
    task: str = Field(min_length=1, max_length=200)
    due_at: datetime
    timezone_name: str = Field(min_length=1, max_length=150)

    @field_validator("task")
    @classmethod
    def task_must_not_be_blank(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("task must not be blank")
        return value

    @field_validator("due_at")
    @classmethod
    def due_at_must_be_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("due_at must include timezone information")
        return value.astimezone(timezone.utc)

    @field_validator("timezone_name")
    @classmethod
    def timezone_must_be_valid(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except Exception as exc:
            raise ValueError("timezone_name must be a valid IANA timezone") from exc
        return value


class CreateRecurringReminderIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal["CREATE_RECURRING_REMINDER"] = "CREATE_RECURRING_REMINDER"
    task: str = Field(min_length=1, max_length=200)
    frequency: Literal["WEEKLY", "MONTHLY", "YEARLY"] = "WEEKLY"
    interval: int = Field(default=1, ge=1, le=60)
    weekdays: tuple[int, ...] = Field(default=(), max_length=7)
    day_of_month: int | None = Field(default=None, ge=1, le=31)
    month: int | None = Field(default=None, ge=1, le=12)
    hour: int = Field(ge=0, le=23)
    minute: int = Field(ge=0, le=59)
    start_local: datetime
    timezone_name: str = Field(min_length=1, max_length=150)

    @field_validator("task")
    @classmethod
    def task_must_not_be_blank(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("task must not be blank")
        return value

    @field_validator("weekdays")
    @classmethod
    def weekdays_must_be_sorted_and_unique(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(day < 0 or day > 6 for day in value):
            raise ValueError("weekdays must be between Monday (0) and Sunday (6)")
        normalized = tuple(sorted(set(value)))
        if normalized != value:
            raise ValueError("weekdays must be sorted and unique")
        return value

    @field_validator("start_local")
    @classmethod
    def start_local_must_be_naive(cls, value: datetime) -> datetime:
        if value.tzinfo is not None and value.utcoffset() is not None:
            raise ValueError("start_local must be timezone-naive")
        return value.replace(tzinfo=None)

    @field_validator("timezone_name")
    @classmethod
    def timezone_must_be_valid(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except Exception as exc:
            raise ValueError("timezone_name must be a valid IANA timezone") from exc
        return value

    @model_validator(mode="after")
    def recurrence_fields_must_match_frequency(self) -> "CreateRecurringReminderIntent":
        if self.frequency == "WEEKLY":
            if not self.weekdays or self.start_local.weekday() not in self.weekdays:
                raise ValueError("weekly start_local must match a selected weekday")
            if self.day_of_month is not None or self.month is not None:
                raise ValueError("weekly recurrence cannot include month fields")
        elif self.frequency == "MONTHLY":
            if self.day_of_month is None or self.start_local.day != self.day_of_month:
                raise ValueError("monthly start_local must match day_of_month")
            if self.weekdays or self.month is not None:
                raise ValueError("monthly recurrence has invalid fields")
        else:
            if (
                self.month is None
                or self.day_of_month is None
                or self.start_local.month != self.month
                or self.start_local.day != self.day_of_month
            ):
                raise ValueError("yearly start_local must match month and day_of_month")
            if self.weekdays:
                raise ValueError("yearly recurrence cannot include weekdays")
        return self


class ListRemindersIntent(BaseModel):
    """A read-only request for all reminders or an inclusive local-date range."""

    model_config = ConfigDict(extra="forbid")

    operation: Literal["LIST_REMINDERS"] = "LIST_REMINDERS"
    start_date: date | None = None
    end_date: date | None = None
    timezone_name: str = Field(min_length=1, max_length=150)

    @field_validator("timezone_name")
    @classmethod
    def timezone_must_be_valid(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except Exception as exc:
            raise ValueError("timezone_name must be a valid IANA timezone") from exc
        return value

    @model_validator(mode="after")
    def date_range_must_be_ordered(self) -> "ListRemindersIntent":
        if (self.start_date is None) != (self.end_date is None):
            raise ValueError("start_date and end_date must be supplied together")
        if self.start_date is None:
            return self
        assert self.end_date is not None
        if self.end_date < self.start_date:
            raise ValueError("end_date must not precede start_date")
        if (self.end_date - self.start_date).days > 366:
            raise ValueError("date range is too large")
        return self

class PendingReminder(BaseModel):
    """Validated, serializable partial data for a one-time or weekly reminder."""

    model_config = ConfigDict(extra="forbid")

    operation: Literal["CREATE_ONE_TIME_REMINDER"] = "CREATE_ONE_TIME_REMINDER"
    task: str | None = Field(default=None, max_length=200)
    date_local: date | None = None
    hour: int | None = Field(default=None, ge=0, le=23)
    minute: int = Field(default=0, ge=0, le=59)
    meridiem: Literal["am", "pm"] | None = None
    recurrence_weekdays: tuple[int, ...] | None = None
    recurrence_frequency: Literal["MONTHLY", "YEARLY"] | None = None
    recurrence_interval: int | None = Field(default=None, ge=1, le=60)
    recurrence_day_of_month: int | None = Field(default=None, ge=1, le=31)
    recurrence_month: int | None = Field(default=None, ge=1, le=12)
    timezone_name: str = Field(min_length=1, max_length=150)
    date_explicit: bool = False

    @field_validator("meridiem", mode="before")
    @classmethod
    def normalise_meridiem(cls, value: object) -> object:
        if isinstance(value, str):
            return value.lower()
        return value

    @field_validator("task")
    @classmethod
    def task_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = " ".join(value.split())
        return value or None

    @field_validator("recurrence_weekdays")
    @classmethod
    def recurrence_weekdays_must_be_sorted_and_unique(
        cls, value: tuple[int, ...] | None
    ) -> tuple[int, ...] | None:
        if value is None:
            return value
        if not value or any(day < 0 or day > 6 for day in value):
            raise ValueError("recurrence weekdays must be between Monday (0) and Sunday (6)")
        if tuple(sorted(set(value))) != value:
            raise ValueError("recurrence weekdays must be sorted and unique")
        return value

    @field_validator("timezone_name")
    @classmethod
    def timezone_must_be_valid(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except Exception as exc:
            raise ValueError("timezone_name must be a valid IANA timezone") from exc
        return value


class PendingReminderFieldUpdate(BaseModel):
    """Only the fields a clarification reply explicitly supplies.

    This is intentionally not a partial ``PendingReminder``.  In particular it
    contains neither an operation nor a timezone, so a model cannot turn a
    follow-up into a complete, authoritative reminder.
    """

    model_config = ConfigDict(extra="forbid")

    task: str | None = Field(default=None, max_length=200)
    date_local: date | None = None
    hour: int | None = Field(default=None, ge=0, le=23)
    minute: int | None = Field(default=None, ge=0, le=59)
    meridiem: Literal["am", "pm"] | None = None
    date_explicit: bool | None = None
    recurrence_frequency: Literal["MONTHLY", "YEARLY"] | None = None
    recurrence_interval: int | None = Field(default=None, ge=1, le=60)
    recurrence_day_of_month: int | None = Field(default=None, ge=1, le=31)
    recurrence_month: int | None = Field(default=None, ge=1, le=12)
    certainty: Literal["CLEAR"] | None = None
    evidence: dict[
        Literal[
            "task", "date_local", "hour", "minute", "meridiem", "date_explicit",
            "recurrence_frequency", "recurrence_interval", "recurrence_day_of_month", "recurrence_month",
        ], str
    ] = Field(default_factory=dict, exclude=True)

    @field_validator("certainty", mode="before")
    @classmethod
    def normalise_certainty(cls, value: object) -> object:
        if isinstance(value, str):
            return value.upper()
        return value

    @field_validator("meridiem", mode="before")
    @classmethod
    def normalise_meridiem(cls, value: object) -> object:
        if isinstance(value, str):
            return value.lower()
        return value

    @field_validator("task")
    @classmethod
    def task_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = " ".join(value.split())
        return value or None

    def supplied_fields(self) -> set[str]:
        """Usable provider values, including ``false`` but never null clears."""
        return {
            field
            for field in self.model_fields_set
            if field not in {"certainty", "evidence"} and getattr(self, field) is not None
        }


class NeedsClarification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal["CREATE_ONE_TIME_REMINDER"] = "CREATE_ONE_TIME_REMINDER"
    pending: PendingReminder
    question: str = Field(min_length=1, max_length=500)


def _missing_pending_fields(pending: PendingReminder | None) -> list[str]:
    if pending is None:
        return []
    missing: list[str] = []
    if pending.task is None:
        missing.append("task")
    if pending.date_local is None:
        missing.append("date")
    if pending.hour is None:
        missing.append("time")
    elif pending.hour <= 12 and pending.meridiem is None:
        missing.append("meridiem")
    return missing


def _last_clarification_question(recent_messages: Sequence[str]) -> str | None:
    for message in reversed(recent_messages):
        if message.startswith("assistant: ") and "?" in message:
            return message.removeprefix("assistant: ")
    return None


@dataclass(frozen=True)
class Interpretation:
    intent: (
        CreateOneTimeReminderIntent | CreateRecurringReminderIntent | ListRemindersIntent | None
    ) = None
    intents: tuple[
        CreateOneTimeReminderIntent | CreateRecurringReminderIntent | ListRemindersIntent, ...
    ] = ()
    clarification: NeedsClarification | None = None
    field_updates: PendingReminderFieldUpdate | None = None
    reschedule_request: Any | None = None
    response_text: str | None = None
    fallback_allowed: bool = True
    clear_context: bool = False

    @property
    def resolved_intents(
        self,
    ) -> tuple[
        CreateOneTimeReminderIntent | CreateRecurringReminderIntent | ListRemindersIntent, ...
    ]:
        """Return one or more validated commands represented by this result."""
        if self.intents:
            return self.intents
        return (self.intent,) if self.intent is not None else ()


def merge_pending_reminder_updates(
    pending: PendingReminder, updates: PendingReminderFieldUpdate
) -> PendingReminder:
    """Apply explicitly supplied follow-up fields to trusted pending state.

    The pending record belongs to the current locked conversation session.  A
    provider is therefore never allowed to replace omitted fields such as the
    task, date, or timezone.
    """
    values = pending.model_dump(mode="python")
    supplied = updates.supplied_fields()
    for field in supplied:
        values[field] = getattr(updates, field)
    if "date_local" in supplied and "date_explicit" not in supplied:
        values["date_explicit"] = updates.date_local is not None
    return PendingReminder.model_validate(values)


class IntentInterpreter(Protocol):
    def interpret(
        self,
        text: str,
        *,
        selected_route: IntentRoute | None = None,
        timezone_name: str,
        now: datetime,
        context: PendingReminder | None = None,
        recent_messages: Sequence[str] = (),
    ) -> Interpretation: ...


_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
}


def _normalise_now(now: datetime) -> datetime:
    if now.tzinfo is None or now.utcoffset() is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def _task_from_remainder(remainder: str) -> str:
    task = remainder.strip(" .,:;")
    task = re.sub(r"^(?:(?:on|to|for|that|about)\s+)+", "", task, flags=re.IGNORECASE)
    return " ".join(task.split())


def _number(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return _NUMBER_WORDS.get(value.lower())


def _ordinal_suffix(value: int) -> str:
    if 10 < value % 100 < 14:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")


def _next_recurring_start_local(
    *,
    frequency: str,
    interval: int,
    weekdays: tuple[int, ...],
    day_of_month: int | None,
    month: int | None,
    hour: int,
    minute: int,
    timezone_name: str,
    now: datetime,
) -> datetime:
    """Derive the next local occurrence for a fully specified recurrence.

    The extraction model supplies the recurrence rule; this application helper
    owns the clock calculation so a model never has to invent ``start_local``.
    """
    local_now = _normalise_now(now).astimezone(ZoneInfo(timezone_name)).replace(tzinfo=None)
    if frequency == "WEEKLY":
        for offset in range(8):
            candidate_date = local_now.date() + timedelta(days=offset)
            if candidate_date.weekday() not in weekdays:
                continue
            candidate = datetime.combine(candidate_date, time(hour, minute))
            if candidate > local_now:
                return candidate
        raise ValueError("could not find weekly recurrence start")

    if day_of_month is None:
        raise ValueError("calendar recurrence requires day_of_month")
    candidate_year = local_now.year
    candidate_month = month or local_now.month
    while True:
        try:
            candidate_date = date(candidate_year, candidate_month, day_of_month)
        except ValueError:
            candidate_date = None
        if candidate_date is not None:
            candidate = datetime.combine(candidate_date, time(hour, minute))
            if candidate > local_now:
                return candidate
        if frequency == "YEARLY":
            candidate_year += interval
        else:
            candidate_month += interval
            candidate_year += (candidate_month - 1) // 12
            candidate_month = (candidate_month - 1) % 12 + 1


def _normalise_meridiem(value: str | None) -> str | None:
    return value.lower().replace(".", "") if value else None


def _valid_clock(hour: int | None, minute: int, meridiem: str | None) -> bool:
    if hour is None or not 0 <= minute <= 59:
        return False
    if meridiem is not None:
        return 1 <= hour <= 12
    return 0 <= hour <= 23


def _clock_to_24(hour: int, meridiem: str) -> int:
    if meridiem == "pm" and hour != 12:
        return hour + 12
    if meridiem == "am" and hour == 12:
        return 0
    return hour


def _clock_label(hour: int, minute: int) -> str:
    return f"{hour:g}" if minute == 0 else f"{hour}:{minute:02d}"


def _date_label(value: date, local_today: date) -> str:
    if value == local_today:
        return "today"
    if value == local_today + timedelta(days=1):
        return "tomorrow"
    return value.strftime("%A")


def resolve_pending_due_at(pending: PendingReminder, *, now: datetime) -> datetime:
    """Resolve a complete pending reminder in its user's local timezone."""
    if pending.date_local is None or pending.hour is None:
        raise ValueError("pending reminder is missing date or time")
    hour = pending.hour
    if pending.meridiem:
        hour = _clock_to_24(hour, pending.meridiem)
    zone = ZoneInfo(pending.timezone_name)
    return datetime.combine(
        pending.date_local,
        time(hour, pending.minute),
        tzinfo=zone,
    )


def _date_from_day(match: re.Match[str] | None, local_now: datetime) -> date:
    assert match is not None
    value = match.group("day").lower()
    if value == "today":
        return local_now.date()
    if value == "tomorrow":
        return local_now.date() + timedelta(days=1)
    target = RuleBasedIntentInterpreter._weekdays[value]
    days_ahead = (target - local_now.weekday()) % 7
    return local_now.date() + timedelta(days=days_ahead)


_COMMON_TASK_VERBS = {
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


def _contains_multiple_tasks(task: str) -> bool:
    """Detect a second action, while allowing people in one task."""
    parts = re.split(r"\s+(?:and|then)\s+", task, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        return False
    second = re.sub(r"^(?:to|please)\s+", "", parts[1].strip(), flags=re.IGNORECASE)
    first_word = second.split(maxsplit=1)[0].lower() if second else ""
    return first_word in _COMMON_TASK_VERBS


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
_CANCEL_FOLLOW_UPS = {"cancel", "cancel it", "never mind", "nevermind", "not now"}
_TASK_CORRECTION = re.compile(
    r"^\s*(?:change|update|replace)\s+(?:the\s+)?(?:task|reminder|it)\s+"
    r"(?:to|with)\s+(?P<task>.+?)\s*$",
    re.IGNORECASE,
)
_REQUIRES_INTERPRETATION = re.compile(
    r"\b(?:after|around|before|each|every|later|monthly|next|recurring|sometime|"
    r"this|tonight|weekly)\b",
    re.IGNORECASE,
)
_COMPOUND_REQUEST = re.compile(
    r"\b(?:and|then)\s+(?:(?:please\s+)?remind\s+me\b|(?:to\s+)?"
    r"(?:buy|call|complete|email|finish|pay|renew|review|send|submit|text|write)\b)",
    re.IGNORECASE,
)


def _same_task(left: str, right: str) -> bool:
    return " ".join(left.lower().split()) == " ".join(right.lower().split())


def _evidence_is_grounded(evidence: str | None, text: str) -> bool:
    """Verify an evidence fragment with bounded, typo-tolerant matching."""
    if not isinstance(evidence, str):
        return False
    normalized_evidence = " ".join(evidence.lower().split())
    normalized_text = " ".join(text.lower().split())
    if not normalized_evidence:
        return False
    if normalized_evidence in normalized_text:
        return True

    evidence_tokens = re.findall(r"[a-z0-9]+", normalized_evidence)
    message_tokens = re.findall(r"[a-z0-9]+", normalized_text)
    if not evidence_tokens or len(evidence_tokens) > len(message_tokens):
        return False

    temporal_words = {
        "today",
        "tomorrow",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    }

    def close_enough(left: str, right: str) -> bool:
        if left == right:
            return True
        if left in temporal_words and right in temporal_words:
            return False
        # Permit only small insertion/deletion/substitution errors.  This is
        # intentionally conservative so unrelated words cannot become evidence.
        limit = max(1, (max(len(left), len(right)) + 3) // 4)
        previous = list(range(len(right) + 1))
        for left_index, left_char in enumerate(left, start=1):
            current = [left_index]
            for right_index, right_char in enumerate(right, start=1):
                current.append(
                    min(
                        previous[right_index] + 1,
                        current[right_index - 1] + 1,
                        previous[right_index - 1] + (left_char != right_char),
                    )
                )
            previous = current
        return previous[-1] <= limit

    for start in range(len(message_tokens) - len(evidence_tokens) + 1):
        candidate = message_tokens[start : start + len(evidence_tokens)]
        if all(close_enough(left, right) for left, right in zip(evidence_tokens, candidate)):
            return True
        # The same small set of supported words may arrive in a different
        # order (for example, "at 8 PM tomorrow").  Permit this only when
        # each token still has a bounded match in the same local window.
        remaining = list(candidate)
        unordered_match = True
        for token in evidence_tokens:
            match_index = next(
                (
                    index
                    for index, candidate_token in enumerate(remaining)
                    if close_enough(token, candidate_token)
                ),
                None,
            )
            if match_index is None:
                unordered_match = False
                break
            remaining.pop(match_index)
        if unordered_match:
            return True
    return False


def restrict_to_explicit_field_updates(
    text: str,
    proposed: PendingReminderFieldUpdate,
    *,
    context: PendingReminder,
    now: datetime,
    allow_task_replacement: bool = False,
    allow_semantic_field_updates: bool = False,
) -> PendingReminderFieldUpdate:
    """Keep only provider updates demonstrably present in the user's reply.

    A model can help recognize a follow-up, but it is not trusted to alter an
    existing reminder field.  Time and date values must agree with deterministic
    text extraction; a task replacement must use an explicit correction phrase.
    """
    raw = text.strip()
    clock = RuleBasedIntentInterpreter._clock.search(raw)
    if clock is None:
        clock = RuleBasedIntentInterpreter._follow_up_clock.search(raw)
    day = RuleBasedIntentInterpreter._day.search(raw)
    permitted: dict[str, Any] = {}

    if clock is not None:
        expected_hour = int(clock.group("hour"))
        expected_minute = int(clock.group("minute") or 0)
        expected_meridiem = _normalise_meridiem(clock.group("meridiem"))
        if proposed.hour == expected_hour:
            permitted["hour"] = proposed.hour
        if proposed.minute == expected_minute:
            permitted["minute"] = proposed.minute
        if proposed.meridiem == expected_meridiem and expected_meridiem is not None:
            permitted["meridiem"] = proposed.meridiem
    elif proposed.meridiem is not None and raw.lower().strip(" .,!?") in {"am", "pm"}:
        # A standalone AM/PM answer is a valid response to an outstanding
        # ambiguous-hour clarification; it has no hour token to match.
        expected_meridiem = _normalise_meridiem(raw)
        if proposed.meridiem == expected_meridiem:
            permitted["meridiem"] = proposed.meridiem

    if day is not None:
        local_now = _normalise_now(now).astimezone(ZoneInfo(context.timezone_name))
        expected_date = _date_from_day(day, local_now)
        if proposed.date_local == expected_date:
            permitted["date_local"] = proposed.date_local
            # The backend derives this flag from the verified date, rather than
            # accepting a model-provided boolean independently.
            permitted["date_explicit"] = True

    # Exact deterministic extraction remains the preferred verifier.  When
    # the intent classifier has selected this locked draft as an update, OSS
    # extraction may also supply a structurally valid field whose source text
    # contains a clear typo (for example, "tomorow").  The model is instructed
    # to return NEEDS_CLARIFICATION instead of guessing when alternatives are
    # plausible; the backend still owns schema, target, and schedule checks.
    semantic_updates_are_grounded = (
        allow_semantic_field_updates and proposed.certainty == "CLEAR"
    )
    if semantic_updates_are_grounded:
        if (
            proposed.date_local is not None
            and "date_local" not in permitted
            and _evidence_is_grounded(proposed.evidence.get("date_local"), raw)
        ):
            permitted["date_local"] = proposed.date_local
            permitted["date_explicit"] = True
        if (
            proposed.hour is not None
            and "hour" not in permitted
            and _evidence_is_grounded(proposed.evidence.get("hour"), raw)
        ):
            if proposed.meridiem is None or 1 <= proposed.hour <= 12:
                permitted["hour"] = proposed.hour
        if (
            proposed.minute is not None
            and "minute" not in permitted
            and _evidence_is_grounded(proposed.evidence.get("minute"), raw)
        ):
            permitted["minute"] = proposed.minute
        if (
            proposed.meridiem is not None
            and "meridiem" not in permitted
            and _evidence_is_grounded(proposed.evidence.get("meridiem"), raw)
        ):
            permitted["meridiem"] = proposed.meridiem
        for field in (
            "recurrence_frequency",
            "recurrence_interval",
            "recurrence_day_of_month",
            "recurrence_month",
        ):
            value = getattr(proposed, field)
            if value is not None and _evidence_is_grounded(
                proposed.evidence.get(field), raw
            ):
                permitted[field] = value

    remainder = raw
    if clock is not None:
        remainder = remainder.replace(clock.group(0), "", 1)
    if day is not None:
        remainder = remainder.replace(day.group(0), "", 1)
    supplied_task = _task_from_remainder(remainder)
    # Detect the correction instruction in the original message.  The task
    # captured by the correction expression is the replacement value and
    # should be validated separately; it does not contain correction language.
    correction = _TASK_CORRECTION.match(remainder)
    if proposed.task:
        if context.task is None and supplied_task and _same_task(proposed.task, supplied_task):
            permitted["task"] = proposed.task
        elif correction is not None:
            corrected_task = _task_from_remainder(correction.group("task"))
            if corrected_task and _same_task(proposed.task, corrected_task):
                permitted["task"] = proposed.task
        # The backend has already selected the user's locked pending draft and
        # the classifier has identified this turn as UPDATE_PENDING.  In that
        # case a nonblank, schema-validated task patch is an explicit
        # replacement; do not require a particular natural-language phrase.
        elif allow_task_replacement and semantic_updates_are_grounded and _evidence_is_grounded(
            proposed.evidence.get("task"), raw
        ):
            permitted["task"] = proposed.task

    return PendingReminderFieldUpdate.model_validate(permitted)


class RuleBasedIntentInterpreter:
    """Conservative parser that refuses to guess incomplete date/time requests."""

    _prefix = re.compile(
        r"^\s*(?:please\s+)?(?:remind\s+me|set\s+(?:a\s+)?reminder)(?:\s+|$)",
        re.IGNORECASE,
    )
    _relative = re.compile(
        r"\bin\s+(?P<amount>\d+|[a-z]+)\s+(?P<unit>minutes?|hours?)\b",
        re.IGNORECASE,
    )
    _tomorrow = re.compile(
        r"\btomorrow\s+at\s+(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*"
        r"(?P<meridiem>a\.m\.|p\.m\.|am|pm)?\b",
        re.IGNORECASE,
    )
    _clock = re.compile(
        r"\bat\s+(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*"
        r"(?P<meridiem>a\.m\.|p\.m\.|am|pm)?\b",
        re.IGNORECASE,
    )
    _follow_up_clock = re.compile(
        r"(?<!\w)(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*"
        r"(?P<meridiem>a\.m\.|p\.m\.|am|pm)?(?=\s|$|[,.])",
        re.IGNORECASE,
    )
    _day = re.compile(
        r"\b(?P<day>today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        re.IGNORECASE,
    )
    _day_part = re.compile(
        r"\b(?:in\s+the\s+)?(?:morning|afternoon|evening|night)\b",
        re.IGNORECASE,
    )
    _recurrence = re.compile(
        r"\bevery\s+(?P<days>(?:day|monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
        r"(?:\s*(?:,|and)\s*(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))*)"
        r"\s+at\s+(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*"
        r"(?P<meridiem>a\.m\.|p\.m\.|am|pm)?"
        r"(?:\s+(?:(?:to|for)\s+)?(?P<task>.+))?$",
        re.IGNORECASE,
    )
    _recurrence_after_task = re.compile(
        r"(?P<task>.+?)\s+every\s+"
        r"(?P<days>(?:day|monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
        r"(?:\s*(?:,|and)\s*(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))*)"
        r"\s+at\s+(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*"
        r"(?P<meridiem>a\.m\.|p\.m\.|am|pm)?$",
        re.IGNORECASE,
    )
    _monthly_recurrence = re.compile(
        r"(?:(?P<before>.+?)\s+)?(?:every\s+month|monthly)\s+on(?:\s+the)?\s+"
        r"(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+at\s+(?P<hour>\d{1,2})"
        r"(?::(?P<minute>\d{2}))?\s*(?P<meridiem>a\.m\.|p\.m\.|am|pm)?"
        r"(?:\s+(?:(?:to|for)\s+)?(?P<after>.+))?$",
        re.IGNORECASE,
    )
    _yearly_recurrence = re.compile(
        r"(?:(?P<before>.+?)\s+)?(?:every\s+year|yearly)\s+on\s+"
        r"(?P<month>january|february|march|april|may|june|july|august|september|"
        r"october|november|december)\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?\s+at\s+"
        r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*"
        r"(?P<meridiem>a\.m\.|p\.m\.|am|pm)?(?:\s+(?:(?:to|for)\s+)?(?P<after>.+))?$",
        re.IGNORECASE,
    )
    _monthly_ordinal_recurrence = re.compile(
        r"(?:(?P<before>.+?)\s+)?on\s+(?:the\s+)?(?P<ordinal>first|second|third|fourth|fifth|"
        r"\d{1,2}(?:st|nd|rd|th)?)\s*(?:day\s+)?of\s+(?:every|each)\s+month"
        r"(?:\s+at\s+(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*"
        r"(?P<meridiem>a\.m\.|p\.m\.|am|pm))?"
        r"(?:\s+(?:(?:to|for)\s+)?(?P<after>.+))?$",
        re.IGNORECASE,
    )
    _ordinal_numbers = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
    _months = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    _multiple_request_separator = re.compile(
        r"\s+and\s+(?=(?:please\s+)?(?:remind\s+me|set\s+(?:a\s+)?reminder)\b)",
        re.IGNORECASE,
    )
    _weekdays = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }
    _list_request = re.compile(
        r"^\s*(?:can\s+you\s+)?(?:what\s+(?:do\s+i\s+have|are\s+my\s+reminders?)|"
        r"(?:list|show)(?:\s+(?:my|me))?\s+(?:all\s+(?:the\s+)?)?(?:recurring\s+)?reminders?|"
        r"what(?:'s|\s+is)\s+(?:(?:on|in)\s+)?(?:my\s+)?schedule|"
        r"what\s+is\s+scheduled|anything\s+coming\s+up)\b",
        re.IGNORECASE,
    )
    _list_range = re.compile(
        r"\b(?:from|between)\s+(?P<start>today|tomorrow|monday|tuesday|wednesday|"
        r"thursday|friday|saturday|sunday|\d{4}-\d{2}-\d{2})\s+"
        r"(?:to|and)\s+(?P<end>today|tomorrow|monday|tuesday|wednesday|thursday|"
        r"friday|saturday|sunday|\d{4}-\d{2}-\d{2})\b",
        re.IGNORECASE,
    )

    def interpret(
        self,
        text: str,
        *,
        selected_route: IntentRoute | None = None,
        timezone_name: str,
        now: datetime,
        context: PendingReminder | None = None,
    ) -> Interpretation:
        raw = text.strip()
        route = selected_route or (
            IntentRoute.UPDATE_PENDING if context is not None else IntentRoute.NEW_REQUEST
        )
        if route is IntentRoute.UNCERTAIN:
            return Interpretation(
                response_text="I’m not sure what you want to do. Please clarify.",
                fallback_allowed=False,
            )
        if route is IntentRoute.UPDATE_PENDING and context is None:
            return Interpretation(
                response_text="There is no unfinished reminder to update.",
                fallback_allowed=False,
            )
        if route is not IntentRoute.UPDATE_PENDING:
            context = None
        # An explicit new reminder command supersedes an unresolved clarification.
        # This gives users a safe way to answer a multi-target question with one
        # complete request instead of merging unrelated text into the old action.
        if context is not None and self._prefix.match(raw):
            context = None
        if context is not None:
            return self._interpret_follow_up(text, context=context, now=now)

        listing = self._interpret_listing(raw, timezone_name=timezone_name, now=now)
        if listing is not None:
            return listing

        # Split independently timed clauses such as "tomorrow at 9 AM to pay
        # rent and at 6 PM to call Sara".  The date before the first clock is
        # shared by every clause; each clause owns the clock and task that
        # follow it.  If any clause is incomplete, keep the result eligible for
        # the structured provider fallback instead of merging task text.
        multiple_times = self._interpret_multiple_time_clauses(
            raw, timezone_name=timezone_name, now=now
        )
        if multiple_times is not None:
            return multiple_times

        # A message may contain several explicitly separated, complete commands.
        # Parse each command independently instead of treating the whole message
        # as an ambiguous target.  We deliberately require the repeated reminder
        # prefix; a plain "and" remains conservative because it may be part of a
        # single task (for example, "call Alice and Bob").
        commands = self._multiple_request_separator.split(raw)
        if len(commands) > 1:
            parsed = [
                self.interpret(
                    command,
                    selected_route=route,
                    timezone_name=timezone_name,
                    now=now,
                    context=None,
                )
                for command in commands
            ]
            if all(item.intent is not None for item in parsed):
                return Interpretation(
                    intents=tuple(item.intent for item in parsed if item.intent is not None)
                )
            clarification = next(
                (item.clarification for item in parsed if item.clarification is not None),
                None,
            )
            if clarification is not None:
                return Interpretation(
                    clarification=clarification,
                    fallback_allowed=False,
                )
            return Interpretation(
                response_text=(
                    "I found multiple reminder requests. Please include a complete "
                    "future date and time for each one."
                ),
                fallback_allowed=False,
            )

        log_event(
            logger,
            logging.INFO,
            "reminder_intent_rules_started",
            input_length=len(raw),
            timezone=timezone_name,
        )
        prefix = self._prefix.match(raw)
        if prefix is None:
            calendar = self._interpret_calendar_recurrence(raw, timezone_name=timezone_name, now=now)
            if calendar is not None:
                return calendar
            recurrence = self._recurrence.search(raw)
            if recurrence is None or recurrence.group("task") is None:
                recurrence = self._recurrence_after_task.search(raw) or recurrence
            if recurrence is not None:
                return self._build_recurring(
                    recurrence,
                    timezone_name=timezone_name,
                    now=_normalise_now(now),
                )
            # Keep bare clock answers and social acknowledgements deterministic,
            # but allow the provider to interpret complex natural language even
            # when it does not contain the literal word "remind".
            simple_time = self._follow_up_clock.fullmatch(raw.strip(" .,!"))
            simple_ack = raw.lower().strip(" .,!?") in (
                _ACKNOWLEDGEMENTS | _CANCEL_FOLLOW_UPS
            )
            return Interpretation(
                response_text=(
                    "Please tell me the reminder task and a future time."
                ),
                fallback_allowed=not (simple_time or simple_ack),
            )

        zone = ZoneInfo(timezone_name)
        reference = _normalise_now(now)
        without_prefix = raw[prefix.end() :]

        calendar = self._interpret_calendar_recurrence(
            without_prefix, timezone_name=timezone_name, now=reference
        )
        if calendar is not None:
            return calendar

        recurrence = self._recurrence.search(without_prefix)
        if recurrence is None or recurrence.group("task") is None:
            recurrence = self._recurrence_after_task.search(without_prefix) or recurrence
        if recurrence is not None:
            return self._build_recurring(
                recurrence,
                timezone_name=timezone_name,
                now=reference,
            )

        # Relative phrases and conversational qualifiers have
        # too much meaning for a small regex parser.  Do not extract a partial
        # intent that could be mistaken for a complete command; let the model
        # return a typed proposal instead.
        if _REQUIRES_INTERPRETATION.search(raw):
            return Interpretation(
                response_text="I need a little more detail to schedule that reminder.",
                fallback_allowed=True,
            )

        relative = self._relative.search(without_prefix)
        if relative is not None:
            amount = _number(relative.group("amount"))
            if amount is None or amount <= 0:
                return Interpretation(
                    response_text="Please provide a valid future reminder time.",
                    fallback_allowed=False,
                )
            unit = relative.group("unit").lower()
            delta = timedelta(minutes=amount if unit.startswith("minute") else amount * 60)
            before_time = without_prefix[: relative.start()]
            after_time = without_prefix[relative.end() :]
            task = _task_from_remainder(
                f"{before_time} {after_time}" if before_time.strip() and after_time.strip()
                else before_time if before_time.strip() else after_time
            )
            return self._build(task, reference + delta, timezone_name)

        tomorrow = self._tomorrow.search(without_prefix)
        if tomorrow is not None:
            if len(list(self._clock.finditer(without_prefix))) > 1:
                return Interpretation(
                    response_text="I found multiple reminder times. Please provide a complete task and time for each.",
                    fallback_allowed=True,
                )
            hour = int(tomorrow.group("hour"))
            minute = int(tomorrow.group("minute") or 0)
            meridiem = _normalise_meridiem(tomorrow.group("meridiem"))
            if not _valid_clock(hour, minute, meridiem):
                return self._invalid_time()
            local_date = reference.astimezone(zone).date() + timedelta(days=1)
            before_time = without_prefix[: tomorrow.start()]
            after_time = without_prefix[tomorrow.end() :]
            task = _task_from_remainder(
                f"{before_time} {after_time}" if before_time.strip() and after_time.strip()
                else before_time if before_time.strip() else after_time
            )
            if meridiem is None:
                return self._pending_result(
                    PendingReminder(
                        task=task or None,
                        date_local=local_date,
                        hour=hour,
                        minute=minute,
                        meridiem=None,
                        timezone_name=timezone_name,
                        date_explicit=True,
                    ),
                    now=now,
                )
            due_local = datetime.combine(
                local_date,
                time(_clock_to_24(hour, meridiem), minute),
                tzinfo=zone,
            )
            return self._build(task, due_local.astimezone(timezone.utc), timezone_name)

        clock = self._clock.search(without_prefix)
        day = self._day.search(without_prefix)
        day_part = self._day_part.search(without_prefix)
        if clock is not None or day is not None or day_part is not None:
            if len(list(self._clock.finditer(without_prefix))) > 1:
                return Interpretation(
                    response_text="I found multiple reminder times. Please provide a complete task and time for each.",
                    fallback_allowed=True,
                )
            local_now = reference.astimezone(zone)
            date_local = (
                _date_from_day(day, local_now)
                if day
                else local_now.date()
                if clock is not None
                else None
            )
            date_explicit = day is not None
            hour = int(clock.group("hour")) if clock else None
            minute = int(clock.group("minute") or 0) if clock else 0
            meridiem = _normalise_meridiem(clock.group("meridiem")) if clock else None
            if clock is not None and not _valid_clock(hour, minute, meridiem):
                return self._invalid_time()
            task_source = without_prefix
            if day is not None:
                task_source = task_source.replace(day.group(0), "", 1)
            if day_part is not None:
                task_source = task_source.replace(day_part.group(0), "", 1)
            if clock is not None:
                task_source = task_source.replace(clock.group(0), "", 1)
            task = _task_from_remainder(task_source)
            pending = PendingReminder(
                task=task or None,
                date_local=date_local,
                hour=hour,
                minute=minute,
                meridiem=meridiem,
                timezone_name=timezone_name,
                date_explicit=date_explicit,
            )
            return self._pending_result(pending, now=now)

        # A command with an ordinary task but no schedule is still useful
        # deterministic input.  Store it as pending rather than asking a model
        # to reconstruct the same partial reminder.
        task = _task_from_remainder(without_prefix)
        if _contains_multiple_tasks(task):
            return self._multiple_target_clarification(timezone_name)
        return self._pending_result(
            PendingReminder(task=task or None, timezone_name=timezone_name),
            now=now,
            fallback_allowed=True,
        )

    def _interpret_calendar_recurrence(
        self, text: str, *, timezone_name: str, now: datetime
    ) -> Interpretation | None:
        ordinal = self._monthly_ordinal_recurrence.search(text)
        if ordinal is not None:
            raw_ordinal = ordinal.group("ordinal").lower()
            day = self._ordinal_numbers.get(raw_ordinal)
            if day is None:
                day = int(re.sub(r"\D", "", raw_ordinal))
            return self._build_calendar_interpretation(
                ordinal,
                frequency="MONTHLY",
                day=day,
                month=None,
                timezone_name=timezone_name,
                now=now,
            )
        match = self._monthly_recurrence.search(text)
        frequency = "MONTHLY"
        month = None
        if match is None:
            match = self._yearly_recurrence.search(text)
            frequency = "YEARLY"
        if match is None:
            return None
        return self._build_calendar_interpretation(
            match,
            frequency=frequency,
            day=int(match.group("day")),
            month=(self._months[match.group("month").lower()] if frequency == "YEARLY" else None),
            timezone_name=timezone_name,
            now=now,
        )

    def _build_calendar_interpretation(
        self,
        match: re.Match[str],
        *,
        frequency: str,
        day: int,
        month: int | None,
        timezone_name: str,
        now: datetime,
    ) -> Interpretation:
        hour_value = match.group("hour")
        hour = int(hour_value) if hour_value is not None else None
        minute = int(match.group("minute") or 0)
        meridiem = _normalise_meridiem(match.group("meridiem"))
        task = _task_from_remainder(
            f"{match.group('before') or ''} {match.group('after') or ''}"
        )
        if hour is None:
            local_now = now.astimezone(ZoneInfo(timezone_name))
            return Interpretation(
                clarification=NeedsClarification(
                    pending=PendingReminder(
                        task=task or None,
                        date_local=local_now.date(),
                        hour=None,
                        recurrence_frequency=frequency,
                        recurrence_day_of_month=day,
                        recurrence_month=month,
                        timezone_name=timezone_name,
                        date_explicit=True,
                    ),
                    question=(
                        f"What time should I use on the {day}{_ordinal_suffix(day)} day "
                        f"of each {'month' if frequency == 'MONTHLY' else 'year'}?"
                    ),
                ),
                fallback_allowed=False,
            )
        if meridiem is None and hour <= 12:
            return Interpretation(
                response_text=f"Do you mean {hour:g} AM or {hour:g} PM for this {frequency.lower()} reminder?",
                fallback_allowed=False,
            )
        if meridiem is not None:
            hour = _clock_to_24(hour, meridiem)
        if frequency == "YEARLY":
            assert month is not None
        try:
            candidate_year = now.astimezone(ZoneInfo(timezone_name)).year
            local_now = now.astimezone(ZoneInfo(timezone_name))
            candidate_month = month or local_now.month
            while True:
                try:
                    candidate_date = date(candidate_year, candidate_month, day)
                except ValueError:
                    candidate_year += 1 if frequency == "YEARLY" else 0
                    if frequency == "MONTHLY":
                        candidate_month += 1
                        if candidate_month > 12:
                            candidate_month = 1
                            candidate_year += 1
                    continue
                candidate_local = datetime.combine(candidate_date, time(hour, minute))
                if candidate_local > local_now.replace(tzinfo=None):
                    break
                if frequency == "YEARLY":
                    candidate_year += 1
                else:
                    candidate_month += 1
                    if candidate_month > 12:
                        candidate_month = 1
                        candidate_year += 1
            if not task:
                return Interpretation(
                    response_text="What should I remind you about on that calendar date?",
                    fallback_allowed=False,
                )
            return Interpretation(
                intent=CreateRecurringReminderIntent(
                    task=task,
                    frequency=frequency,
                    interval=1,
                    weekdays=(),
                    day_of_month=day,
                    month=month,
                    hour=hour,
                    minute=minute,
                    start_local=candidate_local,
                    timezone_name=timezone_name,
                )
            )
        except (TypeError, ValueError, ValidationError):
            return Interpretation(
                response_text="Please provide a valid monthly or yearly reminder date.",
                fallback_allowed=False,
            )

    def _interpret_listing(
        self, raw: str, *, timezone_name: str, now: datetime
    ) -> Interpretation | None:
        if self._list_request.match(raw) is None:
            return None
        zone = ZoneInfo(timezone_name)
        local_today = _normalise_now(now).astimezone(zone).date()
        normalized = raw.lower()

        def resolve(value: str) -> date:
            if value == "today":
                return local_today
            if value == "tomorrow":
                return local_today + timedelta(days=1)
            if value in self._weekdays:
                return local_today + timedelta(
                    days=(self._weekdays[value] - local_today.weekday()) % 7
                )
            return date.fromisoformat(value)

        if "this week" in normalized:
            start = local_today - timedelta(days=local_today.weekday())
            return Interpretation(
                intent=ListRemindersIntent(
                    start_date=start,
                    end_date=start + timedelta(days=6),
                    timezone_name=timezone_name,
                ),
                fallback_allowed=False,
            )
        if "next week" in normalized:
            start = local_today - timedelta(days=local_today.weekday()) + timedelta(days=7)
            return Interpretation(
                intent=ListRemindersIntent(
                    start_date=start,
                    end_date=start + timedelta(days=6),
                    timezone_name=timezone_name,
                ),
                fallback_allowed=False,
            )

        # “all recurring reminders” has no single requested date.  Show the
        # next seven local days, which gives the user a useful projection of
        # every active recurring series without changing the read-only list
        # intent shape.
        if "all" in normalized:
            return Interpretation(
                intent=ListRemindersIntent(timezone_name=timezone_name),
                fallback_allowed=False,
            )

        if "recurring" in normalized and ("all" in normalized or "every" in normalized):
            return Interpretation(
                intent=ListRemindersIntent(
                    start_date=local_today,
                    end_date=local_today + timedelta(days=6),
                    timezone_name=timezone_name,
                ),
                fallback_allowed=False,
            )

        match = self._list_range.search(raw)
        if match is not None:
            try:
                start = resolve(match.group("start").lower())
                end = resolve(match.group("end").lower())
                if end < start:
                    end += timedelta(days=7)
                return Interpretation(
                    intent=ListRemindersIntent(
                        start_date=start,
                        end_date=end,
                        timezone_name=timezone_name,
                    ),
                    fallback_allowed=False,
                )
            except (TypeError, ValueError, ValidationError):
                return Interpretation(
                    response_text="Please provide a valid reminder date range.",
                    fallback_allowed=False,
                )

        for phrase, offset in (("tomorrow", 1), ("today", 0)):
            if phrase in normalized:
                day = local_today + timedelta(days=offset)
                return Interpretation(
                    intent=ListRemindersIntent(
                        start_date=day,
                        end_date=day,
                        timezone_name=timezone_name,
                    ),
                    fallback_allowed=False,
                )

        for weekday, index in self._weekdays.items():
            if re.search(rf"\b{weekday}\b", normalized):
                day = local_today + timedelta(days=(index - local_today.weekday()) % 7)
                return Interpretation(
                    intent=ListRemindersIntent(
                        start_date=day,
                        end_date=day,
                        timezone_name=timezone_name,
                    ),
                    fallback_allowed=False,
                )

        iso_dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", normalized)
        if len(iso_dates) == 1:
            try:
                day = date.fromisoformat(iso_dates[0])
                return Interpretation(
                    intent=ListRemindersIntent(
                        start_date=day,
                        end_date=day,
                        timezone_name=timezone_name,
                    ),
                    fallback_allowed=False,
                )
            except ValueError:
                return Interpretation(
                    response_text="Please provide a valid reminder date.",
                    fallback_allowed=False,
                )

        if re.search(r"\b(?:from|between|on|for)\b", normalized):
            return Interpretation(
                response_text="Please provide the reminder date or date range.",
                fallback_allowed=True,
            )
        return Interpretation(
            intent=ListRemindersIntent(timezone_name=timezone_name),
            fallback_allowed=False,
        )

    def _build_recurring(
        self,
        match: re.Match[str],
        *,
        timezone_name: str,
        now: datetime,
    ) -> Interpretation:
        raw_day_text = match.group("days").lower()
        if raw_day_text.strip() == "day":
            weekdays = tuple(range(7))
        else:
            raw_days = re.findall(
                r"monday|tuesday|wednesday|thursday|friday|saturday|sunday",
                raw_day_text,
            )
            weekdays = tuple(sorted({self._weekdays[day] for day in raw_days}))
        hour = int(match.group("hour"))
        minute = int(match.group("minute") or 0)
        meridiem = _normalise_meridiem(match.group("meridiem"))
        if not _valid_clock(hour, minute, meridiem):
            return self._invalid_time()
        if meridiem is not None:
            hour = _clock_to_24(hour, meridiem)
        task = _task_from_remainder(match.group("task"))
        if not task:
            question = "What should I remind you about every week?"
            return Interpretation(
                response_text=question,
                fallback_allowed=False,
            )
        if meridiem is None and hour <= 12:
            pending = PendingReminder(
                task=task,
                date_local=now.astimezone(ZoneInfo(timezone_name)).date(),
                hour=hour,
                minute=minute,
                recurrence_weekdays=weekdays,
                timezone_name=timezone_name,
            )
            question = f"Do you mean {hour:g} AM or {hour:g} PM every week?"
            return Interpretation(
                clarification=NeedsClarification(pending=pending, question=question),
                response_text=question,
                fallback_allowed=False,
            )
        local_now = now.astimezone(ZoneInfo(timezone_name))
        start_date = local_now.date()
        for offset in range(8):
            candidate = start_date + timedelta(days=offset)
            if candidate.weekday() not in weekdays:
                continue
            candidate_local = datetime.combine(candidate, time(hour, minute))
            if candidate_local > local_now.replace(tzinfo=None):
                start_local = candidate_local
                break
        else:  # pragma: no cover - a tuple of weekdays always yields a date
            raise AssertionError("could not calculate recurring start")
        try:
            WeeklyRecurrence(weekdays, hour, minute)
            return Interpretation(
                intent=CreateRecurringReminderIntent(
                    task=task,
                    weekdays=weekdays,
                    hour=hour,
                    minute=minute,
                    start_local=start_local,
                    timezone_name=timezone_name,
                )
            )
        except ValidationError:
            return Interpretation(
                response_text="Please provide a short recurring reminder task and time.",
                fallback_allowed=False,
            )

    def _interpret_follow_up(
        self, text: str, *, context: PendingReminder, now: datetime
    ) -> Interpretation:
        raw = text.strip()
        normalized = re.sub(r"[.!?]+$", "", raw.lower()).strip()
        if normalized in _CANCEL_FOLLOW_UPS:
            return Interpretation(
                response_text="Okay, I won't create that reminder.",
                fallback_allowed=False,
                clear_context=True,
            )
        clock = self._clock.search(raw)
        if clock is None:
            # Accept natural follow-ups such as "6:55 pm today" and "8 a.m.".
            clock = self._follow_up_clock.search(raw)
        day = self._day.search(raw)
        hour = context.hour
        minute = context.minute
        meridiem = context.meridiem
        date_local = context.date_local
        date_explicit = context.date_explicit
        if clock is not None:
            hour = int(clock.group("hour"))
            minute = int(clock.group("minute") or 0)
            meridiem = _normalise_meridiem(clock.group("meridiem"))
            if not _valid_clock(hour, minute, meridiem):
                return self._invalid_time()
            if not context.date_explicit and meridiem and date_local is not None:
                local_now = _normalise_now(now).astimezone(ZoneInfo(context.timezone_name))
                if meridiem == "am" and date_local == local_now.date():
                    date_local += timedelta(days=1)
        elif normalized in {"am", "pm"}:
            meridiem = normalized
        if day is not None:
            local_now = _normalise_now(now).astimezone(ZoneInfo(context.timezone_name))
            date_local = _date_from_day(day, local_now)
            date_explicit = True

        task = context.task
        remainder = raw
        if clock is not None:
            remainder = remainder.replace(clock.group(0), "", 1)
        elif normalized in {"am", "pm"}:
            remainder = ""
        if day is not None:
            remainder = remainder.replace(day.group(0), "", 1)
        candidate_task = _task_from_remainder(remainder)
        if task is None and candidate_task and candidate_task.lower() not in _ACKNOWLEDGEMENTS:
            task = candidate_task

        pending = context.model_copy(
            update={
                "task": task,
                "date_local": date_local,
                "hour": hour,
                "minute": minute,
                "meridiem": meridiem,
                "recurrence_weekdays": context.recurrence_weekdays,
                "date_explicit": date_explicit,
            }
        )
        return self._pending_result(
            pending,
            now=now,
            fallback_allowed=normalized not in _ACKNOWLEDGEMENTS,
        )

    def apply_field_updates(
        self,
        context: PendingReminder,
        updates: PendingReminderFieldUpdate,
        *,
        now: datetime,
    ) -> Interpretation:
        """Merge a provider's follow-up fields, then run deterministic validation."""
        pending = merge_pending_reminder_updates(context, updates)
        return self._pending_result(pending, now=now, fallback_allowed=False)

    def _pending_result(
        self,
        pending: PendingReminder,
        *,
        now: datetime,
        fallback_allowed: bool = True,
    ) -> Interpretation:
        if pending.recurrence_frequency is not None and pending.date_local is None:
            local_today = _normalise_now(now).astimezone(
                ZoneInfo(pending.timezone_name)
            ).date()
            pending = pending.model_copy(
                update={"date_local": local_today, "date_explicit": True}
            )
        missing: list[str] = []
        if not pending.task:
            missing.append("task")
        if pending.date_local is None:
            missing.append("date")
        if pending.hour is None:
            missing.append("time")
        elif pending.hour <= 12 and pending.meridiem is None:
            missing.append("AM or PM")
        if missing:
            if pending.recurrence_frequency is not None and "time" in missing:
                question = (
                    f"What time should I use on the "
                        f"{pending.recurrence_day_of_month}{_ordinal_suffix(pending.recurrence_day_of_month or 1)} "
                        f"day of each {'month' if pending.recurrence_frequency == 'MONTHLY' else 'year'}?"
                )
                return Interpretation(
                    clarification=NeedsClarification(pending=pending, question=question),
                    response_text=question,
                    fallback_allowed=fallback_allowed,
                )
            if pending.recurrence_weekdays is not None and missing == ["AM or PM"]:
                assert pending.hour is not None
                question = f"Do you mean {pending.hour:g} AM or {pending.hour:g} PM every week?"
                return Interpretation(
                    clarification=NeedsClarification(pending=pending, question=question),
                    response_text=question,
                    fallback_allowed=fallback_allowed,
                )
            if missing == ["AM or PM"]:
                assert pending.date_local is not None and pending.hour is not None
                zone = ZoneInfo(pending.timezone_name)
                local_now = _normalise_now(now).astimezone(zone)
                clock = _clock_label(pending.hour, pending.minute)
                options: list[tuple[str, date]] = []
                for candidate_meridiem in ("am", "pm"):
                    candidate_date = pending.date_local
                    due_local = datetime.combine(
                        candidate_date,
                        time(_clock_to_24(pending.hour, candidate_meridiem), pending.minute),
                        tzinfo=zone,
                    )
                    if due_local <= local_now and not pending.date_explicit:
                        candidate_date += timedelta(days=1)
                        due_local = datetime.combine(
                            candidate_date,
                            time(_clock_to_24(pending.hour, candidate_meridiem), pending.minute),
                            tzinfo=zone,
                        )
                    if due_local > local_now:
                        options.append((candidate_meridiem, candidate_date))
                options.sort(key=lambda option: (option[1], _clock_to_24(pending.hour, option[0])))
                if not options:
                    future_pending = pending.model_copy(
                        update={"date_local": None, "date_explicit": False}
                    )
                    return Interpretation(
                        clarification=NeedsClarification(
                            pending=future_pending,
                            question="That date has passed. What future date should I use?",
                        ),
                        response_text="That date has passed. What future date should I use?",
                        fallback_allowed=fallback_allowed,
                    )
                rendered_options = [
                    f"{clock} {candidate_meridiem.upper()} "
                    f"{_date_label(candidate_date, local_now.date())}"
                    for candidate_meridiem, candidate_date in options
                ]
                question = "Do you mean " + " or ".join(rendered_options) + "?"
            elif "task" in missing and "AM or PM" in missing:
                hour = pending.hour
                question = (
                    f"What should I remind you about? Also, do you mean {hour:g} PM today "
                    f"or {hour:g} AM tomorrow?"
                )
            elif "time" in missing and pending.date_local is not None:
                question = f"What time should I use on {pending.date_local.strftime('%A')}?"
            elif missing == ["date"]:
                question = "What date should I use?"
            elif "task" in missing and len(missing) == 1:
                question = "What should I remind you about?"
            else:
                question = "What should I remind you about, and when?"
            return Interpretation(
                clarification=NeedsClarification(pending=pending, question=question),
                response_text=question,
                fallback_allowed=fallback_allowed,
            )

        if pending.recurrence_weekdays is not None:
            assert pending.date_local is not None and pending.hour is not None
            recurrence_hour = (
                _clock_to_24(pending.hour, pending.meridiem)
                if pending.meridiem is not None
                else pending.hour
            )
            zone = ZoneInfo(pending.timezone_name)
            local_now = _normalise_now(now).astimezone(zone)
            start_date = max(pending.date_local, local_now.date())
            for offset in range(8):
                candidate_date = start_date + timedelta(days=offset)
                if candidate_date.weekday() not in pending.recurrence_weekdays:
                    continue
                start_local = datetime.combine(candidate_date, time(recurrence_hour, pending.minute))
                if start_local > local_now.replace(tzinfo=None):
                    return Interpretation(
                        intent=CreateRecurringReminderIntent(
                            task=pending.task or "",
                            weekdays=pending.recurrence_weekdays,
                            hour=recurrence_hour,
                            minute=pending.minute,
                            start_local=start_local,
                            timezone_name=pending.timezone_name,
                        ),
                        fallback_allowed=False,
                    )
            raise AssertionError("could not calculate recurring start")

        if pending.recurrence_frequency is not None:
            assert pending.date_local is not None and pending.hour is not None
            recurrence_hour = (
                _clock_to_24(pending.hour, pending.meridiem)
                if pending.meridiem is not None
                else pending.hour
            )
            zone = ZoneInfo(pending.timezone_name)
            local_now = _normalise_now(now).astimezone(zone)
            day = pending.recurrence_day_of_month
            month = pending.recurrence_month
            if day is None:
                return Interpretation(
                    response_text="What day of the month should I use?",
                    fallback_allowed=False,
                )
            candidate_year = local_now.year
            candidate_month = month or local_now.month
            while True:
                try:
                    candidate_date = date(candidate_year, candidate_month, day)
                except ValueError:
                    if pending.recurrence_frequency == "YEARLY":
                        candidate_year += 1
                    else:
                        candidate_month += 1
                        if candidate_month > 12:
                            candidate_month = 1
                            candidate_year += 1
                    continue
                candidate_local = datetime.combine(
                    candidate_date, time(recurrence_hour, pending.minute)
                )
                if candidate_local > local_now.replace(tzinfo=None):
                    try:
                        return Interpretation(
                            intent=CreateRecurringReminderIntent(
                                task=pending.task or "",
                                frequency=pending.recurrence_frequency,
                                interval=pending.recurrence_interval or 1,
                                weekdays=(),
                                day_of_month=day,
                                month=month,
                                hour=recurrence_hour,
                                minute=pending.minute,
                                start_local=candidate_local,
                                timezone_name=pending.timezone_name,
                            ),
                            fallback_allowed=False,
                        )
                    except ValidationError:
                        return Interpretation(
                            response_text="Please provide a valid calendar recurrence.",
                            fallback_allowed=False,
                        )
                if pending.recurrence_frequency == "YEARLY":
                    candidate_year += 1
                else:
                    candidate_month += 1
                    if candidate_month > 12:
                        candidate_month = 1
                        candidate_year += 1

        assert pending.date_local is not None and pending.hour is not None
        due_local = resolve_pending_due_at(pending, now=now)
        zone = ZoneInfo(pending.timezone_name)
        if due_local <= _normalise_now(now).astimezone(zone):
            if pending.date_explicit:
                # The user supplied a valid clock time.  Only the selected
                # date is invalid, so retain the task and time and ask for the
                # one replacement field needed to make this future-facing.
                future_pending = pending.model_copy(
                    update={"date_local": None, "date_explicit": False}
                )
                question = "That date has passed. What future date should I use?"
            else:
                # With an implicit "today", the clock time is the expired
                # part.  Keep the task and inferred date for a corrected time.
                future_pending = pending.model_copy(
                    update={"hour": None, "minute": 0, "meridiem": None}
                )
                question = "That time has passed. What future time should I use?"
            return Interpretation(
                clarification=NeedsClarification(
                    pending=future_pending,
                    question=question,
                ),
                response_text=question,
            )
        return self._build(pending.task or "", due_local.astimezone(timezone.utc), pending.timezone_name)

    @staticmethod
    def _invalid_time() -> Interpretation:
        return Interpretation(
            response_text="Please provide a valid future reminder time.",
            fallback_allowed=False,
        )

    def _interpret_multiple_time_clauses(
        self, raw: str, *, timezone_name: str, now: datetime
    ) -> Interpretation | None:
        prefix = self._prefix.match(raw)
        if prefix is None:
            return None
        body = raw[prefix.end() :]
        clocks = list(self._clock.finditer(body))
        if len(clocks) < 2:
            return None

        shared = body[: clocks[0].start()].strip(" ,;:")
        shared_day = self._day.search(shared)
        if shared_day is None:
            # Without a shared date, let the model resolve whether each time
            # belongs to today, tomorrow, or another date.
            return Interpretation(
                response_text="I need a little more detail to schedule those reminders.",
                fallback_allowed=True,
            )
        shared_before_day = shared[: shared_day.start()].strip()
        shared_after_day = shared[shared_day.end() :].strip()
        if shared_before_day or shared_after_day:
            # The first task may precede its date ("call Alice tomorrow at
            # 9 AM"). That is handled by the normal repeated-prefix splitter;
            # this helper is only for a date shared by clock-led clauses.
            return None
        shared_date = shared[shared_day.start() : shared_day.end()]
        commands: list[str] = []
        for index, clock in enumerate(clocks):
            end = clocks[index + 1].start() if index + 1 < len(clocks) else len(body)
            clause = body[clock.start() : end].strip(" ,;:")
            clause = re.sub(r"\s+and\s*$", "", clause, flags=re.IGNORECASE)
            commands.append(f"Remind me {shared_date} {clause}")

        parsed = [
            self.interpret(command, timezone_name=timezone_name, now=now)
            for command in commands
        ]
        if all(item.intent is not None for item in parsed):
            return Interpretation(
                intents=tuple(item.intent for item in parsed if item.intent is not None),
                fallback_allowed=False,
            )
        return Interpretation(
            response_text="I found multiple reminder times. Please provide a complete task and time for each.",
            fallback_allowed=True,
        )

    @staticmethod
    def _multiple_target_clarification(timezone_name: str) -> Interpretation:
        question = (
            "I found more than one reminder task. Please send one task at a time, "
            "or repeat 'remind me' for each reminder."
        )
        return Interpretation(
            clarification=NeedsClarification(
                pending=PendingReminder(timezone_name=timezone_name),
                question=question,
            ),
            response_text=question,
        )

    @staticmethod
    def _build(task: str, due_at: datetime, timezone_name: str) -> Interpretation:
        if not task:
            # The parser has already resolved a complete, user-supplied
            # schedule.  Keep it in the draft while asking only for the task;
            # otherwise the next turn has no way to complete this reminder.
            local_due_at = due_at.astimezone(ZoneInfo(timezone_name))
            hour = local_due_at.hour % 12 or 12
            meridiem = "am" if local_due_at.hour < 12 else "pm"
            question = "What should I remind you about?"
            return Interpretation(
                clarification=NeedsClarification(
                    pending=PendingReminder(
                        date_local=local_due_at.date(),
                        hour=hour,
                        minute=local_due_at.minute,
                        meridiem=meridiem,
                        timezone_name=timezone_name,
                        date_explicit=True,
                    ),
                    question=question,
                ),
                response_text=question,
            )
        try:
            return Interpretation(
                intent=CreateOneTimeReminderIntent(
                    task=task,
                    due_at=due_at,
                    timezone_name=timezone_name,
                )
            )
        except ValidationError:
            return Interpretation(
                response_text="Please provide a short reminder task and future time.",
                fallback_allowed=False,
            )


class OpenAICompatibleIntentInterpreter:
    """Structured intent adapter for OpenAI-compatible chat-completion APIs."""

    def __init__(
        self, base_url: str, api_key: str, model: str, timeout_seconds: float = 10.0
    ) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_seconds

    def interpret(
        self,
        text: str,
        *,
        selected_route: IntentRoute | None = None,
        timezone_name: str,
        now: datetime,
        context: PendingReminder | None = None,
        recent_messages: Sequence[str] = (),
    ) -> Interpretation:
        route = selected_route or (
            IntentRoute.UPDATE_PENDING if context is not None else IntentRoute.NEW_REQUEST
        )
        if route is IntentRoute.UNCERTAIN:
            return Interpretation(
                response_text="I’m not sure what you want to do. Please clarify.",
                fallback_allowed=False,
            )
        if route is IntentRoute.UPDATE_PENDING and context is None:
            return Interpretation(
                response_text="There is no unfinished reminder to update.",
                fallback_allowed=False,
            )
        if route is not IntentRoute.UPDATE_PENDING:
            context = None
        local_now = _normalise_now(now).astimezone(ZoneInfo(timezone_name))
        endpoint_host = urlsplit(self._url).hostname or "unknown"
        log_event(
            logger,
            logging.INFO,
            "reminder_intent_provider_request_started",
            endpoint_host=endpoint_host,
            model=self._model,
            input_length=len(text.strip()),
            timezone=timezone_name,
            has_pending_context=context is not None,
        )
        prompt = (
            """
                Extract structured reminder data from the latest message according to selected_route.

                NEW_REQUEST:
                - CREATE_ONE_TIME_REMINDER: task, due_local.
                - CREATE_ONE_TIME_REMINDERS: separate intents array of {task, due_local}.
                - CREATE_RECURRING_REMINDER: task, frequency (DAILY, WEEKLY, MONTHLY, YEARLY), interval, applicable schedule fields, and start_local.
                LIST_REQUEST:
                - LIST_REMINDERS: return only query parameters. Use start_date and end_date as ISO-8601 dates for a requested date or inclusive date range; omit both to list all reminders. Never return reminder records, tasks, or due times: the application queries its own database.

                RESCHEDULE_REQUEST:
                - RESCHEDULE_REMINDER: target_text, due_local, and scope (OCCURRENCE or SERIES). due_local must be a timezone-naive local ISO-8601 datetime. If the target, time, or scope is ambiguous, return NEEDS_CLARIFICATION with a focused question. Never mutate data.

                UPDATE_PENDING:
                - UPDATE_PENDING_REMINDER: extract only explicitly supplied changes (task, date_local, hour, minute, meridiem, date_explicit, recurrence_frequency, recurrence_interval, recurrence_day_of_month, recurrence_month), with certainty="CLEAR" and exact evidence for each field. Recurrence frequency is MONTHLY or YEARLY; interval is 1-60; day_of_month is 1-31 and month is 1-12. Do not repeat or change other pending fields.

                For UPDATE_PENDING_REMINDER, return an updates object whose field values are primitive values directly, for example: {"operation":"UPDATE_PENDING_REMINDER","updates":{"task":"pay rent","date_local":"2026-11-05","hour":2,"meridiem":"pm","date_explicit":true,"certainty":"CLEAR","evidence":{"task":"pay rent","date_local":"November 5","hour":"2 PM","meridiem":"PM","date_explicit":"November 5"}}}. Do not wrap any updates field in a value/certainty/evidence object; evidence belongs only in the updates.evidence mapping. Include only fields explicitly supplied by the user; do not infer omitted fields or defaults.

                If required fields are missing or ambiguous, return NEEDS_CLARIFICATION with relevant pending fields and a focused question.

                Use provided local time and timezone. Never invent tasks, dates, times, or recurrence rules. Use timezone-naive ISO-8601 local datetimes.

                Return JSON only. The operation must match selected_route.
            """
        )
        prompt += f"\nSelected route: {route.value}."
        prompt += f" Current local date/time is {local_now.isoformat(timespec='minutes')} ({timezone_name})."
        if context is not None:
            prompt += f" Existing pending reminder context is {context.model_dump_json()}"
        if recent_messages:
            prompt += (
                " Relevant recent conversation messages are "
                f"{json.dumps(list(recent_messages)[-34:])}."
            )
        request_payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        log_event(
            logger,
            logging.INFO,
            "reminder_intent_provider_request",
            endpoint_host=endpoint_host,
            model=self._model,
            payload=request_payload,
        )
        try:
            response = httpx.post(
                self._url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=request_payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            content = payload["choices"][0]["message"]["content"]
            log_event(
                logger,
                logging.INFO,
                "reminder_intent_provider_response",
                endpoint_host=endpoint_host,
                model=self._model,
                response_length=len(content) if isinstance(content, str) else None,
            )
            data = json.loads(content) if isinstance(content, str) else content
            if not isinstance(data, dict):
                raise ValueError("provider response was not an object")
            log_event(
                logger,
                logging.INFO,
                "reminder_intent_provider_raw_response",
                endpoint_host=endpoint_host,
                model=self._model,
                route=route,
                provider_response=data,
            )
            operation = data.get("operation")
            if route is IntentRoute.LIST_REQUEST and operation in {
                "LIST_REQUEST",
                "LIST_REMINDER",
                "LIST_REMINDERS_BY_RANGE",
                "RETRIEVE_REMINDERS",
                "SHOW_REMINDERS",
            }:
                operation = "LIST_REMINDERS"
            if route is IntentRoute.RESCHEDULE_REQUEST:
                if operation == "RESCHEDULE_REMINDER":
                    from reminder_agent.rescheduling import RescheduleRequest, RescheduleScope

                    target_text = data.get("target_text")
                    due_local_value = data.get("due_local")
                    scope_value = data.get("scope")
                    if not isinstance(target_text, str) or not target_text.strip():
                        raise ValueError("reschedule intent is missing target_text")
                    if not isinstance(due_local_value, str):
                        raise ValueError("reschedule intent is missing due_local")
                    due_local = datetime.fromisoformat(due_local_value)
                    if due_local.tzinfo is not None and due_local.utcoffset() is not None:
                        raise ValueError("provider returned an offset-bearing local datetime")
                    if scope_value is not None:
                        try:
                            scope = RescheduleScope(scope_value)
                        except ValueError as exc:
                            raise ValueError("reschedule intent has invalid scope") from exc
                    else:
                        scope = None
                    return Interpretation(
                        reschedule_request=RescheduleRequest(
                            target_text=target_text,
                            due_at=due_local.replace(tzinfo=ZoneInfo(timezone_name)).astimezone(
                                timezone.utc
                            ),
                            scope=scope,
                        ),
                        fallback_allowed=False,
                    )
                if operation == "NEEDS_CLARIFICATION":
                    question = data.get("question")
                    if not isinstance(question, str) or not question.strip():
                        raise ValueError("missing reschedule clarification question")
                    return Interpretation(response_text=question.strip(), fallback_allowed=False)
                raise ValueError("provider returned an unsupported reschedule operation")
            if route is IntentRoute.UPDATE_PENDING:
                if operation == "UPDATE_PENDING_REMINDER":
                    raw_updates = data.get("updates")
                    if raw_updates is None:
                        # Some OpenAI-compatible models emit the patch fields
                        # beside ``operation`` despite the nested contract.
                        # Normalize that equivalent shape at the boundary;
                        # all values still pass PendingReminderFieldUpdate and
                        # application-side grounding below.
                        update_fields = {
                            "task",
                            "date_local",
                            "hour",
                            "minute",
                            "meridiem",
                            "date_explicit",
                            "recurrence_frequency",
                            "recurrence_interval",
                            "recurrence_day_of_month",
                            "recurrence_month",
                            "certainty",
                            "evidence",
                        }
                        raw_updates = {
                            key: value for key, value in data.items() if key in update_fields
                        }
                    if not isinstance(raw_updates, dict):
                        raise ValueError("pending update was not an object")
                    updates = PendingReminderFieldUpdate.model_validate(raw_updates)
                    if not updates.supplied_fields():
                        raise ValueError("pending update had no supplied fields")
                    log_event(
                        logger,
                        logging.INFO,
                        "reminder_extraction_completed",
                        model=self._model,
                        operation=operation,
                        field_count=len(updates.supplied_fields()),
                        has_pending_context=True,
                    )
                    return Interpretation(field_updates=updates, fallback_allowed=False)
                if operation == "NEEDS_CLARIFICATION":
                    question = data.get("question")
                    if not isinstance(question, str) or not question.strip():
                        raise ValueError("missing clarification question")
                    # Preserve the application-owned draft.  The extraction
                    # model may request clarification, but cannot replace or
                    # clear trusted pending fields.
                    return Interpretation(
                        clarification=NeedsClarification(pending=context, question=question.strip()),
                        fallback_allowed=False,
                    )
                if operation == "UNKNOWN":
                    return Interpretation(
                        response_text=OUT_OF_SCOPE_RESPONSE,
                        fallback_allowed=False,
                    )
                # A model response must never be authoritative while a
                # clarification is in progress.  Retain the trusted context
                # through the caller's deterministic result instead.
                raise ValueError("provider returned an authoritative pending reminder")
            if operation == "NEEDS_CLARIFICATION":
                raw_pending = data.get("pending") or {}
                if not isinstance(raw_pending, dict):
                    raise ValueError("pending clarification was not an object")
                # The rule parser may already have extracted trusted fields
                # from an incomplete command (for example, the task from
                # "Remind me to submit my assignment").  A model
                # clarification is only a supplement; it must not erase those
                # fields when its pending object omits them or returns null.
                pending_values = {}
                if context is not None:
                    pending_values = context.model_dump(mode="python")
                for field in (
                    "task",
                    "date_local",
                    "hour",
                    "minute",
                    "meridiem",
                    "date_explicit",
                    "recurrence_weekdays",
                    "recurrence_frequency",
                    "recurrence_interval",
                    "recurrence_day_of_month",
                    "recurrence_month",
                ):
                    if field in raw_pending and raw_pending[field] is not None:
                        pending_values[field] = raw_pending[field]
                pending_values["timezone_name"] = timezone_name
                pending = PendingReminder.model_validate(
                    pending_values
                )
                question = data.get("question")
                if not isinstance(question, str) or not question.strip():
                    raise ValueError("missing clarification question")
                log_event(
                    logger,
                    logging.INFO,
                    "reminder_extraction_completed",
                    model=self._model,
                    operation=operation,
                    field_count=sum(value is not None for value in pending_values.values()),
                    has_pending_context=False,
                )
                return Interpretation(
                    clarification=NeedsClarification(pending=pending, question=question)
                )
            if operation == "CREATE_ONE_TIME_REMINDERS":
                raw_intents = data.get("intents")
                if not isinstance(raw_intents, list) or not raw_intents:
                    raise ValueError("multiple reminder intents were not an array")
                intents: list[CreateOneTimeReminderIntent] = []
                for raw_intent in raw_intents:
                    if not isinstance(raw_intent, dict):
                        raise ValueError("multiple reminder intent was not an object")
                    task = raw_intent.get("task")
                    due_local_value = raw_intent.get("due_local")
                    if not isinstance(task, str) or not isinstance(due_local_value, str):
                        raise ValueError("multiple reminder intent is missing fields")
                    due_local = datetime.fromisoformat(due_local_value)
                    if due_local.tzinfo is not None and due_local.utcoffset() is not None:
                        raise ValueError("provider returned an offset-bearing local datetime")
                    intents.append(
                        CreateOneTimeReminderIntent(
                            task=task,
                            due_at=due_local.replace(tzinfo=ZoneInfo(timezone_name)).astimezone(
                                timezone.utc
                            ),
                            timezone_name=timezone_name,
                        )
                    )
                return Interpretation(intents=tuple(intents))
            if operation == "CREATE_RECURRING_REMINDER":
                task = data.get("task")
                weekdays = data.get("weekdays")
                start_local_value = data.get("start_local")
                frequency = str(data.get("frequency", "WEEKLY")).upper()
                if not isinstance(task, str):
                    raise ValueError("recurring intent is missing fields")
                if not isinstance(weekdays, list):
                    weekdays = []
                interval = int(data.get("interval", 1))
                hour = int(data["hour"])
                minute = int(data.get("minute", 0))
                day_of_month = (
                    int(data["day_of_month"])
                    if data.get("day_of_month") is not None
                    else None
                )
                recurrence_month = int(data["month"]) if data.get("month") is not None else None
                if isinstance(start_local_value, str):
                    start_local = datetime.fromisoformat(start_local_value)
                else:
                    start_local = _next_recurring_start_local(
                        frequency=frequency,
                        interval=interval,
                        weekdays=tuple(int(day) for day in weekdays),
                        day_of_month=day_of_month,
                        month=recurrence_month,
                        hour=hour,
                        minute=minute,
                        timezone_name=timezone_name,
                        now=now,
                    )
                intent = CreateRecurringReminderIntent(
                    task=task,
                    frequency=frequency,
                    interval=interval,
                    weekdays=tuple(int(day) for day in weekdays),
                    day_of_month=day_of_month,
                    month=recurrence_month,
                    hour=hour,
                    minute=minute,
                    start_local=start_local,
                    timezone_name=timezone_name,
                )
                return Interpretation(intent=intent)
            if operation == "LIST_REMINDERS":
                start_date_value = data.get("start_date")
                end_date_value = data.get("end_date")
                date_range = data.get("date_range")
                if isinstance(date_range, dict):
                    start_date_value = start_date_value or date_range.get("start")
                    end_date_value = end_date_value or date_range.get("end")
                # Accept the single-date shape emitted by some compatible
                # models, but normalize it into the application's inclusive
                # range representation.  Reminder records in the provider
                # response are deliberately ignored; the database is the
                # authoritative source for every listed reminder.
                single_date_value = data.get("date")
                if single_date_value is not None:
                    if start_date_value is not None or end_date_value is not None:
                        raise ValueError("list intent has conflicting date filters")
                    start_date_value = single_date_value
                    end_date_value = single_date_value
                if (start_date_value is None) != (end_date_value is None):
                    raise ValueError("list intent has an incomplete date range")
                if start_date_value is not None and (
                    not isinstance(start_date_value, str) or not isinstance(end_date_value, str)
                ):
                    raise ValueError("list intent has invalid date range")
                return Interpretation(
                    intent=ListRemindersIntent(
                        start_date=(date.fromisoformat(start_date_value) if start_date_value else None),
                        end_date=(date.fromisoformat(end_date_value) if end_date_value else None),
                        timezone_name=timezone_name,
                    )
                )
            if operation != "CREATE_ONE_TIME_REMINDER":
                result = Interpretation(
                    response_text=OUT_OF_SCOPE_RESPONSE
                )
                log_event(
                    logger,
                    logging.WARNING,
                    "reminder_intent_provider_finished",
                    endpoint_host=endpoint_host,
                    model=self._model,
                    has_intent=False,
                    reason="unsupported_operation",
                )
                return result
            task = data.get("task")
            due_local_value = data.get("due_local")
            if not isinstance(task, str) or not isinstance(due_local_value, str):
                raise ValueError("missing structured fields")
            due_local = datetime.fromisoformat(due_local_value)
            if due_local.tzinfo is not None and due_local.utcoffset() is not None:
                raise ValueError("provider returned an offset-bearing local datetime")
            due_local = due_local.replace(tzinfo=ZoneInfo(timezone_name))
            intent = CreateOneTimeReminderIntent(
                task=task,
                due_at=due_local.astimezone(timezone.utc),
                timezone_name=timezone_name,
            )
            result = Interpretation(intent=intent)
            log_event(
                logger,
                logging.INFO,
                "reminder_extraction_completed",
                model=self._model,
                operation=operation,
                field_count=2,
                has_pending_context=False,
            )
            log_event(
                logger,
                logging.INFO,
                "reminder_intent_provider_finished",
                endpoint_host=endpoint_host,
                model=self._model,
                has_intent=True,
                task_length=len(task),
            )
            return result
        except (
            httpx.HTTPError,
            KeyError,
            TypeError,
            ValueError,
            ValidationError,
            json.JSONDecodeError,
        ) as exc:
            log_event(
                logger,
                logging.ERROR,
                "reminder_intent_provider_failed",
                endpoint_host=endpoint_host,
                model=self._model,
                error_type="provider_or_schema_error",
                has_pending_context=context is not None,
                error_message=(
                    exc.errors(include_url=False)
                    if isinstance(exc, ValidationError)
                    else str(exc)
                ),
            )
            return Interpretation(
                response_text=(
                    "I couldn't understand that reminder. Please provide the task and future time."
                )
            )
        except Exception as exc:  # pragma: no cover - defensive provider boundary
            log_event(
                logger,
                logging.ERROR,
                "reminder_intent_provider_failed",
                endpoint_host=endpoint_host,
                model=self._model,
                error_type=type(exc).__name__,
            )
            return Interpretation(
                response_text=(
                    "I couldn't understand that reminder. Please provide the task and future time."
                )
            )


class ReminderIntentInterpreter:
    """Use deterministic parsing first and an optional configured model second."""

    def __init__(self, model_interpreter: IntentInterpreter | None = None) -> None:
        self._rules = RuleBasedIntentInterpreter()
        self._model = model_interpreter

    def interpret_reschedule(
        self, text: str, *, timezone_name: str, now: datetime
    ) -> Any:
        """Return a non-authoritative reschedule proposal from the extractor, if configured."""

        if not isinstance(self._model, OpenAICompatibleIntentInterpreter):
            return None
        result = self._model.interpret(
            text,
            selected_route=IntentRoute.RESCHEDULE_REQUEST,
            timezone_name=timezone_name,
            now=now,
        )
        return result.reschedule_request or result.response_text

    def interpret(
        self,
        text: str,
        *,
        selected_route: IntentRoute | None = None,
        timezone_name: str,
        now: datetime,
        context: PendingReminder | None = None,
        recent_messages: Sequence[str] = (),
    ) -> Interpretation:
        route = selected_route or (
            IntentRoute.UPDATE_PENDING if context is not None else IntentRoute.NEW_REQUEST
        )
        if route is IntentRoute.UNCERTAIN:
            return Interpretation(
                response_text="I’m not sure what you want to do. Please clarify.",
                fallback_allowed=False,
            )
        extraction_context = context if route is IntentRoute.UPDATE_PENDING else None
        result = self._rules.interpret(
            text,
            selected_route=route,
            timezone_name=timezone_name,
            now=now,
            context=extraction_context,
        )
        if (
            result.intent is not None
            or result.intents
            or self._model is None
            or not result.fallback_allowed
        ):
            return result
        log_event(
            logger,
            logging.INFO,
            "reminder_intent_provider_fallback",
            reason="rules_did_not_match",
        )
        if isinstance(self._model, OpenAICompatibleIntentInterpreter):
            provider_result = self._model.interpret(
                text,
                selected_route=route,
                timezone_name=timezone_name,
                now=now,
                context=extraction_context,
                recent_messages=recent_messages,
            )
        else:
            provider_kwargs = {
                "timezone_name": timezone_name,
                "now": now,
                "context": extraction_context,
            }
            if "selected_route" in inspect.signature(self._model.interpret).parameters:
                provider_kwargs["selected_route"] = route
            provider_result = self._model.interpret(text, **provider_kwargs)
        # A provider response that contains no typed action must not erase the
        # deterministic context or replace a focused clarification with a generic
        # error. Keep the safer rule result in that case.
        if route is IntentRoute.UPDATE_PENDING:
            # During a clarification, only a typed patch (or an explicit
            # cancellation recognized by a provider implementation) can cross
            # the provider boundary.  A provider cannot replace the session
            # with a completed reminder or a newly assembled pending object.
            if (
                provider_result.field_updates is not None
                or provider_result.clear_context
                or provider_result.clarification is not None
            ):
                return provider_result
            return result
        if (
            provider_result.resolved_intents
            or provider_result.clarification is not None
            or provider_result.clear_context
        ):
            if (
                context is None
                and result.clarification is not None
                and provider_result.clarification is not None
            ):
                # Keep fields deterministically extracted from the original
                # command when the provider returns only a partial draft.
                provider_pending = provider_result.clarification.pending
                base_pending = result.clarification.pending
                merged_values = base_pending.model_dump(mode="python")
                for field in (
                    "task",
                    "date_local",
                    "hour",
                    "minute",
                    "meridiem",
                    "date_explicit",
                ):
                    # A Pydantic model supplies defaults (for example,
                    # ``minute=0`` and ``date_explicit=False``) even when the
                    # provider omitted a field.  Only explicitly returned
                    # values are permitted to supplement the trusted draft.
                    value = getattr(provider_pending, field)
                    if field in provider_pending.model_fields_set and value is not None:
                        merged_values[field] = value
                merged_pending = PendingReminder.model_validate(merged_values)
                return Interpretation(
                    clarification=NeedsClarification(
                        pending=merged_pending,
                        question=provider_result.clarification.question,
                    ),
                    response_text=provider_result.response_text,
                    fallback_allowed=provider_result.fallback_allowed,
                )
            return provider_result
        return result
