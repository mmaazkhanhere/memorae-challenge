from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import StrEnum
from typing import Any, Literal, Protocol, Sequence
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from reminder_agent.observability import get_logger, log_event

logger = get_logger()


class IntentRoute(StrEnum):
    NEW_REQUEST = "NEW_REQUEST"
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
            "Return JSON only: {\"operation\": one of NEW_REQUEST, UPDATE_PENDING, "
            "CANCEL_PENDING, SOCIAL, UNRELATED, UNCERTAIN}. Classify intent only; do not "
            "extract reminder fields, suggest values, or execute anything. NEW_REQUEST is an "
            "independent reminder; UPDATE_PENDING changes the supplied draft. If the message "
            "could reasonably mean either, return UNCERTAIN. Use active_pending, missing_fields, "
            "and last_clarification_question to classify a reply to an active clarification."
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


class PendingReminder(BaseModel):
    """Validated, serializable partial data for a one-time reminder."""

    model_config = ConfigDict(extra="forbid")

    operation: Literal["CREATE_ONE_TIME_REMINDER"] = "CREATE_ONE_TIME_REMINDER"
    task: str | None = Field(default=None, max_length=200)
    date_local: date | None = None
    hour: int | None = Field(default=None, ge=0, le=23)
    minute: int = Field(default=0, ge=0, le=59)
    meridiem: Literal["am", "pm"] | None = None
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
    certainty: Literal["CLEAR"] | None = None
    evidence: dict[
        Literal["task", "date_local", "hour", "minute", "meridiem", "date_explicit"], str
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
    intent: CreateOneTimeReminderIntent | None = None
    intents: tuple[CreateOneTimeReminderIntent, ...] = ()
    clarification: NeedsClarification | None = None
    field_updates: PendingReminderFieldUpdate | None = None
    response_text: str | None = None
    fallback_allowed: bool = True
    clear_context: bool = False

    @property
    def resolved_intents(self) -> tuple[CreateOneTimeReminderIntent, ...]:
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
    """Require semantic model patches to cite an exact fragment of this turn."""
    if not isinstance(evidence, str):
        return False
    normalized_evidence = " ".join(evidence.lower().split())
    normalized_text = " ".join(text.lower().split())
    return bool(normalized_evidence) and normalized_evidence in normalized_text


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
        r"^\s*(?:please\s+)?(?:remind\s+me|set\s+(?:a\s+)?reminder)\s+",
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

    def interpret(
        self,
        text: str,
        *,
        timezone_name: str,
        now: datetime,
        context: PendingReminder | None = None,
    ) -> Interpretation:
        raw = text.strip()
        # An explicit new reminder command supersedes an unresolved clarification.
        # This gives users a safe way to answer a multi-target question with one
        # complete request instead of merging unrelated text into the old action.
        if context is not None and self._prefix.match(raw):
            context = None
        if context is not None:
            return self._interpret_follow_up(text, context=context, now=now)

        # A single command containing a second action is not a safe regex
        # parse. Let the model split it into independently validated intents.
        if _COMPOUND_REQUEST.search(raw) and not self._multiple_request_separator.search(raw):
            return self._multiple_target_clarification(timezone_name)

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
            # Keep bare clock answers and social acknowledgements deterministic,
            # but allow the provider to interpret complex natural language even
            # when it does not contain the literal word "remind".
            simple_time = self._follow_up_clock.fullmatch(raw.strip(" .,!"))
            simple_ack = raw.lower().strip(" .,!?") in (
                _ACKNOWLEDGEMENTS | _CANCEL_FOLLOW_UPS
            )
            return Interpretation(
                response_text=(
                    "I can help create reminders. Start with what to remember and when."
                ),
                fallback_allowed=not (simple_time or simple_ack),
            )

        # Relative phrases, recurrence, and conversational qualifiers have
        # too much meaning for a small regex parser.  Do not extract a partial
        # intent that could be mistaken for a complete command; let the model
        # return a typed proposal instead.
        if _REQUIRES_INTERPRETATION.search(raw):
            return Interpretation(
                response_text="I need a little more detail to schedule that reminder.",
                fallback_allowed=True,
            )

        zone = ZoneInfo(timezone_name)
        reference = _normalise_now(now)
        without_prefix = raw[prefix.end() :]

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
            task = _task_from_remainder(before_time if before_time.strip() else after_time)
            if _contains_multiple_tasks(task):
                return self._multiple_target_clarification(timezone_name)
            return self._build(task, reference + delta, timezone_name)

        tomorrow = self._tomorrow.search(without_prefix)
        if tomorrow is not None:
            hour = int(tomorrow.group("hour"))
            minute = int(tomorrow.group("minute") or 0)
            meridiem = _normalise_meridiem(tomorrow.group("meridiem"))
            if not _valid_clock(hour, minute, meridiem):
                return self._invalid_time()
            local_date = reference.astimezone(zone).date() + timedelta(days=1)
            before_time = without_prefix[: tomorrow.start()]
            after_time = without_prefix[tomorrow.end() :]
            task = _task_from_remainder(before_time if before_time.strip() else after_time)
            if _contains_multiple_tasks(task):
                return self._multiple_target_clarification(timezone_name)
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
            if _contains_multiple_tasks(task):
                return self._multiple_target_clarification(timezone_name)
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
        if day is not None:
            local_now = _normalise_now(now).astimezone(ZoneInfo(context.timezone_name))
            date_local = _date_from_day(day, local_now)
            date_explicit = True

        task = context.task
        remainder = raw
        if clock is not None:
            remainder = remainder.replace(clock.group(0), "", 1)
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
            question = "What should I remind you about?"
            return Interpretation(
                clarification=NeedsClarification(
                    pending=PendingReminder(timezone_name=timezone_name),
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
        timezone_name: str,
        now: datetime,
        context: PendingReminder | None = None,
        recent_messages: Sequence[str] = (),
    ) -> Interpretation:
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
            "Return JSON only. For one complete request use operation "
            "CREATE_ONE_TIME_REMINDER with task and due_local. For multiple clearly separated "
            "requests use CREATE_ONE_TIME_REMINDERS with an intents array of {task,due_local}; "
            "never merge separate tasks into one task. For an incomplete or ambiguous "
            "request use operation NEEDS_CLARIFICATION with pending fields task, date_local "
            "(YYYY-MM-DD), hour, minute, meridiem, date_explicit, and a focused question. "
            "For unrelated requests use operation UNKNOWN. Never guess an ambiguous hour. "
            "When existing pending reminder context is supplied, use UPDATE_PENDING_REMINDER "
            "with an updates object containing only fields explicitly "
            "provided by the user (task, date_local, hour, minute, meridiem, date_explicit), "
            "plus certainty=\"CLEAR\" and evidence mapping each updated field to an exact "
            "fragment of the latest user message. "
            "Correct a typo only when it has one clear interpretation. If a date, time, task, "
            "or target is ambiguous, use NEEDS_CLARIFICATION with a focused question and no "
            "updates. "
            "Do not include an existing task, date, or time unless the user explicitly corrects "
            "that exact field in this message. Never repeat, infer, preserve, merge, or clear a "
            "context field. Never return CREATE_ONE_TIME_REMINDER or due_local for a follow-up; "
            "the application owns the pending reminder and will merge and validate the update. "
            "Use UNKNOWN when the follow-up supplies no reminder field. "
            "due_local must be a timezone-naive ISO-8601 local datetime; the application applies "
            f"the user's validated timezone {timezone_name}. Current local time is "
            f"{local_now.isoformat()}."
        )
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
            operation = data.get("operation")
            if context is not None:
                if operation == "UPDATE_PENDING_REMINDER":
                    raw_updates = data.get("updates")
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
                        response_text="I couldn't find a reminder detail in that reply.",
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
            if operation != "CREATE_ONE_TIME_REMINDER":
                result = Interpretation(
                    response_text="Please tell me the reminder task and a future time."
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

    def interpret(
        self,
        text: str,
        *,
        timezone_name: str,
        now: datetime,
        context: PendingReminder | None = None,
        recent_messages: Sequence[str] = (),
    ) -> Interpretation:
        result = self._rules.interpret(
            text, timezone_name=timezone_name, now=now, context=context
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
                timezone_name=timezone_name,
                now=now,
                context=context,
                recent_messages=recent_messages,
            )
        else:
            provider_result = self._model.interpret(
                text, timezone_name=timezone_name, now=now, context=context
            )
        # A provider response that contains no typed action must not erase the
        # deterministic context or replace a focused clarification with a generic
        # error. Keep the safer rule result in that case.
        if context is not None:
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
                    value = getattr(provider_pending, field)
                    if value is not None:
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
