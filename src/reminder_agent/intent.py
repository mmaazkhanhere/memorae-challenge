from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from reminder_agent.observability import get_logger, log_event

logger = get_logger()


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


@dataclass(frozen=True)
class Interpretation:
    intent: CreateOneTimeReminderIntent | None = None
    response_text: str | None = None
    fallback_allowed: bool = True


class IntentInterpreter(Protocol):
    def interpret(
        self, text: str, *, timezone_name: str, now: datetime
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
    task = re.sub(r"^(?:to|for|that|about)\s+", "", task, flags=re.IGNORECASE)
    return " ".join(task.split())


def _number(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return _NUMBER_WORDS.get(value.lower())


class RuleBasedIntentInterpreter:
    """Conservative parser for the relative forms required by VS-03."""

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

    def interpret(self, text: str, *, timezone_name: str, now: datetime) -> Interpretation:
        raw = text.strip()
        log_event(
            logger,
            logging.DEBUG,
            "reminder_intent_rules_started",
            input_length=len(raw),
            timezone=timezone_name,
        )
        prefix = self._prefix.match(raw)
        if prefix is None:
            result = Interpretation(
                response_text="I can help create reminders. Tell me what and when to remember."
            )
            log_event(
                logger,
                logging.DEBUG,
                "reminder_intent_rules_finished",
                matched=False,
                fallback_allowed=result.fallback_allowed,
                reason="not_reminder_prefix",
            )
            return result

        zone = ZoneInfo(timezone_name)
        reference = _normalise_now(now)
        without_prefix = raw[prefix.end() :]

        relative = self._relative.search(without_prefix)
        if relative is not None:
            amount = _number(relative.group("amount"))
            if amount is None or amount <= 0:
                result = Interpretation(
                    response_text="Please provide a valid future reminder time.",
                    fallback_allowed=False,
                )
                log_event(
                    logger,
                    logging.WARNING,
                    "reminder_intent_rules_rejected",
                    schedule_kind="relative",
                    reason="invalid_duration",
                )
                return result
            unit = relative.group("unit").lower()
            delta = timedelta(minutes=amount if unit.startswith("minute") else amount * 60)
            task = _task_from_remainder(without_prefix[relative.end() :])
            result = self._build(task, reference + delta, timezone_name)
            log_event(
                logger,
                logging.INFO if result.intent is not None else logging.WARNING,
                "reminder_intent_rules_finished",
                matched=result.intent is not None,
                schedule_kind="relative",
                task_length=len(task),
                fallback_allowed=result.fallback_allowed,
            )
            return result

        tomorrow = self._tomorrow.search(without_prefix)
        if tomorrow is not None:
            hour = int(tomorrow.group("hour"))
            minute = int(tomorrow.group("minute") or 0)
            meridiem = (tomorrow.group("meridiem") or "").lower().replace(".", "")
            if meridiem:
                if hour < 1 or hour > 12:
                    result = Interpretation(
                        response_text="Please provide a valid future reminder time.",
                        fallback_allowed=False,
                    )
                    log_event(
                        logger,
                        logging.WARNING,
                        "reminder_intent_rules_rejected",
                        schedule_kind="tomorrow_at",
                        reason="invalid_hour",
                    )
                    return result
                if meridiem == "pm" and hour != 12:
                    hour += 12
                if meridiem == "am" and hour == 12:
                    hour = 0
            elif hour > 23:
                result = Interpretation(
                    response_text="Please provide a valid future reminder time.",
                    fallback_allowed=False,
                )
                log_event(
                    logger,
                    logging.WARNING,
                    "reminder_intent_rules_rejected",
                    schedule_kind="tomorrow_at",
                    reason="invalid_hour",
                )
                return result
            local_reference = reference.astimezone(zone)
            local_date = local_reference.date() + timedelta(days=1)
            due_local = datetime.combine(local_date, time(hour, minute), tzinfo=zone)
            task = _task_from_remainder(without_prefix[tomorrow.end() :])
            result = self._build(task, due_local.astimezone(timezone.utc), timezone_name)
            log_event(
                logger,
                logging.INFO if result.intent is not None else logging.WARNING,
                "reminder_intent_rules_finished",
                matched=result.intent is not None,
                schedule_kind="tomorrow_at",
                task_length=len(task),
                fallback_allowed=result.fallback_allowed,
            )
            return result

        result = Interpretation(
            response_text=(
                "Please tell me the reminder task and a future time, such as tomorrow at 9."
            )
        )
        log_event(
            logger,
            logging.DEBUG,
            "reminder_intent_rules_finished",
            matched=False,
            fallback_allowed=result.fallback_allowed,
            reason="unsupported_schedule_shape",
        )
        return result

    @staticmethod
    def _build(task: str, due_at: datetime, timezone_name: str) -> Interpretation:
        if not task:
            return Interpretation(
                response_text="What should I remind you about?",
                fallback_allowed=False,
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

    def interpret(self, text: str, *, timezone_name: str, now: datetime) -> Interpretation:
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
        )
        prompt = (
            "Return JSON only with exactly operation, task, and due_local. operation must be "
            "CREATE_ONE_TIME_REMINDER. due_local must be a timezone-naive ISO-8601 local "
            "datetime (without Z or an offset); the application applies the user's "
            f"validated timezone {timezone_name}. "
            f"Current local time is {local_now.isoformat()}. "
            "If the request is not a clear one-time reminder, return operation UNKNOWN."
        )
        try:
            response = httpx.post(
                self._url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": text},
                    ],
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            content = payload["choices"][0]["message"]["content"]
            data = json.loads(content) if isinstance(content, str) else content
            if not isinstance(data, dict) or data.get("operation") != "CREATE_ONE_TIME_REMINDER":
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
        ):
            log_event(
                logger,
                logging.ERROR,
                "reminder_intent_provider_failed",
                endpoint_host=endpoint_host,
                model=self._model,
                error_type="provider_or_schema_error",
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

    def interpret(self, text: str, *, timezone_name: str, now: datetime) -> Interpretation:
        result = self._rules.interpret(text, timezone_name=timezone_name, now=now)
        if result.intent is not None or self._model is None or not result.fallback_allowed:
            return result
        log_event(
            logger,
            logging.INFO,
            "reminder_intent_provider_fallback",
            reason="rules_did_not_match",
        )
        return self._model.interpret(text, timezone_name=timezone_name, now=now)
