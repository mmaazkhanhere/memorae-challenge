"""Pure deterministic parsing for reminder reschedule requests."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from reminder_agent.features.reminders.recurrence import resolve_local_datetime
from reminder_agent.features.reminders.reschedule.messages import (
    reschedule_message,
)
from reminder_agent.features.reminders.reschedule.models import (
    PendingRescheduleOperation,
    PendingRescheduleProposal,
    RescheduleRequest,
    RescheduleScope,
)

_COMMAND = re.compile(
    r"^\s*(?:please\s+|can\s+you\s+|could\s+you\s+)?"
    r"(?:move|reschedule|change|shift|push|postpone|delay)\s+(?:the\s+)?(?P<target>.+?)\s+"
    r"(?P<separator>to|until|for|at)\s+(?P<when>.+?)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_COMMAND_GREEDY = re.compile(
    r"^\s*(?:please\s+|can\s+you\s+|could\s+you\s+)?"
    r"(?:move|reschedule|change|shift|push|postpone|delay)\s+(?:the\s+)?(?P<target>.+)\s+"
    r"(?P<separator>to|until|for)\s+(?P<when>.+?)\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_INCOMPLETE_COMMAND = re.compile(
    r"^\s*(?:i\s+(?:want|need)\s+(?:you\s+to\s+)?|can\s+you\s+|could\s+you\s+)?"
    r"(?:please\s+)?(?:move|reschedule|change|shift|push|postpone|delay)\s+"
    r"(?:the\s+)?(?P<target>.+?)\s*(?:reminder)?\s*[.!?]*\s*$",
    re.IGNORECASE,
)
_WHEN = re.compile(
    r"^(?:(?P<day>today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"\d{4}-\d{1,2}-\d{1,2})\s+)?(?:at\s+)?(?P<hour>\d{1,2})"
    r"(?::(?P<minute>\d{2}))?\s*(?P<meridiem>a\.?m\.?|p\.?m\.?|am|pm)?$",
    re.IGNORECASE,
)
_OLD_TIME = re.compile(
    r"\s+from\s+(?:(?:today|tomorrow|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday)\s+)?(?:at\s+)?\d{1,2}(?::\d{2})?\s*"
    r"(?:a\.?m\.?|p\.?m\.?|am|pm)\b",
    re.IGNORECASE,
)
_ORDINALS = {
    "first": 0,
    "one": 0,
    "1": 0,
    "second": 1,
    "two": 1,
    "2": 1,
    "third": 2,
    "three": 2,
    "3": 2,
}
_STOP_WORDS = {
    "a",
    "an",
    "the",
    "reminder",
    "reminders",
    "this",
    "occurrence",
    "series",
    "future",
    "all",
    "my",
    "please",
    "one",
    "first",
    "second",
    "third",
    "1",
    "2",
    "3",
}
_DAY_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_OCCURRENCE_SELECTOR = re.compile(
    r"\b(?:(?P<only>only)\s+)?(?P<day>today|tomorrow|monday|tuesday|wednesday|"
    r"thursday|friday|saturday|sunday)'s\s+",
    re.IGNORECASE,
)
_UNSUPPORTED_SERIES_CADENCE = re.compile(
    r"\b(?:on\s+)?(?:the\s+)?(?:\d{1,2}(?:st|nd|rd|th)|first|second|third|"
    r"fourth|last|january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\b",
    re.IGNORECASE,
)


def is_reschedule_command(text: str) -> bool:
    return bool(
        re.match(
            r"^\s*(?:please\s+)?"
            r"(?:move|reschedule|change|shift|push|postpone|delay)\b",
            text,
            re.I,
        )
    )


def has_unsupported_series_cadence_change(text: str) -> bool:
    """Recognize calendar-cadence requests that this narrow API cannot model.

    Named weekdays are supported for weekly series.  Month/day and ordinal
    requests must not reach an LLM proposal that can only express a timestamp,
    because that would turn a cadence request into a silent time-only update.
    """

    if not is_reschedule_command(text) or not re.search(
        r"\b(?:series|all\s+future|every)\b", text, re.I
    ):
        return False
    destination = re.split(r"\s+(?:to|until|for|at)\s+", text, maxsplit=1, flags=re.I)
    return len(destination) == 2 and bool(_UNSUPPORTED_SERIES_CADENCE.search(destination[1]))


def scope_from_text(text: str) -> RescheduleScope | None:
    normalized = text.lower()
    if normalized.strip() in {"all", "all of them", "everything"}:
        return RescheduleScope.SERIES
    if normalized.strip() in {"all future ones", "all the future ones"}:
        return RescheduleScope.SERIES
    if re.search(
        r"\b(?:whole\s+)?series\b|\ball\s+(?:future\s+)?(?:reminders?|occurrences?)\b|"
        r"\ball\s+future\b.*\b(?:reminders?|occurrences?)\b|"
        r"\bupdate\s+all\b|\bevery\b",
        normalized,
    ):
        return RescheduleScope.SERIES
    if re.search(
        r"\b(?:next|upcoming|coming)\s+(?:occurrence|reminder|one)\b|"
        r"\bonly\s+(?:the\s+)?(?:next|upcoming|coming)(?:\s+(?:occurrence|reminder|one))?\b|"
        r"\b(?:only\s+)?(?:this|coming)\s+one\b|\bthis\s+(?:occurrence|reminder)\b",
        normalized,
    ):
        return RescheduleScope.OCCURRENCE
    if normalized.strip() in {"just tomorrow", "only tomorrow", "just the next one"}:
        return RescheduleScope.OCCURRENCE
    return None


def parse_date(raw: str | None, *, now: datetime, zone: ZoneInfo) -> date:
    local_now = now.astimezone(zone)
    if raw is None:
        return local_now.date()
    value = raw.lower()
    if value == "today":
        return local_now.date()
    if value == "tomorrow":
        return local_now.date() + timedelta(days=1)
    if value in _DAY_INDEX:
        delta = (_DAY_INDEX[value] - local_now.weekday()) % 7
        return local_now.date() + timedelta(days=delta or 7)
    return date.fromisoformat(value)


def extract_target_and_occurrence(
    target: str, *, timezone_name: str, now: datetime
) -> tuple[str, RescheduleScope | None, date | None]:
    """Remove an explicit occurrence selector before matching the task."""

    target = _OLD_TIME.sub(" ", target)
    scope = scope_from_text(target)
    occurrence_date = None
    selector = _OCCURRENCE_SELECTOR.search(target)
    if selector is not None:
        occurrence_date = parse_date(
            selector.group("day"), now=now, zone=ZoneInfo(timezone_name)
        )
        # A possessive day selector identifies one concrete recurrence
        # instance; "only" reinforces rather than creates that scope.
        scope = RescheduleScope.OCCURRENCE
        target = target[: selector.start()] + target[selector.end() :]
    if scope is RescheduleScope.SERIES:
        target = re.sub(r"^(?:every\s+)?(?:daily\s+)?", "", target, flags=re.I)
    target = re.sub(
        r"\b(?:all\s+future\s+reminders?|all\s+reminders?|this\s+occurrence|this\s+reminder|"
        r"the\s+whole\s+series|the\s+series|whole\s+series|series|next\s+occurrence|"
        r"upcoming\s+reminder|only\s+this\s+one|only\s+the\s+next|coming\s+one|occurrence)\b",
        " ",
        target,
        flags=re.I,
    )
    return " ".join(target.split()).strip(" .,!?"), scope, occurrence_date


def parse_reschedule_request(
    text: str, *, timezone_name: str, now: datetime
) -> RescheduleRequest | str:
    """Parse the deliberately small, unambiguous VS-09 command language."""

    match = _COMMAND.match(text)
    if match is None:
        incomplete = _INCOMPLETE_COMMAND.match(text)
        if incomplete is None:
            return reschedule_message("missing_reminder_target")
        raw_target = " ".join(incomplete.group("target").split())
        # Leave requests that contain an explicit relative-time expression to
        # the route-specific extractor.  This branch is only for a grounded
        # reminder target with no destination time at all.
        if re.search(
            r"\b(?:to|until|for|at)\s+(?:today|tomorrow|monday|tuesday|wednesday|"
            r"thursday|friday|saturday|sunday|\d)|\bback\s+by\b",
            raw_target,
            re.IGNORECASE,
        ):
            return reschedule_message("missing_reminder_target")
        target = raw_target
        target = re.sub(r"\s+reminders?$", "", target, flags=re.IGNORECASE).strip()
        target, scope, occurrence_date = extract_target_and_occurrence(
            target, timezone_name=timezone_name, now=now
        )
        if not target:
            return reschedule_message("missing_reminder_target")
        return RescheduleRequest(
            target_text=target,
            scope=scope,
            occurrence_date=occurrence_date,
            source_text=text,
        )
    target = " ".join(match.group("target").split())
    target = re.sub(r"^my\s+", "", target, flags=re.I)
    target = re.sub(r"^(?:my\s+)?(?:a\s+)?reminders?\s+to\s+", "", target, flags=re.I)
    if not target:
        return reschedule_message("missing_reminder_target")
    target, scope, occurrence_date = extract_target_and_occurrence(
        target, timezone_name=timezone_name, now=now
    )
    if not target:
        return reschedule_message("missing_reminder_target")

    when = _WHEN.match(match.group("when"))
    if when is None:
        fallback_match = _COMMAND_GREEDY.match(text)
        if fallback_match is not None:
            fallback_when = _WHEN.match(fallback_match.group("when"))
            if fallback_when is not None:
                match = fallback_match
                target = " ".join(match.group("target").split())
                target = re.sub(r"^my\s+", "", target, flags=re.I)
                target = re.sub(
                    r"^(?:my\s+)?(?:a\s+)?reminders?\s+to\s+", "", target, flags=re.I
                )
                target, scope, occurrence_date = extract_target_and_occurrence(
                    target, timezone_name=timezone_name, now=now
                )
                when = fallback_when
    if when is None:
        return "Please give the new time as a future date and time, for example tomorrow at 10 AM."
    hour = int(when.group("hour"))
    minute = int(when.group("minute") or 0)
    meridiem = (when.group("meridiem") or "").lower().replace(".", "")
    if minute > 59 or hour > (12 if meridiem else 23) or (meridiem and hour == 0):
        return "Please provide a valid clock time."
    if not meridiem and 1 <= hour <= 12:
        return f"Do you mean {hour} AM or {hour} PM for the new reminder time?"
    if meridiem == "pm" and hour != 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    zone = ZoneInfo(timezone_name)
    try:
        local_date = parse_date(when.group("day"), now=now, zone=zone)
        local_due = resolve_local_datetime(local_date, time(hour, minute), zone)
    except (TypeError, ValueError):
        return "Please provide a valid future date and time."
    due_at = local_due.astimezone(UTC)
    if when.group("day") is None and due_at <= now.astimezone(UTC):
        local_due = resolve_local_datetime(local_date + timedelta(days=1), time(hour, minute), zone)
        due_at = local_due.astimezone(UTC)
    if due_at <= now.astimezone(UTC):
        return "Please choose a future time for the reminder."
    return RescheduleRequest(
        target_text=target,
        due_at=due_at,
        date_explicit=when.group("day") is not None,
        scope=scope,
        occurrence_date=occurrence_date,
        destination_weekday=_DAY_INDEX.get((when.group("day") or "").lower()),
    )


def parse_pending_reschedule_parameters(
    text: str,
    *,
    timezone_name: str,
    now: datetime,
    destination_date: date | None = None,
) -> PendingRescheduleProposal | None:
    """Recognize only whole, unambiguous parameter-only follow-ups."""

    normalized = text.strip().rstrip(".!?")
    bare_time = re.fullmatch(
        r"(?:at\s+)?\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?|am|pm)",
        normalized,
        re.IGNORECASE,
    )
    if bare_time is not None:
        parsed = parse_reschedule_request(
            (
                f"Move placeholder to {destination_date.isoformat()} at {normalized}"
                if destination_date is not None
                else f"Move placeholder to {normalized}"
            ),
            timezone_name=timezone_name,
            now=now,
        )
        if isinstance(parsed, RescheduleRequest):
            return PendingRescheduleProposal(
                operation=PendingRescheduleOperation.UPDATE_PENDING_PARAMETERS,
                due_at=parsed.due_at,
                offset_minutes=parsed.offset_minutes,
                date_explicit=parsed.date_explicit,
            )
    date_only = re.fullmatch(
        r"(?:no,?\s+)?i\s+meant\s+"
        r"(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        normalized,
        re.IGNORECASE,
    )
    if date_only is not None:
        try:
            value = parse_date(date_only.group(1), now=now, zone=ZoneInfo(timezone_name))
        except ValueError:
            return None
        return PendingRescheduleProposal(
            operation=PendingRescheduleOperation.UPDATE_PENDING_PARAMETERS,
            destination_date=value,
            date_explicit=True,
        )
    match = re.fullmatch(
        r"(?:actually,?\s+)?(?:make|move|change)\s+(?:it|that)(?:\s+to)?\s+(.+)",
        normalized,
        re.IGNORECASE,
    )
    if match is None:
        return None
    parsed = parse_reschedule_request(
        f"Move placeholder to {match.group(1)}", timezone_name=timezone_name, now=now
    )
    if not isinstance(parsed, RescheduleRequest):
        return None
    return PendingRescheduleProposal(
        operation=PendingRescheduleOperation.UPDATE_PENDING_PARAMETERS,
        due_at=parsed.due_at,
        offset_minutes=parsed.offset_minutes,
        date_explicit=parsed.date_explicit,
        scope=parsed.scope,
    )


__all__ = [
    "extract_target_and_occurrence",
    "has_unsupported_series_cadence_change",
    "is_reschedule_command",
    "parse_date",
    "parse_pending_reschedule_parameters",
    "parse_reschedule_request",
    "scope_from_text",
]
