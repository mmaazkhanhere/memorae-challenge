"""Pending reminder creation parsing, grounding, validation, and completion."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, cast
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from reminder_agent.features.intent.contracts import (
    CreateOneTimeReminderIntent,
    CreateRecurringReminderIntent,
    Interpretation,
    NeedsClarification,
    PendingReminder,
    PendingReminderFieldUpdate,
    merge_pending_reminder_updates,
)
from reminder_agent.features.intent.deterministic.patterns import (
    ACKNOWLEDGEMENTS as _ACKNOWLEDGEMENTS,
)
from reminder_agent.features.intent.deterministic.patterns import (
    CANCEL_FOLLOW_UPS as _CANCEL_FOLLOW_UPS,
)
from reminder_agent.features.intent.deterministic.patterns import (
    contains_multiple_tasks as _contains_multiple_tasks,
)
from reminder_agent.features.intent.deterministic.support import (
    Meridiem,
    resolve_pending_due_at,
)
from reminder_agent.features.intent.deterministic.support import (
    clock_label as _clock_label,
)
from reminder_agent.features.intent.deterministic.support import (
    clock_to_24 as _clock_to_24,
)
from reminder_agent.features.intent.deterministic.support import (
    date_from_day as _date_from_day,
)
from reminder_agent.features.intent.deterministic.support import (
    date_label as _date_label,
)
from reminder_agent.features.intent.deterministic.support import (
    normalise_meridiem as _normalise_meridiem,
)
from reminder_agent.features.intent.deterministic.support import (
    normalise_now as _normalise_now,
)
from reminder_agent.features.intent.deterministic.support import (
    ordinal_suffix as _ordinal_suffix,
)
from reminder_agent.features.intent.deterministic.support import (
    task_from_remainder as _task_from_remainder,
)
from reminder_agent.features.intent.deterministic.support import (
    valid_clock as _valid_clock,
)


class PendingIntentMixin:
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
        clock = self._clock.search(raw) or self._follow_up_clock.search(raw)
        day = self._day.search(raw)
        hour: int | None = context.hour
        minute = context.minute
        meridiem = context.meridiem
        date_local = context.date_local
        date_explicit = context.date_explicit
        if clock is not None:
            hour = int(clock.group("hour"))
            minute = int(clock.group("minute") or 0)
            meridiem = cast(Meridiem | None, _normalise_meridiem(clock.group("meridiem")))
            if not _valid_clock(hour, minute, meridiem):
                return self._invalid_time()
            if not context.date_explicit and meridiem and date_local is not None:
                local_now = _normalise_now(now).astimezone(ZoneInfo(context.timezone_name))
                if meridiem == "am" and date_local == local_now.date():
                    date_local += timedelta(days=1)
        elif normalized in {"am", "pm"}:
            meridiem = cast(Meridiem, normalized)
        if day is not None:
            local_now = _normalise_now(now).astimezone(ZoneInfo(context.timezone_name))
            date_local = _date_from_day(day, local_now)
            date_explicit = True

        remainder = raw
        if clock is not None:
            remainder = remainder.replace(clock.group(0), "", 1)
        elif normalized in {"am", "pm"}:
            remainder = ""
        if day is not None:
            remainder = remainder.replace(day.group(0), "", 1)
        candidate_task = _task_from_remainder(remainder)
        task = context.task
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
            local_today = _normalise_now(now).astimezone(ZoneInfo(pending.timezone_name)).date()
            pending = pending.model_copy(update={"date_local": local_today, "date_explicit": True})
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
                pending_hour = pending.hour
                zone = ZoneInfo(pending.timezone_name)
                local_now = _normalise_now(now).astimezone(zone)
                clock = _clock_label(pending.hour, pending.minute)
                options: list[tuple[str, date]] = []
                for candidate_meridiem in ("am", "pm"):
                    candidate_date = pending.date_local
                    due_local = datetime.combine(
                        candidate_date,
                        time(_clock_to_24(pending_hour, candidate_meridiem), pending.minute),
                        tzinfo=zone,
                    )
                    if due_local <= local_now and not pending.date_explicit:
                        candidate_date += timedelta(days=1)
                        due_local = datetime.combine(
                            candidate_date,
                            time(_clock_to_24(pending_hour, candidate_meridiem), pending.minute),
                            tzinfo=zone,
                        )
                    if due_local > local_now:
                        options.append((candidate_meridiem, candidate_date))
                options.sort(key=lambda option: (option[1], _clock_to_24(pending_hour, option[0])))
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

        if _contains_multiple_tasks(pending.task or ""):
            return self._multiple_target_clarification(pending.timezone_name)

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
                start_local = datetime.combine(
                    candidate_date, time(recurrence_hour, pending.minute)
                )
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
        return self._build(pending.task or "", due_local.astimezone(UTC), pending.timezone_name)

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
            self.interpret(command, timezone_name=timezone_name, now=now) for command in commands
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

    def _build(self, task: str, due_at: datetime, timezone_name: str) -> Interpretation:
        if _contains_multiple_tasks(task):
            return self._multiple_target_clarification(timezone_name)
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
                        meridiem=cast(Meridiem, meridiem),
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


_GROUNDING_CLOCK = re.compile(
    r"\bat\s+(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*"
    r"(?P<meridiem>a\.m\.|p\.m\.|am|pm)?\b",
    re.IGNORECASE,
)
_GROUNDING_FOLLOW_UP_CLOCK = re.compile(
    r"(?<!\w)(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*"
    r"(?P<meridiem>a\.m\.|p\.m\.|am|pm)?(?=\s|$|[,.])",
    re.IGNORECASE,
)
_GROUNDING_DAY = re.compile(
    r"\b(?P<day>today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
_GROUNDING_TASK_CORRECTION = re.compile(
    r"^\s*(?:change|update|replace)\s+(?:the\s+)?(?:task|reminder|it)\s+"
    r"(?:to|with)\s+(?P<task>.+?)\s*$",
    re.IGNORECASE,
)


def restrict_to_explicit_field_updates(
    text: str,
    proposed: PendingReminderFieldUpdate,
    *,
    context: PendingReminder,
    now: datetime,
    allow_task_replacement: bool = False,
    allow_semantic_field_updates: bool = False,
) -> PendingReminderFieldUpdate:
    """Keep only model-proposed changes grounded in a pending reply."""

    raw = text.strip()
    clock = _GROUNDING_CLOCK.search(raw) or _GROUNDING_FOLLOW_UP_CLOCK.search(raw)
    day = _GROUNDING_DAY.search(raw)
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
    elif proposed.meridiem is not None and re.sub(r"^[ .,!?]+|[ .,!?]+$", "", raw.lower()) in {
        "am",
        "pm",
    }:
        if proposed.meridiem == _normalise_meridiem(raw):
            permitted["meridiem"] = proposed.meridiem

    if day is not None:
        local_now = _normalise_now(now).astimezone(ZoneInfo(context.timezone_name))
        expected_date = _date_from_day(day, local_now)
        if proposed.date_local == expected_date:
            permitted["date_local"] = proposed.date_local
            permitted["date_explicit"] = True

    semantic_updates_are_grounded = allow_semantic_field_updates and proposed.certainty == "CLEAR"
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
            and (proposed.meridiem is None or 1 <= proposed.hour <= 12)
        ):
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
            if value is not None and _evidence_is_grounded(proposed.evidence.get(field), raw):
                permitted[field] = value

    remainder = raw
    if clock is not None:
        remainder = remainder.replace(clock.group(0), "", 1)
    if day is not None:
        remainder = remainder.replace(day.group(0), "", 1)
    supplied_task = _grounding_task_from_remainder(remainder)
    correction = _GROUNDING_TASK_CORRECTION.match(remainder)
    if proposed.task:
        if context.task is None and supplied_task and _same_task(proposed.task, supplied_task):
            permitted["task"] = proposed.task
        elif correction is not None:
            corrected_task = _grounding_task_from_remainder(correction.group("task"))
            if corrected_task and _same_task(proposed.task, corrected_task):
                permitted["task"] = proposed.task
        elif (
            allow_task_replacement
            and semantic_updates_are_grounded
            and _evidence_is_grounded(proposed.evidence.get("task"), raw)
        ):
            permitted["task"] = proposed.task
    return PendingReminderFieldUpdate.model_validate(permitted)


def _grounding_task_from_remainder(remainder: str) -> str:
    task = remainder.strip(" .,:;")
    task = re.sub(r"^(?:(?:on|to|for|that|about)\s+)+", "", task, flags=re.IGNORECASE)
    return " ".join(task.split())


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
        if all(
            close_enough(left, right)
            for left, right in zip(evidence_tokens, candidate, strict=True)
        ):
            return True
        remaining = list(candidate)
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
                break
            remaining.pop(match_index)
        else:
            return True
    return False


def _same_task(left: str, right: str) -> bool:
    return " ".join(left.lower().split()) == " ".join(right.lower().split())
