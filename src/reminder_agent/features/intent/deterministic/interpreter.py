"""Deterministic, channel-neutral reminder intent parsing."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, time, timedelta
from typing import cast
from zoneinfo import ZoneInfo

from reminder_agent.features.intent.contracts import (
    IntentRoute,
    Interpretation,
    PendingReminder,
)
from reminder_agent.features.intent.deterministic.listing import (
    ListingIntentMixin as _ListingIntentMixin,
)
from reminder_agent.features.intent.deterministic.patterns import (
    ACKNOWLEDGEMENTS as _ACKNOWLEDGEMENTS,
)
from reminder_agent.features.intent.deterministic.patterns import (
    CANCEL_FOLLOW_UPS as _CANCEL_FOLLOW_UPS,
)
from reminder_agent.features.intent.deterministic.patterns import (
    CLOCK,
    DAY,
    DAY_PART,
    FOLLOW_UP_CLOCK,
    LIST_RANGE,
    LIST_REQUEST,
    MONTHLY_ORDINAL_RECURRENCE,
    MONTHLY_RECURRENCE,
    MONTHS,
    MULTIPLE_REQUEST_SEPARATOR,
    ORDINAL_NUMBERS,
    RECURRENCE,
    RECURRENCE_AFTER_TASK,
    RELATIVE,
    SUMMARY_REQUEST,
    TOMORROW,
    YEARLY_RECURRENCE,
    extract_listing_task_query,
)
from reminder_agent.features.intent.deterministic.patterns import (
    COMPOUND_REQUEST as _COMPOUND_REQUEST,
)
from reminder_agent.features.intent.deterministic.patterns import (
    REQUIRES_INTERPRETATION as _REQUIRES_INTERPRETATION,
)
from reminder_agent.features.intent.deterministic.patterns import (
    TASK_CORRECTION as _TASK_CORRECTION,
)
from reminder_agent.features.intent.deterministic.patterns import (
    contains_multiple_tasks as _contains_multiple_tasks,
)
from reminder_agent.features.intent.deterministic.recurrence import (
    RecurrenceIntentMixin as _RecurrenceIntentMixin,
)
from reminder_agent.features.intent.deterministic.support import (
    WEEKDAYS,
    Meridiem,
    resolve_pending_due_at,
)
from reminder_agent.features.intent.deterministic.support import (
    clock_to_24 as _clock_to_24,
)
from reminder_agent.features.intent.deterministic.support import (
    date_from_day as _date_from_day,
)
from reminder_agent.features.intent.deterministic.support import (
    meridiem_from_day_part as _meridiem_from_day_part,
)
from reminder_agent.features.intent.deterministic.support import (
    next_recurring_start_local as _next_recurring_start_local,
)
from reminder_agent.features.intent.deterministic.support import (
    normalise_meridiem as _normalise_meridiem,
)
from reminder_agent.features.intent.deterministic.support import (
    normalise_now as _normalise_now,
)
from reminder_agent.features.intent.deterministic.support import (
    number as _number,
)
from reminder_agent.features.intent.deterministic.support import (
    strip_repeated_temporal_token as _strip_repeated_temporal_token,
)
from reminder_agent.features.intent.deterministic.support import (
    task_from_remainder as _task_from_remainder,
)
from reminder_agent.features.intent.deterministic.support import (
    valid_clock as _valid_clock,
)
from reminder_agent.features.intent.pending.reminder_creation import (
    PendingIntentMixin as _PendingIntentMixin,
)
from reminder_agent.features.memory.messages import save_memory_message
from reminder_agent.infrastructure.observability import get_logger, log_event
from reminder_agent.shared.creation_routing import (
    REMINDER_COMMAND_PREFIX,
    is_disambiguated_reminder_command,
)
from reminder_agent.shared.memory_contracts import SaveMemoryFact
from reminder_agent.shared.memory_text import extract_memory_content

logger = get_logger()

_WEEKDAYS = WEEKDAYS

__all__ = [
    "RuleBasedIntentInterpreter",
    "extract_listing_task_query",
    "resolve_pending_due_at",
    "_normalise_now",
    "_next_recurring_start_local",
    "_COMPOUND_REQUEST",
    "_TASK_CORRECTION",
]


class RuleBasedIntentInterpreter(
    _ListingIntentMixin,
    _RecurrenceIntentMixin,
    _PendingIntentMixin,
):
    """Conservative parser that refuses to guess incomplete date/time requests."""

    _prefix = REMINDER_COMMAND_PREFIX
    _relative = RELATIVE
    _tomorrow = TOMORROW
    _clock = CLOCK
    _follow_up_clock = FOLLOW_UP_CLOCK
    _day = DAY
    _day_part = DAY_PART
    _recurrence = RECURRENCE
    _recurrence_after_task = RECURRENCE_AFTER_TASK
    _monthly_recurrence = MONTHLY_RECURRENCE
    _yearly_recurrence = YEARLY_RECURRENCE
    _monthly_ordinal_recurrence = MONTHLY_ORDINAL_RECURRENCE
    _ordinal_numbers = ORDINAL_NUMBERS
    _months = MONTHS
    _multiple_request_separator = MULTIPLE_REQUEST_SEPARATOR
    _weekdays = WEEKDAYS
    _list_request = LIST_REQUEST
    _summary_request = SUMMARY_REQUEST
    _list_range = LIST_RANGE

    def interpret(
        self,
        text: str,
        *,
        selected_route: IntentRoute | None = None,
        timezone_name: str,
        now: datetime,
        context: PendingReminder | None = None,
    ) -> Interpretation:
        # Telegram users sometimes send copied questions with surrounding
        # quotation marks. They are presentation noise, not intent content.
        raw = text.strip().strip('"“”')
        route = selected_route or (
            IntentRoute.UPDATE_PENDING if context is not None else IntentRoute.NEW_REQUEST
        )
        if route is IntentRoute.UNCERTAIN:
            return Interpretation(
                response_text="IÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢m not sure what you want to do. Please clarify.",
                fallback_allowed=False,
            )
        if route is IntentRoute.UPDATE_PENDING and context is None:
            return Interpretation(
                response_text="There is no unfinished reminder to update.",
                fallback_allowed=False,
            )
        if route is IntentRoute.SUMMARY_REQUEST:
            summary = self._interpret_summary(raw, timezone_name=timezone_name, now=now)
            if summary is not None:
                return summary
            return Interpretation(
                response_text="Please ask for pending, remaining, completed, or overdue reminders.",
                fallback_allowed=False,
            )
        if route is IntentRoute.COMPLETE_REQUEST:
            # Completion is resolved from trusted Telegram delivery context;
            # the extractor must never select an occurrence or invent an ID.
            return Interpretation(fallback_allowed=False)
        if route is IntentRoute.SNOOZE_REQUEST:
            from reminder_agent.features.reminders.snooze.messages import snooze_message
            from reminder_agent.features.reminders.snooze.operations import parse_snooze_duration

            duration = parse_snooze_duration(raw)
            if duration is not None:
                return Interpretation(
                    snooze_duration_seconds=duration.duration_seconds,
                    fallback_allowed=False,
                )
            return Interpretation(
                response_text=snooze_message("missing_duration"),
                fallback_allowed=True,
            )
        if route is IntentRoute.SAVE_MEMORY:
            content = extract_memory_content(raw)
            if content is None:
                return Interpretation(
                    response_text=save_memory_message("clarification"),
                    fallback_allowed=True,
                )
            return Interpretation(
                memory_fact=SaveMemoryFact(content=content),
                fallback_allowed=False,
            )
        if route is IntentRoute.RETRIEVE_MEMORY:
            # Deterministic routing has already established this as a lookup,
            # but no deterministic subject was available. The existing typed
            # extractor may propose that subject; it never supplies an answer.
            return Interpretation(fallback_allowed=True)
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

        summary = self._interpret_summary(raw, timezone_name=timezone_name, now=now)
        if summary is not None:
            return summary

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
            calendar = self._interpret_calendar_recurrence(
                raw, timezone_name=timezone_name, now=now
            )
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
            partial_weekly = self._interpret_partial_weekly(
                raw,
                timezone_name=timezone_name,
                now=_normalise_now(now),
            )
            if partial_weekly is not None:
                return partial_weekly
            # Keep bare clock answers and social acknowledgements deterministic,
            # but allow the provider to interpret complex natural language even
            # when it does not contain the literal word "remind".
            simple_time = self._follow_up_clock.fullmatch(raw.strip(" .,!"))
            simple_ack = raw.lower().strip(" .,!?") in (_ACKNOWLEDGEMENTS | _CANCEL_FOLLOW_UPS)
            return Interpretation(
                response_text=("Please tell me the reminder task and a future time."),
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
        partial_weekly = self._interpret_partial_weekly(
            without_prefix,
            timezone_name=timezone_name,
            now=reference,
        )
        if partial_weekly is not None:
            return partial_weekly

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
                f"{before_time} {after_time}"
                if before_time.strip() and after_time.strip()
                else before_time
                if before_time.strip()
                else after_time
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
            meridiem = meridiem or _meridiem_from_day_part(
                tomorrow.groupdict().get("day_part")
            )
            if not _valid_clock(hour, minute, meridiem):
                return self._invalid_time()
            local_date = reference.astimezone(zone).date() + timedelta(days=1)
            before_time = without_prefix[: tomorrow.start()]
            after_time = without_prefix[tomorrow.end() :]
            task = _task_from_remainder(
                f"{before_time} {after_time}"
                if before_time.strip() and after_time.strip()
                else before_time
                if before_time.strip()
                else after_time
            )
            task = _strip_repeated_temporal_token(task, "tomorrow")
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
                    fallback_allowed=not is_disambiguated_reminder_command(raw),
                )
            due_local = datetime.combine(
                local_date,
                time(_clock_to_24(hour, meridiem), minute),
                tzinfo=zone,
            )
            return self._build(task, due_local.astimezone(UTC), timezone_name)

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
            try:
                date_local = (
                    _date_from_day(day, local_now)
                    if day
                    else local_now.date()
                    if clock is not None
                    else None
                )
            except ValueError:
                return Interpretation(
                    response_text="Please provide a valid future reminder date.",
                    fallback_allowed=False,
                )
            date_explicit = day is not None
            clock_hour: int | None = int(clock.group("hour")) if clock else None
            minute = int(clock.group("minute") or 0) if clock else 0
            meridiem = (
                cast(Meridiem | None, _normalise_meridiem(clock.group("meridiem")))
                if clock
                else None
            )
            meridiem = meridiem or _meridiem_from_day_part(
                clock.groupdict().get("day_part") if clock else None
            )
            if clock is not None and not _valid_clock(clock_hour, minute, meridiem):
                return self._invalid_time()
            task_source = without_prefix
            if day is not None:
                task_source = task_source.replace(day.group(0), "", 1)
            if day_part is not None:
                task_source = task_source.replace(day_part.group(0), "", 1)
            if clock is not None:
                task_source = task_source.replace(clock.group(0), "", 1)
            task = _task_from_remainder(task_source)
            if day is not None:
                task = _strip_repeated_temporal_token(task, day.group("day"))
            pending = PendingReminder(
                task=task or None,
                date_local=date_local,
                hour=clock_hour,
                minute=minute,
                meridiem=meridiem,
                timezone_name=timezone_name,
                date_explicit=date_explicit,
            )
            return self._pending_result(
                pending,
                now=now,
                fallback_allowed=not is_disambiguated_reminder_command(raw),
            )

        # A command with an ordinary task but no schedule is still useful
        # deterministic input.  Store it as pending rather than asking a model
        # to reconstruct the same partial reminder.
        task = _task_from_remainder(without_prefix)
        if _contains_multiple_tasks(task):
            return self._multiple_target_clarification(timezone_name)
        return self._pending_result(
            PendingReminder(task=task or None, timezone_name=timezone_name),
            now=now,
            fallback_allowed=not is_disambiguated_reminder_command(raw),
        )
