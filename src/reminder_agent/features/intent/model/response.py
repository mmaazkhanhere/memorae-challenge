"""Validation and conversion of model provider responses."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, time
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo

from reminder_agent.features.intent.contracts import (
    CreateOneTimeReminderIntent,
    CreateRecurringReminderIntent,
    IntentRoute,
    Interpretation,
    ListRemindersIntent,
    NeedsClarification,
    PendingReminder,
    PendingReminderBatch,
    PendingReminderFieldUpdate,
    ReminderSummaryIntent,
    normalize_task_query,
)
from reminder_agent.features.intent.model.prompts import OPERATION_ENVELOPE_KEYS
from reminder_agent.features.intent.policy import OUT_OF_SCOPE_RESPONSE
from reminder_agent.shared.memory_contracts import RetrieveMemoryQuery, SaveMemoryFact

SummaryMetric = Literal[
    "PENDING_COUNT",
    "TODAY_REMAINING",
    "COMPLETED_COUNT",
    "TODAY_COMPLETED",
    "OVERDUE_COUNT",
]
RecurringFrequency = Literal["WEEKLY", "MONTHLY", "YEARLY"]
_WEEKDAY_ALIASES = {
    "MONDAY": 0,
    "MON": 0,
    "TUESDAY": 1,
    "TUE": 1,
    "TUES": 1,
    "WEDNESDAY": 2,
    "WED": 2,
    "THURSDAY": 3,
    "THU": 3,
    "THURS": 3,
    "FRIDAY": 4,
    "FRI": 4,
    "SATURDAY": 5,
    "SAT": 5,
    "SUNDAY": 6,
    "SUN": 6,
}
_TIME_PATTERN = re.compile(
    r"^(?P<hour>\d{1,2})(?::(?P<minute>\d{1,2}))?\s*(?P<meridiem>AM|PM)?$"
)


def _provider_local_datetime(value: str, *, timezone_name: str) -> datetime:
    """Parse a model datetime and normalize accidental offset-bearing output.

    The contract asks providers for a naive local datetime, but compatible
    models occasionally include ``Z`` or an explicit offset.  Treat that as
    an absolute instant and convert it into the user's local wall-clock time
    before the existing timezone-aware domain conversion.
    """

    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is not None and parsed.utcoffset() is not None:
        parsed = parsed.astimezone(ZoneInfo(timezone_name)).replace(tzinfo=None)
    return parsed


def _normalise_weekdays(value: Any) -> list[int] | None:
    """Normalize provider weekday aliases to the canonical Monday=0 form."""

    if value is None:
        return None
    values = value if isinstance(value, list) else [value]
    weekdays: list[int] = []
    for item in values:
        if isinstance(item, bool):
            raise ValueError("recurring weekday was not an integer or name")
        if isinstance(item, int):
            weekday = item
        elif isinstance(item, str):
            weekday = _WEEKDAY_ALIASES.get(item.strip().upper())
            if weekday is None:
                raise ValueError("recurring weekday was not recognized")
        else:
            raise ValueError("recurring weekday was not an integer or name")
        if weekday < 0 or weekday > 6:
            raise ValueError("recurring weekday was outside 0..6")
        if weekday not in weekdays:
            weekdays.append(weekday)
    return weekdays


def _normalise_time_fields(data: dict[str, Any]) -> tuple[int, int]:
    """Normalize canonical hour/minute or a provider ``time`` alias."""

    if "hour" in data:
        hour_value = data["hour"]
        minute_value = data.get("minute", 0)
        if isinstance(hour_value, bool) or not isinstance(hour_value, int):
            raise ValueError("recurring hour was not an integer")
        if isinstance(minute_value, bool) or not isinstance(minute_value, int):
            raise ValueError("recurring minute was not an integer")
        hour, minute = hour_value, minute_value
    else:
        raw_time = data.get("time")
        if isinstance(raw_time, (int, float)) and not isinstance(raw_time, bool):
            hour, minute = int(raw_time), round((float(raw_time) - int(raw_time)) * 60)
        elif isinstance(raw_time, str):
            match = _TIME_PATTERN.fullmatch(raw_time.strip().upper())
            if match is None:
                raise ValueError("recurring time was not recognized")
            hour = int(match.group("hour"))
            minute = int(match.group("minute") or 0)
            meridiem = match.group("meridiem")
            if meridiem is not None:
                if hour < 1 or hour > 12:
                    raise ValueError("12-hour recurring time was outside 1..12")
                hour = hour % 12 + (12 if meridiem == "PM" else 0)
        else:
            raise ValueError("recurring intent is missing hour")
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        raise ValueError("recurring time was outside the valid range")
    return hour, minute


def _normalise_listing_time(value: Any, field_name: str) -> time | None:
    """Normalize an optional listing time filter to a local wall-clock time."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} was not a time string")
    match = _TIME_PATTERN.fullmatch(value.strip().upper())
    if match is None:
        raise ValueError(f"{field_name} was not recognized")
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    meridiem = match.group("meridiem")
    if meridiem is not None:
        if hour < 1 or hour > 12:
            raise ValueError(f"{field_name} was outside the 12-hour range")
        hour = hour % 12 + (12 if meridiem == "PM" else 0)
    if hour > 23 or minute > 59:
        raise ValueError(f"{field_name} was outside the valid range")
    return time(hour, minute)


def decode_response(content: str | dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Decode JSON content and normalize a one-key operation envelope."""

    data = json.loads(content) if isinstance(content, str) else content
    if not isinstance(data, dict):
        raise ValueError("provider response was not an object")
    envelope_operation: str | None = None
    if "operation" not in data:
        response_type = data.get("type")
        if isinstance(response_type, str):
            normalized_type = response_type.strip().upper()
            if normalized_type in {"CLARIFICATION", "NEEDS_CLARIFICATION"}:
                data = {**data, "operation": "NEEDS_CLARIFICATION"}
                data.pop("type", None)
            elif normalized_type in OPERATION_ENVELOPE_KEYS:
                data = {**data, "operation": normalized_type}
                data.pop("type", None)
        envelope_keys = [
            key
            for key, value in data.items()
            if key in OPERATION_ENVELOPE_KEYS and isinstance(value, dict)
        ]
        if len(envelope_keys) == 1:
            envelope_operation = envelope_keys[0]
            envelope_body = data[envelope_operation]
            data = {**envelope_body, "operation": envelope_operation}
    return data, envelope_operation


def interpret_memory_response(
    data: dict[str, Any],
    *,
    route: IntentRoute,
    model: str,
    logger: logging.Logger,
    log_event_fn: Callable[..., Any],
) -> Interpretation | None:
    """Validate save/retrieve memory responses for the selected route."""

    operation = data.get("operation")
    if route is IntentRoute.SAVE_MEMORY:
        if operation != "SAVE_MEMORY":
            if operation == "NEEDS_CLARIFICATION":
                if set(data) - {"operation", "question"}:
                    raise ValueError("memory clarification has unexpected fields")
                question = data.get("question")
                if not isinstance(question, str) or not question.strip():
                    raise ValueError("missing memory clarification question")
                return Interpretation(response_text=question.strip(), fallback_allowed=False)
            raise ValueError("provider returned an unsupported memory operation")
        if set(data) - {"operation", "content"}:
            raise ValueError("memory intent has unexpected fields")
        memory_fact = SaveMemoryFact.model_validate(data)
        log_event_fn(logger, logging.INFO, "memory_extraction_completed", model=model,
                     content_length=len(memory_fact.content))
        return Interpretation(memory_fact=memory_fact, fallback_allowed=False)
    if route is IntentRoute.RETRIEVE_MEMORY:
        if operation != "RETRIEVE_MEMORY":
            if operation == "NEEDS_CLARIFICATION":
                if set(data) - {"operation", "question"}:
                    raise ValueError("memory lookup clarification has unexpected fields")
                question = data.get("question")
                if not isinstance(question, str) or not question.strip():
                    raise ValueError("missing memory lookup clarification question")
                return Interpretation(response_text=question.strip(), fallback_allowed=False)
            raise ValueError("provider returned an unsupported memory lookup operation")
        if set(data) - {"operation", "query"}:
            raise ValueError("memory lookup intent has unexpected fields")
        memory_query = RetrieveMemoryQuery.model_validate(data)
        log_event_fn(logger, logging.INFO, "memory_lookup_extraction_completed", model=model,
                     query_length=len(memory_query.query))
        return Interpretation(memory_query=memory_query, fallback_allowed=False)
    return None


def interpret_simple_response(
    data: dict[str, Any], *, route: IntentRoute, timezone_name: str
) -> Interpretation | None:
    """Validate snooze and summary responses for the selected route."""

    operation = data.get("operation")
    if route is IntentRoute.SNOOZE_REQUEST:
        if operation == "NEEDS_CLARIFICATION":
            if set(data) - {"operation", "question"}:
                raise ValueError("snooze clarification has unexpected fields")
            question = data.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("missing snooze clarification question")
            return Interpretation(response_text=question.strip(), fallback_allowed=False)
        if operation not in {"SNOOZE_OCCURRENCE", "SNOOZE_REMINDER"}:
            raise ValueError("provider returned an unsupported snooze operation")
        if set(data) - {"operation", "duration_seconds"}:
            raise ValueError("snooze intent has unexpected fields")
        duration_seconds = data.get("duration_seconds")
        if isinstance(duration_seconds, bool) or not isinstance(duration_seconds, int):
            raise ValueError("snooze duration was not an integer")
        from reminder_agent.features.reminders.snooze.operations import SnoozeDuration

        duration = SnoozeDuration(duration_seconds=duration_seconds)
        return Interpretation(snooze_duration_seconds=duration.duration_seconds, fallback_allowed=False)
    if route is IntentRoute.SUMMARY_REQUEST:
        if operation == "NEEDS_CLARIFICATION":
            if set(data) - {"operation", "question"}:
                raise ValueError("summary clarification has unexpected fields")
            question = data.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("missing summary clarification question")
            return Interpretation(response_text=question.strip(), fallback_allowed=False)
        if operation in {"SUMMARY_REQUEST", "REMINDER_SUMMARY", "STATUS_SUMMARY"}:
            operation = "SUMMARY"
        if operation != "SUMMARY":
            raise ValueError("provider returned an unsupported summary operation")
        metric = data.get("metric", data.get("summary_type"))
        if not isinstance(metric, str):
            raise ValueError("summary intent is missing metric")
        aliases = {
            "PENDING": "PENDING_COUNT", "PENDING_REMINDERS": "PENDING_COUNT",
            "OUTSTANDING": "PENDING_COUNT", "REMAINING_TODAY": "TODAY_REMAINING",
            "COMPLETED": "COMPLETED_COUNT", "COMPLETED_TODAY": "TODAY_COMPLETED",
            "OVERDUE": "OVERDUE_COUNT",
        }
        metric = cast(SummaryMetric, aliases.get(metric.upper(), metric.upper()))
        return Interpretation(intent=ReminderSummaryIntent(metric=metric, timezone_name=timezone_name),
                              fallback_allowed=False)
    return None


def interpret_response(
    data: dict[str, Any],
    *,
    route: IntentRoute,
    text: str,
    timezone_name: str,
    now: datetime,
    context: PendingReminder | None,
    recent_messages: Sequence[str],
    pending_reschedule: Any | None,
    model: str,
    endpoint_host: str,
    logger: logging.Logger,
    log_event_fn: Callable[..., Any],
    listing_task_query: Callable[[str], str | None],
    next_recurring_start_local: Callable[..., datetime],
) -> Interpretation:
    """Validate one provider object and convert it to a channel-neutral result."""

    operation = data.get("operation")
    if route is IntentRoute.RESCHEDULE_REQUEST and operation in {
        "RESCHEDULE",
        "RESCHEDULE_REQUEST",
        "MOVE_REMINDER",
        "UPDATE_REMINDER",
        "CHANGE_REMINDER",
        "SHIFT_REMINDER",
    }:
        # Keep provider wording differences inside the untrusted boundary;
        # canonical schema, ownership, and transition validation still run.
        operation = "RESCHEDULE_REMINDER"
        data = {**data, "operation": operation}
    if (
        route is IntentRoute.NEW_REQUEST
        and operation is None
        and set(data) == {"task", "due_local"}
    ):
        # Some OpenAI-compatible providers omit the discriminator even though
        # they return the complete, route-specific one-time proposal. The
        # route is already selected by the classifier and the exact field set
        # keeps this compatibility correction narrowly scoped. All normal
        # datetime, task, ownership, and business validation still follows.
        data = {**data, "operation": "CREATE_ONE_TIME_REMINDER"}
        operation = "CREATE_ONE_TIME_REMINDER"
        log_event_fn(
            logger,
            logging.WARNING,
            "reminder_intent_provider_operation_inferred",
            endpoint_host=endpoint_host,
            model=model,
            route=route,
            operation=operation,
        )
    memory_result = interpret_memory_response(
        data,
        route=route,
        model=model,
        logger=logger,
        log_event_fn=log_event_fn,
    )
    if memory_result is not None:
        return memory_result
    simple_result = interpret_simple_response(
        data,
        route=route,
        timezone_name=timezone_name,
    )
    if simple_result is not None:
        return simple_result
    if route is IntentRoute.LIST_REQUEST and operation in {
        "LIST_REQUEST",
        "LIST_REMINDER",
        "LIST_REMINDERS_BY_RANGE",
        "RETRIEVE_REMINDERS",
        "SHOW_REMINDERS",
    }:
        operation = "LIST_REMINDERS"
    if route is IntentRoute.CANCEL_REQUEST:
        from reminder_agent.features.reminders.cancel.decisions import (
            CancellationRequest,
            CancellationScope,
            ground_model_cancellation_request,
        )

        unexpected_fields = set(data) - {
            "operation",
            "target_text",
            "scope",
            "occurrence_date",
        }
        if operation == "NEEDS_CLARIFICATION":
            if set(data) - {"operation", "question"}:
                raise ValueError("cancellation clarification has unexpected fields")
            question = data.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("missing cancellation clarification question")
            return Interpretation(response_text=question.strip(), fallback_allowed=False)
        if unexpected_fields:
            raise ValueError("cancellation intent has unexpected fields")
        if operation != "CANCEL_REMINDER":
            raise ValueError("provider returned an unsupported cancellation operation")
        target_text = data.get("target_text")
        if not isinstance(target_text, str) or not target_text.strip():
            raise ValueError("cancellation intent is missing target_text")
        scope_value = data.get("scope")
        if scope_value is not None:
            try:
                scope = CancellationScope(scope_value)
            except ValueError as exc:
                raise ValueError("cancellation intent has invalid scope") from exc
        else:
            scope = None
        occurrence_date_value = data.get("occurrence_date")
        if occurrence_date_value is not None:
            if not isinstance(occurrence_date_value, str):
                raise ValueError("cancellation occurrence_date was not a string")
            try:
                occurrence_date = date.fromisoformat(occurrence_date_value)
            except ValueError:
                # Some providers copy the ISO timestamp from cancellation
                # candidate context into the date-only field. Keep the
                # provider boundary tolerant here; grounding below remains
                # authoritative and will discard dates not stated in the
                # user's current request.
                candidate_due_at = datetime.fromisoformat(occurrence_date_value)
                occurrence_date = candidate_due_at.date()
        else:
            occurrence_date = None
        cancellation_request = CancellationRequest(
            target_text=target_text,
            scope=scope,
            occurrence_date=occurrence_date,
        )
        return Interpretation(
            cancellation_request=ground_model_cancellation_request(
                text,
                cancellation_request,
                timezone_name=timezone_name,
                now=now,
                recent_messages=recent_messages,
            ),
            fallback_allowed=False,
        )
    if route is IntentRoute.RESCHEDULE_REQUEST:
        from reminder_agent.features.reminders.reschedule.decisions import (
            PendingRescheduleOperation,
            PendingRescheduleProposal,
            RescheduleRequest,
            RescheduleScope,
        )

        if pending_reschedule is not None:
            pending_operations = {item.value for item in PendingRescheduleOperation}
            if operation not in pending_operations:
                raise ValueError("provider returned an unsupported pending reschedule operation")
            raw_proposal: dict[str, Any] = {"operation": operation}
            for field in (
                "candidate_id",
                "date_explicit",
                "offset_minutes",
                "scope",
                "question",
            ):
                if field in data:
                    raw_proposal[field] = data[field]
            due_local_value = data.get("due_local")
            if due_local_value is not None:
                if not isinstance(due_local_value, str):
                    raise ValueError("pending reschedule due_local was not a string")
                due_local = _provider_local_datetime(
                    due_local_value, timezone_name=timezone_name
                )
                raw_proposal["due_at"] = due_local.replace(
                    tzinfo=ZoneInfo(timezone_name)
                ).astimezone(UTC)
            destination_date = data.get("destination_date")
            if destination_date is not None:
                if not isinstance(destination_date, str):
                    raise ValueError("pending reschedule destination_date was not a string")
                raw_proposal["destination_date"] = date.fromisoformat(destination_date)
            proposal = PendingRescheduleProposal.model_validate(raw_proposal)
            return Interpretation(
                pending_reschedule_proposal=proposal,
                response_text=proposal.question.strip() if proposal.question else None,
                fallback_allowed=False,
            )

        if operation == "RESCHEDULE_REMINDER":
            target_text = data.get("target_text")
            due_local_value = data.get("due_local")
            offset_minutes = data.get("offset_minutes")
            scope_value = data.get("scope")
            date_explicit = data.get("date_explicit")
            destination_date_value = data.get("destination_date")
            destination_date = None
            if destination_date_value is not None:
                if not isinstance(destination_date_value, str):
                    raise ValueError("reschedule destination_date was not a string")
                try:
                    destination_date = date.fromisoformat(destination_date_value)
                except ValueError as exc:
                    raise ValueError("reschedule destination_date was invalid") from exc
            if not isinstance(target_text, str) or not target_text.strip():
                raise ValueError("reschedule intent is missing target_text")
            if (isinstance(due_local_value, str)) == (
                isinstance(offset_minutes, int) and not isinstance(offset_minutes, bool)
            ):
                raise ValueError("reschedule intent needs exactly one time target")
            due_at = None
            if isinstance(due_local_value, str):
                due_local = _provider_local_datetime(
                    due_local_value, timezone_name=timezone_name
                )
                due_at = due_local.replace(tzinfo=ZoneInfo(timezone_name)).astimezone(UTC)
            reschedule_scope: RescheduleScope | None
            if scope_value is not None:
                try:
                    reschedule_scope = RescheduleScope(scope_value)
                except ValueError as exc:
                    raise ValueError("reschedule intent has invalid scope") from exc
            else:
                reschedule_scope = None
            return Interpretation(
                reschedule_request=RescheduleRequest(
                    target_text=target_text,
                    due_at=due_at,
                    offset_minutes=offset_minutes if isinstance(offset_minutes, int) else None,
                    scope=reschedule_scope,
                    date_explicit=(
                        date_explicit
                        if isinstance(date_explicit, bool)
                        else isinstance(due_local_value, str)
                    ),
                    source_text=text,
                ),
                fallback_allowed=False,
            )
        if operation == "NEEDS_CLARIFICATION":
            question = data.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("missing reschedule clarification question")
            target_text = data.get("target_text")
            due_local_value = data.get("due_local")
            offset_minutes = data.get("offset_minutes")
            scope_value = data.get("scope")
            date_explicit = data.get("date_explicit")
            destination_date_value = data.get("destination_date")
            destination_date = None
            if destination_date_value is not None:
                if not isinstance(destination_date_value, str):
                    raise ValueError("reschedule destination_date was not a string")
                try:
                    destination_date = date.fromisoformat(destination_date_value)
                except ValueError as exc:
                    raise ValueError("reschedule destination_date was invalid") from exc
            has_absolute_time = isinstance(due_local_value, str)
            has_relative_time = isinstance(offset_minutes, int) and not isinstance(
                offset_minutes, bool
            )
            if isinstance(target_text, str) and target_text.strip():
                if not has_absolute_time and not has_relative_time:
                    # Preserve the grounded target for durable clarification,
                    # but leave it explicitly non-mutating because no time was
                    # supplied.
                    return Interpretation(
                        reschedule_request=RescheduleRequest(
                            target_text=target_text,
                            scope=(
                                RescheduleScope(scope_value)
                                if scope_value is not None
                                else None
                            ),
                            date_explicit=(
                                date_explicit if isinstance(date_explicit, bool) else False
                            ),
                            destination_date=destination_date,
                            source_text=text,
                        ),
                        response_text=question.strip(),
                        fallback_allowed=False,
                    )
                if has_absolute_time and has_relative_time:
                    raise ValueError("reschedule clarification has multiple time targets")
                due_at = None
                if has_absolute_time:
                    assert isinstance(due_local_value, str)
                    due_local = _provider_local_datetime(
                        due_local_value, timezone_name=timezone_name
                    )
                    due_at = due_local.replace(tzinfo=ZoneInfo(timezone_name)).astimezone(UTC)
                clarification_scope_with_time: RescheduleScope | None = None
                if scope_value is not None:
                    try:
                        clarification_scope_with_time = RescheduleScope(scope_value)
                    except ValueError as exc:
                        raise ValueError("reschedule intent has invalid scope") from exc
                return Interpretation(
                    reschedule_request=RescheduleRequest(
                        target_text=target_text,
                        due_at=due_at,
                        offset_minutes=offset_minutes if has_relative_time else None,
                        scope=clarification_scope_with_time,
                        date_explicit=(
                            date_explicit
                            if isinstance(date_explicit, bool)
                            else isinstance(due_local_value, str)
                        ),
                        source_text=text,
                    ),
                    response_text=question.strip(),
                    fallback_allowed=False,
                )
            return Interpretation(response_text=question.strip(), fallback_allowed=False)
        raise ValueError("provider returned an unsupported reschedule operation")
    if route is IntentRoute.UPDATE_PENDING:
        if operation == "UPDATE_PENDING_REMINDER":
            raw_updates = data.get("updates")
            if raw_updates is None:
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
                raw_updates = {key: value for key, value in data.items() if key in update_fields}
            if not isinstance(raw_updates, dict):
                raise ValueError("pending update was not an object")
            updates = PendingReminderFieldUpdate.model_validate(raw_updates)
            if not updates.supplied_fields():
                raise ValueError("pending update had no supplied fields")
            log_event_fn(
                logger,
                logging.INFO,
                "reminder_extraction_completed",
                model=model,
                operation=operation,
                field_count=len(updates.supplied_fields()),
                has_pending_context=True,
            )
            return Interpretation(field_updates=updates, fallback_allowed=False)
        if operation == "NEEDS_CLARIFICATION":
            question = data.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("missing clarification question")
            if context is None:
                raise ValueError("pending clarification has no trusted context")
            return Interpretation(
                clarification=NeedsClarification(pending=context, question=question.strip()),
                fallback_allowed=False,
            )
        if operation == "UNKNOWN":
            return Interpretation(response_text=OUT_OF_SCOPE_RESPONSE, fallback_allowed=False)
        raise ValueError("provider returned an authoritative pending reminder")
    if route is IntentRoute.LIST_REQUEST and operation == "NEEDS_CLARIFICATION":
        if set(data) - {"operation", "question"}:
            raise ValueError("list clarification has unexpected fields")
        question = data.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("missing list clarification question")
        return Interpretation(response_text=question.strip(), fallback_allowed=False)
    if operation == "NEEDS_CLARIFICATION":
        if set(data) - {"operation", "question", "pending"}:
            raise ValueError("clarification has unexpected fields")
        raw_pending = data.get("pending") or {}
        if not isinstance(raw_pending, dict):
            raise ValueError("pending clarification was not an object")
        pending_fields = {
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
            "reminders",
            "clarification_index",
        }
        if set(raw_pending) - pending_fields:
            raise ValueError("pending clarification has unexpected fields")
        if "reminders" in raw_pending:
            if route is not IntentRoute.NEW_REQUEST or context is not None:
                raise ValueError("multiple reminder drafts are only valid for a new request")
            raw_reminders = raw_pending.get("reminders")
            if not isinstance(raw_reminders, list) or not raw_reminders:
                raise ValueError("pending reminder drafts were not an array")
            clarification_index = raw_pending.get("clarification_index")
            if isinstance(clarification_index, bool) or not isinstance(clarification_index, int):
                raise ValueError("pending clarification index was not an integer")
            reminders: list[PendingReminder] = []
            for raw_reminder in raw_reminders:
                if not isinstance(raw_reminder, dict):
                    raise ValueError("pending reminder draft was not an object")
                if set(raw_reminder) - (pending_fields - {"reminders", "clarification_index"}):
                    raise ValueError("pending reminder draft has unexpected fields")
                pending_values = {
                    key: value
                    for key, value in raw_reminder.items()
                    if key
                    in {
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
                    }
                    and value is not None
                }
                pending_values["timezone_name"] = timezone_name
                reminders.append(PendingReminder.model_validate(pending_values))
            pending: PendingReminder | PendingReminderBatch = PendingReminderBatch(
                reminders=tuple(reminders),
                clarification_index=clarification_index,
            )
            field_count = sum(
                value is not None
                for reminder in reminders
                for value in reminder.model_dump(mode="python").values()
            )
        else:
            pending_values = context.model_dump(mode="python") if context is not None else {}
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
            pending = PendingReminder.model_validate(pending_values)
            field_count = sum(value is not None for value in pending_values.values())
        question = data.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("missing clarification question")
        log_event_fn(
            logger,
            logging.INFO,
            "reminder_extraction_completed",
            model=model,
            operation=operation,
            field_count=field_count,
            has_pending_context=False,
        )
        return Interpretation(clarification=NeedsClarification(pending=pending, question=question))
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
            due_local = _provider_local_datetime(
                due_local_value, timezone_name=timezone_name
            )
            intents.append(
                CreateOneTimeReminderIntent(
                    task=task,
                    due_at=due_local.replace(tzinfo=ZoneInfo(timezone_name)).astimezone(UTC),
                    timezone_name=timezone_name,
                )
            )
        return Interpretation(intents=tuple(intents))
    if operation == "CREATE_RECURRING_REMINDER":
        task = data.get("task")
        # ``weekdays`` is the canonical complete-operation field.  Older model
        # prompt variants emitted the pending-draft name ``recurrence_weekdays``
        # instead, which was silently ignored and left a weekly schedule with
        # no selected day.  Keep this narrow compatibility alias at the
        # provider boundary; the validated intent remains canonical.
        raw_weekdays = data.get("weekdays")
        if raw_weekdays is None:
            raw_weekdays = data.get("day_of_week")
        if raw_weekdays is None:
            raw_weekdays = data.get("recurrence_weekdays")
            if raw_weekdays is not None:
                log_event_fn(
                    logger,
                    logging.WARNING,
                    "reminder_intent_provider_field_alias_normalized",
                    endpoint_host=endpoint_host,
                    model=model,
                    route=route,
                    operation=operation,
                    provider_field="recurrence_weekdays",
                    canonical_field="weekdays",
                )
        weekdays = _normalise_weekdays(raw_weekdays)
        start_local_value = data.get("start_local")
        frequency_value = str(data.get("frequency", "WEEKLY")).upper()
        if frequency_value == "DAILY":
            frequency_value = "WEEKLY"
            weekdays = list(range(7))
        frequency = cast(RecurringFrequency, frequency_value)
        if not isinstance(task, str):
            raise ValueError("recurring intent is missing fields")
        if not isinstance(weekdays, list):
            weekdays = []
        interval = int(data.get("interval", 1))
        hour, minute = _normalise_time_fields(data)
        day_of_month = int(data["day_of_month"]) if data.get("day_of_month") is not None else None
        recurrence_month = int(data["month"]) if data.get("month") is not None else None
        if isinstance(start_local_value, str):
            start_local = _provider_local_datetime(
                start_local_value, timezone_name=timezone_name
            )
        else:
            start_local = next_recurring_start_local(
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
        return Interpretation(
            intent=CreateRecurringReminderIntent(
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
        )
    if operation == "LIST_REMINDERS":
        allowed_fields = {
            "operation",
            "filters",
            "start_date",
            "end_date",
            "date_range",
            "date",
            "task_query",
            "time_after",
            "time_before",
            "weekdays",
            "weekday",
            "recurrence_only",
            "recurrence_series_only",
            "recurrence_frequency",
            "limit",
        }
        unknown_fields = set(data) - allowed_fields
        if unknown_fields:
            raise ValueError("list intent contains unsupported fields")
        nested_filters = data.get("filters")
        if nested_filters is not None and not isinstance(nested_filters, dict):
            raise ValueError("list intent filters were not an object")
        filter_data = dict(nested_filters or {})
        filter_data.update(
            {key: value for key, value in data.items() if key not in {"operation", "filters"}}
        )
        unknown_filter_fields = set(filter_data) - (allowed_fields - {"operation", "filters"})
        if unknown_filter_fields:
            raise ValueError("list intent filters contain unsupported fields")

        start_date_value = filter_data.get("start_date")
        end_date_value = filter_data.get("end_date")
        date_range = filter_data.get("date_range")
        if isinstance(date_range, dict):
            start_date_value = start_date_value or date_range.get("start")
            end_date_value = end_date_value or date_range.get("end")
        single_date_value = filter_data.get("date")
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
        raw_task_query = filter_data.get("task_query")
        explicit_task_query = listing_task_query(text)
        if raw_task_query is not None and not isinstance(raw_task_query, str):
            raise ValueError("list intent has invalid task query")
        if explicit_task_query is not None:
            if not isinstance(raw_task_query, str) or not raw_task_query.strip():
                raise ValueError("list intent discarded an explicit task query")
            if normalize_task_query(raw_task_query).lower() != explicit_task_query.lower():
                raise ValueError("list intent changed an explicit task query")
        time_after = _normalise_listing_time(filter_data.get("time_after"), "time_after")
        time_before = _normalise_listing_time(filter_data.get("time_before"), "time_before")
        weekdays = _normalise_weekdays(
            filter_data.get("weekdays", filter_data.get("weekday"))
        )
        raw_limit = filter_data.get("limit")
        if raw_limit is not None:
            if isinstance(raw_limit, bool) or not isinstance(raw_limit, int):
                raise ValueError("list intent limit was not an integer")
            if raw_limit < 1 or raw_limit > 100:
                raise ValueError("list intent limit was outside 1..100")
        recurrence_frequency = filter_data.get("recurrence_frequency")
        if recurrence_frequency is not None:
            if not isinstance(recurrence_frequency, str):
                raise ValueError("list intent recurrence frequency was not a string")
            recurrence_frequency = recurrence_frequency.upper()
            if recurrence_frequency not in {"WEEKLY", "MONTHLY", "YEARLY"}:
                raise ValueError("list intent recurrence frequency was not recognized")
        recurrence_only = filter_data.get("recurrence_only", False)
        if not isinstance(recurrence_only, bool):
            raise ValueError("list intent recurrence_only was not boolean")
        recurrence_series_only = filter_data.get("recurrence_series_only", False)
        if not isinstance(recurrence_series_only, bool):
            raise ValueError("list intent recurrence_series_only was not boolean")
        return Interpretation(
            intent=ListRemindersIntent(
                start_date=date.fromisoformat(start_date_value) if start_date_value else None,
                end_date=date.fromisoformat(end_date_value) if end_date_value else None,
                task_query=raw_task_query,
                time_after=time_after,
                time_before=time_before,
                weekdays=tuple(weekdays) if weekdays else None,
                recurrence_only=recurrence_only,
                recurrence_series_only=recurrence_series_only,
                recurrence_frequency=recurrence_frequency,
                limit=raw_limit,
                timezone_name=timezone_name,
            )
        )
    if operation != "CREATE_ONE_TIME_REMINDER":
        result = Interpretation(response_text=OUT_OF_SCOPE_RESPONSE)
        log_event_fn(
            logger,
            logging.WARNING,
            "reminder_intent_provider_finished",
            endpoint_host=endpoint_host,
            model=model,
            has_intent=False,
            reason="unsupported_operation",
        )
        return result
    task = data.get("task")
    due_local_value = data.get("due_local")
    if not isinstance(task, str) or not isinstance(due_local_value, str):
        raise ValueError("missing structured fields")
    due_local = _provider_local_datetime(due_local_value, timezone_name=timezone_name)
    one_time_intent = CreateOneTimeReminderIntent(
        task=task,
        due_at=due_local.replace(tzinfo=ZoneInfo(timezone_name)).astimezone(UTC),
        timezone_name=timezone_name,
    )
    result = Interpretation(intent=one_time_intent)
    log_event_fn(
        logger,
        logging.INFO,
        "reminder_extraction_completed",
        model=model,
        operation=operation,
        field_count=2,
        has_pending_context=False,
    )
    log_event_fn(
        logger,
        logging.INFO,
        "reminder_intent_provider_finished",
        endpoint_host=endpoint_host,
        model=model,
        has_intent=True,
        task_length=len(task),
    )
    return result


__all__ = ["decode_response", "interpret_response"]
