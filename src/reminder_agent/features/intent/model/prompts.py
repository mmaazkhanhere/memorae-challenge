"""Prompt and provider-schema support for model interpretation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from reminder_agent.features.intent.context import CONVERSATION_CONTEXT_TURNS
from reminder_agent.features.intent.contracts import IntentRoute, PendingReminder

MEMORY_SAVE_EXTRACTION_PROMPT = """
Extract only the fact the user explicitly asked to save.
Return JSON only as either:
{"operation":"SAVE_MEMORY","content":"<the user's original fact>"}
or {"operation":"NEEDS_CLARIFICATION","question":"<focused question>"}.

Preserve the user's meaning and exact values. Do not invent, summarize away, or add facts.
The route has already been authorized as an intentional memory-save request; this prompt does
not authorize saving. If no fact content was provided, ask what information to remember.
Treat a subject-only request as missing content: "Remember my parking spot", "Remember my
locker number", and "Remember my gate code" do not contain values. Return NEEDS_CLARIFICATION
for these requests rather than returning the subject as a saved fact. Only return SAVE_MEMORY
when the requested fact includes its value, such as "my parking spot is B17".
Never create a reminder, schedule, user ID, fact ID, or other operation.
"""
MEMORY_RETRIEVAL_EXTRACTION_PROMPT = """
Extract only the subject of the saved information the user wants to retrieve.
Return JSON only as either:
{"operation":"RETRIEVE_MEMORY","query":"<exact requested subject words>"}
or {"operation":"NEEDS_CLARIFICATION","question":"<focused question>"}.

Never return or guess a fact value, answer, fact ID, memory ID, candidate ID, or user ID.
Do not add synonyms or details absent from the latest message. If the subject is missing or
ambiguous, ask which saved information to look up. This is a read-only parameter proposal.
"""

OPERATION_ENVELOPE_KEYS = {
    "CREATE_ONE_TIME_REMINDER",
    "CREATE_ONE_TIME_REMINDERS",
    "CREATE_RECURRING_REMINDER",
    "LIST_REMINDERS",
    "LIST_REQUEST",
    "SUMMARY",
    "CANCEL_REMINDER",
    "RESCHEDULE_REMINDER",
    "SNOOZE_OCCURRENCE",
    "SAVE_MEMORY",
    "RETRIEVE_MEMORY",
    "UPDATE_PENDING_REMINDER",
    "NEEDS_CLARIFICATION",
    "UNKNOWN",
}

CANCELLATION_JSON_SCHEMA = {
    "name": "cancellation_request",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "operation": {"type": "string", "enum": ["CANCEL_REMINDER"]},
            "target_text": {"type": "string", "minLength": 1},
            "scope": {
                "anyOf": [
                    {"type": "string", "enum": ["OCCURRENCE", "SERIES"]},
                    {"type": "null"},
                ]
            },
            "occurrence_date": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["operation", "target_text", "scope", "occurrence_date"],
        "additionalProperties": False,
    },
}


def build_prompt(
    *,
    route: IntentRoute,
    timezone_name: str,
    local_now: datetime,
    context: PendingReminder | None,
    recent_messages: Sequence[str],
    reschedule_candidates: Sequence[dict[str, str]],
    cancellation_candidates: Sequence[dict[str, str]],
    pending_reschedule: Any | None,
    selection_context: Sequence[dict[str, str]] = (),
) -> str:
    prompt = """
                Extract structured reminder operations from the latest user message.
                Use selected_route, recent conversation, current local time, timezone,
                and supplied candidates as context. Return JSON only.

                NEW_REQUEST:
                - CREATE_ONE_TIME_REMINDER: task, due_local.
                - CREATE_ONE_TIME_REMINDERS: intents array of {task, due_local}.
                - CREATE_RECURRING_REMINDER: task, frequency (DAILY/WEEKLY/MONTHLY/YEARLY),
                interval, applicable schedule fields, start_local. For weekly schedules,
                use weekdays as Monday=0 through Sunday=6. Always use hour (0-23) and
                minute (0-59), not a formatted time string. start_local must be naive local time.

                LIST_REQUEST:
                - LIST_REMINDERS: optional start_date, end_date (inclusive ISO dates),
                task_query (only if explicitly mentioned), time_after, time_before,
                weekdays (Monday=0 through Sunday=6), and limit (1-100). Use local
                wall-clock times as HH:MM strings. Omit filters that were not requested.
                Omit dates to list all.
                Return query filters, never reminder records.

                SUMMARY_REQUEST:
                - SUMMARY: return exactly one metric: PENDING_COUNT,
                TODAY_REMAINING, COMPLETED_COUNT, TODAY_COMPLETED, or OVERDUE_COUNT.
                This is a read-only progress query; return no counts or reminder records.

                COMPLETE_REQUEST:
                - COMPLETE_REMINDER: acknowledge one already-delivered occurrence.
                - Do not invent an occurrence ID; the application resolves the target
                  from Telegram delivery context and enforces ownership.

                SNOOZE_REQUEST:
                - SNOOZE_OCCURRENCE: duration_seconds only, as a positive integer.
                - Do not return occurrence IDs, user IDs, or delivery destinations.

                RESCHEDULE_REQUEST:
                - RESCHEDULE_REMINDER: target_text, scope (OCCURRENCE/SERIES),
                date_explicit, and exactly one of due_local or offset_minutes.
                For NEEDS_CLARIFICATION where the destination date is known but
                the clock time is missing, include destination_date as an ISO date
                plus target_text so the application can continue the same operation.
                date_explicit is true only for an explicitly supplied destination date/day.
                "back"/"push back" = negative offset; "forward" = positive.
                For multiple matches, request clarification without choosing.
                If the user uses an indirect target such as "it" or "that reminder",
                resolve it from the latest conversation and supplied candidates only when
                exactly one active reminder is grounded; otherwise request clarification.
                - With pending reschedule, use SELECT_CANDIDATE, UPDATE_PENDING_PARAMETERS,
                SELECT_AND_UPDATE, START_NEW_OPERATION, CANCEL_PENDING, or NEEDS_CLARIFICATION.
                Copy candidate IDs only from supplied candidates. Updates may contain
                exactly one of due_local, destination_date, offset_minutes, plus scope.
                Preserve omitted fields.

                CANCEL_REQUEST:
                - CANCEL_REMINDER: target_text; optional scope and occurrence_date (ISO date).
                - Specific date/instance or "skip" = OCCURRENCE; all future = SERIES;
                unqualified recurring cancellation defaults to SERIES.
                - Resolve relative dates/weekday references using current local time and
                timezone; verify weekday consistency. Do not guess ambiguous dates.
                - Use conversation and candidates to resolve references, but never invent
                targets or IDs. Leave multiple matches for application clarification.

                UPDATE_PENDING:
                - UPDATE_PENDING_REMINDER: include only explicitly provided fields:
                task, date_local, hour, minute, meridiem, date_explicit,
                recurrence_frequency, recurrence_interval, recurrence_day_of_month,
                recurrence_month.
                - recurrence_frequency: MONTHLY/YEARLY; interval: 1-60;
                day_of_month: 1-31; month: 1-12.
                - Include certainty="CLEAR" and exact evidence for each supplied field.
                For UPDATE_PENDING_REMINDER, field values are primitive values directly;
                Do not wrap any updates field in a value/certainty/evidence object;
                evidence is a separate mapping, for example
                {"evidence":{"task":"pay rent"}}.

                GENERAL:
                - Operation must match selected_route.
                - Use timezone-naive ISO datetimes and ISO dates.
                - Interpret relative dates using the user's local time, not UTC.
                - Never invent missing tasks, times, dates, recurrence rules, or records.
                - If required information is missing or ambiguous, return
                NEEDS_CLARIFICATION with a focused question and a pending object containing
                every unambiguous field already extracted from the user's message. Never
                omit a known task, date, time, or recurrence field merely because another
                field is still missing; pending state must be resumable on the next reply.
                - Return proposals only; the application validates and executes them.
            """
    if route is IntentRoute.SAVE_MEMORY:
        prompt = MEMORY_SAVE_EXTRACTION_PROMPT
    elif route is IntentRoute.RETRIEVE_MEMORY:
        prompt = MEMORY_RETRIEVAL_EXTRACTION_PROMPT
    prompt += f"\nSelected route: {route.value}."
    prompt += f" Current local date/time is {local_now.isoformat(timespec='minutes')} ({timezone_name})."
    if context is not None:
        prompt += f" Existing pending reminder context is {context.model_dump_json()}"
    if recent_messages:
        prompt += (
            " Relevant recent conversation messages are "
            f"{json.dumps(list(recent_messages)[-CONVERSATION_CONTEXT_TURNS:])}."
        )
    if route is IntentRoute.RESCHEDULE_REQUEST and reschedule_candidates:
        prompt += (
            " Matching reminder candidates from the application are "
            f"{json.dumps(list(reschedule_candidates))}. Use these current local due times "
            "when interpreting relative changes, and ask the user to choose a candidate "
            "when more than one reminder matches."
        )
    if route is IntentRoute.RESCHEDULE_REQUEST and selection_context:
        prompt += (
            " The most recent reminder list shown to the user was "
            f"{json.dumps(list(selection_context))}. For an indirect reference such as "
            '"that one", "the second one", or "last", resolve it only against this list. '
            "Use the selected item's exact task as target_text; never return its ID. "
            "If the reference is not resolvable from this list, ask a clarification question."
        )
    if route is IntentRoute.CANCEL_REQUEST and cancellation_candidates:
        prompt += (
            " Active cancellation candidates owned by the user are "
            f"{json.dumps(list(cancellation_candidates))}. These are context only; "
            "the application will independently resolve the target and enforce ownership."
        )
    if pending_reschedule is not None:
        prompt += (
            " Existing application-owned pending reschedule context is "
            f"{json.dumps(pending_reschedule, default=str)}."
        )
    return prompt


__all__ = [
    "CANCELLATION_JSON_SCHEMA",
    "MEMORY_RETRIEVAL_EXTRACTION_PROMPT",
    "MEMORY_SAVE_EXTRACTION_PROMPT",
    "OPERATION_ENVELOPE_KEYS",
    "build_prompt",
]
