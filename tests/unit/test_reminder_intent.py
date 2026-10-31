import json
from datetime import date, datetime, timezone

import httpx
import pytest

from reminder_agent.intent import (
    CreateOneTimeReminderIntent,
    IntentRoute,
    OpenAICompatibleIntentRouter,
    Interpretation,
    NeedsClarification,
    OpenAICompatibleIntentInterpreter,
    PendingReminderFieldUpdate,
    PendingReminder,
    ReminderIntentInterpreter,
    RuleBasedIntentInterpreter,
    restrict_to_explicit_field_updates,
    resolve_pending_due_at,
)
from reminder_agent.settings import Settings
from reminder_agent.request_policy import OUT_OF_SCOPE_RESPONSE


NOW = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)


def test_intent_router_classifies_only_the_operation_and_limits_history(monkeypatch) -> None:
    captured: dict = {}

    def fake_post(*args, **kwargs):
        captured.update(kwargs["json"])
        return httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={"choices": [{"message": {"content": '{"operation":"NEW_REQUEST"}'}}]},
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    suggestion = OpenAICompatibleIntentRouter(
        "https://api.groq.example/openai/v1", "secret", "openai/gpt-oss-20b"
    ).classify(
        "tommorow",
        pending=PendingReminder(task="call Bilal", hour=4, meridiem="pm", timezone_name="UTC"),
        recent_messages=[
            *(f"message {index}" for index in range(35)),
            "assistant: What date should I use?",
            "user: unrelated earlier reply",
            "assistant: Please clarify the date.",
            "user: another earlier reply",
            "assistant: What date should I use?",
        ],
    )

    assert suggestion.operation is IntentRoute.NEW_REQUEST
    context = json.loads(captured["messages"][1]["content"])
    assert context["active_pending"]["task"] == "call Bilal"
    assert context["missing_fields"] == ["date"]
    assert context["last_clarification_question"] == "What date should I use?"
    assert len(context["recent_messages"]) == 6
    assert "message 33" not in context["recent_messages"]
    assert "message 34" in context["recent_messages"]


def test_pending_update_normalises_meridiem_case() -> None:
    assert PendingReminderFieldUpdate(meridiem="PM").meridiem == "pm"
    assert PendingReminderFieldUpdate(meridiem="AM").meridiem == "am"


def test_pending_update_rejects_invalid_meridiem() -> None:
    with pytest.raises(ValueError):
        PendingReminderFieldUpdate(meridiem="evening")


def test_pending_update_accepts_evidence_for_every_allowed_update_field() -> None:
    update = PendingReminderFieldUpdate(
        date_explicit=True,
        certainty="clear",
        evidence={"date_explicit": "tommorow"},
    )

    assert update.certainty == "CLEAR"
    assert update.evidence == {"date_explicit": "tommorow"}


def test_default_llm_configuration_targets_openai_compatible_api() -> None:
    settings = Settings(_env_file=None)

    assert settings.llm_base_url == "https://api.openai.com/v1"
    assert settings.llm_extract_model == "gpt-5.4-mini-2026-03-17"
    assert settings.llm_intent_model == "openai/gpt-oss-20b"


def test_relative_minutes_are_resolved_from_one_reference_time() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me in five minutes to call the bank.",
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert result.intent is not None
    assert result.intent.task == "call the bank"
    assert result.intent.due_at == datetime(2026, 10, 30, 12, 5, tzinfo=timezone.utc)


def test_relative_minutes_support_task_before_time() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me to call Alice in two minutes.",
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert result.intent is not None
    assert result.intent.task == "call Alice"
    assert result.intent.due_at == datetime(2026, 10, 30, 12, 2, tzinfo=timezone.utc)


def test_tomorrow_at_nine_respects_user_local_timezone() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me tomorrow at 9 AM to renew the domain.",
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert result.intent is not None
    assert result.intent.task == "renew the domain"
    assert result.intent.due_at == datetime(2026, 10, 31, 4, 0, tzinfo=timezone.utc)


def test_missing_task_is_clarified_without_an_intent() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me in five minutes.",
        timezone_name="UTC",
        now=NOW,
    )

    assert result.intent is None
    assert result.response_text == "What should I remind you about?"


def test_task_only_request_uses_model_fallback_for_clarification() -> None:
    class RecordingModel:
        called = False

        def interpret(self, *args, **kwargs):
            self.called = True
            return Interpretation(
                clarification=NeedsClarification(
                    pending=PendingReminder(task="call the bank", timezone_name="UTC"),
                    question="When should I remind you to call the bank?",
                )
            )

    model = RecordingModel()
    result = ReminderIntentInterpreter(model).interpret(
        "Remind me to call the bank",
        timezone_name="UTC",
        now=NOW,
    )

    assert model.called is True
    assert result.clarification is not None
    assert result.clarification.pending.task == "call the bank"
    assert result.clarification.pending.date_local is None
    assert result.clarification.pending.hour is None


def test_tomorrow_morning_is_a_deterministic_incomplete_request() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me tomorrow morning to call the bank",
        timezone_name="UTC",
        now=NOW,
    )

    assert result.intent is None
    assert result.clarification is not None
    assert result.clarification.pending.task == "call the bank"
    assert result.clarification.pending.date_local == date(2026, 10, 31)
    assert result.clarification.pending.hour is None
    assert "time" in result.clarification.question.lower()


def test_day_part_without_task_uses_model_fallback() -> None:
    class RecordingModel:
        called = False

        def interpret(self, *args, **kwargs):
            self.called = True
            return Interpretation(
                clarification=NeedsClarification(
                    pending=PendingReminder(timezone_name="UTC"),
                    question="What should I remind you about, and when?",
                )
            )

    model = RecordingModel()
    result = ReminderIntentInterpreter(model).interpret(
        "Remind me tomorrow morning",
        timezone_name="UTC",
        now=NOW,
    )

    assert model.called is True
    assert result.clarification is not None
    assert result.clarification.pending.task is None


def test_provider_clarification_does_not_erase_rule_extracted_task() -> None:
    class IncompleteProvider:
        def interpret(self, text, *, timezone_name, now, context=None):
            return Interpretation(
                clarification=NeedsClarification(
                    pending=PendingReminder(
                        date_local=date(2026, 10, 31),
                        hour=10,
                        meridiem="am",
                        timezone_name=timezone_name,
                    ),
                    question="What should I remind you about?",
                )
            )

    result = ReminderIntentInterpreter(IncompleteProvider()).interpret(
        "Remind me to submit my assignment.",
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert result.clarification is not None
    assert result.clarification.pending.task == "submit my assignment"
    assert result.clarification.pending.date_local == date(2026, 10, 31)


def test_provider_clarification_normalises_uppercase_meridiem() -> None:
    class UppercaseProvider:
        def interpret(self, text, *, timezone_name, now, context=None):
            return Interpretation(
                clarification=NeedsClarification(
                    pending=PendingReminder(
                        date_local=date(2026, 10, 31),
                        hour=4,
                        meridiem="PM",
                        timezone_name=timezone_name,
                    ),
                    question="What should I remind you about?",
                )
            )

    result = ReminderIntentInterpreter(UppercaseProvider()).interpret(
        "Remind me tomorrow at 4 PM",
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert result.clarification is not None
    assert result.clarification.pending.date_local == date(2026, 10, 31)
    assert result.clarification.pending.hour == 4
    assert result.clarification.pending.meridiem == "pm"


def test_weekday_schedule_with_explicit_meridiem_is_parsed_deterministically() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me on Friday at 6 PM to go to the gym",
        timezone_name="UTC",
        now=NOW,
    )

    assert result.intent is not None
    assert result.intent.task == "go to the gym"
    assert result.intent.due_at == datetime(2026, 10, 30, 18, 0, tzinfo=timezone.utc)


def test_recurring_request_is_not_parsed_deterministically() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me to call John every Friday at 6 PM",
        timezone_name="UTC",
        now=NOW,
    )

    assert result.intent is None
    assert result.fallback_allowed is True


def test_ambiguous_hour_is_clarified_without_guessing_next_morning() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me to call the bank at 7",
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 15, 0, tzinfo=timezone.utc),
    )

    assert result.intent is None
    assert isinstance(result.clarification, NeedsClarification)
    assert result.clarification.pending.task == "call the bank"
    assert result.clarification.pending.hour == 7
    assert result.clarification.pending.meridiem is None
    assert "7 PM today" in result.clarification.question
    assert "7 AM tomorrow" in result.clarification.question
    assert "7 AM today" not in result.clarification.question


def test_tomorrow_clock_without_meridiem_preserves_all_fields_for_clarification() -> None:
    interpreter = RuleBasedIntentInterpreter()
    initial = interpreter.interpret(
        "Remind me tomorrow at 7:44 to call the bank",
        timezone_name="UTC",
        now=NOW,
    )

    assert initial.intent is None
    assert initial.clarification is not None
    assert initial.clarification.pending.task == "call the bank"
    assert initial.clarification.pending.date_local == date(2026, 10, 31)
    assert initial.clarification.pending.hour == 7
    assert initial.clarification.pending.minute == 44
    assert initial.clarification.pending.meridiem is None
    assert "7:44 AM tomorrow" in initial.clarification.question
    assert "7:44 PM tomorrow" in initial.clarification.question

    completed = interpreter.interpret(
        "7:44 PM",
        timezone_name="UTC",
        now=NOW,
        context=initial.clarification.pending,
    )

    assert completed.intent is not None
    assert completed.intent.task == "call the bank"
    assert completed.intent.due_at == datetime(2026, 10, 31, 19, 44, tzinfo=timezone.utc)


def test_ambiguous_time_on_an_expired_explicit_date_asks_only_for_a_new_date() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me Friday at 12 to call the bank",
        timezone_name="UTC",
        now=NOW,
    )

    assert result.clarification is not None
    assert result.clarification.pending.task == "call the bank"
    assert result.clarification.pending.date_local is None
    assert result.clarification.pending.hour == 12
    assert result.clarification.pending.minute == 0
    assert "future date" in result.clarification.question
    assert "AM" not in result.clarification.question
    assert "PM" not in result.clarification.question


def test_weekday_without_hour_is_clarified_and_keeps_task() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me Friday to renew the domain",
        timezone_name="UTC",
        now=NOW,
    )

    assert result.intent is None
    assert result.clarification is not None
    assert result.clarification.pending.task == "renew the domain"
    assert result.clarification.pending.date_local is not None
    assert "time" in result.clarification.question.lower()


def test_clarification_follow_up_merges_once_complete() -> None:
    interpreter = RuleBasedIntentInterpreter()
    initial = interpreter.interpret(
        "Remind me to call the bank at 7",
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 15, 0, tzinfo=timezone.utc),
    )

    assert initial.clarification is not None
    completed = interpreter.interpret(
        "7 PM",
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 15, 0, tzinfo=timezone.utc),
        context=initial.clarification.pending,
    )

    assert completed.clarification is None
    assert completed.intent is not None
    assert completed.intent.task == "call the bank"
    assert completed.intent.due_at == datetime(2026, 10, 30, 19, 0, tzinfo=timezone.utc)


def test_unrelated_follow_up_does_not_fill_missing_task() -> None:
    interpreter = RuleBasedIntentInterpreter()
    initial = interpreter.interpret(
        "Remind me at 7",
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 15, 0, tzinfo=timezone.utc),
    )

    assert initial.clarification is not None
    follow_up = interpreter.interpret(
        "thanks",
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 15, 0, tzinfo=timezone.utc),
        context=initial.clarification.pending,
    )

    assert follow_up.intent is None
    assert follow_up.clarification is not None
    assert follow_up.clarification.pending.task is None


def test_follow_up_cancellation_clears_context_without_creating_reminder() -> None:
    interpreter = RuleBasedIntentInterpreter()
    initial = interpreter.interpret(
        "Remind me to call the bank at 7",
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 15, 0, tzinfo=timezone.utc),
    )

    assert initial.clarification is not None
    canceled = interpreter.interpret(
        "Never mind.",
        timezone_name="UTC",
        now=datetime(2026, 10, 30, 15, 0, tzinfo=timezone.utc),
        context=initial.clarification.pending,
    )

    assert canceled.intent is None
    assert canceled.clarification is None
    assert canceled.clear_context is True
    assert "won't create" in canceled.response_text


def test_pending_time_resolution_uses_the_context_timezone() -> None:
    pending = PendingReminder(
        task="call the bank",
        date_local=date(2026, 10, 31),
        hour=9,
        minute=30,
        meridiem="am",
        timezone_name="Asia/Karachi",
        date_explicit=True,
    )

    assert resolve_pending_due_at(pending, now=NOW) == datetime(
        2026, 10, 31, 4, 30, tzinfo=timezone.utc
    )


def test_provider_updates_cannot_change_unmentioned_pending_fields() -> None:
    context = PendingReminder(
        task="call the bank",
        date_local=date(2026, 10, 31),
        hour=7,
        minute=0,
        meridiem="pm",
        timezone_name="UTC",
        date_explicit=True,
    )
    verified = restrict_to_explicit_field_updates(
        "8 AM",
        PendingReminderFieldUpdate(
            task="book a holiday",
            date_local=date(2026, 11, 2),
            hour=8,
            minute=0,
            meridiem="am",
        ),
        context=context,
        now=NOW,
    )

    assert verified.model_dump(exclude_none=True) == {
        "hour": 8,
        "minute": 0,
        "meridiem": "am",
    }


def test_provider_task_update_requires_an_explicit_correction() -> None:
    context = PendingReminder(
        task="call the bank",
        date_local=date(2026, 10, 31),
        hour=7,
        timezone_name="UTC",
        date_explicit=True,
    )
    verified = restrict_to_explicit_field_updates(
        "Change the reminder to call the doctor at 8 AM",
        PendingReminderFieldUpdate(task="call the doctor", hour=8, minute=0, meridiem="am"),
        context=context,
        now=NOW,
    )

    assert verified.task == "call the doctor"
    assert verified.hour == 8
    assert verified.meridiem == "am"


def test_standalone_meridiem_is_accepted_for_pending_time_clarification() -> None:
    context = PendingReminder(
        task="visit the dentist",
        date_local=date(2026, 10, 31),
        hour=8,
        timezone_name="UTC",
    )
    verified = restrict_to_explicit_field_updates(
        "PM",
        PendingReminderFieldUpdate(meridiem="PM"),
        context=context,
        now=NOW,
    )

    assert verified.meridiem == "pm"


def test_classifier_selected_update_can_replace_pending_task() -> None:
    context = PendingReminder(
        task="email Hassan",
        date_local=date(2026, 10, 31),
        hour=8,
        meridiem="pm",
        timezone_name="UTC",
    )
    verified = restrict_to_explicit_field_updates(
        "email Bilal",
        PendingReminderFieldUpdate(
            task="email Bilal", certainty="CLEAR", evidence={"task": "email Bilal"}
        ),
        context=context,
        now=NOW,
        allow_task_replacement=True,
            allow_semantic_field_updates=True,
    )

    assert verified.task == "email Bilal"


@pytest.mark.parametrize(
    ("message", "date_evidence"),
    [
        ("tomorow at 8 PM", "tomorrow"),
        ("tmorrow at 8 PM", "tomorrow"),
        ("at 8 PM tommorow", "tomorrow at 8 PM"),
    ],
)
def test_semantic_update_accepts_grounded_typo_or_word_order(
    message: str, date_evidence: str
) -> None:
    context = PendingReminder(
        task="call Ayesha",
        timezone_name="UTC",
    )
    verified = restrict_to_explicit_field_updates(
        message,
        PendingReminderFieldUpdate(
            date_local=date(2026, 10, 31),
            hour=8,
            minute=0,
            meridiem="PM",
            certainty="CLEAR",
            evidence={
                "date_local": date_evidence,
                "hour": date_evidence,
                "minute": date_evidence,
                "meridiem": date_evidence,
            },
        ),
        context=context,
        now=NOW,
        allow_semantic_field_updates=True,
    )

    assert verified.date_local == date(2026, 10, 31)
    assert verified.hour == 8
    assert verified.meridiem == "pm"


def test_semantic_update_does_not_accept_typo_without_classifier_authorization() -> None:
    context = PendingReminder(task="call Ayesha", timezone_name="UTC")
    verified = restrict_to_explicit_field_updates(
        "tomorow",
        PendingReminderFieldUpdate(date_local=date(2026, 10, 31)),
        context=context,
        now=NOW,
    )

    assert verified.date_local is None


def test_semantic_update_requires_clear_grounded_evidence() -> None:
    context = PendingReminder(task="call Ayesha", timezone_name="UTC")
    verified = restrict_to_explicit_field_updates(
        "tomorow",
        PendingReminderFieldUpdate(
            date_local=date(2026, 10, 31), certainty="CLEAR", evidence={"date_local": "today"}
        ),
        context=context,
        now=NOW,
        allow_semantic_field_updates=True,
    )

    assert verified.date_local is None


def test_semantic_update_rejects_a_different_temporal_word_as_evidence() -> None:
    context = PendingReminder(task="call Ayesha", timezone_name="UTC")
    verified = restrict_to_explicit_field_updates(
        "Sunday",
        PendingReminderFieldUpdate(
            date_local=date(2026, 11, 2), certainty="CLEAR", evidence={"date_local": "Monday"}
        ),
        context=context,
        now=NOW,
        allow_semantic_field_updates=True,
    )

    assert verified.date_local is None


def test_semantic_task_update_accepts_a_grounded_spelling_variation() -> None:
    context = PendingReminder(task="email Hassan", timezone_name="UTC")
    verified = restrict_to_explicit_field_updates(
        "emial Bilal",
        PendingReminderFieldUpdate(
            task="email Bilal", certainty="CLEAR", evidence={"task": "email Bilal"}
        ),
        context=context,
        now=NOW,
        allow_task_replacement=True,
        allow_semantic_field_updates=True,
    )

    assert verified.task == "email Bilal"


def test_clarification_question_mentions_only_the_missing_date() -> None:
    pending = PendingReminder(
        task="prepare slides",
        hour=4,
        meridiem="pm",
        timezone_name="UTC",
    )

    result = RuleBasedIntentInterpreter().apply_field_updates(
        pending, PendingReminderFieldUpdate(), now=NOW
    )

    assert result.clarification is not None
    assert result.clarification.pending.task == "prepare slides"
    assert result.clarification.pending.hour == 4
    assert result.clarification.pending.meridiem == "pm"
    assert result.clarification.pending.date_local is None
    assert result.clarification.question == "What date should I use?"


def test_complex_follow_up_uses_model_fallback_with_pending_context() -> None:
    class RecordingModel:
        def __init__(self) -> None:
            self.context = None

        def interpret(self, text, *, timezone_name, now, context=None):
            self.context = context
            return Interpretation(
                clarification=NeedsClarification(
                    pending=PendingReminder(
                        task="call the bank",
                        date_local=date(2026, 10, 30),
                        hour=7,
                        minute=0,
                        meridiem="pm",
                        timezone_name=timezone_name,
                    ),
                    question="What time should I use?",
                )
            )

    model = RecordingModel()
    context = PendingReminder(
        task="call the bank",
        date_local=date(2026, 10, 30),
        hour=7,
        minute=0,
        timezone_name="UTC",
    )
    result = ReminderIntentInterpreter(model).interpret(
        "Could you make sure this happens after dinner?",
        timezone_name="UTC",
        now=NOW,
        context=context,
    )

    assert model.context == context
    assert result.clarification is not None


def test_complex_natural_language_without_reminder_keyword_uses_model_fallback() -> None:
    class RecordingModel:
        called = False

        def interpret(self, text, *, timezone_name, now, context=None):
            self.called = True
            return Interpretation(
                intent=CreateOneTimeReminderIntent(
                    task="call John",
                    due_at=datetime(2026, 10, 30, 19, 0, tzinfo=timezone.utc),
                    timezone_name=timezone_name,
                )
            )

    model = RecordingModel()
    result = ReminderIntentInterpreter(model).interpret(
        "Could you make sure I call John around dinner time tonight?",
        timezone_name="UTC",
        now=NOW,
    )

    assert model.called is True
    assert result.intent is not None
    assert result.intent.task == "call John"


def test_multiple_people_in_one_task_are_not_split_into_multiple_reminders() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me to call Alice and Bob tomorrow at 9 AM",
        timezone_name="UTC",
        now=NOW,
    )

    assert result.clarification is None
    assert result.intent is not None
    assert result.intent.task == "call Alice and Bob"


def test_multiple_tasks_with_one_schedule_are_one_reminder() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me to call Alice and email Bob tomorrow at 9 AM",
        timezone_name="UTC",
        now=NOW,
    )

    assert result.clarification is None
    assert result.intent is not None
    assert result.intent.task == "call Alice and email Bob"


def test_multiple_tasks_with_one_schedule_inherit_the_same_time() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me to call Ali tomorrow at 9 AM and to pay the bill",
        timezone_name="UTC",
        now=NOW,
    )

    assert result.intent is not None
    assert result.intent.task == "call Ali and to pay the bill"
    assert result.intent.due_at == datetime(2026, 10, 31, 9, 0, tzinfo=timezone.utc)


def test_explicit_multiple_reminder_requests_are_parsed_independently() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me to call Alice tomorrow at 9 AM and remind me to email Bob tomorrow at 10 AM",
        timezone_name="UTC",
        now=NOW,
    )

    assert result.clarification is None
    assert result.intent is None
    assert len(result.resolved_intents) == 2
    assert [intent.task for intent in result.resolved_intents] == [
        "call Alice",
        "email Bob",
    ]
    assert [intent.due_at for intent in result.resolved_intents] == [
        datetime(2026, 10, 31, 9, 0, tzinfo=timezone.utc),
        datetime(2026, 10, 31, 10, 0, tzinfo=timezone.utc),
    ]


def test_shared_date_multiple_times_are_parsed_independently() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me tomorrow at 9 AM to pay rent and at 6 PM to call Sara",
        timezone_name="UTC",
        now=datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc),
    )

    assert [intent.task for intent in result.resolved_intents] == [
        "pay rent",
        "call Sara",
    ]
    assert [intent.due_at for intent in result.resolved_intents] == [
        datetime(2026, 11, 1, 9, 0, tzinfo=timezone.utc),
        datetime(2026, 11, 1, 18, 0, tzinfo=timezone.utc),
    ]


def test_compound_single_prefix_uses_model_fallback() -> None:
    class RecordingModel:
        called = False

        def interpret(self, text, *, timezone_name, now, context=None):
            self.called = True
            return Interpretation(
                intents=(
                    CreateOneTimeReminderIntent(
                        task="call Alice",
                        due_at=datetime(2026, 10, 31, 9, 0, tzinfo=timezone.utc),
                        timezone_name=timezone_name,
                    ),
                    CreateOneTimeReminderIntent(
                        task="email Bob",
                        due_at=datetime(2026, 10, 31, 10, 0, tzinfo=timezone.utc),
                        timezone_name=timezone_name,
                    ),
                )
            )

    model = RecordingModel()
    result = ReminderIntentInterpreter(model).interpret(
        "Remind me to call Alice tomorrow at 9 AM and email Bob tomorrow at 10 AM",
        timezone_name="UTC",
        now=NOW,
    )

    assert model.called is True
    assert [intent.task for intent in result.resolved_intents] == ["call Alice", "email Bob"]


def test_complete_single_command_supersedes_multi_target_clarification() -> None:
    interpreter = RuleBasedIntentInterpreter()
    initial = interpreter.interpret(
        "Remind me to call Alice and email Bob tomorrow at 9 AM",
        timezone_name="UTC",
        now=NOW,
    )

    assert initial.clarification is None
    assert initial.intent is not None
    assert initial.intent.task == "call Alice and email Bob"


def test_follow_up_accepts_minute_and_day_phrasing() -> None:
    interpreter = RuleBasedIntentInterpreter()
    initial = interpreter.interpret(
        "Remind me to call John at 6:55",
        timezone_name="UTC",
        now=datetime(2026, 10, 29, 15, 0, tzinfo=timezone.utc),
    )

    assert initial.clarification is not None
    completed = interpreter.interpret(
        "6:55 pm today",
        timezone_name="UTC",
        now=datetime(2026, 10, 29, 15, 0, tzinfo=timezone.utc),
        context=initial.clarification.pending,
    )

    assert completed.intent is not None
    assert completed.intent.due_at == datetime(2026, 10, 29, 18, 55, tzinfo=timezone.utc)


def test_bare_time_does_not_start_a_provider_clarification_session() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "7 pm",
        timezone_name="UTC",
        now=NOW,
    )

    assert result.intent is None
    assert result.clarification is None
    assert result.fallback_allowed is False


def test_friday_follow_up_keeps_the_explicit_future_date() -> None:
    interpreter = RuleBasedIntentInterpreter()
    thursday = datetime(2026, 10, 29, 13, 0, tzinfo=timezone.utc)
    initial = interpreter.interpret(
        "Remind me Friday to renew the domain",
        timezone_name="UTC",
        now=thursday,
    )

    assert initial.clarification is not None
    completed = interpreter.interpret(
        "8 a.m.",
        timezone_name="UTC",
        now=thursday,
        context=initial.clarification.pending,
    )

    assert completed.intent is not None
    assert completed.intent.due_at == datetime(2026, 10, 30, 8, 0, tzinfo=timezone.utc)


def test_past_follow_up_keeps_task_and_asks_for_a_new_future_time() -> None:
    interpreter = RuleBasedIntentInterpreter()
    initial = interpreter.interpret(
        "Remind me to stretch at 7",
        timezone_name="UTC",
        now=datetime(2026, 10, 29, 15, 0, tzinfo=timezone.utc),
    )

    assert initial.clarification is not None
    past = interpreter.interpret(
        "7 pm",
        timezone_name="UTC",
        now=datetime(2026, 10, 29, 19, 2, tzinfo=timezone.utc),
        context=initial.clarification.pending,
    )

    assert past.intent is None
    assert past.clarification is not None
    assert past.clarification.pending.task == "stretch"
    assert past.clarification.pending.date_local == date(2026, 10, 29)
    assert past.clarification.pending.hour is None
    assert "passed" in past.clarification.question

    completed = interpreter.interpret(
        "8 pm",
        timezone_name="UTC",
        now=datetime(2026, 10, 29, 19, 2, tzinfo=timezone.utc),
        context=past.clarification.pending,
    )
    assert completed.intent is not None
    assert completed.intent.due_at == datetime(2026, 10, 29, 20, 0, tzinfo=timezone.utc)


def test_explicit_past_date_keeps_task_and_time_and_asks_only_for_date() -> None:
    interpreter = RuleBasedIntentInterpreter()
    initial = interpreter.interpret(
        "Remind me Friday at 8 AM to call the bank",
        timezone_name="UTC",
        now=NOW,
    )

    assert initial.intent is None
    assert initial.clarification is not None
    assert initial.clarification.pending.task == "call the bank"
    assert initial.clarification.pending.date_local is None
    assert initial.clarification.pending.hour == 8
    assert initial.clarification.pending.meridiem == "am"
    assert "future date" in initial.clarification.question

    completed = interpreter.interpret(
        "Monday",
        timezone_name="UTC",
        now=NOW,
        context=initial.clarification.pending,
    )

    assert completed.intent is not None
    assert completed.intent.task == "call the bank"
    assert completed.intent.due_at == datetime(2026, 11, 2, 8, 0, tzinfo=timezone.utc)


def test_pydantic_intent_rejects_naive_due_time() -> None:
    try:
        CreateOneTimeReminderIntent(
            task="call",
            due_at="2026-10-30T12:05:00",
            timezone_name="UTC",
        )
    except ValueError as exc:
        assert "timezone" in str(exc)
    else:  # pragma: no cover - assertion makes the intended validation explicit
        raise AssertionError("naive due time was accepted")


def test_openai_compatible_interpreter_accepts_structured_response(monkeypatch) -> None:
    def fake_post(*args, **kwargs):
        assert args[0] == "https://llm.example/v1/chat/completions"
        assert kwargs["headers"]["Authorization"] == "Bearer secret"
        assert kwargs["json"]["model"] == "provider-model"
        return httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"operation":"CREATE_ONE_TIME_REMINDER",'
                                '"task":"call the bank","due_local":"2026-10-31T09:00:00"}'
                            )
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    result = OpenAICompatibleIntentInterpreter(
        "https://llm.example/v1",
        "secret",
        "provider-model",
    ).interpret(
        "Please remind me to call the bank tomorrow morning",
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert result.intent is not None
    assert result.intent.task == "call the bank"
    assert result.intent.due_at == datetime(2026, 10, 31, 4, 0, tzinfo=timezone.utc)


def test_openai_compatible_interpreter_returns_only_provider_field_updates_for_context(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"operation":"UPDATE_PENDING_REMINDER",'
                                '"updates":{"meridiem":"PM"}}'
                            )
                        }
                    }
                ]
            },
        ),
    )
    context = PendingReminder(
        task="call the bank",
        date_local=date(2026, 10, 30),
        hour=7,
        timezone_name="UTC",
    )

    result = OpenAICompatibleIntentInterpreter(
        "https://llm.example/v1", "secret", "provider-model"
    ).interpret("in the evening", timezone_name="UTC", now=NOW, context=context)

    assert result.intent is None
    assert result.clarification is None
    assert result.field_updates == PendingReminderFieldUpdate(meridiem="pm")

    # The service-side state manager, not the provider, combines this patch
    # with the pending task/date/time and only then validates completion.
    completed = RuleBasedIntentInterpreter().apply_field_updates(
        context, result.field_updates, now=NOW
    )
    assert completed.intent is not None
    assert completed.intent.task == "call the bank"
    assert completed.intent.due_at == datetime(2026, 10, 30, 19, 0, tzinfo=timezone.utc)


def test_openai_compatible_interpreter_preserves_pending_on_ambiguous_update(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"operation":"NEEDS_CLARIFICATION",'
                                '"question":"Which Friday do you mean?"}'
                            )
                        }
                    }
                ]
            },
        ),
    )
    context = PendingReminder(
        task="call Ayesha",
        date_local=date(2026, 10, 31),
        hour=8,
        meridiem="pm",
        timezone_name="UTC",
    )

    result = OpenAICompatibleIntentInterpreter(
        "https://llm.example/v1", "secret", "provider-model"
    ).interpret("next Friday", timezone_name="UTC", now=NOW, context=context)

    assert result.field_updates is None
    assert result.clarification is not None
    assert result.clarification.question == "Which Friday do you mean?"
    assert result.clarification.pending == context


def test_openai_compatible_interpreter_rejects_completed_reminder_during_clarification(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: httpx.Response(
            200,
            request=httpx.Request("POST", args[0]),
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"operation":"CREATE_ONE_TIME_REMINDER",'
                                '"task":"wrong task",'
                                '"due_local":"2026-10-31T08:00:00"}'
                            )
                        }
                    }
                ]
            },
        ),
    )
    context = PendingReminder(
        task="call the bank",
        date_local=date(2026, 10, 30),
        hour=7,
        timezone_name="UTC",
    )

    result = OpenAICompatibleIntentInterpreter(
        "https://llm.example/v1", "secret", "provider-model"
    ).interpret("8 AM", timezone_name="UTC", now=NOW, context=context)

    assert result.intent is None
    assert result.field_updates is None


def test_openai_compatible_interpreter_rejects_offset_bearing_local_time(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: httpx.Response(
            200,
            request=httpx.Request("POST", "https://llm.example/v1/chat/completions"),
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"operation":"CREATE_ONE_TIME_REMINDER",'
                                '"task":"call the bank",'
                                '"due_local":"2026-10-31T09:00:00+00:00"}'
                            )
                        }
                    }
                ]
            },
        ),
    )

    result = OpenAICompatibleIntentInterpreter(
        "https://llm.example/v1", "secret", "provider-model"
    ).interpret("remind me", timezone_name="Asia/Karachi", now=NOW)

    assert result.intent is None
    assert result.response_text


def test_openai_compatible_interpreter_rejects_unknown_operation(monkeypatch) -> None:
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: httpx.Response(
            200,
            request=httpx.Request("POST", "https://llm.example/v1/chat/completions"),
            json={"choices": [{"message": {"content": '{"operation":"DELETE_ALL"}'}}]},
        ),
    )

    result = OpenAICompatibleIntentInterpreter(
        "https://llm.example/v1", "secret", "provider-model"
    ).interpret("delete my reminders", timezone_name="UTC", now=NOW)

    assert result.intent is None
    assert result.response_text == OUT_OF_SCOPE_RESPONSE
