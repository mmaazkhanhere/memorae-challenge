from datetime import datetime, timezone

import httpx

from reminder_agent.intent import (
    CreateOneTimeReminderIntent,
    OpenAICompatibleIntentInterpreter,
    RuleBasedIntentInterpreter,
)
from reminder_agent.settings import Settings


NOW = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)


def test_default_llm_configuration_targets_openai_compatible_api() -> None:
    settings = Settings(_env_file=None)

    assert settings.llm_base_url == "https://api.openai.com/v1"
    assert settings.llm_model == "gpt-6-luna"


def test_relative_minutes_are_resolved_from_one_reference_time() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me in five minutes to call the bank.",
        timezone_name="Asia/Karachi",
        now=NOW,
    )

    assert result.intent is not None
    assert result.intent.task == "call the bank"
    assert result.intent.due_at == datetime(2026, 10, 30, 12, 5, tzinfo=timezone.utc)


def test_tomorrow_at_nine_respects_user_local_timezone() -> None:
    result = RuleBasedIntentInterpreter().interpret(
        "Remind me tomorrow at 9 to renew the domain.",
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
    assert result.response_text
