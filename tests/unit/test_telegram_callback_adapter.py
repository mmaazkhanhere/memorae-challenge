from uuid import uuid4

from reminder_agent.channels.telegram.callbacks import callback_data
from reminder_agent.channels.telegram.completion_callbacks import parse_completion_callback
from reminder_agent.channels.telegram.provider import TelegramUpdate


def _update(data: str, *, chat_type: str = "private") -> TelegramUpdate:
    return TelegramUpdate.model_validate(
        {
            "update_id": 42,
            "callback_query": {
                "id": "callback-42",
                "from": {"id": 1001},
                "data": data,
                "message": {
                    "message_id": 77,
                    "from": {"id": 999},
                    "chat": {"id": 2001, "type": chat_type},
                    "text": "Reminder",
                },
            },
        }
    )


def test_callback_adapter_normalizes_signed_completion_payload() -> None:
    occurrence_id = uuid4()

    result = parse_completion_callback(
        _update(callback_data(occurrence_id, "secret")),
        callback_secret="secret",
    )

    assert result is not None
    assert result.occurrence_id == occurrence_id
    assert result.chat_id == 2001
    assert result.provider_message_id == "77"


def test_callback_adapter_rejects_forged_and_unsupported_payloads() -> None:
    signed = callback_data(uuid4(), "secret")
    forged = signed[:-1] + ("0" if signed[-1] != "0" else "1")

    assert parse_completion_callback(_update(forged), callback_secret="secret") is None
    assert (
        parse_completion_callback(_update("cancel:unsupported"), callback_secret="secret") is None
    )


def test_callback_adapter_rejects_non_private_callbacks() -> None:
    result = parse_completion_callback(
        _update(callback_data(uuid4(), "secret"), chat_type="group"),
        callback_secret="secret",
    )

    assert result is None
