"""Telegram keyboard builders."""

PHONE_KEYBOARD = {
    "keyboard": [
        [{"text": "Share my phone number", "request_contact": True}],
        [{"text": "Skip for now"}],
    ],
    "resize_keyboard": True,
    "one_time_keyboard": True,
}
REMOVE_KEYBOARD = {"remove_keyboard": True}


def done_keyboard(callback_data: str) -> dict:
    """Build the inline completion action for a delivered reminder."""

    return {"inline_keyboard": [[{"text": "Done", "callback_data": callback_data}]]}
