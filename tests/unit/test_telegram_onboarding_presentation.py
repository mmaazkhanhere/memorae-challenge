from types import SimpleNamespace

from reminder_agent.application.contracts import OutboundResponse
from reminder_agent.channels.telegram import webhook
from reminder_agent.channels.telegram.outbound import (
    PHONE_KEYBOARD,
    REMOVE_KEYBOARD,
)


def test_semantic_onboarding_hints_render_existing_telegram_keyboards(update_factory):
    update = webhook.TelegramUpdate.model_validate(update_factory(1801, text="/start"))

    phone_response = OutboundResponse(
        content="Share your phone.",
        response_kind="PHONE_LINK_PROMPT",
        presentation_hint="PHONE_CONTACT_REQUEST",
    )
    remove_response = OutboundResponse(
        content="Your phone is linked.",
        response_kind="PHONE_LINKED",
        presentation_hint="REMOVE_KEYBOARD",
    )

    assert webhook._telegram_reply_markup(None, update, phone_response) == PHONE_KEYBOARD
    assert webhook._telegram_reply_markup(None, update, remove_response) == REMOVE_KEYBOARD


def test_legacy_staged_onboarding_markup_is_replayed_unchanged(monkeypatch, update_factory):
    update = webhook.TelegramUpdate.model_validate(update_factory(1802, text="/start"))
    legacy_markup = {"keyboard": [[{"text": "legacy"}]], "resize_keyboard": True}
    monkeypatch.setattr(
        webhook,
        "find_idempotency_record",
        lambda *_args, **_kwargs: SimpleNamespace(response={"reply_markup": legacy_markup}),
    )
    response = OutboundResponse(
        content="legacy",
        response_kind="REPLAY",
        presentation_hint="LEGACY_STAGED",
    )

    assert webhook._telegram_reply_markup(None, update, response) == legacy_markup
