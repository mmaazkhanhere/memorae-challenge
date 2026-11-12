from reminder_agent.features.onboarding import message_templates


def test_message_catalog_contains_variants_for_each_onboarding_operation() -> None:
    assert message_templates.ONBOARDING_MESSAGES
    assert all(messages for messages in message_templates.ONBOARDING_MESSAGES.values())


def test_render_message_selects_and_interpolates_a_variant(monkeypatch) -> None:
    monkeypatch.setattr(
        message_templates.random,
        "choice",
        lambda messages: messages[0],
    )

    rendered = message_templates.render_message("welcome", name="Ada")

    assert rendered == message_templates.ONBOARDING_MESSAGES["welcome"][0].format(name="Ada")


def test_reminder_example_is_shown_after_phone_decision() -> None:
    example = "Remind me to stretch in 5 minutes."

    assert all(example not in message for message in message_templates.ONBOARDING_MESSAGES["onboarding_complete"])
    assert all(example in message for message in message_templates.ONBOARDING_MESSAGES["phone_linked"])
    assert all(example in message for message in message_templates.ONBOARDING_MESSAGES["phone_link_skipped"])
