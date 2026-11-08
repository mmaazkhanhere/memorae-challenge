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
