from reminder_agent.features.reminders.reminder_creation_response import stage_creation_response


def test_creation_response_staging_preserves_joined_text_and_response_kind() -> None:
    staged = stage_creation_response(
        confirmations=("first confirmation", "second confirmation"),
        stage_response=lambda **kwargs: kwargs,
    )

    assert staged == {
        "text": "first confirmation\nsecond confirmation",
        "response_kind": "confirmation",
    }
