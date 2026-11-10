import pytest

from reminder_agent.features.onboarding.onboarding_validation import (
    InvalidTimezone,
    validate_timezone,
)


def test_valid_iana_timezone_is_accepted() -> None:
    assert validate_timezone(" Asia/Karachi ") == "Asia/Karachi"


@pytest.mark.parametrize("value", ["", "not/a-timezone", "PST", "UTC+5"])
def test_invalid_timezone_is_rejected(value: str) -> None:
    with pytest.raises(InvalidTimezone):
        validate_timezone(value)
