import pytest

from reminder_agent.onboarding import (
    InvalidPhoneNumber,
    InvalidPhoneRegion,
    normalize_phone_number,
    validate_phone_region,
)


def test_normalize_international_phone_to_e164() -> None:
    assert normalize_phone_number(" +44 20 8366 1177 ") == "+442083661177"


def test_normalize_national_phone_with_configured_region() -> None:
    assert normalize_phone_number("03001234567", default_region="pk") == "+923001234567"


@pytest.mark.parametrize("value", ["", "not-a-number", "12345", "+999123456"])
def test_invalid_phone_is_rejected(value: str) -> None:
    with pytest.raises(InvalidPhoneNumber):
        normalize_phone_number(value)


def test_national_phone_without_default_region_is_rejected() -> None:
    with pytest.raises(InvalidPhoneNumber):
        normalize_phone_number("03001234567")


def test_phone_region_is_normalized_to_uppercase() -> None:
    assert validate_phone_region(" pk ") == "PK"


def test_unknown_phone_region_is_rejected() -> None:
    with pytest.raises(InvalidPhoneRegion):
        validate_phone_region("ZZ")
