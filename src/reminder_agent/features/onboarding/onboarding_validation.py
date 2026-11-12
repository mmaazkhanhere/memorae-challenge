"""Pure timezone and phone validation for onboarding."""

from __future__ import annotations

from zoneinfo import ZoneInfo, available_timezones

import phonenumbers
from phonenumbers import NumberParseException


class InvalidTimezone(ValueError):
    pass


class InvalidPhoneNumber(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class InvalidPhoneRegion(ValueError):
    pass


def validate_timezone(value: str) -> str:
    candidate = value.strip()
    if not candidate or candidate not in available_timezones():
        raise InvalidTimezone(candidate)
    try:
        ZoneInfo(candidate)
    except Exception as exc:  # pragma: no cover - defensive guard for platform tzdata
        raise InvalidTimezone(candidate) from exc
    return candidate


def normalize_phone_number(value: str, *, default_region: str | None = None) -> str:
    """Return a valid phone number in E.164 form without retaining raw input."""
    candidate = value.strip()
    if not candidate:
        raise InvalidPhoneNumber("empty phone number")
    region = default_region.strip().upper() if default_region else None
    if (
        region is None
        and not candidate.startswith("+")
        and any(character.isdigit() for character in candidate)
    ):
        raise InvalidPhoneNumber("missing_region")
    try:
        parsed = phonenumbers.parse(candidate, region)
    except NumberParseException as exc:
        raise InvalidPhoneNumber("unparseable") from exc
    if not phonenumbers.is_possible_number(parsed) or not phonenumbers.is_valid_number(parsed):
        raise InvalidPhoneNumber("invalid")
    normalized = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    if not normalized.startswith("+"):
        raise InvalidPhoneNumber("phone number is not international")
    return normalized


def validate_phone_region(value: str) -> str:
    region = value.strip().upper()
    if region not in phonenumbers.SUPPORTED_REGIONS:
        raise InvalidPhoneRegion(region)
    return region


def phone_shape(value: str) -> dict[str, int | bool]:
    """Return non-sensitive shape diagnostics for observability."""
    stripped = value.strip()
    return {
        "phone_length": len(stripped),
        "phone_digit_count": sum(character.isdigit() for character in stripped),
        "phone_has_plus": stripped.startswith("+"),
    }
