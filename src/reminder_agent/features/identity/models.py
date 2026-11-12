"""Provider-neutral identity value objects.

These types deliberately do not mirror SQLAlchemy models.  They describe the
identity concepts used by application policies without introducing a second
persistence model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class IdentityType(StrEnum):
    """Supported persisted identity kinds."""

    TELEGRAM = "TELEGRAM"
    PHONE = "PHONE"


class IdentityStatus(StrEnum):
    """Lifecycle states for an external identity."""

    PENDING = "PENDING"
    LINKED = "LINKED"
    UNLINKED = "UNLINKED"


@dataclass(frozen=True)
class IdentityLinkDecision:
    """Pure result of evaluating an identity-link request."""

    allowed: bool
    reason: str | None = None
