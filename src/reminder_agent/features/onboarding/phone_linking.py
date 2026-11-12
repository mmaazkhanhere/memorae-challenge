"""Pure onboarding phone-link decision orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PhoneLinkDecision:
    """Validated phone-linking outcome for any communication channel."""

    outcome: str
    normalized: str | None = None
    reason: str | None = None
    region: str | None = None
    replaced_existing: bool = False


def process_phone_link(
    db: Any,
    *,
    user: Any,
    phone_number: str,
    default_region: str | None,
    pending: Any = None,
    prepare_phone_link_fn: Callable[..., Any],
    link_phone_identity_fn: Callable[..., Any],
) -> PhoneLinkDecision:
    """Coordinate validation and linking through injected application ports.

    Persistence, identity ownership, and transaction policy stay in the
    application services supplied by the caller. This feature-level function
    contains only the provider-neutral decision and result mapping.
    """

    preparation = prepare_phone_link_fn(
        db,
        user=user,
        phone_number=phone_number,
        default_region=default_region,
    )
    if preparation.outcome != "NORMALIZED":
        return PhoneLinkDecision(
            outcome=preparation.outcome,
            reason=preparation.reason,
            region=preparation.region,
        )

    assert preparation.normalized is not None
    mutation = link_phone_identity_fn(
        db,
        user=user,
        normalized=preparation.normalized,
        pending=pending,
    )
    return PhoneLinkDecision(
        outcome=mutation.outcome,
        normalized=preparation.normalized,
        reason=mutation.reason,
        region=preparation.region,
        replaced_existing=mutation.replaced_existing,
    )


__all__ = ["PhoneLinkDecision", "process_phone_link"]
