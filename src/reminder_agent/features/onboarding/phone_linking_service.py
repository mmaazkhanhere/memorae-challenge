"""Application service that injects identity persistence into phone linking."""

from __future__ import annotations

from reminder_agent.application.identity.link import link_phone_identity, prepare_phone_link
from reminder_agent.features.onboarding.phone_linking import (
    PhoneLinkDecision,
)
from reminder_agent.features.onboarding.phone_linking import (
    process_phone_link as _process_phone_link,
)


def process_phone_link(
    db,
    *,
    user,
    phone_number: str,
    default_region: str | None,
    pending=None,
) -> PhoneLinkDecision:
    """Link a phone through application-owned persistence ports."""

    return _process_phone_link(
        db,
        user=user,
        phone_number=phone_number,
        default_region=default_region,
        pending=pending,
        prepare_phone_link_fn=prepare_phone_link,
        link_phone_identity_fn=link_phone_identity,
    )


__all__ = ["PhoneLinkDecision", "process_phone_link"]
