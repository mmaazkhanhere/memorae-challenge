"""Pure business rules for identity linking and lifecycle transitions."""

from __future__ import annotations

from reminder_agent.features.identity.models import (
    IdentityLinkDecision,
    IdentityStatus,
    IdentityType,
)


def can_link_identity(
    *,
    identity_type: IdentityType | str,
    verified: bool,
    already_owned_by_other_user: bool,
    already_linked_to_user: bool = False,
) -> IdentityLinkDecision:
    """Decide whether verified identity evidence may be linked.

    Matching claims alone never authorize a link.  Persistence uniqueness and
    transaction handling remain application/infrastructure responsibilities.
    """

    if not verified:
        return IdentityLinkDecision(False, "verification_required")
    if already_owned_by_other_user:
        return IdentityLinkDecision(False, "identity_owned_by_other_user")
    if already_linked_to_user:
        return IdentityLinkDecision(True, "already_linked")
    if str(identity_type) not in {item.value for item in IdentityType}:
        return IdentityLinkDecision(False, "unsupported_identity_type")
    return IdentityLinkDecision(True)


def can_transition_identity(*, current: IdentityStatus | str, target: IdentityStatus | str) -> bool:
    """Allow only valid identity lifecycle transitions."""

    current_value = str(current)
    target_value = str(target)
    if current_value == target_value:
        return True
    return (current_value, target_value) in {
        (IdentityStatus.PENDING.value, IdentityStatus.LINKED.value),
        (IdentityStatus.PENDING.value, IdentityStatus.UNLINKED.value),
        (IdentityStatus.LINKED.value, IdentityStatus.UNLINKED.value),
        (IdentityStatus.UNLINKED.value, IdentityStatus.LINKED.value),
    }
