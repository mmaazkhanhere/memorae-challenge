"""Channel-neutral application contracts and feature-oriented operations.

Channel adapters authenticate provider requests and resolve canonical users
before constructing :class:`AuthenticatedRequestContext`. Services revalidate
that persisted relationship and never trust LLM or provider payload identity.
"""

from reminder_agent.application.contracts import (
    ApplicationRequest,
    ApplicationResult,
    AuthenticatedRequestContext,
    AuthenticatedUserContext,
    CancelReminderCommand,
    CompleteReminderCommand,
    DeliveredReminderContext,
    DurableResponse,
    IncomingMessage,
    ListRemindersCommand,
    OutboundResponse,
    OutgoingResponse,
    RequestSource,
    RescheduleReminderCommand,
    SaveMemoryCommand,
    SnoozeReminderCommand,
    VerifiedChannelIdentity,
)
from reminder_agent.application.conversation import (
    ConversationEntryResult,
    interpret_for_route,
    process_authenticated_conversation,
)
from reminder_agent.application.delivery import (
    ResponseDeliveryPort,
    SqlAlchemyResponseDelivery,
)
from reminder_agent.application.identity.link import (
    PhoneLinkPreparation,
    prepare_phone_link,
)
from reminder_agent.application.identity.resolve import (
    ResolvedIdentity as ResolvedLinkedIdentity,
)
from reminder_agent.application.identity.resolve import (
    resolve_linked_identity,
)
from reminder_agent.features.memory.saving import save_memory_for_request
from reminder_agent.features.onboarding.phone_linking_service import (
    PhoneLinkDecision,
    process_phone_link,
)
from reminder_agent.features.onboarding.request_contracts import (
    OnboardingAction,
    OnboardingIdentityContext,
    OnboardingRequest,
    classify_onboarding_text,
)
from reminder_agent.features.onboarding.service import handle_onboarding_request
from reminder_agent.features.reminders.cancel.service import cancel_reminder_for_request
from reminder_agent.features.reminders.listing.service import (
    list_reminders_for_request,
    list_reminders_from_interpretation,
)
from reminder_agent.features.reminders.reschedule.service import (
    reschedule_reminder_for_request,
    reschedule_reminder_for_user,
)
from reminder_agent.features.reminders.service import (
    complete_reminder_for_request,
    snooze_reminder_for_request,
)
from reminder_agent.features.reminders.snooze.service import handle_snooze_clarification
from reminder_agent.features.reminders.summary.service import (
    summarize_reminders_from_interpretation,
)

__all__ = [
    "ApplicationRequest",
    "ApplicationResult",
    "AuthenticatedRequestContext",
    "AuthenticatedUserContext",
    "CancelReminderCommand",
    "CompleteReminderCommand",
    "ConversationEntryResult",
    "DurableResponse",
    "DeliveredReminderContext",
    "IncomingMessage",
    "ListRemindersCommand",
    "RequestSource",
    "RescheduleReminderCommand",
    "SaveMemoryCommand",
    "SnoozeReminderCommand",
    "OutgoingResponse",
    "OutboundResponse",
    "ResponseDeliveryPort",
    "SqlAlchemyResponseDelivery",
    "PhoneLinkPreparation",
    "PhoneLinkDecision",
    "OnboardingAction",
    "OnboardingIdentityContext",
    "OnboardingRequest",
    "ResolvedLinkedIdentity",
    "VerifiedChannelIdentity",
    "cancel_reminder_for_request",
    "complete_reminder_for_request",
    "handle_snooze_clarification",
    "interpret_for_route",
    "list_reminders_for_request",
    "list_reminders_from_interpretation",
    "reschedule_reminder_for_user",
    "reschedule_reminder_for_request",
    "snooze_reminder_for_request",
    "summarize_reminders_from_interpretation",
    "process_authenticated_conversation",
    "prepare_phone_link",
    "process_phone_link",
    "handle_onboarding_request",
    "classify_onboarding_text",
    "resolve_linked_identity",
    "save_memory_for_request",
]
