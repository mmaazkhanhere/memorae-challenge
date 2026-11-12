from datetime import UTC, datetime, timedelta

from reminder_agent.application import (
    ApplicationRequest,
    AuthenticatedRequestContext,
    CancelReminderCommand,
    CompleteReminderCommand,
    DeliveredReminderContext,
    OutboundResponse,
    RequestSource,
    RescheduleReminderCommand,
    SnoozeReminderCommand,
    VerifiedChannelIdentity,
    cancel_reminder_for_request,
    complete_reminder_for_request,
    prepare_phone_link,
    reschedule_reminder_for_request,
    snooze_reminder_for_request,
)
from reminder_agent.features.onboarding.identity_operations import link_phone_identity
from reminder_agent.features.reminders.cancel.decisions import CancellationScope
from reminder_agent.features.reminders.cancel.messages import REMINDER_CANCEL_MESSAGES
from reminder_agent.features.reminders.complete.messages import REMINDER_COMPLETION_MESSAGES
from reminder_agent.features.reminders.reschedule.decisions import RescheduleScope
from reminder_agent.infrastructure.database.models import (
    ChannelIdentity,
    Notification,
    Reminder,
    ReminderOccurrence,
    User,
)

NOW = datetime(2026, 11, 3, 12, tzinfo=UTC)


def _phone_request(user, identity, *, key: str) -> ApplicationRequest:
    return ApplicationRequest(
        context=AuthenticatedRequestContext(
            source=RequestSource.WEB_API,
            canonical_user_id=user.id,
            verified_identity=VerifiedChannelIdentity(
                identity_id=identity.id,
                channel_type="PHONE",
            ),
            external_session_id="phone-session:ada",
        ),
        message_text="done",
        idempotency_key=key,
    )


def _seed_phone_delivery(db, *, suffix: str = "ada"):
    user = User(name="Ada", timezone="UTC", onboarding_status="ACTIVE")
    db.add(user)
    db.flush()
    identity = ChannelIdentity(
        user_id=user.id,
        channel_type="PHONE",
        external_id=f"phone:{suffix}",
        status="LINKED",
    )
    db.add(identity)
    db.flush()
    reminder = Reminder(
        user_id=user.id,
        task="call the bank",
        schedule_type="ONE_TIME",
        start_local="2026-11-03T12:00:00",
        timezone="UTC",
    )
    db.add(reminder)
    db.flush()
    occurrence = ReminderOccurrence(
        reminder_id=reminder.id,
        scheduled_at=NOW - timedelta(minutes=1),
        next_due_at=NOW - timedelta(minutes=1),
        status="NOTIFIED",
    )
    db.add(occurrence)
    db.flush()
    notification = Notification(
        occurrence_id=occurrence.id,
        destination_identity_id=identity.id,
        sequence=1,
        due_at=occurrence.scheduled_at,
        status="SENT",
        sent_at=occurrence.scheduled_at,
        provider_message_id="phone-message-1",
    )
    db.add(notification)
    db.flush()
    return user, identity, reminder, occurrence, notification


def test_fake_channel_completion_uses_verified_delivery_context_and_replay_is_safe(db):
    user, identity, reminder, occurrence, _notification = _seed_phone_delivery(db)
    request = _phone_request(user, identity, key="phone-complete-1")
    command = CompleteReminderCommand(
        occurrence_id=occurrence.id,
        delivery=DeliveredReminderContext(
            identity_id=identity.id,
            channel_type="PHONE",
            external_destination_id="phone:ada",
            provider_message_id="phone-message-1",
        ),
    )

    first = complete_reminder_for_request(db, request=request, command=command, now=NOW)
    second = complete_reminder_for_request(db, request=request, command=command, now=NOW)

    assert first.response_kind == "completion"
    assert isinstance(first, OutboundResponse)
    assert first.content in {
        message.format(task="call the bank")
        for message in REMINDER_COMPLETION_MESSAGES["reminder_completed"]
    }
    assert second.content in set(REMINDER_COMPLETION_MESSAGES["reminder_already_completed"])
    assert db.get(Reminder, reminder.id).status == "COMPLETED"


def test_fake_channel_snooze_rechecks_identity_ownership(db):
    user, identity, _reminder, occurrence, _notification = _seed_phone_delivery(db)
    other = User(name="Bob", timezone="UTC", onboarding_status="ACTIVE")
    db.add(other)
    db.flush()
    forged = ApplicationRequest(
        context=AuthenticatedRequestContext(
            source=RequestSource.WEB_API,
            canonical_user_id=other.id,
            verified_identity=VerifiedChannelIdentity(
                identity_id=identity.id,
                channel_type="PHONE",
            ),
            external_session_id="phone-session:bob",
        ),
        message_text="snooze",
        idempotency_key="phone-snooze-forged",
    )
    command = SnoozeReminderCommand(
        occurrence_id=occurrence.id,
        duration_seconds=600,
        delivery=DeliveredReminderContext(
            identity_id=identity.id,
            channel_type="PHONE",
            external_destination_id="phone:ada",
            provider_message_id="phone-message-1",
        ),
    )

    result = snooze_reminder_for_request(db, request=forged, command=command, now=NOW)

    assert result.response_kind == "AUTHORIZATION_FAILURE"
    assert occurrence.status == "NOTIFIED"


def test_fake_channel_cancellation_and_rescheduling_use_application_commands(db):
    user, identity, reminder, occurrence, notification = _seed_phone_delivery(db)
    occurrence.status = "PENDING"
    notification.status = "PENDING"
    request = _phone_request(user, identity, key="phone-mutation-1")

    canceled = cancel_reminder_for_request(
        db,
        request=request,
        command=CancelReminderCommand(
            reminder_id=reminder.id,
            occurrence_id=occurrence.id,
            scope=CancellationScope.OCCURRENCE,
        ),
        now=NOW,
    )

    assert canceled.response_kind == "CANCELLATION_REMINDER"
    assert canceled.content in {
        template.format(task="call the bank")
        for template in REMINDER_CANCEL_MESSAGES["one_time_cancelled"]
    }

    user2, identity2, reminder2, occurrence2, notification2 = _seed_phone_delivery(
        db, suffix="ada-2"
    )
    occurrence2.status = "PENDING"
    notification2.status = "PENDING"
    rescheduled = reschedule_reminder_for_request(
        db,
        request=_phone_request(user2, identity2, key="phone-mutation-2"),
        command=RescheduleReminderCommand(
            reminder_id=reminder2.id,
            occurrence_id=occurrence2.id,
            due_at=NOW + timedelta(hours=2),
            scope=RescheduleScope.OCCURRENCE,
        ),
        now=NOW,
    )

    assert rescheduled.response_kind == "RESCHEDULE_REMINDER"
    assert rescheduled.resource_id == reminder2.id


def test_fake_channel_onboarding_links_identity_without_telegram(db):
    user = User(name="Ada", timezone="UTC", onboarding_status="ACTIVE")
    db.add(user)
    db.flush()

    prepared = prepare_phone_link(
        db,
        user=user,
        phone_number="+44 20 8366 1177",
        default_region=None,
    )
    linked = link_phone_identity(db, user=user, normalized=prepared.normalized or "")

    assert prepared.outcome == "NORMALIZED"
    assert prepared.normalized == "+442083661177"
    assert linked.outcome == "LINKED"

    # A non-Telegram channel consumes the semantic result without rendering a
    # provider keyboard or callback payload.
    response = OutboundResponse(
        content="Phone linked.",
        response_kind=linked.outcome,
        presentation_hint="REMOVE_KEYBOARD",
    )
    assert response.content == "Phone linked."
    assert response.presentation_hint == "REMOVE_KEYBOARD"
