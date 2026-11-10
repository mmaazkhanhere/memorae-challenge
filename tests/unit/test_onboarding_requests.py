
from sqlalchemy import select

from reminder_agent.application.contracts import RequestSource
from reminder_agent.channels.telegram.provider import TelegramUpdate
from reminder_agent.channels.telegram.updates import to_onboarding_request
from reminder_agent.features.onboarding.request_contracts import (
    OnboardingAction,
    OnboardingIdentityContext,
    OnboardingRequest,
    classify_onboarding_text,
)
from reminder_agent.features.onboarding.service import handle_onboarding_request
from reminder_agent.infrastructure.database.models import User


def test_onboarding_command_classifier_is_channel_neutral() -> None:
    assert classify_onboarding_text("/start") == (OnboardingAction.START, None)
    assert classify_onboarding_text("/link_phone") == (OnboardingAction.LINK_PHONE, None)
    assert classify_onboarding_text("skip for now") == (OnboardingAction.SKIP_PHONE, None)
    assert classify_onboarding_text("/phone_region PK") == (
        OnboardingAction.PHONE_REGION,
        "PK",
    )
    assert classify_onboarding_text("03001234567", has_contact=True) == (
        OnboardingAction.CONTACT,
        None,
    )
    assert classify_onboarding_text("voice input", has_voice=True) == (
        OnboardingAction.VOICE,
        None,
    )


def test_fake_channel_can_construct_onboarding_request_without_telegram() -> None:
    request = OnboardingRequest(
        source=RequestSource.WEB_API,
        identity=OnboardingIdentityContext(
            channel_type="PHONE",
            external_id="+442083661177",
            destination_id="fake-phone-session",
        ),
        idempotency_key="fake-onboarding-1",
        request_hash="stable-request-hash",
        action=OnboardingAction.LINK_PHONE,
    )

    assert request.identity.channel_type == "PHONE"
    assert request.action is OnboardingAction.LINK_PHONE
    assert request.request_hash == "stable-request-hash"


def test_telegram_adapter_translates_transport_fields_only() -> None:
    update = TelegramUpdate.model_validate(
        {
            "update_id": 901,
            "message": {
                "message_id": 901,
                "from": {"id": 1001, "first_name": "Ada"},
                "chat": {"id": 2001, "type": "private"},
                "text": "/phone_region PK",
            },
        }
    )

    request = to_onboarding_request(update)

    assert request is not None
    assert request.action is OnboardingAction.PHONE_REGION
    assert request.argument == "PK"
    assert request.identity.external_id == "1001"
    assert request.identity.destination_id == "2001"
    assert request.request_hash


def test_onboarding_application_service_accepts_a_nontelegram_request(
    session_factory, fixed_now
) -> None:
    identity = OnboardingIdentityContext(
        channel_type="PHONE",
        external_id="+442083661177",
        destination_id="phone-session-1",
    )
    start = OnboardingRequest(
        source=RequestSource.VAPI,
        identity=identity,
        idempotency_key="voice-onboarding-1",
        request_hash="voice-onboarding-hash-1",
        action=OnboardingAction.START,
        display_name="Ada",
    )
    with session_factory() as db:
        response = handle_onboarding_request(db, start, now=fixed_now)

    assert response is not None
    assert "Ada" in response.content
    assert "time zone" in response.content.lower()

    timezone_request = OnboardingRequest(
        source=RequestSource.VAPI,
        identity=identity,
        idempotency_key="voice-onboarding-2",
        request_hash="voice-onboarding-hash-2",
        action=OnboardingAction.MESSAGE,
        text="UTC",
    )
    with session_factory() as db:
        response = handle_onboarding_request(db, timezone_request, now=fixed_now)
        user = db.scalar(select(User).where(User.name == "Ada"))

    assert response is not None
    assert response.presentation_hint == "PHONE_CONTACT_REQUEST"
    assert user is not None
    assert user.onboarding_status == "ACTIVE"
    assert user.timezone == "UTC"
