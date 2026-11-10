from __future__ import annotations

from datetime import timedelta

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from reminder_agent.application.main import create_app
from reminder_agent.application.voice_call_authorization import resolve_vapi_call_context
from reminder_agent.infrastructure.config.settings import Settings, validate_runtime_settings
from reminder_agent.infrastructure.database.models import (
    ChannelIdentity,
    User,
    VoiceCallAuthorization,
)

PHONE = "+923001234567"
TOKEN = "vapi-test-token-" + "x" * 32


def _settings(*, enabled: bool = True) -> Settings:
    return Settings(
        _env_file=None,
        database_url="sqlite://",
        telegram_bot_token="test-token",
        telegram_webhook_secret="test-secret",
        vapi_enabled=enabled,
        vapi_webhook_bearer_token=TOKEN,
        llm_api_key=None,
        embedding_provider="disabled",
        embedding_api_key=None,
    )


def _seed_linked_user(session_factory, *, phone: str = PHONE) -> tuple[User, ChannelIdentity]:
    db = session_factory()
    try:
        user = User(
            name="Ada",
            timezone="UTC",
            onboarding_status="ACTIVE",
        )
        db.add(user)
        db.flush()
        identity = ChannelIdentity(
            user_id=user.id,
            channel_type="PHONE",
            external_id=phone,
            status="LINKED",
        )
        db.add(identity)
        db.commit()
        return user, identity
    finally:
        db.close()


def _payload(call_id: str = "call-test-1", phone: str | None = PHONE) -> dict:
    caller = {} if phone is None else {"phoneNumber": phone}
    return {
        "message": {
            "type": "assistant-request",
            "call": {"id": call_id, "from": caller},
        }
    }


def _customer_payload(call_id: str = "call-customer-1", phone: str = PHONE) -> dict:
    return {
        "message": {
            "type": "assistant-request",
            "call": {"id": call_id, "customer": {"number": phone}},
        }
    }


def test_authorized_assistant_request_creates_one_server_side_binding(
    session_factory, fixed_now
) -> None:
    _seed_linked_user(session_factory)
    app = create_app(
        _settings(),
        session_factory=session_factory,
        clock=lambda: fixed_now,
    )

    with TestClient(app) as client:
        response = client.post(
            "/webhook/vapi",
            json=_payload(),
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["assistant"]["name"] == "Memorae"
    assert "server" not in body["assistant"]
    rendered = response.text
    assert PHONE not in rendered
    db = session_factory()
    try:
        assert db.scalar(select(func.count()).select_from(VoiceCallAuthorization)) == 1
    finally:
        db.close()


def test_replayed_assistant_request_is_idempotent(session_factory, fixed_now) -> None:
    _seed_linked_user(session_factory)
    app = create_app(_settings(), session_factory=session_factory, clock=lambda: fixed_now)
    with TestClient(app) as client:
        first = client.post(
            "/webhook/vapi",
            json=_payload(),
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        second = client.post(
            "/webhook/vapi",
            json=_payload(),
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert first.json() == second.json()
    db = session_factory()
    try:
        assert db.scalar(select(func.count()).select_from(VoiceCallAuthorization)) == 1
    finally:
        db.close()


def test_customer_number_authorizes_an_inbound_call(session_factory, fixed_now) -> None:
    _seed_linked_user(session_factory)
    app = create_app(_settings(), session_factory=session_factory, clock=lambda: fixed_now)

    with TestClient(app) as client:
        response = client.post(
            "/webhook/vapi",
            json=_customer_payload(),
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert response.status_code == 200
    assert response.json()["assistant"]["name"] == "Memorae"
    db = session_factory()
    try:
        assert db.scalar(select(func.count()).select_from(VoiceCallAuthorization)) == 1
    finally:
        db.close()


def test_unknown_caller_is_rejected_without_authorization(session_factory, fixed_now) -> None:
    app = create_app(_settings(), session_factory=session_factory, clock=lambda: fixed_now)
    with TestClient(app) as client:
        response = client.post(
            "/webhook/vapi",
            json=_payload(phone="+923001234568"),
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "error": "We could not verify this call. Please link this number in Telegram."
    }
    db = session_factory()
    try:
        assert db.scalar(select(func.count()).select_from(VoiceCallAuthorization)) == 0
    finally:
        db.close()


def test_invalid_credentials_are_rejected_before_payload_processing(
    session_factory, fixed_now
) -> None:
    app = create_app(_settings(), session_factory=session_factory, clock=lambda: fixed_now)
    with TestClient(app) as client:
        response = client.post(
            "/webhook/vapi",
            json={"not": "a vapi message"},
            headers={"Authorization": "Bearer wrong"},
        )

    assert response.status_code == 401


def test_missing_caller_id_fails_closed(session_factory, fixed_now) -> None:
    _seed_linked_user(session_factory)
    app = create_app(_settings(), session_factory=session_factory, clock=lambda: fixed_now)
    with TestClient(app) as client:
        response = client.post(
            "/webhook/vapi",
            json=_payload(phone=None),
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert response.status_code == 200
    assert "could not verify" in response.json()["error"]


def test_malformed_authenticated_message_returns_bad_request(session_factory, fixed_now) -> None:
    app = create_app(_settings(), session_factory=session_factory, clock=lambda: fixed_now)
    with TestClient(app) as client:
        response = client.post(
            "/webhook/vapi",
            json={"message": {"type": "assistant-request"}},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    # A syntactically valid assistant-request without a call is a safe reject,
    # not an authorization or a provider error.
    assert response.status_code == 200
    assert "could not verify" in response.json()["error"]


def test_disabled_vapi_endpoint_does_not_authorize(session_factory, fixed_now) -> None:
    _seed_linked_user(session_factory)
    app = create_app(
        _settings(enabled=False), session_factory=session_factory, clock=lambda: fixed_now
    )
    with TestClient(app) as client:
        response = client.post(
            "/webhook/vapi",
            json=_payload(),
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
    assert response.status_code == 404


def test_enabled_vapi_requires_a_strong_webhook_credential() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://user:password@db.example/app",
        telegram_bot_token="123456:production-token",
        telegram_webhook_secret="s" * 32,
        llm_api_key="production-llm-key",
        stt_api_key="production-stt-key",
        vapi_enabled=True,
        vapi_webhook_bearer_token="short",
    )
    try:
        validate_runtime_settings(settings)
    except RuntimeError as exc:
        assert "VAPI_WEBHOOK_BEARER_TOKEN" in str(exc)
    else:
        raise AssertionError("enabled Vapi accepted a weak webhook credential")


def test_resolver_revalidates_identity_and_expiry(session_factory, fixed_now) -> None:
    user, identity = _seed_linked_user(session_factory)
    db = session_factory()
    try:
        authorization = VoiceCallAuthorization(
            provider_call_id="call-resolve",
            user_id=user.id,
            phone_identity_id=identity.id,
            status="ACTIVE",
            created_at=fixed_now,
            expires_at=fixed_now + timedelta(minutes=5),
        )
        db.add(authorization)
        db.commit()
        context = resolve_vapi_call_context(
            db,
            provider_call_id="call-resolve",
            now=fixed_now,
        )
        assert context is not None
        assert context.canonical_user_id == user.id
        persisted_identity = db.get(ChannelIdentity, identity.id)
        assert persisted_identity is not None
        persisted_identity.status = "UNLINKED"
        db.commit()
        assert (
            resolve_vapi_call_context(
                db,
                provider_call_id="call-resolve",
                now=fixed_now,
            )
            is None
        )
    finally:
        db.close()
