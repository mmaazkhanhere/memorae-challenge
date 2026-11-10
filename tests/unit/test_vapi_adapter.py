from __future__ import annotations

from datetime import timedelta

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from reminder_agent.application.main import create_app
from reminder_agent.application.voice_call_authorization import resolve_vapi_call_context
from reminder_agent.features.intent import (
    IntentRoute,
    IntentRouteSuggestion,
    Interpretation,
    RetrieveMemoryQuery,
    RuleBasedIntentInterpreter,
)
from reminder_agent.features.memory.messages import MEMORY_SAVE_MESSAGES
from reminder_agent.infrastructure.config.settings import Settings, validate_runtime_settings
from reminder_agent.infrastructure.database.models import (
    ChannelIdentity,
    ConversationSession,
    IdempotencyRecord,
    MemoryFact,
    Notification,
    Reminder,
    ReminderOccurrence,
    User,
    VoiceCallAuthorization,
)

PHONE = "+923001234567"
TOKEN = "vapi-test-token-" + "x" * 32


class _ExternalEmbeddingProvider:
    model = "fake-external-v1"
    dimensions = 1536
    is_external = True

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [1.0, *([0.0] * 1535)]


class _MemoryLookupFallbackHarness:
    """Stand in for the configured model branches in this Vapi regression."""

    def __init__(self) -> None:
        self._deterministic = RuleBasedIntentInterpreter()

    def classify(self, *_args, **_kwargs) -> IntentRouteSuggestion:
        return IntentRouteSuggestion(operation=IntentRoute.RETRIEVE_MEMORY)

    def interpret(self, text: str, *, selected_route: IntentRoute, **kwargs) -> Interpretation:
        if selected_route is IntentRoute.RETRIEVE_MEMORY:
            return Interpretation(
                memory_query=RetrieveMemoryQuery(query="birthday"),
                fallback_allowed=False,
            )
        return self._deterministic.interpret(text, selected_route=selected_route, **kwargs)


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


def _tool_payload(
    *,
    call_id: str = "call-test-1",
    tool_call_id: str = "tool-test-1",
    utterance: str = "Remind me in five minutes to call the bank",
) -> dict:
    return {
        "message": {
            "type": "tool-calls",
            "call": {"id": call_id},
            "toolCallList": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": "submit_reminder_utterance",
                        "arguments": {"utterance": utterance},
                    },
                }
            ],
        }
    }


def _seed_linked_user_with_telegram(session_factory, *, phone: str = PHONE):
    user, phone_identity = _seed_linked_user(session_factory, phone=phone)
    db = session_factory()
    try:
        telegram_identity = ChannelIdentity(
            user_id=user.id,
            channel_type="TELEGRAM",
            external_id="1001",
            telegram_chat_id=2001,
            status="LINKED",
        )
        db.add(telegram_identity)
        db.commit()
        return user, phone_identity, telegram_identity
    finally:
        db.close()


def _memory_tool_payload(
    *,
    name: str,
    utterance: str,
    call_id: str = "call-test-1",
    tool_call_id: str = "memory-tool-1",
) -> dict:
    payload = _tool_payload(
        call_id=call_id,
        tool_call_id=tool_call_id,
        utterance=utterance,
    )
    payload["message"]["toolCallList"][0]["function"]["name"] = name
    return payload


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
    assert body["assistant"]["model"]["tools"][0]["function"]["name"] == (
        "submit_reminder_utterance"
    )
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


def test_tool_call_creates_telegram_reminder_for_authorized_call(
    session_factory, fixed_now
) -> None:
    _seed_linked_user_with_telegram(session_factory)
    app = create_app(_settings(), session_factory=session_factory, clock=lambda: fixed_now)

    with TestClient(app) as client:
        authorized = client.post(
            "/webhook/vapi",
            json=_payload(),
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert authorized.status_code == 200
        tool_response = client.post(
            "/webhook/vapi",
            json=_tool_payload(),
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert tool_response.status_code == 200
    body = tool_response.json()
    assert body["results"][0]["toolCallId"] == "tool-test-1"
    assert "call the bank" in body["results"][0]["result"]
    assert "Telegram" in body["results"][0]["result"]
    db = session_factory()
    try:
        reminder = db.scalar(select(Reminder))
        notification = db.scalar(select(Notification))
        assert reminder is not None and reminder.task == "call the bank"
        assert notification is not None
        telegram = db.scalar(
            select(ChannelIdentity).where(ChannelIdentity.channel_type == "TELEGRAM")
        )
        assert telegram is not None and notification.destination_identity_id == telegram.id
    finally:
        db.close()


def test_replayed_tool_call_returns_committed_confirmation_without_duplicate(
    session_factory, fixed_now
) -> None:
    _seed_linked_user_with_telegram(session_factory)
    app = create_app(_settings(), session_factory=session_factory, clock=lambda: fixed_now)
    with TestClient(app) as client:
        headers = {"Authorization": f"Bearer {TOKEN}"}
        client.post("/webhook/vapi", json=_payload(), headers=headers)
        first = client.post("/webhook/vapi", json=_tool_payload(), headers=headers)
        second = client.post("/webhook/vapi", json=_tool_payload(), headers=headers)

    assert first.json() == second.json()
    db = session_factory()
    try:
        assert db.scalar(select(func.count()).select_from(Reminder)) == 1
        assert db.scalar(select(func.count()).select_from(Notification)) == 1
    finally:
        db.close()


def test_reused_tool_call_id_with_different_request_is_rejected(
    session_factory, fixed_now
) -> None:
    _seed_linked_user_with_telegram(session_factory)
    app = create_app(_settings(), session_factory=session_factory, clock=lambda: fixed_now)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    with TestClient(app) as client:
        client.post("/webhook/vapi", json=_payload(), headers=headers)
        first = client.post("/webhook/vapi", json=_tool_payload(), headers=headers)
        second = client.post(
            "/webhook/vapi",
            json=_tool_payload(utterance="Remind me in ten minutes to walk the dog"),
            headers=headers,
        )

    assert "already used" not in first.json()["results"][0].get("error", "")
    assert "already used" in second.json()["results"][0]["error"]
    db = session_factory()
    try:
        assert db.scalar(select(func.count()).select_from(Reminder)) == 1
    finally:
        db.close()


def test_distinct_tool_calls_in_one_call_create_distinct_reminders(
    session_factory, fixed_now
) -> None:
    _seed_linked_user_with_telegram(session_factory)
    app = create_app(_settings(), session_factory=session_factory, clock=lambda: fixed_now)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    with TestClient(app) as client:
        client.post("/webhook/vapi", json=_payload(), headers=headers)
        first = client.post(
            "/webhook/vapi",
            json=_tool_payload(tool_call_id="tool-distinct-1"),
            headers=headers,
        )
        second = client.post(
            "/webhook/vapi",
            json=_tool_payload(
                tool_call_id="tool-distinct-2",
                utterance="Remind me in ten minutes to walk the dog",
            ),
            headers=headers,
        )

    assert "call the bank" in first.json()["results"][0]["result"]
    assert "walk the dog" in second.json()["results"][0]["result"]
    db = session_factory()
    try:
        assert db.scalar(select(func.count()).select_from(Reminder)) == 2
    finally:
        db.close()


def test_tool_call_rejects_action_arguments_instead_of_trusting_llm_fields(
    session_factory, fixed_now
) -> None:
    _seed_linked_user_with_telegram(session_factory)
    app = create_app(_settings(), session_factory=session_factory, clock=lambda: fixed_now)
    payload = _tool_payload()
    payload["message"]["toolCallList"][0]["function"]["arguments"] = {
        "utterance": "Remind me in five minutes to call the bank",
        "user_id": "another-user",
    }
    with TestClient(app) as client:
        client.post(
            "/webhook/vapi",
            json=_payload(),
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        response = client.post(
            "/webhook/vapi",
            json=payload,
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    assert response.status_code == 200
    assert "understand" in response.json()["results"][0]["error"]
    db = session_factory()
    try:
        assert db.scalar(select(func.count()).select_from(Reminder)) == 0
    finally:
        db.close()


def test_vapi_memory_save_creates_a_fact_without_a_reminder(
    session_factory, fixed_now
) -> None:
    _seed_linked_user_with_telegram(session_factory)
    app = create_app(_settings(), session_factory=session_factory, clock=lambda: fixed_now)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    payload = _memory_tool_payload(
        name="submit_memory_save_utterance",
        utterance="Remember the Wi-Fi password is Andorra2026",
    )
    with TestClient(app) as client:
        client.post("/webhook/vapi", json=_payload(), headers=headers)
        response = client.post("/webhook/vapi", json=payload, headers=headers)

    assert response.status_code == 200
    assert response.json()["results"][0]["result"] in MEMORY_SAVE_MESSAGES["confirmation"]
    db = session_factory()
    try:
        fact = db.scalar(select(MemoryFact))
        assert fact is not None
        assert fact.content == "the Wi-Fi password is Andorra2026"
        assert db.scalar(select(func.count()).select_from(Reminder)) == 0
        record = db.scalar(
            select(IdempotencyRecord).where(
                IdempotencyRecord.source == "VAPI",
                IdempotencyRecord.idempotency_key == "call-test-1:memory-tool-1",
            )
        )
        assert record is not None
        assert "Andorra2026" not in str(record.response)
    finally:
        db.close()


def test_replayed_vapi_memory_save_does_not_duplicate_the_fact(
    session_factory, fixed_now
) -> None:
    _seed_linked_user_with_telegram(session_factory)
    app = create_app(_settings(), session_factory=session_factory, clock=lambda: fixed_now)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    payload = _memory_tool_payload(
        name="submit_memory_save_utterance",
        utterance="Remember my favorite color is blue",
    )
    with TestClient(app) as client:
        client.post("/webhook/vapi", json=_payload(), headers=headers)
        first = client.post("/webhook/vapi", json=payload, headers=headers)
        second = client.post("/webhook/vapi", json=payload, headers=headers)

    assert first.json() == second.json()
    db = session_factory()
    try:
        assert db.scalar(select(func.count()).select_from(MemoryFact)) == 1
    finally:
        db.close()


def test_vapi_memory_lookup_returns_owner_scoped_fact(
    session_factory, fixed_now
) -> None:
    _seed_linked_user_with_telegram(session_factory)
    app = create_app(_settings(), session_factory=session_factory, clock=lambda: fixed_now)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    with TestClient(app) as client:
        client.post("/webhook/vapi", json=_payload(), headers=headers)
        saved = client.post(
            "/webhook/vapi",
            json=_memory_tool_payload(
                name="submit_memory_save_utterance",
                utterance="Remember my favorite color is blue",
                tool_call_id="memory-save-1",
            ),
            headers=headers,
        )
        looked_up = client.post(
            "/webhook/vapi",
            json=_memory_tool_payload(
                name="submit_memory_lookup_utterance",
                utterance="What's my favorite color?",
                tool_call_id="memory-lookup-1",
            ),
            headers=headers,
        )

    assert saved.json()["results"][0]["result"] in MEMORY_SAVE_MESSAGES["confirmation"]
    assert "blue" in looked_up.json()["results"][0]["result"]


def test_vapi_memory_lookup_returns_birthday_fact(
    session_factory, fixed_now
) -> None:
    _seed_linked_user_with_telegram(session_factory)
    harness = _MemoryLookupFallbackHarness()
    app = create_app(
        _settings(),
        session_factory=session_factory,
        reminder_interpreter=harness,
        intent_router=harness,
        clock=lambda: fixed_now,
    )
    headers = {"Authorization": f"Bearer {TOKEN}"}
    with TestClient(app) as client:
        client.post("/webhook/vapi", json=_payload(), headers=headers)
        client.post(
            "/webhook/vapi",
            json=_memory_tool_payload(
                name="submit_memory_save_utterance",
                utterance="Remember that my birthday is January 10",
                tool_call_id="birthday-save-1",
            ),
            headers=headers,
        )
        response = client.post(
            "/webhook/vapi",
            json=_memory_tool_payload(
                name="submit_memory_lookup_utterance",
                utterance="When is my birthday?",
                tool_call_id="birthday-lookup-1",
            ),
            headers=headers,
        )

    assert response.status_code == 200
    assert "January 10" in response.json()["results"][0]["result"]


def test_vapi_memory_lookup_speaks_sensitive_facts(
    session_factory, fixed_now
) -> None:
    _seed_linked_user_with_telegram(session_factory)
    provider = _ExternalEmbeddingProvider()
    app = create_app(
        _settings(),
        session_factory=session_factory,
        embedding_provider=provider,
        clock=lambda: fixed_now,
    )
    headers = {"Authorization": f"Bearer {TOKEN}"}
    with TestClient(app) as client:
        client.post("/webhook/vapi", json=_payload(), headers=headers)
        client.post(
            "/webhook/vapi",
            json=_memory_tool_payload(
                name="submit_memory_save_utterance",
                utterance="Remember the Wi-Fi password is Andorra2026",
                tool_call_id="memory-save-sensitive",
            ),
            headers=headers,
        )
        response = client.post(
            "/webhook/vapi",
            json=_memory_tool_payload(
                name="submit_memory_lookup_utterance",
                utterance="What's my Wi-Fi password?",
                tool_call_id="memory-lookup-sensitive",
            ),
            headers=headers,
        )

    result = response.json()["results"][0]["result"]
    assert "Andorra2026" in result
    db = session_factory()
    try:
        fact = db.scalar(select(MemoryFact))
        assert fact is not None and fact.embedding_status == "INDEXED"
        assert provider.calls == ["the Wi-Fi password is Andorra2026"]
    finally:
        db.close()


def test_vapi_memory_clarification_follow_up_saves_original_operation(
    session_factory, fixed_now
) -> None:
    _seed_linked_user_with_telegram(session_factory)
    app = create_app(_settings(), session_factory=session_factory, clock=lambda: fixed_now)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    with TestClient(app) as client:
        client.post("/webhook/vapi", json=_payload(), headers=headers)
        first = client.post(
            "/webhook/vapi",
            json=_memory_tool_payload(
                name="submit_memory_save_utterance",
                utterance="Remember the Wi-Fi password",
                tool_call_id="memory-clarify-1",
            ),
            headers=headers,
        )
        second = client.post(
            "/webhook/vapi",
            json=_memory_tool_payload(
                name="submit_memory_save_utterance",
                utterance="Andorra2026",
                tool_call_id="memory-clarify-2",
            ),
            headers=headers,
        )

    assert "What information" in first.json()["results"][0]["result"]
    assert second.json()["results"][0]["result"] in MEMORY_SAVE_MESSAGES["confirmation"]
    db = session_factory()
    try:
        assert db.scalar(select(func.count()).select_from(MemoryFact)) == 1
        fact = db.scalar(select(MemoryFact))
        assert fact is not None and fact.content == "Andorra2026"
    finally:
        db.close()


def test_voice_clarification_follow_up_completes_the_same_call_draft(
    session_factory, fixed_now
) -> None:
    _seed_linked_user_with_telegram(session_factory)
    app = create_app(_settings(), session_factory=session_factory, clock=lambda: fixed_now)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    with TestClient(app) as client:
        client.post("/webhook/vapi", json=_payload(), headers=headers)
        first = client.post(
            "/webhook/vapi",
            json=_tool_payload(
                tool_call_id="tool-clarify-1",
                utterance="Remind me to call the bank",
            ),
            headers=headers,
        )
        second = client.post(
            "/webhook/vapi",
            json=_tool_payload(
                tool_call_id="tool-clarify-2",
                utterance="tomorrow at 9 AM",
            ),
            headers=headers,
        )

    assert "What should" in first.json()["results"][0]["result"]
    assert "call the bank" in second.json()["results"][0]["result"]
    db = session_factory()
    try:
        assert db.scalar(select(func.count()).select_from(Reminder)) == 1
        assert db.scalar(select(func.count()).select_from(ConversationSession)) == 1
        session = db.scalar(select(ConversationSession))
        assert session is not None and session.pending_clarification is None
    finally:
        db.close()


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


def _seed_voice_reminder(session_factory, *, user_id, telegram_identity_id, fixed_now, task, status="PENDING"):
    reminder = Reminder(
        user_id=user_id,
        task=task,
        schedule_type="ONE_TIME",
        start_local="2026-10-31T13:00:00",
        timezone="UTC",
        status="ACTIVE",
    )
    db = session_factory()
    try:
        db.add(reminder)
        db.flush()
        occurrence = ReminderOccurrence(
            reminder_id=reminder.id,
            scheduled_at=fixed_now + timedelta(days=1),
            next_due_at=fixed_now + timedelta(days=1),
            status=status,
        )
        db.add(occurrence)
        db.flush()
        db.add(
            Notification(
                occurrence_id=occurrence.id,
                destination_identity_id=telegram_identity_id,
                sequence=1,
                due_at=occurrence.scheduled_at,
                status="SENT" if status == "NOTIFIED" else "PENDING",
                sent_at=occurrence.scheduled_at if status == "NOTIFIED" else None,
                provider_message_id="voice-provider-message" if status == "NOTIFIED" else None,
            )
        )
        db.commit()
        return reminder.id, occurrence.id
    finally:
        db.close()


def test_vapi_lists_and_reschedules_existing_reminders(session_factory, fixed_now) -> None:
    user, _phone, telegram = _seed_linked_user_with_telegram(session_factory)
    reminder_id, _ = _seed_voice_reminder(
        session_factory,
        user_id=user.id,
        telegram_identity_id=telegram.id,
        fixed_now=fixed_now,
        task="call the bank",
    )
    app = create_app(_settings(), session_factory=session_factory, clock=lambda: fixed_now)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    with TestClient(app) as client:
        client.post("/webhook/vapi", json=_payload(), headers=headers)
        listed = client.post(
            "/webhook/vapi",
            json=_tool_payload(tool_call_id="voice-list-1", utterance="What do I have tomorrow?"),
            headers=headers,
        )
        moved = client.post(
            "/webhook/vapi",
            json=_tool_payload(
                tool_call_id="voice-move-1",
                utterance="Move the bank one to tomorrow at 14:00",
            ),
            headers=headers,
        )
    assert "call the bank" in listed.json()["results"][0]["result"]
    assert "call the bank" in moved.json()["results"][0]["result"]
    db = session_factory()
    try:
        assert db.get(Reminder, reminder_id).status == "ACTIVE"
        assert db.scalar(select(ReminderOccurrence).where(ReminderOccurrence.reminder_id == reminder_id)).next_due_at.hour == 14
    finally:
        db.close()


def test_vapi_completes_and_snoozes_the_linked_telegram_delivery(session_factory, fixed_now) -> None:
    user, _phone, telegram = _seed_linked_user_with_telegram(session_factory)
    _seed_voice_reminder(
        session_factory,
        user_id=user.id,
        telegram_identity_id=telegram.id,
        fixed_now=fixed_now,
        task="take medicine",
        status="NOTIFIED",
    )
    _seed_voice_reminder(
        session_factory,
        user_id=user.id,
        telegram_identity_id=telegram.id,
        fixed_now=fixed_now,
        task="send the report",
        status="NOTIFIED",
    )
    app = create_app(_settings(), session_factory=session_factory, clock=lambda: fixed_now)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    with TestClient(app) as client:
        client.post("/webhook/vapi", json=_payload(), headers=headers)
        snoozed = client.post(
            "/webhook/vapi",
            json=_tool_payload(
                tool_call_id="voice-snooze-1", utterance="Snooze take medicine for 10 minutes"
            ),
            headers=headers,
        )
        completed = client.post(
            "/webhook/vapi",
            json=_tool_payload(tool_call_id="voice-complete-1", utterance="Done"),
            headers=headers,
        )
    assert "10" in snoozed.json()["results"][0]["result"]
    assert "report" in completed.json()["results"][0]["result"]
    db = session_factory()
    try:
        occurrence = db.scalar(select(ReminderOccurrence).where(ReminderOccurrence.status == "PENDING"))
        assert occurrence is not None
        assert db.scalar(select(func.count()).select_from(Notification).where(Notification.status == "PENDING")) == 1
    finally:
        db.close()


def test_vapi_cancels_an_existing_reminder(session_factory, fixed_now) -> None:
    user, _phone, telegram = _seed_linked_user_with_telegram(session_factory)
    reminder_id, _ = _seed_voice_reminder(
        session_factory,
        user_id=user.id,
        telegram_identity_id=telegram.id,
        fixed_now=fixed_now,
        task="attend the standup",
    )
    app = create_app(_settings(), session_factory=session_factory, clock=lambda: fixed_now)
    headers = {"Authorization": f"Bearer {TOKEN}"}
    with TestClient(app) as client:
        client.post("/webhook/vapi", json=_payload(), headers=headers)
        response = client.post(
            "/webhook/vapi",
            json=_tool_payload(tool_call_id="voice-cancel-1", utterance="Cancel the standup"),
            headers=headers,
        )
    assert "standup" in response.json()["results"][0]["result"]
    db = session_factory()
    try:
        assert db.get(Reminder, reminder_id).status == "CANCELED"
    finally:
        db.close()
