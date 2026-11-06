from datetime import timedelta

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from reminder_agent.conversation_router import ConversationRoute, route_message
from reminder_agent.intent import (
    IntentRoute,
    IntentRouteSuggestion,
    Interpretation,
    ReminderIntentInterpreter,
)
from reminder_agent.main import create_app
from reminder_agent.memories import MemoryIntentEvidence, SaveMemoryFact

from reminder_agent.db.models import (
    ConversationSession,
    IdempotencyRecord,
    MemoryFact,
    Notification,
    Reminder,
    ReminderOccurrence,
    User,
)


HEADERS = {"X-Telegram-Bot-Api-Secret-Token": "test-secret"}


def post_update(client, update: dict):
    return client.post("/webhook/telegram", json=update, headers=HEADERS)


def onboard(api_client, update_factory, *, user_id: int, chat_id: int, start: int) -> None:
    assert post_update(
        api_client,
        update_factory(start, user_id=user_id, chat_id=chat_id, text="/start"),
    ).status_code == 200
    assert post_update(
        api_client,
        update_factory(
            start + 1,
            user_id=user_id,
            chat_id=chat_id,
            text="Asia/Karachi",
        ),
    ).status_code == 200


def test_explicit_memory_save_creates_one_owned_fact_and_no_reminder_rows(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory, user_id=1001, chat_id=2001, start=100)

    response = post_update(
        api_client,
        update_factory(102, text="Remember my gate code is 4512"),
    )

    assert response.status_code == 200
    assert telegram_client.messages[-1][1] == "I saved that information."
    fact = db.scalar(select(MemoryFact))
    assert fact is not None
    assert fact.content == "my gate code is 4512"
    assert db.scalar(select(func.count()).select_from(Reminder)) == 0
    assert db.scalar(select(func.count()).select_from(ReminderOccurrence)) == 0
    assert db.scalar(select(func.count()).select_from(Notification)) == 0
    record = db.scalar(
        select(IdempotencyRecord).where(IdempotencyRecord.idempotency_key == "102")
    )
    assert record is not None
    assert record.operation == "SAVE_MEMORY"
    assert record.response is not None
    assert "request_text" not in record.response


def test_replayed_memory_update_does_not_duplicate_fact_or_confirmation(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory, user_id=1001, chat_id=2001, start=200)
    update = update_factory(202, text="Remember my Wi-Fi password is Andorra2026")

    assert post_update(api_client, update).status_code == 200
    sent_count = len(telegram_client.messages)
    assert post_update(api_client, update).status_code == 200

    assert len(telegram_client.messages) == sent_count
    assert db.scalar(select(func.count()).select_from(MemoryFact)) == 1


def test_memory_fact_survives_database_session_and_application_restart(
    api_client, telegram_client, session_factory, update_factory, fixed_now
) -> None:
    onboard(api_client, update_factory, user_id=1001, chat_id=2001, start=250)
    update = update_factory(252, text="Remember my emergency contact is Sam")
    assert post_update(api_client, update).status_code == 200
    sent_count = len(telegram_client.messages)

    # Close the original database session and verify the committed fact from a
    # newly opened session before constructing a fresh application instance.
    with session_factory() as reopened_db:
        fact = reopened_db.scalar(select(MemoryFact))
        assert fact is not None
        assert fact.content == "my emergency contact is Sam"

    restarted_app = create_app(
        api_client.app.state.settings,
        telegram_client=telegram_client,
        session_factory=session_factory,
        clock=lambda: fixed_now,
    )
    with TestClient(restarted_app) as restarted_client:
        assert post_update(restarted_client, update).status_code == 200

    with session_factory() as final_db:
        assert final_db.scalar(select(func.count()).select_from(MemoryFact)) == 1
    assert len(telegram_client.messages) == sent_count


def test_memory_facts_are_owned_separately_between_users(
    api_client, db, update_factory
) -> None:
    onboard(api_client, update_factory, user_id=1001, chat_id=2001, start=300)
    onboard(api_client, update_factory, user_id=1002, chat_id=2002, start=400)
    assert post_update(
        api_client,
        update_factory(302, user_id=1001, chat_id=2001, text="Remember first user's fact"),
    ).status_code == 200
    assert post_update(
        api_client,
        update_factory(402, user_id=1002, chat_id=2002, text="Remember second user's fact"),
    ).status_code == 200

    facts = db.scalars(select(MemoryFact).order_by(MemoryFact.content)).all()
    assert [fact.content for fact in facts] == ["first user's fact", "second user's fact"]
    assert facts[0].user_id != facts[1].user_id


def test_incomplete_memory_save_asks_for_fact_without_mutation(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory, user_id=1001, chat_id=2001, start=500)

    assert post_update(api_client, update_factory(502, text="Remember")).status_code == 200

    assert telegram_client.messages[-1][1] == "What information should I remember?"
    assert db.scalar(select(func.count()).select_from(MemoryFact)) == 0
    session = db.scalar(select(ConversationSession))
    assert session is not None
    assert session.pending_clarification == {"operation": "SAVE_MEMORY", "content": None}

    assert post_update(
        api_client,
        update_factory(503, text="my passport is in the safe"),
    ).status_code == 200
    assert db.scalar(select(func.count()).select_from(MemoryFact)) == 1
    db.expire_all()
    assert db.scalar(select(ConversationSession)).pending_clarification is None


def test_pending_memory_can_be_canceled_without_mutation(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory, user_id=1001, chat_id=2001, start=600)
    assert post_update(api_client, update_factory(602, text="Remember")).status_code == 200
    assert post_update(api_client, update_factory(603, text="Never mind")).status_code == 200

    assert telegram_client.messages[-1][1] == "Okay, I won't save anything."
    assert db.scalar(select(func.count()).select_from(MemoryFact)) == 0
    db.expire_all()
    assert db.scalar(select(ConversationSession)).pending_clarification is None


def test_memory_save_database_failure_is_truthful_and_leaves_no_fact(
    api_client, telegram_client, db, update_factory, monkeypatch
) -> None:
    onboard(api_client, update_factory, user_id=1001, chat_id=2001, start=700)

    from sqlalchemy.exc import OperationalError
    import reminder_agent.reminders as reminders

    def fail(*args, **kwargs):
        raise OperationalError("insert", {}, RuntimeError("database down"))

    monkeypatch.setattr(reminders, "save_memory_fact", fail)
    assert post_update(
        api_client,
        update_factory(702, text="Remember a fact that cannot be saved"),
    ).status_code == 200

    assert telegram_client.messages[-1][1] == "I couldn't save that information. Please try again."
    assert db.scalar(select(func.count()).select_from(MemoryFact)) == 0


def test_unexpected_memory_save_failure_rolls_back_and_never_confirms(
    api_client, telegram_client, db, update_factory, monkeypatch
) -> None:
    onboard(api_client, update_factory, user_id=1001, chat_id=2001, start=750)

    import reminder_agent.reminders as reminders
    from reminder_agent.memories import save_memory_fact as real_save_memory_fact

    def fail_after_flush(db_session, *, user_id, content, now):
        real_save_memory_fact(db_session, user_id=user_id, content=content, now=now)
        raise RuntimeError("unexpected failure containing a secret")

    monkeypatch.setattr(reminders, "save_memory_fact", fail_after_flush)
    assert post_update(
        api_client,
        update_factory(752, text="Remember my vault code is Secret-4512"),
    ).status_code == 200

    assert telegram_client.messages[-1][1] == "I couldn't save that information. Please try again."
    assert not any(message[1] == "I saved that information." for message in telegram_client.messages)
    assert db.scalar(select(func.count()).select_from(MemoryFact)) == 0
    assert db.scalar(
        select(func.count()).select_from(IdempotencyRecord).where(
            IdempotencyRecord.idempotency_key == "752"
        )
    ) == 0


def test_explicit_memory_save_uses_rules_without_calling_classifier(
    api_client, db, update_factory
) -> None:
    onboard(api_client, update_factory, user_id=1001, chat_id=2001, start=800)

    class FailingRouter:
        def classify(self, *args, **kwargs):
            raise AssertionError("explicit memory save must not invoke classification")

    api_client.app.state.intent_router = FailingRouter()
    assert post_update(
        api_client,
        update_factory(802, text="Remember my emergency contact is Sam"),
    ).status_code == 200
    assert db.scalar(select(func.count()).select_from(MemoryFact)) == 1


def test_unambiguous_reminder_and_memory_commands_do_not_cross_write_or_call_llm(
    api_client, db, update_factory
) -> None:
    onboard(api_client, update_factory, user_id=1001, chat_id=2001, start=1600)

    class FailingRouter:
        def classify(self, *args, **kwargs):  # pragma: no cover - assertion below
            raise AssertionError("unambiguous commands must not invoke classification")

    class FailingExtractor:
        def interpret(self, *args, **kwargs):  # pragma: no cover - assertion below
            raise AssertionError("deterministic commands must not invoke extraction fallback")

    api_client.app.state.intent_router = FailingRouter()
    api_client.app.state.reminder_interpreter = ReminderIntentInterpreter(
        model_interpreter=FailingExtractor()
    )
    timed_reminder = update_factory(1602, text="Remember to call Ali tomorrow at 9 AM")
    assert post_update(api_client, timed_reminder).status_code == 200
    reminder = db.scalar(select(Reminder))
    assert reminder is not None and reminder.task == "call Ali"
    assert db.scalar(select(func.count()).select_from(MemoryFact)) == 0

    assert post_update(
        api_client, update_factory(1603, text="Remember to call Ali")
    ).status_code == 200
    assert db.scalar(select(func.count()).select_from(Reminder)) == 1
    session = db.scalar(select(ConversationSession))
    assert session is not None
    assert session.pending_clarification["task"] == "call Ali"

    assert post_update(
        api_client, update_factory(1604, text="Remember my gate code is 4512")
    ).status_code == 200
    assert db.scalar(select(func.count()).select_from(MemoryFact)) == 1
    assert db.scalar(select(func.count()).select_from(Reminder)) == 1

    assert post_update(
        api_client, update_factory(1605, text="Remember that my birthday is January 10")
    ).status_code == 200
    assert db.scalar(select(func.count()).select_from(MemoryFact)) == 2
    assert db.scalar(select(func.count()).select_from(Reminder)) == 1

    saved_reminder = update_factory(1606, text="Save a reminder to call Ali tomorrow at 9")
    assert post_update(api_client, saved_reminder).status_code == 200
    assert db.scalar(select(func.count()).select_from(MemoryFact)) == 2
    # The command is deterministically a reminder request. "9" lacks an
    # AM/PM marker, so the existing schedule validator keeps it as a pending
    # reminder rather than guessing a delivery time.
    assert db.scalar(select(func.count()).select_from(Reminder)) == 1
    db.expire_all()
    session = db.scalar(select(ConversationSession))
    assert session is not None and session.pending_clarification["task"] == "call Ali"
    assert post_update(api_client, saved_reminder).status_code == 200
    assert db.scalar(select(func.count()).select_from(MemoryFact)) == 2
    assert db.scalar(select(func.count()).select_from(Reminder)) == 1


def test_colliding_or_uncertain_creation_text_uses_classifier_without_writes(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory, user_id=1001, chat_id=2001, start=1700)
    calls: list[str] = []

    class UncertainRouter:
        def classify(self, text, **kwargs):
            calls.append(text)
            return IntentRouteSuggestion(operation=IntentRoute.UNCERTAIN)

    class FailingExtractor:
        def interpret(self, *args, **kwargs):  # pragma: no cover - assertion below
            raise AssertionError("uncertain intent must not reach extraction")

    api_client.app.state.intent_router = UncertainRouter()
    api_client.app.state.reminder_interpreter = FailingExtractor()
    for update_id, text in (
        (1702, "Remember to keep my gate code in mind"),
        (1703, "Don't forget Ali's birthday"),
    ):
        assert route_message(text, pending=None).route is ConversationRoute.MODEL_FALLBACK
        assert post_update(api_client, update_factory(update_id, text=text)).status_code == 200

    assert calls == [
        "Remember to keep my gate code in mind",
        "Don't forget Ali's birthday",
    ]
    assert "not sure" in telegram_client.messages[-1][1].lower()
    assert db.scalar(select(func.count()).select_from(MemoryFact)) == 0
    assert db.scalar(select(func.count()).select_from(Reminder)) == 0
    assert db.scalar(select(func.count()).select_from(ReminderOccurrence)) == 0
    assert db.scalar(select(func.count()).select_from(Notification)) == 0


def test_saved_memory_secret_is_omitted_from_idempotency_and_recent_history(
    api_client, db, update_factory
) -> None:
    onboard(api_client, update_factory, user_id=1001, chat_id=2001, start=850)
    secret = "my Wi-Fi password is Secret-4512"
    assert post_update(
        api_client,
        update_factory(852, text=f"Remember {secret}"),
    ).status_code == 200

    record = db.scalar(select(IdempotencyRecord).where(IdempotencyRecord.idempotency_key == "852"))
    assert record is not None and record.response is not None
    assert "request_text" not in record.response

    from reminder_agent.reminders import _recent_messages

    user = db.scalar(select(User))
    assert user is not None
    assert secret not in "\n".join(_recent_messages(db, user_id=user.id))


def test_explicit_memory_save_supersedes_pending_reminder_without_creating_one(
    api_client, db, update_factory
) -> None:
    onboard(api_client, update_factory, user_id=1001, chat_id=2001, start=900)
    assert post_update(api_client, update_factory(902, text="Remind me")).status_code == 200
    assert post_update(
        api_client,
        update_factory(903, text="Remember my passport is in the safe"),
    ).status_code == 200

    assert db.scalar(select(func.count()).select_from(MemoryFact)) == 1
    assert db.scalar(select(func.count()).select_from(Reminder)) == 0
    db.expire_all()
    session = db.scalar(select(ConversationSession))
    assert session is not None and session.pending_clarification is None


def test_indirect_intent_uses_classifier_and_existing_typed_extractor(
    api_client, db, update_factory
) -> None:
    onboard(api_client, update_factory, user_id=1001, chat_id=2001, start=1000)
    calls: list[IntentRoute] = []

    class MemoryRouter:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(
                operation=IntentRoute.SAVE_MEMORY,
                memory_intent_evidence=MemoryIntentEvidence(
                    kind="EXPLICIT_SAVE_REQUEST",
                    confidence="CLEAR",
                    evidence_text="Keep in mind",
                ),
            )

    class MemoryExtractor:
        def interpret(self, text, *, selected_route, **kwargs):
            calls.append(selected_route)
            return Interpretation(memory_fact=SaveMemoryFact(content="gate code is 4512"))

    api_client.app.state.intent_router = MemoryRouter()
    api_client.app.state.reminder_interpreter = MemoryExtractor()
    assert post_update(
        api_client,
        update_factory(1002, text="Keep in mind that my gate code is 4512"),
    ).status_code == 200

    fact = db.scalar(select(MemoryFact))
    assert fact is not None and fact.content == "gate code is 4512"
    assert calls == [IntentRoute.SAVE_MEMORY]
    assert db.scalar(select(func.count()).select_from(Reminder)) == 0


def test_keep_in_mind_completes_model_fallback_memory_sequence_once(
    api_client, telegram_client, db, update_factory, monkeypatch
) -> None:
    onboard(api_client, update_factory, user_id=1001, chat_id=2001, start=1400)
    text = "Keep in mind that my gate code is 4512"
    events: list[str] = []

    assert route_message(text, pending=None).route is ConversationRoute.MODEL_FALLBACK

    class MemoryRouter:
        def classify(self, latest_text, **kwargs):
            events.append("classifier")
            assert latest_text == text
            suggestion = IntentRouteSuggestion(
                operation=IntentRoute.SAVE_MEMORY,
                memory_intent_evidence=MemoryIntentEvidence(
                    kind="EXPLICIT_SAVE_REQUEST",
                    confidence="CLEAR",
                    evidence_text="Keep in mind",
                ),
            )
            assert suggestion.operation is not IntentRoute.NEW_REQUEST
            return suggestion

    class MemoryExtractor:
        def interpret(self, latest_text, *, selected_route, **kwargs):
            events.append("memory_extraction")
            assert latest_text == text
            assert selected_route is IntentRoute.SAVE_MEMORY
            fact = SaveMemoryFact(content="my gate code is 4512")
            assert fact.model_dump(mode="json") == {
                "operation": "SAVE_MEMORY",
                "content": "my gate code is 4512",
            }
            return Interpretation(memory_fact=fact)

    import reminder_agent.reminders as reminders

    original_save = reminders.save_memory_fact

    def save_with_trace(*args, **kwargs):
        events.append("authorized_transactional_memory_service")
        return original_save(*args, **kwargs)

    monkeypatch.setattr(reminders, "save_memory_fact", save_with_trace)
    api_client.app.state.intent_router = MemoryRouter()
    api_client.app.state.reminder_interpreter = MemoryExtractor()

    update = update_factory(1402, text=text)
    assert post_update(api_client, update).status_code == 200
    assert events == [
        "classifier",
        "memory_extraction",
        "authorized_transactional_memory_service",
    ]
    assert telegram_client.messages[-1][1] == "I saved that information."

    fact = db.scalar(select(MemoryFact))
    assert fact is not None
    assert fact.content == "my gate code is 4512"
    owner = db.scalar(select(User))
    assert owner is not None and fact.user_id == owner.id
    assert db.scalar(select(func.count()).select_from(MemoryFact)) == 1
    assert db.scalar(select(func.count()).select_from(Reminder)) == 0
    assert db.scalar(select(func.count()).select_from(ReminderOccurrence)) == 0
    assert db.scalar(select(func.count()).select_from(Notification)) == 0

    message_count = len(telegram_client.messages)
    event_count = len(events)
    assert post_update(api_client, update).status_code == 200
    assert len(telegram_client.messages) == message_count
    assert len(events) == event_count
    assert db.scalar(select(func.count()).select_from(MemoryFact)) == 1


def test_model_routed_memory_save_for_dog_name_creates_only_one_fact(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory, user_id=1001, chat_id=2001, start=1450)
    text = "I'd like you to remember that my dog's name is Bruno"

    class MemoryRouter:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(
                operation=IntentRoute.SAVE_MEMORY,
                memory_intent_evidence=MemoryIntentEvidence(
                    kind="EXPLICIT_SAVE_REQUEST",
                    confidence="CLEAR",
                    evidence_text="remember",
                ),
            )

    class MemoryExtractor:
        def interpret(self, latest_text, *, selected_route, **kwargs):
            assert latest_text == text
            assert selected_route is IntentRoute.SAVE_MEMORY
            return Interpretation(
                memory_fact=SaveMemoryFact(content="my dog's name is Bruno")
            )

    api_client.app.state.intent_router = MemoryRouter()
    api_client.app.state.reminder_interpreter = ReminderIntentInterpreter(
        model_interpreter=MemoryExtractor()
    )

    assert post_update(api_client, update_factory(1452, text=text)).status_code == 200
    assert telegram_client.messages[-1][1] == "I saved that information."
    assert db.scalar(select(func.count()).select_from(MemoryFact)) == 1
    assert db.scalar(select(func.count()).select_from(Reminder)) == 0
    assert db.scalar(select(func.count()).select_from(ReminderOccurrence)) == 0
    assert db.scalar(select(func.count()).select_from(Notification)) == 0
    fact = db.scalar(select(MemoryFact))
    assert fact is not None and fact.content == "my dog's name is Bruno"


def test_uncertain_memory_classification_clarifies_without_extraction_or_mutation(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory, user_id=1001, chat_id=2001, start=1500)

    class UncertainRouter:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(operation=IntentRoute.UNCERTAIN)

    class FailingExtractor:
        def interpret(self, *args, **kwargs):  # pragma: no cover - assertion below
            raise AssertionError("uncertain classification must not invoke extraction")

    api_client.app.state.intent_router = UncertainRouter()
    api_client.app.state.reminder_interpreter = FailingExtractor()
    assert post_update(
        api_client,
        update_factory(1502, text="Keep in mind that my gate code is 4512"),
    ).status_code == 200

    assert "not sure" in telegram_client.messages[-1][1].lower()
    assert db.scalar(select(func.count()).select_from(MemoryFact)) == 0
    assert db.scalar(select(func.count()).select_from(Reminder)) == 0
    assert db.scalar(select(func.count()).select_from(ReminderOccurrence)) == 0
    assert db.scalar(select(func.count()).select_from(Notification)) == 0


def test_classifier_cannot_turn_an_ordinary_statement_into_memory_permission(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory, user_id=1001, chat_id=2001, start=1100)

    class OvereagerRouter:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(operation=IntentRoute.SAVE_MEMORY)

    class FailingExtractor:
        def interpret(self, *args, **kwargs):  # pragma: no cover - assertion below
            raise AssertionError("ordinary statements must not reach extraction")

    api_client.app.state.intent_router = OvereagerRouter()
    api_client.app.state.reminder_interpreter = FailingExtractor()
    assert post_update(
        api_client,
        update_factory(1102, text="My gate code is 4512"),
    ).status_code == 200

    assert db.scalar(select(func.count()).select_from(MemoryFact)) == 0
    assert "what information" in telegram_client.messages[-1][1].lower()


def test_malformed_llm_memory_content_clarifies_without_mutation(
    api_client, telegram_client, db, update_factory
) -> None:
    onboard(api_client, update_factory, user_id=1001, chat_id=2001, start=1200)

    class MemoryRouter:
        def classify(self, *args, **kwargs):
            return IntentRouteSuggestion(
                operation=IntentRoute.SAVE_MEMORY,
                memory_intent_evidence=MemoryIntentEvidence(
                    kind="EXPLICIT_SAVE_REQUEST",
                    confidence="CLEAR",
                    evidence_text="keep this in mind",
                ),
            )

    class MalformedExtractor:
        def interpret(self, *args, **kwargs):
            return Interpretation(response_text="Tell me what information to remember.")

    api_client.app.state.intent_router = MemoryRouter()
    api_client.app.state.reminder_interpreter = MalformedExtractor()
    assert post_update(
        api_client,
        update_factory(1202, text="Please keep this in mind"),
    ).status_code == 200

    assert db.scalar(select(func.count()).select_from(MemoryFact)) == 0
    assert db.scalar(select(func.count()).select_from(Reminder)) == 0
    assert db.scalar(select(func.count()).select_from(ReminderOccurrence)) == 0
    assert db.scalar(select(func.count()).select_from(Notification)) == 0
    assert telegram_client.messages[-1][1] == "Tell me what information to remember."
    assert "reminder task" not in telegram_client.messages[-1][1].lower()
    session = db.scalar(select(ConversationSession))
    assert session is not None
    assert session.pending_clarification == {"operation": "SAVE_MEMORY", "content": None}


def test_expired_pending_memory_does_not_save_follow_up_as_a_new_fact(
    api_client, db, update_factory, fixed_now
) -> None:
    onboard(api_client, update_factory, user_id=1001, chat_id=2001, start=1300)
    assert post_update(api_client, update_factory(1302, text="Remember")).status_code == 200

    session = db.scalar(select(ConversationSession))
    assert session is not None
    session.updated_at = fixed_now - timedelta(seconds=901)
    db.commit()

    assert post_update(
        api_client,
        update_factory(1303, text="my passport is in the safe"),
    ).status_code == 200
    assert db.scalar(select(func.count()).select_from(MemoryFact)) == 0
    db.expire_all()
    assert db.scalar(select(ConversationSession)).pending_clarification is None
