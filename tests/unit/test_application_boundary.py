from __future__ import annotations

import ast
import inspect
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import func, select

from reminder_agent import application
from reminder_agent.application import (
    ApplicationRequest,
    ApplicationResult,
    AuthenticatedRequestContext,
    CancelReminderCommand,
    ConversationEntryResult,
    DurableResponse,
    ListRemindersCommand,
    RequestSource,
    SaveMemoryCommand,
    VerifiedChannelIdentity,
    cancel_reminder_for_request,
    interpret_for_route,
    list_reminders_for_request,
    process_authenticated_conversation,
    save_memory_for_request,
)
from reminder_agent.application import conversation_coordinator as reminders
from reminder_agent.channels.telegram.provider import TelegramUpdate
from reminder_agent.features.intent import ReminderIntentInterpreter
from reminder_agent.features.intent.deterministic.routing import ConversationRoute
from reminder_agent.features.memory import adapters as memory_adapters
from reminder_agent.features.reminders.cancel.decisions import CancellationScope
from reminder_agent.features.reminders.cancel.messages import REMINDER_CANCEL_MESSAGES
from reminder_agent.infrastructure.db.models import (
    ChannelIdentity,
    IdempotencyRecord,
    MemoryFact,
    Notification,
    Reminder,
    ReminderOccurrence,
    User,
)


def _active_telegram_context(db):
    user = User(name="Ada", timezone="UTC", onboarding_status="ACTIVE")
    db.add(user)
    db.flush()
    identity = ChannelIdentity(
        user_id=user.id,
        channel_type="TELEGRAM",
        external_id="1001",
        telegram_chat_id=2001,
        status="LINKED",
    )
    db.add(identity)
    db.flush()
    return user, identity


def _request_for(
    user: User, identity: ChannelIdentity, *, update_id: int = 700
) -> ApplicationRequest:
    return ApplicationRequest(
        context=AuthenticatedRequestContext(
            source=RequestSource.TELEGRAM,
            canonical_user_id=user.id,
            verified_identity=VerifiedChannelIdentity(
                identity_id=identity.id,
                channel_type=identity.channel_type,
            ),
            external_session_id="2001",
        ),
        message_text="Show me all reminders",
        idempotency_key=str(update_id),
    )


def _pending_one_time_reminder(db, *, user: User, identity: ChannelIdentity, due_at):
    reminder = Reminder(
        user_id=user.id,
        task="call the bank",
        schedule_type="ONE_TIME",
        start_local=due_at.isoformat(),
        timezone="UTC",
        status="ACTIVE",
    )
    db.add(reminder)
    db.flush()
    occurrence = ReminderOccurrence(
        reminder_id=reminder.id,
        scheduled_at=due_at,
        next_due_at=due_at,
        status="PENDING",
    )
    db.add(occurrence)
    db.flush()
    db.add(
        Notification(
            occurrence_id=occurrence.id,
            destination_identity_id=identity.id,
            sequence=1,
            due_at=due_at,
            status="PENDING",
            attempt_count=0,
        )
    )
    db.flush()
    return reminder, occurrence


def test_application_boundary_is_independent_of_telegram_payloads_and_clients() -> None:
    application_paths = (
        Path(application.__path__[0]).glob("*.py")
        if hasattr(application, "__path__")
        else (Path(inspect.getsourcefile(application) or ""),)
    )
    telegram_imports = []
    for path in application_paths:
        if path.name in {"bootstrap.py", "conversation_coordinator.py"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        telegram_imports.extend(
            node
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and node.module.startswith("reminder_agent.channels.telegram")
            )
            or (
                isinstance(node, ast.Import)
                and any(
                    alias.name.startswith("reminder_agent.channels.telegram")
                    for alias in node.names
                )
            )
        )

    assert telegram_imports == []


def test_application_public_interface_is_reexported_from_the_package() -> None:
    expected = {
        "ApplicationRequest": ApplicationRequest,
        "ApplicationResult": ApplicationResult,
        "AuthenticatedRequestContext": AuthenticatedRequestContext,
        "CancelReminderCommand": CancelReminderCommand,
        "ConversationEntryResult": ConversationEntryResult,
        "DurableResponse": DurableResponse,
        "ListRemindersCommand": ListRemindersCommand,
        "RequestSource": RequestSource,
        "SaveMemoryCommand": SaveMemoryCommand,
        "VerifiedChannelIdentity": VerifiedChannelIdentity,
        "cancel_reminder_for_request": cancel_reminder_for_request,
        "interpret_for_route": interpret_for_route,
        "list_reminders_for_request": list_reminders_for_request,
        "process_authenticated_conversation": process_authenticated_conversation,
        "save_memory_for_request": save_memory_for_request,
    }

    assert set(expected).issubset(application.__all__)
    for name, value in expected.items():
        assert getattr(application, name) is value


def test_list_application_service_rechecks_context_and_keeps_reads_non_idempotent(
    session_factory, fixed_now
) -> None:
    with session_factory.begin() as db:
        user, identity = _active_telegram_context(db)
        other = User(name="Bob", timezone="UTC", onboarding_status="ACTIVE")
        db.add(other)
        db.flush()

    command = ListRemindersCommand(
        start_date=None,
        end_date=None,
        task_query=None,
        timezone_name="UTC",
    )
    with session_factory() as db:
        result = list_reminders_for_request(
            db,
            request=_request_for(user, identity),
            command=command,
            now=fixed_now,
        )
        forged = list_reminders_for_request(
            db,
            request=ApplicationRequest(
                context=AuthenticatedRequestContext(
                    source=RequestSource.TELEGRAM,
                    canonical_user_id=other.id,
                    verified_identity=VerifiedChannelIdentity(
                        identity_id=identity.id,
                        channel_type=identity.channel_type,
                    ),
                    external_session_id="2001",
                ),
                message_text="Show me all reminders",
                idempotency_key="701",
            ),
            command=command,
            now=fixed_now,
        )

        assert result.response_kind == "REMINDER_LIST"
        assert result.content == "You're all caught up! No upcoming reminders for now."
        assert forged.response_kind == "AUTHORIZATION_FAILURE"
        assert db.scalar(select(func.count()).select_from(IdempotencyRecord)) == 0


def test_memory_application_service_rechecks_context_and_stages_only_owned_fact(
    session_factory, fixed_now
) -> None:
    with session_factory.begin() as db:
        user, identity = _active_telegram_context(db)
        other = User(name="Bob", timezone="UTC", onboarding_status="ACTIVE")
        db.add(other)
        db.flush()

    with session_factory.begin() as db:
        result = save_memory_for_request(
            db,
            request=_request_for(user, identity, update_id=703),
            command=SaveMemoryCommand(content="my gate code is 4512"),
            now=fixed_now,
        )
        forged = save_memory_for_request(
            db,
            request=ApplicationRequest(
                context=AuthenticatedRequestContext(
                    source=RequestSource.TELEGRAM,
                    canonical_user_id=other.id,
                    verified_identity=VerifiedChannelIdentity(
                        identity_id=identity.id,
                        channel_type=identity.channel_type,
                    ),
                    external_session_id="2001",
                ),
                message_text="Remember my gate code is 4512",
                idempotency_key="704",
            ),
            command=SaveMemoryCommand(content="another user's fact"),
            now=fixed_now,
        )

    with session_factory() as db:
        fact = db.scalar(select(MemoryFact))
        assert result.response_kind == "MEMORY_CONFIRMATION"
        assert result.durable_response == DurableResponse(
            content="I saved that information.",
            response_kind="MEMORY_CONFIRMATION",
        )
        assert fact is not None
        assert result.resource_id == fact.id
        assert fact.user_id == user.id
        assert fact.content == "my gate code is 4512"
        assert forged.response_kind == "AUTHORIZATION_FAILURE"
        assert db.scalar(select(func.count()).select_from(IdempotencyRecord)) == 0


def test_memory_application_service_stages_no_write_after_caller_rollback(
    session_factory, fixed_now
) -> None:
    with session_factory.begin() as db:
        user, identity = _active_telegram_context(db)

    with pytest.raises(RuntimeError, match="force rollback"):
        with session_factory.begin() as db:
            save_memory_for_request(
                db,
                request=_request_for(user, identity, update_id=705),
                command=SaveMemoryCommand(content="rollback-only fact"),
                now=fixed_now,
            )
            raise RuntimeError("force rollback")

    with session_factory() as db:
        assert db.scalar(select(func.count()).select_from(MemoryFact)) == 0


def test_cancellation_application_service_rechecks_context_and_stages_owned_changes(
    session_factory, fixed_now
) -> None:
    with session_factory.begin() as db:
        user, identity = _active_telegram_context(db)
        other = User(name="Bob", timezone="UTC", onboarding_status="ACTIVE")
        db.add(other)
        db.flush()
        reminder, occurrence = _pending_one_time_reminder(
            db,
            user=user,
            identity=identity,
            due_at=fixed_now + timedelta(days=1),
        )

    command = CancelReminderCommand(
        reminder_id=reminder.id,
        occurrence_id=occurrence.id,
        scope=CancellationScope.OCCURRENCE,
    )
    with session_factory.begin() as db:
        result = cancel_reminder_for_request(
            db,
            request=_request_for(user, identity, update_id=707),
            command=command,
            now=fixed_now,
        )
        forged = cancel_reminder_for_request(
            db,
            request=ApplicationRequest(
                context=AuthenticatedRequestContext(
                    source=RequestSource.TELEGRAM,
                    canonical_user_id=other.id,
                    verified_identity=VerifiedChannelIdentity(
                        identity_id=identity.id,
                        channel_type=identity.channel_type,
                    ),
                    external_session_id="2001",
                ),
                message_text="Cancel the bank reminder",
                idempotency_key="708",
            ),
            command=command,
            now=fixed_now,
        )

    with session_factory() as db:
        canceled_reminder = db.get(Reminder, reminder.id)
        canceled_occurrence = db.get(ReminderOccurrence, occurrence.id)
        notification = db.scalar(select(Notification))
        assert result.response_kind == "CANCELLATION_REMINDER"
        assert result.durable_response.content in {
            template.format(task="call the bank")
            for template in REMINDER_CANCEL_MESSAGES["one_time_cancelled"]
        }
        assert result.resource_id == reminder.id
        assert result.affected_count == 1
        assert forged.response_kind == "AUTHORIZATION_FAILURE"
        assert canceled_reminder is not None and canceled_reminder.status == "CANCELED"
        assert canceled_occurrence is not None and canceled_occurrence.status == "CANCELED"
        assert notification is not None and notification.status == "CANCELED"
        assert db.scalar(select(func.count()).select_from(IdempotencyRecord)) == 0


def test_telegram_listing_uses_the_authenticated_application_contract(
    session_factory, fixed_now, monkeypatch, update_factory
) -> None:
    with session_factory.begin() as db:
        user, identity = _active_telegram_context(db)

    captured: dict[str, object] = {}

    def boundary_spy(db, *, request, interpreter, now, **_kwargs):
        captured["request"] = request
        captured["now"] = now
        return ConversationEntryResult(
            route=ConversationRoute.LIST_REQUEST,
            response=ApplicationResult(
                content="Boundary list response", response_kind="REMINDER_LIST"
            ),
        )

    monkeypatch.setattr(reminders, "process_authenticated_conversation", boundary_spy)
    update = TelegramUpdate.model_validate(update_factory(702, text="Show me all reminders"))

    with session_factory() as db:
        reply = reminders.handle_reminder_text(
            db,
            update,
            user_id=user.id,
            identity_id=identity.id,
            telegram_user_id=1001,
            interpreter=ReminderIntentInterpreter(),
            now=fixed_now,
        )

    request = captured["request"]
    assert isinstance(request, ApplicationRequest)
    assert request.context.source is RequestSource.TELEGRAM
    assert request.context.canonical_user_id == user.id
    assert request.context.verified_identity.identity_id == identity.id
    assert request.context.verified_identity.channel_type == "TELEGRAM"
    assert request.context.external_session_id == "2001"
    assert request.message_text == "Show me all reminders"
    assert request.idempotency_key == "702"
    assert captured["now"] == fixed_now
    assert reply.chat_id == 2001
    assert reply.text == "Boundary list response"


def test_telegram_memory_save_uses_the_authenticated_application_contract(
    session_factory, fixed_now, monkeypatch, update_factory
) -> None:
    with session_factory.begin() as db:
        user, identity = _active_telegram_context(db)

    captured: dict[str, object] = {}

    def boundary_spy(db, *, request, command, now):
        captured["request"] = request
        captured["command"] = command
        captured["now"] = now
        return ApplicationResult(
            content="I saved that information.",
            response_kind="MEMORY_CONFIRMATION",
            resource_id=uuid4(),
        )

    monkeypatch.setattr(memory_adapters, "save_memory_for_request", boundary_spy)
    update = TelegramUpdate.model_validate(
        update_factory(706, text="Remember my gate code is 4512")
    )

    with session_factory() as db:
        reply = reminders.handle_reminder_text(
            db,
            update,
            user_id=user.id,
            identity_id=identity.id,
            telegram_user_id=1001,
            interpreter=ReminderIntentInterpreter(),
            now=fixed_now,
        )

    request = captured["request"]
    assert isinstance(request, ApplicationRequest)
    assert request.context.source is RequestSource.TELEGRAM
    assert request.context.canonical_user_id == user.id
    assert request.context.verified_identity.identity_id == identity.id
    assert request.context.external_session_id == "2001"
    assert request.message_text == "Remember my gate code is 4512"
    assert request.idempotency_key == "706"
    assert captured["command"] == SaveMemoryCommand(content="my gate code is 4512")
    assert captured["now"] == fixed_now
    assert reply.chat_id == 2001
    assert reply.text == "I saved that information."
