from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from reminder_agent.application.conversations import save_pending_clarification_session
from reminder_agent.application.pending_contracts import PendingCompletion, PendingSnooze
from reminder_agent.features.reminders.complete.pending import prepare_pending_completion
from reminder_agent.features.reminders.snooze.service import handle_snooze_clarification
from reminder_agent.infrastructure.database.models import ConversationSession, User

NOW = datetime(2026, 11, 3, 12, tzinfo=UTC)


def test_pending_completion_preparation_isolates_channel_session_and_matches_target(db):
    user = User(name="Ada", timezone="UTC", onboarding_status="ACTIVE")
    db.add(user)
    db.commit()
    pending = PendingCompletion(candidate_occurrence_ids=(uuid4(),))
    save_pending_clarification_session(
        db,
        user_id=user.id,
        external_session_id="session-1",
        pending=pending,
        now=NOW,
        channel_type="TELEGRAM",
    )
    db.commit()

    candidate = (object(), object(), object())

    def resolver(*_args, **_kwargs):
        return [candidate]

    def matcher(*_args, **_kwargs):
        return [candidate]

    assert (
        prepare_pending_completion(
            db,
            user_id=user.id,
            identity_id=uuid4(),
            channel_type="VOICE_CALL",
            external_session_id="session-1",
            chat_id=None,
            external_destination_id="voice-1",
            text="James",
            now=NOW,
            session_ttl_seconds=900,
            resolve_by_occurrence_fn=resolver,
            match_candidates_fn=matcher,
        )
        is None
    )

    preparation = prepare_pending_completion(
        db,
        user_id=user.id,
        identity_id=uuid4(),
        channel_type="TELEGRAM",
        external_session_id="session-1",
        chat_id=2001,
        external_destination_id=None,
        text="James",
        now=NOW,
        session_ttl_seconds=900,
        resolve_by_occurrence_fn=resolver,
        match_candidates_fn=matcher,
    )
    assert preparation is not None
    assert preparation.candidates == (candidate,)
    assert preparation.matches == (candidate,)


def test_snooze_workflow_persists_pending_session_and_stages_response(db):
    user = User(name="Ada", timezone="UTC", onboarding_status="ACTIVE")
    db.add(user)
    db.commit()
    pending = PendingSnooze(candidate_occurrence_ids=(uuid4(),), duration_seconds=None)

    response = handle_snooze_clarification(
        db,
        user_id=user.id,
        channel_type="TELEGRAM",
        external_session_id="2001",
        pending=pending,
        response_text="How long should I snooze that reminder?",
        response_kind="snooze_clarification",
        now=NOW,
        session_ttl_seconds=900,
        upsert_pending=lambda **kwargs: save_pending_clarification_session(db, **kwargs),
        find_existing=lambda: None,
        stage_response=lambda **kwargs: kwargs,
        recover_from_race=lambda: None,
    )

    session = db.scalar(
        select(ConversationSession).where(
            ConversationSession.user_id == user.id,
            ConversationSession.external_session_id == "2001",
        )
    )
    assert response["response_kind"] == "snooze_clarification"
    assert session is not None
    assert session.pending_clarification == pending.model_dump(mode="json")


def test_snooze_workflow_replays_existing_response_without_staging(db):
    staged = False

    def stage_response(**_kwargs):
        nonlocal staged
        staged = True
        return "new"

    response = handle_snooze_clarification(
        db,
        user_id=uuid4(),
        channel_type="TELEGRAM",
        external_session_id="2002",
        pending=None,
        response_text="unused",
        response_kind="snooze_response",
        now=NOW,
        session_ttl_seconds=900,
        upsert_pending=lambda **_kwargs: None,
        find_existing=lambda: "replayed",
        stage_response=stage_response,
        recover_from_race=lambda: None,
    )

    assert response == "replayed"
    assert staged is False


def test_snooze_workflow_recovers_a_response_staging_integrity_race(db):
    response = handle_snooze_clarification(
        db,
        user_id=uuid4(),
        channel_type="TELEGRAM",
        external_session_id="2003",
        pending=None,
        response_text="unused",
        response_kind="snooze_response",
        now=NOW,
        session_ttl_seconds=900,
        upsert_pending=lambda **_kwargs: None,
        find_existing=lambda: None,
        stage_response=lambda **_kwargs: (_ for _ in ()).throw(
            IntegrityError("stage", {}, RuntimeError("duplicate"))
        ),
        recover_from_race=lambda: "recovered",
    )

    assert response == "recovered"
