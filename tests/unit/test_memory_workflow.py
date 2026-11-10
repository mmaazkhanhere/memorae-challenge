from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from reminder_agent.application import (
    ApplicationRequest,
    ApplicationResult,
    AuthenticatedRequestContext,
    RequestSource,
    VerifiedChannelIdentity,
)
from reminder_agent.features.memory.create.workflow import run_memory_save_workflow
from reminder_agent.features.memory.create.validation import validated_memory_content


class _Responses:
    def __init__(self) -> None:
        self.events: list[str] = []

    def existing(self):
        return None

    def recover_from_race(self):
        return None

    def direct(self, text: str):
        self.events.append(f"direct:{text}")
        return text

    def stage(self, *, user_id, text=None, durable_response=None, response_kind, operation):
        self.events.append(f"stage:{response_kind}:{operation}")
        return durable_response.content if durable_response is not None else text


def test_incomplete_memory_subject_requires_value_before_save() -> None:
    for text in (
        "Remember my locker number",
        "Remember my locker number is",
        "Remember my parking spot",
    ):
        assert (
            validated_memory_content(
                text=text,
                pending=None,
                model_content=None,
                from_model=False,
            )
            is None
        )

    assert (
        validated_memory_content(
            text="Remember my parking spot",
            pending=None,
            model_content="my parking spot",
            from_model=True,
        )
        is None
    )


def test_memory_workflow_keeps_validation_before_transactional_save(session_factory) -> None:
    user_id = uuid4()
    identity_id = uuid4()
    request = ApplicationRequest(
        context=AuthenticatedRequestContext(
            source=RequestSource.TELEGRAM,
            canonical_user_id=user_id,
            verified_identity=VerifiedChannelIdentity(
                identity_id=identity_id,
                channel_type="TELEGRAM",
            ),
            external_session_id="2001",
        ),
        message_text="Remember my passport is in the safe",
        idempotency_key="902",
    )
    responses = _Responses()
    events: list[str] = []

    def save_memory(db, *, request, command, now):
        events.append(f"save:{command.content}")
        return ApplicationResult(
            content="I saved that information.",
            response_kind="MEMORY_CONFIRMATION",
            resource_id=uuid4(),
        )

    with session_factory() as db:
        result = run_memory_save_workflow(
            db,
            request=request,
            user_id=user_id,
            pending=None,
            text=request.message_text,
            now=datetime(2026, 10, 30, 12, 0, tzinfo=UTC),
            session_ttl_seconds=900,
            responses=responses,
            upsert_clarification_session=lambda *args, **kwargs: None,
            save_memory_operation=save_memory,
        )

    assert result == "I saved that information."
    assert events == ["save:my passport is in the safe"]
    assert responses.events == ["stage:memory_confirmation:SAVE_MEMORY"]
