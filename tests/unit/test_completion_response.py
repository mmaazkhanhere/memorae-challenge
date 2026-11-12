import pytest
from sqlalchemy.exc import IntegrityError

from reminder_agent.features.reminders.complete.response import (
    commit_completion_response,
    stage_completion_response,
)


class FailingCommitSession:
    def __init__(self) -> None:
        self.commits = 0

    def commit(self) -> None:
        self.commits += 1
        raise IntegrityError("commit", {}, RuntimeError("duplicate"))


def test_completion_response_staging_preserves_durable_operation_metadata() -> None:
    calls: list[dict[str, str]] = []

    def stage_response(**kwargs: str) -> str:
        calls.append(kwargs)
        return "reply"

    result = stage_completion_response(
        stage_response=stage_response,
        text="Marked as done: call the bank.",
        response_kind="completion",
    )

    assert result == "reply"
    assert calls == [
        {
            "text": "Marked as done: call the bank.",
            "response_kind": "completion",
            "operation": "COMPLETE_REMINDER",
        }
    ]


def test_completion_response_replays_duplicate_commit_winner() -> None:
    db = FailingCommitSession()
    replay = object()

    result = commit_completion_response(
        db,  # type: ignore[arg-type]
        reply=object(),
        recover_from_race=lambda: replay,
    )

    assert result is replay
    assert db.commits == 1


def test_completion_response_reraises_unknown_commit_failure() -> None:
    db = FailingCommitSession()

    with pytest.raises(IntegrityError):
        commit_completion_response(
            db,  # type: ignore[arg-type]
            reply=object(),
            recover_from_race=lambda: None,
        )
