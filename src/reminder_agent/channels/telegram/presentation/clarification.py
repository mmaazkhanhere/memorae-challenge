"""Telegram clarification presentation helpers."""

from __future__ import annotations

from collections.abc import Sequence


def clarification_text(question: str, *, choices: Sequence[str] = ()) -> str:
    """Render a validated application clarification as Telegram text."""

    if not choices:
        return question
    options = "\n".join(f"{index}. {choice}" for index, choice in enumerate(choices, 1))
    return f"{question}\n{options}"


__all__ = ["clarification_text"]
