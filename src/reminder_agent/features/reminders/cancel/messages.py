"""Personalized user-facing messages for reminder cancellation."""

from __future__ import annotations

from random import choice

REMINDER_CANCEL_MESSAGES: dict[str, list[str]] = {
    "missing_target": [
        "Of course! Which reminder would you like me to cancel?",
        "Sure thing! Which reminder should I remove?",
        "Got it! Just tell me which reminder you'd like to cancel.",
    ],
    "no_match": [
        "I couldn't find an active reminder matching that. Could you describe it another way?",
        "Hmm, I don't see that reminder on your active list. Could you give me a little more detail?",
        "I couldn't find a reminder matching that description. Mind trying a different phrase?",
    ],
    "uncertain_match": [
        "I'm not quite sure which reminder you mean. Could you tell me a little more about it?",
        "I want to make sure I cancel the right one. Could you share a bit more of the reminder's task?",
        "That sounds close to a reminder on your list, but I'm not certain it's the right one. Could you be a little more specific?",
    ],
    "multiple_matches": [
        "I found a few reminders that match. Which one would you like me to cancel?",
        "Looks like you've got more than one reminder for that. Which one did you mean?",
        "I've found a couple of possibilities. Which reminder should I cancel?",
    ],
    "partial_ambiguous_match": [
        "A few reminders look like they could be the one. Which would you like to cancel?",
        "I found some possible matches, but I don't want to cancel the wrong one. Which did you mean?",
        "There are a few reminders that sound similar. Could you pick the one you want to cancel?",
    ],
    "pending_selection_ambiguous": [
        "Just to be sure I cancel the right one, which reminder did you mean?",
        "I still need to know which one to cancel. Could you choose from the reminders below?",
        "Which of these reminders should I cancel? You can reply with its number.",
    ],
    "reminder_inactive": [
        "Looks like that reminder isn't active anymore, so there's nothing to cancel.",
        "That reminder is no longer active. I haven't made any changes.",
        "I couldn't cancel that one because it's no longer active.",
    ],
    "cancellation_abandoned": [
        "No problem! I've left your reminders as they are.",
        "Got it! I won't cancel anything.",
        "All good! Your reminders will stay as scheduled.",
        "Of course! Nothing has been cancelled.",
    ],
    "authorization_failed": [
        "I couldn't verify your Telegram account, so I haven't cancelled anything.",
        "I wasn't able to confirm your Telegram identity, so I couldn't cancel the reminder.",
    ],
    "missing_idempotency_key": [
        "I couldn't process that cancellation request. Could you try again?",
        "Something went wrong while processing your request. Please try cancelling the reminder again.",
    ],
    "cancellation_failed": [
        "I couldn't cancel that reminder because {reason}. Please try again.",
        "Something prevented me from cancelling that reminder: {reason}.",
        "I wasn't able to cancel that one. Here's why: {reason}.",
    ],
    "one_time_cancelled": [
        "All set! I've cancelled your reminder to {task}.",
        "Got it! Your '{task}' reminder has been cancelled.",
        "Done! I've removed the reminder to {task} from your upcoming schedule.",
        "Of course! You won't receive that '{task}' reminder anymore.",
    ],
    "recurring_series_cancelled": [
        "All set! I've cancelled your recurring '{task}' reminder and its {count} upcoming {occurrence_word}.",
        "Got it! I've stopped all future '{task}' reminders. {count} upcoming {occurrence_word} have been cancelled.",
        "Done! Your recurring '{task}' reminder is cancelled, along with its {count} pending {occurrence_word}.",
    ],
    "recurring_occurrence_cancelled": [
        "Got it! I've cancelled the upcoming '{task}' reminder. The rest of the series is still active.",
        "All set! I've skipped the selected '{task}' reminder. Your future reminders will continue as usual.",
        "Done! That '{task}' occurrence is cancelled. Your recurring schedule hasn't changed.",
    ],
}


def cancel_message(key: str, **values: object) -> str:
    """Select and format one cancellation message variant."""

    try:
        template = choice(REMINDER_CANCEL_MESSAGES[key])
    except KeyError as exc:
        raise ValueError(f"unknown cancellation message key: {key}") from exc
    return template.format(**values)


__all__ = ["REMINDER_CANCEL_MESSAGES", "cancel_message"]
