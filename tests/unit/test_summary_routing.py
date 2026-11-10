from datetime import UTC, datetime

from reminder_agent.features.intent.contracts import IntentRoute
from reminder_agent.features.intent.deterministic.interpreter import RuleBasedIntentInterpreter
from reminder_agent.features.intent.deterministic.routing import ConversationRoute, route_message


def test_quoted_pending_summary_routes_deterministically() -> None:
    text = '"How many reminders do I have pending?"'
    assert route_message(text, pending=None).route is ConversationRoute.SUMMARY_REQUEST

    result = RuleBasedIntentInterpreter().interpret(
        text,
        selected_route=IntentRoute.SUMMARY_REQUEST,
        timezone_name="Asia/Karachi",
        now=datetime(2026, 11, 9, 12, tzinfo=UTC),
    )
    assert result.intent is not None
    assert result.intent.metric == "PENDING_COUNT"
