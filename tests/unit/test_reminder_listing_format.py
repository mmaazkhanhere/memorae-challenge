from datetime import UTC, date, datetime
from uuid import uuid4

from reminder_agent.features.reminders.listing.filters import (
    ListedReminder,
    format_reminder_list,
)


def test_specific_date_listing_does_not_repeat_date_in_single_item() -> None:
    text = format_reminder_list(
        [
            ListedReminder(
                uuid4(),
                "take nap",
                datetime(2026, 11, 10, 14, 10, tzinfo=UTC),
            )
        ],
        start_date=date(2026, 11, 10),
        end_date=date(2026, 11, 10),
        timezone_name="Asia/Karachi",
        now=datetime(2026, 11, 9, 12, tzinfo=UTC),
    )

    assert "tomorrow:" in text
    assert "- 19:10 PKT - take nap" in text
    assert "- tomorrow at" not in text
