from datetime import date, datetime, timezone
from uuid import UUID

from reminder_agent.db.models import Reminder, ReminderOccurrence, User
from reminder_agent.reminder_listing import (
    ListedReminder,
    format_reminder_list,
    list_reminders_in_range,
)


def test_listing_projects_unmaterialized_recurring_instance_and_excludes_terminal_rows(
    session_factory,
):
    with session_factory.begin() as db:
        user = User(name="Ada", timezone="UTC", onboarding_status="ACTIVE")
        other = User(name="Bob", timezone="UTC", onboarding_status="ACTIVE")
        db.add_all([user, other])
        db.flush()
        recurring = Reminder(
            user_id=user.id,
            task="weekly review",
            schedule_type="RECURRING",
            start_local="2026-11-02T09:00",
            timezone="UTC",
            recurrence_rule="FREQ=WEEKLY;BYDAY=MO;BYHOUR=09;BYMINUTE=00",
            status="ACTIVE",
        )
        one_time = Reminder(
            user_id=user.id,
            task="completed task",
            schedule_type="ONE_TIME",
            start_local="2026-11-03T09:00+00:00",
            timezone="UTC",
            status="ACTIVE",
        )
        hidden = Reminder(
            user_id=other.id,
            task="private task",
            schedule_type="ONE_TIME",
            start_local="2026-11-03T09:00+00:00",
            timezone="UTC",
            status="ACTIVE",
        )
        canceled = Reminder(
            user_id=user.id,
            task="canceled parent",
            schedule_type="ONE_TIME",
            start_local="2026-11-03T10:00+00:00",
            timezone="UTC",
            status="CANCELED",
        )
        db.add_all([recurring, one_time, hidden, canceled])
        db.flush()
        db.add(
            ReminderOccurrence(
                reminder_id=one_time.id,
                scheduled_at=datetime(2026, 11, 3, 9, tzinfo=timezone.utc),
                next_due_at=datetime(2026, 11, 3, 9, tzinfo=timezone.utc),
                status="COMPLETED",
            )
        )
        db.flush()
        db.add(
            ReminderOccurrence(
                reminder_id=canceled.id,
                scheduled_at=datetime(2026, 11, 3, 10, tzinfo=timezone.utc),
                next_due_at=datetime(2026, 11, 3, 10, tzinfo=timezone.utc),
                status="PENDING",
            )
        )
        db.flush()
        db.add(
            ReminderOccurrence(
                reminder_id=hidden.id,
                scheduled_at=datetime(2026, 11, 3, 11, tzinfo=timezone.utc),
                next_due_at=datetime(2026, 11, 3, 11, tzinfo=timezone.utc),
                status="PENDING",
            )
        )

    with session_factory() as db:
        result = list_reminders_in_range(
            db,
            user_id=user.id,
            start_date=date(2026, 11, 2),
            end_date=date(2026, 11, 8),
            timezone_name="UTC",
        )

    assert [(item.task, item.due_at) for item in result] == [
        ("weekly review", datetime(2026, 11, 2, 9, tzinfo=timezone.utc))
    ]


def test_listing_uses_local_midnight_boundaries_across_dst(session_factory):
    with session_factory.begin() as db:
        user = User(name="DST", timezone="America/New_York", onboarding_status="ACTIVE")
        db.add(user)
        db.flush()
        reminder = Reminder(
            user_id=user.id,
            task="DST appointment",
            schedule_type="ONE_TIME",
            start_local="2027-03-14T09:00-04:00",
            timezone="America/New_York",
            status="ACTIVE",
        )
        db.add(reminder)
        db.flush()
        db.add(
            ReminderOccurrence(
                reminder_id=reminder.id,
                scheduled_at=datetime(2027, 3, 14, 13, tzinfo=timezone.utc),
                next_due_at=datetime(2027, 3, 14, 13, tzinfo=timezone.utc),
                status="PENDING",
            )
        )

    with session_factory() as db:
        result = list_reminders_in_range(
            db,
            user_id=user.id,
            start_date=date(2027, 3, 14),
            end_date=date(2027, 3, 14),
            timezone_name="America/New_York",
        )

    assert len(result) == 1
    assert result[0].due_at == datetime(2027, 3, 14, 13, tzinfo=timezone.utc)


def test_empty_listing_is_explicit_and_formatting_is_local():
    assert format_reminder_list(
        [], start_date=None, end_date=None, timezone_name="UTC"
    ) == "You have no upcoming active reminders."
    assert format_reminder_list(
        [], start_date=date(2026, 11, 1), end_date=date(2026, 11, 1), timezone_name="UTC"
    ) == "You have no active reminders on 2026-11-01."
    assert "2026-11-01 09:00 UTC" in format_reminder_list(
        [
            ListedReminder(
                reminder_id=UUID(int=0),
                due_at=datetime(2026, 11, 1, 9, tzinfo=timezone.utc),
                task="call the bank",
            )
        ],
        start_date=date(2026, 11, 1),
        end_date=date(2026, 11, 1),
        timezone_name="UTC",
    )
