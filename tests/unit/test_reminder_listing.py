from datetime import date, datetime, timezone
from uuid import UUID

import pytest

from reminder_agent.db.models import (
    Reminder,
    ReminderOccurrence,
    ReminderRecurrenceException,
    User,
)
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


def test_listing_task_filter_combines_with_date_and_owner_scope(session_factory):
    with session_factory.begin() as db:
        ada = User(name="Ada", timezone="UTC", onboarding_status="ACTIVE")
        bob = User(name="Bob", timezone="UTC", onboarding_status="ACTIVE")
        db.add_all([ada, bob])
        db.flush()
        rows = [
            (ada, "doctor appointment", datetime(2026, 11, 1, 9, tzinfo=timezone.utc)),
            (ada, "call John", datetime(2026, 11, 1, 10, tzinfo=timezone.utc)),
            (ada, "gym workout", datetime(2026, 11, 2, 9, tzinfo=timezone.utc)),
            (bob, "doctor appointment", datetime(2026, 11, 1, 8, tzinfo=timezone.utc)),
        ]
        for owner, task, due_at in rows:
            reminder = Reminder(
                user_id=owner.id, task=task, schedule_type="ONE_TIME",
                start_local=due_at.isoformat(), timezone="UTC", status="ACTIVE",
            )
            db.add(reminder)
            db.flush()
            db.add(ReminderOccurrence(
                reminder_id=reminder.id, scheduled_at=due_at, next_due_at=due_at, status="PENDING"
            ))

    with session_factory() as db:
        doctors = list_reminders_in_range(
            db, user_id=ada.id, start_date=date(2026, 11, 1), end_date=date(2026, 11, 1),
            task_query="doctor's appointments", timezone_name="UTC",
        )
        gym = list_reminders_in_range(
            db, user_id=ada.id, start_date=None, end_date=None, task_query="gym", timezone_name="UTC",
            now=datetime(2026, 10, 30, tzinfo=timezone.utc),
        )
        missing = list_reminders_in_range(
            db, user_id=ada.id, start_date=None, end_date=None, task_query="dentist", timezone_name="UTC",
            now=datetime(2026, 10, 30, tzinfo=timezone.utc),
        )

    assert [item.task for item in doctors] == ["doctor appointment"]
    assert [item.task for item in gym] == ["gym workout"]
    assert missing == []
    assert format_reminder_list(
        missing, start_date=None, end_date=None, task_query="dentist", timezone_name="UTC"
    ) == 'You have no upcoming active reminders matching "dentist".'


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


def test_listing_rejects_a_start_date_without_an_end_date(session_factory):
    with session_factory() as db, pytest.raises(
        ValueError, match="start_date and end_date must be supplied together"
    ):
        list_reminders_in_range(
            db,
            user_id=UUID(int=1),
            start_date=date(2026, 11, 1),
            end_date=None,
            timezone_name="UTC",
        )


def test_listing_rejects_an_end_date_before_its_start_date(session_factory):
    with session_factory() as db, pytest.raises(
        ValueError, match="end_date must not precede start_date"
    ):
        list_reminders_in_range(
            db,
            user_id=UUID(int=1),
            start_date=date(2026, 11, 2),
            end_date=date(2026, 11, 1),
            timezone_name="UTC",
        )


def test_all_reminders_formatting_lists_nonempty_results():
    rendered = format_reminder_list(
        [
            ListedReminder(
                reminder_id=UUID(int=1),
                due_at=datetime(2026, 11, 1, 9, tzinfo=timezone.utc),
                task="call the bank",
            ),
            ListedReminder(
                reminder_id=UUID(int=2),
                due_at=datetime(2026, 11, 2, 10, 30, tzinfo=timezone.utc),
                task="pay rent",
            ),
        ],
        start_date=None,
        end_date=None,
        timezone_name="UTC",
    )

    lines = rendered.splitlines()
    assert lines[0] == "All upcoming active reminders (UTC):"
    assert "2026-11-01 09:00 UTC" in lines[1]
    assert lines[1].endswith("call the bank")
    assert "2026-11-02 10:30 UTC" in lines[2]
    assert lines[2].endswith("pay rent")


def test_multi_day_range_formatting_includes_dates_and_each_reminder():
    rendered = format_reminder_list(
        [
            ListedReminder(
                reminder_id=UUID(int=1),
                due_at=datetime(2026, 11, 1, 9, tzinfo=timezone.utc),
                task="call the bank",
            ),
            ListedReminder(
                reminder_id=UUID(int=2),
                due_at=datetime(2026, 11, 3, 17, tzinfo=timezone.utc),
                task="send the report",
            ),
        ],
        start_date=date(2026, 11, 1),
        end_date=date(2026, 11, 3),
        timezone_name="UTC",
    )

    lines = rendered.splitlines()
    assert lines[0] == "Reminders from 2026-11-01 to 2026-11-03 (UTC):"
    assert "2026-11-01 09:00 UTC" in lines[1]
    assert lines[1].endswith("call the bank")
    assert "2026-11-03 17:00 UTC" in lines[2]
    assert lines[2].endswith("send the report")


def test_existing_recurring_occurrence_suppresses_matching_projected_instance(session_factory):
    with session_factory.begin() as db:
        user = User(name="Ada", timezone="UTC", onboarding_status="ACTIVE")
        db.add(user)
        db.flush()
        reminder = Reminder(
            user_id=user.id,
            task="weekly review",
            schedule_type="RECURRING",
            start_local="2026-11-02T09:00",
            timezone="UTC",
            recurrence_rule="FREQ=WEEKLY;BYDAY=MO;BYHOUR=09;BYMINUTE=00",
            status="ACTIVE",
        )
        db.add(reminder)
        db.flush()
        db.add(
            ReminderOccurrence(
                reminder_id=reminder.id,
                scheduled_at=datetime(2026, 11, 2, 9, tzinfo=timezone.utc),
                next_due_at=datetime(2026, 11, 2, 9, tzinfo=timezone.utc),
                recurrence_instance_key="2026-11-02T09:00",
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


def test_listing_excludes_unmaterialized_canceled_recurring_instance(session_factory):
    with session_factory.begin() as db:
        user = User(name="Ada", timezone="UTC", onboarding_status="ACTIVE")
        db.add(user)
        db.flush()
        reminder = Reminder(
            user_id=user.id,
            task="review finances",
            schedule_type="RECURRING",
            start_local="2026-11-06T10:00",
            timezone="UTC",
            recurrence_rule="FREQ=WEEKLY;BYDAY=FR;BYHOUR=10;BYMINUTE=00",
            status="ACTIVE",
        )
        db.add(reminder)
        db.flush()
        db.add(
            ReminderRecurrenceException(
                reminder_id=reminder.id,
                occurrence_date=date(2027, 1, 8),
            )
        )

    with session_factory() as db:
        result = list_reminders_in_range(
            db,
            user_id=user.id,
            start_date=date(2027, 1, 8),
            end_date=date(2027, 1, 8),
            timezone_name="UTC",
        )

    assert result == []


def test_listing_includes_only_occurrences_at_the_requested_local_midnight_boundary(
    session_factory,
):
    with session_factory.begin() as db:
        user = User(name="Ada", timezone="Asia/Karachi", onboarding_status="ACTIVE")
        db.add(user)
        db.flush()
        reminders = []
        for task, due_at in (
            ("before range", datetime(2026, 10, 31, 18, 59, tzinfo=timezone.utc)),
            ("at local midnight", datetime(2026, 10, 31, 19, 0, tzinfo=timezone.utc)),
            ("next local midnight", datetime(2026, 11, 1, 19, 0, tzinfo=timezone.utc)),
        ):
            reminder = Reminder(
                user_id=user.id,
                task=task,
                schedule_type="ONE_TIME",
                start_local=due_at.isoformat(),
                timezone="Asia/Karachi",
                status="ACTIVE",
            )
            db.add(reminder)
            db.flush()
            reminders.append(reminder)
            db.add(
                ReminderOccurrence(
                    reminder_id=reminder.id,
                    scheduled_at=due_at,
                    next_due_at=due_at,
                    status="PENDING",
                )
            )

    with session_factory() as db:
        result = list_reminders_in_range(
            db,
            user_id=user.id,
            start_date=date(2026, 11, 1),
            end_date=date(2026, 11, 1),
            timezone_name="Asia/Karachi",
        )

    assert [(item.task, item.due_at) for item in result] == [
        ("at local midnight", datetime(2026, 10, 31, 19, tzinfo=timezone.utc))
    ]
