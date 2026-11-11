from datetime import UTC, date, datetime, time
from uuid import UUID

import pytest

from reminder_agent.features.reminders.listing.filters import (
    REMINDER_LIST_MESSAGES,
    ListedReminder,
    format_reminder_list,
    format_reminder_summary,
    list_reminders_in_range,
    summarize_reminders,
)
from reminder_agent.infrastructure.database.models import (
    Reminder,
    ReminderOccurrence,
    ReminderRecurrenceException,
    User,
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
                scheduled_at=datetime(2026, 11, 3, 9, tzinfo=UTC),
                next_due_at=datetime(2026, 11, 3, 9, tzinfo=UTC),
                status="COMPLETED",
            )
        )
        db.flush()
        db.add(
            ReminderOccurrence(
                reminder_id=canceled.id,
                scheduled_at=datetime(2026, 11, 3, 10, tzinfo=UTC),
                next_due_at=datetime(2026, 11, 3, 10, tzinfo=UTC),
                status="PENDING",
            )
        )
        db.flush()
        db.add(
            ReminderOccurrence(
                reminder_id=hidden.id,
                scheduled_at=datetime(2026, 11, 3, 11, tzinfo=UTC),
                next_due_at=datetime(2026, 11, 3, 11, tzinfo=UTC),
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
        ("weekly review", datetime(2026, 11, 2, 9, tzinfo=UTC))
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
                scheduled_at=datetime(2027, 3, 14, 13, tzinfo=UTC),
                next_due_at=datetime(2027, 3, 14, 13, tzinfo=UTC),
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
    assert result[0].due_at == datetime(2027, 3, 14, 13, tzinfo=UTC)


def test_listing_task_filter_combines_with_date_and_owner_scope(session_factory):
    with session_factory.begin() as db:
        ada = User(name="Ada", timezone="UTC", onboarding_status="ACTIVE")
        bob = User(name="Bob", timezone="UTC", onboarding_status="ACTIVE")
        db.add_all([ada, bob])
        db.flush()
        rows = [
            (ada, "doctor appointment", datetime(2026, 11, 1, 9, tzinfo=UTC)),
            (ada, "call John", datetime(2026, 11, 1, 10, tzinfo=UTC)),
            (ada, "gym workout", datetime(2026, 11, 2, 9, tzinfo=UTC)),
            (bob, "doctor appointment", datetime(2026, 11, 1, 8, tzinfo=UTC)),
        ]
        for owner, task, due_at in rows:
            reminder = Reminder(
                user_id=owner.id,
                task=task,
                schedule_type="ONE_TIME",
                start_local=due_at.isoformat(),
                timezone="UTC",
                status="ACTIVE",
            )
            db.add(reminder)
            db.flush()
            db.add(
                ReminderOccurrence(
                    reminder_id=reminder.id,
                    scheduled_at=due_at,
                    next_due_at=due_at,
                    status="PENDING",
                )
            )

    with session_factory() as db:
        doctors = list_reminders_in_range(
            db,
            user_id=ada.id,
            start_date=date(2026, 11, 1),
            end_date=date(2026, 11, 1),
            task_query="doctor's appointments",
            timezone_name="UTC",
        )
        gym = list_reminders_in_range(
            db,
            user_id=ada.id,
            start_date=None,
            end_date=None,
            task_query="gym",
            timezone_name="UTC",
            now=datetime(2026, 10, 30, tzinfo=UTC),
        )
        missing = list_reminders_in_range(
            db,
            user_id=ada.id,
            start_date=None,
            end_date=None,
            task_query="dentist",
            timezone_name="UTC",
            now=datetime(2026, 10, 30, tzinfo=UTC),
        )

    assert [item.task for item in doctors] == ["doctor appointment"]
    assert [item.task for item in gym] == ["gym workout"]
    assert missing == []
    rendered = format_reminder_list(
        missing, start_date=None, end_date=None, task_query="dentist", timezone_name="UTC"
    )
    assert rendered in {
        message.format(query="dentist") for message in REMINDER_LIST_MESSAGES["empty_task_search"]
    }


def test_listing_filters_by_local_time_weekday_and_limit(session_factory):
    with session_factory.begin() as db:
        user = User(name="Filters", timezone="UTC", onboarding_status="ACTIVE")
        db.add(user)
        db.flush()
        rows = [
            ("weekday evening", datetime(2026, 11, 9, 18, 30, tzinfo=UTC)),
            ("weekday morning", datetime(2026, 11, 10, 8, tzinfo=UTC)),
            ("saturday evening", datetime(2026, 11, 14, 20, tzinfo=UTC)),
            ("sunday evening", datetime(2026, 11, 15, 19, tzinfo=UTC)),
        ]
        for task, due_at in rows:
            reminder = Reminder(
                user_id=user.id,
                task=task,
                schedule_type="ONE_TIME",
                start_local=due_at.isoformat(),
                timezone="UTC",
                status="ACTIVE",
            )
            db.add(reminder)
            db.flush()
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
            start_date=None,
            end_date=None,
            timezone_name="UTC",
            time_after=time(18),
            weekdays=(5, 6),
            limit=1,
            now=datetime(2026, 11, 8, tzinfo=UTC),
        )

    assert [item.task for item in result] == ["saturday evening"]


def test_empty_listing_is_explicit_and_formatting_is_local():
    assert format_reminder_list([], start_date=None, end_date=None, timezone_name="UTC") in {
        message for message in REMINDER_LIST_MESSAGES["empty_upcoming"]
    }
    assert format_reminder_list(
        [], start_date=date(2026, 11, 1), end_date=date(2026, 11, 1), timezone_name="UTC"
    ) in {
        message.format(date="2026-11-01")
        for message in REMINDER_LIST_MESSAGES["empty_specific_date"]
    }
    assert "09:00 UTC - call the bank" in format_reminder_list(
        [
            ListedReminder(
                reminder_id=UUID(int=0),
                due_at=datetime(2026, 11, 1, 9, tzinfo=UTC),
                task="call the bank",
            )
        ],
        start_date=date(2026, 11, 1),
        end_date=date(2026, 11, 1),
        timezone_name="UTC",
    )


def test_summary_counts_are_owner_scoped_and_today_keeps_delivered_work(
    session_factory,
):
    now = datetime(2026, 11, 1, 12, tzinfo=UTC)
    with session_factory.begin() as db:
        ada = User(name="Ada", timezone="UTC", onboarding_status="ACTIVE")
        bob = User(name="Bob", timezone="UTC", onboarding_status="ACTIVE")
        db.add_all([ada, bob])
        db.flush()
        rows = [
            (ada, "pending task", datetime(2026, 11, 1, 9, tzinfo=UTC), "PENDING"),
            (ada, "delivered task", datetime(2026, 11, 1, 10, tzinfo=UTC), "NOTIFIED"),
            (ada, "completed task", datetime(2026, 10, 31, 10, tzinfo=UTC), "COMPLETED"),
            (bob, "private task", datetime(2026, 11, 1, 11, tzinfo=UTC), "PENDING"),
        ]
        for owner, task, due_at, status in rows:
            reminder = Reminder(
                user_id=owner.id,
                task=task,
                schedule_type="ONE_TIME",
                start_local=due_at.isoformat(),
                timezone="UTC",
                status="COMPLETED" if status == "COMPLETED" else "ACTIVE",
            )
            db.add(reminder)
            db.flush()
            db.add(
                ReminderOccurrence(
                    reminder_id=reminder.id,
                    scheduled_at=due_at,
                    next_due_at=due_at,
                    status=status,
                    updated_at=(
                        datetime(2026, 11, 1, 11, tzinfo=UTC) if status == "COMPLETED" else due_at
                    ),
                )
            )

    with session_factory() as db:
        pending = summarize_reminders(
            db,
            user_id=ada.id,
            metric="PENDING_COUNT",
            timezone_name="UTC",
            now=now,
        )
        remaining = summarize_reminders(
            db,
            user_id=ada.id,
            metric="TODAY_REMAINING",
            timezone_name="UTC",
            now=now,
        )
        completed = summarize_reminders(
            db,
            user_id=ada.id,
            metric="COMPLETED_COUNT",
            timezone_name="UTC",
            now=now,
        )

    assert pending.count == 2
    assert remaining.count == 2
    assert {item.task for item in remaining.reminders} == {"pending task", "delivered task"}
    assert completed.count == 1
    assert format_reminder_summary(remaining, timezone_name="UTC").startswith(
        "You still have 2 tasks to do today:"
    )


def test_listing_rejects_a_start_date_without_an_end_date(session_factory):
    with (
        session_factory() as db,
        pytest.raises(ValueError, match="start_date and end_date must be supplied together"),
    ):
        list_reminders_in_range(
            db,
            user_id=UUID(int=1),
            start_date=date(2026, 11, 1),
            end_date=None,
            timezone_name="UTC",
        )


def test_listing_rejects_an_end_date_before_its_start_date(session_factory):
    with (
        session_factory() as db,
        pytest.raises(ValueError, match="end_date must not precede start_date"),
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
                due_at=datetime(2026, 11, 1, 9, tzinfo=UTC),
                task="call the bank",
            ),
            ListedReminder(
                reminder_id=UUID(int=2),
                due_at=datetime(2026, 11, 2, 10, 30, tzinfo=UTC),
                task="pay rent",
            ),
        ],
        start_date=None,
        end_date=None,
        timezone_name="UTC",
    )

    lines = rendered.splitlines()
    assert lines[0] in {
        message.format(count=2, reminder_word="reminders")
        for message in REMINDER_LIST_MESSAGES["upcoming"]
    }
    assert lines[1] == "2026-11-01:"
    assert lines[2] == "- 09:00 UTC - call the bank"
    assert lines[3] == "2026-11-02:"
    assert lines[4] == "- 10:30 UTC - pay rent"


def test_multi_day_range_formatting_includes_dates_and_each_reminder():
    rendered = format_reminder_list(
        [
            ListedReminder(
                reminder_id=UUID(int=1),
                due_at=datetime(2026, 11, 1, 9, tzinfo=UTC),
                task="call the bank",
            ),
            ListedReminder(
                reminder_id=UUID(int=2),
                due_at=datetime(2026, 11, 3, 17, tzinfo=UTC),
                task="send the report",
            ),
        ],
        start_date=date(2026, 11, 1),
        end_date=date(2026, 11, 3),
        timezone_name="UTC",
    )

    lines = rendered.splitlines()
    assert lines[0] in {
        message.format(
            count=2,
            reminder_word="reminders",
            start_date="2026-11-01",
            end_date="2026-11-03",
        )
        for message in REMINDER_LIST_MESSAGES["date_range"]
    }
    assert lines[1] == "2026-11-01:"
    assert lines[2] == "- 09:00 UTC - call the bank"
    assert lines[3] == "2026-11-03:"
    assert lines[4] == "- 17:00 UTC - send the report"


def test_listing_uses_user_timezone_for_relative_dates_and_singular_grammar():
    rendered = format_reminder_list(
        [
            ListedReminder(
                reminder_id=UUID(int=1),
                due_at=datetime(2026, 11, 10, 10, tzinfo=UTC),
                task="call the bank",
            )
        ],
        start_date=None,
        end_date=None,
        timezone_name="Asia/Karachi",
        now=datetime(2026, 11, 9, 23, 30, tzinfo=UTC),
    )

    assert "1 reminders" not in rendered
    assert "today at 15:00 PKT" in rendered


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
                scheduled_at=datetime(2026, 11, 2, 9, tzinfo=UTC),
                next_due_at=datetime(2026, 11, 2, 9, tzinfo=UTC),
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
        ("weekly review", datetime(2026, 11, 2, 9, tzinfo=UTC))
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
            ("before range", datetime(2026, 10, 31, 18, 59, tzinfo=UTC)),
            ("at local midnight", datetime(2026, 10, 31, 19, 0, tzinfo=UTC)),
            ("next local midnight", datetime(2026, 11, 1, 19, 0, tzinfo=UTC)),
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
        ("at local midnight", datetime(2026, 10, 31, 19, tzinfo=UTC))
    ]
