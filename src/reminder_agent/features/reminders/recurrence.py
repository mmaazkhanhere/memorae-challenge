"""Small, strict recurrence engine for weekly, monthly, and yearly schedules."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

WEEKDAY_NAMES = (
    "MO",
    "TU",
    "WE",
    "TH",
    "FR",
    "SA",
    "SU",
)
WEEKDAY_BY_NAME = {name: index for index, name in enumerate(WEEKDAY_NAMES)}


@dataclass(frozen=True)
class WeeklyRecurrence:
    weekdays: tuple[int, ...]
    hour: int
    minute: int
    interval: int = 1

    def __post_init__(self) -> None:
        if not self.weekdays or any(day < 0 or day > 6 for day in self.weekdays):
            raise ValueError("weekly recurrence requires valid weekdays")
        if tuple(sorted(set(self.weekdays))) != self.weekdays:
            raise ValueError("weekly recurrence weekdays must be sorted and unique")
        if not 0 <= self.hour <= 23 or not 0 <= self.minute <= 59:
            raise ValueError("weekly recurrence has an invalid local time")
        if self.interval < 1:
            raise ValueError("weekly recurrence interval must be positive")

    @property
    def canonical_rule(self) -> str:
        days = ",".join(WEEKDAY_NAMES[day] for day in self.weekdays)
        interval = f";INTERVAL={self.interval}" if self.interval != 1 else ""
        return (
            f"FREQ=WEEKLY{interval};BYDAY={days};BYHOUR={self.hour:02d};BYMINUTE={self.minute:02d}"
        )


@dataclass(frozen=True)
class MonthlyRecurrence:
    day_of_month: int
    hour: int
    minute: int
    interval: int = 1

    def __post_init__(self) -> None:
        if not 1 <= self.day_of_month <= 31:
            raise ValueError("monthly recurrence has an invalid day")
        if not 0 <= self.hour <= 23 or not 0 <= self.minute <= 59:
            raise ValueError("monthly recurrence has an invalid local time")
        if self.interval < 1:
            raise ValueError("monthly recurrence interval must be positive")

    @property
    def canonical_rule(self) -> str:
        interval = f";INTERVAL={self.interval}" if self.interval != 1 else ""
        return (
            f"FREQ=MONTHLY{interval};BYMONTHDAY={self.day_of_month:02d};"
            f"BYHOUR={self.hour:02d};BYMINUTE={self.minute:02d}"
        )


@dataclass(frozen=True)
class YearlyRecurrence:
    month: int
    day_of_month: int
    hour: int
    minute: int
    interval: int = 1

    def __post_init__(self) -> None:
        date(2000, self.month, self.day_of_month)
        if not 0 <= self.hour <= 23 or not 0 <= self.minute <= 59:
            raise ValueError("yearly recurrence has an invalid local time")
        if self.interval < 1:
            raise ValueError("yearly recurrence interval must be positive")

    @property
    def canonical_rule(self) -> str:
        interval = f";INTERVAL={self.interval}" if self.interval != 1 else ""
        return (
            f"FREQ=YEARLY{interval};BYMONTH={self.month:02d};BYMONTHDAY={self.day_of_month:02d};"
            f"BYHOUR={self.hour:02d};BYMINUTE={self.minute:02d}"
        )


def parse_weekly_rule(rule: str) -> WeeklyRecurrence:
    """Parse only the canonical weekly rule emitted by the application."""

    fields = {}
    for part in rule.split(";"):
        if "=" not in part:
            raise ValueError("invalid recurrence rule")
        key, value = part.split("=", 1)
        fields[key.upper()] = value.upper()
    if fields.get("FREQ") != "WEEKLY":
        raise ValueError("only weekly recurrence is supported")
    raw_days = fields.get("BYDAY", "").split(",")
    if not raw_days or any(day not in WEEKDAY_BY_NAME for day in raw_days):
        raise ValueError("weekly recurrence requires valid BYDAY values")
    try:
        hour = int(fields["BYHOUR"])
        minute = int(fields["BYMINUTE"])
    except (KeyError, ValueError) as exc:
        raise ValueError("weekly recurrence requires BYHOUR and BYMINUTE") from exc
    return WeeklyRecurrence(
        tuple(sorted({WEEKDAY_BY_NAME[day] for day in raw_days})),
        hour,
        minute,
        int(fields.get("INTERVAL", "1")),
    )


def _rule_fields(rule: str) -> dict[str, str]:
    fields = {}
    for part in rule.split(";"):
        if "=" not in part:
            raise ValueError("invalid recurrence rule")
        key, value = part.split("=", 1)
        fields[key.upper()] = value.upper()
    return fields


def parse_recurrence_rule(
    rule: str,
) -> WeeklyRecurrence | MonthlyRecurrence | YearlyRecurrence:
    fields = _rule_fields(rule)
    try:
        hour = int(fields["BYHOUR"])
        minute = int(fields["BYMINUTE"])
    except (KeyError, ValueError) as exc:
        raise ValueError("recurrence requires BYHOUR and BYMINUTE") from exc
    frequency = fields.get("FREQ")
    try:
        interval = int(fields.get("INTERVAL", "1"))
    except ValueError as exc:
        raise ValueError("recurrence interval must be an integer") from exc
    if frequency == "WEEKLY":
        raw_days = fields.get("BYDAY", "").split(",")
        if not raw_days or any(day not in WEEKDAY_BY_NAME for day in raw_days):
            raise ValueError("weekly recurrence requires valid BYDAY values")
        return WeeklyRecurrence(
            tuple(sorted({WEEKDAY_BY_NAME[day] for day in raw_days})), hour, minute, interval
        )
    if frequency == "MONTHLY":
        return MonthlyRecurrence(int(fields["BYMONTHDAY"]), hour, minute, interval)
    if frequency == "YEARLY":
        return YearlyRecurrence(
            int(fields["BYMONTH"]), int(fields["BYMONTHDAY"]), hour, minute, interval
        )
    raise ValueError("unsupported recurrence frequency")


def resolve_local_datetime(local_date: date, local_time: time, zone: ZoneInfo) -> datetime:
    """Resolve a wall-clock time with an explicit DST policy.

    Fall-back overlaps use the earlier instant (fold=0).  Spring-forward gaps
    shift forward by the DST gap, preserving the requested minutes.
    """

    naive = datetime.combine(local_date, local_time)
    candidates = [naive.replace(tzinfo=zone, fold=fold) for fold in (0, 1)]
    valid: list[datetime] = []
    for candidate in candidates:
        round_trip = candidate.astimezone(UTC).astimezone(zone).replace(tzinfo=None)
        if round_trip == naive:
            valid.append(candidate)
    if valid:
        return min(valid, key=lambda value: value.astimezone(UTC))

    # A nonexistent local time has two different offsets.  Moving by their
    # difference produces the corresponding wall-clock time after the gap.
    offsets = [candidate.utcoffset() for candidate in candidates]
    if offsets[0] is None or offsets[1] is None or offsets[1] <= offsets[0]:
        raise ValueError("could not resolve nonexistent local time")
    shifted = naive + (offsets[1] - offsets[0])
    return shifted.replace(tzinfo=zone, fold=0)


def recurrence_due_for_local_date(
    recurrence: WeeklyRecurrence | MonthlyRecurrence | YearlyRecurrence,
    *,
    start_local: datetime,
    occurrence_date: date,
    timezone_name: str,
) -> tuple[datetime, datetime] | None:
    """Return a recurrence's local/UTC due values for one valid local date."""

    if occurrence_date < start_local.date():
        return None
    if isinstance(recurrence, WeeklyRecurrence):
        weeks_since_start = (occurrence_date - start_local.date()).days // 7
        matches = (
            occurrence_date.weekday() in recurrence.weekdays
            and weeks_since_start % recurrence.interval == 0
        )
    elif isinstance(recurrence, MonthlyRecurrence):
        months_since_start = (
            (occurrence_date.year - start_local.year) * 12
            + occurrence_date.month
            - start_local.month
        )
        matches = (
            occurrence_date.day == recurrence.day_of_month
            and months_since_start % recurrence.interval == 0
        )
    else:
        years_since_start = occurrence_date.year - start_local.year
        matches = (
            occurrence_date.month == recurrence.month
            and occurrence_date.day == recurrence.day_of_month
            and years_since_start % recurrence.interval == 0
        )
    if not matches:
        return None
    local_due = resolve_local_datetime(
        occurrence_date, time(recurrence.hour, recurrence.minute), ZoneInfo(timezone_name)
    )
    return local_due, local_due.astimezone(UTC)


def iter_weekly_occurrences(
    recurrence: WeeklyRecurrence,
    *,
    start_local: datetime,
    now: datetime,
    timezone_name: str,
    horizon_days: int,
) -> Iterator[tuple[date, datetime, datetime]]:
    """Yield future UTC/local recurrence instances through a rolling horizon."""

    yield from iter_recurrence_occurrences(
        recurrence,
        start_local=start_local,
        now=now,
        timezone_name=timezone_name,
        horizon_days=horizon_days,
    )


def iter_recurrence_occurrences(
    recurrence: WeeklyRecurrence | MonthlyRecurrence | YearlyRecurrence,
    *,
    start_local: datetime,
    now: datetime,
    timezone_name: str,
    horizon_days: int,
) -> Iterator[tuple[date, datetime, datetime]]:
    zone = ZoneInfo(timezone_name)
    local_now = now.astimezone(zone)
    end_date = local_now.date() + timedelta(days=horizon_days)
    if isinstance(recurrence, WeeklyRecurrence):
        current = max(start_local.date(), local_now.date())
        while current <= end_date:
            weeks_since_start = (current - start_local.date()).days // 7
            if (
                current.weekday() in recurrence.weekdays
                and weeks_since_start % recurrence.interval == 0
            ):
                local_due = resolve_local_datetime(
                    current, time(recurrence.hour, recurrence.minute), zone
                )
                due_utc = local_due.astimezone(UTC)
                if due_utc > now:
                    yield current, local_due, due_utc
            current += timedelta(days=1)
        return

    current = max(start_local.date().replace(day=1), local_now.date().replace(day=1))
    while current <= end_date:
        if isinstance(recurrence, YearlyRecurrence) and current.month != recurrence.month:
            current = date(
                current.year + (1 if current.month >= recurrence.month else 0), recurrence.month, 1
            )
            if current <= local_now.date().replace(day=1):
                current = date(current.year + 1, recurrence.month, 1)
        day = recurrence.day_of_month
        months_since_start = (
            (current.year - start_local.year) * 12 + current.month - start_local.month
        )
        years_since_start = current.year - start_local.year
        matches_interval = (
            months_since_start % recurrence.interval == 0
            if isinstance(recurrence, MonthlyRecurrence)
            else years_since_start % recurrence.interval == 0
        )
        if matches_interval and current.month == getattr(recurrence, "month", current.month):
            try:
                candidate = date(current.year, current.month, day)
            except ValueError:
                candidate = None
            if candidate is not None and start_local.date() <= candidate <= end_date:
                local_due = resolve_local_datetime(
                    candidate, time(recurrence.hour, recurrence.minute), zone
                )
                due_utc = local_due.astimezone(UTC)
                if due_utc > now:
                    yield candidate, local_due, due_utc
        if isinstance(recurrence, MonthlyRecurrence):
            current = date(
                current.year + (1 if current.month == 12 else 0),
                1 if current.month == 12 else current.month + 1,
                1,
            )
        else:
            current = date(current.year + 1, 1, 1)
