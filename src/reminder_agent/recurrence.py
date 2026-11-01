"""Small, strict recurrence engine used by VS-07.

The challenge only needs weekly wall-clock schedules.  Keeping the accepted
rule vocabulary deliberately small makes model output safe to validate and
keeps recurrence calculations independent from provider integrations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
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

    def __post_init__(self) -> None:
        if not self.weekdays or any(day < 0 or day > 6 for day in self.weekdays):
            raise ValueError("weekly recurrence requires valid weekdays")
        if tuple(sorted(set(self.weekdays))) != self.weekdays:
            raise ValueError("weekly recurrence weekdays must be sorted and unique")
        if not 0 <= self.hour <= 23 or not 0 <= self.minute <= 59:
            raise ValueError("weekly recurrence has an invalid local time")

    @property
    def canonical_rule(self) -> str:
        days = ",".join(WEEKDAY_NAMES[day] for day in self.weekdays)
        return f"FREQ=WEEKLY;BYDAY={days};BYHOUR={self.hour:02d};BYMINUTE={self.minute:02d}"


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
    return WeeklyRecurrence(tuple(sorted({WEEKDAY_BY_NAME[day] for day in raw_days})), hour, minute)


def resolve_local_datetime(local_date: date, local_time: time, zone: ZoneInfo) -> datetime:
    """Resolve a wall-clock time with an explicit DST policy.

    Fall-back overlaps use the earlier instant (fold=0).  Spring-forward gaps
    shift forward by the DST gap, preserving the requested minutes.
    """

    naive = datetime.combine(local_date, local_time)
    candidates = [naive.replace(tzinfo=zone, fold=fold) for fold in (0, 1)]
    valid: list[datetime] = []
    for candidate in candidates:
        round_trip = candidate.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None)
        if round_trip == naive:
            valid.append(candidate)
    if valid:
        return min(valid, key=lambda value: value.astimezone(timezone.utc))

    # A nonexistent local time has two different offsets.  Moving by their
    # difference produces the corresponding wall-clock time after the gap.
    offsets = [candidate.utcoffset() for candidate in candidates]
    if offsets[0] is None or offsets[1] is None or offsets[1] <= offsets[0]:
        raise ValueError("could not resolve nonexistent local time")
    shifted = naive + (offsets[1] - offsets[0])
    return shifted.replace(tzinfo=zone, fold=0)


def iter_weekly_occurrences(
    recurrence: WeeklyRecurrence,
    *,
    start_local: datetime,
    now: datetime,
    timezone_name: str,
    horizon_days: int,
):
    """Yield future UTC/local recurrence instances through a rolling horizon."""

    zone = ZoneInfo(timezone_name)
    local_now = now.astimezone(zone)
    first_date = max(start_local.date(), local_now.date())
    end_date = local_now.date() + timedelta(days=horizon_days)
    current = first_date
    while current <= end_date:
        if current.weekday() in recurrence.weekdays:
            local_due = resolve_local_datetime(
                current, time(recurrence.hour, recurrence.minute), zone
            )
            due_utc = local_due.astimezone(timezone.utc)
            if due_utc > now:
                yield current, local_due, due_utc
        current += timedelta(days=1)
