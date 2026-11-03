from datetime import date, datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    DateTime,
    Date,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint(
            "onboarding_status IN ('PENDING', 'ACTIVE')",
            name="ck_users_onboarding_status",
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    timezone: Mapped[str | None] = mapped_column(String(150), nullable=True)
    phone_region: Mapped[str | None] = mapped_column(String(2), nullable=True)
    onboarding_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="PENDING", server_default="PENDING"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )


class ChannelIdentity(Base):
    __tablename__ = "channel_identities"
    __table_args__ = (
        CheckConstraint("channel_type IN ('TELEGRAM', 'PHONE')", name="ck_identity_channel_type"),
        CheckConstraint("status IN ('PENDING', 'LINKED', 'UNLINKED')", name="ck_identity_status"),
        Index("ix_channel_identities_user_id", "user_id"),
        Index(
            "uq_channel_identities_active_external",
            "channel_type",
            "external_id",
            unique=True,
            postgresql_where=text("status = 'LINKED'"),
            sqlite_where=text("status = 'LINKED'"),
        ),
        Index(
            "uq_channel_identities_active_phone_user",
            "user_id",
            unique=True,
            postgresql_where=text("channel_type = 'PHONE' AND status = 'LINKED'"),
            sqlite_where=text("channel_type = 'PHONE' AND status = 'LINKED'"),
        ),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    channel_type: Mapped[str] = mapped_column(String(20), nullable=False)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="LINKED", server_default="LINKED"
    )
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    telegram_chat_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class Reminder(Base):
    __tablename__ = "reminders"
    __table_args__ = (
        CheckConstraint(
            "schedule_type IN ('ONE_TIME', 'RECURRING')", name="ck_reminders_schedule_type"
        ),
        CheckConstraint(
            "status IN ('ACTIVE', 'COMPLETED', 'CANCELED')", name="ck_reminders_status"
        ),
        CheckConstraint("version >= 1", name="ck_reminders_version"),
        CheckConstraint(
            "(schedule_type = 'RECURRING' AND recurrence_rule IS NOT NULL) "
            "OR (schedule_type = 'ONE_TIME' AND recurrence_rule IS NULL)",
            name="ck_reminders_recurrence_rule",
        ),
        Index("ix_reminders_user_status", "user_id", "status"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    task: Mapped[str] = mapped_column(Text, nullable=False)
    schedule_type: Mapped[str] = mapped_column(String(20), nullable=False)
    start_local: Mapped[str] = mapped_column(String(64), nullable=False)
    timezone: Mapped[str] = mapped_column(String(150), nullable=False)
    recurrence_rule: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="ACTIVE", server_default="ACTIVE"
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )


class ReminderOccurrence(Base):
    __tablename__ = "reminder_occurrences"
    __table_args__ = (
        CheckConstraint(
            "status IN ('PENDING', 'NOTIFIED', 'COMPLETED', 'CANCELED')",
            name="ck_occurrences_status",
        ),
        UniqueConstraint("reminder_id", "scheduled_at", name="uq_occurrence_reminder_scheduled"),
        UniqueConstraint(
            "reminder_id",
            "recurrence_instance_key",
            name="uq_occurrence_reminder_recurrence_key",
        ),
        Index("ix_occurrences_due_status", "next_due_at", "status"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    reminder_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("reminders.id", ondelete="CASCADE"), nullable=False
    )
    scheduled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    next_due_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    recurrence_instance_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="PENDING", server_default="PENDING"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )


class ReminderRecurrenceException(Base):
    """A durable, local-calendar skip for one recurring instance."""

    __tablename__ = "reminder_recurrence_exceptions"
    __table_args__ = (
        UniqueConstraint(
            "reminder_id", "occurrence_date", name="uq_recurrence_exception_reminder_date"
        ),
        Index("ix_recurrence_exceptions_reminder_date", "reminder_id", "occurrence_date"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    reminder_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("reminders.id", ondelete="CASCADE"), nullable=False
    )
    occurrence_date: Mapped[date] = mapped_column(Date, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class Notification(Base):
    __tablename__ = "notifications"
    __table_args__ = (
        CheckConstraint(
            "status IN ('PENDING', 'SENDING', 'SENT', 'FAILED', 'UNKNOWN', 'CANCELED')",
            name="ck_notifications_status",
        ),
        CheckConstraint("attempt_count >= 0", name="ck_notifications_attempt_count"),
        CheckConstraint("sequence >= 1", name="ck_notifications_sequence"),
        UniqueConstraint("occurrence_id", "sequence", name="uq_notification_occurrence_sequence"),
        Index("ix_notifications_due_status", "due_at", "status"),
        Index("ix_notifications_claimable", "status", "due_at", "claimed_until"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    occurrence_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("reminder_occurrences.id", ondelete="CASCADE"),
        nullable=False,
    )
    destination_identity_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("channel_identities.id", ondelete="RESTRICT"),
        nullable=False,
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    due_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="PENDING", server_default="PENDING"
    )
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    provider_message_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    claimed_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    claimed_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error_kind: Mapped[str | None] = mapped_column(String(80), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    attempt_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )


class ConversationSession(Base):
    __tablename__ = "conversation_sessions"
    __table_args__ = (
        CheckConstraint(
            "channel_type IN ('TELEGRAM', 'VOICE_CALL')",
            name="ck_conversation_sessions_channel_type",
        ),
        CheckConstraint(
            "status IN ('ACTIVE', 'AWAITING_CLARIFICATION', 'ENDED')",
            name="ck_conversation_sessions_status",
        ),
        UniqueConstraint(
            "user_id",
            "channel_type",
            "external_session_id",
            name="uq_conversation_sessions_external",
        ),
        Index("ix_conversation_sessions_user_status", "user_id", "status"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    channel_type: Mapped[str] = mapped_column(String(20), nullable=False)
    external_session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    active_reminder_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("reminders.id", ondelete="SET NULL"), nullable=True
    )
    pending_clarification: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(
        String(30), nullable=False, default="ACTIVE", server_default="ACTIVE"
    )
    version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default="1"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )


class IdempotencyRecord(Base):
    __tablename__ = "idempotency_records"
    __table_args__ = (
        CheckConstraint("source IN ('TELEGRAM', 'VAPI')", name="ck_idempotency_source"),
        CheckConstraint(
            "status IN ('PROCESSING', 'SUCCEEDED', 'FAILED')",
            name="ck_idempotency_status",
        ),
        CheckConstraint(
            "delivery_status IN ('PENDING', 'ATTEMPTED', 'SENT', 'FAILED', 'UNKNOWN')",
            name="ck_idempotency_delivery_status",
        ),
        Index("ix_idempotency_user_id", "user_id"),
        Index(
            "ix_idempotency_delivery_retry",
            "source",
            "delivery_status",
            "delivery_next_attempt_at",
        ),
        UniqueConstraint("source", "idempotency_key", name="uq_idempotency_source_key"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    source: Mapped[str] = mapped_column(String(20), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    user_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    operation: Mapped[str] = mapped_column(String(50), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="SUCCEEDED", server_default="SUCCEEDED"
    )
    delivery_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="PENDING", server_default="PENDING"
    )
    delivery_attempted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    delivery_attempt_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    delivery_next_attempt_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    delivery_claimed_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    delivery_claimed_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    delivery_last_error_kind: Mapped[str | None] = mapped_column(String(80), nullable=True)
    response: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
