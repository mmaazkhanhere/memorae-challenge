from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
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


class IdempotencyRecord(Base):
    __tablename__ = "idempotency_records"
    __table_args__ = (
        CheckConstraint("source IN ('TELEGRAM', 'VAPI')", name="ck_idempotency_source"),
        CheckConstraint(
            "status IN ('PROCESSING', 'SUCCEEDED', 'FAILED')",
            name="ck_idempotency_status",
        ),
        CheckConstraint(
            "delivery_status IN ('PENDING', 'ATTEMPTED', 'SENT', 'UNKNOWN')",
            name="ck_idempotency_delivery_status",
        ),
        Index("ix_idempotency_user_id", "user_id"),
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
    response: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
