"""SQLAlchemy models for the shared user database (mecdm_superapp_user_db)."""

from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Text,
    func,
)

# Note: ForeignKey("user.id") is NOT declared in the ORM models because the
# BetterAuth "user" table has no SQLAlchemy model in this codebase.
# The FK constraints exist in the actual database (created by migration SQL).
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class ChatSession(Base):
    __tablename__ = "chat_session"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False, default="New Chat")
    adk_session_id: Mapped[str | None] = mapped_column(Text, unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        server_default=func.now(),
    )


class ChatMessage(Base):
    __tablename__ = "chat_message"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    chat_session_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("chat_session.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    thinking: Mapped[str | None] = mapped_column(Text)
    tool_calls: Mapped[dict | None] = mapped_column(JSONB)
    activity_events: Mapped[dict | None] = mapped_column(JSONB)
    attachments: Mapped[list | None] = mapped_column(JSONB)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name="ck_role"),
    )


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    user_id: Mapped[str] = mapped_column(Text, nullable=False)
    chat_session_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("chat_session.id", ondelete="SET NULL")
    )
    message_id: Mapped[str | None] = mapped_column(Text)
    score: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint("score BETWEEN 1 AND 5", name="ck_score_range"),
    )


class UserPreferences(Base):
    __tablename__ = "user_preferences"

    user_id: Mapped[str] = mapped_column(Text, primary_key=True)
    chat_position: Mapped[str] = mapped_column(
        Text, nullable=False, default="dock-right"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        server_default=func.now(),
    )


# ── Dashboard config tables ──────────────────────────────────────────────────


class DashboardStat(Base):
    __tablename__ = "dashboard_stat"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str] = mapped_column(Text, nullable=False)
    query: Mapped[dict] = mapped_column(JSONB, nullable=False)
    chart: Mapped[dict] = mapped_column(JSONB, nullable=False)
    refresh_interval: Mapped[int | None] = mapped_column(Integer)
    is_system: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_by: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        server_default=func.now(),
    )


class GeographyDashboardConfig(Base):
    __tablename__ = "geography_dashboard_config"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    geo_level: Mapped[str] = mapped_column(Text, nullable=False)
    district_code_lgd: Mapped[int | None] = mapped_column(BigInteger)
    block_code_lgd: Mapped[int | None] = mapped_column(BigInteger)
    stat_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    is_override: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    label: Mapped[str | None] = mapped_column(Text)
    created_by: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        server_default=func.now(),
    )


class UserDashboardConfig(Base):
    __tablename__ = "user_dashboard_config"

    user_id: Mapped[str] = mapped_column(Text, primary_key=True)
    stat_ids: Mapped[list[str] | None] = mapped_column(ARRAY(Text))
    hidden_stat_ids: Mapped[list[str]] = mapped_column(
        ARRAY(Text), default=list
    )
    extra_stat_ids: Mapped[list[str]] = mapped_column(
        ARRAY(Text), default=list
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=_utcnow,
        onupdate=_utcnow,
        server_default=func.now(),
    )


# ── Query audit log ──────────────────────────────────────────────────────────


class QueryAuditLog(Base):
    __tablename__ = "query_audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_params: Mapped[dict | None] = mapped_column(JSONB)
    source: Mapped[str] = mapped_column(Text, nullable=False)
    user_id: Mapped[str | None] = mapped_column(Text)
    stat_id: Mapped[str | None] = mapped_column(Text)
    original_question: Mapped[str | None] = mapped_column(Text)
    rows_returned: Mapped[int | None] = mapped_column(Integer)
    execution_ms: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )


# ── Golden SQL examples ──────────────────────────────────────────────────────


class GoldenSql(Base):
    __tablename__ = "golden_sql"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    sql_query: Mapped[str] = mapped_column(Text, nullable=False)
    explanation: Mapped[str | None] = mapped_column(Text)
    tables_used: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_by: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
