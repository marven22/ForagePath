from __future__ import annotations

import json
import os
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import Boolean, Date, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


ROOT = Path(__file__).resolve().parents[1]


class Base(DeclarativeBase):
    pass


class FeedbackRecord(Base):
    __tablename__ = "farmer_feedback"
    feedback_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    submitted_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    release_id: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    state_code: Mapped[str] = mapped_column(String(2), nullable=False, index=True)
    cell_id: Mapped[str] = mapped_column(String(80), nullable=False)
    planting_class: Mapped[str] = mapped_column(String(80), nullable=False)
    selected_plants_json: Mapped[str] = mapped_column(Text, nullable=False)
    helpful_rating: Mapped[int] = mapped_column(Integer, nullable=False)
    intended_action: Mapped[str] = mapped_column(String(40), nullable=False)
    comments: Mapped[str | None] = mapped_column(Text)


class OutcomeRecord(Base):
    __tablename__ = "farmer_outcomes"
    outcome_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    feedback_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    submitted_at_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    observation_date: Mapped[date] = mapped_column(Date, nullable=False)
    plants_attempted_json: Mapped[str] = mapped_column(Text, nullable=False)
    establishment_fraction: Mapped[float | None] = mapped_column(Float)
    flowering_observed: Mapped[bool | None] = mapped_column(Boolean)
    pollinators_observed: Mapped[bool | None] = mapped_column(Boolean)
    observed_groups_json: Mapped[str] = mapped_column(Text, nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)
    verification_status: Mapped[str] = mapped_column(String(24), nullable=False, default="unreviewed")


def feedback_url() -> str:
    configured = os.getenv("FEEDBACK_DATABASE_URL")
    if configured:
        return configured.replace("postgresql://", "postgresql+psycopg://", 1)
    path = ROOT / "data" / "app" / "feedback.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


class FeedbackStore:
    def __init__(self, url: str | None = None) -> None:
        self.engine = create_engine(url or feedback_url(), pool_pre_ping=True)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def initialize(self) -> None:
        Base.metadata.create_all(self.engine)

    def save_feedback(self, payload: dict[str, Any]) -> str:
        identifier = str(uuid.uuid4())
        record = FeedbackRecord(
            feedback_id=identifier,
            submitted_at_utc=datetime.now(timezone.utc),
            release_id=payload["release_id"], state_code=payload["state_code"],
            cell_id=payload["cell_id"], planting_class=payload["planting_class"],
            selected_plants_json=json.dumps(payload["selected_plants"]),
            helpful_rating=payload["helpful_rating"], intended_action=payload["intended_action"],
            comments=payload.get("comments"),
        )
        with self.sessions.begin() as session:
            session.add(record)
        return identifier

    def save_outcome(self, payload: dict[str, Any]) -> str:
        identifier = str(uuid.uuid4())
        record = OutcomeRecord(
            outcome_id=identifier, feedback_id=payload["feedback_id"],
            submitted_at_utc=datetime.now(timezone.utc), observation_date=payload["observation_date"],
            plants_attempted_json=json.dumps(payload["plants_attempted"]),
            establishment_fraction=payload.get("establishment_fraction"),
            flowering_observed=payload.get("flowering_observed"),
            pollinators_observed=payload.get("pollinators_observed"),
            observed_groups_json=json.dumps(payload.get("observed_groups", [])),
            notes=payload.get("notes"), verification_status="unreviewed",
        )
        with self.sessions.begin() as session:
            if session.get(FeedbackRecord, payload["feedback_id"]) is None:
                raise LookupError("The feedback reference does not exist.")
            session.add(record)
        return identifier
