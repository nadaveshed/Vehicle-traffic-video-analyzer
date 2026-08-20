"""Persistence model.  SQLite by default - swap the URL for Postgres and
nothing else changes."""
from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import (JSON, Boolean, DateTime, Enum, Float, ForeignKey,
                        Integer, String, Text)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from repository.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Zone(Base):
    """The virtual rectangle.  Stored normalised so one zone definition works
    for any resolution of the same camera view."""

    __tablename__ = "zones"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    x: Mapped[float] = mapped_column(Float)
    y: Mapped[float] = mapped_column(Float)
    w: Mapped[float] = mapped_column(Float)
    h: Mapped[float] = mapped_column(Float)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    jobs: Mapped[list["Job"]] = relationship(back_populates="zone")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    video_path: Mapped[str] = mapped_column(String(512))
    zone_id: Mapped[int | None] = mapped_column(ForeignKey("zones.id"), nullable=True)
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.PENDING)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)

    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    frame_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    calibration: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    validation: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    artifacts: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    zone: Mapped[Zone | None] = relationship(back_populates="jobs")
    events: Mapped[list["CaptureEvent"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )


class CaptureEvent(Base):
    """One vehicle's closest-approach moment."""

    __tablename__ = "capture_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"), index=True)
    track_id: Mapped[int] = mapped_column(Integer, index=True)
    class_name: Mapped[str] = mapped_column(String(40))
    vehicle_type: Mapped[str] = mapped_column(String(40), default="car", index=True)
    is_emergency: Mapped[bool] = mapped_column(Boolean, default=False)
    emergency_score: Mapped[float] = mapped_column(Float, default=0.0)

    frame_index: Mapped[int] = mapped_column(Integer)
    timestamp: Mapped[float] = mapped_column(Float)
    refined_timestamp: Mapped[float] = mapped_column(Float)
    timestamp_hms: Mapped[str] = mapped_column(String(20))

    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)
    center_x: Mapped[float] = mapped_column(Float)
    center_y: Mapped[float] = mapped_column(Float)

    distance_m: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)

    first_frame: Mapped[int] = mapped_column(Integer)
    last_frame: Mapped[int] = mapped_column(Integer)
    frames_tracked: Mapped[int] = mapped_column(Integer)

    entered_zone: Mapped[bool] = mapped_column(Boolean, default=False)
    zone_entry_frame: Mapped[int | None] = mapped_column(Integer, nullable=True)
    zone_exit_frame: Mapped[int | None] = mapped_column(Integer, nullable=True)
    exits_view: Mapped[bool] = mapped_column(Boolean, default=False)
    truncated_start: Mapped[bool] = mapped_column(Boolean, default=False)
    truncated_end: Mapped[bool] = mapped_column(Boolean, default=False)

    agreement_frames: Mapped[int] = mapped_column(Integer, default=0)
    agreement_distance_pct: Mapped[float] = mapped_column(Float, default=0.0)
    naive_area_frame: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metric_picks: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    series: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    snapshot_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    crop_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    plot_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    contact_sheet_path: Mapped[str | None] = mapped_column(String(512), nullable=True)

    job: Mapped[Job] = relationship(back_populates="events")
