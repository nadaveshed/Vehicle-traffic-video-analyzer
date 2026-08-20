"""Request / response contracts for the REST layer."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from repository.models import JobStatus


# --------------------------------------------------------------------- zones
class ZoneCreate(BaseModel):
    """The virtual rectangle, in normalised frame coordinates (0..1)."""

    name: str = Field(..., max_length=120, examples=["overpass-near-lane"])
    x: float = Field(..., ge=0.0, le=1.0, examples=[0.15])
    y: float = Field(..., ge=0.0, le=1.0, examples=[0.55])
    w: float = Field(..., gt=0.0, le=1.0, examples=[0.85])
    h: float = Field(..., gt=0.0, le=1.0, examples=[0.45])
    description: str | None = None


class ZoneOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    x: float
    y: float
    w: float
    h: float
    description: str | None
    created_at: datetime


# ---------------------------------------------------------------------- jobs
class JobCreate(BaseModel):
    video_path: str = Field(
        ..., description="Path to the clip, absolute or relative to the project root.",
        examples=["data/highway.mp4"],
    )
    zone_id: int | None = Field(
        None, description="Capture zone to use. Omitted -> the default zone from config."
    )


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    video_path: str
    zone_id: int | None
    status: JobStatus
    progress: float
    message: str | None
    width: int | None
    height: int | None
    fps: float | None
    frame_count: int | None
    calibration: dict | None
    validation: dict | None
    artifacts: dict | None
    created_at: datetime
    finished_at: datetime | None


# -------------------------------------------------------------------- events
class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float


class EventOut(BaseModel):
    """One vehicle's closest-approach moment - the deliverable of the system."""

    id: int
    job_id: int
    track_id: int
    class_name: str
    vehicle_type: str
    is_emergency: bool
    emergency_score: float

    frame_index: int
    timestamp_sec: float
    timestamp_refined_sec: float
    timestamp_hms: str

    bbox: BoundingBox
    center: tuple[float, float]

    distance_m: float
    confidence: float

    first_frame: int
    last_frame: int
    frames_tracked: int
    entered_zone: bool
    zone_entry_frame: int | None
    zone_exit_frame: int | None
    exits_view: bool
    truncated_start: bool
    truncated_end: bool

    agreement_frames: int
    agreement_distance_pct: float
    naive_area_frame: int | None
    metric_picks: list[dict] | None

    media: dict[str, str | None]

    @classmethod
    def from_row(cls, row) -> "EventOut":
        return cls(
            id=row.id,
            job_id=row.job_id,
            track_id=row.track_id,
            class_name=row.class_name,
            vehicle_type=row.vehicle_type,
            is_emergency=row.is_emergency,
            emergency_score=row.emergency_score,
            frame_index=row.frame_index,
            timestamp_sec=row.timestamp,
            timestamp_refined_sec=row.refined_timestamp,
            timestamp_hms=row.timestamp_hms,
            bbox=BoundingBox(
                x1=row.x1, y1=row.y1, x2=row.x2, y2=row.y2,
                width=row.x2 - row.x1, height=row.y2 - row.y1,
            ),
            center=(row.center_x, row.center_y),
            distance_m=row.distance_m,
            confidence=row.confidence,
            first_frame=row.first_frame,
            last_frame=row.last_frame,
            frames_tracked=row.frames_tracked,
            entered_zone=row.entered_zone,
            zone_entry_frame=row.zone_entry_frame,
            zone_exit_frame=row.zone_exit_frame,
            exits_view=row.exits_view,
            truncated_start=row.truncated_start,
            truncated_end=row.truncated_end,
            agreement_frames=row.agreement_frames,
            agreement_distance_pct=row.agreement_distance_pct,
            naive_area_frame=row.naive_area_frame,
            metric_picks=row.metric_picks,
            media={
                "snapshot": f"/api/v1/events/{row.id}/snapshot" if row.snapshot_path else None,
                "crop": f"/api/v1/events/{row.id}/crop" if row.crop_path else None,
                "plot": f"/api/v1/events/{row.id}/plot" if row.plot_path else None,
                "contact_sheet": (
                    f"/api/v1/events/{row.id}/contact-sheet" if row.contact_sheet_path else None
                ),
            },
        )


class SeriesOut(BaseModel):
    """The full per-frame depth curve behind one event."""

    event_id: int
    track_id: int
    fps: float
    series: dict
