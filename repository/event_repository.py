from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from core.geometry import format_timestamp
from core.models import ClosestApproach
from repository.models import CaptureEvent


class EventRepository:
    def __init__(self, session: Session):
        self.session = session

    def clear_for_job(self, job_id: int) -> None:
        self.session.execute(delete(CaptureEvent).where(CaptureEvent.job_id == job_id))
        self.session.commit()

    def add_many(self, job_id: int, approaches: list[ClosestApproach]) -> list[CaptureEvent]:
        rows = []
        for a in approaches:
            x1, y1, x2, y2 = a.box
            rows.append(
                CaptureEvent(
                    job_id=job_id,
                    track_id=a.track_id,
                    class_name=a.class_name,
                    vehicle_type=a.vehicle_type,
                    is_emergency=a.is_emergency,
                    emergency_score=a.emergency_score,
                    frame_index=a.frame_index,
                    timestamp=a.timestamp,
                    refined_timestamp=a.refined_timestamp,
                    timestamp_hms=format_timestamp(a.refined_timestamp),
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    center_x=a.center[0], center_y=a.center[1],
                    distance_m=a.distance_m,
                    confidence=a.confidence,
                    first_frame=a.first_frame,
                    last_frame=a.last_frame,
                    frames_tracked=a.frames_tracked,
                    entered_zone=a.entered_zone,
                    zone_entry_frame=a.zone_entry_frame,
                    zone_exit_frame=a.zone_exit_frame,
                    exits_view=a.exits_view,
                    truncated_start=a.truncated_start,
                    truncated_end=a.truncated_end,
                    agreement_frames=a.agreement_frames,
                    agreement_distance_pct=a.agreement_distance_pct,
                    naive_area_frame=a.naive_area_frame,
                    metric_picks=[{"name": p.name, "frame": p.frame_index, "value": p.value}
                                  for p in a.picks],
                    series=a.series,
                )
            )
        self.session.add_all(rows)
        self.session.commit()
        for r in rows:
            self.session.refresh(r)
        return rows

    def get(self, event_id: int) -> CaptureEvent | None:
        return self.session.get(CaptureEvent, event_id)

    def list_for_job(self, job_id: int) -> list[CaptureEvent]:
        return list(
            self.session.scalars(
                select(CaptureEvent)
                .where(CaptureEvent.job_id == job_id)
                .order_by(CaptureEvent.timestamp)
            )
        )

    def update_media(self, event_id: int, **paths: str | None) -> None:
        event = self.get(event_id)
        if event is None:
            return
        for key, value in paths.items():
            if value is not None:
                setattr(event, key, value)
        self.session.commit()
