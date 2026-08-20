from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from api.dependencies import EventRepoDep, JobRepoDep
from api.schemas import EventOut, SeriesOut

router = APIRouter(prefix="/events", tags=["events"])


def _load(events: EventRepoDep, event_id: int):
    row = events.get(event_id)
    if row is None:
        raise HTTPException(404, "event not found")
    return row


def _file(row, attr: str, media_type: str) -> FileResponse:
    path = getattr(row, attr, None)
    if not path or not Path(path).exists():
        raise HTTPException(404, f"{attr} is not available")
    return FileResponse(path, media_type=media_type)


@router.get("/{event_id}", response_model=EventOut)
def get_event(event_id: int, events: EventRepoDep):
    return EventOut.from_row(_load(events, event_id))


@router.get("/{event_id}/series", response_model=SeriesOut)
def get_series(event_id: int, events: EventRepoDep, jobs: JobRepoDep):
    """Per-frame depth estimates for this vehicle - the raw material of the plot."""
    row = _load(events, event_id)
    job = jobs.get(row.job_id)
    return SeriesOut(
        event_id=row.id,
        track_id=row.track_id,
        fps=(job.fps if job and job.fps else 0.0),
        series=row.series or {},
    )


@router.get("/{event_id}/snapshot")
def snapshot(event_id: int, events: EventRepoDep):
    """The captured frame: the moment this vehicle was closest to the camera."""
    return _file(_load(events, event_id), "snapshot_path", "image/jpeg")


@router.get("/{event_id}/crop")
def crop(event_id: int, events: EventRepoDep):
    return _file(_load(events, event_id), "crop_path", "image/jpeg")


@router.get("/{event_id}/plot")
def plot(event_id: int, events: EventRepoDep):
    return _file(_load(events, event_id), "plot_path", "image/png")


@router.get("/{event_id}/contact-sheet")
def contact_sheet(event_id: int, events: EventRepoDep):
    return _file(_load(events, event_id), "contact_sheet_path", "image/jpeg")
