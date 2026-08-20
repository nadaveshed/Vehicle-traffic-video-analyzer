from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from fastapi.responses import FileResponse, HTMLResponse

from api.dependencies import (EventRepoDep, JobRepoDep, SettingsDep,
                              ZoneRepoDep)
from api.schemas import EventOut, JobCreate, JobOut
from services.pipeline_service import PipelineService

router = APIRouter(prefix="/jobs", tags=["jobs"])

PLOT_KEYS = {
    "calibration": "calibration_plot",
    "agreement": "agreement_plot",
    "timeline": "timeline_plot",
}


def _run_pipeline(job_id: int) -> None:
    PipelineService().run(job_id)


@router.post("", response_model=JobOut, status_code=status.HTTP_202_ACCEPTED)
def create_job(
    payload: JobCreate,
    background: BackgroundTasks,
    jobs: JobRepoDep,
    zones: ZoneRepoDep,
    settings: SettingsDep,
):
    """Submit a clip. Processing runs in the background; poll the job for progress."""
    video = settings.resolve(payload.video_path)
    if not video.exists():
        raise HTTPException(404, f"video not found: {video}")

    zone = zones.get(payload.zone_id) if payload.zone_id else zones.ensure_default()
    if zone is None:
        raise HTTPException(404, "zone not found")

    job = jobs.create(payload.video_path, zone.id)
    background.add_task(_run_pipeline, job.id)
    return job


@router.get("", response_model=list[JobOut])
def list_jobs(jobs: JobRepoDep):
    return jobs.list()


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: int, jobs: JobRepoDep):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


@router.get("/{job_id}/events", response_model=list[EventOut])
def job_events(job_id: int, jobs: JobRepoDep, events: EventRepoDep):
    if jobs.get(job_id) is None:
        raise HTTPException(404, "job not found")
    return [EventOut.from_row(r) for r in events.list_for_job(job_id)]


@router.get("/{job_id}/validation")
def job_validation(job_id: int, jobs: JobRepoDep):
    """The numbers that show the picked frames are the right ones."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return {"job_id": job_id, "calibration": job.calibration, "validation": job.validation}


def _require_job(jobs: JobRepoDep, job_id: int):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


def _artifact(job, key: str) -> Path:
    path = (job.artifacts or {}).get(key)
    if not path or not Path(path).exists():
        raise HTTPException(404, f"artifact {key} is not available for this job")
    return Path(path)


@router.get("/{job_id}/report", response_class=HTMLResponse)
def job_report(job_id: int, jobs: JobRepoDep):
    job = _require_job(jobs, job_id)
    return HTMLResponse(_artifact(job, "report").read_text(encoding="utf-8"))


@router.get("/{job_id}/video")
def job_video(job_id: int, jobs: JobRepoDep):
    job = _require_job(jobs, job_id)
    return FileResponse(_artifact(job, "annotated_video"), media_type="video/mp4")


@router.get("/{job_id}/results.csv")
def job_csv(job_id: int, jobs: JobRepoDep):
    job = _require_job(jobs, job_id)
    return FileResponse(
        _artifact(job, "results_csv"),
        media_type="text/csv",
        filename=f"job_{job_id}_results.csv",
    )


@router.get("/{job_id}/plots/{name}")
def job_plot(job_id: int, name: str, jobs: JobRepoDep):
    job = _require_job(jobs, job_id)
    key = PLOT_KEYS.get(name)
    if key is None:
        raise HTTPException(404, "unknown plot; try calibration | agreement | timeline")
    return FileResponse(_artifact(job, key), media_type="image/png")
