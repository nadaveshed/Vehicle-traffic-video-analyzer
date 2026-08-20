from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.models import Calibration, VideoMeta
from repository.models import Job, JobStatus


class JobRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, video_path: str, zone_id: int | None) -> Job:
        job = Job(video_path=video_path, zone_id=zone_id, status=JobStatus.PENDING)
        self.session.add(job)
        self.session.commit()
        self.session.refresh(job)
        return job

    def get(self, job_id: int) -> Job | None:
        return self.session.get(Job, job_id)

    def list(self, limit: int = 100) -> list[Job]:
        return list(self.session.scalars(select(Job).order_by(Job.id.desc()).limit(limit)))

    def set_status(self, job_id: int, status: JobStatus, message: str | None = None) -> None:
        job = self.get(job_id)
        if job is None:
            return
        job.status = status
        if message is not None:
            job.message = message
        if status in (JobStatus.COMPLETED, JobStatus.FAILED):
            job.finished_at = datetime.now(timezone.utc)
        self.session.commit()

    def set_progress(self, job_id: int, progress: float, message: str | None = None) -> None:
        job = self.get(job_id)
        if job is None:
            return
        job.progress = round(progress, 4)
        if message is not None:
            job.message = message
        self.session.commit()

    def set_video_meta(self, job_id: int, meta: VideoMeta) -> None:
        job = self.get(job_id)
        if job is None:
            return
        job.width, job.height = meta.width, meta.height
        job.fps, job.frame_count = meta.fps, meta.frame_count
        self.session.commit()

    def set_calibration(self, job_id: int, calib: Calibration) -> None:
        job = self.get(job_id)
        if job is None:
            return
        job.calibration = {
            "horizon_y": calib.horizon_y,
            "camera_height_m": calib.camera_height_m,
            "focal_px": calib.focal_px,
            "r_squared": calib.r_squared,
            "n_samples": calib.n_samples,
            "invert_from": calib.invert_from,
            "curve": {"width_px": calib.knots_w, "bottom_row_px": calib.knots_y},
        }
        self.session.commit()

    def set_validation(self, job_id: int, validation: dict) -> None:
        job = self.get(job_id)
        if job is None:
            return
        job.validation = validation
        self.session.commit()

    def set_artifacts(self, job_id: int, artifacts: dict) -> None:
        job = self.get(job_id)
        if job is None:
            return
        job.artifacts = {**(job.artifacts or {}), **artifacts}
        self.session.commit()
