"""Orchestration: video in, capture events + proof material out.

    track  ->  calibrate  ->  gate on the zone  ->  locate closest approach
           ->  persist  ->  render the visual/numeric aids

Nothing here knows about HTTP; the API and the CLI both call `run`.
"""
from __future__ import annotations

import csv
import json
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

from config.settings import Settings, get_settings
from core.geometry import NormalizedRect, format_timestamp
from core.models import ClosestApproach, VideoMeta
from repository.database import session_scope
from repository.event_repository import EventRepository
from repository.job_repository import JobRepository
from repository.models import JobStatus
from repository.zone_repository import ZoneRepository
from services.calibration_service import CalibrationService
from services.detection_cache import load_detections, save_detections
from services.emergency_service import EmergencyService
from services.plot_service import PlotService
from services.proximity_service import ProximityService
from services.tracking_service import TrackingService, group_by_track
from services.visualization_service import VisualizationService
from services.zone_service import ZoneService


class PipelineService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.tracking = TrackingService(self.settings)
        self.calibration = CalibrationService(self.settings)
        self.proximity = ProximityService(self.settings)
        self.visualization = VisualizationService(self.settings)
        self.emergency = EmergencyService(self.settings)

    # ------------------------------------------------------------------ run
    def run(self, job_id: int, detections_cache: Path | None = None) -> None:
        """Process a job.

        `detections_cache` re-uses the tracker output of an earlier run, so the
        zone, the depth model or the smoothing can be changed and the effect
        measured against byte-identical detections.
        """
        s = self.settings
        try:
            with session_scope() as session:
                jobs = JobRepository(session)
                job = jobs.get(job_id)
                if job is None:
                    return
                video_path = str(s.resolve(job.video_path))
                zone_row = ZoneRepository(session).get(job.zone_id) if job.zone_id else None
                rect = (
                    NormalizedRect(zone_row.x, zone_row.y, zone_row.w, zone_row.h)
                    if zone_row
                    else NormalizedRect(s.zone_x, s.zone_y, s.zone_w, s.zone_h)
                )
                jobs.set_status(job_id, JobStatus.RUNNING, "starting")

            out_dir = s.output_dir / f"job_{job_id:04d}"
            out_dir.mkdir(parents=True, exist_ok=True)

            # ---------------------------------------------- detect + track --
            def track_progress(p: float, msg: str) -> None:
                self._progress(job_id, 0.05 + 0.45 * p, msg)

            if detections_cache is not None:
                self._progress(job_id, 0.45, f"reusing detections from {detections_cache.name}")
                meta, detections = load_detections(detections_cache)
                meta.path = video_path
            else:
                meta, detections = self.tracking.run(video_path, track_progress)
            with session_scope() as session:
                JobRepository(session).set_video_meta(job_id, meta)
            if s.cache_detections:
                save_detections(out_dir / "detections.json", meta, detections)

            # -------------------------------------------------- calibrate --
            self._progress(job_id, 0.52, "calibrating camera geometry")
            zone = ZoneService(rect)
            tracks = group_by_track(detections)
            zone_tracks = {
                tid for tid, dets in tracks.items() if zone.visit(dets, meta).entered
            }
            calib = self.calibration.calibrate(meta, detections, zone_tracks)
            with session_scope() as session:
                JobRepository(session).set_calibration(job_id, calib)

            # ------------------------------------- closest approach per id --
            self._progress(job_id, 0.56, "locating closest approach per vehicle")
            approaches: list[ClosestApproach] = []
            for track in tracks.values():
                result = self.proximity.analyse(track, meta, calib, zone)
                if result is not None:
                    approaches.append(result)
            approaches.sort(key=lambda a: a.refined_timestamp)

            # ------------------------------------ emergency vehicle pass ---
            # COCO has no class for these, so they are picked out by their
            # beacon.  Runs after tracking so it only has to look at frames
            # where a tracked vehicle is actually big enough to read.
            if s.detect_emergency:
                self._progress(job_id, 0.59, "checking for emergency beacons")
                verdicts = self.emergency.classify(video_path, tracks, meta)
                for a in approaches:
                    verdict = verdicts.get(a.track_id)
                    if verdict is None:
                        continue
                    a.emergency_score = verdict.score
                    a.is_emergency = verdict.is_emergency
                    if verdict.is_emergency:
                        a.vehicle_type = "emergency"

            validation = self._validate(approaches, meta, calib)

            with session_scope() as session:
                events_repo = EventRepository(session)
                events_repo.clear_for_job(job_id)
                rows = events_repo.add_many(job_id, approaches)
                JobRepository(session).set_validation(job_id, validation)
                id_by_track = {r.track_id: r.id for r in rows}

            # ------------------------------------------------ proof material -
            self._progress(job_id, 0.62, "rendering snapshots and contact sheets")
            media = self.visualization.render_event_stills(
                video_path, approaches, meta, rect, out_dir
            )

            plots = PlotService(out_dir)
            for a in approaches:
                media.setdefault(a.track_id, {})["plot_path"] = str(
                    plots.event_plot(a, meta)
                )
            artifacts = {
                "calibration_plot": str(plots.calibration_plot(detections, calib, meta)),
                "agreement_plot": str(plots.agreement_plot(approaches, meta)),
                "timeline_plot": str(plots.timeline_plot(approaches, meta)),
            }

            with session_scope() as session:
                events_repo = EventRepository(session)
                for track_id, paths in media.items():
                    if track_id in id_by_track:
                        events_repo.update_media(id_by_track[track_id], **paths)

            artifacts |= self._export_tables(out_dir, approaches, meta, calib, validation, rect)

            if s.write_annotated_video:
                self._progress(job_id, 0.72, "rendering annotated video")

                def render_progress(p: float, msg: str) -> None:
                    self._progress(job_id, 0.72 + 0.22 * p, msg)

                video_out = self.visualization.render_video(
                    video_path, out_dir / "annotated.mp4", meta, detections,
                    approaches, rect, calib, render_progress,
                )
                artifacts["annotated_video"] = str(video_out)

            self._progress(job_id, 0.96, "writing report")
            from services.report_service import ReportService

            report = ReportService(out_dir).build(
                job_id, meta, calib, rect, approaches, validation, artifacts, media
            )
            artifacts["report"] = str(report)

            with session_scope() as session:
                jobs = JobRepository(session)
                jobs.set_artifacts(job_id, artifacts)
                jobs.set_progress(job_id, 1.0)
                jobs.set_status(
                    job_id, JobStatus.COMPLETED,
                    f"{len(approaches)} vehicles captured",
                )
        except Exception as exc:  # noqa: BLE001 - surface the failure on the job
            with session_scope() as session:
                JobRepository(session).set_status(
                    job_id, JobStatus.FAILED,
                    f"{exc}\n{traceback.format_exc(limit=6)}",
                )
            raise

    # -------------------------------------------------------------- helpers
    def _progress(self, job_id: int, value: float, message: str) -> None:
        with session_scope() as session:
            JobRepository(session).set_progress(job_id, value, message)

    @staticmethod
    def _validate(approaches: list[ClosestApproach], meta: VideoMeta, calib) -> dict:
        """The numbers that answer 'how do we know it found the right frame?'"""
        if not approaches:
            return {"vehicles": 0}

        def stats(items):
            spread = np.array([a.agreement_frames for a in items], dtype=float)
            if not len(spread):
                return {}
            return {
                "median": float(np.median(spread)),
                "mean": round(float(spread.mean()), 2),
                "p90": float(np.percentile(spread, 90)),
                "max": float(spread.max()),
                "within_1_frame_pct": round(float((spread <= 1).mean() * 100), 1),
                "within_3_frames_pct": round(float((spread <= 3).mean() * 100), 1),
            }

        def depth_stats(items):
            pct = np.array([a.agreement_distance_pct for a in items], dtype=float)
            if not len(pct):
                return {}
            return {
                "median_pct": round(float(np.median(pct)), 3),
                "mean_pct": round(float(pct.mean()), 3),
                "p90_pct": round(float(np.percentile(pct, 90)), 3),
                "max_pct": round(float(pct.max()), 3),
                "within_1_pct": round(float((pct <= 1.0).mean() * 100), 1),
                "within_5_pct": round(float((pct <= 5.0).mean() * 100), 1),
            }

        spread = np.array([a.agreement_frames for a in approaches], dtype=float)
        naive = np.array(
            [abs(a.naive_area_frame - a.frame_index) for a in approaches
             if a.naive_area_frame is not None], dtype=float
        )
        zone_only = [a for a in approaches if a.entered_zone]

        return {
            "vehicles": len(approaches),
            "vehicles_by_type": {
                kind: sum(1 for a in approaches if a.vehicle_type == kind)
                for kind in sorted({a.vehicle_type for a in approaches})
            },
            "emergency_vehicles": sum(1 for a in approaches if a.is_emergency),
            "vehicles_triggered_by_zone": len(zone_only),
            "vehicles_leaving_the_frame": sum(1 for a in approaches if a.exits_view),
            "vehicles_truncated_by_clip_bounds": sum(
                1 for a in approaches if a.truncated_start or a.truncated_end
            ),
            "cross_metric_spread_frames_zone_only": stats(zone_only),
            "cross_metric_depth_spread": depth_stats(approaches),
            "cross_metric_depth_spread_zone_only": depth_stats(zone_only),
            "cross_metric_spread_frames": {
                "median": float(np.median(spread)),
                "mean": round(float(spread.mean()), 2),
                "p90": float(np.percentile(spread, 90)),
                "max": float(spread.max()),
                "within_1_frame_pct": round(float((spread <= 1).mean() * 100), 1),
                "within_3_frames_pct": round(float((spread <= 3).mean() * 100), 1),
            },
            "cross_metric_spread_seconds": {
                "median": round(float(np.median(spread)) / meta.fps, 4),
                "p90": round(float(np.percentile(spread, 90)) / meta.fps, 4),
            },
            "naive_area_baseline_error_frames": {
                "median": float(np.median(naive)) if len(naive) else None,
                "mean": round(float(naive.mean()), 2) if len(naive) else None,
                "max": float(naive.max()) if len(naive) else None,
                "disagreed_pct": round(float((naive > 0).mean() * 100), 1) if len(naive) else None,
            },
            "calibration": {
                "horizon_y_px": round(calib.horizon_y, 1),
                "r_squared": round(calib.r_squared, 4),
                "camera_height_m": round(calib.camera_height_m, 2),
                "focal_px": round(calib.focal_px, 1),
                "n_samples": calib.n_samples,
            },
            "timing_precision_seconds": round(1.0 / meta.fps, 4),
        }

    @staticmethod
    def _export_tables(
        out_dir: Path,
        approaches: list[ClosestApproach],
        meta: VideoMeta,
        calib,
        validation: dict,
        rect: NormalizedRect,
    ) -> dict:
        csv_path = out_dir / "results.csv"
        json_path = out_dir / "results.json"

        fields = [
            "track_id", "vehicle_type", "class", "is_emergency", "emergency_score",
            "closest_frame", "timestamp_sec", "timestamp_refined_sec",
            "timestamp_hms", "x1", "y1", "x2", "y2", "center_x", "center_y",
            "box_width_px", "box_height_px", "distance_m", "confidence",
            "first_frame", "last_frame", "frames_tracked", "entered_zone",
            "zone_entry_frame", "zone_exit_frame", "exits_view",
            "truncated_start", "truncated_end",
            "pick_ground", "pick_width", "pick_diag", "agreement_frames",
            "agreement_distance_pct",
            "naive_area_frame", "naive_area_error_frames",
        ]
        rows = []
        for a in approaches:
            picks = {p.name: p.frame_index for p in a.picks}
            rows.append({
                "track_id": a.track_id,
                "vehicle_type": a.vehicle_type,
                "class": a.class_name,
                "is_emergency": a.is_emergency,
                "emergency_score": a.emergency_score,
                "closest_frame": a.frame_index,
                "timestamp_sec": round(a.timestamp, 4),
                "timestamp_refined_sec": round(a.refined_timestamp, 4),
                "timestamp_hms": format_timestamp(a.refined_timestamp),
                "x1": round(a.box[0], 1), "y1": round(a.box[1], 1),
                "x2": round(a.box[2], 1), "y2": round(a.box[3], 1),
                "center_x": round(a.center[0], 1), "center_y": round(a.center[1], 1),
                "box_width_px": round(a.box[2] - a.box[0], 1),
                "box_height_px": round(a.box[3] - a.box[1], 1),
                "distance_m": round(a.distance_m, 2),
                "confidence": round(a.confidence, 3),
                "first_frame": a.first_frame,
                "last_frame": a.last_frame,
                "frames_tracked": a.frames_tracked,
                "entered_zone": a.entered_zone,
                "zone_entry_frame": a.zone_entry_frame,
                "zone_exit_frame": a.zone_exit_frame,
                "exits_view": a.exits_view,
                "truncated_start": a.truncated_start,
                "truncated_end": a.truncated_end,
                "pick_ground": picks.get("ground"),
                "pick_width": picks.get("width"),
                "pick_diag": picks.get("diag"),
                "agreement_frames": a.agreement_frames,
                "agreement_distance_pct": a.agreement_distance_pct,
                "naive_area_frame": a.naive_area_frame,
                "naive_area_error_frames": (
                    a.naive_area_frame - a.frame_index if a.naive_area_frame is not None else None
                ),
            })

        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        json_path.write_text(
            json.dumps(
                {
                    "video": asdict(meta),
                    "capture_zone_normalized": {"x": rect.x, "y": rect.y, "w": rect.w, "h": rect.h},
                    "calibration": asdict(calib),
                    "validation": validation,
                    "vehicles": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"results_csv": str(csv_path), "results_json": str(json_path)}
