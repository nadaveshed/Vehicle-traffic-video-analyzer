"""Detection + multi-object tracking.

YOLO11 supplies the detections, ByteTrack (bundled with ultralytics) keeps a
stable id on each vehicle across frames.  The service's only job is to turn
that stream into plain `Detection` records - everything downstream is pure
numpy and has no idea a neural network was involved.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

from config.settings import Settings
from core.geometry import clip_flags
from core.models import Detection, VideoMeta
from services.video_service import probe

ProgressCb = Callable[[float, str], None]


def _select_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class TrackingService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.settings.model_name)
        return self._model

    def run(
        self,
        video_path: str | Path,
        progress: ProgressCb | None = None,
    ) -> tuple[VideoMeta, list[Detection]]:
        s = self.settings
        meta = probe(video_path)
        device = _select_device(s.device)
        if progress:
            progress(0.0, f"loading {s.model_name} on device={device}")

        detections: list[Detection] = []
        stream = self.model.track(
            source=str(video_path),
            stream=True,
            persist=True,
            tracker=s.tracker_config,
            classes=s.vehicle_classes,
            conf=s.conf_threshold,
            iou=s.iou_threshold,
            imgsz=s.imgsz,
            device=device,
            verbose=False,
        )

        for frame_index, result in enumerate(stream):
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                names = result.names
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int)
                cls = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                for box, tid, cid, cf in zip(xyxy, ids, cls, confs):
                    b = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
                    detections.append(
                        Detection(
                            frame_index=frame_index,
                            timestamp=frame_index / meta.fps,
                            track_id=int(tid),
                            class_id=int(cid),
                            class_name=str(names.get(int(cid), int(cid))),
                            confidence=float(cf),
                            box=b,
                            clip=clip_flags(b, meta.width, meta.height, s.clip_margin_px),
                        )
                    )
            if progress and meta.frame_count and frame_index % 25 == 0:
                progress(
                    min(frame_index / meta.frame_count, 1.0),
                    f"tracking frame {frame_index}/{meta.frame_count}",
                )

        if progress:
            progress(1.0, f"tracked {len({d.track_id for d in detections})} vehicles")
        return meta, detections


def group_by_track(detections: list[Detection]) -> dict[int, list[Detection]]:
    tracks: dict[int, list[Detection]] = {}
    for d in detections:
        tracks.setdefault(d.track_id, []).append(d)
    for dets in tracks.values():
        dets.sort(key=lambda d: d.frame_index)
    return tracks
