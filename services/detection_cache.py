"""Persist the raw detections next to the job output.

Detection and tracking dominate the runtime; the analysis that follows is
milliseconds. Caching the tracker output means the depth model, the zone or the
smoothing can be re-tuned and re-run against identical detections - which also
makes any change in the results attributable to the change in the analysis
rather than to a different pass of the detector.
"""
from __future__ import annotations

import json
from pathlib import Path

from core.geometry import ClipFlags
from core.models import Detection, VideoMeta


def save_detections(path: Path, meta: VideoMeta, detections: list[Detection]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video": {
            "path": meta.path, "width": meta.width, "height": meta.height,
            "fps": meta.fps, "frame_count": meta.frame_count,
        },
        "detections": [
            [d.frame_index, d.track_id, d.class_id, d.class_name, round(d.confidence, 4),
             round(d.box[0], 2), round(d.box[1], 2), round(d.box[2], 2), round(d.box[3], 2),
             int(d.clip.left), int(d.clip.right), int(d.clip.top), int(d.clip.bottom)]
            for d in detections
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_detections(path: Path) -> tuple[VideoMeta, list[Detection]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    v = payload["video"]
    meta = VideoMeta(v["path"], v["width"], v["height"], v["fps"], v["frame_count"])
    detections = [
        Detection(
            frame_index=row[0],
            timestamp=row[0] / meta.fps,
            track_id=row[1],
            class_id=row[2],
            class_name=row[3],
            confidence=row[4],
            box=(row[5], row[6], row[7], row[8]),
            clip=ClipFlags(bool(row[9]), bool(row[10]), bool(row[11]), bool(row[12])),
        )
        for row in payload["detections"]
    ]
    return meta, detections
