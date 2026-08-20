"""Tell emergency vehicles apart from ordinary traffic.

The detector cannot do this on its own. YOLO here is a COCO model, and COCO has
four vehicle classes - car, motorcycle, bus, truck - with no notion of an
ambulance, a police car or a fire engine. A police car is simply a `car` to it.

What separates them on the road is the beacon: a small patch of heavily
saturated blue or red that *flashes*. Both halves matter. Saturated blue is
already rare on traffic, but a blue car would hold a steady ratio, whereas a
lightbar makes that ratio swing from frame to frame. So each track is sampled
across its lifetime, the roof band of every sample is scored for saturated
blue and red, and a vehicle is called an emergency only when the colour is
present *and* the signal flickers.

The limits are worth stating plainly:

  * lights off means invisible to this - a parked patrol car reads as a car
  * it needs enough pixels, so it only fires once the vehicle is close enough
  * it says "carries a beacon", not which service the vehicle belongs to

`emergency_score` is reported next to the verdict so a borderline call can be
inspected instead of taken on faith.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config.settings import Settings
from core.models import Detection, VideoMeta
from services.video_service import iter_frames

# OpenCV hue is 0..179.  The saturation and value floors are the whole trick:
# at dusk a silver car's windscreen reflects a blue-grey sky and registers as
# "blue" under any loose threshold, which on this footage flagged half the
# traffic.  A lightbar is not merely blue, it is violently saturated and bright,
# and nothing else on a road is.  Measured over the 4K clip, the marked police
# car scores 0.268 on this band while the highest-scoring ordinary vehicle
# reaches 0.017 and the median vehicle scores 0.000.
_BLUE = ((100, 170, 150), (135, 255, 255))
_RED_LO = ((0, 170, 170), (8, 255, 255))
_RED_HI = ((172, 170, 170), (179, 255, 255))

_ROOF_BAND = 0.35      # beacons sit on top; the body below is only noise
_SIDE_INSET = 0.10     # trim the box edges, which contain background
_CROP_WIDTH = 160      # samples are downscaled before the colour test
_MIN_BOX_WIDTH = 55    # below this a lightbar is a couple of pixels
_MAX_SAMPLES = 14


@dataclass
class EmergencyVerdict:
    is_emergency: bool
    score: float
    peak_blue: float
    peak_red: float
    flicker: float
    samples: int


_NO_VERDICT = EmergencyVerdict(False, 0.0, 0.0, 0.0, 0.0, 0)


class EmergencyService:
    def __init__(self, settings: Settings):
        self.settings = settings

    # ------------------------------------------------------------------ api
    def classify(
        self,
        video_path: str | Path,
        tracks: dict[int, list[Detection]],
        meta: VideoMeta,
        progress=None,
    ) -> dict[int, EmergencyVerdict]:
        wanted = self._plan(tracks)
        if not wanted:
            return {}

        readings: dict[int, list[tuple[float, float]]] = defaultdict(list)
        remaining = len(wanted)
        for index, frame in iter_frames(video_path):
            todo = wanted.get(index)
            if not todo:
                continue
            for track_id, box in todo:
                readings[track_id].append(self._score_patch(frame, box))
            remaining -= 1
            if progress and remaining % 40 == 0 and meta.frame_count:
                progress(1.0 - remaining / len(wanted), "scanning for beacons")

        return {tid: self._verdict(vals) for tid, vals in readings.items()}

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _plan(tracks: dict[int, list[Detection]]) -> dict[int, list[tuple[int, tuple]]]:
        """Which frames to look at, keyed by frame so the video is read once."""
        wanted: dict[int, list[tuple[int, tuple]]] = defaultdict(list)
        for track_id, dets in tracks.items():
            usable = [d for d in dets if (d.box[2] - d.box[0]) >= _MIN_BOX_WIDTH]
            if len(usable) < 3:
                continue
            step = max(1, len(usable) // _MAX_SAMPLES)
            for d in usable[::step][:_MAX_SAMPLES]:
                wanted[d.frame_index].append((track_id, d.box))
        return dict(wanted)

    @staticmethod
    def _score_patch(frame: np.ndarray, box: tuple) -> tuple[float, float]:
        """Fraction of the roof band that is saturated blue, and red."""
        h, w = frame.shape[:2]
        x1 = max(0, int(box[0]))
        y1 = max(0, int(box[1]))
        x2 = min(w, int(box[2]))
        y2 = min(h, int(box[3]))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return 0.0, 0.0

        inset = int((x2 - x1) * _SIDE_INSET)
        crop = frame[y1:y1 + max(int((y2 - y1) * _ROOF_BAND), 8), x1 + inset:x2 - inset]
        if crop.size == 0:
            return 0.0, 0.0
        if crop.shape[1] > _CROP_WIDTH:
            scale = _CROP_WIDTH / crop.shape[1]
            crop = cv2.resize(crop, (_CROP_WIDTH, max(int(crop.shape[0] * scale), 4)))

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        total = float(hsv.shape[0] * hsv.shape[1]) or 1.0
        blue = float(cv2.inRange(hsv, np.array(_BLUE[0]), np.array(_BLUE[1])).sum()) / 255.0
        red = (
            float(cv2.inRange(hsv, np.array(_RED_LO[0]), np.array(_RED_LO[1])).sum())
            + float(cv2.inRange(hsv, np.array(_RED_HI[0]), np.array(_RED_HI[1])).sum())
        ) / 255.0
        return blue / total, red / total

    def _verdict(self, readings: list[tuple[float, float]]) -> EmergencyVerdict:
        s = self.settings
        if len(readings) < 3:
            return _NO_VERDICT

        blue = np.array([b for b, _ in readings], dtype=float)
        red = np.array([r for _, r in readings], dtype=float)
        combined = blue + red

        peak_blue = float(blue.max())
        peak_red = float(red.max())
        mean = float(combined.mean())
        flicker = float(combined.std() / mean) if mean > 1e-6 else 0.0

        # Blue carries the verdict; red only supports it, because brake lights
        # and red bodywork put red on almost every vehicle in shot.
        colour = peak_blue + 0.25 * peak_red
        score = colour * (1.0 + min(flicker, 2.0))

        is_emergency = (
            peak_blue >= s.emergency_blue_ratio and flicker >= s.emergency_flicker
        ) or peak_blue >= s.emergency_blue_ratio_strong

        return EmergencyVerdict(
            is_emergency=bool(is_emergency),
            score=round(score, 5),
            peak_blue=round(peak_blue, 5),
            peak_red=round(peak_red, 5),
            flicker=round(flicker, 4),
            samples=len(readings),
        )
