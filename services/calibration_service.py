"""Recover the camera model from the detections themselves.

A vehicle's apparent width `w_px` and the image row of its wheels `y_bottom`
are both governed by how far away it is, so the two are functionally related
and that relation is everything the depth estimate needs to know about the
installation.

The textbook derivation - flat road, level camera - makes the relation a
straight line. Neither premise survives contact with real footage: road cameras
look downwards, and a motorway crests and curves, so beyond a certain range
vehicles stop descending the frame at all. Fitting a line, or a parabola, to
data shaped like that puts the horizon in the wrong place and makes every
distance derived from it wrong.

So the curve is measured instead of assumed. Detections are binned by apparent
width, the median contact row is taken in each bin, and the result is forced
monotone. This needs no knowledge of the camera and no site survey, and it
gives back numbers that can be checked by eye:

  * `horizon_y`       the row beyond which apparent size stops changing
  * `camera_height_m` from the near-field slope; should look like a real mount
  * `r_squared`       how well the curve explains the scatter it was fitted to

The fit uses only vehicles that visit the capture zone. Traffic on an opposite
carriageway sits at a different elevation and forms its own branch, which would
drag a global fit off.
"""
from __future__ import annotations

import numpy as np

from config.settings import Settings
from core.geometry import box_size, focal_from_fov, road_contact_point
from core.models import Calibration, Detection, VideoMeta

_MIN_SAMPLES = 60
_N_BINS = 24
_MIN_PER_BIN = 12
_RISING_SLOPE = 0.25      # px of row per px of width; below this the curve is flat
_NEAR_FIELD_KNOTS = 4


class CalibrationService:
    def __init__(self, settings: Settings):
        self.settings = settings

    def calibrate(
        self,
        meta: VideoMeta,
        detections: list[Detection],
        track_ids: set[int] | None = None,
    ) -> Calibration:
        s = self.settings
        focal = s.focal_length_px or focal_from_fov(meta.width, s.horizontal_fov_deg)

        sample = [
            d for d in detections
            if not d.clip.any
            and d.class_name == "car"
            and (track_ids is None or d.track_id in track_ids)
        ]
        if len(sample) < _MIN_SAMPLES:
            sample = [d for d in detections if not d.clip.any and d.class_name == "car"]
        if len(sample) < _MIN_SAMPLES:
            return self._fallback(meta, focal, len(sample))

        widths = np.array([box_size(d.box)[0] for d in sample], dtype=float)
        bottoms = np.array([road_contact_point(d.box)[1] for d in sample], dtype=float)

        knots_w, knots_y = self._binned_medians(widths, bottoms)
        if len(knots_w) < 4:
            return self._fallback(meta, focal, len(sample))

        # Monotone: a wider vehicle is nearer, so it can never sit higher.
        knots_y = np.maximum.accumulate(knots_y)

        pred = np.interp(widths, knots_w, knots_y)
        ss_res = float(np.sum((bottoms - pred) ** 2))
        ss_tot = float(np.sum((bottoms - bottoms.mean()) ** 2)) or 1.0

        slopes = np.diff(knots_y) / np.maximum(np.diff(knots_w), 1e-6)
        rising = np.flatnonzero(slopes >= _RISING_SLOPE)
        invert_from = int(rising[0]) if len(rising) else 0

        near = slopes[-_NEAR_FIELD_KNOTS:] if len(slopes) >= _NEAR_FIELD_KNOTS else slopes
        camera_height = float(np.median(near)) * s.vehicle_width_m
        if not (0.5 <= camera_height <= 60.0):
            camera_height = 7.0     # a typical motorway footbridge

        return Calibration(
            horizon_y=float(knots_y[0]),
            knots_w=[round(float(v), 2) for v in knots_w],
            knots_y=[round(float(v), 2) for v in knots_y],
            invert_from=invert_from,
            focal_px=float(focal),
            camera_height_m=camera_height,
            r_squared=1.0 - ss_res / ss_tot,
            n_samples=len(sample),
        )

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _binned_medians(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Median y per quantile bin of x - robust to the detector's outliers."""
        edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, _N_BINS + 1)))
        kx, ky = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            inside = (x >= lo) & (x <= hi)
            if inside.sum() >= _MIN_PER_BIN:
                kx.append(float(np.median(x[inside])))
                ky.append(float(np.median(y[inside])))
        return np.array(kx), np.array(ky)

    @staticmethod
    def _fallback(meta: VideoMeta, focal: float, n: int) -> Calibration:
        """Too little clean geometry: assume the usual framing for a road camera."""
        horizon = meta.height / 3.0
        widths = np.linspace(20.0, meta.width * 0.25, 12)
        rows = np.linspace(horizon, meta.height, 12)
        return Calibration(
            horizon_y=horizon,
            knots_w=[round(float(v), 2) for v in widths],
            knots_y=[round(float(v), 2) for v in rows],
            invert_from=0,
            focal_px=focal,
            camera_height_m=7.0,
            r_squared=0.0,
            n_samples=n,
        )
