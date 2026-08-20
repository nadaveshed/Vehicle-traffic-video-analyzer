"""Turn a track into the single frame where the vehicle was closest.

Definition used throughout
--------------------------
The closest approach is the frame that minimises the estimated camera distance
**among the frames in which the vehicle is wholly inside the image**. Once a
box touches an image border the vehicle is only partly visible: the detector
then reports the width of the visible fragment, not of the vehicle, so every
size-based depth estimate silently breaks. Vehicles that go on to leave the
frame are flagged `exits_view` - they kept approaching, but past that frame no
honest measurement exists.

Three depth estimates are computed per frame, each from a different part of the
box, so no two share a failure mode:

    ground   from the box's bottom edge, through the calibrated curve
    width    Z = f · W_real / w_px
    diag     Z = f · sqrt(W·H) / sqrt(area)

They are fused with a per-frame median, smoothed, and the minimum is located to
sub-frame precision with a parabola fit. The three picks are kept separately:
their agreement is what makes the chosen frame checkable.

The baseline this replaces - `argmax(box area)` over every frame - is computed
alongside, because on a vehicle that leaves through the bottom of the frame it
fires on a cropped box and reports a bounding box that is not the vehicle.
"""
from __future__ import annotations

import math
import warnings

import numpy as np

from config.settings import Settings
from core.geometry import (BBox, box_area, box_size, parabolic_vertex,
                           road_contact_point)
from core.models import (Calibration, ClosestApproach, Detection, MetricPick,
                         VideoMeta)
from services.zone_service import ZoneService

_METRICS = ("ground", "width", "diag")


def _interp_nan(y: np.ndarray) -> np.ndarray:
    """Fill interior NaNs so the smoother sees a continuous signal."""
    y = y.astype(float).copy()
    ok = ~np.isnan(y)
    if ok.sum() < 2:
        return y
    idx = np.arange(len(y))
    y[~ok] = np.interp(idx[~ok], idx[ok], y[ok])
    return y


def _smooth(y: np.ndarray, window: int, poly: int) -> np.ndarray:
    n = len(y)
    if n < 5:
        return y
    win = min(window, n if n % 2 == 1 else n - 1)
    if win <= poly:
        return y
    try:
        from scipy.signal import savgol_filter

        return savgol_filter(y, win, poly)
    except Exception:
        return np.convolve(y, np.ones(3) / 3.0, mode="same")


class ProximityService:
    def __init__(self, settings: Settings):
        self.settings = settings

    # ------------------------------------------------------------------ api
    def analyse(
        self,
        track: list[Detection],
        meta: VideoMeta,
        calib: Calibration,
        zone: ZoneService,
    ) -> ClosestApproach | None:
        s = self.settings
        if len(track) < s.min_track_length:
            return None

        frames = np.array([d.frame_index for d in track])
        boxes: list[BBox] = [d.box for d in track]
        widths = np.array([box_size(b)[0] for b in boxes], dtype=float)
        areas = np.array([box_area(b) for b in boxes], dtype=float)
        bottoms = np.array([road_contact_point(b)[1] for b in boxes], dtype=float)
        measurable = np.array([not d.clip.any for d in track])

        cls = track[len(track) // 2].class_name
        W = s.class_width_m.get(cls, s.vehicle_width_m)
        Hv = s.class_height_m.get(cls, s.vehicle_height_m)
        f = calib.focal_px

        # ---- three independent depth estimates; unmeasurable frames -> NaN ---
        w_from_bottom = np.asarray(calib.width_from_bottom(bottoms), dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            d_ground = np.where(
                measurable & (w_from_bottom > 0), f * W / w_from_bottom, np.nan
            )
            d_width = np.where(measurable, f * W / np.maximum(widths, 1e-6), np.nan)
            d_diag = np.where(
                measurable,
                f * math.sqrt(W * Hv) / np.sqrt(np.maximum(areas, 1e-6)),
                np.nan,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fused = np.nanmedian(np.vstack([d_ground, d_width, d_diag]), axis=0)
            # Kept only for the plot: what the estimate looks like once the box
            # is cropped, which is exactly where it stops meaning anything.
            unreliable = np.where(
                measurable, np.nan, f * W / np.maximum(widths, 1e-6)
            )

        if not np.any(np.isfinite(fused)):
            return None

        fused_s = _smooth(_interp_nan(fused), s.smooth_window, s.smooth_poly)
        fused_s = np.where(np.isnan(fused), np.nan, fused_s)

        # ------------------------ the virtual rectangle gates the search ---
        visit = zone.visit(track, meta)
        window = np.ones(len(track), dtype=bool)
        if visit.entered:
            gated = (frames >= visit.entry_frame) & (frames <= visit.exit_frame)
            if np.any(gated & ~np.isnan(fused_s)):
                window = gated

        searchable = window & ~np.isnan(fused_s)
        if not searchable.any():
            return None

        i_best = int(np.argmin(np.where(searchable, fused_s, np.inf)))

        # --------------------------------- sub-frame minimum via parabola --
        offset = 0.0
        if 0 < i_best < len(fused_s) - 1:
            trio = fused_s[i_best - 1: i_best + 2]
            gaps = np.diff(frames[i_best - 1: i_best + 2])
            if not np.isnan(trio).any() and np.all(gaps == 1):
                offset = parabolic_vertex(float(trio[0]), float(trio[1]), float(trio[2]))

        best = track[i_best]
        picks = [
            MetricPick(name, *self._argmin_of(arr, searchable, frames))
            for name, arr in zip(_METRICS, (d_ground, d_width, d_diag))
        ]
        valid = [p.frame_index for p in picks if p.frame_index >= 0]
        agreement = max(valid) - min(valid) if len(valid) > 1 else 0

        # How much disagreement actually costs.  In stop-and-go traffic a
        # vehicle can sit at nearly the same distance for a second at a time;
        # the three estimates then pick frames far apart while agreeing on the
        # distance to within noise, and a frame count on its own would make a
        # well-behaved result look like a failure.  This measures the spread in
        # depth across the frames they chose, which is the quantity that
        # matters.
        by_frame = dict(zip(frames.tolist(), fused_s.tolist()))
        at_picks = [by_frame.get(fi) for fi in valid]
        at_picks = [v for v in at_picks if v is not None and np.isfinite(v)]
        if len(at_picks) > 1 and min(at_picks) > 0:
            agreement_distance_pct = (max(at_picks) - min(at_picks)) / min(at_picks) * 100.0
        else:
            agreement_distance_pct = 0.0

        naive_area_frame = int(frames[int(np.argmax(np.where(window, areas, -np.inf)))])

        cx = (best.box[0] + best.box[2]) / 2.0
        cy = (best.box[1] + best.box[3]) / 2.0

        return ClosestApproach(
            track_id=best.track_id,
            class_name=best.class_name,
            vehicle_type=best.class_name,
            frame_index=best.frame_index,
            timestamp=best.frame_index / meta.fps,
            refined_timestamp=(best.frame_index + offset) / meta.fps,
            box=best.box,
            center=(cx, cy),
            contact_point=road_contact_point(best.box),
            distance_m=float(fused_s[i_best]),
            confidence=best.confidence,
            first_frame=int(frames[0]),
            last_frame=int(frames[-1]),
            frames_tracked=len(track),
            entered_zone=visit.entered,
            zone_entry_frame=visit.entry_frame,
            zone_exit_frame=visit.exit_frame,
            exits_view=bool(not measurable[-1]),
            truncated_start=bool(best.frame_index == 0),
            truncated_end=bool(best.frame_index >= meta.frame_count - 1),
            picks=picks,
            naive_area_frame=naive_area_frame,
            agreement_frames=int(agreement),
            agreement_distance_pct=round(float(agreement_distance_pct), 3),
            series={
                "frames": frames.tolist(),
                "timestamps": (frames / meta.fps).round(4).tolist(),
                "fused": _json_floats(fused),
                "fused_smooth": _json_floats(fused_s),
                "unreliable": _json_floats(unreliable),
                "ground": _json_floats(d_ground),
                "width": _json_floats(d_width),
                "diag": _json_floats(d_diag),
                "area": areas.round(1).tolist(),
                "boxes": [[round(v, 1) for v in b] for b in boxes],
                "cropped": [bool(v) for v in ~measurable],
                "in_zone": [bool(v) for v in window],
            },
        )

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _argmin_of(
        arr: np.ndarray, searchable: np.ndarray, frames: np.ndarray
    ) -> tuple[int, float]:
        usable = searchable & ~np.isnan(arr)
        if not usable.any():
            return -1, float("nan")
        i = int(np.argmin(np.where(usable, arr, np.inf)))
        return int(frames[i]), float(arr[i])


def _json_floats(a: np.ndarray) -> list[float | None]:
    return [None if not np.isfinite(v) else round(float(v), 4) for v in a]
