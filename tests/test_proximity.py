"""Ground-truth test for the closest-approach solver.

A real clip has no labels, so correctness is checked against a synthetic
vehicle whose trajectory we choose: depth follows a known curve with a known
minimum, the box is projected through the same pinhole model the camera would
use, detector jitter is added, and the box is cropped at the image border
exactly as OpenCV would deliver it.

The solver must land on the true minimum; `argmax(area)` must not.

    python -m pytest tests -q          (or)      python tests/test_proximity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import Settings  # noqa: E402
from core.geometry import (NormalizedRect, clip_flags, focal_from_fov,  # noqa: E402
                           parabolic_vertex)
from core.models import Calibration, Detection, VideoMeta  # noqa: E402
from services.proximity_service import ProximityService  # noqa: E402
from services.zone_service import ZoneService  # noqa: E402

W, H, FPS = 1920, 1080, 25.0
VEHICLE_W, VEHICLE_H = 1.82, 1.50
CAM_H = 7.0
FOV = 50.0


# A 7 m camera with a 50 deg FOV sees the road contact point leave the bottom
# of a 1080-row frame at roughly Z = 20 m, so the default trajectory bottoms out
# above that and `test_survives_the_vehicle_leaving_the_frame` deliberately does
# not.
def _synthesise(
    n_frames: int = 90,
    z_min: float = 22.0,
    z_start: float = 80.0,
    cpa_at: int = 62,
    noise_px: float = 1.5,
    seed: int = 7,
) -> tuple[list[Detection], int]:
    """A vehicle approaching, reaching `z_min` at frame `cpa_at`, then receding."""
    rng = np.random.default_rng(seed)
    f = focal_from_fov(W, FOV)
    horizon = H / 3.0

    dets: list[Detection] = []
    for i in range(n_frames):
        # V-shaped depth profile with a genuine interior minimum.
        t = (i - cpa_at) / cpa_at
        z = z_min + (z_start - z_min) * abs(t) ** 1.6

        w_px = f * VEHICLE_W / z
        h_px = f * VEHICLE_H / z
        y_bottom = horizon + f * CAM_H / z
        x_center = W / 2 + 40 * t

        x1 = x_center - w_px / 2 + rng.normal(0, noise_px)
        x2 = x_center + w_px / 2 + rng.normal(0, noise_px)
        y2 = y_bottom + rng.normal(0, noise_px)
        y1 = y2 - h_px + rng.normal(0, noise_px)

        # What OpenCV would actually hand us: the box, clipped to the frame.
        box = (max(0.0, x1), max(0.0, y1), min(float(W), x2), min(float(H), y2))
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            continue
        dets.append(
            Detection(
                frame_index=i,
                timestamp=i / FPS,
                track_id=1,
                class_id=2,
                class_name="car",
                confidence=0.9,
                box=box,
                clip=clip_flags(box, W, H, 3),
            )
        )
    return dets, cpa_at


def _calibration(n: int) -> Calibration:
    """The exact camera used by `_synthesise`, expressed as the measured curve.

    In the synthetic world y_bottom = horizon + (CAM_H / VEHICLE_W) * w_px, so
    the knots lie on a straight line and the solver is handed a perfect model -
    which is the point: any error the tests catch is the solver's, not the
    calibration's.
    """
    widths = np.linspace(20.0, 600.0, 24)
    rows = H / 3.0 + (CAM_H / VEHICLE_W) * widths
    return Calibration(
        horizon_y=H / 3.0,
        knots_w=[float(v) for v in widths],
        knots_y=[float(v) for v in rows],
        invert_from=0,
        focal_px=focal_from_fov(W, FOV),
        camera_height_m=CAM_H,
        r_squared=1.0,
        n_samples=n,
    )


def _analyse(dets, zone: ZoneService | None = None):
    settings = Settings()
    meta = VideoMeta("synthetic", W, H, FPS, 200)
    zone = zone or ZoneService(NormalizedRect(0.0, 0.0, 1.0, 1.0))
    return ProximityService(settings).analyse(dets, meta, _calibration(len(dets)), zone), meta


# --------------------------------------------------------------------- tests
def test_finds_interior_minimum():
    dets, truth = _synthesise()
    result, _ = _analyse(dets)
    assert result is not None
    assert abs(result.frame_index - truth) <= 2, (
        f"picked frame {result.frame_index}, truth {truth}"
    )


def test_recovers_depth_within_ten_percent():
    dets, _ = _synthesise()
    result, _ = _analyse(dets)
    assert abs(result.distance_m - 22.0) / 22.0 < 0.10, result.distance_m


def test_three_metrics_agree():
    dets, _ = _synthesise()
    result, _ = _analyse(dets)
    picks = [p.frame_index for p in result.picks if p.frame_index >= 0]
    assert len(picks) == 3
    assert max(picks) - min(picks) <= 3, picks


def test_never_reports_a_cropped_box():
    """The hard case: the vehicle leaves through the bottom of the frame.

    The contract is that the reported frame always carries a complete bounding
    box, that it is the last such frame before the vehicle goes, and that the
    vehicle is flagged as having continued past it.
    """
    dets, _ = _synthesise(n_frames=70, z_min=9.0, cpa_at=69, z_start=80.0, noise_px=1.0)
    assert any(d.clip.any for d in dets), "the case is meant to include cropped boxes"

    result, _ = _analyse(dets)
    assert result is not None

    picked = next(d for d in dets if d.frame_index == result.frame_index)
    assert not picked.clip.any, "reported a box that the frame border had cut"

    last_clean = max(d.frame_index for d in dets if not d.clip.any)
    assert result.frame_index >= last_clean - 2, (
        f"picked {result.frame_index}, last fully-visible frame is {last_clean}"
    )
    assert result.exits_view, "the vehicle leaves the frame and should be flagged"


def test_zone_gates_the_search():
    """A rectangle covering only the far half must not pick a near-field frame."""
    dets, _ = _synthesise()
    result, _ = _analyse(dets, ZoneService(NormalizedRect(0.0, 0.47, 1.0, 0.13)))
    assert result is not None
    assert result.entered_zone
    assert result.zone_entry_frame <= result.frame_index <= result.zone_exit_frame


def test_parabolic_vertex_is_exact_on_a_parabola():
    # y = (x - 0.3)^2 sampled at -1, 0, 1 -> the minimum sits at +0.3
    f = lambda x: (x - 0.3) ** 2  # noqa: E731
    assert abs(parabolic_vertex(f(-1), f(0), f(1)) - 0.3) < 1e-9
    assert parabolic_vertex(1.0, 1.0, 1.0) == 0.0


def test_short_tracks_are_dropped():
    dets, _ = _synthesise(n_frames=4)
    result, _ = _analyse(dets)
    assert result is None


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"  FAIL  {name}: {exc}")
    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
