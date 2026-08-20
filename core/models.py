"""Domain objects passed between services (framework-free)."""
from __future__ import annotations

from dataclasses import dataclass, field

from core.geometry import BBox, ClipFlags


@dataclass
class Detection:
    frame_index: int
    timestamp: float
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    box: BBox
    clip: ClipFlags


@dataclass
class VideoMeta:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


@dataclass
class Calibration:
    """The camera model, recovered from the detections themselves.

    Rather than assume a flat road and a level camera - neither of which holds
    for a real motorway view, where the carriageway crests and the camera looks
    down - the relation between a vehicle's apparent width and the image row of
    its wheels is measured directly and stored as a monotone curve. That curve
    *is* the camera model for this installation.

    `knots_w` / `knots_y` hold the curve. `invert_from` is the first knot past
    which the curve actually rises: below it, distant vehicles all sit on the
    same row and the bottom edge carries no depth information, so the inverse
    abstains instead of guessing.
    """

    horizon_y: float
    knots_w: list[float]
    knots_y: list[float]
    invert_from: int
    focal_px: float
    camera_height_m: float
    r_squared: float
    n_samples: int

    def bottom_from_width(self, w_px):
        """Where a vehicle of this apparent width meets the road, in image rows."""
        import numpy as np

        kw, ky = np.asarray(self.knots_w), np.asarray(self.knots_y)
        return np.interp(np.asarray(w_px, dtype=float), kw, ky)

    def width_from_bottom(self, y_bottom):
        """Inverse: the apparent width a vehicle on this row would have.

        NaN where the curve is flat (too far away to tell) - the caller drops
        those samples rather than trusting them.
        """
        import numpy as np

        y = np.asarray(y_bottom, dtype=float)
        kw = np.asarray(self.knots_w)[self.invert_from:]
        ky = np.asarray(self.knots_y)[self.invert_from:]
        if len(kw) < 2:
            return np.full(y.shape, np.nan)

        out = np.interp(y, ky, kw)
        # Straight-line extrapolation past the nearest knot, so a vehicle that
        # comes closer than anything in the fit still gets an estimate.
        tail = (ky[-1] - ky[-2]) / max(kw[-1] - kw[-2], 1e-6)
        beyond = y > ky[-1]
        out = np.where(beyond, kw[-1] + (y - ky[-1]) / max(tail, 1e-6), out)
        return np.where(y < ky[0], np.nan, out)


@dataclass
class MetricPick:
    name: str
    frame_index: int
    value: float


@dataclass
class ClosestApproach:
    """The answer for one vehicle."""

    track_id: int
    class_name: str            # what COCO called it
    vehicle_type: str          # what we report: car/truck/bus/motorcycle/emergency
    frame_index: int
    timestamp: float
    refined_timestamp: float
    box: BBox
    center: tuple[float, float]
    contact_point: tuple[float, float]
    distance_m: float
    confidence: float
    first_frame: int
    last_frame: int
    frames_tracked: int
    entered_zone: bool
    zone_entry_frame: int | None
    zone_exit_frame: int | None
    exits_view: bool
    truncated_start: bool = False
    truncated_end: bool = False
    picks: list[MetricPick] = field(default_factory=list)
    naive_area_frame: int | None = None
    agreement_frames: int = 0
    agreement_distance_pct: float = 0.0
    is_emergency: bool = False
    emergency_score: float = 0.0
    series: dict = field(default_factory=dict)
