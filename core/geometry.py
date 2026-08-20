"""Pure geometry helpers - no I/O, no framework, trivially unit-testable."""
from __future__ import annotations

import math
from dataclasses import dataclass

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


@dataclass(frozen=True)
class NormalizedRect:
    """The 'virtual rectangle' (capture zone), stored resolution-independently."""

    x: float
    y: float
    w: float
    h: float

    def to_pixels(self, width: int, height: int) -> BBox:
        return (self.x * width, self.y * height,
                (self.x + self.w) * width, (self.y + self.h) * height)

    def contains_px(self, px: float, py: float, width: int, height: int) -> bool:
        x1, y1, x2, y2 = self.to_pixels(width, height)
        return x1 <= px <= x2 and y1 <= py <= y2


def road_contact_point(box: BBox) -> tuple[float, float]:
    """Bottom-centre of the box - where the vehicle meets the road.

    This is the only point on a bounding box whose image position maps to a
    well-defined point on the ground plane, so every depth estimate anchors
    here rather than on the box centre.
    """
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def box_size(box: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (max(x2 - x1, 1e-6), max(y2 - y1, 1e-6))


def box_area(box: BBox) -> float:
    w, h = box_size(box)
    return w * h


@dataclass(frozen=True)
class ClipFlags:
    """Which frame borders the box is touching."""

    left: bool
    right: bool
    top: bool
    bottom: bool

    @property
    def any(self) -> bool:
        return self.left or self.right or self.top or self.bottom

    @property
    def horizontal(self) -> bool:
        return self.left or self.right


def clip_flags(box: BBox, width: int, height: int, margin: int = 3) -> ClipFlags:
    x1, y1, x2, y2 = box
    return ClipFlags(
        left=x1 <= margin,
        right=x2 >= width - margin,
        top=y1 <= margin,
        bottom=y2 >= height - margin,
    )


def focal_from_fov(image_width: int, fov_deg: float) -> float:
    """Pinhole focal length in pixels from a horizontal field of view."""
    return (image_width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def parabolic_vertex(y0: float, y1: float, y2: float) -> float:
    """Sub-sample offset of the minimum of a parabola through three points.

    Returns a value in [-1, 1] relative to the middle sample; 0.0 when the
    three points are collinear.  Used to place the closest-approach instant
    between two frames instead of snapping it to a frame boundary.
    """
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return 0.0
    return max(-1.0, min(1.0, 0.5 * (y0 - y2) / denom))


def format_timestamp(seconds: float) -> str:
    m, s = divmod(max(seconds, 0.0), 60.0)
    return f"{int(m):02d}:{s:06.3f}"
