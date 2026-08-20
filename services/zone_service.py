"""The virtual rectangle and the arm/trigger state machine.

A vehicle 'arrives' when its road-contact point (box bottom-centre) crosses
into the rectangle.  The rectangle is the *gate*: it decides which stretch of
the clip is searched.  The exact frame of closest approach is then picked
inside that window by `ProximityService` - the zone says roughly when, the
geometry says exactly when.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.geometry import NormalizedRect, road_contact_point
from core.models import Detection, VideoMeta


@dataclass
class ZoneVisit:
    entered: bool
    entry_frame: int | None
    exit_frame: int | None
    frames_inside: int


class ZoneService:
    def __init__(self, rect: NormalizedRect):
        self.rect = rect

    def is_inside(self, det: Detection, meta: VideoMeta) -> bool:
        cx, cy = road_contact_point(det.box)
        return self.rect.contains_px(cx, cy, meta.width, meta.height)

    def visit(self, track: list[Detection], meta: VideoMeta) -> ZoneVisit:
        """First contiguous-ish dwell of this track inside the rectangle.

        We take the first entry and the last frame still inside, so a detector
        hiccup in the middle of the dwell does not split one vehicle into two
        capture events.
        """
        inside = [d.frame_index for d in track if self.is_inside(d, meta)]
        if not inside:
            return ZoneVisit(False, None, None, 0)
        return ZoneVisit(True, min(inside), max(inside), len(inside))
