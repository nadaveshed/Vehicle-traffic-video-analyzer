"""Everything the examiner looks at.

Four aids, each attacking the same claim from a different angle:

  1. annotated video      - the rectangle, every track, and a flash at the
                            exact capture frame
  2. snapshot + crop      - the captured frame itself, per vehicle
  3. distance-vs-time plot- the curve with the chosen minimum marked, next to
                            the frame a naive argmax(area) would have chosen
  4. contact sheet        - five frames around the capture so the middle one
                            being the biggest is obvious at a glance
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from config.settings import Settings
from core.geometry import NormalizedRect, format_timestamp
from core.models import Calibration, ClosestApproach, Detection, VideoMeta
from services.video_service import FrameReader, iter_frames

# BGR
COL_TRACKED = (200, 170, 90)
COL_ARMED = (60, 210, 245)
COL_CAPTURE = (60, 60, 255)
COL_ZONE = (255, 200, 40)
COL_EMERGENCY = (230, 60, 200)   # emergency vehicles keep this colour throughout
COL_TEXT = (255, 255, 255)
COL_PANEL = (32, 28, 26)

FLASH_FRAMES = 12
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(img, text, org, scale=0.5, color=COL_TEXT, bg=COL_PANEL, thickness=1, pad=4):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x, y - th - pad), (x + tw + 2 * pad, y + base + pad // 2), bg, -1)
    cv2.putText(img, text, (x + pad, y), FONT, scale, color, thickness, cv2.LINE_AA)
    return tw + 2 * pad


def _dashed_line(img, p1, p2, color, thickness=1, dash=14):
    x1, y1 = p1
    x2, y2 = p2
    length = int(np.hypot(x2 - x1, y2 - y1))
    if length == 0:
        return
    for i in range(0, length, dash * 2):
        a = i / length
        b = min((i + dash) / length, 1.0)
        cv2.line(
            img,
            (int(x1 + (x2 - x1) * a), int(y1 + (y2 - y1) * a)),
            (int(x1 + (x2 - x1) * b), int(y1 + (y2 - y1) * b)),
            color, thickness, cv2.LINE_AA,
        )


def draw_zone(frame: np.ndarray, rect: NormalizedRect, armed: int = 0) -> None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in rect.to_pixels(w, h))
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), COL_ZONE, -1)
    cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), COL_ZONE, 2, cv2.LINE_AA)
    for cx, cy in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
        cv2.drawMarker(frame, (cx, cy), COL_ZONE, cv2.MARKER_CROSS, 18, 2)
    tag = "CAPTURE ZONE" + (f"  -  {armed} armed" if armed else "")
    _label(frame, tag, (x1 + 8, y1 + 24), 0.6, COL_ZONE, COL_PANEL, 2)


class VisualizationService:
    def __init__(self, settings: Settings):
        self.settings = settings

    # ------------------------------------------------------------- video ---
    def render_video(
        self,
        video_path: str,
        out_path: Path,
        meta: VideoMeta,
        detections: list[Detection],
        approaches: list[ClosestApproach],
        rect: NormalizedRect,
        calib: Calibration,
        progress=None,
    ) -> Path:
        s = self.settings
        by_frame: dict[int, list[Detection]] = defaultdict(list)
        for d in detections:
            by_frame[d.frame_index].append(d)

        cpa_by_track = {a.track_id: a for a in approaches}
        dist_by_key = {
            (a.track_id, f): v
            for a in approaches
            for f, v in zip(a.series["frames"], a.series["fused_smooth"])
            if v is not None
        }
        zone_by_track = {
            a.track_id: (a.zone_entry_frame, a.zone_exit_frame)
            for a in approaches
            if a.entered_zone
        }

        scale = s.annotated_video_scale
        out_w, out_h = int(meta.width * scale), int(meta.height * scale)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), meta.fps, (out_w, out_h)
        )

        captures_so_far = 0
        for idx, frame in iter_frames(video_path):
            dets = by_frame.get(idx, [])
            armed = sum(
                1 for d in dets
                if d.track_id in zone_by_track
                and zone_by_track[d.track_id][0] <= idx <= zone_by_track[d.track_id][1]
            )
            draw_zone(frame, rect, armed)
            _dashed_line(
                frame, (0, int(calib.horizon_y)), (meta.width, int(calib.horizon_y)),
                (150, 150, 150), 1,
            )
            _label(frame, "fitted horizon", (12, int(calib.horizon_y) - 8), 0.45,
                   (190, 190, 190))

            for d in dets:
                cpa = cpa_by_track.get(d.track_id)
                x1, y1, x2, y2 = (int(v) for v in d.box)
                in_zone = (
                    d.track_id in zone_by_track
                    and zone_by_track[d.track_id][0] <= idx <= zone_by_track[d.track_id][1]
                )
                is_capture = cpa is not None and 0 <= idx - cpa.frame_index < FLASH_FRAMES

                emergency = cpa is not None and cpa.is_emergency
                if is_capture:
                    color, thick = COL_CAPTURE, 4
                elif emergency:
                    color, thick = COL_EMERGENCY, 3
                elif in_zone:
                    color, thick = COL_ARMED, 3
                else:
                    color, thick = COL_TRACKED, 2

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick, cv2.LINE_AA)
                cx = (x1 + x2) // 2
                cv2.drawMarker(frame, (cx, y2), color, cv2.MARKER_TRIANGLE_UP, 14, 2)

                dist = dist_by_key.get((d.track_id, idx))
                kind = cpa.vehicle_type if cpa is not None else d.class_name
                text = f"#{d.track_id} {kind.upper() if emergency else kind}"
                if dist is not None:
                    text += f" {dist:5.1f}m"
                _label(frame, text, (x1, max(y1 - 6, 18)), 0.52, COL_TEXT, color, 1)

                if is_capture:
                    age = idx - cpa.frame_index
                    ring = int(6 + age * 3)
                    cv2.rectangle(frame, (x1 - ring, y1 - ring), (x2 + ring, y2 + ring),
                                  COL_CAPTURE, max(1, 3 - age // 4), cv2.LINE_AA)
                    note = "  (leaves frame)" if cpa.exits_view else ""
                    # A vehicle at its closest is usually at the bottom of the
                    # frame, so the caption goes above the box when there is no
                    # room under it.
                    caption_y = y2 + 30 if y2 + 40 < meta.height else max(y1 - 12, 30)
                    _label(frame,
                           f"CLOSEST  f{cpa.frame_index}  "
                           f"t={format_timestamp(cpa.refined_timestamp)}  "
                           f"{cpa.distance_m:.1f}m{note}",
                           (min(x1, meta.width - 520), caption_y), 0.6, COL_TEXT,
                           COL_CAPTURE, 2)
                    if age == 0:
                        captures_so_far += 1

            self._hud(frame, idx, meta, len(dets), armed, captures_so_far)
            writer.write(cv2.resize(frame, (out_w, out_h)))
            if progress and meta.frame_count and idx % 25 == 0:
                progress(idx / meta.frame_count, f"rendering frame {idx}/{meta.frame_count}")

        writer.release()
        return out_path

    def _hud(self, frame, idx, meta: VideoMeta, n_tracked, n_armed, n_captured):
        lines = [
            f"frame {idx:5d} / {meta.frame_count}   t={format_timestamp(idx / meta.fps)}",
            f"tracked {n_tracked:2d}   armed {n_armed:2d}   captured {n_captured:3d}",
        ]
        pad = 10
        box_h = 26 * len(lines) + pad
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (520, box_h), COL_PANEL, -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (pad, 24 + i * 26), FONT, 0.62, COL_TEXT, 1, cv2.LINE_AA)

    # ---------------------------------------------------- stills per event --
    def render_event_stills(
        self,
        video_path: str,
        approaches: list[ClosestApproach],
        meta: VideoMeta,
        rect: NormalizedRect,
        out_dir: Path,
    ) -> dict[int, dict[str, str]]:
        """Snapshot, crop and contact sheet for every capture event."""
        snaps = out_dir / "snapshots"
        crops = out_dir / "crops"
        sheets = out_dir / "contact_sheets"
        for d in (snaps, crops, sheets):
            d.mkdir(parents=True, exist_ok=True)

        by_track = {a.track_id: a for a in approaches}
        offsets = (-30, -15, 0, 8, 16)
        wanted: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for a in approaches:
            for off in offsets:
                wanted[max(0, min(meta.frame_count - 1, a.frame_index + off))].append(
                    (a.track_id, off)
                )

        boxes_by_track = {
            a.track_id: dict(zip(a.series["frames"], a.series["boxes"]))
            for a in approaches
        }
        out: dict[int, dict[str, str]] = {}
        sheet_tiles: dict[int, list] = defaultdict(list)
        cell_w = 420
        cell_h = int(cell_w * meta.height / meta.width)

        with FrameReader(video_path) as reader:
            for frame_no in sorted(wanted):
                frame = reader.read(frame_no)
                if frame is None:
                    continue
                for track_id, off in wanted[frame_no]:
                    a = by_track[track_id]
                    tile = frame.copy()
                    if off == 0:
                        x1, y1, x2, y2 = (int(v) for v in a.box)
                        draw_zone(tile, rect)
                        cv2.rectangle(tile, (x1, y1), (x2, y2), COL_CAPTURE, 4, cv2.LINE_AA)
                        _label(tile,
                               f"#{a.track_id} {a.vehicle_type}  frame {a.frame_index}  "
                               f"t={format_timestamp(a.refined_timestamp)}  "
                               f"{a.distance_m:.1f} m",
                               (x1, max(y1 - 10, 24)), 0.8, COL_TEXT,
                               COL_EMERGENCY if a.is_emergency else COL_CAPTURE, 2)
                        snap_path = snaps / f"track_{a.track_id:03d}_f{a.frame_index:05d}.jpg"
                        cv2.imwrite(str(snap_path), tile, [cv2.IMWRITE_JPEG_QUALITY, 88])

                        pad = 24
                        cx1 = max(0, x1 - pad)
                        cy1 = max(0, y1 - pad)
                        cx2 = min(meta.width, x2 + pad)
                        cy2 = min(meta.height, y2 + pad)
                        crop = frame[cy1:cy2, cx1:cx2]
                        crop_path = crops / f"track_{a.track_id:03d}.jpg"
                        if crop.size:
                            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                        out.setdefault(a.track_id, {})["snapshot_path"] = str(snap_path)
                        out.setdefault(a.track_id, {})["crop_path"] = str(crop_path)

                    # Keep only the down-scaled cell: holding five full frames
                    # per vehicle would run into gigabytes on a long clip.  The
                    # vehicle is boxed in *every* tile, otherwise the reader
                    # cannot tell which of the cars on the road is this track.
                    cell = cv2.resize(frame, (cell_w, cell_h))
                    box = boxes_by_track.get(a.track_id, {}).get(frame_no)
                    if box is not None:
                        sx, sy = cell_w / meta.width, cell_h / meta.height
                        bx1, by1, bx2, by2 = box
                        cv2.rectangle(
                            cell,
                            (int(bx1 * sx), int(by1 * sy)),
                            (int(bx2 * sx), int(by2 * sy)),
                            COL_CAPTURE if off == 0 else COL_ARMED,
                            3 if off == 0 else 2, cv2.LINE_AA,
                        )
                    sheet_tiles[track_id].append((off, frame_no, cell))

        for track_id, tiles in sheet_tiles.items():
            a = by_track[track_id]
            path = self._contact_sheet(a, tiles, meta, sheets)
            out.setdefault(track_id, {})["contact_sheet_path"] = str(path)
        return out

    def _contact_sheet(self, a: ClosestApproach, tiles, meta: VideoMeta, out_dir: Path) -> Path:
        series_frames = a.series["frames"]
        series_dist = a.series["fused_smooth"]
        lookup = dict(zip(series_frames, series_dist))

        tiles = sorted(tiles, key=lambda t: t[0])
        rendered = []
        for off, frame_no, img in tiles:
            cell_h, cell_w = img.shape[:2]
            is_cpa = off == 0
            border = COL_CAPTURE if is_cpa else (90, 90, 90)
            cv2.rectangle(img, (0, 0), (cell_w - 1, cell_h - 1), border, 6 if is_cpa else 2)
            dt = (frame_no - a.frame_index) / meta.fps
            dist = lookup.get(frame_no)
            cap = f"f{frame_no}  {dt:+.2f}s"
            if dist is not None:
                cap += f"  {dist:.1f}m"
            if is_cpa:
                cap = "CLOSEST  " + cap
            _label(img, cap, (8, cell_h - 12), 0.5, COL_TEXT, border, 1)
            rendered.append(img)

        sheet = cv2.hconcat(rendered)
        header = np.full((46, sheet.shape[1], 3), COL_PANEL, np.uint8)
        cv2.putText(
            header,
            f"track #{a.track_id} ({a.vehicle_type})   closest at frame {a.frame_index}"
            f"   t={format_timestamp(a.refined_timestamp)}   d={a.distance_m:.1f} m",
            (12, 31), FONT, 0.68, COL_TEXT, 1, cv2.LINE_AA,
        )
        sheet = cv2.vconcat([header, sheet])
        path = out_dir / f"track_{a.track_id:03d}.jpg"
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return path
