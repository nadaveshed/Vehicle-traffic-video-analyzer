"""Thin wrapper around OpenCV video I/O."""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from core.models import VideoMeta


def probe(path: str | Path) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    meta = VideoMeta(
        path=str(path),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=float(cap.get(cv2.CAP_PROP_FPS)) or 25.0,
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return meta


def iter_frames(path: str | Path) -> Iterator[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield index, frame
            index += 1
    finally:
        cap.release()


class FrameReader:
    """Random access to frames, with a tiny cache.

    OpenCV seeking on a long-GOP mp4 is expensive, so callers that need many
    scattered frames should sort their requests; this class keeps the last
    grab around because contact sheets ask for neighbouring frames.
    """

    def __init__(self, path: str | Path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {path}")
        self._last_index: int | None = None
        self._last_frame: np.ndarray | None = None

    def read(self, index: int) -> np.ndarray | None:
        if index == self._last_index and self._last_frame is not None:
            return self._last_frame.copy()
        if self._last_index is not None and 0 < index - self._last_index <= 4:
            for _ in range(index - self._last_index - 1):
                self.cap.read()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            return None
        self._last_index, self._last_frame = index, frame
        return frame.copy()

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> "FrameReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
