"""Application configuration.

All tunables live here so the pipeline can be re-pointed at a different
video / camera geometry without touching the services.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CPA_", env_file=".env", extra="ignore")

    # ---------------------------------------------------------------- paths
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    output_dir: Path = BASE_DIR / "output"
    database_url: str = f"sqlite:///{(BASE_DIR / 'output' / 'cpa.db').as_posix()}"

    default_video: str = "data/highway.mp4"

    # ------------------------------------------------------------- detector
    model_name: str = "yolo11s.pt"
    device: str = "auto"              # "auto" | "cpu" | "0"
    imgsz: int = 960
    conf_threshold: float = 0.35
    iou_threshold: float = 0.5
    tracker_config: str = "bytetrack.yaml"
    # COCO ids: 2 car, 3 motorcycle, 5 bus, 7 truck
    vehicle_classes: list[int] = Field(default_factory=lambda: [2, 3, 5, 7])

    # ---------------------------------------------- capture zone (defaults)
    # Normalised rectangle (0..1) covering the stretch of road nearest the
    # camera.  A vehicle "arrives" when its road-contact point enters it.
    zone_x: float = 0.15
    zone_y: float = 0.55
    zone_w: float = 0.85
    zone_h: float = 0.45

    # ---------------------------------------------------------- geometry
    vehicle_width_m: float = 1.82     # mean passenger-car width (fallback)
    vehicle_height_m: float = 1.50
    # Real-world width per COCO class - a bus is not 1.8 m wide, and using one
    # number for everything biases the metric distance of trucks and bikes.
    class_width_m: dict[str, float] = Field(
        default_factory=lambda: {
            "car": 1.82, "motorcycle": 0.80, "bus": 2.55, "truck": 2.45,
        }
    )
    class_height_m: dict[str, float] = Field(
        default_factory=lambda: {
            "car": 1.50, "motorcycle": 1.40, "bus": 3.20, "truck": 3.00,
        }
    )
    horizontal_fov_deg: float = 50.0  # used only to turn pixels into metres
    focal_length_px: float | None = None   # set -> overrides the FOV estimate

    # ------------------------------------------------- emergency beacons
    # COCO has no emergency class, so these are detected by their lightbar:
    # saturated blue on the roof band that flickers between samples.
    detect_emergency: bool = True
    emergency_blue_ratio: float = 0.050        # with flicker, this is enough
    emergency_blue_ratio_strong: float = 0.100  # blue this loud needs no flicker
    emergency_flicker: float = 0.25            # std/mean across samples

    # ---------------------------------------------------------- filtering
    min_track_length: int = 6         # frames; shorter tracks are noise
    clip_margin_px: int = 3           # box within N px of a border == clipped
    smooth_window: int = 9            # Savitzky-Golay window (odd)
    smooth_poly: int = 2

    # ------------------------------------------------------------- runtime
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cache_detections: bool = True     # re-analyse without re-running YOLO
    write_annotated_video: bool = True
    annotated_video_scale: float = 0.75

    def resolve(self, p: str | Path) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (self.base_dir / p)


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.output_dir.mkdir(parents=True, exist_ok=True)
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s
