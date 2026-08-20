from __future__ import annotations

from fastapi import APIRouter

from api.dependencies import SettingsDep

router = APIRouter(tags=["health"])


@router.get("/health")
def health(settings: SettingsDep) -> dict:
    device = "cpu"
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        pass
    return {
        "status": "ok",
        "model": settings.model_name,
        "tracker": settings.tracker_config,
        "device": device,
        "default_zone": {
            "x": settings.zone_x, "y": settings.zone_y,
            "w": settings.zone_w, "h": settings.zone_h,
        },
    }
