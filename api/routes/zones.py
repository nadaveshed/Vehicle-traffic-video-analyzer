from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from api.dependencies import ZoneRepoDep
from api.schemas import ZoneCreate, ZoneOut
from core.geometry import NormalizedRect

router = APIRouter(prefix="/zones", tags=["zones"])


@router.get("", response_model=list[ZoneOut])
def list_zones(repo: ZoneRepoDep):
    repo.ensure_default()
    return repo.list()


@router.post("", response_model=ZoneOut, status_code=status.HTTP_201_CREATED)
def create_zone(payload: ZoneCreate, repo: ZoneRepoDep):
    if payload.x + payload.w > 1.0 or payload.y + payload.h > 1.0:
        raise HTTPException(422, "rectangle extends past the frame")
    return repo.create(
        payload.name,
        NormalizedRect(payload.x, payload.y, payload.w, payload.h),
        payload.description,
    )


@router.get("/{zone_id}", response_model=ZoneOut)
def get_zone(zone_id: int, repo: ZoneRepoDep):
    zone = repo.get(zone_id)
    if zone is None:
        raise HTTPException(404, "zone not found")
    return zone


@router.delete("/{zone_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_zone(zone_id: int, repo: ZoneRepoDep):
    if not repo.delete(zone_id):
        raise HTTPException(404, "zone not found")
