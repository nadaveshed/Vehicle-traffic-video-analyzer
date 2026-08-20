from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from config.settings import get_settings
from core.geometry import NormalizedRect
from repository.models import Zone


class ZoneRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, name: str, rect: NormalizedRect, description: str | None = None) -> Zone:
        zone = Zone(name=name, x=rect.x, y=rect.y, w=rect.w, h=rect.h, description=description)
        self.session.add(zone)
        self.session.commit()
        self.session.refresh(zone)
        return zone

    def get(self, zone_id: int) -> Zone | None:
        return self.session.get(Zone, zone_id)

    def list(self) -> list[Zone]:
        return list(self.session.scalars(select(Zone).order_by(Zone.id)))

    def delete(self, zone_id: int) -> bool:
        zone = self.get(zone_id)
        if zone is None:
            return False
        self.session.delete(zone)
        self.session.commit()
        return True

    def ensure_default(self) -> Zone:
        """The zone shipped in config - created once, then reused."""
        existing = self.session.scalar(select(Zone).where(Zone.name == "default"))
        if existing:
            return existing
        s = get_settings()
        return self.create(
            "default",
            NormalizedRect(s.zone_x, s.zone_y, s.zone_w, s.zone_h),
            "Stretch of road nearest the camera; a vehicle's road-contact "
            "point entering this rectangle arms the capture.",
        )
