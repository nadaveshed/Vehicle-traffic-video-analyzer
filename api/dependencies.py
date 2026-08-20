"""FastAPI dependencies - repositories bound to a request-scoped session."""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from config.settings import Settings, get_settings
from repository.database import get_session
from repository.event_repository import EventRepository
from repository.job_repository import JobRepository
from repository.zone_repository import ZoneRepository

SessionDep = Annotated[Session, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def get_zone_repository(session: SessionDep) -> ZoneRepository:
    return ZoneRepository(session)


def get_job_repository(session: SessionDep) -> JobRepository:
    return JobRepository(session)


def get_event_repository(session: SessionDep) -> EventRepository:
    return EventRepository(session)


ZoneRepoDep = Annotated[ZoneRepository, Depends(get_zone_repository)]
JobRepoDep = Annotated[JobRepository, Depends(get_job_repository)]
EventRepoDep = Annotated[EventRepository, Depends(get_event_repository)]
