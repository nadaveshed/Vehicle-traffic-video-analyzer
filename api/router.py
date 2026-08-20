from __future__ import annotations

from fastapi import APIRouter

from api.routes import events, health, jobs, zones

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(zones.router)
api_router.include_router(jobs.router)
api_router.include_router(events.router)
