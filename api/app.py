"""FastAPI application factory."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.router import api_router
from repository.database import init_db, session_scope
from repository.zone_repository import ZoneRepository

DESCRIPTION = """
Finds, for every vehicle in a traffic clip, the exact frame at which it was
closest to the camera.

A **virtual rectangle** is pinned to the stretch of road nearest the camera.
When the road-contact point of a vehicle enters it the capture is armed; the
exact frame is then chosen inside that window by three independent geometric
depth estimates, and the frame, timestamp and bounding box are stored along
with the snapshot and the evidence needed to check the answer.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    with session_scope() as session:
        ZoneRepository(session).ensure_default()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Vehicle Closest-Approach API",
        description=DESCRIPTION,
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router)

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/docs")

    return app


app = create_app()
