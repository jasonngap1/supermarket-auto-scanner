from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import bootstrap  # noqa: F401  (must run before torch)
from app.api import captures as captures_api
from app.api import items as items_api
from app.api import meta as meta_api
from app.api import scan as scan_api
from app.api.health import router as health_router
from app.config import Settings, get_settings
from app.db import init_db
from app.dependencies import AppState, build_default_state
from app.services.catalog import seed_catalog


def create_app(
    settings: Settings | None = None, services: AppState | None = None
) -> FastAPI:
    settings = settings or get_settings()
    services = services or build_default_state(settings)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.media_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        fresh_database = not settings.db_path.exists()
        init_db(settings)
        if fresh_database:
            seed_catalog(settings)
        if services.registry is not None and settings.warm_models:
            services.registry.warm()
        yield

    app = FastAPI(title="Supermarket Auto-Scanner", lifespan=lifespan)
    app.state.settings = settings
    app.state.services = services
    for router in (
        health_router,
        meta_api.router,
        items_api.router,
        captures_api.router,
        scan_api.router,
    ):
        app.include_router(router, prefix="/api")
    app.mount("/media", StaticFiles(directory=settings.media_dir), name="media")
    return app


app = create_app()
