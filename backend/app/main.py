from __future__ import annotations

from fastapi import FastAPI

from .core.config import settings
from .db import Base, engine


def create_app() -> FastAPI:
    app = FastAPI(title="Enterprise Support Copilot API")

    @app.on_event("startup")
    def on_startup() -> None:
        Base.metadata.create_all(bind=engine)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "env": settings.app_env}

    return app


app = create_app()

