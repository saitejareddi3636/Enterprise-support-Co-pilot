from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    app_env: str
    app_port: int
    database_url: str

    @classmethod
    def from_env(cls) -> "Settings":
        app_env = os.getenv("APP_ENV", "local")
        app_port = int(os.getenv("APP_PORT", "8000"))
        database_url = os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg2://postgres:postgres@localhost:5432/enterprise_support",
        )
        return cls(app_env=app_env, app_port=app_port, database_url=database_url)


settings = Settings.from_env()

