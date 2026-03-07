"""FastAPI application factory for the GraphQL API."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from strawberry.fastapi import GraphQLRouter

from infrastructure.config import get_database_config, get_storage_config
from infrastructure.database.person_log_db import PersonLogDB
from interfaces.api.schema import schema


def create_app(config: dict) -> FastAPI:
    """Build and return the FastAPI application.

    Parameters
    ----------
    config : dict
        The fully-resolved configuration dictionary (from ``load_config``).

    Returns
    -------
    FastAPI
        Configured application with GraphQL endpoint, static image serving,
        and CORS middleware.
    """
    db_config = get_database_config(config)
    storage_config = get_storage_config(config)

    db = PersonLogDB(db_config["path"])
    images_dir = Path(storage_config["person_images_dir"])
    images_dir.mkdir(parents=True, exist_ok=True)

    # ── GraphQL context ───────────────────────────────────────────────

    async def get_context() -> dict:
        return {"db": db}

    graphql_app = GraphQLRouter(schema, context_getter=get_context)

    # ── FastAPI app ───────────────────────────────────────────────────

    app = FastAPI(title="Camera Surveillance API")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(graphql_app, prefix="/graphql")

    # Serve person crop images as static files
    app.mount("/images", StaticFiles(directory=str(images_dir)), name="images")

    return app
