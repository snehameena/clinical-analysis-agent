"""
FastAPI application factory with lifespan, CORS, and routers.
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import health, pipeline
from src.env import load_env

load_env()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: pre-compile graph. Shutdown: cleanup."""
    from src.api.dependencies import get_graph
    get_graph()  # warm up the singleton
    logging.getLogger(__name__).info("Pipeline graph compiled")
    yield
    logging.getLogger(__name__).info("Shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Healthcare Content Intelligence Pipeline",
        description="Multi-agent clinical evidence synthesis API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(health.router)
    app.include_router(pipeline.router)

    return app


app = create_app()
