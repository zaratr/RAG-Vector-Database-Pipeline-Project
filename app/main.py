"""FastAPI application entrypoint."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.db import Base, engine
from app.api import routes_documents, routes_query
from app.config import get_settings
from app.core.logging import logger

settings = get_settings()

# Ensure tables exist
Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.app_name, debug=settings.debug)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_documents.router)
app.include_router(routes_query.router)


@app.get("/")
async def root():
    logger.info("Health check")
    return {"status": "ok", "app": settings.app_name}
