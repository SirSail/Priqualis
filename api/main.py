"""Priqualis API - FastAPI application for healthcare claim validation."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import autofix, reports, similar, validate
from priqualis.core.config import get_settings

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Priqualis API")
    yield
    logger.info("🛑 Shutting down Priqualis API")

_settings = get_settings()

app = FastAPI(
    title=_settings.api_title,
    description="Pre-submission compliance validator for healthcare claims",
    version=_settings.api_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(validate.router, prefix="/api/v1", tags=["Validation"])
app.include_router(similar.router, prefix="/api/v1", tags=["Similarity"])
app.include_router(autofix.router, prefix="/api/v1", tags=["AutoFix"])
app.include_router(reports.router, prefix="/api/v1", tags=["Reports"])

@app.get("/")
async def root():
    return {"name": _settings.api_title, "version": _settings.api_version, "docs": "/docs"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": _settings.api_version}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=_settings.api_host, port=_settings.api_port, reload=_settings.is_development)
