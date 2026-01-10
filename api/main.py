"""
Priqualis API - Main Application.

FastAPI application for healthcare claim validation and compliance.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import validate, similar, autofix, reports

logger = logging.getLogger(__name__)


# =============================================================================
# Lifespan
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    # Startup
    logger.info("🚀 Starting Priqualis API")
    yield
    # Shutdown
    logger.info("🛑 Shutting down Priqualis API")


# =============================================================================
# Application
# =============================================================================


app = FastAPI(
    title="Priqualis API",
    description="Pre-submission compliance validator for healthcare claims",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# =============================================================================
# Middleware
# =============================================================================


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Routers
# =============================================================================


app.include_router(validate.router, prefix="/api/v1", tags=["Validation"])
app.include_router(similar.router, prefix="/api/v1", tags=["Similarity"])
app.include_router(autofix.router, prefix="/api/v1", tags=["AutoFix"])
app.include_router(reports.router, prefix="/api/v1", tags=["Reports"])


# =============================================================================
# Root Endpoints
# =============================================================================


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Priqualis API",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "0.1.0",
    }


# =============================================================================
# Run with uvicorn
# =============================================================================


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
