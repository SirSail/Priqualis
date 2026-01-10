"""
API Dependencies.

Dependency injection for FastAPI services.
"""

import logging
from functools import lru_cache
from pathlib import Path

from priqualis.autofix import PatchApplier, PatchGenerator
from priqualis.core.config import get_settings, Settings
from priqualis.etl import ClaimImporter
from priqualis.rules import RuleEngine
from priqualis.search import (
    BM25Index,
    EmbeddingService,
    SimilarityService,
    VectorStore,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Settings
# =============================================================================


@lru_cache
def get_api_settings() -> Settings:
    """Get cached application settings."""
    return get_settings()


# =============================================================================
# Core Services
# =============================================================================


@lru_cache
def get_rule_engine() -> RuleEngine:
    """Get cached rule engine instance."""
    settings = get_api_settings()
    rules_path = Path(settings.rules_config_path)
    logger.info("Initializing RuleEngine from %s", rules_path)
    return RuleEngine(rules_path)


@lru_cache
def get_importer() -> ClaimImporter:
    """Get claim importer instance."""
    return ClaimImporter(strict=False)


# =============================================================================
# AutoFix Services
# =============================================================================


@lru_cache
def get_patch_generator() -> PatchGenerator:
    """Get patch generator instance."""
    return PatchGenerator()


@lru_cache
def get_patch_applier() -> PatchApplier:
    """Get patch applier instance."""
    settings = get_api_settings()
    audit_dir = Path(settings.data_processed_path) / "audit" if settings.data_processed_path else None
    return PatchApplier(audit_dir=audit_dir)


# =============================================================================
# Search Services
# =============================================================================


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Get embedding service instance."""
    return EmbeddingService(
        model_name="intfloat/multilingual-e5-small",
        device="cpu",
    )


@lru_cache
def get_similarity_service() -> SimilarityService:
    """Get similarity service instance."""
    embeddings = get_embedding_service()
    return SimilarityService(
        bm25_index=BM25Index(),
        vector_store=VectorStore(in_memory=True),
        embedding_service=embeddings,
    )


# =============================================================================
# Cleanup
# =============================================================================


def clear_caches() -> None:
    """Clear all LRU caches (for testing)."""
    get_api_settings.cache_clear()
    get_rule_engine.cache_clear()
    get_importer.cache_clear()
    get_patch_generator.cache_clear()
    get_patch_applier.cache_clear()
    get_embedding_service.cache_clear()
    get_similarity_service.cache_clear()
