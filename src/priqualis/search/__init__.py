"""
Search module for Priqualis.

Provides hybrid similarity search (BM25 + Vector) for healthcare claims.
"""

from priqualis.search.bm25 import BM25Index, SimpleTokenizer
from priqualis.search.hybrid import HybridSearch, linear_fusion, reciprocal_rank_fusion
from priqualis.search.models import (
    AttributeDiff,
    CaseStatus,
    DiffType,
    IndexingStats,
    SearchQuery,
    SearchResult,
    SearchSource,
    SimilarCase,
)
from priqualis.search.rerank import Reranker, get_reranker
from priqualis.search.service import (
    ClaimIndexer,
    SimilarityService,
    create_similarity_service,
)
from priqualis.search.vector import (
    EmbeddingService,
    VectorStore,
    get_embedding_service,
    get_vector_store,
)

__all__ = [
    # Models
    "AttributeDiff",
    "CaseStatus",
    "DiffType",
    "IndexingStats",
    "SearchQuery",
    "SearchResult",
    "SearchSource",
    "SimilarCase",
    # BM25
    "BM25Index",
    "SimpleTokenizer",
    # Vector
    "EmbeddingService",
    "VectorStore",
    "get_embedding_service",
    "get_vector_store",
    # Hybrid
    "HybridSearch",
    "reciprocal_rank_fusion",
    "linear_fusion",
    # Rerank
    "Reranker",
    "get_reranker",
    # Service
    "ClaimIndexer",
    "SimilarityService",
    "create_similarity_service",
]