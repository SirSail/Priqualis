"""
Hybrid Search for Priqualis.

Combines BM25 sparse retrieval with vector dense retrieval.
"""

import logging
from typing import Literal

import numpy as np

from priqualis.search.bm25 import BM25Index
from priqualis.search.models import SearchQuery, SearchResult, SearchSource
from priqualis.search.vector import EmbeddingService, VectorStore

logger = logging.getLogger(__name__)


# =============================================================================
# Fusion Functions
# =============================================================================


def reciprocal_rank_fusion(
    results_list: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion (RRF).

    Combines multiple ranked lists using: score = sum(1 / (k + rank_i))

    Args:
        results_list: List of ranked result lists [(case_id, score), ...]
        k: RRF constant (default 60)

    Returns:
        Fused results sorted by RRF score descending
    """
    scores: dict[str, float] = {}

    for results in results_list:
        for rank, (case_id, _) in enumerate(results, start=1):
            if case_id not in scores:
                scores[case_id] = 0.0
            scores[case_id] += 1.0 / (k + rank)

    # Sort by fused score
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused


def linear_fusion(
    bm25_results: list[tuple[str, float]],
    vector_results: list[tuple[str, float]],
    alpha: float = 0.5,
) -> list[tuple[str, float]]:
    """
    Linear interpolation fusion.

    score = alpha * bm25_norm + (1 - alpha) * vector_norm

    Args:
        bm25_results: BM25 results
        vector_results: Vector results
        alpha: BM25 weight (0-1)

    Returns:
        Fused results sorted by score descending
    """
    # Normalize scores to 0-1 range
    def normalize(results: list[tuple[str, float]]) -> dict[str, float]:
        if not results:
            return {}
        scores = [s for _, s in results]
        min_s, max_s = min(scores), max(scores)
        range_s = max_s - min_s if max_s != min_s else 1.0
        return {cid: (s - min_s) / range_s for cid, s in results}

    bm25_norm = normalize(bm25_results)
    vector_norm = normalize(vector_results)

    # Combine all case IDs
    all_ids = set(bm25_norm.keys()) | set(vector_norm.keys())

    # Calculate fused scores
    fused = []
    for case_id in all_ids:
        bm25_score = bm25_norm.get(case_id, 0.0)
        vector_score = vector_norm.get(case_id, 0.0)
        combined = alpha * bm25_score + (1 - alpha) * vector_score
        fused.append((case_id, combined))

    # Sort by score
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused


# =============================================================================
# Hybrid Search
# =============================================================================


class HybridSearch:
    """
    Combine BM25 and vector search with fusion.

    Pipeline:
    1. BM25 retrieval → top-200 candidates
    2. Vector retrieval → top-50 candidates
    3. RRF or Linear fusion
    4. Return top-k
    """

    def __init__(
        self,
        bm25_index: BM25Index,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        alpha: float = 0.5,
    ):
        """
        Initialize hybrid search.

        Args:
            bm25_index: BM25 sparse index
            vector_store: Qdrant vector store
            embedding_service: Embedding generator
            alpha: BM25 weight for linear fusion (0-1)
        """
        self.bm25 = bm25_index
        self.vectors = vector_store
        self.embeddings = embedding_service
        self.alpha = alpha
        self._claim_cache: dict[str, dict] = {}

    def set_claim_cache(self, claims: dict[str, dict]) -> None:
        """Set claim data cache for result enrichment."""
        self._claim_cache = claims

    def search(
        self,
        query: SearchQuery,
        top_k: int = 50,
        fusion: Literal["rrf", "linear"] = "rrf",
        bm25_candidates: int = 200,
        vector_candidates: int = 50,
    ) -> list[SearchResult]:
        """
        Execute hybrid search.

        Args:
            query: Search query
            top_k: Number of results to return
            fusion: Fusion method ("rrf" or "linear")
            bm25_candidates: Number of BM25 candidates
            vector_candidates: Number of vector candidates

        Returns:
            List of SearchResult sorted by score
        """
        logger.debug("Hybrid search for: %s", query.text[:100])

        # 1. BM25 retrieval
        bm25_results: list[tuple[str, float]] = []
        if self.bm25.is_built:
            bm25_results = self.bm25.search(query.text, top_k=bm25_candidates)
            logger.debug("BM25 returned %d results", len(bm25_results))

        # 2. Vector retrieval
        vector_results: list[tuple[str, float]] = []
        try:
            query_embedding = self.embeddings.embed_single(query.text)
            vector_raw = self.vectors.search(
                query_embedding,
                top_k=vector_candidates,
                filters={"jgp_code": query.jgp_code} if query.jgp_code else None,
            )
            vector_results = [(cid, score) for cid, score, _ in vector_raw]
            logger.debug("Vector returned %d results", len(vector_results))
        except Exception as e:
            logger.warning("Vector search failed: %s", e)

        # 3. Fusion
        if fusion == "rrf":
            fused = reciprocal_rank_fusion([bm25_results, vector_results])
        else:
            fused = linear_fusion(bm25_results, vector_results, self.alpha)

        # 4. Build SearchResult objects
        results = []
        for rank, (case_id, score) in enumerate(fused[:top_k], start=1):
            # Determine source
            in_bm25 = case_id in {r[0] for r in bm25_results}
            in_vector = case_id in {r[0] for r in vector_results}

            if in_bm25 and in_vector:
                source = SearchSource.HYBRID
            elif in_bm25:
                source = SearchSource.BM25
            else:
                source = SearchSource.VECTOR

            results.append(SearchResult(
                case_id=case_id,
                score=score,
                source=source,
                rank=rank,
                claim_data=self._claim_cache.get(case_id, {}),
            ))

        logger.info("Hybrid search returned %d results", len(results))
        return results

    def search_bm25_only(
        self,
        query: SearchQuery,
        top_k: int = 50,
    ) -> list[SearchResult]:
        """BM25-only search (fallback if vectors unavailable)."""
        if not self.bm25.is_built:
            return []

        results = self.bm25.search(query.text, top_k=top_k)
        return [
            SearchResult(
                case_id=case_id,
                score=score,
                source=SearchSource.BM25,
                rank=rank,
                claim_data=self._claim_cache.get(case_id, {}),
            )
            for rank, (case_id, score) in enumerate(results, start=1)
        ]

    def search_vector_only(
        self,
        query: SearchQuery,
        top_k: int = 50,
    ) -> list[SearchResult]:
        """Vector-only search."""
        query_embedding = self.embeddings.embed_single(query.text)
        results = self.vectors.search(query_embedding, top_k=top_k)

        return [
            SearchResult(
                case_id=case_id,
                score=score,
                source=SearchSource.VECTOR,
                rank=rank,
                claim_data=payload,
            )
            for rank, (case_id, score, payload) in enumerate(results, start=1)
        ]
