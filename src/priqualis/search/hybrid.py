"""Hybrid Search for Priqualis."""

import logging
from typing import Literal
from priqualis.search.bm25 import BM25Index
from priqualis.search.models import SearchQuery, SearchResult, SearchSource
from priqualis.search.vector import EmbeddingService, VectorStore

logger = logging.getLogger(__name__)

def reciprocal_rank_fusion(results_list: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    scores = {}
    for res in results_list:
        for r, (cid, _) in enumerate(res, 1): scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + r)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def linear_fusion(bm25_res: list[tuple[str, float]], vec_res: list[tuple[str, float]], alpha: float = 0.5) -> list[tuple[str, float]]:
    def norm(res):
        if not res: return {}
        vals = [s for _, s in res]
        mi, ma = min(vals), max(vals)
        r = ma - mi if ma != mi else 1.0
        return {c: (s - mi) / r for c, s in res}

    bm, vec = norm(bm25_res), norm(vec_res)
    fused = [(cid, alpha * bm.get(cid, 0.0) + (1 - alpha) * vec.get(cid, 0.0)) for cid in set(bm) | set(vec)]
    return sorted(fused, key=lambda x: x[1], reverse=True)

class HybridSearch:
    """Combine BM25 and vector search."""

    def __init__(self, bm25_index: BM25Index, vector_store: VectorStore, embedding_service: EmbeddingService, alpha: float = 0.5):
        self.bm25, self.vectors, self.embeddings, self.alpha = bm25_index, vector_store, embedding_service, alpha
        self._claim_cache = {}

    def set_claim_cache(self, claims: dict[str, dict]) -> None: self._claim_cache = claims

    def search(self, query: SearchQuery, top_k: int = 50, fusion: Literal["rrf", "linear"] = "rrf", bm25_c: int = 200, vector_c: int = 50):
        bm25_res = self.bm25.search(query.text, top_k=bm25_c) if self.bm25.is_built else []
        
        vec_res = []
        try:
            vec_res = [(c, s) for c, s, _ in self.vectors.search(self.embeddings.embed_single(query.text), top_k=vector_c, filters={"jgp_code": query.jgp_code} if query.jgp_code else None)]
        except Exception as e: logger.warning("Vector search failed: %s", e)

        fused = reciprocal_rank_fusion([bm25_res, vec_res]) if fusion == "rrf" else linear_fusion(bm25_res, vec_res, self.alpha)
        
        results = []
        for r, (cid, s) in enumerate(fused[:top_k], 1):
            src = SearchSource.HYBRID if (cid in dict(bm25_res) and cid in dict(vec_res)) else (SearchSource.BM25 if cid in dict(bm25_res) else SearchSource.VECTOR)
            results.append(SearchResult(cid, s, src, r, self._claim_cache.get(cid, {})))
            
        return results

    def search_bm25_only(self, query: SearchQuery, top_k: int = 50):
        return [SearchResult(cid, s, SearchSource.BM25, r, self._claim_cache.get(cid, {})) for r, (cid, s) in enumerate(self.bm25.search(query.text, top_k=top_k), 1)] if self.bm25.is_built else []

    def search_vector_only(self, query: SearchQuery, top_k: int = 50):
        return [SearchResult(c, s, SearchSource.VECTOR, r, p) for r, (c, s, p) in enumerate(self.vectors.search(self.embeddings.embed_single(query.text), top_k=top_k), 1)]
