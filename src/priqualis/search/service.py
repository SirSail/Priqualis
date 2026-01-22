"""Similarity Service for Priqualis."""

import logging
from priqualis.search.bm25 import BM25Index
from priqualis.search.hybrid import HybridSearch
from priqualis.search.models import AttributeDiff, CaseStatus, DiffType, SearchQuery, SimilarCase
from priqualis.search.rerank import Reranker
from priqualis.search.vector import EmbeddingService, VectorStore

logger = logging.getLogger(__name__)

class SimilarityService:
    """Service for finding similar approved cases."""

    def __init__(self, bm25_index: BM25Index | None = None, vector_store: VectorStore | None = None, 
                 embedding_service: EmbeddingService | None = None, reranker: Reranker | None = None, alpha: float = 0.5):
        self.bm25 = bm25_index or BM25Index()
        self.vectors = vector_store or VectorStore(in_memory=True)
        self.embeddings = embedding_service or EmbeddingService()
        self.reranker = reranker
        self.hybrid = HybridSearch(self.bm25, self.vectors, self.embeddings, alpha)
        self._approved_cases = {}

    def load_approved_cases(self, cases: list[dict]) -> int:
        self._approved_cases = {c["case_id"]: c for c in cases if "case_id" in c}
        self.hybrid.set_claim_cache(self._approved_cases)
        return len(self._approved_cases)

    def find_similar(self, claim: dict, top_k: int = 5, include_diffs: bool = True) -> list[SimilarCase]:
        query = SearchQuery.from_claim(claim)
        results = self.hybrid.search(query, top_k=top_k * 2)
        
        if self.reranker and len(results) > top_k: results = self.reranker.rerank(query.text, results, top_k=top_k)
        else: results = results[:top_k]

        cases = []
        for r in results:
            match = r.claim_data or self._approved_cases.get(r.case_id, {})
            cases.append(SimilarCase(r.case_id, r.score, self._compute_diffs(claim, match) if include_diffs and match else [], 
                                   match.get("jgp_code"), CaseStatus.APPROVED, match))
        return cases

    def _compute_diffs(self, query: dict, match: dict) -> list[AttributeDiff]:
        diffs = []
        for f in ["icd10_main", "icd10_secondary", "jgp_code", "procedures", "admission_mode", "department_code"]:
            q, m = query.get(f), match.get(f)
            if not q and not m: continue
            
            dt = DiffType.EXTRA if (q and not m) else (DiffType.MISSING if (not q and m) else (DiffType.DIFFERENT if q != m else None))
            if dt: diffs.append(AttributeDiff(f, q, m, dt))
        return diffs

class ClaimIndexer:
    """Index approved claims."""

    def __init__(self, bm25_index: BM25Index, vector_store: VectorStore, embedding_service: EmbeddingService):
        self.bm25, self.vectors, self.embeddings = bm25_index, vector_store, embedding_service

    def index_claims(self, claims: list[dict]) -> dict:
        import time; start = time.time()
        
        bm25_docs = [(c.get("case_id"), self.embeddings.claim_to_text(c)) for c in claims if c.get("case_id")]
        if not bm25_docs: return {"total": 0, "error": "No valid claims"}
        
        self.bm25.build(bm25_docs)
        
        try:
            self.vectors.create_collection(self.embeddings.dimension)
            items = [(cid, emb, claims[i]) for i, (cid, txt) in enumerate(bm25_docs) for emb in [self.embeddings.embed_batch([t for _, t in bm25_docs])[i]]]
            self.vectors.upsert_batch(items)
        except Exception as e: logger.error("Vector index error: %s", e)

        return {"total": len(claims), "duration": round(time.time()-start, 2)}

def create_similarity_service(rerank_enabled: bool = False, in_memory: bool = True) -> SimilarityService:
    from priqualis.search.rerank import get_reranker
    return SimilarityService(reranker=get_reranker(rerank_enabled), vector_store=VectorStore(in_memory=in_memory))
