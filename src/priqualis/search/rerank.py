"""Reranker for Priqualis."""

import logging
from typing import Any
from priqualis.search.models import SearchResult

logger = logging.getLogger(__name__)

class Reranker:
    """Cross-encoder reranker."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str = "cpu"):
        self.model_name, self.device, self._model = model_name, device, None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            logger.info("Loading cross-encoder: %s", self.model_name)
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        if not candidates: return []
        
        pairs = [(query, self._claim_to_text(c.claim_data) if c.claim_data else f"Case {c.case_id}") for c in candidates]
        scores = self.model.predict(pairs)
        
        scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [SearchResult(c.case_id, float(s), c.source, i, c.claim_data) for i, (c, s) in enumerate(scored[:top_k], 1)]

    def _claim_to_text(self, claim: dict[str, Any]) -> str:
        parts = []
        if jgp := claim.get("jgp_code"): parts.append(f"JGP:{jgp}")
        if icd := claim.get("icd10_main"): parts.append(f"ICD10:{icd}")
        if procs := claim.get("procedures", []):
            parts.append(f"PROC:{procs[:3]}" if isinstance(procs, list) else f"PROC:{procs}")
        return " ".join(parts) if parts else "EMPTY"

def get_reranker(enabled: bool = True) -> Reranker | None:
    return Reranker() if enabled else None
