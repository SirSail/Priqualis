"""
Vector Search Module (Dense Retrieval).
Handles embedding generation (e5-small) and Qdrant vector storage/Search.
"""
import logging
from functools import lru_cache
from typing import Any

import numpy as np

from priqualis.core.config import get_settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", device: str = "cpu", batch_size: int = 32):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
        self._cache: dict[str, np.ndarray] = {}

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def embed_single(self, text: str) -> np.ndarray:
        if text in self._cache: return self._cache[text]
        embedding = self.model.encode([text], convert_to_numpy=True)[0]
        self._cache[text] = embedding
        return embedding

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=len(texts) > 100)

    def claim_to_text(self, claim: dict[str, Any]) -> str:
        parts = []
        if jgp := claim.get("jgp_code"): parts.append(f"JGP:{jgp}")
        if icd_main := claim.get("icd10_main"): parts.append(f"ICD10:{icd_main}")
        
        icd_secondary = claim.get("icd10_secondary", [])
        if icd_secondary:
            if isinstance(icd_secondary, str): icd_secondary = [s.strip() for s in icd_secondary.split(",") if s.strip()]
            for code in icd_secondary[:5]: parts.append(f"ICD10:{code}")

        procedures = claim.get("procedures", [])
        if procedures:
            if isinstance(procedures, str): procedures = [p.strip() for p in procedures.split(",") if p.strip()]
            for proc in procedures[:5]: parts.append(f"PROC:{proc}")

        if dept := claim.get("department_code"): parts.append(f"DEPT:{dept}")
        return " ".join(parts) if parts else "EMPTY"

    def clear_cache(self) -> None:
        self._cache.clear()

class VectorStore:
    def __init__(self, collection: str = "claims", host: str | None = None, port: int = 6333, api_key: str | None = None, in_memory: bool = False):
        self.collection, self.host, self.port, self.api_key, self.in_memory = collection, host, port, api_key, in_memory
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient
            if self.in_memory or self.host is None:
                logger.info("Using in-memory Qdrant")
                self._client = QdrantClient(":memory:")
            else:
                logger.info("Connecting to Qdrant at %s:%d", self.host, self.port)
                self._client = QdrantClient(host=self.host, port=self.port, api_key=self.api_key)
        return self._client

    def create_collection(self, vector_size: int = 384) -> None:
        from qdrant_client.models import Distance, HnswConfigDiff, VectorParams
        if any(c.name == self.collection for c in self.client.get_collections().collections): return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
        )

    def upsert(self, case_id: str, vector: np.ndarray, payload: dict[str, Any]) -> None:
        from qdrant_client.models import PointStruct
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=hash(case_id) % (2**63), vector=vector.tolist(), payload={"case_id": case_id, **payload})],
        )

    def upsert_batch(self, items: list[tuple[str, np.ndarray, dict]], batch_size: int = 100) -> int:
        from qdrant_client.models import PointStruct
        points = [PointStruct(id=hash(cid) % (2**63), vector=v.tolist(), payload={"case_id": cid, **p}) for cid, v, p in items]
        for i in range(0, len(points), batch_size):
            self.client.upsert(collection_name=self.collection, points=points[i : i + batch_size])
        return len(points)

    def search(self, vector: np.ndarray, top_k: int = 50, filters: dict[str, Any] | None = None) -> list[tuple[str, float, dict[str, Any]]]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        query_filter = Filter(must=[FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()]) if filters else None
        
        results = self.client.query_points(collection_name=self.collection, query=vector.tolist(), limit=top_k, query_filter=query_filter).points
        return [(r.payload.get("case_id", "unknown"), r.score, r.payload) for r in results]

    def delete(self, case_ids: list[str]) -> None:
        from qdrant_client.models import PointIdsList
        self.client.delete(collection_name=self.collection, points_selector=PointIdsList(points=[hash(cid) % (2**63) for cid in case_ids]))

    def count(self) -> int:
        return self.client.get_collection(self.collection).points_count

@lru_cache
def get_embedding_service() -> EmbeddingService:
    s = get_settings()
    return EmbeddingService(model_name=getattr(s, "embedding_model", "intfloat/multilingual-e5-small"), device=getattr(s, "embedding_device", "cpu"))

@lru_cache
def get_vector_store() -> VectorStore:
    s = get_settings()
    return VectorStore(
        collection=getattr(s, "qdrant_collection", "claims"),
        host=getattr(s, "qdrant_host", None),
        port=getattr(s, "qdrant_port", 6333),
        in_memory=getattr(s, "qdrant_in_memory", True),
    )
