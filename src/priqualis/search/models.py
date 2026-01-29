import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

class SearchSource(str, Enum):
    BM25 = "bm25"; VECTOR = "vector"; HYBRID = "hybrid"

class DiffType(str, Enum):
    MISSING = "missing"; DIFFERENT = "different"; EXTRA = "extra"

class CaseStatus(str, Enum):
    APPROVED = "approved"; REJECTED = "rejected"; PENDING = "pending"

@dataclass(slots=True)
class SearchQuery:
    case_id: str
    text: str
    jgp_code: str | None = None
    icd10_codes: list[str] = field(default_factory=list)
    procedures: list[str] = field(default_factory=list)
    filters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_claim(cls, claim: dict[str, Any]) -> "SearchQuery":
        parts = []
        if jgp := claim.get("jgp_code"): parts.append(f"JGP:{jgp}")
        if icd := claim.get("icd10_main"): parts.append(f"ICD10:{icd}")
        
        icd_sec = claim.get("icd10_secondary", [])
        if isinstance(icd_sec, str): icd_sec = [s.strip() for s in icd_sec.split(",") if s.strip()]
        parts.extend([f"ICD10:{c}" for c in icd_sec])

        procs = claim.get("procedures", [])
        if isinstance(procs, str): procs = [p.strip() for p in procs.split(",") if p.strip()]
        parts.extend([f"PROC:{p}" for p in procs])

        if dept := claim.get("department_code"): parts.append(f"DEPT:{dept}")

        return cls(
            case_id=claim.get("case_id", "unknown"),
            text=" ".join(parts),
            jgp_code=jgp,
            icd10_codes=[icd] + icd_sec if icd else icd_sec,
            procedures=procs
        )

@dataclass(slots=True)
class SearchResult:
    case_id: str
    score: float
    source: SearchSource
    rank: int
    claim_data: dict[str, Any] = field(default_factory=dict)
    def __lt__(self, other): return self.score > other.score

@dataclass(slots=True)
class AttributeDiff:
    field: str; query_value: Any; match_value: Any; diff_type: DiffType

@dataclass(slots=True)
class SimilarCase:
    case_id: str
    similarity_score: float
    attribute_diffs: list[AttributeDiff] = field(default_factory=list)
    jgp_code: str | None = None
    status: CaseStatus = CaseStatus.APPROVED
    claim_data: dict[str, Any] = field(default_factory=dict)
    @property
    def diff_count(self) -> int: return len(self.attribute_diffs)

@dataclass(slots=True)
class IndexingStats:
    total_claims: int = 0
    indexed_bm25: int = 0
    indexed_vector: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    @property
    def success_rate(self) -> float:
        return 1.0 if self.total_claims == 0 else (self.indexed_bm25 + self.indexed_vector) / (2 * self.total_claims)
