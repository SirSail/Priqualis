"""Similar cases search endpoints."""

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api.deps import get_similarity_service
from priqualis.search import SimilarityService

logger = logging.getLogger(__name__)
router = APIRouter()

class SimilarRequest(BaseModel):
    claim: dict[str, Any] = Field(..., description="Query claim")
    top_k: int = Field(5, ge=1, le=20)
    include_diffs: bool = True

class AttributeDiffDTO(BaseModel):
    field: str
    query_value: Any
    match_value: Any
    diff_type: str

class SimilarCaseDTO(BaseModel):
    case_id: str
    similarity_score: float
    jgp_code: str | None = None
    status: str
    attribute_diffs: list[AttributeDiffDTO] = []

class SimilarResponse(BaseModel):
    query_case_id: str
    similar_cases: list[SimilarCaseDTO]
    search_time_ms: float

@router.post("/similar", response_model=SimilarResponse)
async def find_similar(
    request: SimilarRequest, 
    service: SimilarityService = Depends(get_similarity_service)
) -> SimilarResponse:
    start_time = time.perf_counter()
    similar = service.find_similar(claim=request.claim, top_k=request.top_k, include_diffs=request.include_diffs)

    similar_dtos = []
    for case in similar:
        diffs = [
            AttributeDiffDTO(
                field=d.field, query_value=d.query_value, match_value=d.match_value,
                diff_type=d.diff_type.value if hasattr(d.diff_type, "value") else str(d.diff_type),
            )
            for d in case.attribute_diffs
        ]
        similar_dtos.append(SimilarCaseDTO(
            case_id=case.case_id, similarity_score=case.similarity_score, jgp_code=case.jgp_code,
            status=case.status.value if hasattr(case.status, "value") else str(case.status),
            attribute_diffs=diffs,
        ))

    return SimilarResponse(
        query_case_id=request.claim.get("case_id", "unknown"),
        similar_cases=similar_dtos,
        search_time_ms=(time.perf_counter() - start_time) * 1000,
    )

@router.post("/similar/batch")
async def find_similar_batch(
    claims: list[dict[str, Any]], top_k: int = 5,
    service: SimilarityService = Depends(get_similarity_service),
) -> list[SimilarResponse]:
    results = []
    for claim in claims:
        start_time = time.perf_counter()
        similar = service.find_similar(claim, top_k=top_k)
        similar_dtos = [
            SimilarCaseDTO(
                case_id=c.case_id, similarity_score=c.similarity_score, jgp_code=c.jgp_code,
                status=c.status.value if hasattr(c.status, "value") else str(c.status),
            )
            for c in similar
        ]
        results.append(SimilarResponse(
            query_case_id=claim.get("case_id", "unknown"),
            similar_cases=similar_dtos,
            search_time_ms=(time.perf_counter() - start_time) * 1000,
        ))
    return results
