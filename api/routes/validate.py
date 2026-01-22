"""Validation endpoints."""

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, File, UploadFile
from pydantic import BaseModel, Field

from api.deps import get_fpa_tracker, get_importer, get_rule_engine
from priqualis.etl import ClaimImporter
from priqualis.etl.schemas import ClaimBatch
from priqualis.rules import RuleEngine
from priqualis.shadow import FPATracker

logger = logging.getLogger(__name__)
router = APIRouter()

class ValidateBatchRequest(BaseModel):
    claims: list[dict[str, Any]] = Field(..., min_length=1)
    rules: list[str] | None = Field(None, description="Specific rule IDs to run")

class ValidationResultDTO(BaseModel):
    rule_id: str
    case_id: str
    state: str
    message: str | None = None
    impact_score: float | None = None

class ValidateBatchResponse(BaseModel):
    batch_id: str
    total_claims: int
    violations: int
    warnings: int
    passed: int
    pass_rate: float
    results: list[ValidationResultDTO]
    processing_time_ms: float

@router.post("/validate", response_model=ValidateBatchResponse)
async def validate_batch(
    request: ValidateBatchRequest,
    rule_engine: RuleEngine = Depends(get_rule_engine),
    fpa_tracker: FPATracker = Depends(get_fpa_tracker),
) -> ValidateBatchResponse:
    start_time = time.perf_counter()
    from priqualis.etl.schemas import ClaimRecord

    records = []
    for claim in request.claims:
        try:
            records.append(ClaimRecord(**claim))
        except Exception as e:
            logger.warning("Skipping invalid claim: %s", e)

    batch = ClaimBatch(records=records)
    report = rule_engine.validate(batch)

    results = [
        ValidationResultDTO(
            rule_id=r.rule_id, case_id=r.case_id,
            state=r.state if isinstance(r.state, str) else r.state.value,
            message=r.message, impact_score=r.impact_score,
        )
        for r in report.results
    ]

    processing_time = (time.perf_counter() - start_time) * 1000
    batch_id = f"batch_{int(time.time())}"
    fpa_tracker.record_submission(batch_id, [r.case_id for r in records])

    return ValidateBatchResponse(
        batch_id=batch_id, total_claims=report.total_records,
        violations=report.violation_count, warnings=report.warning_count,
        passed=len(report.satisfied), pass_rate=report.pass_rate,
        results=results, processing_time_ms=processing_time,
    )

@router.post("/validate/file", response_model=ValidateBatchResponse)
async def validate_file(
    file: UploadFile = File(...),
    rule_engine: RuleEngine = Depends(get_rule_engine),
    importer: ClaimImporter = Depends(get_importer),
) -> ValidateBatchResponse:
    import tempfile
    from pathlib import Path

    start_time = time.perf_counter()
    suffix = Path(file.filename).suffix if file.filename else ".csv"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        batch = importer.import_file(tmp_path)
        report = rule_engine.validate(batch)

        results = [
            ValidationResultDTO(
                rule_id=r.rule_id, case_id=r.case_id,
                state=r.state if isinstance(r.state, str) else r.state.value,
                message=r.message, impact_score=r.impact_score,
            )
            for r in report.results
        ]

        return ValidateBatchResponse(
            batch_id=f"batch_{int(time.time())}", total_claims=report.total_records,
            violations=report.violation_count, warnings=report.warning_count,
            passed=len(report.satisfied), pass_rate=report.pass_rate,
            results=results, processing_time_ms=(time.perf_counter() - start_time) * 1000,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

@router.get("/validate/rules")
async def list_rules(rule_engine: RuleEngine = Depends(get_rule_engine)) -> list[dict]:
    return [
        {"rule_id": r.rule_id, "name": r.name, "description": r.description, "severity": r.severity, "enabled": r.enabled}
        for r in rule_engine.rules
    ]
