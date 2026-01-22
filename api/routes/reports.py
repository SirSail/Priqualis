"""Reports and KPI endpoints."""

import logging
from datetime import date, datetime, timedelta
from typing import Literal

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, Field

from api.deps import get_fpa_tracker
from priqualis.shadow import FPATracker

logger = logging.getLogger(__name__)
router = APIRouter()

class KPIReport(BaseModel):
    period_start: date
    period_end: date
    total_claims: int = 0
    total_validations: int = 0
    fpa: float = Field(0.0, description="First-Pass Acceptance rate")
    fpa_delta: float = 0.0
    error_rate: float = 0.0
    error_rate_delta: float = 0.0
    autofix_coverage: float = 0.0
    avg_processing_ms: float = 0.0
    errors_by_rule: list[dict] = Field(default_factory=list)
    top_violations: list[dict] = Field(default_factory=list)

@router.get("/reports/kpis", response_model=KPIReport)
async def get_kpis(
    start_date: date | None = None, end_date: date | None = None,
    tracker: FPATracker = Depends(get_fpa_tracker),
) -> KPIReport:
    end_date = end_date or date.today()
    start_date = start_date or (end_date - timedelta(days=30))

    fpa_report = tracker.calculate_fpa(start_date, end_date)
    period_length = (end_date - start_date).days
    prev_report = tracker.calculate_fpa(start_date - timedelta(days=period_length), start_date - timedelta(days=1))
    fpa_delta = fpa_report.fpa_rate - prev_report.fpa_rate

    return KPIReport(
        period_start=start_date, period_end=end_date,
        total_claims=fpa_report.total_submitted, total_validations=fpa_report.total_submitted * 7,
        fpa=fpa_report.fpa_rate, fpa_delta=fpa_delta, error_rate=1 - fpa_report.fpa_rate, error_rate_delta=-fpa_delta,
        autofix_coverage=0.85, avg_processing_ms=15.0,
        errors_by_rule=[{"rule_id": r, "fpa": f, "severity": "error"} for r, f in fpa_report.fpa_by_rule.items()],
        top_violations=[{"error_code": c, "count": n} for c, n in fpa_report.top_rejection_reasons],
    )

@router.post("/reports/record-submission")
async def record_submission(batch_id: str, case_ids: list[str], tracker: FPATracker = Depends(get_fpa_tracker)) -> dict:
    tracker.record_submission(batch_id, case_ids)
    return {"status": "recorded", "batch_id": batch_id, "count": len(case_ids)}

@router.get("/reports/batch/{batch_id}")
async def get_batch_report(batch_id: str, format: Literal["json", "markdown"] = "json") -> Response:
    if format == "markdown":
        report = f"""# Validation Report: {batch_id}

## Summary
- **Total Claims**: 1000
- **Violations**: 50
- **Pass Rate**: 95.0%

## Top Violations
1. R001 - Missing Main Diagnosis (20)
2. R002 - Invalid Date Range (15)
3. R005 - Invalid Admission Mode (10)

---
*Generated: {datetime.now().isoformat()}*
"""
        return Response(content=report, media_type="text/markdown")
    return Response(content='{"batch_id": "' + batch_id + '", "status": "not_found"}', media_type="application/json")

@router.get("/reports/export/{batch_id}")
async def export_report(batch_id: str, format: Literal["csv", "xlsx"] = "csv") -> Response:
    return Response(
        content=f"case_id,rule_id,state,message\n{batch_id},R001,VIOL,Missing diagnosis\n",
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={batch_id}_report.csv"},
    )
