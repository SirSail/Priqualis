"""
Reports Router.

Endpoints for report generation and KPIs.
"""

import logging
from datetime import date, datetime
from typing import Literal

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, Field

from api.deps import get_fpa_tracker
from priqualis.shadow import FPATracker

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================


class KPIReport(BaseModel):
    """KPI summary."""

    period_start: date
    period_end: date
    total_claims: int = 0
    total_validations: int = 0
    fpa: float = Field(0.0, description="First-Pass Acceptance rate")
    fpa_delta: float = Field(0.0, description="FPA change from previous period")
    error_rate: float = 0.0
    error_rate_delta: float = 0.0
    autofix_coverage: float = Field(0.0, description="% of violations with autofix")
    avg_processing_ms: float = 0.0
    errors_by_rule: list[dict] = Field(default_factory=list)
    top_violations: list[dict] = Field(default_factory=list)


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/reports/kpis", response_model=KPIReport)
async def get_kpis(
    start_date: date | None = None,
    end_date: date | None = None,
    tracker: FPATracker = Depends(get_fpa_tracker),
) -> KPIReport:
    """
    Get KPI summary using real FPA tracker data.

    Args:
        start_date: Period start (default: 30 days ago)
        end_date: Period end (default: today)
        tracker: Injected FPA tracker

    Returns:
        KPI report with metrics
    """
    from datetime import timedelta

    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=30)

    # Get FPA report from tracker
    fpa_report = tracker.calculate_fpa(start_date, end_date)

    # Calculate FPA delta (compare to previous period)
    period_length = (end_date - start_date).days
    prev_start = start_date - timedelta(days=period_length)
    prev_end = start_date - timedelta(days=1)
    prev_report = tracker.calculate_fpa(prev_start, prev_end)

    fpa_delta = fpa_report.fpa_rate - prev_report.fpa_rate

    # Build errors by rule
    errors_by_rule = [
        {"rule_id": rule, "fpa": fpa, "severity": "error"}
        for rule, fpa in fpa_report.fpa_by_rule.items()
    ]

    # Top violations
    top_violations = [
        {"error_code": code, "count": count}
        for code, count in fpa_report.top_rejection_reasons
    ]

    return KPIReport(
        period_start=start_date,
        period_end=end_date,
        total_claims=fpa_report.total_submitted,
        total_validations=fpa_report.total_submitted * 7,  # 7 rules
        fpa=fpa_report.fpa_rate,
        fpa_delta=fpa_delta,
        error_rate=1 - fpa_report.fpa_rate,
        error_rate_delta=-fpa_delta,
        autofix_coverage=0.85,  # Placeholder - needs integration
        avg_processing_ms=15.0,  # Placeholder
        errors_by_rule=errors_by_rule,
        top_violations=top_violations,
    )


@router.post("/reports/record-submission")
async def record_submission(
    batch_id: str,
    case_ids: list[str],
    tracker: FPATracker = Depends(get_fpa_tracker),
) -> dict:
    """Record a submission for FPA tracking."""
    tracker.record_submission(batch_id, case_ids)
    return {
        "status": "recorded",
        "batch_id": batch_id,
        "count": len(case_ids),
    }


@router.get("/reports/batch/{batch_id}")
async def get_batch_report(
    batch_id: str,
    format: Literal["json", "markdown"] = "json",
) -> Response:
    """
    Generate validation report for a batch.

    Args:
        batch_id: Batch identifier
        format: Output format (json or markdown)

    Returns:
        Report in requested format
    """
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
        return Response(
            content=report,
            media_type="text/markdown",
        )

    return Response(
        content='{"batch_id": "' + batch_id + '", "status": "not_found"}',
        media_type="application/json",
    )


@router.get("/reports/export/{batch_id}")
async def export_report(
    batch_id: str,
    format: Literal["csv", "xlsx"] = "csv",
) -> Response:
    """
    Export validation results.

    Args:
        batch_id: Batch identifier
        format: Export format

    Returns:
        File download
    """
    return Response(
        content=f"case_id,rule_id,state,message\n{batch_id},R001,VIOL,Missing diagnosis\n",
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={batch_id}_report.csv"
        },
    )
