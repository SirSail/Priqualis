"""
AutoFix Router.

Endpoints for patch generation and application.
"""

import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from api.deps import get_patch_applier, get_patch_generator
from priqualis.autofix import PatchApplier, PatchGenerator

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================


class PatchOperationDTO(BaseModel):
    """Patch operation."""

    op: str
    field: str
    value: Any = None
    old_value: Any = None


class PatchDTO(BaseModel):
    """Patch definition."""

    case_id: str
    rule_id: str
    changes: list[PatchOperationDTO]
    rationale: str
    confidence: float = 0.5
    risk_note: str = "Formal verification remains with provider."


class GeneratePatchRequest(BaseModel):
    """Request to generate patches."""

    violations: list[dict[str, Any]] = Field(..., min_length=1)
    records: dict[str, dict[str, Any]] = Field(..., description="case_id → record map")


class ApplyPatchRequest(BaseModel):
    """Request to apply a patch."""

    case_id: str
    record: dict[str, Any]
    patch: PatchDTO
    mode: Literal["dry-run", "commit"] = "dry-run"


class AuditEntryDTO(BaseModel):
    """Audit entry."""

    patch_id: str
    case_id: str
    rule_id: str
    user: str
    applied_at: str


class ApplyPatchResponse(BaseModel):
    """Response from applying a patch."""

    case_id: str
    applied: bool
    modified_record: dict[str, Any]
    audit_entry: AuditEntryDTO | None = None


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/autofix/generate", response_model=list[PatchDTO])
async def generate_patches(
    request: GeneratePatchRequest,
    generator: PatchGenerator = Depends(get_patch_generator),
) -> list[PatchDTO]:
    """
    Generate patches for violations.

    Args:
        request: Violations and records
        generator: Injected patch generator

    Returns:
        List of generated patches
    """
    from priqualis.rules.models import RuleResult, RuleState

    # Convert violations to RuleResult
    rule_results = []
    for v in request.violations:
        rule_results.append(RuleResult(
            rule_id=v.get("rule_id", "R001"),
            case_id=v.get("case_id", "unknown"),
            state=RuleState.VIOL,
            message=v.get("message"),
            autofix_hint=v.get("autofix_hint"),
        ))

    # Generate patches
    patches = generator.generate_batch(rule_results, request.records)

    # Convert to DTOs
    return [
        PatchDTO(
            case_id=p.case_id,
            rule_id=p.rule_id,
            changes=[
                PatchOperationDTO(
                    op=c.op if isinstance(c.op, str) else c.op.value,
                    field=c.field,
                    value=c.value,
                    old_value=c.old_value,
                )
                for c in p.changes
            ],
            rationale=p.rationale,
            confidence=p.confidence,
            risk_note=p.risk_note,
        )
        for p in patches
    ]


@router.post("/autofix/apply", response_model=ApplyPatchResponse)
async def apply_patch(
    request: ApplyPatchRequest,
    applier: PatchApplier = Depends(get_patch_applier),
) -> ApplyPatchResponse:
    """
    Apply a patch to a record.

    Args:
        request: Patch and target record
        applier: Injected patch applier

    Returns:
        Modified record and audit info
    """
    from priqualis.autofix.generator import Patch, PatchOperation
    from priqualis.rules.models import AutoFixOperation

    # Convert DTO to Patch
    patch = Patch(
        case_id=request.patch.case_id,
        rule_id=request.patch.rule_id,
        changes=[
            PatchOperation(
                op=AutoFixOperation(c.op),
                field=c.field,
                value=c.value,
                old_value=c.old_value,
            )
            for c in request.patch.changes
        ],
        rationale=request.patch.rationale,
        confidence=request.patch.confidence,
        risk_note=request.patch.risk_note,
    )

    # Apply
    modified = applier.apply(patch, request.record, mode=request.mode)

    # Create audit entry for commits
    audit_dto = None
    if request.mode == "commit":
        entry = applier.create_audit_entry(patch, user="api")
        audit_dto = AuditEntryDTO(
            patch_id=entry.patch_id,
            case_id=entry.case_id,
            rule_id=entry.rule_id,
            user=entry.user,
            applied_at=entry.applied_at.isoformat(),
        )

    return ApplyPatchResponse(
        case_id=request.case_id,
        applied=request.mode == "commit",
        modified_record=modified,
        audit_entry=audit_dto,
    )


@router.get("/autofix/audit/{case_id}")
async def get_audit_trail(
    case_id: str,
    applier: PatchApplier = Depends(get_patch_applier),
) -> list[AuditEntryDTO]:
    """Get audit trail for a case."""
    entries = [e for e in applier.applied_patches if e.case_id == case_id]
    return [
        AuditEntryDTO(
            patch_id=e.patch_id,
            case_id=e.case_id,
            rule_id=e.rule_id,
            user=e.user,
            applied_at=e.applied_at.isoformat(),
        )
        for e in entries
    ]
