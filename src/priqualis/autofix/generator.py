"""Patch Generator for Priqualis."""

import logging
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field
from priqualis.rules.models import AutoFixOperation, RuleResult

logger = logging.getLogger(__name__)

class PatchOperation(BaseModel):
    op: AutoFixOperation
    field: str
    value: Any = None
    old_value: Any | None = None
    model_config = {"use_enum_values": True}

class Patch(BaseModel):
    case_id: str
    rule_id: str
    changes: list[PatchOperation]
    rationale: str
    risk_note: str = "Formal verification remains with provider."
    confidence: float = 0.5
    created_at: datetime = Field(default_factory=datetime.now)
    applied: bool = False

    @property
    def is_safe(self) -> bool: return self.confidence >= 0.8
    def to_yaml_dict(self) -> dict: return self.model_dump(mode='json')

class AuditEntry(BaseModel):
    patch_id: str; case_id: str; rule_id: str; user: str
    changes: list[PatchOperation] = Field(default_factory=list)
    applied_at: datetime = Field(default_factory=datetime.now); approved: bool = False

DEFAULT_VALUES = {
    "icd10_main": "Z00.0", "icd10_secondary": [], "jgp_code": "A01", 
    "jgp_name": "General Medicine", "tariff_value": 0.0, "admission_mode": "planned", 
    "department_code": "4000", "procedures": []
}

SUGGESTED_FIXES = {
    "R001": {"field": "icd10_main", "value": "Z00.0", "rationale": "Added placeholder diagnosis."},
    "R002": {"field": "discharge_date", "value": None, "rationale": "Set discharge to admission date."},
    "R003": {"field": "jgp_code", "value": "A01", "rationale": "Added default JGP."},
    "R005": {"field": "admission_mode", "value": "planned", "rationale": "Set mode to planned."},
    "R006": {"field": "department_code", "value": "4000", "rationale": "Added default dept."},
}

class PatchGenerator:
    """Generates patches for rule violations."""

    def __init__(self, default_values: dict | None = None, suggested_fixes: dict | None = None):
        self.defaults = default_values or DEFAULT_VALUES
        self.suggested = suggested_fixes or SUGGESTED_FIXES

    def generate(self, violation: RuleResult, record: dict) -> Patch | None:
        if violation.is_satisfied or not violation.autofix_hint: return None
        
        ops = self._gen_ops(violation, record)
        if not ops: return None

        fix = self.suggested.get(violation.rule_id, {})
        return Patch(
            case_id=violation.case_id, rule_id=violation.rule_id, changes=ops,
            rationale=fix.get("rationale", f"Fix for {violation.rule_id}"),
            confidence=self._conf(violation, ops)
        )

    def _gen_ops(self, v: RuleResult, rec: dict) -> list[PatchOperation]:
        hint = v.autofix_hint
        fix = self.suggested.get(v.rule_id, {})
        f, val = fix.get("field"), fix.get("value")
        if not f: return []

        old = rec.get(f)
        if hint == "add_if_absent" and not old:
            return [PatchOperation(op="add_if_absent", field=f, value=val or self.defaults.get(f), old_value=old)]
        elif hint == "set":
            if f == "discharge_date" and val is None: val = rec.get("admission_date")
            return [PatchOperation(op="set", field=f, value=val, old_value=old)] if val is not None else []
        elif hint == "remove" and old is not None:
            return [PatchOperation(op="remove", field=f, value=None, old_value=old)]
        return []

    def _conf(self, v: RuleResult, ops: list[PatchOperation]) -> float:
        base = 0.5 + (0.1 if len(ops) == 1 else 0) + (0.2 if ops and ops[0].op == "add_if_absent" else 0)
        return min(base + (0.1 if v.rule_id in self.suggested else 0), 1.0)

    def generate_batch(self, violations: list[RuleResult], records: dict) -> list[Patch]:
        patches = []
        for v in violations:
            if p := self.generate(v, records.get(v.case_id, {})): patches.append(p)
        logger.info("Generated %d patches", len(patches))
        return patches
