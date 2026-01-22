"""Rule Models for Priqualis."""

import logging
from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

class RuleSeverity(str, Enum): ERROR = "error"; WARNING = "warning"
class RuleState(str, Enum): SAT = "SAT"; VIOL = "VIOL"; WARN = "WARN"
class AutoFixOperation(str, Enum): ADD_IF_ABSENT = "add_if_absent"; SET = "set"; REMOVE = "remove"; REPLACE = "replace"

class ViolationAction(BaseModel):
    message: str
    autofix_hint: AutoFixOperation | None = None
    suggested_value: str | None = None

class RuleDefinition(BaseModel):
    rule_id: str = Field(..., pattern=r"^R\d{3}$")
    name: str; description: str
    severity: RuleSeverity = RuleSeverity.ERROR
    condition: str
    on_violation: ViolationAction
    jgp_groups: list[str] | None = None
    enabled: bool = True; version: str = "1.0"
    model_config = {"use_enum_values": True}

    @field_validator("condition")
    def validate_cond(cls, v):
        try: compile(v, "<rule>", "eval")
        except SyntaxError as e: raise ValueError(f"Invalid syntax: {e}")
        return v

    @property
    def autofix_hint(self): return self.on_violation.autofix_hint

class RuleResult(BaseModel):
    rule_id: str; case_id: str; state: RuleState
    message: str | None = None; impact_score: float | None = None
    autofix_hint: AutoFixOperation | None = None
    executed_at: datetime = Field(default_factory=datetime.now)
    model_config = {"use_enum_values": True}

    @property
    def is_violation(self): return self.state == RuleState.VIOL
    @property
    def is_warning(self): return self.state == RuleState.WARN
    @property
    def is_satisfied(self): return self.state == RuleState.SAT

class ValidationReport(BaseModel):
    source_file: str | None = None
    total_records: int = 0; total_rules: int = 0
    results: list[RuleResult] = Field(default_factory=list)
    executed_at: datetime = Field(default_factory=datetime.now)

    @property
    def violations(self): return [r for r in self.results if r.state == RuleState.VIOL]
    @property
    def warnings(self): return [r for r in self.results if r.state == RuleState.WARN]
    @property
    def satisfied(self): return [r for r in self.results if r.state == RuleState.SAT]
    @property
    def violation_count(self): return len(self.violations)
    @property
    def warning_count(self): return len(self.warnings)
    @property
    def pass_rate(self): return len(self.satisfied) / len(self.results) if self.results else 1.0

    def summary(self) -> dict:
        return {"total": self.total_records, "checks": len(self.results), "violations": self.violation_count, "rate": f"{self.pass_rate:.1%}"}
