"""Autofix Models."""

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field
from priqualis.rules.models import AutoFixOperation

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
