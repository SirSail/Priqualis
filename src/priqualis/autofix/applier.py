"""Patch Applier for Priqualis."""

import copy, logging, uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
import yaml
from priqualis.autofix.generator import AuditEntry, Patch, PatchOperation
from priqualis.core.exceptions import AutoFixError

logger = logging.getLogger(__name__)

class ApplyMode(str, Enum): DRY_RUN = "dry-run"; COMMIT = "commit"

class PatchApplier:
    """Applies patches with audit trail."""

    def __init__(self, audit_dir: Path | None = None):
        self.audit_dir = audit_dir
        self._applied_patches = []

    def apply(self, patch: Patch, record: dict, mode: ApplyMode | str = "dry-run") -> dict:
        mode = ApplyMode(mode) if isinstance(mode, str) else mode
        rec = record.model_dump() if hasattr(record, "model_dump") else record
        
        if rec.get("case_id") != patch.case_id: raise AutoFixError(f"ID mismatch: {patch.case_id} vs {rec.get('case_id')}")
        working = copy.deepcopy(rec) if mode == ApplyMode.DRY_RUN else rec

        for op in patch.changes: self._apply_op(op, working)
        logger.info("Applied patch %s (%s)", patch.rule_id, mode.value)
        return working

    def _apply_op(self, op: PatchOperation, rec: dict):
        parts = op.field.split(".")
        tgt = rec
        for p in parts[:-1]: tgt = tgt.setdefault(p, {})
        key = parts[-1]
        
        if op.op == "add_if_absent" and not tgt.get(key): tgt[key] = op.value
        elif op.op == "set" or op.op == "replace": tgt[key] = op.value
        elif op.op == "remove" and key in tgt: del tgt[key]

    def create_audit(self, patch: Patch, user: str = "system", approved: bool = False) -> AuditEntry:
        entry = AuditEntry(patch_id=str(uuid.uuid4())[:8], case_id=patch.case_id, rule_id=patch.rule_id, user=user, changes=patch.changes, approved=approved)
        self._applied_patches.append(entry)
        if self.audit_dir:
            (p := self.audit_dir / f"audit_{entry.applied_at.strftime('%Y%m%d_%H%M%S')}_{entry.patch_id}.yaml").parent.mkdir(parents=True, exist_ok=True)
            try: p.write_text(yaml.dump(entry.model_dump(mode='json'), allow_unicode=True))
            except Exception as e: logger.error("Audit save failed: %s", e)
        return entry

    def apply_batch(self, patches: list[Patch], records: dict, mode: str = "dry-run", user: str = "system") -> dict:
        res, mode = {}, ApplyMode(mode)
        for p in patches:
            if r := records.get(p.case_id):
                try:
                    res[p.case_id] = self.apply(p, r, mode)
                    if mode == ApplyMode.COMMIT: self.create_audit(p, user)
                except Exception as e: logger.error("Apply failed %s: %s", p.case_id, e)
        return res

def apply_patch(patch: Patch, record: dict, mode: str = "dry-run") -> dict:
    return PatchApplier().apply(patch, record, mode)
