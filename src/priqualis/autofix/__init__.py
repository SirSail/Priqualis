"""
AutoFix module for Priqualis.

Generates and applies patches to fix rule violations.
"""

from priqualis.autofix.models import (
    AuditEntry,
    Patch,
    PatchOperation,
)
from priqualis.autofix.generator import (
    DEFAULT_VALUES,
    SUGGESTED_FIXES,
    PatchGenerator,
)
from priqualis.autofix.applier import (
    ApplyMode,
    PatchApplier,
    apply_patch,
    export_patches_yaml,
)

__all__ = [
    "Patch", "PatchOperation", "AuditEntry",
    "PatchGenerator", "DEFAULT_VALUES", "SUGGESTED_FIXES",
    "PatchApplier", "ApplyMode", "apply_patch", "export_patches_yaml",
]