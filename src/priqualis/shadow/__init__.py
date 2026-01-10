"""
Shadow mode - NFZ rejection tracking and FPA analytics.

This module imports payer rejections and tracks First-Pass Acceptance
rates over time to measure validation effectiveness.
"""

from priqualis.shadow.fpa import (
    FPAReport,
    FPATrend,
    FPATracker,
    NFZ_ERROR_MAPPING,
    RejectionImporter,
    RejectionRecord,
)

__all__ = [
    "FPAReport",
    "FPATrend",
    "FPATracker",
    "NFZ_ERROR_MAPPING",
    "RejectionImporter",
    "RejectionRecord",
]