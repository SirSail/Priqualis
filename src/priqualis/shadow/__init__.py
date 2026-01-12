"""
Shadow mode - NFZ rejection tracking and FPA analytics.

This module imports payer rejections and tracks First-Pass Acceptance
rates over time to measure validation effectiveness.
"""

from priqualis.shadow.alerts import (
    Alert,
    AlertConfig,
    AlertManager,
    AnomalyDetector,
)
from priqualis.shadow.fpa import (
    NFZ_ERROR_MAPPING,
    FPAReport,
    FPATracker,
    FPATrend,
    RejectionImporter,
    RejectionRecord,
)

__all__ = [
    # FPA
    "FPAReport",
    "FPATrend",
    "FPATracker",
    "NFZ_ERROR_MAPPING",
    "RejectionImporter",
    "RejectionRecord",
    # Alerts
    "Alert",
    "AlertConfig",
    "AnomalyDetector",
    "AlertManager",
]