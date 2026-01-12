"""
Priqualis Reports Module.

Generates validation reports in various formats (Markdown, PDF, JSON).
"""

from .generator import (
    ReportConfig,
    ReportGenerator,
    generate_batch_report,
)

__all__ = [
    "ReportGenerator",
    "ReportConfig",
    "generate_batch_report",
]
