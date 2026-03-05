"""
Structured reporting for validation and generation runs.

This module provides standardized JSON-serializable report structures
for tracking inputs, outputs, and intermediate states.
"""

from .run_report import (
    RunReport,
    create_run_report,
    save_run_report,
)
from .drift_report import (
    DriftReport,
    compute_drift_report,
)

__all__ = [
    "RunReport",
    "create_run_report",
    "save_run_report",
    "DriftReport",
    "compute_drift_report",
]
