"""
Validity checks for void meshes and embedded domains.

This module provides additional validity checks for void meshes,
including component analysis, domain containment, and shell integrity.
"""

from .void_checks import (
    check_void_components,
    check_void_inside_domain,
    check_shell_watertight,
    check_minimum_diameter,
    report_diameter_shrink_and_drift,
    run_void_validity_checks,
    CheckStatus,
    CheckResult,
    VoidValidityReport,
    DiameterReport,
)

__all__ = [
    "check_void_components",
    "check_void_inside_domain",
    "check_shell_watertight",
    "check_minimum_diameter",
    "report_diameter_shrink_and_drift",
    "run_void_validity_checks",
    "CheckStatus",
    "CheckResult",
    "VoidValidityReport",
    "DiameterReport",
]
