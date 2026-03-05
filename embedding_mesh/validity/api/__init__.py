"""
Public API for validity checking and repair.

This module provides the main entry points for validating and repairing
vascular network meshes. All operations are parameterized with policy objects.
"""

from .validate import (
    validate_mesh,
    validate_network,
    validate_artifacts,
    ValidationPolicy,
)
from .repair import (
    repair_mesh,
    RepairPolicy,
)
from .pipeline import (
    validate_repair_validate,
    run_full_pipeline,
)

__all__ = [
    "validate_mesh",
    "validate_network",
    "validate_artifacts",
    "ValidationPolicy",
    "repair_mesh",
    "RepairPolicy",
    "validate_repair_validate",
    "run_full_pipeline",
]
