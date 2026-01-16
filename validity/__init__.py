"""
Agentic Organ Generation - Validity Checking Library (Part B)

This module provides comprehensive validity checking for organ structures,
organized into two main stages:

1. Pre-Embedding Checks: Validate the structure BEFORE embedding into a domain
   - Mesh watertightness, manifoldness, surface quality
   - Murray's law compliance at bifurcations
   - Flow plausibility (Poiseuille solver, Reynolds number)
   - Collision/self-intersection detection

2. Post-Embedding Checks: Validate the structure AFTER embedding into a domain
   - Outlet openness (ports accessible from exterior)
   - Channel continuity (fluid can flow through)
   - Trapped fluid detection
   - Printability constraints (min channel diameter, wall thickness)

Main Entry Points:
    - run_pre_embedding_validation(): Validate structure before embedding
    - run_post_embedding_validation(): Validate embedded structure
    - validate_and_repair_geometry(): Full pipeline with repair

Example:
    >>> from validity import run_pre_embedding_validation, run_post_embedding_validation
    >>> from validity.pre_embedding import mesh_checks, flow_checks
    >>>
    >>> # Pre-embedding validation
    >>> pre_report = run_pre_embedding_validation(
    ...     mesh_path="structure.stl",
    ...     network=vascular_network,
    ... )
    >>> print(f"Pre-embedding status: {pre_report.status}")
    >>>
    >>> # Post-embedding validation
    >>> post_report = run_post_embedding_validation(
    ...     embedded_mesh_path="domain_with_void.stl",
    ...     manufacturing_config={"min_channel_diameter": 0.5, "min_wall_thickness": 0.3}
    ... )
    >>> print(f"Post-embedding status: {post_report.status}")
"""

from .orchestrators import (
    run_pre_embedding_validation,
    run_post_embedding_validation,
    ValidationReport,
    ValidationConfig,
)
from .runner import (
    run_validity_checks,
    ValidityReport,
    CheckResult,
)

# validate_and_repair_geometry requires pymeshfix which is an optional dependency.
# We use lazy import to avoid breaking imports when pymeshfix is not installed.
# Users who need mesh repair should import it explicitly:
#   from validity.pipeline import validate_and_repair_geometry
# or install pymeshfix: pip install pymeshfix


def __getattr__(name):
    """Lazy import for optional dependencies."""
    if name == "validate_and_repair_geometry":
        from .pipeline import validate_and_repair_geometry
        return validate_and_repair_geometry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Orchestrators
    "run_pre_embedding_validation",
    "run_post_embedding_validation",
    "ValidationReport",
    "ValidationConfig",
    # G1 FIX: Canonical validity runner
    "run_validity_checks",
    "ValidityReport",
    "CheckResult",
    # Legacy pipeline (requires pymeshfix - lazy imported)
    "validate_and_repair_geometry",
]
