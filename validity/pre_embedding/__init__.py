"""
Pre-Embedding Validity Checks

These checks validate the structure BEFORE it is embedded into a domain.
They focus on the intrinsic properties of the vascular network and mesh.

Modules:
    - mesh_checks: Watertightness, manifoldness, surface quality
    - graph_checks: Murray's law, branch order, collisions
    - flow_checks: Poiseuille solver, flow plausibility, Reynolds number
"""

from .mesh_checks import (
    check_watertightness,
    check_manifoldness,
    check_surface_quality,
    check_degenerate_faces,
    run_all_mesh_checks,
)
from .graph_checks import (
    check_murrays_law,
    check_branch_order,
    check_collisions,
    check_self_intersections,
    run_all_graph_checks,
)
from .flow_checks import (
    check_flow_plausibility,
    check_reynolds_number,
    check_pressure_monotonicity,
    run_all_flow_checks,
)

__all__ = [
    # Mesh checks
    "check_watertightness",
    "check_manifoldness",
    "check_surface_quality",
    "check_degenerate_faces",
    "run_all_mesh_checks",
    # Graph checks
    "check_murrays_law",
    "check_branch_order",
    "check_collisions",
    "check_self_intersections",
    "run_all_graph_checks",
    # Flow checks
    "check_flow_plausibility",
    "check_reynolds_number",
    "check_pressure_monotonicity",
    "run_all_flow_checks",
]
