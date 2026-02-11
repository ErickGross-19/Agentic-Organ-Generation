"""
Convenience wrapper for the ODC (Optimized Directed Colonization) backend.

Exposes configuration knobs so you can experiment from a flat dictionary
without touching backend internals.

Usage
-----
    from test.odc_runner import run_odc
    network, stats = run_odc({"seed": 42})
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from generation.core.domain import BoxDomain, CylinderDomain
from generation.core.network import VascularNetwork
from generation.core.types import Point3D
from generation.tissue.hierarchical import TissueLevel, HierarchicalTissueSpec
from generation.ops.odc import run_odc_colonization, ODCResult
from generation.ops.murray_propagation import propagate_murray_radii


DOMAIN_BUILDERS = {
    "cylinder": lambda p: CylinderDomain(
        radius=p.get("domain_radius", 0.005),
        height=p.get("domain_height", 0.010),
        center=Point3D(*p.get("domain_center", [0.0, 0.0, 0.0])),
    ),
    "box": lambda p: BoxDomain(
        x_min=p.get("x_min", -0.005),
        x_max=p.get("x_max", 0.005),
        y_min=p.get("y_min", -0.005),
        y_max=p.get("y_max", 0.005),
        z_min=p.get("z_min", -0.005),
        z_max=p.get("z_max", 0.005),
    ),
}


def _collect_stats(
    network: VascularNetwork,
    odc_result: ODCResult,
    elapsed: float,
) -> Dict[str, Any]:
    n_nodes = len(network.nodes)
    n_segments = len(network.segments)
    n_terminals = sum(1 for n in network.nodes.values() if n.node_type == "terminal")
    n_inlets = sum(1 for n in network.nodes.values() if n.node_type == "inlet")
    n_junctions = sum(1 for n in network.nodes.values() if n.node_type == "junction")

    lengths = [seg.length for seg in network.segments.values()]
    radii = [seg.geometry.mean_radius() for seg in network.segments.values()]

    return {
        "nodes": n_nodes,
        "segments": n_segments,
        "terminals": n_terminals,
        "inlets": n_inlets,
        "junctions": n_junctions,
        "total_length_m": sum(lengths) if lengths else 0.0,
        "mean_segment_length_m": float(np.mean(lengths)) if lengths else 0.0,
        "min_radius_m": float(np.min(radii)) if radii else 0.0,
        "max_radius_m": float(np.max(radii)) if radii else 0.0,
        "mean_radius_m": float(np.mean(radii)) if radii else 0.0,
        "iterations_used": odc_result.iterations_used,
        "levels_reached": odc_result.levels_reached,
        "elapsed_seconds": elapsed,
    }


def run_odc(
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[VascularNetwork, Dict[str, Any]]:
    """
    Generate a vascular network using ODC with hierarchical tissue targeting.

    Parameters (all optional, sensible defaults provided)
    -----------------------------------------------------
    Domain
        domain_type : str        "cylinder" | "box"  (default "cylinder")
        domain_radius : float    Cylinder radius       (default 0.005 m)
        domain_height : float    Cylinder height        (default 0.010 m)
        domain_center : list     [x, y, z] center       (default [0,0,0])

    Inlet
        inlet_position : list    [x, y, z] in meters
        inlet_radius : float     Inlet vessel radius
        vessel_type : str        "arterial" | "venous"

    Tissue levels (hierarchical)
        tissue_levels : list[dict]
            Each dict: {"priority": int, "points": [[x,y,z],...], "label": str,
                        "weight": float, "coverage_threshold": float}
        auto_generate_levels : bool    Auto-generate if no tissue_levels (default True)
        auto_n_levels : int            Number of auto levels (default 3)
        auto_points_per_level : int    Points per auto level (default 200)

    Growth control
        influence_radius : float       (default 0.015)
        kill_radius : float            (default 0.003)
        step_size : float              (default 0.005)
        max_steps : int                (default 500)
        bifurcation_probability : float (default 0.7)
        max_children_per_node : int    (default 2)
        taper_factor : float           (default 0.95)

    Murray propagation
        apply_murray : bool            (default True)
        murray_exponent : float        (default 3.0)
        terminal_radius : float        (default 0.0003)

    Base
        seed : int | None

    Returns
    -------
    network : VascularNetwork
    stats : dict   summary statistics including timing and level coverage
    """
    if params is None:
        params = {}

    domain_type = params.get("domain_type", "cylinder")
    domain = DOMAIN_BUILDERS[domain_type](params)

    tissue_levels_raw = params.get("tissue_levels", None)
    if tissue_levels_raw is not None:
        levels = [TissueLevel.from_dict(ld) for ld in tissue_levels_raw]
        tissue_spec = HierarchicalTissueSpec(levels=levels)
    else:
        from generation.tissue.samplers import generate_hierarchical_from_strategy
        tissue_spec = generate_hierarchical_from_strategy(
            domain,
            n_levels=params.get("auto_n_levels", 3),
            points_per_level=params.get("auto_points_per_level", 200),
            seed=params.get("seed"),
        )

    augment = params.get("augment_with_filler", True)
    if augment:
        from generation.tissue.samplers import sample_hierarchical_tissue_points
        tissue_spec, _ = sample_hierarchical_tissue_points(
            domain,
            tissue_spec,
            augment_with_filler=True,
            filler_n_points=params.get("filler_n_points", 500),
            seed=params.get("seed"),
        )

    bounds = domain.get_bounds()
    inlet_pos = params.get(
        "inlet_position",
        [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, bounds[5]],
    )
    if isinstance(inlet_pos, list):
        inlet_pos = tuple(inlet_pos)

    ports = {
        "inlets": [{
            "position": inlet_pos,
            "radius": params.get("inlet_radius", 0.001),
            "vessel_type": params.get("vessel_type", "arterial"),
            "direction": params.get("inlet_direction", [0, 0, -1]),
        }],
        "outlets": [],
    }

    odc_params = {
        "influence_radius": params.get("influence_radius", 0.015),
        "kill_radius": params.get("kill_radius", 0.003),
        "step_size": params.get("step_size", 0.005),
        "max_steps": params.get("max_steps", 500),
        "bifurcation_probability": params.get("bifurcation_probability", 0.7),
        "max_children_per_node": params.get("max_children_per_node", 2),
        "taper_factor": params.get("taper_factor", 0.95),
        "min_radius": params.get("min_radius", 0.0003),
        "vessel_type": params.get("vessel_type", "arterial"),
        "murray_exponent": params.get("murray_exponent", 3.0),
        "terminal_radius": params.get("terminal_radius", 0.0003),
        "max_stall_steps": params.get("max_stall_steps", 30),
        "smoothing_weight": params.get("smoothing_weight", 0.4),
        "max_curvature_deg": params.get("max_curvature_deg", 45.0),
    }
    if params.get("min_clearance") is not None:
        odc_params["min_clearance"] = params["min_clearance"]

    seed = params.get("seed", None)

    t0 = time.perf_counter()
    odc_result = run_odc_colonization(
        domain=domain,
        tissue_spec=tissue_spec,
        ports=ports,
        params=odc_params,
        seed=seed,
    )

    network = odc_result.network

    if params.get("apply_murray", True):
        propagate_murray_radii(
            network,
            terminal_radius=params.get("terminal_radius", 0.0003),
            gamma=params.get("murray_exponent", 3.0),
        )

    elapsed = time.perf_counter() - t0
    stats = _collect_stats(network, odc_result, elapsed)
    return network, stats
