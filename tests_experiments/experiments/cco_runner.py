"""
Convenience wrapper for the CCO Hybrid backend (includes NLP optimization).

Exposes every configuration knob from CCOConfig and BackendConfig
so you can experiment from a flat dictionary without touching backend internals.

Usage
-----
    from cco_runner import run_cco, run_cco_dual_tree
    network, stats = run_cco({"use_nlp_optimization": True, "nlp_solver": "SLSQP"})
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from generation.core.domain import BoxDomain, CylinderDomain, EllipsoidDomain
from generation.core.network import VascularNetwork
from generation.core.types import Point3D
from generation.backends.cco_hybrid_backend import (
    CCOHybridBackend,
    CCOConfig,
)


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
    "ellipsoid": lambda p: EllipsoidDomain(
        semi_axis_a=p.get("semi_a", 0.005),
        semi_axis_b=p.get("semi_b", 0.005),
        semi_axis_c=p.get("semi_c", 0.005),
        center=Point3D(*p.get("domain_center", [0.0, 0.0, 0.0])),
    ),
}


def _build_config(params: Dict[str, Any]) -> CCOConfig:
    return CCOConfig(
        # --- BackendConfig base ---
        seed=params.get("seed", None),
        min_segment_length=params.get("min_segment_length", 0.0005),
        max_segment_length=params.get("max_segment_length", 0.020),
        min_radius=params.get("min_radius", 0.0001),
        min_terminal_separation=params.get("min_terminal_separation", 0.0005),
        check_collisions=params.get("check_collisions", True),
        collision_clearance=params.get("collision_clearance", 0.0001),
        # --- Murray's law ---
        murray_exponent=params.get("murray_exponent", 3.0),
        # --- Cost function weights ---
        cost_length_weight=params.get("cost_length_weight", 1.0),
        cost_radius_weight=params.get("cost_radius_weight", 1.0),
        boundary_penalty_weight=params.get("boundary_penalty_weight", 10.0),
        # --- Optimization ---
        optimization_grid_resolution=params.get("optimization_grid_resolution", 10),
        candidate_edges_k=params.get("candidate_edges_k", 50),
        # --- Accelerators ---
        use_partial_binding=params.get("use_partial_binding", True),
        use_collision_triage=params.get("use_collision_triage", True),
        # --- Collision ---
        collision_check_enabled=params.get("collision_check_enabled", True),
        # --- Geometry parameters ---
        candidate_search_radius=params.get("candidate_search_radius", 0.05),
        boundary_penalty_threshold=params.get("boundary_penalty_threshold", 0.002),
        initial_radius_taper=params.get("initial_radius_taper", 0.8),
        fallback_murray_split_ratio=params.get("fallback_murray_split_ratio", 0.8),
        outlet_end_radius_taper=params.get("outlet_end_radius_taper", 0.9),
        single_child_taper=params.get("single_child_taper", 0.9),
        anastomosis_max_length=params.get("anastomosis_max_length", 0.015),
        # --- Dual-tree ---
        min_terminal_separation_same_type=params.get("min_terminal_separation_same_type", None),
        min_terminal_separation_cross_type=params.get("min_terminal_separation_cross_type", None),
        encourage_av_proximity=params.get("encourage_av_proximity", False),
        # --- NLP optimization ---
        use_nlp_optimization=params.get("use_nlp_optimization", False),
        nlp_solver=params.get("nlp_solver", "SLSQP"),
        nlp_tolerance=params.get("nlp_tolerance", 1e-6),
        max_nlp_iterations=params.get("max_nlp_iterations", 100),
        nlp_use_grid_initial_guess=params.get("nlp_use_grid_initial_guess", True),
        nlp_grid_resolution_for_guess=params.get("nlp_grid_resolution_for_guess", 5),
        # --- Trifurcation ---
        enable_trifurcation=params.get("enable_trifurcation", False),
        trifurcation_cost_threshold=params.get("trifurcation_cost_threshold", 0.8),
        # --- Generation control ---
        max_consecutive_failures=params.get("max_consecutive_failures", 50),
        default_inlet_radius=params.get("default_inlet_radius", 0.002),
    )


def _collect_stats(network: VascularNetwork, elapsed: float) -> Dict[str, Any]:
    n_nodes = len(network.nodes)
    n_segments = len(network.segments)
    n_terminals = sum(1 for n in network.nodes.values() if n.node_type == "terminal")
    n_inlets = sum(1 for n in network.nodes.values() if n.node_type == "inlet")
    n_junctions = sum(1 for n in network.nodes.values() if n.node_type == "junction")
    n_arterial = sum(1 for n in network.nodes.values() if n.vessel_type == "arterial")
    n_venous = sum(1 for n in network.nodes.values() if n.vessel_type == "venous")

    lengths = [seg.length for seg in network.segments.values()]
    radii = [seg.geometry.mean_radius() for seg in network.segments.values()]

    return {
        "nodes": n_nodes,
        "segments": n_segments,
        "terminals": n_terminals,
        "inlets": n_inlets,
        "junctions": n_junctions,
        "arterial_nodes": n_arterial,
        "venous_nodes": n_venous,
        "total_length_m": sum(lengths) if lengths else 0.0,
        "mean_segment_length_m": float(np.mean(lengths)) if lengths else 0.0,
        "min_radius_m": float(np.min(radii)) if radii else 0.0,
        "max_radius_m": float(np.max(radii)) if radii else 0.0,
        "mean_radius_m": float(np.mean(radii)) if radii else 0.0,
        "elapsed_seconds": elapsed,
    }


def run_cco(
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[VascularNetwork, Dict[str, Any]]:
    """
    Generate a single-tree vascular network using CCO hybrid.

    Parameters (all optional, sensible defaults provided)
    -----------------------------------------------------
    Domain
        domain_type : str        "cylinder" | "box" | "ellipsoid"  (default "cylinder")
        domain_radius : float    Cylinder / ellipsoid radius       (default 0.005 m)
        domain_height : float    Cylinder height                   (default 0.010 m)
        domain_center : list     [x, y, z] center                  (default [0,0,0])
        x_min … z_max : float    Box bounds
        semi_a/b/c : float       Ellipsoid semi-axes

    Inlet
        inlet_position : list    [x, y, z] in meters               (default top of domain)
        inlet_radius : float     Inlet vessel radius in meters     (default 0.001)
        vessel_type : str        "arterial" | "venous"              (default "arterial")

    Murray's law
        murray_exponent : float  (default 3.0)

    Cost function
        cost_length_weight : float        (default 1.0)
        cost_radius_weight : float        (default 1.0)
        boundary_penalty_weight : float   (default 10.0)

    Optimization
        optimization_grid_resolution : int   grid res for bifurcation search  (default 10)
        candidate_edges_k : int              max candidate edges              (default 50)

    Accelerators
        use_partial_binding : bool     partial binding optimization  (default True)
        use_collision_triage : bool    collision triage              (default True)

    Collision
        collision_clearance : float       min clearance in m  (default 0.0001)
        collision_check_enabled : bool    (default True)

    Geometry
        candidate_search_radius : float        (default 0.05)
        boundary_penalty_threshold : float     (default 0.002)
        initial_radius_taper : float           (default 0.8)
        fallback_murray_split_ratio : float    (default 0.8)
        outlet_end_radius_taper : float        (default 0.9)
        single_child_taper : float             (default 0.9)
        anastomosis_max_length : float         (default 0.015)

    NLP optimization  (gradient-based instead of grid search)
        use_nlp_optimization : bool              (default False)
        nlp_solver : str                         "SLSQP" | "trust-constr" | "L-BFGS-B"
        nlp_tolerance : float                    (default 1e-6)
        max_nlp_iterations : int                 (default 100)
        nlp_use_grid_initial_guess : bool        (default True)
        nlp_grid_resolution_for_guess : int      (default 5)

    Trifurcation (3-way splits)
        enable_trifurcation : bool               (default False)
        trifurcation_cost_threshold : float      (default 0.8)

    Generation control
        max_consecutive_failures : int   (default 50)
        default_inlet_radius : float     (default 0.002)

    Base config
        seed : int | None
        min_segment_length : float   (default 0.0005)
        max_segment_length : float   (default 0.020)
        min_radius : float           (default 0.0001)
        min_terminal_separation : float (default 0.0005)
        check_collisions : bool      (default True)

    Other
        num_outlets : int   target terminal count (default 50)
        rng_seed : int      separate RNG seed

    Returns
    -------
    network : VascularNetwork
    stats : dict   summary statistics including timing
    """
    if params is None:
        params = {}

    domain_type = params.get("domain_type", "cylinder")
    domain = DOMAIN_BUILDERS[domain_type](params)
    config = _build_config(params)
    backend = CCOHybridBackend()

    bounds = domain.get_bounds()
    inlet_pos = np.array(params.get(
        "inlet_position",
        [
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            bounds[5],
        ],
    ))
    inlet_rad = params.get("inlet_radius", 0.001)
    num_outlets = params.get("num_outlets", 50)
    rng_seed = params.get("rng_seed", params.get("seed", None))

    t0 = time.perf_counter()
    network = backend.generate(
        domain=domain,
        num_outlets=num_outlets,
        inlet_position=inlet_pos,
        inlet_radius=inlet_rad,
        vessel_type=params.get("vessel_type", "arterial"),
        config=config,
        rng_seed=rng_seed,
    )
    elapsed = time.perf_counter() - t0
    stats = _collect_stats(network, elapsed)
    return network, stats


def run_cco_dual_tree(
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[VascularNetwork, Dict[str, Any]]:
    """
    Generate a dual arterial-venous network using CCO hybrid.

    Extra parameters (in addition to all single-tree params)
    --------------------------------------------------------
    arterial_outlets : int       (default 30)
    venous_outlets : int         (default 30)
    arterial_inlet : list        [x,y,z] position (default top-center of domain)
    venous_outlet : list         [x,y,z] position (default bottom-center of domain)
    arterial_radius : float      (default 0.001)
    venous_radius : float        (default 0.001)
    create_anastomoses : bool    (default False)
    num_anastomoses : int        (default 0)

    Dual-tree specific
        min_terminal_separation_same_type : float | None
        min_terminal_separation_cross_type : float | None
        encourage_av_proximity : bool   (default False)
    """
    if params is None:
        params = {}

    domain_type = params.get("domain_type", "cylinder")
    domain = DOMAIN_BUILDERS[domain_type](params)
    config = _build_config(params)
    backend = CCOHybridBackend()

    bounds = domain.get_bounds()
    cx = (bounds[0] + bounds[1]) / 2
    cy = (bounds[2] + bounds[3]) / 2

    arterial_inlet = np.array(params.get("arterial_inlet", [cx, cy, bounds[5]]))
    venous_outlet = np.array(params.get("venous_outlet", [cx, cy, bounds[4]]))
    rng_seed = params.get("rng_seed", params.get("seed", None))

    t0 = time.perf_counter()
    network = backend.generate_dual_tree(
        domain=domain,
        arterial_outlets=params.get("arterial_outlets", 30),
        venous_outlets=params.get("venous_outlets", 30),
        arterial_inlet=arterial_inlet,
        venous_outlet=venous_outlet,
        arterial_radius=params.get("arterial_radius", 0.001),
        venous_radius=params.get("venous_radius", 0.001),
        config=config,
        rng_seed=rng_seed,
        create_anastomoses=params.get("create_anastomoses", False),
        num_anastomoses=params.get("num_anastomoses", 0),
    )
    elapsed = time.perf_counter() - t0
    stats = _collect_stats(network, elapsed)
    return network, stats
