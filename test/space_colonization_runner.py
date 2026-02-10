"""
Convenience wrapper for the Space Colonization backend.

Exposes every configuration knob from SpaceColonizationConfig and BackendConfig
so you can experiment from a flat dictionary without touching backend internals.

Usage
-----
    from space_colonization_runner import run_space_colonization
    network, stats = run_space_colonization({"num_attractors": 2000, "step_size": 0.003})
"""

import sys, os, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from generation.core.domain import BoxDomain, CylinderDomain, EllipsoidDomain
from generation.core.network import VascularNetwork
from generation.core.types import Point3D
from generation.backends.space_colonization_backend import (
    SpaceColonizationBackend,
    SpaceColonizationConfig,
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


def _build_config(params: Dict[str, Any]) -> SpaceColonizationConfig:
    return SpaceColonizationConfig(
        seed=params.get("seed", None),
        min_segment_length=params.get("min_segment_length", 0.0005),
        max_segment_length=params.get("max_segment_length", 0.020),
        min_radius=params.get("min_radius", 0.0001),
        min_terminal_separation=params.get("min_terminal_separation", 0.0005),
        check_collisions=params.get("check_collisions", True),
        collision_clearance=params.get("collision_clearance", 0.0002),
        attraction_distance=params.get("attraction_distance", 0.010),
        kill_distance=params.get("kill_distance", 0.002),
        step_size=params.get("step_size", 0.002),
        num_attractors=params.get("num_attractors", 1000),
        max_iterations=params.get("max_iterations", 500),
        branch_angle_deg=params.get("branch_angle_deg", 30.0),
        multi_inlet_mode=params.get("multi_inlet_mode", "blended"),
        collision_merge_distance=params.get("collision_merge_distance", 0.0003),
        max_inlets=params.get("max_inlets", 10),
        multi_inlet_blend_sigma=params.get("multi_inlet_blend_sigma", 0.0),
        directional_bias=params.get("directional_bias", 0.5),
        max_deviation_deg=params.get("max_deviation_deg", 60.0),
        taper_factor=params.get("taper_factor", 0.95),
        encourage_bifurcation=params.get("encourage_bifurcation", False),
        max_children_per_node=params.get("max_children_per_node", 2),
        bifurcation_probability=params.get("bifurcation_probability", 0.7),
        min_attractions_for_bifurcation=params.get("min_attractions_for_bifurcation", 3),
        bifurcation_angle_threshold_deg=params.get("bifurcation_angle_threshold_deg", 40.0),
        max_steps=params.get("max_steps", 100),
        progress=params.get("progress", False),
        kdtree_rebuild_tip_every=params.get("kdtree_rebuild_tip_every", 1),
        kdtree_rebuild_all_nodes_every=params.get("kdtree_rebuild_all_nodes_every", 10),
        kdtree_rebuild_all_nodes_min_new_nodes=params.get("kdtree_rebuild_all_nodes_min_new_nodes", 5),
        stall_steps_per_inlet=params.get("stall_steps_per_inlet", 10),
        interleaving_strategy=params.get("interleaving_strategy", "round_robin"),
        partitioned_directional_bias=params.get("partitioned_directional_bias", 1.0),
        partitioned_max_deviation_deg=params.get("partitioned_max_deviation_deg", 30.0),
        partitioned_cone_angle_deg=params.get("partitioned_cone_angle_deg", 30.0),
        partitioned_cylinder_radius=params.get("partitioned_cylinder_radius", 0.001),
    )


def _collect_stats(network: VascularNetwork, elapsed: float) -> Dict[str, Any]:
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
        "elapsed_seconds": elapsed,
    }


def run_space_colonization(
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[VascularNetwork, Dict[str, Any]]:
    """
    Generate a vascular network using space colonization.

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

    Multi-inlet  (pass ``inlets`` list of dicts instead of single inlet)
        inlets : list[dict]      Each dict has position, radius, direction (optional)
        multi_inlet_mode : str   "blended" | "partitioned_xy" | "forest"

    Growth control
        num_attractors : int            (default 1000)
        attraction_distance : float     influence radius in m (default 0.010)
        kill_distance : float           kill radius in m      (default 0.002)
        step_size : float               growth step in m      (default 0.002)
        max_iterations : int            outer loop limit      (default 500)
        max_steps : int                 inner step limit      (default 100)
        branch_angle_deg : float        (default 30.0)
        directional_bias : float        0–1 (default 0.5)
        max_deviation_deg : float       (default 60.0)

    Bifurcation
        encourage_bifurcation : bool                  (default False)
        max_children_per_node : int                   (default 2)
        bifurcation_probability : float               (default 0.7)
        min_attractions_for_bifurcation : int         (default 3)
        bifurcation_angle_threshold_deg : float       (default 40.0)

    Radius / taper
        min_radius : float       minimum vessel radius (default 0.0001)
        taper_factor : float     per-generation taper  (default 0.95)

    Blended multi-inlet
        multi_inlet_blend_sigma : float   (default 0 → auto)
        collision_merge_distance : float  (default 0.0003)

    Partitioned multi-inlet
        partitioned_directional_bias : float      (default 1.0)
        partitioned_max_deviation_deg : float     (default 30.0)
        partitioned_cone_angle_deg : float        (default 30.0)
        partitioned_cylinder_radius : float       (default 0.001)

    Performance
        kdtree_rebuild_tip_every : int                 (default 1)
        kdtree_rebuild_all_nodes_every : int           (default 10)
        kdtree_rebuild_all_nodes_min_new_nodes : int   (default 5)
        stall_steps_per_inlet : int                    (default 10)
        interleaving_strategy : str                    "round_robin" | "weighted"
        progress : bool                                show tqdm bar (default False)

    Base
        seed : int | None
        min_segment_length : float   (default 0.0005)
        max_segment_length : float   (default 0.020)
        min_terminal_separation : float (default 0.0005)
        check_collisions : bool      (default True)
        collision_clearance : float  (default 0.0002)

    Other
        num_outlets : int   target outlet count, scales attractors (default 50)
        rng_seed : int      separate RNG seed for the run

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
    backend = SpaceColonizationBackend()

    inlets = params.get("inlets", None)
    num_outlets = params.get("num_outlets", 50)
    rng_seed = params.get("rng_seed", params.get("seed", None))

    t0 = time.perf_counter()

    if inlets and len(inlets) > 1:
        network = backend.generate_multi_inlet(
            domain=domain,
            num_outlets=num_outlets,
            inlets=inlets,
            vessel_type=params.get("vessel_type", "arterial"),
            config=config,
            rng_seed=rng_seed,
        )
    else:
        if inlets and len(inlets) == 1:
            inlet_pos = np.array(inlets[0].get("position", [0.0, 0.0, 0.0]))
            inlet_rad = inlets[0].get("radius", 0.001)
        else:
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


def run_space_colonization_dual_tree(
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[VascularNetwork, Dict[str, Any]]:
    """
    Generate a dual arterial-venous network using space colonization.

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
    """
    if params is None:
        params = {}

    domain_type = params.get("domain_type", "cylinder")
    domain = DOMAIN_BUILDERS[domain_type](params)
    config = _build_config(params)
    backend = SpaceColonizationBackend()

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
