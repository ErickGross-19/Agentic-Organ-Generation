"""
Bifurcating tree vascular network generation for MorphoStruct.

This module wraps AOG's scaffold top-down backend to create regular
bifurcating trees with configurable branching patterns and radius tapering.
"""

import manifold3d as m3d
import numpy as np
from typing import Tuple, Dict, Any
import logging

from generation.backends.scaffold_topdown_backend import (
    ScaffoldTopDownBackend,
    ScaffoldTopDownConfig,
    CollisionOnlineConfig,
)
from generation.core.domain import CylinderDomain

from .mesh_adapter import network_to_manifold, calculate_network_stats

logger = logging.getLogger(__name__)


def generate_bifurcating_tree_from_dict(
    params: Dict[str, Any]
) -> Tuple[m3d.Manifold, Dict[str, Any]]:
    """
    Generate regular bifurcating tree from parameter dictionary.

    This function follows MorphoStruct's generator convention:
    - Takes a dict of parameters
    - Returns (manifold, stats) tuple

    Parameters
    ----------
    params : dict
        Parameter dictionary with keys matching BifurcatingTreeParams model:
        - root_position: List[float] - Root position [x,y,z] (m)
        - root_direction: List[float] - Root direction (normalized)
        - root_radius: float - Root radius (m)
        - branching_levels: int - Branching depth
        - branches_per_node: int - Branches per bifurcation
        - branching_angle_deg: float - Branching angle (deg)
        - segment_length: float - Segment length (m)
        - taper_segment_length: bool - Taper segment length per level
        - length_taper_factor: float - Length taper factor
        - radius_mode: str - Radius calculation mode
        - murray_exponent: float - Murray's law exponent
        - min_terminal_radius: float - Min terminal radius (m)
        - add_variation: bool - Add random variation
        - angle_variation_deg: float - Angle variation ±deg
        - length_variation_pct: float - Length variation ±%
        - radial_resolution: int - Tube radial resolution
        - random_seed: int - Random seed

    Returns
    -------
    manifold : m3d.Manifold
        The generated vascular tree geometry
    stats : dict
        Statistics about the generated network
    """
    # Extract parameters with defaults
    root_position = np.array(params.get("root_position", [0.0, 0.0, 0.001]))
    root_direction = np.array(params.get("root_direction", [0.0, 0.0, -1.0]))
    root_radius = params.get("root_radius", 0.0002)

    branching_levels = params.get("branching_levels", 5)
    branches_per_node = params.get("branches_per_node", 2)
    branching_angle_deg = params.get("branching_angle_deg", 35.0)

    segment_length = params.get("segment_length", 0.0003)
    taper_segment_length = params.get("taper_segment_length", True)
    length_taper_factor = params.get("length_taper_factor", 0.85)

    radius_mode = params.get("radius_mode", "murray")
    murray_exponent = params.get("murray_exponent", 3.0)
    min_terminal_radius = params.get("min_terminal_radius", 0.00003)

    add_variation = params.get("add_variation", False)
    angle_variation_deg = params.get("angle_variation_deg", 10.0)
    length_variation_pct = params.get("length_variation_pct", 15.0)

    random_seed = params.get("random_seed", 42)

    # Normalize root direction
    root_direction = root_direction / np.linalg.norm(root_direction)

    # Create domain (large enough to contain the tree)
    # Estimate tree extent based on levels and segment lengths
    max_depth = segment_length * branching_levels
    if taper_segment_length:
        # Geometric series sum
        max_depth = segment_length * (1 - length_taper_factor**branching_levels) / (1 - length_taper_factor)

    domain_radius = max_depth * 2  # Conservative estimate
    domain_height = max_depth * 3

    domain = CylinderDomain(
        center=[0.0, 0.0, -max_depth/2],
        radius=domain_radius,
        height=domain_height,
        axis_direction=[0.0, 0.0, 1.0]
    )

    # Calculate radius ratio based on mode
    if radius_mode == "murray":
        # Murray's law: r_parent^n = sum(r_child^n)
        # For equal children: r_child = r_parent / (num_children)^(1/n)
        radius_ratio = 1.0 / (branches_per_node ** (1.0 / murray_exponent))
    elif radius_mode == "linear":
        # Simple linear taper
        radius_ratio = 0.8
    elif radius_mode == "fixed":
        # Fixed radius (no taper)
        radius_ratio = 1.0
    else:
        # Default to Murray's law
        radius_ratio = 1.0 / (branches_per_node ** (1.0 / 3.0))

    # Calculate step decay for length tapering
    step_decay = length_taper_factor if taper_segment_length else 1.0

    # Configure collision avoidance (light collision checking for regular trees)
    collision_config = CollisionOnlineConfig(
        enabled=True,
        buffer_abs_m=min_terminal_radius * 0.5,  # Small buffer
        buffer_rel=0.1,
        rotation_attempts=8,
        reduction_factors=[0.8],
        max_attempts_per_child=10,
        on_fail="terminate_branch",  # Stop branch on collision
    )

    # Create scaffold top-down config
    config = ScaffoldTopDownConfig(
        primary_axis=tuple(root_direction),
        splits=branches_per_node,
        levels=branching_levels,
        ratio=radius_ratio,
        step_length=segment_length,
        step_decay=step_decay,
        spread=segment_length * 0.5,  # Lateral spread for branching
        spread_decay=0.9,
        cone_angle_deg=branching_angle_deg,
        jitter_deg=angle_variation_deg if add_variation else 0.0,
        curvature=0.0,  # Straight segments for regular tree
        curve_samples=3,
        min_radius=min_terminal_radius,
        collision_online=collision_config,
        branch_plane_mode="local",  # 3D branching
    )

    # Generate network using AOG backend
    logger.info(f"Generating bifurcating tree with {branching_levels} levels, {branches_per_node} branches/node")

    backend = ScaffoldTopDownBackend()

    try:
        network = backend.generate(
            domain=domain,
            num_outlets=0,  # Not used (determined by levels and splits)
            inlet_position=root_position,
            inlet_radius=root_radius,
            vessel_type="arterial",
            config=config,
            rng_seed=random_seed,
        )

        logger.info(f"Generated tree with {len(network.nodes)} nodes, {len(network.segments)} segments")

    except Exception as e:
        logger.error(f"Bifurcating tree generation failed: {e}")
        raise ValueError(f"Network generation failed: {e}")

    # Convert network to manifold
    try:
        manifold, mesh_stats = network_to_manifold(
            network,
            segments_per_circle=params.get("radial_resolution", 12),
            add_node_spheres=False,  # Avoid bulb artifacts
            cap_ends=True,
        )
    except Exception as e:
        logger.error(f"Mesh conversion failed: {e}")
        raise ValueError(f"Failed to convert network to mesh: {e}")

    # Calculate comprehensive stats
    network_stats = calculate_network_stats(network)

    # Calculate volume
    try:
        volume_m3 = manifold.volume() if hasattr(manifold, "volume") else 0.0
        volume_mm3 = volume_m3 * 1e9  # m³ to mm³
    except Exception as e:
        logger.warning(f"Could not calculate volume: {e}")
        volume_mm3 = 0.0

    # Combine all stats
    stats = {
        "scaffold_type": "bifurcating_tree",
        "triangle_count": mesh_stats.get("mesh_faces", 0),
        "volume_mm3": volume_mm3,
        "network_nodes": mesh_stats.get("network_nodes", 0),
        "network_segments": mesh_stats.get("network_segments", 0),
        "total_length_m": network_stats.get("total_length_m", 0.0),
        "min_radius_m": network_stats.get("min_radius_m", 0.0),
        "max_radius_m": network_stats.get("max_radius_m", 0.0),
        "terminal_count": network_stats.get("terminal_count", 0),
        "branching_levels": branching_levels,
        "branches_per_node": branches_per_node,
    }

    logger.info(f"Bifurcating tree complete: {stats['network_nodes']} nodes, {stats['triangle_count']} triangles")

    return manifold, stats
