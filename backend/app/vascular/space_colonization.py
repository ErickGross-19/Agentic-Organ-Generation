"""
Space colonization vascular network generation for MorphoStruct.

This module wraps AOG's space colonization backend to work with
MorphoStruct's parameter format and manifold3d geometry.
"""

import manifold3d as m3d
import numpy as np
from typing import Tuple, Dict, Any, List
import logging

from generation.backends.space_colonization_backend import (
    SpaceColonizationBackend,
    SpaceColonizationConfig,
)
from generation.core.domain import CylinderDomain, BoxDomain
from aog_policies import TissueSamplingPolicy

from .mesh_adapter import network_to_manifold, calculate_network_stats

logger = logging.getLogger(__name__)


def generate_space_colonization_from_dict(
    params: Dict[str, Any]
) -> Tuple[m3d.Manifold, Dict[str, Any]]:
    """
    Generate space colonization vascular network from parameter dictionary.

    This function follows MorphoStruct's generator convention:
    - Takes a dict of parameters
    - Returns (manifold, stats) tuple

    Parameters
    ----------
    params : dict
        Parameter dictionary with keys matching SpaceColonizationParams model:
        - inlets: List[dict] - Inlet specifications
        - num_attractors: int - Number of tissue attraction points
        - influence_radius: float - Attractor influence radius (m)
        - kill_radius: float - Perfusion kill radius (m)
        - step_size: float - Growth step size (m)
        - max_iterations: int - Maximum growth iterations
        - enable_bifurcation: bool - Enable bifurcation
        - bifurcation_probability: float - Bifurcation probability
        - min_attractors_for_split: int - Min attractors for bifurcation
        - max_children_per_node: int - Max children per node
        - min_radius: float - Minimum vessel radius (m)
        - max_radius: float - Maximum vessel radius (m)
        - taper_factor: float - Radius taper per generation
        - multi_inlet_mode: str - Multi-inlet mode
        - directional_bias: float - Directional growth bias
        - max_deviation_deg: float - Max deviation angle
        - radial_resolution: int - Tube radial resolution
        - random_seed: int - Random seed

    Returns
    -------
    manifold : m3d.Manifold
        The generated vascular network geometry
    stats : dict
        Statistics about the generated network
    """
    # Extract parameters with defaults
    inlets_data = params.get("inlets", [{
        "position": [0.0, 0.0, 0.001],
        "radius": 0.0002,
        "direction": [0.0, 0.0, -1.0]
    }])

    # Parse inlet specifications
    inlets = []
    for inlet_spec in inlets_data:
        inlets.append({
            "position": np.array(inlet_spec.get("position", [0.0, 0.0, 0.001])),
            "radius": inlet_spec.get("radius", 0.0002),
            "direction": np.array(inlet_spec.get("direction", [0.0, 0.0, -1.0])),
        })

    # Create domain
    # For now, use a cylindrical domain
    # TODO: Support custom domain meshes via domain_id
    domain_radius = params.get("domain_radius", 0.005)  # 5mm default
    domain_height = params.get("domain_height", 0.002)  # 2mm default

    domain = CylinderDomain(
        center=[0.0, 0.0, 0.0],
        radius=domain_radius,
        height=domain_height,
        axis_direction=[0.0, 0.0, 1.0]
    )

    # Create AOG space colonization config
    config = SpaceColonizationConfig(
        num_attractors=params.get("num_attractors", 50000),
        attraction_distance=params.get("influence_radius", 0.002),
        kill_distance=params.get("kill_radius", 0.00025),
        step_size=params.get("step_size", 0.00018),
        max_iterations=params.get("max_iterations", 300),
        encourage_bifurcation=params.get("enable_bifurcation", True),
        bifurcation_probability=params.get("bifurcation_probability", 0.35),
        min_attractions_for_bifurcation=params.get("min_attractors_for_split", 8),
        max_children_per_node=params.get("max_children_per_node", 2),
        min_radius=params.get("min_radius", 0.00003),
        taper_factor=params.get("taper_factor", 0.95),
        multi_inlet_mode=params.get("multi_inlet_mode", "blended"),
        directional_bias=params.get("directional_bias", 0.35),
        max_deviation_deg=params.get("max_deviation_deg", 70.0),
    )

    # Generate network using AOG backend
    logger.info(f"Generating space colonization network with {config.num_attractors} attractors")

    backend = SpaceColonizationBackend()

    try:
        # For multi-inlet, call generate_multi_inlet
        if len(inlets) > 1:
            network = backend.generate_multi_inlet(
                domain=domain,
                inlets=inlets,
                config=config,
                rng_seed=params.get("random_seed", 42),
            )
        else:
            # Single inlet
            network = backend.generate(
                domain=domain,
                num_outlets=0,  # Not used in space colonization
                inlet_position=inlets[0]["position"],
                inlet_radius=inlets[0]["radius"],
                config=config,
                rng_seed=params.get("random_seed", 42),
            )

        logger.info(f"Generated network with {len(network.nodes)} nodes, {len(network.segments)} segments")

    except Exception as e:
        logger.error(f"Space colonization generation failed: {e}")
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
        "scaffold_type": "space_colonization",
        "triangle_count": mesh_stats.get("mesh_faces", 0),
        "volume_mm3": volume_mm3,
        "network_nodes": mesh_stats.get("network_nodes", 0),
        "network_segments": mesh_stats.get("network_segments", 0),
        "total_length_m": network_stats.get("total_length_m", 0.0),
        "min_radius_m": network_stats.get("min_radius_m", 0.0),
        "max_radius_m": network_stats.get("max_radius_m", 0.0),
        "terminal_count": network_stats.get("terminal_count", 0),
        "num_inlets": len(inlets),
    }

    logger.info(f"Space colonization complete: {stats['network_nodes']} nodes, {stats['triangle_count']} triangles")

    return manifold, stats
