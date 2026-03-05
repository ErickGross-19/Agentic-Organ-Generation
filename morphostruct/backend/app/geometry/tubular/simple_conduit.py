"""
Tubular conduit scaffold generator.

Generates hollow cylindrical scaffolds for:
- Nerve conduits
- Vascular grafts
- Tubular organ structures

Supports various inner surface textures (smooth, grooved, porous).
"""

import manifold3d as m3d
import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class TubularConduitParams:
    """Parameters for tubular conduit generation."""

    outer_diameter_mm: float = 5.0
    wall_thickness_mm: float = 0.5
    length_mm: float = 20.0
    inner_texture: Literal['smooth', 'grooved', 'porous'] = 'smooth'
    groove_count: int = 8
    groove_depth_mm: float = 0.15
    resolution: int = 32


def generate_tubular_conduit(params: TubularConduitParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate a tubular conduit (hollow tube) scaffold.

    For nerve conduits, vascular grafts, etc.

    Args:
        params: TubularConduitParams with generation parameters

    Returns:
        Tuple of (manifold, stats_dict)

    Raises:
        ValueError: If wall thickness exceeds radius
    """
    outer_radius = params.outer_diameter_mm / 2
    inner_radius = outer_radius - params.wall_thickness_mm

    if inner_radius <= 0:
        raise ValueError("Wall thickness exceeds radius - no inner channel possible")

    # Create outer cylinder (aligned along Z axis)
    outer = m3d.Manifold.cylinder(
        params.length_mm,
        outer_radius,
        outer_radius,
        params.resolution
    )

    # Create inner cylinder (slightly longer for clean boolean)
    inner = m3d.Manifold.cylinder(
        params.length_mm + 0.2,
        inner_radius,
        inner_radius,
        params.resolution
    ).translate([0, 0, -0.1])

    # Basic tube
    tube = outer - inner

    # Add inner surface texture if requested
    if params.inner_texture == 'grooved' and params.groove_count > 0:
        # Create longitudinal grooves on inner surface
        grooves = []
        for i in range(params.groove_count):
            angle = 2 * np.pi * i / params.groove_count
            # Position groove at inner wall
            groove_center_radius = inner_radius + params.groove_depth_mm / 2
            x = groove_center_radius * np.cos(angle)
            y = groove_center_radius * np.sin(angle)

            # Create groove as thin cylinder
            groove = m3d.Manifold.cylinder(
                params.length_mm + 0.2,
                params.groove_depth_mm / 2,
                params.groove_depth_mm / 2,
                8
            ).translate([x, y, -0.1])
            grooves.append(groove)

        # Subtract grooves from tube
        from .core import batch_union
        all_grooves = batch_union(grooves)
        tube = tube - all_grooves

    elif params.inner_texture == 'porous':
        # Add small pores along inner wall
        pores = []
        pore_radius = 0.1  # 100um pores
        pore_spacing = 0.5  # 500um spacing

        # Spiral pattern of pores
        n_rings = int(params.length_mm / pore_spacing)
        n_angular = int(2 * np.pi * inner_radius / pore_spacing)

        for ring in range(1, n_rings):
            z = ring * pore_spacing
            for i in range(n_angular):
                angle = 2 * np.pi * i / n_angular + (ring * 0.3)  # Offset each ring
                x = inner_radius * np.cos(angle)
                y = inner_radius * np.sin(angle)

                # Radial pore
                pore = m3d.Manifold.cylinder(
                    params.wall_thickness_mm + 0.1,
                    pore_radius,
                    pore_radius,
                    6
                ).rotate([0, 90, np.degrees(angle)]).translate([x, y, z])
                pores.append(pore)

        if pores:
            from .core import batch_union
            all_pores = batch_union(pores)
            tube = tube - all_pores

    # Calculate statistics
    mesh = tube.to_mesh()
    volume = tube.volume() if hasattr(tube, 'volume') else 0

    stats = {
        'triangle_count': len(mesh.tri_verts) // 3 if hasattr(mesh, 'tri_verts') else 0,
        'volume_mm3': volume,
        'inner_diameter_mm': inner_radius * 2,
        'scaffold_type': 'tubular_conduit'
    }

    return tube, stats


def generate_tubular_conduit_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate a tubular conduit scaffold from dictionary parameters.

    Args:
        params: Dictionary of parameters (will be unpacked into TubularConduitParams)

    Returns:
        Tuple of (manifold, stats_dict)
    """
    return generate_tubular_conduit(TubularConduitParams(**params))
