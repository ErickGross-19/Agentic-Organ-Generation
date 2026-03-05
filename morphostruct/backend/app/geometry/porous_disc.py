"""
Porous disc scaffold generator.

Generates cylindrical disc scaffolds with pore patterns for tissue engineering.
Supports hexagonal and grid pore arrangements with configurable porosity.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Literal

try:
    import manifold3d as m3d
    HAS_MANIFOLD = True
except ImportError:
    m3d = None
    HAS_MANIFOLD = False

from .core import batch_union


@dataclass
class PorousDiscParams:
    """
    Parameters for porous disc scaffold generation.

    Attributes:
        diameter_mm: Disc diameter in millimeters (1-50)
        height_mm: Disc height/thickness in millimeters (0.5-10)
        pore_diameter_um: Pore diameter in microns (50-500)
        pore_spacing_um: Center-to-center pore spacing in microns (100-1000).
            If porosity_target is set, this is auto-calculated and ignored.
        pore_pattern: Pore arrangement pattern ('hexagonal' or 'grid')
        porosity_target: Target porosity (0.1-0.9). If set, pore_spacing_um is
            auto-calculated to achieve this porosity. If None, uses pore_spacing_um directly.
        resolution: Number of segments for cylindrical surfaces (8-64)
    """
    diameter_mm: float = 10.0
    height_mm: float = 2.0
    pore_diameter_um: float = 200.0
    pore_spacing_um: float = 400.0
    pore_pattern: Literal['hexagonal', 'grid'] = 'hexagonal'
    porosity_target: float | None = None
    resolution: int = 16


def generate_hexagonal_positions(radius: float, spacing: float) -> list[tuple[float, float]]:
    """
    Generate hexagonal grid of points within a circle.

    Hexagonal packing provides optimal density and uniform spacing.
    Each row is offset by half the spacing for tessellation.

    Args:
        radius: Maximum radius for pore placement (mm)
        spacing: Center-to-center spacing between pores (mm)

    Returns:
        List of (x, y) positions in mm
    """
    positions = []
    # Hexagonal pattern: rows offset by half spacing
    row_height = spacing * np.sqrt(3) / 2
    y = -radius
    row = 0

    while y <= radius:
        x_offset = (spacing / 2) if (row % 2) else 0
        x = -radius + x_offset

        while x <= radius:
            if x*x + y*y <= radius*radius:  # Inside circle
                positions.append((x, y))
            x += spacing

        y += row_height
        row += 1

    return positions


def generate_grid_positions(radius: float, spacing: float) -> list[tuple[float, float]]:
    """
    Generate square grid of points within a circle.

    Simple orthogonal grid pattern, easier to analyze but lower packing density.

    Args:
        radius: Maximum radius for pore placement (mm)
        spacing: Center-to-center spacing between pores (mm)

    Returns:
        List of (x, y) positions in mm
    """
    positions = []
    n = int(radius / spacing) + 1

    for i in range(-n, n+1):
        for j in range(-n, n+1):
            x, y = i * spacing, j * spacing
            if x*x + y*y <= radius*radius:
                positions.append((x, y))

    return positions


def generate_porous_disc(params: PorousDiscParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate a porous disc scaffold.

    Creates a cylindrical disc with a regular array of cylindrical pores.
    The pore pattern can be hexagonal (optimal packing) or grid (orthogonal).

    Algorithm:
        1. Create base cylinder (disc) with specified diameter and height
        2. Generate pore positions based on chosen pattern
        3. Create cylindrical voids at each pore position
        4. Boolean subtract all voids from base disc

    Args:
        params: PorousDiscParams configuration object

    Returns:
        Tuple of (manifold, statistics_dict)

        The statistics dict contains:
            - triangle_count: Number of triangles in mesh
            - volume_mm3: Final scaffold volume
            - porosity: Fraction of volume that is pores (0-1)
            - pore_count: Number of pores created
            - scaffold_type: 'porous_disc'

    Raises:
        ImportError: If manifold3d is not installed
    """
    if not HAS_MANIFOLD:
        raise ImportError(
            "manifold3d is required for geometry operations. "
            "Install with: pip install manifold3d"
        )

    # Convert units
    radius_mm = params.diameter_mm / 2
    pore_radius_mm = params.pore_diameter_um / 2000  # um to mm

    # Calculate spacing from porosity_target if specified
    if params.porosity_target is not None:
        # Clamp porosity to valid range
        target_porosity = max(0.1, min(0.9, params.porosity_target))

        # Calculate spacing to achieve target porosity
        # For hexagonal packing: porosity = (π * r²) / (s² * √3/2)
        # Solving for s: s = r * sqrt(2π / (porosity * √3))
        # For grid packing: porosity = (π * r²) / s²
        # Solving for s: s = r * sqrt(π / porosity)
        if params.pore_pattern == 'hexagonal':
            spacing_mm = pore_radius_mm * np.sqrt(2 * np.pi / (target_porosity * np.sqrt(3)))
        else:  # grid
            spacing_mm = pore_radius_mm * np.sqrt(np.pi / target_porosity)

        # Ensure minimum spacing (pores can't overlap)
        min_spacing = pore_radius_mm * 2.1  # 5% margin
        spacing_mm = max(spacing_mm, min_spacing)
    else:
        spacing_mm = params.pore_spacing_um / 1000  # um to mm

    # Create base disc
    base = m3d.Manifold.cylinder(
        params.height_mm,
        radius_mm,
        radius_mm,
        params.resolution
    )

    # Generate pore positions
    if params.pore_pattern == 'hexagonal':
        positions = generate_hexagonal_positions(radius_mm - pore_radius_mm, spacing_mm)
    else:
        positions = generate_grid_positions(radius_mm - pore_radius_mm, spacing_mm)

    # Create pore cylinders
    pores = []
    for x, y in positions:
        pore = m3d.Manifold.cylinder(
            params.height_mm + 0.1,  # Slightly taller to ensure clean cut
            pore_radius_mm,
            pore_radius_mm,
            max(6, params.resolution // 2)  # Fewer segments for pores
        ).translate([x, y, -0.05])
        pores.append(pore)

    # Union all pores and subtract from base
    if pores:
        all_pores = batch_union(pores)
        result = base - all_pores
    else:
        result = base

    # Calculate statistics
    mesh = result.to_mesh()
    volume = result.volume() if hasattr(result, 'volume') else 0
    base_volume = np.pi * radius_mm**2 * params.height_mm
    porosity = 1 - (volume / base_volume) if base_volume > 0 else 0

    stats = {
        'triangle_count': len(mesh.tri_verts) // 3 if hasattr(mesh, 'tri_verts') else 0,
        'volume_mm3': volume,
        'porosity': porosity,
        'pore_count': len(positions),
        'scaffold_type': 'porous_disc'
    }

    return result, stats


def generate_porous_disc_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Convenience function to generate porous disc from dictionary parameters.

    Useful for API endpoints that receive JSON payloads.

    Args:
        params: Dictionary of parameters matching PorousDiscParams fields

    Returns:
        Tuple of (manifold, statistics_dict)

    Example:
        params = {
            'diameter_mm': 15.0,
            'height_mm': 3.0,
            'pore_diameter_um': 250.0,
            'pore_spacing_um': 500.0,
            'pore_pattern': 'hexagonal'
        }
        manifold, stats = generate_porous_disc_from_dict(params)
    """
    # Filter to only valid PorousDiscParams fields
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(PorousDiscParams)}
    filtered_params = {k: v for k, v in params.items() if k in valid_fields}
    return generate_porous_disc(PorousDiscParams(**filtered_params))
