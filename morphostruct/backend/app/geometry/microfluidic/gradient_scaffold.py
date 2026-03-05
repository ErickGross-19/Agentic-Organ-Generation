"""
Gradient scaffold generator with continuous porosity/stiffness gradients.

Creates scaffolds with varying pore size and density along a specified axis.
Useful for tissue engineering applications requiring mechanical or biological gradients.

Features:
- Linear, radial, and axial gradient directions
- Linear, exponential, or sigmoid gradient functions
- Continuous variation in pore size and stiffness
- Multi-zone architecture for interfacial tissue engineering
- Stiffness gradient modeling for osteochondral applications
"""

from __future__ import annotations
import manifold3d as m3d
import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional, List
from ..core import batch_union


@dataclass
class GradientScaffoldParams:
    """
    Parameters for gradient scaffold generation.

    Based on tissue engineering gradient scaffold designs for interfacial
    tissue regeneration (e.g., osteochondral, tendon-bone, ligament-bone).
    """
    # === Gradient Direction and Type ===
    gradient_direction: Literal['linear', 'radial', 'axial'] = 'linear'
    gradient_axis: Literal['x', 'y', 'z'] = 'z'  # For linear gradient
    gradient_type: Literal['linear', 'exponential', 'sigmoid'] = 'linear'

    # === Porosity Gradient ===
    starting_porosity: float = 0.65  # 0.3-0.95 typical
    ending_porosity: float = 0.90  # 0.3-0.95 typical
    porosity_exponent: float = 1.0  # For exponential gradients

    # === Pore Size Gradient ===
    min_pore_size_um: float = 100.0  # um (100+ um for vascularization)
    max_pore_size_um: float = 350.0  # 200-500 um for cartilage-like
    pore_shape: Literal['spherical', 'ellipsoidal', 'cylindrical'] = 'spherical'
    pore_aspect_ratio: float = 1.0  # For ellipsoidal pores

    # === Zone Architecture ===
    num_zones: int = 3  # 2-5 discrete zones
    transition_zone_width_mm: float = 1.0  # 0.5-3 mm transition regions
    enable_discrete_zones: bool = False  # False = continuous gradient
    zone_boundaries: Optional[List[float]] = None  # Custom zone boundaries (normalized 0-1)

    # === Stiffness Gradient (design parameters) ===
    enable_stiffness_gradient: bool = False
    stiffness_gradient_min_pa: float = 30000.0  # Pa (20-50 kPa cartilage ECM)
    stiffness_gradient_max_pa: float = 300000000.0  # Pa (100-500 MPa trabecular bone)
    stiffness_correlation: Literal['direct', 'inverse'] = 'inverse'  # To porosity

    # === Scaffold Geometry ===
    scaffold_thickness_mm: float = 5.0  # Total thickness along gradient
    scaffold_shape: Literal['rectangular', 'cylindrical', 'custom'] = 'rectangular'
    bounding_box_mm: tuple[float, float, float] = (10.0, 10.0, 5.0)  # x, y, z
    scaffold_diameter_mm: float = 10.0  # For cylindrical shape

    # === Grid and Pore Distribution ===
    grid_spacing_mm: float = 1.5  # Base spacing between pores
    pore_base_size_mm: float = 0.5  # Reference pore size
    enable_stochastic_variation: bool = False
    position_noise: float = 0.1  # Random position jitter (0-1)
    size_variance: float = 0.1  # Random size variation (0-1)

    # === Interconnectivity ===
    enable_interconnections: bool = False
    interconnection_diameter_ratio: float = 0.3  # Relative to pore size
    min_interconnections_per_pore: int = 3

    # === Quality/Resolution ===
    resolution: int = 12
    seed: int = 42


def calculate_pore_size(
    position: float,
    length: float,
    start_porosity: float,
    end_porosity: float,
    min_pore_size: float,
    max_pore_size: float,
    gradient_type: Literal['linear', 'exponential', 'sigmoid'],
    porosity_exponent: float = 1.0
) -> tuple[float, float]:
    """
    Calculate pore size and porosity at a given position based on gradient function.

    Args:
        position: Position along gradient axis (0 to length)
        length: Total length of gradient axis
        start_porosity: Porosity at position 0
        end_porosity: Porosity at position length
        min_pore_size: Minimum pore size (mm)
        max_pore_size: Maximum pore size (mm)
        gradient_type: Type of gradient function
        porosity_exponent: Exponent for exponential gradient (controls steepness)

    Returns:
        Tuple of (pore_size, porosity) at the given position
    """
    # Normalized position [0, 1]
    t = position / length if length > 0 else 0
    t = np.clip(t, 0, 1)

    # Calculate porosity based on gradient type
    if gradient_type == 'linear':
        porosity = start_porosity + (end_porosity - start_porosity) * t

    elif gradient_type == 'exponential':
        # Apply porosity_exponent to normalized position for steepness control
        # porosity_exponent < 1: faster initial change, slower at end
        # porosity_exponent > 1: slower initial change, faster at end
        t_exp = t ** porosity_exponent
        porosity = start_porosity + (end_porosity - start_porosity) * t_exp

    elif gradient_type == 'sigmoid':
        # Sigmoid interpolation (smooth S-curve)
        sigmoid_t = 1 / (1 + np.exp(-10 * (t - 0.5)))
        porosity = start_porosity + (end_porosity - start_porosity) * sigmoid_t

    else:
        # Default to linear
        porosity = start_porosity + (end_porosity - start_porosity) * t

    # Clamp porosity to valid range
    porosity = np.clip(porosity, 0.01, 0.95)

    # Calculate pore size proportional to porosity
    # Higher porosity = larger pores
    pore_size = min_pore_size + (max_pore_size - min_pore_size) * (porosity - start_porosity) / (end_porosity - start_porosity + 1e-6)
    pore_size = np.clip(pore_size, min_pore_size, max_pore_size)

    return pore_size, porosity


def calculate_radial_gradient(
    x: float, y: float,
    center_x: float, center_y: float,
    max_radius: float,
    start_porosity: float,
    end_porosity: float,
    min_pore_size: float,
    max_pore_size: float,
    gradient_type: Literal['linear', 'exponential', 'sigmoid'],
    porosity_exponent: float = 1.0
) -> tuple[float, float]:
    """Calculate pore size for radial gradient (center to edge)."""
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    normalized_r = r / max_radius if max_radius > 0 else 0
    return calculate_pore_size(
        normalized_r * max_radius, max_radius,
        start_porosity, end_porosity,
        min_pore_size, max_pore_size,
        gradient_type,
        porosity_exponent
    )


def calculate_zone_porosity(
    position: float,
    length: float,
    num_zones: int,
    starting_porosity: float,
    ending_porosity: float,
    zone_boundaries: Optional[List[float]],
    transition_width: float,
    enable_discrete: bool,
    gradient_type: Literal['linear', 'exponential', 'sigmoid'] = 'linear',
    porosity_exponent: float = 1.0
) -> float:
    """
    Calculate porosity at a position using discrete zone system with smooth transitions.

    Args:
        position: Position along gradient axis
        length: Total length of gradient axis
        num_zones: Number of discrete zones (2-7 typical)
        starting_porosity: Porosity at position 0
        ending_porosity: Porosity at position length
        zone_boundaries: Custom zone boundaries (normalized 0-1), or None for equal zones
        transition_width: Width of smooth transition between zones (mm)
        enable_discrete: If False, returns continuous gradient
        gradient_type: Type of gradient function (for continuous mode)
        porosity_exponent: Exponent for exponential gradient

    Returns:
        Porosity value at the given position
    """
    t = position / length if length > 0 else 0
    t = np.clip(t, 0, 1)

    if not enable_discrete:
        # Continuous gradient - use standard calculation
        if gradient_type == 'exponential':
            t_exp = t ** porosity_exponent
            return starting_porosity + (ending_porosity - starting_porosity) * t_exp
        elif gradient_type == 'sigmoid':
            sigmoid_t = 1 / (1 + np.exp(-10 * (t - 0.5)))
            return starting_porosity + (ending_porosity - starting_porosity) * sigmoid_t
        else:
            return starting_porosity + (ending_porosity - starting_porosity) * t

    # Discrete zone system
    if zone_boundaries is None or len(zone_boundaries) == 0:
        # Equal zone widths
        boundaries = [i / num_zones for i in range(1, num_zones)]
    else:
        boundaries = sorted(zone_boundaries)

    # Find which zone we're in
    zone_idx = 0
    for boundary in boundaries:
        if t < boundary:
            break
        zone_idx += 1

    # Calculate zone-specific porosity (each zone has a constant base porosity)
    zone_porosity = starting_porosity + (ending_porosity - starting_porosity) * (zone_idx + 0.5) / num_zones

    # Apply smooth transition at boundaries using smoothstep
    transition_t = transition_width / length if length > 0 else 0

    for i, boundary in enumerate(boundaries):
        if abs(t - boundary) < transition_t:
            # We're in a transition region
            # Calculate porosities of adjacent zones
            zone_below = starting_porosity + (ending_porosity - starting_porosity) * (i + 0.5) / num_zones
            zone_above = starting_porosity + (ending_porosity - starting_porosity) * (i + 1.5) / num_zones

            # Smoothstep interpolation
            blend_t = (t - boundary + transition_t) / (2 * transition_t)
            blend_t = np.clip(blend_t, 0, 1)
            # Smoothstep: 3t^2 - 2t^3
            smooth_t = blend_t * blend_t * (3 - 2 * blend_t)

            zone_porosity = zone_below + (zone_above - zone_below) * smooth_t
            break

    return np.clip(zone_porosity, 0.01, 0.95)


def create_cylinder_between_points(
    p1: np.ndarray,
    p2: np.ndarray,
    radius: float,
    resolution: int
) -> m3d.Manifold:
    """
    Create a cylinder connecting two points.

    Args:
        p1: Start point (x, y, z)
        p2: End point (x, y, z)
        radius: Cylinder radius
        resolution: Number of segments for circular cross-section

    Returns:
        Manifold cylinder oriented from p1 to p2
    """
    direction = p2 - p1
    length = np.linalg.norm(direction)

    if length < 1e-9:
        # Points too close, return tiny sphere instead
        return m3d.Manifold.sphere(radius, resolution).translate(p1.tolist())

    # Create cylinder along z-axis, then rotate and translate
    cylinder = m3d.Manifold.cylinder(length, radius, radius, resolution)

    # Calculate rotation to align z-axis with direction
    direction_normalized = direction / length
    z_axis = np.array([0, 0, 1])

    # Rotation axis is cross product
    rotation_axis = np.cross(z_axis, direction_normalized)
    rotation_axis_norm = np.linalg.norm(rotation_axis)

    if rotation_axis_norm < 1e-9:
        # Vectors are parallel
        if direction_normalized[2] < 0:
            # Pointing in -z direction, flip
            cylinder = cylinder.rotate([180, 0, 0])
    else:
        # Calculate rotation angle
        rotation_axis = rotation_axis / rotation_axis_norm
        cos_angle = np.dot(z_axis, direction_normalized)
        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

        # Convert axis-angle to Euler (simplified approach using rotation matrix)
        # For manifold3d, we use rotate([x_deg, y_deg, z_deg])
        # A more robust approach: use quaternion or rotation matrix
        # Here we use a simplified method suitable for most orientations

        # Create rotation using Rodrigues' formula conceptually
        # manifold3d rotate expects Euler angles, so we compute them
        if abs(rotation_axis[2]) < 0.99:
            # Tilt rotation around the computed axis
            # Approximate with sequential rotations
            # First rotate in xy plane to align rotation axis with x
            xy_angle = np.degrees(np.arctan2(rotation_axis[1], rotation_axis[0]))
            # Then rotate around x by the computed angle
            # Finally rotate back in xy
            cylinder = cylinder.rotate([0, 0, -xy_angle])
            cylinder = cylinder.rotate([angle_deg, 0, 0])
            cylinder = cylinder.rotate([0, 0, xy_angle])
        else:
            # Rotation axis is nearly along z, rotate around z
            cylinder = cylinder.rotate([0, angle_deg * np.sign(rotation_axis[2]), 0])

    # Translate to start from p1
    cylinder = cylinder.translate(p1.tolist())

    return cylinder


def create_interconnections(
    pore_positions: List[np.ndarray],
    pore_sizes: List[float],
    diameter_ratio: float,
    min_connections: int,
    resolution: int
) -> tuple[List[m3d.Manifold], int]:
    """
    Create interconnection channels between pores.

    Based on literature: interconnection diameter should be 0.2-0.5 of pore diameter,
    minimum 50-100 um for bone ingrowth, with 3-6 connections per pore for good
    nutrient flow.

    Args:
        pore_positions: List of pore center positions
        pore_sizes: List of pore diameters (mm)
        diameter_ratio: Interconnection diameter as ratio of pore diameter (0.2-0.5)
        min_connections: Minimum connections per pore (3-6 typical)
        resolution: Resolution for cylinder generation

    Returns:
        Tuple of (list of interconnection manifolds, total connection count)
    """
    if len(pore_positions) < 2:
        return [], 0

    interconnections = []
    connection_set = set()  # Track unique connections (i, j) where i < j

    for i, (pos_i, size_i) in enumerate(zip(pore_positions, pore_sizes)):
        # Find nearest neighbors
        distances = []
        for j, pos_j in enumerate(pore_positions):
            if i != j:
                dist = np.linalg.norm(pos_i - pos_j)
                distances.append((dist, j))

        distances.sort(key=lambda x: x[0])

        # Create connections to nearest neighbors
        num_connections = min(min_connections, len(distances))
        for k in range(num_connections):
            dist, j = distances[k]

            # Avoid duplicate connections
            conn_key = (min(i, j), max(i, j))
            if conn_key in connection_set:
                continue
            connection_set.add(conn_key)

            pos_j = pore_positions[j]
            size_j = pore_sizes[j]

            # Interconnection diameter based on smaller pore
            conn_diameter = min(size_i, size_j) * diameter_ratio

            # Minimum interconnection size for cell migration (50 um = 0.05 mm)
            conn_diameter = max(conn_diameter, 0.05)

            # Only create connection if distance is reasonable (not too far)
            # Max distance is 3x the sum of radii
            max_dist = 1.5 * (size_i + size_j)
            if dist > max_dist:
                continue

            # Create cylinder between pore centers
            conn = create_cylinder_between_points(
                pos_i, pos_j, conn_diameter / 2, resolution
            )
            interconnections.append(conn)

    return interconnections, len(connection_set)


def generate_gradient_scaffold(params: GradientScaffoldParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate a gradient scaffold with varying porosity.

    Creates a grid of pores with sizes varying according to the specified
    gradient direction and function type. Supports discrete zones with smooth
    transitions and pore interconnections for enhanced cell migration.

    Args:
        params: GradientScaffoldParams specifying geometry and gradient

    Returns:
        Tuple of (manifold, stats_dict)
        - manifold: The 3D geometry
        - stats_dict: Dictionary with triangle_count, volume_mm3, relative_density,
                     pore_count, gradient_type, scaffold_type, interconnection_count,
                     zone_count, stiffness_correlation

    Raises:
        ValueError: If no pores are generated
    """
    rng = np.random.default_rng(params.seed)

    dx, dy, dz = params.bounding_box_mm
    spacing = params.grid_spacing_mm

    # Convert pore sizes from um to mm
    min_pore_mm = params.min_pore_size_um / 1000.0
    max_pore_mm = params.max_pore_size_um / 1000.0

    # Use pore_base_size_mm as fallback when min/max are 0
    if min_pore_mm <= 0:
        min_pore_mm = params.pore_base_size_mm * 0.5
    if max_pore_mm <= 0:
        max_pore_mm = params.pore_base_size_mm * 2.0

    # Determine gradient axis and length
    # Use scaffold_thickness_mm if provided and positive
    if params.gradient_direction == 'linear':
        if params.gradient_axis == 'x':
            default_length = dx
            axis_idx = 0
        elif params.gradient_axis == 'y':
            default_length = dy
            axis_idx = 1
        else:  # 'z'
            default_length = dz
            axis_idx = 2
        # Override with scaffold_thickness_mm if specified
        gradient_length = params.scaffold_thickness_mm if params.scaffold_thickness_mm > 0 else default_length
    elif params.gradient_direction == 'radial':
        gradient_length = min(dx, dy) / 2  # Use radius
        axis_idx = -1  # Special case for radial
    else:  # 'axial'
        default_length = dz
        gradient_length = params.scaffold_thickness_mm if params.scaffold_thickness_mm > 0 else default_length
        axis_idx = 2

    # Calculate number of pores in each dimension
    nx = max(1, int(dx / spacing))
    ny = max(1, int(dy / spacing))
    nz = max(1, int(dz / spacing))

    # Determine actual zone count
    actual_zone_count = params.num_zones if params.enable_discrete_zones else 1

    # Generate pore positions and sizes
    pores = []
    pore_positions: List[np.ndarray] = []  # Track for interconnections
    pore_sizes: List[float] = []  # Track for interconnections
    pore_count = 0
    total_pore_volume = 0

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Calculate position
                x = -dx/2 + (i + 0.5) * dx / nx
                y = -dy/2 + (j + 0.5) * dy / ny
                z = -dz/2 + (k + 0.5) * dz / nz

                # Add position noise if enabled
                if params.enable_stochastic_variation:
                    noise_scale = params.position_noise * spacing
                    x += rng.uniform(-noise_scale, noise_scale)
                    y += rng.uniform(-noise_scale, noise_scale)
                    z += rng.uniform(-noise_scale, noise_scale)

                # Get position along gradient
                if params.gradient_direction == 'radial':
                    pore_size, porosity = calculate_radial_gradient(
                        x, y, 0, 0, gradient_length,
                        params.starting_porosity, params.ending_porosity,
                        min_pore_mm, max_pore_mm, params.gradient_type,
                        params.porosity_exponent
                    )
                else:
                    if params.gradient_axis == 'x':
                        gradient_pos = x + dx/2
                    elif params.gradient_axis == 'y':
                        gradient_pos = y + dy/2
                    else:
                        gradient_pos = z + dz/2

                    # Use zone system if discrete zones enabled
                    if params.enable_discrete_zones:
                        porosity = calculate_zone_porosity(
                            gradient_pos, gradient_length,
                            params.num_zones,
                            params.starting_porosity, params.ending_porosity,
                            params.zone_boundaries,
                            params.transition_zone_width_mm,
                            params.enable_discrete_zones,
                            params.gradient_type,
                            params.porosity_exponent
                        )
                        # Calculate pore size from porosity
                        pore_size = min_pore_mm + (max_pore_mm - min_pore_mm) * (
                            (porosity - params.starting_porosity) /
                            (params.ending_porosity - params.starting_porosity + 1e-6)
                        )
                        pore_size = np.clip(pore_size, min_pore_mm, max_pore_mm)
                    else:
                        pore_size, porosity = calculate_pore_size(
                            gradient_pos, gradient_length,
                            params.starting_porosity, params.ending_porosity,
                            min_pore_mm, max_pore_mm, params.gradient_type,
                            params.porosity_exponent
                        )

                # Add size variance if enabled
                if params.enable_stochastic_variation:
                    size_factor = 1.0 + rng.uniform(-params.size_variance, params.size_variance)
                    pore_size *= size_factor

                # Track pore position and size for interconnections
                pore_positions.append(np.array([x, y, z]))
                pore_sizes.append(pore_size)

                # Create pore based on shape
                if params.pore_shape == 'spherical':
                    pore = m3d.Manifold.sphere(pore_size / 2, params.resolution)
                elif params.pore_shape == 'ellipsoidal':
                    pore = m3d.Manifold.sphere(pore_size / 2, params.resolution)
                    pore = pore.scale([1, 1, params.pore_aspect_ratio])
                else:  # cylindrical
                    pore = m3d.Manifold.cylinder(pore_size, pore_size / 2, pore_size / 2, params.resolution)

                pore = pore.translate([x, y, z])
                pores.append(pore)
                pore_count += 1

                # Track pore volume
                if params.pore_shape == 'spherical':
                    total_pore_volume += (4/3) * np.pi * (pore_size/2)**3
                elif params.pore_shape == 'ellipsoidal':
                    total_pore_volume += (4/3) * np.pi * (pore_size/2)**2 * (pore_size/2 * params.pore_aspect_ratio)
                else:
                    total_pore_volume += np.pi * (pore_size/2)**2 * pore_size

    if not pores:
        raise ValueError("No pores generated")

    # Create interconnections if enabled
    interconnection_count = 0
    if params.enable_interconnections and len(pore_positions) >= 2:
        interconnections, interconnection_count = create_interconnections(
            pore_positions,
            pore_sizes,
            params.interconnection_diameter_ratio,
            params.min_interconnections_per_pore,
            params.resolution
        )
        # Add interconnections to pores for combined subtraction
        if interconnections:
            pores.extend(interconnections)

    # Union all pores (and interconnections)
    pores_combined = batch_union(pores)

    # Create bounding volume based on shape
    if params.scaffold_shape == 'cylindrical':
        bbox = m3d.Manifold.cylinder(dz, params.scaffold_diameter_mm/2, params.scaffold_diameter_mm/2, params.resolution)
        bbox = bbox.translate([0, 0, -dz/2])
        solid_volume = np.pi * (params.scaffold_diameter_mm/2)**2 * dz
    else:
        bbox = m3d.Manifold.cube([dx, dy, dz], center=True)
        solid_volume = dx * dy * dz

    # Subtract pores from solid block
    result = bbox - pores_combined

    # Calculate statistics
    mesh = result.to_mesh()
    volume = result.volume() if hasattr(result, 'volume') else 0
    relative_density = volume / solid_volume if solid_volume > 0 else 0
    actual_porosity = 1 - relative_density

    # Calculate average stiffness if gradient enabled
    avg_stiffness = None
    if params.enable_stiffness_gradient:
        avg_stiffness = (params.stiffness_gradient_min_pa + params.stiffness_gradient_max_pa) / 2

    stats = {
        'triangle_count': len(mesh.tri_verts) // 3 if hasattr(mesh, 'tri_verts') else 0,
        'volume_mm3': volume,
        'relative_density': relative_density,
        'actual_porosity': actual_porosity,
        'pore_count': pore_count,
        'gradient_type': params.gradient_type,
        'gradient_direction': params.gradient_direction,
        'num_zones': params.num_zones,
        'zone_count': actual_zone_count,  # Actual zones used (1 if continuous)
        'enable_discrete_zones': params.enable_discrete_zones,
        'starting_porosity': params.starting_porosity,
        'ending_porosity': params.ending_porosity,
        'porosity_exponent': params.porosity_exponent,
        'min_pore_size_um': params.min_pore_size_um,
        'max_pore_size_um': params.max_pore_size_um,
        'scaffold_type': 'gradient_scaffold',
        # Interconnection stats
        'enable_interconnections': params.enable_interconnections,
        'interconnection_count': interconnection_count,
        'interconnection_diameter_ratio': params.interconnection_diameter_ratio,
        # Stiffness correlation for downstream mechanical simulation
        'stiffness_correlation': params.stiffness_correlation,
    }

    if avg_stiffness is not None:
        stats['average_stiffness_pa'] = avg_stiffness
        stats['stiffness_gradient_min_pa'] = params.stiffness_gradient_min_pa
        stats['stiffness_gradient_max_pa'] = params.stiffness_gradient_max_pa

    return result, stats


def generate_gradient_scaffold_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate gradient scaffold from dictionary parameters (for API compatibility).

    Args:
        params: Dictionary with keys matching GradientScaffoldParams fields

    Returns:
        Tuple of (manifold, stats_dict)
    """
    # Handle bounding_box variations
    bbox = params.get('bounding_box_mm', params.get('bounding_box', params.get('dimensions_mm', params.get('dimensions', (10.0, 10.0, 5.0)))))
    if isinstance(bbox, dict):
        bbox = (bbox.get('x', 10.0), bbox.get('y', 10.0), bbox.get('z', 5.0))

    # Handle gradient_direction variations (map old x/y/z to linear)
    gradient_direction = params.get('gradient_direction', 'linear')
    gradient_axis = params.get('gradient_axis', 'z')

    # Support old style where gradient_direction was x/y/z
    if gradient_direction in ['x', 'y', 'z']:
        gradient_axis = gradient_direction
        gradient_direction = 'linear'

    p = GradientScaffoldParams(
        # Gradient direction and type
        gradient_direction=gradient_direction,
        gradient_axis=gradient_axis,
        gradient_type=params.get('gradient_type', 'linear'),

        # Porosity gradient
        starting_porosity=params.get('starting_porosity', params.get('start_porosity', 0.65)),
        ending_porosity=params.get('ending_porosity', params.get('end_porosity', 0.90)),
        porosity_exponent=params.get('porosity_exponent', 1.0),

        # Pore size gradient
        min_pore_size_um=params.get('min_pore_size_um', params.get('min_pore_size', 100.0)),
        max_pore_size_um=params.get('max_pore_size_um', params.get('max_pore_size', 350.0)),
        pore_shape=params.get('pore_shape', 'spherical'),
        pore_aspect_ratio=params.get('pore_aspect_ratio', 1.0),

        # Zone architecture
        num_zones=params.get('num_zones', 3),
        transition_zone_width_mm=params.get('transition_zone_width_mm', params.get('transition_zone_width', 1.0)),
        enable_discrete_zones=params.get('enable_discrete_zones', False),
        zone_boundaries=params.get('zone_boundaries', None),

        # Stiffness gradient
        enable_stiffness_gradient=params.get('enable_stiffness_gradient', False),
        stiffness_gradient_min_pa=params.get('stiffness_gradient_min_pa', params.get('stiffness_gradient_min', 30000.0)),
        stiffness_gradient_max_pa=params.get('stiffness_gradient_max_pa', params.get('stiffness_gradient_max', 300000000.0)),
        stiffness_correlation=params.get('stiffness_correlation', 'inverse'),

        # Scaffold geometry
        scaffold_thickness_mm=params.get('scaffold_thickness_mm', params.get('scaffold_thickness', 5.0)),
        scaffold_shape=params.get('scaffold_shape', 'rectangular'),
        bounding_box_mm=tuple(bbox),
        scaffold_diameter_mm=params.get('scaffold_diameter_mm', params.get('scaffold_diameter', 10.0)),

        # Grid and pore distribution
        grid_spacing_mm=params.get('grid_spacing_mm', params.get('grid_spacing', 1.5)),
        pore_base_size_mm=params.get('pore_base_size_mm', params.get('pore_base_size', 0.5)),
        enable_stochastic_variation=params.get('enable_stochastic_variation', False),
        position_noise=params.get('position_noise', 0.1),
        size_variance=params.get('size_variance', 0.1),

        # Interconnectivity
        enable_interconnections=params.get('enable_interconnections', False),
        interconnection_diameter_ratio=params.get('interconnection_diameter_ratio', 0.3),
        min_interconnections_per_pore=params.get('min_interconnections_per_pore', 3),

        # Quality/Resolution
        resolution=params.get('resolution', 12),
        seed=params.get('seed', params.get('random_seed', 42)),
    )

    return generate_gradient_scaffold(p)
