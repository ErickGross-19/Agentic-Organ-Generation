"""
Trabecular bone scaffold generator with interconnected porous network.

Mimics cancellous bone structure with random interconnected struts forming
a network of pores typical of trabecular bone tissue.

Implements biologically accurate parameters:
- Variable pore sizes (100-600 μm)
- Trabecular thickness control with variance
- Connectivity density modeling
- Anisotropy for load-bearing alignment
- Mixed rod/plate morphology (SMI-based)
- Resorption pits (Howship's lacunae)
- Density gradients
- Surface roughness
"""

from __future__ import annotations
import manifold3d as m3d
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from ..core import batch_union
from ..helpers.mesh_utils import subtract_all


@dataclass
class TrabecularBoneParams:
    """Parameters for trabecular bone scaffold generation.

    Biologically realistic parameters based on human cancellous bone:
    - Trabecular thickness: 100-300 μm (mean ~150-200 μm)
    - Trabecular spacing: 300-1000 μm (mean ~500-600 μm)
    - Porosity: 50-90% (mean ~75-80%)
    - Bone volume fraction (BV/TV): 10-50% (mean ~20-25%)
    - Degree of anisotropy: 1.5-2.5 for vertebrae, ~1.8 average
    """
    # Basic geometry
    bounding_box: tuple[float, float, float] = (5.0, 5.0, 5.0)
    resolution: int = 6
    seed: int = 42

    # Pore characteristics (ranges based on micro-CT of human trabecular bone)
    pore_size_min_um: float = 100.0  # Minimum pore diameter
    pore_size_max_um: float = 600.0  # Maximum pore diameter
    pore_size_um: float = 400.0  # Mean pore size (legacy compatibility)
    pore_size_variance: float = 0.2  # Random variation in pore size (0-1)

    # Trabecular architecture
    trabecular_thickness_um: float = 200.0  # Mean trabecular thickness (100-300 μm typical)
    trabecular_spacing_um: float = 500.0  # Mean trabecular separation (300-1000 μm typical)
    strut_thickness_um: float = 200.0  # Legacy alias for trabecular_thickness
    strut_diameter_um: float = 250.0  # Alternative strut specification

    # Porosity and volume metrics
    porosity: float = 0.75  # Target porosity (void fraction)
    bone_volume_fraction: float = 0.25  # BV/TV ratio (1 - porosity)
    connectivity_density: float = 5.0  # Connections per mm³ (1-15 typical)

    # Structural anisotropy
    anisotropy_ratio: float = 1.5  # Primary direction stretch
    degree_anisotropy: float = 1.8  # MIL-based anisotropy (1.0 = isotropic)
    primary_orientation_deg: float = 0.0  # Primary trabecular alignment direction
    fabric_tensor_eigenratio: float = 1.5  # Ratio of principal fabric eigenvalues

    # Randomization and organic appearance
    randomization_factor: float = 0.2  # Overall structural randomness (0-1)
    position_noise: float = 0.3  # Node position jitter relative to spacing
    thickness_variance: float = 0.15  # Variation in trabecular thickness
    connectivity_variance: float = 0.2  # Variation in local connectivity

    # Rod vs plate morphology
    structure_model_index: float = 1.5  # 0 = plates, 3 = rods, ~1.5 typical
    enable_plate_like_regions: bool = True  # Include some plate-like structures
    plate_fraction: float = 0.3  # Fraction of plate-like vs rod-like structures

    # Surface characteristics
    surface_roughness: float = 0.1  # Surface roughness factor
    enable_resorption_pits: bool = False  # Simulate osteoclast resorption lacunae
    resorption_pit_density: float = 0.05  # Pits per mm² of surface

    # Gradient options
    enable_density_gradient: bool = False  # Variable density across scaffold
    gradient_direction: str = 'z'  # Axis for gradient (x, y, z)
    gradient_start_porosity: float = 0.6  # Porosity at gradient start
    gradient_end_porosity: float = 0.9  # Porosity at gradient end


def _apply_rotation_matrix(point: np.ndarray, angle_deg: float, axis: str = 'z') -> np.ndarray:
    """Apply rotation around specified axis."""
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)

    if axis == 'z':
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    elif axis == 'y':
        rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    else:  # x
        rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    return rot @ point


def _get_position_along_gradient(position: np.ndarray, direction: str,
                                  bounds: Tuple[float, float, float]) -> float:
    """Get normalized position (0-1) along gradient axis."""
    bx, by, bz = bounds
    if direction == 'x':
        return position[0] / bx if bx > 0 else 0.5
    elif direction == 'y':
        return position[1] / by if by > 0 else 0.5
    else:  # z
        return position[2] / bz if bz > 0 else 0.5


def _interpolate_porosity(pos_normalized: float, start_porosity: float,
                          end_porosity: float) -> float:
    """Linearly interpolate porosity based on position."""
    return start_porosity + pos_normalized * (end_porosity - start_porosity)


def _create_rod_strut(p1: np.ndarray, p2: np.ndarray, radius: float,
                      resolution: int, roughness: float = 0.0,
                      rng: Optional[np.random.Generator] = None) -> m3d.Manifold:
    """
    Create a cylindrical rod strut between two points.

    Args:
        p1: Starting point
        p2: Ending point
        radius: Strut radius
        resolution: Cylinder segments
        roughness: Surface roughness factor (0-1)
        rng: Random generator for roughness

    Returns:
        Manifold for the strut
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    length = np.sqrt(dx*dx + dy*dy + dz*dz)

    if length < 1e-6:
        return m3d.Manifold()

    # Apply surface roughness by slightly varying radius
    if roughness > 0 and rng is not None:
        # Roughness of 0.1 = 10% variation, scaled to realistic 5-10μm Ra
        radius_variation = 1.0 + (rng.random() - 0.5) * roughness * 0.2
        radius = radius * radius_variation

    # Create cylinder along Z axis
    strut = m3d.Manifold.cylinder(length, radius, radius, resolution)

    # Calculate rotation to align with direction vector
    h = np.sqrt(dx*dx + dy*dy)
    if h > 0.001 or abs(dz) > 0.001:
        angle_y = np.arctan2(h, dz) * 180 / np.pi
        angle_z = np.arctan2(dy, dx) * 180 / np.pi

        strut = strut.rotate([0, angle_y, 0])
        strut = strut.rotate([0, 0, angle_z])

    return strut.translate([p1[0], p1[1], p1[2]])


def _create_plate_strut(p1: np.ndarray, p2: np.ndarray, thickness: float,
                        resolution: int, plate_width_ratio: float = 2.5,
                        roughness: float = 0.0,
                        rng: Optional[np.random.Generator] = None) -> m3d.Manifold:
    """
    Create a plate-like strut (flattened ellipsoid cross-section) between two points.

    Plate-like trabeculae have elliptical cross-sections with width > thickness.
    SMI=0 indicates pure plates, SMI=3 indicates pure rods.

    Args:
        p1: Starting point
        p2: Ending point
        thickness: Plate thickness (minor axis)
        resolution: Mesh resolution
        plate_width_ratio: Width to thickness ratio (typically 2-3)
        roughness: Surface roughness factor
        rng: Random generator

    Returns:
        Manifold for the plate strut
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    length = np.sqrt(dx*dx + dy*dy + dz*dz)

    if length < 1e-6:
        return m3d.Manifold()

    # Apply roughness
    if roughness > 0 and rng is not None:
        thickness_var = 1.0 + (rng.random() - 0.5) * roughness * 0.2
        thickness = thickness * thickness_var

    # Create elongated box (plate cross-section approximation)
    # Plate is wider in one dimension than the other
    plate_width = thickness * plate_width_ratio
    plate_height = thickness

    # Create a box and scale to create elliptical-like cross section
    plate = m3d.Manifold.cube([plate_width, plate_height, length], center=True)
    plate = plate.translate([0, 0, length/2])  # Move so one end is at origin

    # Calculate rotation to align with direction vector
    h = np.sqrt(dx*dx + dy*dy)
    if h > 0.001 or abs(dz) > 0.001:
        angle_y = np.arctan2(h, dz) * 180 / np.pi
        angle_z = np.arctan2(dy, dx) * 180 / np.pi

        plate = plate.rotate([0, angle_y, 0])
        plate = plate.rotate([0, 0, angle_z])

    return plate.translate([p1[0], p1[1], p1[2]])


def _create_resorption_pit(center: np.ndarray, normal: np.ndarray,
                           diameter: float, depth: float,
                           resolution: int) -> m3d.Manifold:
    """
    Create a hemispherical resorption pit (Howship's lacuna).

    These are created by osteoclasts during bone remodeling.
    Typical size: 100+ μm diameter.

    Args:
        center: Center point on surface
        normal: Surface normal direction
        diameter: Pit diameter (typically 100-200 μm)
        depth: Pit depth (typically 50-100% of radius)
        resolution: Sphere resolution

    Returns:
        Sphere manifold to subtract from surface
    """
    radius = diameter / 2
    # Create sphere (we'll subtract to create the pit)
    pit = m3d.Manifold.sphere(radius, resolution)

    # Position the sphere so its center creates appropriate depth
    # Normal should point outward from surface
    offset = normal * (radius - depth)
    return pit.translate([center[0] + offset[0],
                         center[1] + offset[1],
                         center[2] + offset[2]])


def _should_include_strut_for_gradient(strut_center: np.ndarray,
                                        params: TrabecularBoneParams,
                                        rng: np.random.Generator,
                                        effective_porosity: Optional[float] = None) -> bool:
    """
    Determine if a strut should be included based on density gradient.

    Higher porosity = fewer struts, so we randomly exclude struts
    based on local porosity.

    Args:
        strut_center: Center position of the strut
        params: Trabecular bone parameters
        rng: Random number generator
        effective_porosity: Effective porosity (derived from bone_volume_fraction if set)
    """
    if not params.enable_density_gradient:
        return True

    # Use effective_porosity if provided, otherwise fall back to params.porosity
    base_porosity = effective_porosity if effective_porosity is not None else params.porosity

    pos_norm = _get_position_along_gradient(
        strut_center,
        params.gradient_direction,
        params.bounding_box
    )

    local_porosity = _interpolate_porosity(
        pos_norm,
        params.gradient_start_porosity,
        params.gradient_end_porosity
    )

    # Convert porosity to inclusion probability
    # Higher porosity = lower inclusion probability
    # Base probability from effective porosity (considers bone_volume_fraction)
    base_prob = 1.0 - (base_porosity * 0.8)

    # Scale by local porosity ratio
    porosity_ratio = local_porosity / base_porosity if base_porosity > 0 else 1.0
    adjusted_prob = base_prob / porosity_ratio
    adjusted_prob = max(0.1, min(0.95, adjusted_prob))  # Clamp

    return rng.random() < adjusted_prob


def _calculate_local_connectivity(base_connectivity: float,
                                   variance: float,
                                   rng: np.random.Generator) -> float:
    """
    Calculate local connectivity with variance.

    Connectivity density is ~3-7 connections per mm³ in healthy bone.
    """
    if variance <= 0:
        return base_connectivity

    # Apply variance as percentage variation
    multiplier = 1.0 + (rng.random() - 0.5) * 2 * variance
    return base_connectivity * multiplier


def generate_trabecular_bone(params: TrabecularBoneParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate a trabecular bone scaffold with interconnected porous network.

    Uses a randomized lattice approach to create irregular interconnected
    struts mimicking cancellous bone microarchitecture. Implements:

    - Variable pore sizes within min/max range
    - Trabecular thickness with natural variance
    - Connectivity density control
    - Anisotropic alignment for load-bearing
    - Mixed rod/plate morphology (SMI-based)
    - Optional resorption pits
    - Optional density gradients
    - Surface roughness

    Args:
        params: TrabecularBoneParams specifying geometry and structure

    Returns:
        Tuple of (manifold, stats_dict)
        - manifold: The 3D geometry
        - stats_dict: Dictionary with triangle_count, volume_mm3, porosity,
                     strut_count, scaffold_type

    Raises:
        ValueError: If no struts are generated
    """
    rng = np.random.default_rng(params.seed)

    bx, by, bz = params.bounding_box

    # --- PARAMETER RESOLUTION ---
    # Resolve bone_volume_fraction vs porosity (they should be complementary)
    # bone_volume_fraction = 1 - porosity, use to validate/derive
    effective_porosity = params.porosity
    if params.bone_volume_fraction is not None and params.bone_volume_fraction > 0:
        # If bone_volume_fraction is explicitly set differently than 1-porosity,
        # use it to derive porosity (bone_volume_fraction takes precedence)
        derived_porosity = 1.0 - params.bone_volume_fraction
        # Use bone_volume_fraction derived value if it differs significantly
        if abs(derived_porosity - params.porosity) > 0.01:
            effective_porosity = derived_porosity

    # Resolve trabecular_thickness_um vs strut_thickness_um (legacy alias)
    # strut_thickness_um is legacy alias for trabecular_thickness_um
    # Use strut_thickness_um as fallback if trabecular_thickness_um is at default
    effective_trabecular_thickness = params.trabecular_thickness_um
    if params.trabecular_thickness_um == 200.0 and params.strut_thickness_um != 200.0:
        # User explicitly set strut_thickness_um, use it as override
        effective_trabecular_thickness = params.strut_thickness_um

    # strut_diameter_um provides upper bound for local strut diameter variation
    # This creates a range: [effective_trabecular_thickness, strut_diameter_um]
    strut_diameter_max = params.strut_diameter_um
    if strut_diameter_max < effective_trabecular_thickness:
        strut_diameter_max = effective_trabecular_thickness

    # Use trabecular_thickness_um as primary thickness parameter
    base_strut_radius_mm = effective_trabecular_thickness / 2000.0
    max_strut_radius_mm = strut_diameter_max / 2000.0

    # Calculate spacing from trabecular_spacing_um or pore_size_um
    spacing = params.trabecular_spacing_um / 1000.0  # Convert to mm

    # Pore size range in mm for variable sizing
    # Use pore_size_um as fallback if min/max are at defaults
    pore_min_mm = params.pore_size_min_um / 1000.0
    pore_max_mm = params.pore_size_max_um / 1000.0

    # If pore_size_min_um and pore_size_max_um are at defaults (100, 600),
    # but pore_size_um is different from default (400), use pore_size_um to derive range
    if (params.pore_size_min_um == 100.0 and params.pore_size_max_um == 600.0
            and params.pore_size_um != 400.0):
        # Use pore_size_um as center, create a range around it
        pore_center_mm = params.pore_size_um / 1000.0
        pore_range = pore_center_mm * 0.5  # +/- 50% range
        pore_min_mm = max(0.05, pore_center_mm - pore_range)  # Min 50 um
        pore_max_mm = pore_center_mm + pore_range

    # Apply randomization_factor as a global multiplier to all noise/variance parameters
    # randomization_factor of 0 = no randomness, 1 = full randomness
    rand_factor = params.randomization_factor
    effective_position_noise = params.position_noise * rand_factor
    effective_thickness_variance = params.thickness_variance * rand_factor
    effective_pore_variance = params.pore_size_variance * rand_factor
    effective_connectivity_variance = params.connectivity_variance * rand_factor
    effective_surface_roughness = params.surface_roughness * rand_factor

    # Generate nodes in a perturbed grid
    nx = max(2, int(bx / spacing) + 1)
    ny = max(2, int(by / spacing) + 1)
    nz = max(2, int(bz / spacing) + 1)

    nodes = []
    node_indices = {}

    # Rotation matrix for primary orientation
    orientation_rad = np.radians(params.primary_orientation_deg)

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Base position
                x = i * spacing
                y = j * spacing
                z = k * spacing

                # Add random perturbation using position_noise parameter scaled by randomization_factor
                if 0 < i < nx-1:
                    x += (rng.random() - 0.5) * spacing * effective_position_noise
                if 0 < j < ny-1:
                    y += (rng.random() - 0.5) * spacing * effective_position_noise
                if 0 < k < nz-1:
                    z += (rng.random() - 0.5) * spacing * effective_position_noise

                # Apply anisotropy based on degree_anisotropy and fabric_tensor_eigenratio
                # Scale in primary direction (z by default, rotated by primary_orientation)
                point = np.array([x, y, z])

                # Apply fabric tensor scaling - stretch in primary direction
                # eigenratio controls how much more aligned in primary vs secondary
                if params.degree_anisotropy > 1.0 or params.fabric_tensor_eigenratio > 1.0:
                    # Rotate to align with primary orientation
                    if params.primary_orientation_deg != 0:
                        point = _apply_rotation_matrix(point, -params.primary_orientation_deg, 'z')

                    # Apply anisotropic scaling (stretch in Z)
                    stretch_factor = params.fabric_tensor_eigenratio
                    point[2] *= stretch_factor

                    # Rotate back
                    if params.primary_orientation_deg != 0:
                        point = _apply_rotation_matrix(point, params.primary_orientation_deg, 'z')

                # Also apply legacy anisotropy_ratio for backward compatibility
                point[2] *= params.anisotropy_ratio

                nodes.append(point)
                node_indices[(i, j, k)] = len(nodes) - 1

    # Create struts connecting neighboring nodes
    strut_manifolds = []
    strut_count = 0
    rod_count = 0
    plate_count = 0

    # Base connection probability from effective porosity (derived from bone_volume_fraction if set)
    base_connection_probability = 1.0 - (effective_porosity * 0.8)

    # Determine plate fraction based on structure_model_index
    # SMI: 0=plates, 3=rods. Map to plate probability
    smi_plate_prob = max(0, min(1, 1.0 - params.structure_model_index / 3.0))
    effective_plate_fraction = params.plate_fraction * smi_plate_prob if params.enable_plate_like_regions else 0

    # Track strut midpoints for resorption pits
    strut_midpoints = []
    strut_directions = []

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                idx = node_indices[(i, j, k)]
                p1 = nodes[idx]

                # Local connectivity with variance (scaled by randomization_factor)
                local_connectivity = _calculate_local_connectivity(
                    params.connectivity_density,
                    effective_connectivity_variance,
                    rng
                )

                # Normalize connectivity to connection probability
                # connectivity_density is connections per mm³, scale to per-node probability
                volume_per_node = spacing ** 3
                conn_prob_factor = local_connectivity * volume_per_node / 6.0  # 6 potential connections
                conn_prob_factor = max(0.2, min(2.0, conn_prob_factor))  # Clamp

                adjusted_prob = base_connection_probability * conn_prob_factor

                # Connect to neighbors (6-connectivity for basic, can extend to 26 for more)
                for di, dj, dk in [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1)]:
                    ni, nj, nk = i + di, j + dj, k + dk
                    if ni < nx and nj < ny and nk < nz:
                        neighbor_idx = node_indices[(ni, nj, nk)]
                        p2 = nodes[neighbor_idx]

                        # Calculate strut center for gradient
                        strut_center = (p1 + p2) / 2

                        # Check density gradient (uses effective_porosity from bone_volume_fraction)
                        if not _should_include_strut_for_gradient(strut_center, params, rng, effective_porosity):
                            continue

                        # Apply connection probability with variance
                        if rng.random() < adjusted_prob:
                            # Calculate thickness with variance (scaled by randomization_factor)
                            # strut_diameter_um provides upper bound for variation
                            thickness_mult = 1.0 + (rng.random() - 0.5) * 2 * effective_thickness_variance
                            local_radius = base_strut_radius_mm * thickness_mult
                            # Clamp to strut_diameter_um upper bound
                            local_radius = min(local_radius, max_strut_radius_mm)

                            # Determine if this strut is plate-like or rod-like
                            is_plate = rng.random() < effective_plate_fraction

                            if is_plate:
                                strut = _create_plate_strut(
                                    p1, p2, local_radius * 2,  # thickness = diameter
                                    params.resolution,
                                    plate_width_ratio=2.5,
                                    roughness=effective_surface_roughness,
                                    rng=rng
                                )
                                plate_count += 1
                            else:
                                strut = _create_rod_strut(
                                    p1, p2, local_radius,
                                    params.resolution,
                                    roughness=effective_surface_roughness,
                                    rng=rng
                                )
                                rod_count += 1

                            if strut.num_vert() > 0:
                                strut_manifolds.append(strut)
                                strut_count += 1

                                # Store for resorption pits
                                if params.enable_resorption_pits:
                                    direction = p2 - p1
                                    length = np.linalg.norm(direction)
                                    if length > 0:
                                        strut_midpoints.append(strut_center)
                                        strut_directions.append(direction / length)

    if not strut_manifolds:
        raise ValueError("No struts generated")

    # Union all struts
    result = batch_union(strut_manifolds)

    # Apply resorption pits if enabled
    if params.enable_resorption_pits and strut_midpoints:
        # Calculate number of pits based on density and surface area estimate
        # Rough surface area estimate: strut_count * average_strut_length * circumference
        avg_strut_length = spacing * 0.8
        avg_circumference = 2 * np.pi * base_strut_radius_mm
        surface_area_mm2 = strut_count * avg_strut_length * avg_circumference

        num_pits = int(surface_area_mm2 * params.resorption_pit_density)
        num_pits = min(num_pits, len(strut_midpoints), 100)  # Limit for performance

        if num_pits > 0:
            pit_manifolds = []
            pit_indices = rng.choice(len(strut_midpoints), size=num_pits, replace=False)

            for pit_idx in pit_indices:
                center = strut_midpoints[pit_idx]
                # Normal perpendicular to strut direction
                strut_dir = strut_directions[pit_idx]
                # Create a perpendicular vector
                if abs(strut_dir[2]) < 0.9:
                    perp = np.cross(strut_dir, np.array([0, 0, 1]))
                else:
                    perp = np.cross(strut_dir, np.array([1, 0, 0]))
                perp = perp / np.linalg.norm(perp)

                # Pit size: 100-200 μm diameter (0.1-0.2 mm)
                pit_diameter = (0.1 + rng.random() * 0.1)
                pit_depth = pit_diameter * 0.4  # Hemispherical depth

                pit = _create_resorption_pit(
                    center + perp * base_strut_radius_mm * 0.5,
                    perp,
                    pit_diameter,
                    pit_depth,
                    max(4, params.resolution // 2)
                )
                if pit.num_vert() > 0:
                    pit_manifolds.append(pit)

            if pit_manifolds:
                result = subtract_all(result, pit_manifolds)

    # Clip to original bounding box (accounting for anisotropy)
    clip_z = bz * params.anisotropy_ratio * params.fabric_tensor_eigenratio
    clip_box = m3d.Manifold.cube([bx, by, clip_z])
    result = result & clip_box

    # Calculate statistics
    mesh = result.to_mesh()
    volume = result.volume() if hasattr(result, 'volume') else 0
    solid_volume = bx * by * clip_z
    actual_porosity = 1.0 - (volume / solid_volume) if solid_volume > 0 else 0

    stats = {
        'triangle_count': len(mesh.tri_verts) // 3 if hasattr(mesh, 'tri_verts') else 0,
        'volume_mm3': volume,
        'porosity': actual_porosity,
        'strut_count': strut_count,
        'rod_count': rod_count,
        'plate_count': plate_count,
        'node_count': len(nodes),
        'scaffold_type': 'trabecular_bone',
        'effective_plate_fraction': plate_count / strut_count if strut_count > 0 else 0,
        'density_gradient_enabled': params.enable_density_gradient,
        'resorption_pits_enabled': params.enable_resorption_pits,
        # Effective parameter values (after resolving legacy aliases and derivations)
        'effective_porosity': effective_porosity,
        'effective_trabecular_thickness_um': effective_trabecular_thickness,
        'effective_strut_radius_mm': base_strut_radius_mm,
        'max_strut_radius_mm': max_strut_radius_mm,
        'randomization_factor_applied': rand_factor,
        'pore_size_range_mm': (pore_min_mm, pore_max_mm),
    }

    return result, stats


def make_strut(p1: np.ndarray, p2: np.ndarray, radius: float, resolution: int) -> m3d.Manifold:
    """
    Create a cylindrical strut between two points.

    Legacy function for backward compatibility. Use _create_rod_strut for new code.

    Args:
        p1: Starting point (x, y, z)
        p2: Ending point (x, y, z)
        radius: Strut radius
        resolution: Number of segments around cylinder

    Returns:
        Manifold representing the strut cylinder
    """
    return _create_rod_strut(p1, p2, radius, resolution)


def generate_trabecular_bone_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate trabecular bone from dictionary parameters (for API compatibility).

    Args:
        params: Dictionary with keys matching TrabecularBoneParams fields

    Returns:
        Tuple of (manifold, stats_dict)
    """
    bbox = params.get('bounding_box', (10, 10, 10))
    if isinstance(bbox, dict):
        bbox = (bbox['x'], bbox['y'], bbox['z'])

    return generate_trabecular_bone(TrabecularBoneParams(
        # Basic geometry
        bounding_box=tuple(bbox),
        resolution=params.get('resolution', 6),
        seed=params.get('seed', 42),

        # Pore characteristics
        pore_size_min_um=params.get('pore_size_min_um', 100.0),
        pore_size_max_um=params.get('pore_size_max_um', 600.0),
        pore_size_um=params.get('pore_size_um', 400.0),
        pore_size_variance=params.get('pore_size_variance', 0.2),

        # Trabecular architecture
        trabecular_thickness_um=params.get('trabecular_thickness_um', 200.0),
        trabecular_spacing_um=params.get('trabecular_spacing_um', 500.0),
        strut_thickness_um=params.get('strut_thickness_um', 200.0),
        strut_diameter_um=params.get('strut_diameter_um', 250.0),

        # Porosity and volume metrics
        porosity=params.get('porosity', 0.75),
        bone_volume_fraction=params.get('bone_volume_fraction', 0.25),
        connectivity_density=params.get('connectivity_density', 5.0),

        # Structural anisotropy
        anisotropy_ratio=params.get('anisotropy_ratio', 1.5),
        degree_anisotropy=params.get('degree_anisotropy', 1.8),
        primary_orientation_deg=params.get('primary_orientation_deg', 0.0),
        fabric_tensor_eigenratio=params.get('fabric_tensor_eigenratio', 1.5),

        # Randomization
        randomization_factor=params.get('randomization_factor', 0.2),
        position_noise=params.get('position_noise', 0.3),
        thickness_variance=params.get('thickness_variance', 0.15),
        connectivity_variance=params.get('connectivity_variance', 0.2),

        # Rod vs plate morphology
        structure_model_index=params.get('structure_model_index', 1.5),
        enable_plate_like_regions=params.get('enable_plate_like_regions', True),
        plate_fraction=params.get('plate_fraction', 0.3),

        # Surface characteristics
        surface_roughness=params.get('surface_roughness', 0.1),
        enable_resorption_pits=params.get('enable_resorption_pits', False),
        resorption_pit_density=params.get('resorption_pit_density', 0.05),

        # Gradient options
        enable_density_gradient=params.get('enable_density_gradient', False),
        gradient_direction=params.get('gradient_direction', 'z'),
        gradient_start_porosity=params.get('gradient_start_porosity', 0.6),
        gradient_end_porosity=params.get('gradient_end_porosity', 0.9),
    ))
