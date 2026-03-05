"""
Osteochondral scaffold generator with gradient structure.

Creates a multi-layer scaffold transitioning from bone to calcified cartilage
to cartilage, with gradient porosity matching native osteochondral tissue.
"""

from __future__ import annotations
import manifold3d as m3d
import numpy as np
from dataclasses import dataclass
from typing import Literal


@dataclass
class OsteochondralParams:
    """Parameters for osteochondral scaffold generation.

    Biologically realistic parameters based on native osteochondral tissue:
    - Articular cartilage: 2-4 mm thick
    - Calcified cartilage: 0.1-0.3 mm thick (tidemark to cement line)
    - Subchondral bone plate: 0.2-0.5 mm thick
    - Trabecular bone: Variable depth, 50-90% porosity
    """
    # Basic geometry
    diameter: float = 8.0  # Scaffold diameter in mm
    resolution: int = 16
    seed: int = 42

    # Zone depths (mm)
    cartilage_depth: float = 2.5  # Total articular cartilage thickness
    calcified_cartilage_depth: float = 0.2  # Calcified cartilage zone thickness
    subchondral_plate_depth: float = 0.3  # Subchondral bone plate thickness
    bone_depth: float = 3.0  # Trabecular bone depth
    transition_width: float = 1.0  # Gradient transition zone width

    # Cartilage zone ratios (superficial/middle/deep within cartilage_depth)
    superficial_zone_ratio: float = 0.15  # 10-20% of cartilage
    middle_zone_ratio: float = 0.50  # 40-60% of cartilage
    deep_zone_ratio: float = 0.35  # 30-40% of cartilage

    # Porosity per zone
    cartilage_porosity: float = 0.85  # Uncalcified cartilage porosity
    calcified_cartilage_porosity: float = 0.65  # Lower porosity in calcified zone
    subchondral_plate_porosity: float = 0.10  # Dense cortical-like bone
    bone_porosity: float = 0.70  # Trabecular bone porosity

    # Pore size gradients (mm)
    superficial_pore_size: float = 0.15  # Small pores in superficial zone
    middle_pore_size: float = 0.25  # Medium pores in middle zone
    deep_pore_size: float = 0.35  # Larger pores in deep zone
    bone_pore_size: float = 0.50  # Trabecular bone pore size

    # Gradient and transition properties
    gradient_type: Literal['linear', 'exponential', 'sigmoid'] = 'linear'
    gradient_sharpness: float = 1.0  # Controls transition steepness (1.0 = normal)

    # Tidemark and cement line features
    enable_tidemark: bool = True  # Interface between calcified/uncalcified cartilage
    tidemark_thickness: float = 0.01  # Tidemark thickness (~10 μm)
    enable_cement_line: bool = True  # Interface between calcified cartilage/bone
    cement_line_thickness: float = 0.005  # Cement line (~5 μm)

    # Collagen fiber orientation by zone
    superficial_fiber_angle_deg: float = 0.0  # Tangential (parallel to surface)
    deep_fiber_angle_deg: float = 90.0  # Perpendicular to surface

    # Channel features
    enable_vascular_channels: bool = True  # Vascular channels in bone zone
    vascular_channel_diameter: float = 0.1  # Blood vessel channel diameter
    vascular_channel_spacing: float = 0.8  # Spacing between vascular channels

    # Subchondral bone properties
    subchondral_trabecular_thickness: float = 0.15  # Thinner trabeculae near surface
    subchondral_connectivity: float = 0.8  # Higher connectivity near plate

    # Mechanical property indicators
    enable_stiffness_gradient: bool = False  # Visual indicator of stiffness zones
    cartilage_modulus_indicator: float = 1.0  # Relative cartilage stiffness
    bone_modulus_indicator: float = 20.0  # Relative bone stiffness

    # Randomization
    position_noise: float = 0.1  # Random variation in pore positions
    pore_size_variance: float = 0.15  # Random variation in pore sizes


def _apply_gradient(t: float, gradient_type: str, sharpness: float) -> float:
    """
    Apply gradient transformation to interpolation parameter.

    Args:
        t: Linear interpolation parameter (0-1)
        gradient_type: 'linear', 'exponential', or 'sigmoid'
        sharpness: Controls steepness of transition (1.0 = normal)

    Returns:
        Transformed interpolation parameter
    """
    if gradient_type == 'linear':
        return t ** sharpness
    elif gradient_type == 'exponential':
        return (np.exp(sharpness * t) - 1) / (np.exp(sharpness) - 1)
    elif gradient_type == 'sigmoid':
        # Sigmoid centered at 0.5
        k = 10 * sharpness
        return 1 / (1 + np.exp(-k * (t - 0.5)))
    return t


def _interpolate_fiber_angle(z: float, z_deep_start: float, z_superficial_end: float,
                             deep_angle: float, superficial_angle: float) -> float:
    """
    Interpolate fiber angle based on z-position within cartilage.

    Args:
        z: Current z-position
        z_deep_start: Z where deep zone starts
        z_superficial_end: Z where superficial zone ends (top)
        deep_angle: Fiber angle in deep zone (typically 90°)
        superficial_angle: Fiber angle in superficial zone (typically 0°)

    Returns:
        Interpolated angle in degrees
    """
    if z <= z_deep_start:
        return deep_angle
    if z >= z_superficial_end:
        return superficial_angle

    t = (z - z_deep_start) / (z_superficial_end - z_deep_start)
    return deep_angle + t * (superficial_angle - deep_angle)


def generate_osteochondral(params: OsteochondralParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate an osteochondral scaffold with biologically accurate multi-layer structure.

    Creates 6+ distinct layers (bottom to top):
    1. Trabecular bone - porous bone with vascular channels
    2. Subchondral plate - dense cortical-like bone (5.8% porosity)
    3. Cement line - thin interdigitated interface (optional)
    4. Calcified cartilage - mineralized cartilage (15-25% porosity)
    5. Tidemark - semipermeable boundary (optional)
    6. Deep cartilage zone - perpendicular fibers, 400-500μm pores
    7. Middle cartilage zone - transitional, 300-400μm pores
    8. Superficial cartilage zone - parallel fibers, 200-300μm pores

    Uses fiber-oriented pore channels and zone-specific porosity/pore sizes
    based on native osteochondral tissue architecture.

    Args:
        params: OsteochondralParams specifying geometry and structure

    Returns:
        Tuple of (manifold, stats_dict)
        - manifold: The 3D geometry
        - stats_dict: Dictionary with detailed layer information
    """
    rng = np.random.default_rng(params.seed)

    # Calculate cartilage sub-zone heights
    # Normalize ratios to ensure they sum to 1.0
    total_ratio = params.superficial_zone_ratio + params.middle_zone_ratio + params.deep_zone_ratio
    superficial_ratio = params.superficial_zone_ratio / total_ratio
    middle_ratio = params.middle_zone_ratio / total_ratio
    deep_ratio = params.deep_zone_ratio / total_ratio

    # Available cartilage depth after accounting for calcified cartilage
    uncalcified_cartilage_depth = params.cartilage_depth

    superficial_height = uncalcified_cartilage_depth * superficial_ratio
    middle_height = uncalcified_cartilage_depth * middle_ratio
    deep_height = uncalcified_cartilage_depth * deep_ratio

    # Calculate total height with all layers
    total_height = (
        params.bone_depth +
        params.subchondral_plate_depth +
        (params.cement_line_thickness if params.enable_cement_line else 0) +
        params.calcified_cartilage_depth +
        (params.tidemark_thickness if params.enable_tidemark else 0) +
        deep_height +
        middle_height +
        superficial_height
    )

    radius = params.diameter / 2

    # Create base cylinder
    base = m3d.Manifold.cylinder(total_height, radius, radius, params.resolution)

    # Track z-position as we build up from bottom
    z_current = 0.0
    pore_manifolds = []
    layer_count = 0
    layer_info = {}

    # ========== LAYER 1: TRABECULAR BONE ==========
    z_bone_start = z_current
    z_bone_end = z_current + params.bone_depth

    pore_manifolds.extend(create_zone_pores_advanced(
        z_start=z_bone_start,
        z_end=z_bone_end,
        radius=radius,
        pore_radius=params.bone_pore_size / 2,
        pore_spacing=params.bone_pore_size * 1.5,
        porosity=params.bone_porosity,
        resolution=params.resolution,
        fiber_angle_deg=90.0,  # Vertical orientation in bone
        position_noise=params.position_noise,
        pore_size_variance=params.pore_size_variance,
        rng=rng
    ))

    layer_info['trabecular_bone'] = {
        'z_start': z_bone_start,
        'z_end': z_bone_end,
        'depth': params.bone_depth,
        'porosity': params.bone_porosity,
        'pore_size': params.bone_pore_size
    }
    layer_count += 1
    z_current = z_bone_end

    # ========== VASCULAR CHANNELS IN BONE ==========
    if params.enable_vascular_channels:
        vascular_channels = create_vascular_channels(
            z_start=z_bone_start,
            z_end=z_bone_end,
            radius=radius,
            channel_diameter=params.vascular_channel_diameter,
            channel_spacing=params.vascular_channel_spacing,
            resolution=params.resolution,
            rng=rng
        )
        pore_manifolds.extend(vascular_channels)
        layer_info['vascular_channels'] = {
            'count': len(vascular_channels),
            'diameter': params.vascular_channel_diameter,
            'spacing': params.vascular_channel_spacing
        }

    # ========== LAYER 2: SUBCHONDRAL PLATE ==========
    z_plate_start = z_current
    z_plate_end = z_current + params.subchondral_plate_depth

    # Subchondral plate is very dense (low porosity ~5.8%)
    # Use trabecular thickness parameter for pore sizing
    plate_pore_radius = params.subchondral_trabecular_thickness * 0.5
    # Higher connectivity = denser pore network = smaller spacing
    plate_spacing = params.subchondral_trabecular_thickness * (3.0 - params.subchondral_connectivity)

    pore_manifolds.extend(create_zone_pores_advanced(
        z_start=z_plate_start,
        z_end=z_plate_end,
        radius=radius,
        pore_radius=plate_pore_radius,
        pore_spacing=plate_spacing,
        porosity=params.subchondral_plate_porosity,
        resolution=params.resolution,
        fiber_angle_deg=90.0,
        position_noise=params.position_noise * 0.5,  # Less variation in dense plate
        pore_size_variance=params.pore_size_variance * 0.5,
        rng=rng
    ))

    layer_info['subchondral_plate'] = {
        'z_start': z_plate_start,
        'z_end': z_plate_end,
        'depth': params.subchondral_plate_depth,
        'porosity': params.subchondral_plate_porosity,
        'trabecular_thickness': params.subchondral_trabecular_thickness,
        'connectivity': params.subchondral_connectivity
    }
    layer_count += 1
    z_current = z_plate_end

    # ========== LAYER 3: CEMENT LINE (OPTIONAL) ==========
    if params.enable_cement_line:
        z_cement_start = z_current
        z_cement_end = z_current + params.cement_line_thickness

        # Cement line is modeled as a thin, wavy interface
        # Create interdigitated pattern with small channels
        cement_pores = create_interdigitated_layer(
            z_start=z_cement_start,
            z_end=z_cement_end,
            radius=radius,
            resolution=params.resolution,
            wave_amplitude=params.cement_line_thickness * 0.3,
            wave_frequency=8,
            rng=rng
        )
        pore_manifolds.extend(cement_pores)

        layer_info['cement_line'] = {
            'z_start': z_cement_start,
            'z_end': z_cement_end,
            'thickness': params.cement_line_thickness
        }
        layer_count += 1
        z_current = z_cement_end

    # ========== LAYER 4: CALCIFIED CARTILAGE ==========
    z_calcified_start = z_current
    z_calcified_end = z_current + params.calcified_cartilage_depth

    # Calcified cartilage: 15-25% porosity, intermediate pore size
    calcified_pore_radius = params.deep_pore_size * 0.6

    pore_manifolds.extend(create_zone_pores_advanced(
        z_start=z_calcified_start,
        z_end=z_calcified_end,
        radius=radius,
        pore_radius=calcified_pore_radius,
        pore_spacing=calcified_pore_radius * 3.0,
        porosity=params.calcified_cartilage_porosity,
        resolution=params.resolution,
        fiber_angle_deg=90.0,  # Perpendicular in calcified zone
        position_noise=params.position_noise,
        pore_size_variance=params.pore_size_variance,
        rng=rng
    ))

    layer_info['calcified_cartilage'] = {
        'z_start': z_calcified_start,
        'z_end': z_calcified_end,
        'depth': params.calcified_cartilage_depth,
        'porosity': params.calcified_cartilage_porosity
    }
    layer_count += 1
    z_current = z_calcified_end

    # ========== LAYER 5: TIDEMARK (OPTIONAL) ==========
    if params.enable_tidemark:
        z_tidemark_start = z_current
        z_tidemark_end = z_current + params.tidemark_thickness

        # Tidemark is a thin semipermeable boundary
        # Model as a thin layer with very small pores
        tidemark_pores = create_thin_porous_layer(
            z_start=z_tidemark_start,
            z_end=z_tidemark_end,
            radius=radius,
            resolution=params.resolution,
            pore_density=0.3,  # Semipermeable
            rng=rng
        )
        pore_manifolds.extend(tidemark_pores)

        layer_info['tidemark'] = {
            'z_start': z_tidemark_start,
            'z_end': z_tidemark_end,
            'thickness': params.tidemark_thickness
        }
        layer_count += 1
        z_current = z_tidemark_end

    # ========== LAYER 6: DEEP CARTILAGE ZONE ==========
    z_deep_start = z_current
    z_deep_end = z_current + deep_height

    # Deep zone: perpendicular fibers (90°), larger pores
    pore_manifolds.extend(create_zone_pores_advanced(
        z_start=z_deep_start,
        z_end=z_deep_end,
        radius=radius,
        pore_radius=params.deep_pore_size / 2,
        pore_spacing=params.deep_pore_size * 1.2,
        porosity=params.cartilage_porosity * 0.9,  # Slightly lower porosity
        resolution=params.resolution,
        fiber_angle_deg=params.deep_fiber_angle_deg,
        position_noise=params.position_noise,
        pore_size_variance=params.pore_size_variance,
        rng=rng
    ))

    layer_info['deep_cartilage'] = {
        'z_start': z_deep_start,
        'z_end': z_deep_end,
        'depth': deep_height,
        'pore_size': params.deep_pore_size,
        'fiber_angle': params.deep_fiber_angle_deg
    }
    layer_count += 1
    z_current = z_deep_end

    # ========== LAYER 7: MIDDLE CARTILAGE ZONE ==========
    z_middle_start = z_current
    z_middle_end = z_current + middle_height

    # Middle zone: transitional fiber angle (45° average)
    middle_fiber_angle = (params.deep_fiber_angle_deg + params.superficial_fiber_angle_deg) / 2

    pore_manifolds.extend(create_zone_pores_advanced(
        z_start=z_middle_start,
        z_end=z_middle_end,
        radius=radius,
        pore_radius=params.middle_pore_size / 2,
        pore_spacing=params.middle_pore_size * 1.2,
        porosity=params.cartilage_porosity,
        resolution=params.resolution,
        fiber_angle_deg=middle_fiber_angle,
        position_noise=params.position_noise,
        pore_size_variance=params.pore_size_variance,
        rng=rng
    ))

    layer_info['middle_cartilage'] = {
        'z_start': z_middle_start,
        'z_end': z_middle_end,
        'depth': middle_height,
        'pore_size': params.middle_pore_size,
        'fiber_angle': middle_fiber_angle
    }
    layer_count += 1
    z_current = z_middle_end

    # ========== LAYER 8: SUPERFICIAL CARTILAGE ZONE ==========
    z_superficial_start = z_current
    z_superficial_end = z_current + superficial_height

    # Superficial zone: tangential fibers (0°), smallest pores
    pore_manifolds.extend(create_zone_pores_advanced(
        z_start=z_superficial_start,
        z_end=z_superficial_end,
        radius=radius,
        pore_radius=params.superficial_pore_size / 2,
        pore_spacing=params.superficial_pore_size * 1.2,
        porosity=params.cartilage_porosity * 1.1,  # Slightly higher porosity at surface
        resolution=params.resolution,
        fiber_angle_deg=params.superficial_fiber_angle_deg,
        position_noise=params.position_noise,
        pore_size_variance=params.pore_size_variance,
        rng=rng
    ))

    layer_info['superficial_cartilage'] = {
        'z_start': z_superficial_start,
        'z_end': z_superficial_end,
        'depth': superficial_height,
        'pore_size': params.superficial_pore_size,
        'fiber_angle': params.superficial_fiber_angle_deg
    }
    layer_count += 1

    # ========== STIFFNESS GRADIENT INDICATORS (OPTIONAL) ==========
    if params.enable_stiffness_gradient:
        # Add visual markers indicating stiffness zones
        # Higher stiffness = smaller marker rings, lower stiffness = larger rings
        stiffness_markers = create_stiffness_indicators(
            bone_z_start=z_bone_start,
            bone_z_end=z_bone_end,
            cartilage_z_start=z_deep_start,
            cartilage_z_end=z_superficial_end,
            radius=radius,
            bone_modulus=params.bone_modulus_indicator,
            cartilage_modulus=params.cartilage_modulus_indicator,
            resolution=params.resolution
        )
        pore_manifolds.extend(stiffness_markers)
        layer_info['stiffness_gradient'] = {
            'enabled': True,
            'bone_modulus_indicator': params.bone_modulus_indicator,
            'cartilage_modulus_indicator': params.cartilage_modulus_indicator
        }

    # ========== GRADIENT TRANSITION ZONES ==========
    # Add gradient transitions between major zones
    # Transition between subchondral plate and calcified cartilage
    if params.transition_width > 0:
        gradient_pores = create_gradient_transition(
            z_start=z_plate_end - params.transition_width * 0.5,
            z_end=z_calcified_start + params.transition_width * 0.5,
            radius=radius,
            start_porosity=params.subchondral_plate_porosity,
            end_porosity=params.calcified_cartilage_porosity,
            start_pore_size=plate_pore_radius,
            end_pore_size=calcified_pore_radius,
            gradient_type=params.gradient_type,
            gradient_sharpness=params.gradient_sharpness,
            resolution=params.resolution,
            rng=rng
        )
        pore_manifolds.extend(gradient_pores)

    # ========== SUBTRACT PORES FROM BASE ==========
    result = base
    for pore in pore_manifolds:
        result = result - pore

    # ========== CALCULATE STATISTICS ==========
    mesh = result.to_mesh()
    volume = result.volume() if hasattr(result, 'volume') else 0

    # Calculate volumes by zone
    bone_volume = volume * (params.bone_depth / total_height)
    cartilage_volume = volume * (params.cartilage_depth / total_height)
    calcified_volume = volume * (params.calcified_cartilage_depth / total_height)
    plate_volume = volume * (params.subchondral_plate_depth / total_height)

    stats = {
        'triangle_count': len(mesh.tri_verts) // 3 if hasattr(mesh, 'tri_verts') else 0,
        'volume_mm3': volume,
        'total_height_mm': total_height,
        'layer_count': layer_count,
        'bone_volume_mm3': bone_volume,
        'cartilage_volume_mm3': cartilage_volume,
        'calcified_cartilage_volume_mm3': calcified_volume,
        'subchondral_plate_volume_mm3': plate_volume,
        'pore_count': len(pore_manifolds),
        'scaffold_type': 'osteochondral',
        'layers': layer_info,
        'tidemark_enabled': params.enable_tidemark,
        'cement_line_enabled': params.enable_cement_line,
        'vascular_channels_enabled': params.enable_vascular_channels
    }

    return result, stats


def create_zone_pores(
    z_start: float,
    z_end: float,
    radius: float,
    pore_radius: float,
    pore_spacing: float,
    porosity: float,
    resolution: int
) -> list[m3d.Manifold]:
    """
    Create pores for a specific zone (legacy function for backwards compatibility).

    Args:
        z_start: Start height of zone
        z_end: End height of zone
        radius: Outer radius of scaffold
        pore_radius: Radius of individual pores
        pore_spacing: Spacing between pore centers
        porosity: Target porosity for this zone
        resolution: Angular resolution

    Returns:
        List of pore manifolds
    """
    return create_zone_pores_advanced(
        z_start=z_start,
        z_end=z_end,
        radius=radius,
        pore_radius=pore_radius,
        pore_spacing=pore_spacing,
        porosity=porosity,
        resolution=resolution,
        fiber_angle_deg=90.0,
        position_noise=0.0,
        pore_size_variance=0.0,
        rng=np.random.default_rng(42)
    )


def create_zone_pores_advanced(
    z_start: float,
    z_end: float,
    radius: float,
    pore_radius: float,
    pore_spacing: float,
    porosity: float,
    resolution: int,
    fiber_angle_deg: float = 90.0,
    position_noise: float = 0.0,
    pore_size_variance: float = 0.0,
    rng: np.random.Generator = None
) -> list[m3d.Manifold]:
    """
    Create pores for a specific zone with fiber orientation and randomization.

    Args:
        z_start: Start height of zone
        z_end: End height of zone
        radius: Outer radius of scaffold
        pore_radius: Radius of individual pores
        pore_spacing: Spacing between pore centers
        porosity: Target porosity for this zone
        resolution: Angular resolution
        fiber_angle_deg: Fiber orientation angle (0° = tangential, 90° = perpendicular)
        position_noise: Random variation in pore positions (0-1)
        pore_size_variance: Random variation in pore sizes (0-1)
        rng: Random number generator

    Returns:
        List of pore manifolds
    """
    if rng is None:
        rng = np.random.default_rng(42)

    pores = []
    z_height = z_end - z_start

    if z_height <= 0 or pore_radius <= 0:
        return pores

    # Calculate radial and angular distribution
    n_radial = max(1, int(radius / pore_spacing))
    n_vertical = max(1, int(z_height / pore_spacing))

    # Adjust density based on porosity
    density_factor = porosity / 0.7  # Normalized to reference porosity

    # Convert fiber angle to radians
    fiber_angle_rad = np.radians(fiber_angle_deg)

    for i in range(n_radial):
        r = (i + 0.5) * (radius / n_radial) * 0.8  # Stay within bounds
        circumference = 2 * np.pi * r
        n_angular = max(3, int(circumference / pore_spacing * density_factor))

        for j in range(n_angular):
            angle = j * 2 * np.pi / n_angular

            for k in range(n_vertical):
                z = z_start + (k + 0.5) * (z_height / n_vertical)

                # Apply position noise
                if position_noise > 0:
                    noise_r = rng.uniform(-position_noise, position_noise) * pore_spacing * 0.3
                    noise_angle = rng.uniform(-np.pi/6, np.pi/6) * position_noise
                    noise_z = rng.uniform(-position_noise, position_noise) * pore_spacing * 0.3
                    r_final = max(0.1, r + noise_r)
                    angle_final = angle + noise_angle
                    z_final = max(z_start + pore_radius, min(z_end - pore_radius, z + noise_z))
                else:
                    r_final = r
                    angle_final = angle
                    z_final = z

                # Pore position
                x = r_final * np.cos(angle_final)
                y = r_final * np.sin(angle_final)

                # Apply pore size variance
                if pore_size_variance > 0:
                    size_factor = 1.0 + rng.uniform(-pore_size_variance, pore_size_variance)
                    actual_pore_radius = pore_radius * max(0.5, size_factor)
                else:
                    actual_pore_radius = pore_radius

                # Create oriented pore based on fiber angle
                # 90° = vertical ellipsoid, 0° = horizontal ellipsoid
                if abs(fiber_angle_deg - 90.0) < 5.0:
                    # Vertical orientation - use vertical ellipsoid
                    pore = create_oriented_pore(
                        actual_pore_radius, resolution,
                        stretch_z=1.5, stretch_xy=1.0
                    )
                elif abs(fiber_angle_deg) < 5.0:
                    # Horizontal orientation - use horizontal ellipsoid
                    pore = create_oriented_pore(
                        actual_pore_radius, resolution,
                        stretch_z=0.7, stretch_xy=1.3
                    )
                else:
                    # Intermediate angle - spherical with slight stretch
                    stretch = 1.0 + 0.3 * np.sin(fiber_angle_rad)
                    pore = create_oriented_pore(
                        actual_pore_radius, resolution,
                        stretch_z=stretch, stretch_xy=1.0
                    )

                pore = pore.translate([x, y, z_final])
                pores.append(pore)

    return pores


def create_oriented_pore(
    radius: float,
    resolution: int,
    stretch_z: float = 1.0,
    stretch_xy: float = 1.0
) -> m3d.Manifold:
    """
    Create an oriented ellipsoidal pore.

    Args:
        radius: Base radius of the pore
        resolution: Angular resolution
        stretch_z: Stretch factor in Z direction
        stretch_xy: Stretch factor in XY plane

    Returns:
        Ellipsoidal pore manifold
    """
    # Create sphere and scale to ellipsoid
    sphere = m3d.Manifold.sphere(radius, resolution)
    return sphere.scale([stretch_xy, stretch_xy, stretch_z])


def create_vascular_channels(
    z_start: float,
    z_end: float,
    radius: float,
    channel_diameter: float,
    channel_spacing: float,
    resolution: int,
    rng: np.random.Generator
) -> list[m3d.Manifold]:
    """
    Create vertical vascular channels in the bone zone.

    Based on biological data: ~6.25 channels/mm² density, 25-35μm diameter.

    Args:
        z_start: Bottom of bone zone
        z_end: Top of bone zone
        radius: Scaffold radius
        channel_diameter: Vascular channel diameter
        channel_spacing: Spacing between channels
        resolution: Angular resolution
        rng: Random number generator

    Returns:
        List of channel manifolds
    """
    channels = []
    channel_radius = channel_diameter / 2
    channel_height = z_end - z_start

    if channel_height <= 0 or channel_radius <= 0:
        return channels

    # Calculate grid of channel positions
    n_radial = max(1, int((radius - channel_radius) / channel_spacing))

    for i in range(n_radial):
        r = (i + 0.5) * channel_spacing
        if r >= radius - channel_radius:
            continue

        circumference = 2 * np.pi * r
        n_angular = max(3, int(circumference / channel_spacing))

        for j in range(n_angular):
            angle = j * 2 * np.pi / n_angular

            # Add slight randomization
            r_noise = rng.uniform(-0.1, 0.1) * channel_spacing
            angle_noise = rng.uniform(-0.1, 0.1) * (2 * np.pi / n_angular)

            r_final = r + r_noise
            angle_final = angle + angle_noise

            x = r_final * np.cos(angle_final)
            y = r_final * np.sin(angle_final)

            # Create vertical cylinder for channel
            channel = m3d.Manifold.cylinder(channel_height, channel_radius, channel_radius, resolution)
            channel = channel.translate([x, y, z_start])
            channels.append(channel)

    return channels


def create_interdigitated_layer(
    z_start: float,
    z_end: float,
    radius: float,
    resolution: int,
    wave_amplitude: float,
    wave_frequency: int,
    rng: np.random.Generator
) -> list[m3d.Manifold]:
    """
    Create an interdigitated (wavy) interface layer like the cement line.

    The cement line between calcified cartilage and subchondral bone
    is highly interdigitated to increase mechanical interlocking.

    Args:
        z_start: Bottom of layer
        z_end: Top of layer
        radius: Scaffold radius
        resolution: Angular resolution
        wave_amplitude: Height of interdigitation waves
        wave_frequency: Number of waves around circumference
        rng: Random number generator

    Returns:
        List of pore manifolds creating wavy interface
    """
    pores = []
    layer_thickness = z_end - z_start

    if layer_thickness <= 0:
        return pores

    # Create small interconnection pores along the wavy interface
    n_radial = max(3, int(radius / 0.3))

    for i in range(n_radial):
        r = (i + 0.5) * (radius / n_radial) * 0.9
        n_angular = max(8, wave_frequency * 2)

        for j in range(n_angular):
            angle = j * 2 * np.pi / n_angular

            # Calculate wavy z position
            wave_offset = wave_amplitude * np.sin(wave_frequency * angle + rng.uniform(0, np.pi/4))
            z = z_start + layer_thickness / 2 + wave_offset

            x = r * np.cos(angle)
            y = r * np.sin(angle)

            # Create small spherical pore
            pore_radius = layer_thickness * 0.3
            pore = m3d.Manifold.sphere(pore_radius, max(8, resolution // 2))
            pore = pore.translate([x, y, z])
            pores.append(pore)

    return pores


def create_thin_porous_layer(
    z_start: float,
    z_end: float,
    radius: float,
    resolution: int,
    pore_density: float,
    rng: np.random.Generator
) -> list[m3d.Manifold]:
    """
    Create a thin semipermeable layer like the tidemark.

    The tidemark is a thin boundary (~5-10μm) between calcified
    and uncalcified cartilage with selective permeability.

    Args:
        z_start: Bottom of layer
        z_end: Top of layer
        radius: Scaffold radius
        resolution: Angular resolution
        pore_density: Relative density of pores (0-1)
        rng: Random number generator

    Returns:
        List of tiny pore manifolds
    """
    pores = []
    layer_thickness = z_end - z_start

    if layer_thickness <= 0:
        return pores

    # Create very small pores for semipermeable membrane effect
    pore_radius = layer_thickness * 0.4
    spacing = layer_thickness * 3.0 / pore_density

    n_radial = max(2, int(radius / spacing))

    for i in range(n_radial):
        r = (i + 0.5) * (radius / n_radial) * 0.85
        circumference = 2 * np.pi * r
        n_angular = max(6, int(circumference / spacing))

        for j in range(n_angular):
            # Random skip based on density
            if rng.random() > pore_density:
                continue

            angle = j * 2 * np.pi / n_angular
            angle += rng.uniform(-0.1, 0.1)

            x = r * np.cos(angle)
            y = r * np.sin(angle)
            z = z_start + layer_thickness / 2

            pore = m3d.Manifold.sphere(pore_radius, max(6, resolution // 2))
            pore = pore.translate([x, y, z])
            pores.append(pore)

    return pores


def create_gradient_transition(
    z_start: float,
    z_end: float,
    radius: float,
    start_porosity: float,
    end_porosity: float,
    start_pore_size: float,
    end_pore_size: float,
    gradient_type: str,
    gradient_sharpness: float,
    resolution: int,
    rng: np.random.Generator,
    n_layers: int = 5
) -> list[m3d.Manifold]:
    """
    Create gradient transition between two zones.

    Args:
        z_start: Start of transition
        z_end: End of transition
        radius: Scaffold radius
        start_porosity: Porosity at z_start
        end_porosity: Porosity at z_end
        start_pore_size: Pore size at z_start
        end_pore_size: Pore size at z_end
        gradient_type: 'linear', 'exponential', or 'sigmoid'
        gradient_sharpness: Controls steepness
        resolution: Angular resolution
        rng: Random number generator
        n_layers: Number of interpolation layers

    Returns:
        List of pore manifolds
    """
    pores = []
    z_height = z_end - z_start

    if z_height <= 0:
        return pores

    layer_height = z_height / n_layers

    for i in range(n_layers):
        t = i / (n_layers - 1) if n_layers > 1 else 0.5

        # Apply gradient transformation
        t_transformed = _apply_gradient(t, gradient_type, gradient_sharpness)

        # Interpolate properties
        layer_porosity = start_porosity + t_transformed * (end_porosity - start_porosity)
        layer_pore_size = start_pore_size + t_transformed * (end_pore_size - start_pore_size)

        layer_z_start = z_start + i * layer_height
        layer_z_end = layer_z_start + layer_height

        pores.extend(create_zone_pores_advanced(
            z_start=layer_z_start,
            z_end=layer_z_end,
            radius=radius,
            pore_radius=layer_pore_size,
            pore_spacing=layer_pore_size * 2.0,
            porosity=layer_porosity,
            resolution=resolution,
            fiber_angle_deg=90.0,
            position_noise=0.05,
            pore_size_variance=0.1,
            rng=rng
        ))

    return pores


def create_stiffness_indicators(
    bone_z_start: float,
    bone_z_end: float,
    cartilage_z_start: float,
    cartilage_z_end: float,
    radius: float,
    bone_modulus: float,
    cartilage_modulus: float,
    resolution: int,
    n_markers: int = 3
) -> list[m3d.Manifold]:
    """
    Create visual indicators of stiffness gradient.

    Creates ring-like markers at different z-levels with size inversely
    proportional to local stiffness (larger rings = softer tissue).

    Args:
        bone_z_start: Bottom of bone zone
        bone_z_end: Top of bone zone
        cartilage_z_start: Bottom of cartilage zone
        cartilage_z_end: Top of cartilage zone
        radius: Scaffold radius
        bone_modulus: Relative stiffness of bone (typically ~20)
        cartilage_modulus: Relative stiffness of cartilage (typically ~1)
        resolution: Angular resolution
        n_markers: Number of marker rings per zone

    Returns:
        List of indicator manifolds
    """
    indicators = []

    # Bone zone markers (small rings = high stiffness)
    bone_ring_radius = 0.05 / (bone_modulus / 10)  # Inversely scaled by modulus
    for i in range(n_markers):
        z = bone_z_start + (i + 0.5) * (bone_z_end - bone_z_start) / n_markers
        ring = m3d.Manifold.cylinder(bone_ring_radius, radius * 0.95, radius * 0.95, resolution)
        ring = ring.translate([0, 0, z - bone_ring_radius / 2])
        indicators.append(ring)

    # Cartilage zone markers (larger rings = low stiffness)
    cartilage_ring_radius = 0.05 / (cartilage_modulus / 10 + 0.1)  # Inversely scaled
    for i in range(n_markers):
        z = cartilage_z_start + (i + 0.5) * (cartilage_z_end - cartilage_z_start) / n_markers
        ring = m3d.Manifold.cylinder(cartilage_ring_radius, radius * 0.95, radius * 0.95, resolution)
        ring = ring.translate([0, 0, z - cartilage_ring_radius / 2])
        indicators.append(ring)

    return indicators


def generate_osteochondral_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate osteochondral scaffold from dictionary parameters (for API compatibility).

    Args:
        params: Dictionary with keys matching OsteochondralParams fields

    Returns:
        Tuple of (manifold, stats_dict)
    """
    return generate_osteochondral(OsteochondralParams(
        # Basic geometry
        diameter=params.get('diameter', 8.0),
        resolution=params.get('resolution', 16),
        seed=params.get('seed', 42),

        # Zone depths
        cartilage_depth=params.get('cartilage_depth', 2.5),
        calcified_cartilage_depth=params.get('calcified_cartilage_depth', 0.2),
        subchondral_plate_depth=params.get('subchondral_plate_depth', 0.3),
        bone_depth=params.get('bone_depth', 3.0),
        transition_width=params.get('transition_width', 1.0),

        # Cartilage zone ratios
        superficial_zone_ratio=params.get('superficial_zone_ratio', 0.15),
        middle_zone_ratio=params.get('middle_zone_ratio', 0.50),
        deep_zone_ratio=params.get('deep_zone_ratio', 0.35),

        # Porosity per zone
        cartilage_porosity=params.get('cartilage_porosity', 0.85),
        calcified_cartilage_porosity=params.get('calcified_cartilage_porosity', 0.65),
        subchondral_plate_porosity=params.get('subchondral_plate_porosity', 0.10),
        bone_porosity=params.get('bone_porosity', 0.70),

        # Pore size gradients
        superficial_pore_size=params.get('superficial_pore_size', 0.15),
        middle_pore_size=params.get('middle_pore_size', 0.25),
        deep_pore_size=params.get('deep_pore_size', 0.35),
        bone_pore_size=params.get('bone_pore_size', 0.50),

        # Gradient properties
        gradient_type=params.get('gradient_type', 'linear'),
        gradient_sharpness=params.get('gradient_sharpness', 1.0),

        # Tidemark and cement line
        enable_tidemark=params.get('enable_tidemark', True),
        tidemark_thickness=params.get('tidemark_thickness', 0.01),
        enable_cement_line=params.get('enable_cement_line', True),
        cement_line_thickness=params.get('cement_line_thickness', 0.005),

        # Fiber orientation
        superficial_fiber_angle_deg=params.get('superficial_fiber_angle_deg', 0.0),
        deep_fiber_angle_deg=params.get('deep_fiber_angle_deg', 90.0),

        # Channel features
        enable_vascular_channels=params.get('enable_vascular_channels', True),
        vascular_channel_diameter=params.get('vascular_channel_diameter', 0.1),
        vascular_channel_spacing=params.get('vascular_channel_spacing', 0.8),

        # Subchondral bone properties
        subchondral_trabecular_thickness=params.get('subchondral_trabecular_thickness', 0.15),
        subchondral_connectivity=params.get('subchondral_connectivity', 0.8),

        # Mechanical indicators
        enable_stiffness_gradient=params.get('enable_stiffness_gradient', False),
        cartilage_modulus_indicator=params.get('cartilage_modulus_indicator', 1.0),
        bone_modulus_indicator=params.get('bone_modulus_indicator', 20.0),

        # Randomization
        position_noise=params.get('position_noise', 0.1),
        pore_size_variance=params.get('pore_size_variance', 0.15),
    ))
