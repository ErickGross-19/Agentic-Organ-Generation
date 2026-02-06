"""
Articular cartilage scaffold generator with zonal architecture.

Creates a three-zone scaffold mimicking native articular cartilage:
- Superficial zone: tangential collagen orientation, small pores, high cell density
- Middle zone: oblique orientation, medium pores, moderate cell density
- Deep zone: perpendicular orientation, large pores, low cell density

Implements biologically accurate features:
- Zone-specific fiber orientations (0°, 45°, 90°)
- Porosity gradients (70%, 80%, 85%)
- Collagen fiber bundles with configurable diameter/spacing
- Optional vertical channels for scaffold perfusion
- Tidemark layer at calcified cartilage interface
- Surface texture for articular gliding
- Zone boundary blurring for smooth transitions
"""

from __future__ import annotations
import manifold3d as m3d
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from ..core import batch_union


@dataclass
class ArticularCartilageParams:
    """Parameters for articular cartilage scaffold generation.

    Biologically realistic parameters based on native articular cartilage:
    - Total thickness: 1-5 mm depending on joint location (2-3 mm typical for knee)
    - Superficial zone: 10-20% of thickness, tangential collagen fibers
    - Middle zone: 40-60% of thickness, oblique/random collagen fibers
    - Deep zone: 30-40% of thickness, perpendicular collagen fibers
    - Cell density decreases from superficial to deep zone
    - Proteoglycan content increases from superficial to deep zone
    """
    # Basic geometry
    diameter: float = 8.0  # Scaffold diameter in mm
    resolution: int = 16
    seed: int = 42

    # Overall thickness (based on joint location)
    total_thickness: float = 2.5  # Total thickness in mm (2500 μm)
    total_thickness_um: float = 2500.0  # Total thickness in micrometers

    # Zone thickness ratios (must sum to 1.0)
    superficial_thickness_ratio: float = 0.15  # 10-20% of total
    middle_thickness_ratio: float = 0.50  # 40-60% of total
    deep_thickness_ratio: float = 0.35  # 30-40% of total
    zone_ratios: tuple[float, float, float] = (0.15, 0.50, 0.35)  # Legacy compatibility

    # Collagen fiber orientation per zone (degrees from surface)
    superficial_fiber_orientation_deg: float = 0.0  # Tangential (parallel to surface)
    middle_fiber_orientation_deg: float = 45.0  # Oblique orientation
    deep_fiber_orientation_deg: float = 90.0  # Perpendicular to surface
    fiber_orientation_variance_deg: float = 15.0  # Random variance in orientation

    # Pore characteristics per zone (mm)
    superficial_pore_size: float = 0.10  # Smaller pores for dense superficial zone
    middle_pore_size: float = 0.20  # Medium pores
    deep_pore_size: float = 0.30  # Larger pores for columnar structure
    pore_gradient: tuple[float, float, float] = (0.10, 0.20, 0.30)  # Legacy compatibility

    # Porosity per zone (controls PORE DENSITY, not just pore size)
    superficial_porosity: float = 0.70  # Lower porosity in superficial zone
    middle_porosity: float = 0.80  # Moderate porosity
    deep_porosity: float = 0.85  # Higher porosity for nutrient diffusion

    # Cell density per zone (cells per mm³) - for visualization/planning
    superficial_cell_density: float = 10000.0  # Highest density
    middle_cell_density: float = 5000.0  # Moderate density
    deep_cell_density: float = 3000.0  # Lowest density

    # Proteoglycan content indicators (relative units)
    superficial_proteoglycan: float = 0.3  # Lowest content
    middle_proteoglycan: float = 0.7  # Moderate content
    deep_proteoglycan: float = 1.0  # Highest content

    # Collagen fiber properties
    collagen_fiber_diameter_um: float = 50.0  # Fiber bundle diameter
    collagen_fiber_spacing_um: float = 100.0  # Spacing between fibers
    enable_fiber_bundles: bool = True  # Show fiber bundle structures

    # Channel features for nutrient diffusion
    enable_vertical_channels: bool = True  # Channels in deep zone
    vertical_channel_diameter: float = 0.15  # Channel diameter in mm
    vertical_channel_spacing: float = 0.5  # Spacing between channels
    enable_horizontal_channels: bool = False  # Tangential channels in superficial zone

    # Surface characteristics
    surface_roughness: float = 0.05  # Articular surface roughness
    enable_surface_texture: bool = True  # Textured articular surface

    # Tidemark (calcified cartilage interface)
    enable_tidemark_layer: bool = False  # Include calcified cartilage zone
    tidemark_thickness: float = 0.1  # Calcified zone thickness (mm)
    tidemark_porosity: float = 0.60  # Lower porosity for calcified zone

    # Randomization and organic appearance
    position_noise: float = 0.15  # Random jitter in pore positions
    pore_size_variance: float = 0.2  # Random variation in pore sizes
    zone_boundary_blur: float = 0.1  # Soft boundaries between zones


def generate_articular_cartilage(params: ArticularCartilageParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate an articular cartilage scaffold with zonal architecture.

    Creates three distinct zones with varying pore characteristics:
    1. Superficial zone (top) - small pores, tangential orientation (0°)
    2. Middle zone - medium pores, oblique orientation (45°)
    3. Deep zone (bottom) - large pores, perpendicular orientation (90°)

    All parameters are now implemented:
    - Fiber orientations affect pore elongation direction
    - Porosity values control pore density per zone
    - Collagen fiber bundles create oriented channels
    - Vertical channels for perfusion (optional)
    - Tidemark layer at base (optional)
    - Surface texture for articulation
    - Zone boundary blur for smooth transitions
    - Cell density creates chondrocyte lacunae markers
    - Proteoglycan content modulates effective pore sizes
    - Legacy arrays (zone_ratios, pore_gradient) serve as fallbacks

    Args:
        params: ArticularCartilageParams specifying geometry and structure

    Returns:
        Tuple of (manifold, stats_dict)
        - manifold: The 3D geometry
        - stats_dict: Dictionary with triangle_count, volume_mm3, zone_count,
                     zone_thicknesses, scaffold_type
    """
    # Initialize random generator with seed for reproducibility
    rng = np.random.default_rng(params.seed)

    radius = params.diameter / 2

    # =========================================================================
    # TOTAL THICKNESS: Use total_thickness_um as validation/fallback
    # If total_thickness differs significantly from total_thickness_um/1000,
    # use total_thickness_um as the authoritative source
    # =========================================================================
    total_thickness_from_um = params.total_thickness_um / 1000.0  # Convert um to mm
    if abs(params.total_thickness - total_thickness_from_um) > 0.01:
        # Use um value if there's a mismatch (um is more precise)
        total_height = total_thickness_from_um
    else:
        total_height = params.total_thickness

    # =========================================================================
    # ZONE RATIOS: Use legacy zone_ratios array as fallback
    # If individual ratios are at defaults and zone_ratios differs, use array
    # =========================================================================
    default_ratios = (0.15, 0.50, 0.35)
    individual_at_default = (
        abs(params.superficial_thickness_ratio - 0.15) < 0.001 and
        abs(params.middle_thickness_ratio - 0.50) < 0.001 and
        abs(params.deep_thickness_ratio - 0.35) < 0.001
    )
    legacy_differs = (
        abs(params.zone_ratios[0] - default_ratios[0]) > 0.001 or
        abs(params.zone_ratios[1] - default_ratios[1]) > 0.001 or
        abs(params.zone_ratios[2] - default_ratios[2]) > 0.001
    )

    if individual_at_default and legacy_differs:
        # Use legacy array values
        sup_ratio_raw = params.zone_ratios[0]
        mid_ratio_raw = params.zone_ratios[1]
        deep_ratio_raw = params.zone_ratios[2]
    else:
        # Use individual params
        sup_ratio_raw = params.superficial_thickness_ratio
        mid_ratio_raw = params.middle_thickness_ratio
        deep_ratio_raw = params.deep_thickness_ratio

    # Normalize zone ratios
    total_ratio = sup_ratio_raw + mid_ratio_raw + deep_ratio_raw
    superficial_ratio = sup_ratio_raw / total_ratio
    middle_ratio = mid_ratio_raw / total_ratio
    deep_ratio = deep_ratio_raw / total_ratio

    # =========================================================================
    # PORE GRADIENT: Use legacy pore_gradient array as fallback
    # =========================================================================
    default_pores = (0.10, 0.20, 0.30)
    pores_at_default = (
        abs(params.superficial_pore_size - 0.10) < 0.001 and
        abs(params.middle_pore_size - 0.20) < 0.001 and
        abs(params.deep_pore_size - 0.30) < 0.001
    )
    legacy_pores_differ = (
        abs(params.pore_gradient[0] - default_pores[0]) > 0.001 or
        abs(params.pore_gradient[1] - default_pores[1]) > 0.001 or
        abs(params.pore_gradient[2] - default_pores[2]) > 0.001
    )

    if pores_at_default and legacy_pores_differ:
        # Use legacy array values
        superficial_pore_size = params.pore_gradient[0]
        middle_pore_size = params.pore_gradient[1]
        deep_pore_size = params.pore_gradient[2]
    else:
        # Use individual params
        superficial_pore_size = params.superficial_pore_size
        middle_pore_size = params.middle_pore_size
        deep_pore_size = params.deep_pore_size

    # =========================================================================
    # PROTEOGLYCAN MODULATION: Adjust pore sizes based on proteoglycan content
    # Higher proteoglycan = smaller pores (proteoglycans attract water, fill space)
    # Scale: 0.0 proteoglycan = 1.0x pore size, 1.0 proteoglycan = 0.7x pore size
    # =========================================================================
    proteoglycan_scale_sup = 1.0 - 0.3 * params.superficial_proteoglycan
    proteoglycan_scale_mid = 1.0 - 0.3 * params.middle_proteoglycan
    proteoglycan_scale_deep = 1.0 - 0.3 * params.deep_proteoglycan

    effective_superficial_pore_size = superficial_pore_size * proteoglycan_scale_sup
    effective_middle_pore_size = middle_pore_size * proteoglycan_scale_mid
    effective_deep_pore_size = deep_pore_size * proteoglycan_scale_deep

    # Calculate zone thicknesses
    deep_thickness = deep_ratio * total_height
    middle_thickness = middle_ratio * total_height
    superficial_thickness = superficial_ratio * total_height

    # Calculate tidemark thickness (subtracted from deep zone if enabled)
    tidemark_thickness = params.tidemark_thickness if params.enable_tidemark_layer else 0

    # Create base cylinder
    base = m3d.Manifold.cylinder(total_height, radius, radius, params.resolution)

    # Collect all pore geometries
    all_pores = []
    pore_counts = {'deep': 0, 'middle': 0, 'superficial': 0, 'tidemark': 0, 'vertical_channels': 0, 'lacunae': 0}

    # Zone boundaries (from bottom to top)
    z_tidemark_start = 0
    z_tidemark_end = tidemark_thickness
    z_deep_start = z_tidemark_end
    z_deep_end = z_deep_start + deep_thickness - tidemark_thickness
    z_middle_start = z_deep_end
    z_middle_end = z_middle_start + middle_thickness
    z_superficial_start = z_middle_end
    z_superficial_end = total_height

    # =========================================================================
    # TIDEMARK LAYER (if enabled) - at base, lower porosity
    # =========================================================================
    if params.enable_tidemark_layer and tidemark_thickness > 0:
        tidemark_pores = create_zone_pores(
            z_start=z_tidemark_start,
            z_end=z_tidemark_end,
            radius=radius,
            pore_size=effective_deep_pore_size * 0.6,  # Smaller pores, proteoglycan-adjusted
            porosity=params.tidemark_porosity,
            fiber_angle_deg=90.0,  # Perpendicular
            fiber_variance_deg=5.0,  # Minimal variance in calcified zone
            position_noise=params.position_noise * 0.5,
            pore_size_variance=params.pore_size_variance * 0.5,
            resolution=params.resolution,
            rng=rng
        )
        all_pores.extend(tidemark_pores)
        pore_counts['tidemark'] = len(tidemark_pores)

    # =========================================================================
    # PARALLEL ZONE PROCESSING
    # Create pores and fiber bundles for all zones concurrently
    # =========================================================================

    # Create RNG for extra features (surface texture, lacunae, etc.)
    rng_extras = np.random.default_rng(params.seed + 3)

    # Zone parameters for parallel execution
    deep_blur_top = params.zone_boundary_blur * deep_thickness
    middle_blur_bottom = params.zone_boundary_blur * middle_thickness
    middle_blur_top = params.zone_boundary_blur * middle_thickness
    superficial_blur_bottom = params.zone_boundary_blur * superficial_thickness

    def process_deep_zone():
        """Process deep zone pores and fiber bundles."""
        zone_pores = []
        local_rng = np.random.default_rng(params.seed)

        pores = create_zone_pores(
            z_start=z_deep_start, z_end=z_deep_end, radius=radius,
            pore_size=effective_deep_pore_size, porosity=params.deep_porosity,
            fiber_angle_deg=params.deep_fiber_orientation_deg,
            fiber_variance_deg=params.fiber_orientation_variance_deg,
            position_noise=params.position_noise, pore_size_variance=params.pore_size_variance,
            resolution=params.resolution, rng=local_rng,
            blend_top=deep_blur_top, target_porosity_top=params.middle_porosity
        )
        zone_pores.extend(pores)

        if params.enable_fiber_bundles:
            fibers = create_fiber_bundles(
                z_start=z_deep_start, z_end=z_deep_end, radius=radius,
                fiber_diameter_mm=params.collagen_fiber_diameter_um / 1000.0,
                fiber_spacing_mm=params.collagen_fiber_spacing_um / 1000.0,
                fiber_angle_deg=params.deep_fiber_orientation_deg,
                resolution=params.resolution, rng=local_rng
            )
            zone_pores.extend(fibers)

        return ('deep', pores, zone_pores)

    def process_middle_zone():
        """Process middle zone pores and fiber bundles."""
        zone_pores = []
        local_rng = np.random.default_rng(params.seed + 1)

        pores = create_zone_pores(
            z_start=z_middle_start, z_end=z_middle_end, radius=radius,
            pore_size=effective_middle_pore_size, porosity=params.middle_porosity,
            fiber_angle_deg=params.middle_fiber_orientation_deg,
            fiber_variance_deg=params.fiber_orientation_variance_deg,
            position_noise=params.position_noise, pore_size_variance=params.pore_size_variance,
            resolution=params.resolution, rng=local_rng,
            blend_bottom=middle_blur_bottom, target_porosity_bottom=params.deep_porosity,
            blend_top=middle_blur_top, target_porosity_top=params.superficial_porosity
        )
        zone_pores.extend(pores)

        if params.enable_fiber_bundles:
            fibers = create_fiber_bundles(
                z_start=z_middle_start, z_end=z_middle_end, radius=radius,
                fiber_diameter_mm=params.collagen_fiber_diameter_um / 1000.0,
                fiber_spacing_mm=params.collagen_fiber_spacing_um / 1000.0,
                fiber_angle_deg=params.middle_fiber_orientation_deg,
                resolution=params.resolution, rng=local_rng
            )
            zone_pores.extend(fibers)

        return ('middle', pores, zone_pores)

    def process_superficial_zone():
        """Process superficial zone pores and fiber bundles."""
        zone_pores = []
        local_rng = np.random.default_rng(params.seed + 2)

        pores = create_zone_pores(
            z_start=z_superficial_start, z_end=z_superficial_end, radius=radius,
            pore_size=effective_superficial_pore_size, porosity=params.superficial_porosity,
            fiber_angle_deg=params.superficial_fiber_orientation_deg,
            fiber_variance_deg=params.fiber_orientation_variance_deg,
            position_noise=params.position_noise, pore_size_variance=params.pore_size_variance,
            resolution=params.resolution, rng=local_rng,
            blend_bottom=superficial_blur_bottom, target_porosity_bottom=params.middle_porosity
        )
        zone_pores.extend(pores)

        if params.enable_fiber_bundles:
            fibers = create_fiber_bundles(
                z_start=z_superficial_start, z_end=z_superficial_end, radius=radius,
                fiber_diameter_mm=params.collagen_fiber_diameter_um / 1000.0 * 0.5,
                fiber_spacing_mm=params.collagen_fiber_spacing_um / 1000.0 * 0.7,
                fiber_angle_deg=params.superficial_fiber_orientation_deg,
                resolution=params.resolution, rng=local_rng
            )
            zone_pores.extend(fibers)

        return ('superficial', pores, zone_pores)

    # Execute zone processing in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(process_deep_zone),
            executor.submit(process_middle_zone),
            executor.submit(process_superficial_zone)
        ]

        for future in as_completed(futures):
            zone_name, pores_only, zone_pores = future.result()
            all_pores.extend(zone_pores)
            pore_counts[zone_name] = len(pores_only)

    # =========================================================================
    # VERTICAL CHANNELS (if enabled) - for scaffold perfusion
    # =========================================================================
    if params.enable_vertical_channels:
        vertical_channels = create_vertical_channels(
            z_start=z_deep_start, z_end=z_deep_end, radius=radius,
            channel_diameter=params.vertical_channel_diameter,
            channel_spacing=params.vertical_channel_spacing,
            resolution=params.resolution
        )
        all_pores.extend(vertical_channels)
        pore_counts['vertical_channels'] = len(vertical_channels)

    # =========================================================================
    # HORIZONTAL CHANNELS IN SUPERFICIAL ZONE (if enabled)
    # =========================================================================
    if params.enable_horizontal_channels:
        horizontal_channels = create_horizontal_channels(
            z_start=z_superficial_start, z_end=z_superficial_end, radius=radius,
            pore_radius=params.superficial_pore_size * 0.5,
            resolution=params.resolution, rng=rng_extras
        )
        all_pores.extend(horizontal_channels)

    # =========================================================================
    # SURFACE TEXTURE (if enabled)
    # =========================================================================
    if params.enable_surface_texture and params.surface_roughness > 0:
        surface_features = create_surface_texture(
            z_top=total_height, radius=radius, roughness=params.surface_roughness,
            resolution=params.resolution, rng=rng_extras
        )
        all_pores.extend(surface_features)

    # =========================================================================
    # CHONDROCYTE LACUNAE (based on cell density)
    # =========================================================================
    lacunae = create_chondrocyte_lacunae(
        z_deep_start=z_deep_start, z_deep_end=z_deep_end,
        z_middle_start=z_middle_start, z_middle_end=z_middle_end,
        z_superficial_start=z_superficial_start, z_superficial_end=z_superficial_end,
        radius=radius,
        superficial_cell_density=params.superficial_cell_density,
        middle_cell_density=params.middle_cell_density,
        deep_cell_density=params.deep_cell_density,
        resolution=params.resolution, rng=rng_extras
    )
    all_pores.extend(lacunae)
    pore_counts['lacunae'] = len(lacunae)

    # =========================================================================
    # SUBTRACT ALL PORES FROM BASE
    # Parallel batch processing for improved performance
    # =========================================================================
    result = base
    if all_pores:
        # Split pores into batches and union each batch in parallel
        batch_size = 400
        batches = [all_pores[i:i + batch_size] for i in range(0, len(all_pores), batch_size)]

        # Parallel union of batches
        def union_batch(batch):
            return batch_union(batch)

        with ThreadPoolExecutor(max_workers=min(4, len(batches))) as executor:
            batch_results = list(executor.map(union_batch, batches))

        # Combine batch results and subtract from base
        valid_batches = [b for b in batch_results if b is not None]
        if valid_batches:
            # Union all batch results together
            combined_pores = batch_union(valid_batches)
            if combined_pores is not None:
                result = result - combined_pores

    # =========================================================================
    # CALCULATE STATISTICS
    # Note: We defer to_mesh() since it's the slowest operation (99% of time).
    # Triangle count is estimated from pore count; actual count available on export.
    # =========================================================================
    volume = result.volume() if hasattr(result, 'volume') else 0
    base_volume = np.pi * radius**2 * total_height
    actual_porosity = 1.0 - (volume / base_volume) if base_volume > 0 else 0

    # Estimate triangle count (actual mesh conversion deferred to export)
    # Each pore adds ~100-300 triangles depending on resolution
    estimated_triangles = int(len(all_pores) * 150 * (params.resolution / 16))

    stats = {
        'triangle_count': estimated_triangles,  # Estimated; actual on export
        'volume_mm3': volume,
        'zone_count': 3 + (1 if params.enable_tidemark_layer else 0),
        'zone_thicknesses_mm': {
            'superficial': superficial_thickness,
            'middle': middle_thickness,
            'deep': deep_thickness - tidemark_thickness,
            'tidemark': tidemark_thickness
        },
        'pore_counts': pore_counts,
        'total_pore_count': sum(pore_counts.values()),
        'target_porosity': {
            'superficial': params.superficial_porosity,
            'middle': params.middle_porosity,
            'deep': params.deep_porosity,
            'tidemark': params.tidemark_porosity if params.enable_tidemark_layer else None
        },
        'actual_porosity': actual_porosity,
        'fiber_orientations_deg': {
            'superficial': params.superficial_fiber_orientation_deg,
            'middle': params.middle_fiber_orientation_deg,
            'deep': params.deep_fiber_orientation_deg
        },
        'features_enabled': {
            'fiber_bundles': params.enable_fiber_bundles,
            'vertical_channels': params.enable_vertical_channels,
            'horizontal_channels': params.enable_horizontal_channels,
            'tidemark_layer': params.enable_tidemark_layer,
            'surface_texture': params.enable_surface_texture
        },
        'proteoglycan_modulation': {
            'superficial_scale': proteoglycan_scale_sup,
            'middle_scale': proteoglycan_scale_mid,
            'deep_scale': proteoglycan_scale_deep
        },
        'effective_pore_sizes_mm': {
            'superficial': effective_superficial_pore_size,
            'middle': effective_middle_pore_size,
            'deep': effective_deep_pore_size
        },
        'cell_densities_per_mm3': {
            'superficial': params.superficial_cell_density,
            'middle': params.middle_cell_density,
            'deep': params.deep_cell_density
        },
        'scaffold_type': 'articular_cartilage'
    }

    return result, stats


def create_zone_pores(
    z_start: float,
    z_end: float,
    radius: float,
    pore_size: float,
    porosity: float,
    fiber_angle_deg: float,
    fiber_variance_deg: float,
    position_noise: float,
    pore_size_variance: float,
    resolution: int,
    rng: np.random.Generator,
    blend_bottom: float = 0.0,
    target_porosity_bottom: float = 0.0,
    blend_top: float = 0.0,
    target_porosity_top: float = 0.0
) -> List[m3d.Manifold]:
    """
    Create pores for one cartilage zone with fiber orientation.

    Pores are elongated in the direction of the fiber angle to simulate
    the orientation of collagen fibers within each zone.

    Args:
        z_start: Start Z coordinate of zone
        z_end: End Z coordinate of zone
        radius: Outer radius of scaffold
        pore_size: Base pore size (diameter) in mm
        porosity: Target porosity (0-1) controlling pore density
        fiber_angle_deg: Fiber orientation angle (0=horizontal, 90=vertical)
        fiber_variance_deg: Random variance in fiber angle
        position_noise: Random jitter in pore positions (0-1)
        pore_size_variance: Random variance in pore sizes (0-1)
        resolution: Angular resolution for geometry
        rng: Random number generator
        blend_bottom: Height of blend region at bottom
        target_porosity_bottom: Porosity to blend toward at bottom
        blend_top: Height of blend region at top
        target_porosity_top: Porosity to blend toward at top

    Returns:
        List of pore manifolds
    """
    pores = []
    height = z_end - z_start

    if height <= 0 or pore_size <= 0:
        return pores

    # Calculate pore spacing based on porosity
    # Higher porosity = more pores = closer spacing
    base_spacing = pore_size / (porosity ** 0.33) * 1.5

    # Calculate number of pores needed
    n_radial = max(1, int(radius / base_spacing))
    n_vertical = max(1, int(height / base_spacing))

    pore_radius = pore_size / 2

    # Pre-create sphere template for reuse (major optimization - avoids recreating geometry)
    sphere_template = m3d.Manifold.sphere(1.0, resolution)

    for i in range(n_radial):
        r = (i + 0.5) * (radius / n_radial) * 0.85
        circumference = 2 * np.pi * r
        n_angular = max(3, int(circumference / base_spacing))

        for j in range(n_angular):
            base_angle = j * 2 * np.pi / n_angular

            for k in range(n_vertical):
                base_z = z_start + (k + 0.5) * (height / n_vertical)

                # Calculate local porosity with boundary blending
                local_porosity = porosity
                z_relative = base_z - z_start

                if blend_bottom > 0 and z_relative < blend_bottom:
                    blend_factor = z_relative / blend_bottom
                    local_porosity = (1 - blend_factor) * target_porosity_bottom + blend_factor * porosity

                if blend_top > 0 and (height - z_relative) < blend_top:
                    blend_factor = (height - z_relative) / blend_top
                    local_porosity = (1 - blend_factor) * target_porosity_top + blend_factor * porosity

                # Skip some pores probabilistically based on local porosity variation
                if rng.random() > local_porosity / porosity:
                    continue

                # Apply position noise
                angle = base_angle + (rng.random() - 0.5) * 2 * np.pi * position_noise / n_angular
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                z = base_z + (rng.random() - 0.5) * 2 * (height / n_vertical) * position_noise

                # Clamp z within zone
                z = max(z_start + pore_radius, min(z_end - pore_radius, z))

                # Apply pore size variance
                current_pore_radius = pore_radius * (1 + (rng.random() - 0.5) * 2 * pore_size_variance)
                current_pore_radius = max(pore_radius * 0.5, current_pore_radius)

                # Calculate actual fiber angle with variance
                actual_fiber_angle = fiber_angle_deg + (rng.random() - 0.5) * 2 * fiber_variance_deg

                # Create oriented pore using template (optimized)
                pore = create_oriented_pore_from_template(
                    sphere_template,
                    x, y, z,
                    current_pore_radius,
                    actual_fiber_angle,
                    rng
                )
                pores.append(pore)

    return pores


def create_oriented_pore_from_template(
    sphere_template: m3d.Manifold,
    x: float,
    y: float,
    z: float,
    radius: float,
    fiber_angle_deg: float,
    rng: np.random.Generator
) -> m3d.Manifold:
    """
    Create a pore using a pre-created sphere template (optimized version).

    Preserves full biological accuracy with proper elongation and orientation
    while reusing the template sphere for performance.

    Args:
        sphere_template: Pre-created unit sphere to reuse
        x, y, z: Pore center position
        radius: Base pore radius
        fiber_angle_deg: Fiber orientation (0=horizontal, 90=vertical)
        rng: Random number generator

    Returns:
        Manifold representing the oriented pore
    """
    # Full biological elongation factor (1.5-2.0 along fiber direction)
    elongation = 1.5 + rng.random() * 0.5
    fiber_angle_rad = np.radians(fiber_angle_deg)

    if fiber_angle_deg < 15:
        # Superficial zone: horizontal ellipsoid (tangent to surface)
        random_azimuth = rng.random() * 2 * np.pi
        scale_x = max(0.5, abs(elongation * np.cos(random_azimuth)))
        scale_y = max(0.5, abs(elongation * np.sin(random_azimuth)))
        scale_z = 1.0
        pore = sphere_template.scale([radius * scale_x, radius * scale_y, radius * scale_z])

    elif fiber_angle_deg > 75:
        # Deep zone: vertical ellipsoid (perpendicular to surface)
        pore = sphere_template.scale([radius, radius, radius * elongation])

    else:
        # Middle zone: oblique orientation
        scale_x = 1.0 + (elongation - 1.0) * np.cos(fiber_angle_rad)
        scale_z = 1.0 + (elongation - 1.0) * np.sin(fiber_angle_rad)
        rotation_xy = (rng.random() - 0.5) * 30  # ±15 degrees
        pore = sphere_template.scale([radius * scale_x, radius, radius * scale_z])
        pore = pore.rotate([0, 0, rotation_xy])

    return pore.translate([x, y, z])


def create_fiber_bundles(
    z_start: float,
    z_end: float,
    radius: float,
    fiber_diameter_mm: float,
    fiber_spacing_mm: float,
    fiber_angle_deg: float,
    resolution: int,
    rng: np.random.Generator,
    max_fibers: int = 20000  # Performance limit per zone
) -> List[m3d.Manifold]:
    """
    Create collagen fiber bundle channels.

    These represent the spaces between collagen fiber bundles, modeled as
    elongated cylindrical channels oriented along the fiber direction.

    Args:
        z_start: Start Z coordinate
        z_end: End Z coordinate
        radius: Outer radius of scaffold
        fiber_diameter_mm: Diameter of fiber channel in mm
        fiber_spacing_mm: Spacing between fibers in mm
        fiber_angle_deg: Fiber orientation angle
        resolution: Angular resolution
        rng: Random number generator
        max_fibers: Maximum fibers per zone (performance limit)

    Returns:
        List of fiber channel manifolds
    """
    fibers = []
    height = z_end - z_start

    if height <= 0 or fiber_diameter_mm <= 0:
        return fibers

    fiber_radius = fiber_diameter_mm / 2

    # Number of fibers based on spacing
    n_fibers_radial = max(2, int(radius / fiber_spacing_mm))
    n_fibers_vertical = max(2, int(height / fiber_spacing_mm))

    # Pre-calculate fiber lengths for each zone type
    vertical_fiber_length = min(height * 0.8, fiber_spacing_mm * 2)
    horizontal_fiber_length = min(radius * 0.3, fiber_spacing_mm * 2)
    oblique_fiber_length = min(height * 0.4, fiber_spacing_mm * 2)

    # Pre-create cylinder templates for each orientation (major optimization)
    if fiber_angle_deg > 75:
        cylinder_template = m3d.Manifold.cylinder(vertical_fiber_length, fiber_radius, fiber_radius, resolution)
    elif fiber_angle_deg < 15:
        cylinder_template = m3d.Manifold.cylinder(horizontal_fiber_length, fiber_radius, fiber_radius, resolution)
        cylinder_template = cylinder_template.rotate([0, 90, 0])  # Pre-rotate to horizontal
    else:
        cylinder_template = m3d.Manifold.cylinder(oblique_fiber_length, fiber_radius, fiber_radius, resolution)
        cylinder_template = cylinder_template.rotate([0, 90 - fiber_angle_deg, 0])  # Pre-tilt

    for i in range(n_fibers_radial):
        r = (i + 0.5) * (radius / n_fibers_radial) * 0.8
        n_angular = max(4, int(2 * np.pi * r / fiber_spacing_mm))

        for j in range(n_angular):
            angle = j * 2 * np.pi / n_angular + (rng.random() - 0.5) * 0.2

            for k in range(n_fibers_vertical):
                z = z_start + (k + 0.5) * (height / n_fibers_vertical)

                x = r * np.cos(angle)
                y = r * np.sin(angle)

                # Skip some randomly for organic appearance
                if rng.random() > 0.7:
                    continue

                # Create fiber from template with position-specific transforms
                if fiber_angle_deg > 75:
                    # Vertical fiber (deep zone)
                    fiber = cylinder_template.translate([x, y, z - vertical_fiber_length/2])
                elif fiber_angle_deg < 15:
                    # Horizontal fiber (superficial zone)
                    fiber = cylinder_template.rotate([0, 0, np.degrees(angle)])
                    fiber = fiber.translate([x, y, z])
                else:
                    # Oblique fiber (middle zone)
                    fiber = cylinder_template.rotate([0, 0, np.degrees(angle)])
                    fiber = fiber.translate([x, y, z])

                fibers.append(fiber)

    return fibers


def create_vertical_channels(
    z_start: float,
    z_end: float,
    radius: float,
    channel_diameter: float,
    channel_spacing: float,
    resolution: int
) -> List[m3d.Manifold]:
    """
    Create vertical through-channels for scaffold perfusion.

    NOTE: Mature articular cartilage is AVASCULAR. These channels are only
    for engineered scaffold designs to allow nutrient perfusion and cell
    seeding. Only created if enable_vertical_channels=True.

    Args:
        z_start: Start Z coordinate
        z_end: End Z coordinate
        radius: Outer radius of scaffold
        channel_diameter: Diameter of channels in mm
        channel_spacing: Spacing between channels in mm
        resolution: Angular resolution

    Returns:
        List of vertical channel manifolds
    """
    channels = []
    height = z_end - z_start
    channel_radius = channel_diameter / 2

    if height <= 0 or channel_diameter <= 0:
        return channels

    # Create grid of vertical channels
    n_radial = max(1, int(radius / channel_spacing))

    for i in range(n_radial):
        r = (i + 0.5) * (radius / n_radial) * 0.7  # Stay well within bounds
        circumference = 2 * np.pi * r
        n_angular = max(3, int(circumference / channel_spacing))

        for j in range(n_angular):
            angle = j * 2 * np.pi / n_angular
            x = r * np.cos(angle)
            y = r * np.sin(angle)

            # Vertical cylinder through the zone
            channel = m3d.Manifold.cylinder(height * 1.1, channel_radius, channel_radius, resolution)
            channel = channel.translate([x, y, z_start - height * 0.05])
            channels.append(channel)

    return channels


def create_horizontal_channels(
    z_start: float,
    z_end: float,
    radius: float,
    pore_radius: float,
    resolution: int,
    rng: np.random.Generator
) -> List[m3d.Manifold]:
    """
    Create tangential (horizontal) channels for superficial zone.

    These represent the tangential orientation of collagen fibers in
    the superficial zone, parallel to the articular surface.

    Args:
        z_start: Start Z coordinate
        z_end: End Z coordinate
        radius: Outer radius of scaffold
        pore_radius: Radius of channels
        resolution: Angular resolution
        rng: Random number generator

    Returns:
        List of horizontal channel manifolds
    """
    pores = []
    height = z_end - z_start
    spacing = pore_radius * 3
    n_layers = max(2, int(height / spacing))

    for layer in range(n_layers):
        z = z_start + (layer + 0.5) * (height / n_layers)

        # Radial channels at this height
        n_radial = 8 + int(rng.random() * 4)  # 8-12 channels
        for i in range(n_radial):
            angle = i * 2 * np.pi / n_radial + (rng.random() - 0.5) * 0.3

            # Channel from near center to edge
            length = radius * 0.75
            cylinder = m3d.Manifold.cylinder(length, pore_radius, pore_radius, resolution)

            # Rotate to horizontal and point outward
            cylinder = cylinder.rotate([0, 90, 0])  # Make horizontal
            cylinder = cylinder.rotate([0, 0, np.degrees(angle)])  # Rotate around Z
            cylinder = cylinder.translate([0, 0, z])

            pores.append(cylinder)

    return pores


def create_surface_texture(
    z_top: float,
    radius: float,
    roughness: float,
    resolution: int,
    rng: np.random.Generator
) -> List[m3d.Manifold]:
    """
    Create surface texture for articular surface.

    Native articular cartilage has an extremely smooth surface (0.1-1μm Ra).
    This creates very subtle undulations to simulate the smooth articular
    gliding surface. Does NOT add prominent bumps.

    Args:
        z_top: Top Z coordinate (articular surface)
        radius: Outer radius of scaffold
        roughness: Surface roughness factor (very small values)
        resolution: Angular resolution
        rng: Random number generator

    Returns:
        List of surface feature manifolds (shallow indentations)
    """
    features = []

    # Very subtle surface undulations
    # Only add if roughness is non-negligible
    if roughness < 0.001:
        return features

    # Create small shallow indentations
    indent_radius = roughness * 2  # Very small radius
    indent_depth = roughness * 0.5  # Very shallow

    n_features = max(5, int(radius / (roughness * 10)))
    for _ in range(n_features):
        r = rng.random() * radius * 0.9
        angle = rng.random() * 2 * np.pi
        x = r * np.cos(angle)
        y = r * np.sin(angle)

        # Small sphere just at surface level (creates shallow dimple)
        feature = m3d.Manifold.sphere(indent_radius, max(4, resolution // 4))
        feature = feature.scale([1.0, 1.0, 0.3])  # Flatten
        feature = feature.translate([x, y, z_top - indent_depth])
        features.append(feature)

    return features


def create_chondrocyte_lacunae(
    z_deep_start: float,
    z_deep_end: float,
    z_middle_start: float,
    z_middle_end: float,
    z_superficial_start: float,
    z_superficial_end: float,
    radius: float,
    superficial_cell_density: float,
    middle_cell_density: float,
    deep_cell_density: float,
    resolution: int,
    rng: np.random.Generator
) -> List[m3d.Manifold]:
    """
    Create chondrocyte lacunae (cell housing voids) based on cell density.

    Chondrocyte lacunae are small (~10-15 μm diameter) spherical voids in the
    cartilage matrix that house individual chondrocytes. Higher cell density
    zones have more lacunae markers.

    Cell density is given in cells/mm³. We scale down to create representative
    markers (not 1:1 since that would be too many small features).

    Args:
        z_deep_start, z_deep_end: Deep zone Z boundaries
        z_middle_start, z_middle_end: Middle zone Z boundaries
        z_superficial_start, z_superficial_end: Superficial zone Z boundaries
        radius: Outer radius of scaffold
        superficial_cell_density: Cells/mm³ in superficial zone
        middle_cell_density: Cells/mm³ in middle zone
        deep_cell_density: Cells/mm³ in deep zone
        resolution: Angular resolution
        rng: Random number generator

    Returns:
        List of lacunae manifolds (small spherical voids)
    """
    lacunae = []

    # Lacuna dimensions (biological range: 10-15 μm diameter)
    # We use slightly larger for visibility in scaffold
    lacuna_diameter_mm = 0.015  # 15 μm = 0.015 mm
    lacuna_radius = lacuna_diameter_mm / 2

    # Scale factor: we don't create one lacuna per cell (way too many)
    # Instead, create representative markers at ~1% of actual density
    # This provides visual indication of cell density differences
    scale_factor = 0.0001  # 0.01% of actual density

    # Calculate zone volumes (cylindrical approximation)
    def zone_volume(z_start: float, z_end: float) -> float:
        height = z_end - z_start
        return np.pi * (radius * 0.85) ** 2 * height  # Use inner 85% radius

    # Pre-create lacuna templates for each zone type (optimization)
    lacuna_resolution = max(4, resolution // 4)
    sphere_template = m3d.Manifold.sphere(lacuna_radius, lacuna_resolution)

    # Pre-scale templates for each zone
    superficial_template = sphere_template.scale([1.2, 1.2, 0.6])  # Flattened
    middle_template = sphere_template  # Spherical
    deep_template = sphere_template.scale([0.8, 0.8, 1.3])  # Elongated vertically

    # Create lacunae for each zone
    zone_specs = [
        ('superficial', z_superficial_start, z_superficial_end, superficial_cell_density, superficial_template),
        ('middle', z_middle_start, z_middle_end, middle_cell_density, middle_template),
        ('deep', z_deep_start, z_deep_end, deep_cell_density, deep_template),
    ]

    for zone_name, z_start, z_end, cell_density, template in zone_specs:
        height = z_end - z_start
        if height <= 0:
            continue

        volume = zone_volume(z_start, z_end)
        n_lacunae = max(1, int(cell_density * volume * scale_factor))

        # Cap at reasonable number to avoid performance issues
        n_lacunae = min(n_lacunae, 200)

        for _ in range(n_lacunae):
            # Random position within zone
            r = rng.random() ** 0.5 * radius * 0.85  # Square root for uniform area distribution
            angle = rng.random() * 2 * np.pi
            z = z_start + rng.random() * height

            x = r * np.cos(angle)
            y = r * np.sin(angle)

            # Use pre-scaled template
            lacuna = template.translate([x, y, z])
            lacunae.append(lacuna)

    return lacunae


def generate_articular_cartilage_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate articular cartilage from dictionary parameters (for API compatibility).

    Args:
        params: Dictionary with keys matching ArticularCartilageParams fields

    Returns:
        Tuple of (manifold, stats_dict)
    """
    zone_ratios = params.get('zone_ratios', [0.15, 0.50, 0.35])
    if isinstance(zone_ratios, list):
        zone_ratios = tuple(zone_ratios)

    pore_gradient = params.get('pore_gradient', [0.10, 0.20, 0.30])
    if isinstance(pore_gradient, list):
        pore_gradient = tuple(pore_gradient)

    return generate_articular_cartilage(ArticularCartilageParams(
        # Basic geometry
        diameter=params.get('diameter', 8.0),
        resolution=params.get('resolution', 16),
        seed=params.get('seed', 42),

        # Thickness
        total_thickness=params.get('total_thickness', 2.5),
        total_thickness_um=params.get('total_thickness_um', 2500.0),

        # Zone ratios
        superficial_thickness_ratio=params.get('superficial_thickness_ratio', 0.15),
        middle_thickness_ratio=params.get('middle_thickness_ratio', 0.50),
        deep_thickness_ratio=params.get('deep_thickness_ratio', 0.35),
        zone_ratios=zone_ratios,

        # Fiber orientation
        superficial_fiber_orientation_deg=params.get('superficial_fiber_orientation_deg', 0.0),
        middle_fiber_orientation_deg=params.get('middle_fiber_orientation_deg', 45.0),
        deep_fiber_orientation_deg=params.get('deep_fiber_orientation_deg', 90.0),
        fiber_orientation_variance_deg=params.get('fiber_orientation_variance_deg', 15.0),

        # Pore sizes
        superficial_pore_size=params.get('superficial_pore_size', 0.10),
        middle_pore_size=params.get('middle_pore_size', 0.20),
        deep_pore_size=params.get('deep_pore_size', 0.30),
        pore_gradient=pore_gradient,

        # Porosity
        superficial_porosity=params.get('superficial_porosity', 0.70),
        middle_porosity=params.get('middle_porosity', 0.80),
        deep_porosity=params.get('deep_porosity', 0.85),

        # Cell density
        superficial_cell_density=params.get('superficial_cell_density', 10000.0),
        middle_cell_density=params.get('middle_cell_density', 5000.0),
        deep_cell_density=params.get('deep_cell_density', 3000.0),

        # Proteoglycan content
        superficial_proteoglycan=params.get('superficial_proteoglycan', 0.3),
        middle_proteoglycan=params.get('middle_proteoglycan', 0.7),
        deep_proteoglycan=params.get('deep_proteoglycan', 1.0),

        # Collagen fibers
        collagen_fiber_diameter_um=params.get('collagen_fiber_diameter_um', 50.0),
        collagen_fiber_spacing_um=params.get('collagen_fiber_spacing_um', 100.0),
        enable_fiber_bundles=params.get('enable_fiber_bundles', True),

        # Channels
        enable_vertical_channels=params.get('enable_vertical_channels', True),
        vertical_channel_diameter=params.get('vertical_channel_diameter', 0.15),
        vertical_channel_spacing=params.get('vertical_channel_spacing', 0.5),
        enable_horizontal_channels=params.get('enable_horizontal_channels', False),

        # Surface
        surface_roughness=params.get('surface_roughness', 0.05),
        enable_surface_texture=params.get('enable_surface_texture', True),

        # Tidemark
        enable_tidemark_layer=params.get('enable_tidemark_layer', False),
        tidemark_thickness=params.get('tidemark_thickness', 0.1),
        tidemark_porosity=params.get('tidemark_porosity', 0.60),

        # Randomization
        position_noise=params.get('position_noise', 0.15),
        pore_size_variance=params.get('pore_size_variance', 0.2),
        zone_boundary_blur=params.get('zone_boundary_blur', 0.1),
    ))
