"""
Tendon/ligament scaffold generator with aligned parallel fiber bundles.

Creates a structure with aligned parallel fibers exhibiting characteristic
crimped (sinusoidal) patterns typical of tendon and ligament tissue.
"""

from __future__ import annotations
import manifold3d as m3d
import numpy as np
from dataclasses import dataclass
from ..core import batch_union


@dataclass
class TendonLigamentParams:
    """Parameters for tendon/ligament scaffold generation.

    Biologically realistic parameters based on native tendon/ligament structure:
    - Hierarchical organization: collagen molecules -> fibrils -> fibers -> fascicles -> tendon
    - Fiber diameter: 1-20 um (fibrils), 50-500 um (fibers), 1-3 mm (fascicles)
    - Crimp pattern: wavelength 20-200 um, amplitude ~10% of wavelength
    - Typical tendon length: 10-200 mm depending on location
    - Porosity: relatively low, ~10-30% depending on tissue
    """
    # Basic geometry (mm)
    length: float = 30.0  # Total length
    width: float = 8.0  # Total width
    thickness: float = 3.0  # Total thickness
    resolution: int = 12
    seed: int = 42

    # Fascicle organization (highest level bundles)
    bundle_count: int = 5  # Number of primary fascicles
    fascicle_diameter: float = 1.5  # Fascicle diameter (mm)
    fascicle_spacing: float = 0.3  # Gap between fascicles (mm)
    enable_fascicle_boundaries: bool = True  # Show endotenon boundaries

    # Fiber properties (within fascicles)
    fiber_diameter: float = 0.15  # Fiber diameter (mm)
    fiber_diameter_um: float = 150.0  # Fiber diameter in micrometers
    fiber_spacing: float = 0.4  # Spacing between fiber centers (mm)
    fiber_spacing_um: float = 400.0  # Spacing in micrometers
    fibers_per_fascicle: int = 20  # Number of fibers per fascicle

    # Fibril properties (within fibers) - for high detail scaffolds
    fibril_diameter_um: float = 5.0  # Individual fibril diameter
    fibril_spacing_um: float = 10.0  # Spacing between fibrils
    enable_fibril_detail: bool = False  # Include fibril-level detail

    # Crimp pattern (sinusoidal wave)
    crimp_amplitude: float = 0.3  # Wave amplitude (mm)
    crimp_amplitude_um: float = 300.0  # Wave amplitude in micrometers
    crimp_wavelength: float = 2.0  # Wave period (mm)
    crimp_wavelength_um: float = 300.0  # Wave period in micrometers
    crimp_angle_deg: float = 10.0  # Crimp angle (typically 5-15 degrees)
    crimp_variance: float = 0.1  # Random variation in crimp pattern

    # Endotenon (interfascicular matrix)
    enable_endotenon: bool = True  # Connective tissue between fascicles
    endotenon_thickness: float = 0.05  # Endotenon layer thickness (mm)
    endotenon_porosity: float = 0.5  # Higher porosity for blood/nerve supply

    # Epitenon (outer sheath)
    enable_epitenon: bool = True  # Outer connective tissue sheath
    epitenon_thickness: float = 0.1  # Epitenon layer thickness (mm)
    epitenon_porosity: float = 0.3  # Moderate porosity

    # Paratenon (sliding layer - for certain tendons)
    enable_paratenon: bool = False  # Loose areolar tissue layer
    paratenon_thickness: float = 0.2  # Paratenon layer thickness

    # Collagen alignment
    primary_fiber_angle_deg: float = 0.0  # Main fiber direction (along length)
    fiber_angle_variance_deg: float = 5.0  # Slight angular variation
    enable_cross_links: bool = True  # Cross-linking between fibers
    cross_link_density: float = 0.1  # Cross-links per unit length

    # Vascular features
    enable_vascular_channels: bool = True  # Blood vessel channels
    vascular_channel_diameter: float = 0.08  # Small vessel diameter (mm)
    vascular_channel_spacing: float = 1.5  # Spacing between vessels
    vascular_channel_pattern: str = 'longitudinal'  # longitudinal or spiral

    # Insertion zones (enthesis)
    enable_enthesis_transition: bool = False  # Bone insertion zone
    enthesis_length: float = 3.0  # Length of transition zone (mm)
    enthesis_mineralization_gradient: float = 0.5  # Mineral content gradient

    # Mechanical property indicators
    target_stiffness_mpa: float = 1000.0  # Target elastic modulus indicator
    target_ultimate_stress_mpa: float = 100.0  # Ultimate tensile strength indicator

    # Porosity
    porosity: float = 0.15  # Overall porosity (tendons are relatively dense)
    pore_size_um: float = 50.0  # Interfiber pore size

    # Surface characteristics
    surface_roughness: float = 0.02  # Surface texture
    enable_surface_texture: bool = True  # Textured exterior

    # Randomization
    position_noise: float = 0.05  # Random position jitter
    diameter_variance: float = 0.1  # Variation in fiber diameter
    spacing_variance: float = 0.1  # Variation in fiber spacing


def generate_tendon_ligament(params: TendonLigamentParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate a tendon/ligament scaffold with aligned crimped fibers.

    Creates parallel fiber bundles with sinusoidal crimp pattern:
    - Fibers aligned along length direction
    - Crimp amplitude creates characteristic wave pattern
    - Multiple bundles organized in parallel
    - Fascicle boundaries (endotenon)
    - Tissue layers (epitenon, paratenon)
    - Vascular channels
    - Enthesis transition zones

    Args:
        params: TendonLigamentParams specifying geometry and structure

    Returns:
        Tuple of (manifold, stats_dict)
        - manifold: The 3D geometry
        - stats_dict: Dictionary with triangle_count, volume_mm3, fiber_count,
                     bundle_count, scaffold_type
    """
    # Initialize RNG with seed for reproducibility
    rng = np.random.default_rng(params.seed)

    # Apply unit conversions: use _um params if they differ from defaults while mm params are default
    # This allows users to specify either mm or um values
    fiber_diameter = _resolve_um_param(
        params.fiber_diameter, 0.15, params.fiber_diameter_um, 150.0
    )
    fiber_spacing = _resolve_um_param(
        params.fiber_spacing, 0.4, params.fiber_spacing_um, 400.0
    )
    crimp_amplitude = _resolve_um_param(
        params.crimp_amplitude, 0.3, params.crimp_amplitude_um, 300.0
    )
    crimp_wavelength = _resolve_um_param(
        params.crimp_wavelength, 2.0, params.crimp_wavelength_um, 2000.0
    )

    # Apply mechanical property influence on fiber characteristics
    # Higher stiffness target = denser packing (reduce spacing variance)
    # Higher ultimate stress target = thicker fibers (increase diameter)
    stiffness_factor = params.target_stiffness_mpa / 1000.0  # Normalize to typical value
    stress_factor = params.target_ultimate_stress_mpa / 100.0  # Normalize to typical value

    # Adjust spacing variance: higher stiffness = more uniform (lower variance)
    adjusted_spacing_variance = params.spacing_variance / max(0.5, stiffness_factor)

    # Adjust diameter variance and base diameter: higher stress = slightly thicker fibers
    diameter_multiplier = 1.0 + 0.1 * (stress_factor - 1.0)  # +-10% per stress factor
    fiber_diameter *= np.clip(diameter_multiplier, 0.8, 1.3)

    fiber_radius = fiber_diameter / 2
    all_components = []
    total_fiber_count = 0
    fascicle_boundaries = []
    cross_links_list = []
    fibril_components = []

    # Convert primary fiber angle to radians
    primary_angle_rad = np.radians(params.primary_fiber_angle_deg)
    angle_variance_rad = np.radians(params.fiber_angle_variance_deg)

    # Calculate fascicle layout
    # Distribute fascicles in the cross-section
    fascicle_positions = _calculate_fascicle_positions(
        params.width, params.thickness, params.bundle_count,
        params.fascicle_diameter, params.fascicle_spacing, rng
    )

    for bundle_idx, (fascicle_y, fascicle_z) in enumerate(fascicle_positions):
        # Apply fiber angle with variance for this fascicle
        fascicle_angle = primary_angle_rad + rng.uniform(-angle_variance_rad, angle_variance_rad)

        # Create fascicle boundary (endotenon) if enabled
        if params.enable_fascicle_boundaries and params.enable_endotenon:
            boundary = _create_fascicle_boundary(
                length=params.length,
                center_y=fascicle_y,
                center_z=fascicle_z,
                diameter=params.fascicle_diameter,
                thickness=params.endotenon_thickness,
                porosity=params.endotenon_porosity,
                resolution=params.resolution,
                rng=rng
            )
            if boundary is not None and boundary.num_vert() > 0:
                fascicle_boundaries.append(boundary)

        # Create fibers within this fascicle
        bundle_fibers, fiber_positions = _create_fascicle_fibers(
            length=params.length,
            fascicle_center_y=fascicle_y,
            fascicle_center_z=fascicle_z,
            fascicle_diameter=params.fascicle_diameter,
            fiber_radius=fiber_radius,
            fiber_spacing=fiber_spacing,
            fibers_per_fascicle=params.fibers_per_fascicle,
            crimp_amplitude=crimp_amplitude,
            crimp_wavelength=crimp_wavelength,
            crimp_angle_deg=params.crimp_angle_deg,
            crimp_variance=params.crimp_variance,
            fiber_angle=fascicle_angle,
            resolution=params.resolution,
            position_noise=params.position_noise,
            diameter_variance=params.diameter_variance,
            spacing_variance=adjusted_spacing_variance,
            rng=rng
        )
        all_components.extend(bundle_fibers)
        total_fiber_count += len(bundle_fibers)

        # Create fibril detail if enabled (sub-fiber texture)
        if params.enable_fibril_detail:
            for fiber_y, fiber_z in fiber_positions:
                fibrils = _create_fibril_channels(
                    length=params.length,
                    fiber_y=fiber_y,
                    fiber_z=fiber_z,
                    fiber_radius=fiber_radius,
                    fibril_diameter_um=params.fibril_diameter_um,
                    fibril_spacing_um=params.fibril_spacing_um,
                    crimp_amplitude=crimp_amplitude,
                    crimp_wavelength=crimp_wavelength,
                    resolution=max(4, params.resolution // 2),
                    rng=rng
                )
                fibril_components.extend(fibrils)

        # Create cross-links between fibers if enabled
        if params.enable_cross_links and len(fiber_positions) > 1:
            links = _create_cross_links(
                fiber_positions=fiber_positions,
                fiber_radius=fiber_radius,
                length=params.length,
                cross_link_density=params.cross_link_density,
                resolution=params.resolution,
                rng=rng
            )
            cross_links_list.extend(links)

    # Add fascicle boundaries
    all_components.extend(fascicle_boundaries)

    # Add cross-links
    all_components.extend(cross_links_list)

    # Add fibril components (these are channels to subtract later)
    # Store separately for subtraction
    fibril_channels = fibril_components

    # Add vascular channels if enabled
    if params.enable_vascular_channels:
        vascular = _create_vascular_channels(
            length=params.length,
            width=params.width,
            thickness=params.thickness,
            channel_diameter=params.vascular_channel_diameter,
            channel_spacing=params.vascular_channel_spacing,
            pattern=params.vascular_channel_pattern,
            resolution=params.resolution,
            rng=rng
        )
        all_components.extend(vascular)

    # Add enthesis transition zones if enabled
    if params.enable_enthesis_transition:
        enthesis = _create_enthesis_zones(
            length=params.length,
            width=params.width,
            thickness=params.thickness,
            enthesis_length=params.enthesis_length,
            mineralization_gradient=params.enthesis_mineralization_gradient,
            resolution=params.resolution
        )
        all_components.extend(enthesis)

    # Add porosity (distributed pores) if porosity > 0
    if params.porosity > 0:
        pores = _create_pores(
            length=params.length,
            width=params.width,
            thickness=params.thickness,
            porosity=params.porosity,
            pore_size_um=params.pore_size_um,
            resolution=params.resolution,
            rng=rng
        )
        # Pores are subtracted later; store them separately
        pore_manifolds = pores
    else:
        pore_manifolds = []

    if not all_components:
        raise ValueError("No fibers or components generated")

    # Union all solid components
    result = batch_union(all_components)

    # Add tissue layers (epitenon, paratenon)
    if params.enable_epitenon or params.enable_paratenon:
        tissue_layers = _create_tissue_layers(
            length=params.length,
            width=params.width,
            thickness=params.thickness,
            enable_epitenon=params.enable_epitenon,
            epitenon_thickness=params.epitenon_thickness,
            epitenon_porosity=params.epitenon_porosity,
            enable_paratenon=params.enable_paratenon,
            paratenon_thickness=params.paratenon_thickness,
            resolution=params.resolution,
            rng=rng
        )
        if tissue_layers:
            result = batch_union([result] + tissue_layers)

    # Subtract pores if any
    if pore_manifolds:
        pore_union = batch_union(pore_manifolds)
        result = result - pore_union

    # Subtract fibril channels if enabled (creates surface texture on fibers)
    if fibril_channels:
        fibril_union = batch_union(fibril_channels)
        result = result - fibril_union

    # Add surface texture if enabled
    if params.enable_surface_texture and params.surface_roughness > 0:
        result = _apply_surface_texture(
            manifold=result,
            roughness=params.surface_roughness,
            resolution=params.resolution,
            rng=rng
        )

    # Calculate statistics
    mesh = result.to_mesh()
    volume = result.volume() if hasattr(result, 'volume') else 0

    stats = {
        'triangle_count': len(mesh.tri_verts) // 3 if hasattr(mesh, 'tri_verts') else 0,
        'volume_mm3': volume,
        'fiber_count': total_fiber_count,
        'bundle_count': params.bundle_count,
        'fascicle_boundary_count': len(fascicle_boundaries),
        'cross_link_count': len(cross_links_list),
        'fibril_channel_count': len(fibril_channels) if params.enable_fibril_detail else 0,
        'scaffold_type': 'tendon_ligament'
    }

    return result, stats


def _resolve_um_param(mm_value: float, mm_default: float, um_value: float, um_default: float) -> float:
    """
    Resolve parameter value, preferring _um if explicitly set while mm is default.

    If the mm value differs from default, use it directly.
    If mm is at default but um differs from its default, convert um to mm.
    Otherwise use the mm value.

    Args:
        mm_value: Current value in millimeters
        mm_default: Default value in millimeters
        um_value: Current value in micrometers
        um_default: Default value in micrometers

    Returns:
        Value in millimeters
    """
    # Check if mm was explicitly set (differs from default)
    mm_is_default = abs(mm_value - mm_default) < 1e-9

    # Check if um was explicitly set (differs from default)
    um_is_default = abs(um_value - um_default) < 1e-9

    if not mm_is_default:
        # mm was explicitly set, use it
        return mm_value
    elif not um_is_default:
        # um was explicitly set, convert to mm
        return um_value / 1000.0
    else:
        # Both at defaults, use mm
        return mm_value


def _create_fibril_channels(
    length: float,
    fiber_y: float,
    fiber_z: float,
    fiber_radius: float,
    fibril_diameter_um: float,
    fibril_spacing_um: float,
    crimp_amplitude: float,
    crimp_wavelength: float,
    resolution: int,
    rng: np.random.Generator
) -> list[m3d.Manifold]:
    """
    Create fibril-level channels/grooves along a fiber surface.

    Fibrils are the sub-fiber structural units (~5um diameter vs 150um for fibers).
    This creates small longitudinal channels on the fiber surface to represent
    the fibrillar texture.

    Args:
        length: Total length of fiber
        fiber_y: Y position of fiber center
        fiber_z: Z position of fiber center
        fiber_radius: Radius of the parent fiber
        fibril_diameter_um: Diameter of fibrils in micrometers
        fibril_spacing_um: Spacing between fibrils in micrometers
        crimp_amplitude: Crimp amplitude of parent fiber
        crimp_wavelength: Crimp wavelength of parent fiber
        resolution: Angular resolution
        rng: Random number generator

    Returns:
        List of small channel manifolds to subtract from fiber
    """
    channels = []

    # Convert um to mm
    fibril_diameter = fibril_diameter_um / 1000.0
    fibril_spacing = fibril_spacing_um / 1000.0
    fibril_radius = fibril_diameter / 2

    # Calculate number of fibrils around the fiber circumference
    circumference = 2 * np.pi * fiber_radius
    n_fibrils = max(4, int(circumference / fibril_spacing))

    # Limit for performance
    n_fibrils = min(n_fibrils, 12)

    # Create channels at evenly spaced positions around the fiber
    for i in range(n_fibrils):
        angle = 2 * np.pi * i / n_fibrils

        # Channel position on fiber surface (slightly inside to create groove)
        # The channel runs along the fiber length, offset radially
        channel_offset_y = (fiber_radius - fibril_radius * 0.5) * np.cos(angle)
        channel_offset_z = (fiber_radius - fibril_radius * 0.5) * np.sin(angle)

        channel_y = fiber_y + channel_offset_y
        channel_z = fiber_z + channel_offset_z

        # Create a simplified straight channel (crimp effect is minimal at this scale)
        # Use fewer segments for performance
        n_segments = max(5, int(length / (crimp_wavelength / 2)))

        path_points = []
        phase_offset = rng.uniform(0, 2 * np.pi)

        for j in range(n_segments + 1):
            x = (j / n_segments) * length
            # Apply scaled crimp pattern
            z_offset = crimp_amplitude * np.sin(2 * np.pi * x / crimp_wavelength + phase_offset)

            path_points.append(np.array([
                x,
                channel_y,
                channel_z + z_offset
            ]))

        # Create channel segments
        for j in range(len(path_points) - 1):
            seg = _make_segment(path_points[j], path_points[j + 1], fibril_radius, resolution)
            if seg is not None and seg.num_vert() > 0:
                channels.append(seg)

    return channels


def _calculate_fascicle_positions(
    width: float,
    thickness: float,
    bundle_count: int,
    fascicle_diameter: float,
    fascicle_spacing: float,
    rng: np.random.Generator
) -> list[tuple[float, float]]:
    """Calculate positions for fascicles in cross-section."""
    positions = []

    if bundle_count == 1:
        return [(0.0, 0.0)]

    # Arrange fascicles in a grid pattern within the cross-section
    # Calculate how many rows and columns we need
    total_fascicle_width = fascicle_diameter + fascicle_spacing
    n_cols = max(1, int(width / total_fascicle_width))
    n_rows = max(1, int((bundle_count + n_cols - 1) / n_cols))

    fascicle_idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if fascicle_idx >= bundle_count:
                break

            # Calculate position
            y = (col - (n_cols - 1) / 2) * total_fascicle_width
            z = (row - (n_rows - 1) / 2) * total_fascicle_width

            # Add slight random offset for natural appearance
            y += rng.uniform(-fascicle_spacing * 0.1, fascicle_spacing * 0.1)
            z += rng.uniform(-fascicle_spacing * 0.1, fascicle_spacing * 0.1)

            # Ensure within bounds
            max_y = (width - fascicle_diameter) / 2
            max_z = (thickness - fascicle_diameter) / 2
            y = np.clip(y, -max_y, max_y)
            z = np.clip(z, -max_z, max_z)

            positions.append((y, z))
            fascicle_idx += 1

    return positions


def _create_fascicle_boundary(
    length: float,
    center_y: float,
    center_z: float,
    diameter: float,
    thickness: float,
    porosity: float,
    resolution: int,
    rng: np.random.Generator
) -> m3d.Manifold | None:
    """
    Create a cylindrical shell representing the fascicle boundary (endotenon).

    The endotenon is the connective tissue between fascicles.
    When porosity > 0, adds small pores throughout the endotenon for
    blood vessel and nerve supply pathways.

    Args:
        length: Length of the boundary
        center_y: Y position of center
        center_z: Z position of center
        diameter: Inner diameter (fascicle diameter)
        thickness: Thickness of endotenon layer
        porosity: Porosity of endotenon (0-1)
        resolution: Angular resolution
        rng: Random number generator

    Returns:
        Manifold representing the porous endotenon shell
    """
    outer_radius = diameter / 2 + thickness
    inner_radius = diameter / 2

    # Create outer cylinder
    outer = m3d.Manifold.cylinder(length, outer_radius, outer_radius, resolution)
    # Create inner cylinder to subtract
    inner = m3d.Manifold.cylinder(length, inner_radius, inner_radius, resolution)

    # Create shell by subtraction
    shell = outer - inner

    # Add porosity if specified
    if porosity > 0:
        # Create pores in the endotenon
        pore_radius = thickness * 0.3  # Pores are ~30% of layer thickness
        mid_radius = (outer_radius + inner_radius) / 2

        # Calculate number of pores based on porosity and shell volume
        shell_volume = np.pi * (outer_radius**2 - inner_radius**2) * length
        target_pore_volume = shell_volume * porosity
        single_pore_volume = (4/3) * np.pi * pore_radius**3
        n_pores = max(1, int(target_pore_volume / single_pore_volume / 2))

        # Limit for performance
        n_pores = min(n_pores, 50)

        pores = []
        for _ in range(n_pores):
            # Random position along length and around circumference
            x = rng.uniform(pore_radius, length - pore_radius)
            angle = rng.uniform(0, 2 * np.pi)
            y = mid_radius * np.cos(angle)
            z = mid_radius * np.sin(angle)

            pore = m3d.Manifold.sphere(pore_radius, max(4, resolution // 2))
            pore = pore.translate([x, y, z])
            pores.append(pore)

        if pores:
            pore_union = batch_union(pores)
            shell = shell - pore_union

    # Rotate to align along X axis (length direction)
    shell = shell.rotate([0, 90, 0])

    # Translate to position
    shell = shell.translate([0, center_y, center_z])

    return shell


def _create_fascicle_fibers(
    length: float,
    fascicle_center_y: float,
    fascicle_center_z: float,
    fascicle_diameter: float,
    fiber_radius: float,
    fiber_spacing: float,
    fibers_per_fascicle: int,
    crimp_amplitude: float,
    crimp_wavelength: float,
    crimp_angle_deg: float,
    crimp_variance: float,
    fiber_angle: float,
    resolution: int,
    position_noise: float,
    diameter_variance: float,
    spacing_variance: float,
    rng: np.random.Generator
) -> tuple[list[m3d.Manifold], list[tuple[float, float]]]:
    """
    Create fibers within a single fascicle.

    Returns:
        Tuple of (fiber manifolds, fiber center positions for cross-linking)
    """
    fibers = []
    fiber_positions = []

    # Calculate fiber distribution within fascicle
    fascicle_radius = fascicle_diameter / 2 - fiber_radius

    # Distribute fibers in a packed arrangement within the fascicle
    actual_spacing = fiber_spacing * (1 + rng.uniform(-spacing_variance, spacing_variance))
    n_across = max(1, int(fascicle_diameter / actual_spacing))

    fiber_idx = 0
    for i in range(n_across):
        for j in range(n_across):
            if fiber_idx >= fibers_per_fascicle:
                break

            # Position relative to fascicle center
            rel_y = (i - (n_across - 1) / 2) * actual_spacing
            rel_z = (j - (n_across - 1) / 2) * actual_spacing

            # Check if within fascicle radius
            dist_from_center = np.sqrt(rel_y**2 + rel_z**2)
            if dist_from_center > fascicle_radius:
                continue

            # Apply position noise
            rel_y += rng.uniform(-position_noise, position_noise) * actual_spacing
            rel_z += rng.uniform(-position_noise, position_noise) * actual_spacing

            # Absolute position
            y_pos = fascicle_center_y + rel_y
            z_pos = fascicle_center_z + rel_z

            # Apply diameter variance
            actual_radius = fiber_radius * (1 + rng.uniform(-diameter_variance, diameter_variance))

            # Apply crimp variance
            actual_amplitude = crimp_amplitude * (1 + rng.uniform(-crimp_variance, crimp_variance))
            actual_wavelength = crimp_wavelength * (1 + rng.uniform(-crimp_variance / 2, crimp_variance / 2))

            # Phase offset for natural appearance
            phase_offset = rng.uniform(0, 2 * np.pi)

            # Create crimped fiber with angle
            fiber = _create_crimped_fiber_with_angle(
                length=length,
                y_base=y_pos,
                z_base=z_pos,
                radius=actual_radius,
                crimp_amplitude=actual_amplitude,
                crimp_wavelength=actual_wavelength,
                crimp_angle_deg=crimp_angle_deg,
                fiber_angle=fiber_angle,
                phase_offset=phase_offset,
                resolution=resolution
            )

            if fiber is not None and fiber.num_vert() > 0:
                fibers.append(fiber)
                fiber_positions.append((y_pos, z_pos))
                fiber_idx += 1

    return fibers, fiber_positions


def _create_crimped_fiber_with_angle(
    length: float,
    y_base: float,
    z_base: float,
    radius: float,
    crimp_amplitude: float,
    crimp_wavelength: float,
    crimp_angle_deg: float,
    fiber_angle: float,
    phase_offset: float,
    resolution: int
) -> m3d.Manifold | None:
    """
    Create a single crimped fiber with specified crimp angle and fiber angle.

    The crimp_angle_deg affects the sharpness of the crimp pattern.
    The fiber_angle rotates the entire fiber in the YZ plane.
    """
    # Number of segments along length
    n_segments = max(10, int(length / (crimp_wavelength / 4)))

    # Crimp angle affects amplitude scaling
    # Higher crimp angle = more pronounced crimp
    angle_factor = np.tan(np.radians(crimp_angle_deg)) / np.tan(np.radians(20))  # Normalize to 20 deg baseline
    effective_amplitude = crimp_amplitude * angle_factor

    # Generate path points
    path_points = []
    for i in range(n_segments + 1):
        x = (i / n_segments) * length
        # Sinusoidal crimp in Z direction with phase offset
        z_offset = effective_amplitude * np.sin(2 * np.pi * x / crimp_wavelength + phase_offset)
        z = z_base + z_offset

        # Apply fiber angle rotation in YZ plane
        y_rotated = y_base * np.cos(fiber_angle) - z_offset * np.sin(fiber_angle)
        z_rotated = z_base + y_base * np.sin(fiber_angle) + z_offset * np.cos(fiber_angle)

        path_points.append(np.array([x, y_base + (y_rotated - y_base) * 0.1, z]))

    # Create cylinders between consecutive points
    segments = []
    for i in range(len(path_points) - 1):
        p1 = path_points[i]
        p2 = path_points[i + 1]

        segment = _make_segment(p1, p2, radius, resolution)
        if segment is not None and segment.num_vert() > 0:
            segments.append(segment)

    # Union all segments
    if not segments:
        return None

    if len(segments) == 1:
        return segments[0]

    result = segments[0]
    for seg in segments[1:]:
        result = result + seg

    return result


def _make_segment(p1: np.ndarray, p2: np.ndarray, radius: float, resolution: int) -> m3d.Manifold | None:
    """
    Create a cylindrical segment between two points.
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    length = np.sqrt(dx*dx + dy*dy + dz*dz)

    if length < 1e-6:
        return None

    # Create cylinder along Z axis
    segment = m3d.Manifold.cylinder(length, radius, radius, resolution)

    # Calculate rotation to align with direction vector
    h = np.sqrt(dx*dx + dy*dy)
    if h > 0.001 or abs(dz) > 0.001:
        angle_y = np.arctan2(h, dz) * 180 / np.pi
        angle_z = np.arctan2(dy, dx) * 180 / np.pi

        segment = segment.rotate([0, angle_y, 0])
        segment = segment.rotate([0, 0, angle_z])

    return segment.translate([p1[0], p1[1], p1[2]])


def _create_cross_links(
    fiber_positions: list[tuple[float, float]],
    fiber_radius: float,
    length: float,
    cross_link_density: float,
    resolution: int,
    rng: np.random.Generator
) -> list[m3d.Manifold]:
    """
    Create cross-links between adjacent fibers.

    Cross-links are small cylinders connecting neighboring fibers.
    """
    links = []
    link_radius = fiber_radius * 0.3  # Cross-links are thinner than fibers

    # For each pair of adjacent fibers
    for i in range(len(fiber_positions)):
        for j in range(i + 1, len(fiber_positions)):
            y1, z1 = fiber_positions[i]
            y2, z2 = fiber_positions[j]

            # Distance between fiber centers
            dist = np.sqrt((y2 - y1)**2 + (z2 - z1)**2)

            # Only create links between nearby fibers
            if dist > fiber_radius * 6:
                continue

            # Number of links along length based on density
            n_links = max(1, int(length * cross_link_density / 10))

            for k in range(n_links):
                # Random x position along length
                x_pos = rng.uniform(0.1 * length, 0.9 * length)

                # Create link from fiber i to fiber j
                p1 = np.array([x_pos, y1, z1])
                p2 = np.array([x_pos, y2, z2])

                link = _make_segment(p1, p2, link_radius, max(4, resolution // 2))
                if link is not None and link.num_vert() > 0:
                    links.append(link)

    return links


def _create_vascular_channels(
    length: float,
    width: float,
    thickness: float,
    channel_diameter: float,
    channel_spacing: float,
    pattern: str,
    resolution: int,
    rng: np.random.Generator
) -> list[m3d.Manifold]:
    """
    Create vascular channels running through the tendon.

    Channels run in the endotenon regions between fascicles.
    """
    channels = []
    channel_radius = channel_diameter / 2

    # Calculate number of channels based on spacing
    n_channels_y = max(1, int(width / channel_spacing))
    n_channels_z = max(1, int(thickness / channel_spacing))

    for i in range(n_channels_y):
        for j in range(n_channels_z):
            # Position in cross-section
            y_pos = (i - (n_channels_y - 1) / 2) * channel_spacing
            z_pos = (j - (n_channels_z - 1) / 2) * channel_spacing

            # Add slight randomness
            y_pos += rng.uniform(-channel_spacing * 0.1, channel_spacing * 0.1)
            z_pos += rng.uniform(-channel_spacing * 0.1, channel_spacing * 0.1)

            if pattern == 'longitudinal':
                # Straight channel along length
                channel = m3d.Manifold.cylinder(length, channel_radius, channel_radius, resolution)
                channel = channel.rotate([0, 90, 0])
                channel = channel.translate([0, y_pos, z_pos])
            elif pattern == 'spiral':
                # Spiral channel
                channel = _create_spiral_channel(
                    length=length,
                    y_center=y_pos,
                    z_center=z_pos,
                    radius=channel_radius,
                    spiral_radius=channel_spacing * 0.3,
                    resolution=resolution
                )
            else:
                # Default to longitudinal
                channel = m3d.Manifold.cylinder(length, channel_radius, channel_radius, resolution)
                channel = channel.rotate([0, 90, 0])
                channel = channel.translate([0, y_pos, z_pos])

            if channel is not None and channel.num_vert() > 0:
                channels.append(channel)

    return channels


def _create_spiral_channel(
    length: float,
    y_center: float,
    z_center: float,
    radius: float,
    spiral_radius: float,
    resolution: int
) -> m3d.Manifold | None:
    """Create a spiral vascular channel."""
    n_segments = max(20, int(length * 4))
    n_turns = length / 5  # One turn every 5mm

    path_points = []
    for i in range(n_segments + 1):
        t = i / n_segments
        x = t * length
        angle = 2 * np.pi * n_turns * t
        y = y_center + spiral_radius * np.cos(angle)
        z = z_center + spiral_radius * np.sin(angle)
        path_points.append(np.array([x, y, z]))

    # Create tube along path
    segments = []
    for i in range(len(path_points) - 1):
        seg = _make_segment(path_points[i], path_points[i + 1], radius, resolution)
        if seg is not None and seg.num_vert() > 0:
            segments.append(seg)

    if not segments:
        return None

    return batch_union(segments)


def _create_enthesis_zones(
    length: float,
    width: float,
    thickness: float,
    enthesis_length: float,
    mineralization_gradient: float,
    resolution: int
) -> list[m3d.Manifold]:
    """
    Create enthesis (bone insertion) transition zones at tendon ends.

    The enthesis has 4 zones:
    1. Dense tendon
    2. Unmineralized fibrocartilage
    3. Mineralized fibrocartilage
    4. Bone

    This is modeled as gradient density zones with increasing solid fraction.
    """
    zones = []

    # Create zones at both ends
    for end in [0, 1]:  # 0 = start, 1 = end
        zone_x = 0 if end == 0 else length - enthesis_length

        # Create 4 sub-zones within the enthesis
        sub_zone_length = enthesis_length / 4

        for zone_idx in range(4):
            # Each zone has increasing density (modeled by decreasing porosity)
            zone_density = 0.6 + 0.1 * zone_idx  # 60%, 70%, 80%, 90%

            # Zone dimensions - slightly smaller for inner zones
            zone_width = width * (0.9 + 0.025 * zone_idx)
            zone_thickness = thickness * (0.9 + 0.025 * zone_idx)

            # Create zone as a box
            x_start = zone_x + zone_idx * sub_zone_length
            zone_box = m3d.Manifold.cube([sub_zone_length, zone_width, zone_thickness], True)

            # Position the zone
            zone_box = zone_box.translate([
                x_start + sub_zone_length / 2,
                0,
                0
            ])

            zones.append(zone_box)

    return zones


def _create_tissue_layers(
    length: float,
    width: float,
    thickness: float,
    enable_epitenon: bool,
    epitenon_thickness: float,
    epitenon_porosity: float,
    enable_paratenon: bool,
    paratenon_thickness: float,
    resolution: int,
    rng: np.random.Generator
) -> list[m3d.Manifold]:
    """
    Create outer tissue layers (epitenon and paratenon).

    Epitenon: Inner sheath directly around the tendon (with optional porosity)
    Paratenon: Outer sliding layer (when present)

    Args:
        length: Total length of the tendon
        width: Width of the tendon
        thickness: Thickness of the tendon
        enable_epitenon: Whether to create epitenon layer
        epitenon_thickness: Thickness of epitenon layer
        epitenon_porosity: Porosity of epitenon (0-1)
        enable_paratenon: Whether to create paratenon layer
        paratenon_thickness: Thickness of paratenon layer
        resolution: Angular resolution
        rng: Random number generator

    Returns:
        List of tissue layer manifolds
    """
    layers = []

    # Calculate radii for elliptical cross-section
    inner_ry = width / 2
    inner_rz = thickness / 2

    if enable_epitenon:
        # Epitenon is a thin shell around the tendon
        outer_ry = inner_ry + epitenon_thickness
        outer_rz = inner_rz + epitenon_thickness

        # Create as a hollow elliptical cylinder
        # Approximate with scaled cylinders
        avg_outer_r = (outer_ry + outer_rz) / 2
        avg_inner_r = (inner_ry + inner_rz) / 2

        outer = m3d.Manifold.cylinder(length, avg_outer_r, avg_outer_r, resolution)
        inner = m3d.Manifold.cylinder(length, avg_inner_r, avg_inner_r, resolution)

        epitenon = outer - inner

        # Add porosity to epitenon if specified
        if epitenon_porosity > 0:
            pore_radius = epitenon_thickness * 0.25  # Pores are ~25% of layer thickness
            mid_radius = (avg_outer_r + avg_inner_r) / 2

            # Calculate number of pores based on porosity
            shell_volume = np.pi * (avg_outer_r**2 - avg_inner_r**2) * length
            target_pore_volume = shell_volume * epitenon_porosity
            single_pore_volume = (4/3) * np.pi * pore_radius**3
            n_pores = max(1, int(target_pore_volume / single_pore_volume / 2))

            # Limit for performance
            n_pores = min(n_pores, 80)

            pores = []
            for _ in range(n_pores):
                # Random position along length and around circumference
                x = rng.uniform(pore_radius, length - pore_radius)
                angle = rng.uniform(0, 2 * np.pi)
                y = mid_radius * np.cos(angle)
                z = mid_radius * np.sin(angle)

                pore = m3d.Manifold.sphere(pore_radius, max(4, resolution // 2))
                pore = pore.translate([x, y, z])
                pores.append(pore)

            if pores:
                pore_union = batch_union(pores)
                epitenon = epitenon - pore_union

        epitenon = epitenon.rotate([0, 90, 0])
        epitenon = epitenon.scale([1, outer_ry / avg_outer_r, outer_rz / avg_outer_r])

        layers.append(epitenon)

        # Update inner radii for paratenon
        inner_ry = outer_ry
        inner_rz = outer_rz

    if enable_paratenon:
        # Paratenon is the outermost layer
        outer_ry = inner_ry + paratenon_thickness
        outer_rz = inner_rz + paratenon_thickness

        avg_outer_r = (outer_ry + outer_rz) / 2
        avg_inner_r = (inner_ry + inner_rz) / 2

        outer = m3d.Manifold.cylinder(length, avg_outer_r, avg_outer_r, resolution)
        inner = m3d.Manifold.cylinder(length, avg_inner_r, avg_inner_r, resolution)

        paratenon = outer - inner
        paratenon = paratenon.rotate([0, 90, 0])
        paratenon = paratenon.scale([1, outer_ry / avg_outer_r, outer_rz / avg_outer_r])

        layers.append(paratenon)

    return layers


def _create_pores(
    length: float,
    width: float,
    thickness: float,
    porosity: float,
    pore_size_um: float,
    resolution: int,
    rng: np.random.Generator
) -> list[m3d.Manifold]:
    """
    Create distributed pores throughout the scaffold.

    Pores are spherical voids to be subtracted from the solid.
    """
    pores = []
    pore_radius = pore_size_um / 1000 / 2  # Convert um to mm

    # Calculate volume and number of pores needed
    total_volume = length * width * thickness
    target_pore_volume = total_volume * porosity
    single_pore_volume = (4/3) * np.pi * pore_radius**3
    n_pores = max(1, int(target_pore_volume / single_pore_volume / 3))  # Divide by 3 for overlapping effect

    # Limit pores for performance
    n_pores = min(n_pores, 500)

    for _ in range(n_pores):
        # Random position within bounds
        x = rng.uniform(pore_radius, length - pore_radius)
        y = rng.uniform(-width/2 + pore_radius, width/2 - pore_radius)
        z = rng.uniform(-thickness/2 + pore_radius, thickness/2 - pore_radius)

        # Create pore (sphere)
        pore = m3d.Manifold.sphere(pore_radius, resolution)
        pore = pore.translate([x, y, z])
        pores.append(pore)

    return pores


def _apply_surface_texture(
    manifold: m3d.Manifold,
    roughness: float,
    resolution: int,
    rng: np.random.Generator
) -> m3d.Manifold:
    """
    Apply surface texture to the manifold.

    Creates small bumps on the surface for cell attachment.
    Roughness parameter (0-0.1, default 0.02) controls the size
    and density of surface features.

    Biological context: Surface roughness (1-100 μm features) is
    important for cell attachment and migration in tissue scaffolds.
    """
    if roughness <= 0:
        return manifold

    # Get bounding box for surface placement
    bbox = manifold.bounding_box()
    min_pt = bbox.min
    max_pt = bbox.max

    # Feature size based on roughness (roughness=0.02 → ~20μm features)
    # Scale to mm: roughness * 0.5 gives feature radius in mm
    feature_radius = roughness * 0.5

    # Spacing between features (4x the radius for balanced coverage)
    spacing = feature_radius * 4.0

    # Create bumps on top and bottom surfaces (z-direction)
    bumps = []

    # Limit the number of features for performance
    max_features = 200
    feature_count = 0

    # Sample points on top and bottom surfaces
    x_min, y_min, z_min = min_pt
    x_max, y_max, z_max = max_pt

    # Generate features on top surface (z_max)
    x = x_min
    while x <= x_max and feature_count < max_features:
        y = y_min
        while y <= y_max and feature_count < max_features:
            # Add some randomness to position
            jitter_x = rng.uniform(-spacing * 0.3, spacing * 0.3)
            jitter_y = rng.uniform(-spacing * 0.3, spacing * 0.3)

            bump = m3d.Manifold.sphere(feature_radius, resolution)
            bump = bump.translate([x + jitter_x, y + jitter_y, z_max])
            bumps.append(bump)
            feature_count += 1

            y += spacing
        x += spacing

    # Generate features on bottom surface (z_min)
    x = x_min
    while x <= x_max and feature_count < max_features:
        y = y_min
        while y <= y_max and feature_count < max_features:
            # Add some randomness to position
            jitter_x = rng.uniform(-spacing * 0.3, spacing * 0.3)
            jitter_y = rng.uniform(-spacing * 0.3, spacing * 0.3)

            bump = m3d.Manifold.sphere(feature_radius, resolution)
            bump = bump.translate([x + jitter_x, y + jitter_y, z_min])
            bumps.append(bump)
            feature_count += 1

            y += spacing
        x += spacing

    # Union bumps to manifold
    if bumps:
        bump_union = batch_union(bumps)
        manifold = manifold + bump_union

    return manifold


# Legacy functions for backward compatibility

def create_crimped_fiber_bundle(
    length: float,
    width: float,
    thickness: float,
    fiber_radius: float,
    fiber_spacing: float,
    crimp_amplitude: float,
    crimp_wavelength: float,
    fibers_per_bundle: int,
    bundle_offset_y: float,
    resolution: int
) -> list[m3d.Manifold]:
    """
    Create a bundle of crimped parallel fibers (legacy function).

    Args:
        length: Length along fiber direction (X axis)
        width: Width of bundle (Y axis)
        thickness: Thickness of bundle (Z axis)
        fiber_radius: Radius of individual fibers
        fiber_spacing: Spacing between fiber centers
        crimp_amplitude: Amplitude of sinusoidal crimp
        crimp_wavelength: Wavelength of crimp pattern
        fibers_per_bundle: Number of fibers in this bundle
        bundle_offset_y: Y-axis offset for this bundle
        resolution: Angular and path resolution

    Returns:
        List of crimped fiber manifolds
    """
    fibers = []

    # Distribute fibers in cross-section
    n_width = max(1, int(width / fiber_spacing))
    n_thickness = max(1, int(thickness / fiber_spacing))

    fiber_idx = 0
    for i in range(n_width):
        for j in range(n_thickness):
            if fiber_idx >= fibers_per_bundle:
                break

            # Fiber position in cross-section
            y_pos = bundle_offset_y + (i - (n_width - 1) / 2) * fiber_spacing
            z_pos = (j - (n_thickness - 1) / 2) * fiber_spacing

            # Add slight random offset for natural appearance
            y_pos += (np.random.random() - 0.5) * fiber_spacing * 0.2
            z_pos += (np.random.random() - 0.5) * fiber_spacing * 0.2

            # Create crimped fiber path
            fiber = create_crimped_fiber(
                length=length,
                y_base=y_pos,
                z_base=z_pos,
                radius=fiber_radius,
                crimp_amplitude=crimp_amplitude,
                crimp_wavelength=crimp_wavelength,
                resolution=resolution
            )

            if fiber.num_vert() > 0:
                fibers.append(fiber)
                fiber_idx += 1

    return fibers


def create_crimped_fiber(
    length: float,
    y_base: float,
    z_base: float,
    radius: float,
    crimp_amplitude: float,
    crimp_wavelength: float,
    resolution: int
) -> m3d.Manifold:
    """
    Create a single crimped fiber following sinusoidal path (legacy function).

    Args:
        length: Total length of fiber
        y_base: Base Y position
        z_base: Base Z position (crimp oscillates in Z direction)
        radius: Fiber radius
        crimp_amplitude: Amplitude of sine wave
        crimp_wavelength: Wavelength of crimp
        resolution: Number of segments for smooth curve

    Returns:
        Manifold representing crimped fiber
    """
    # Number of segments along length
    n_segments = max(10, int(length / (crimp_wavelength / 4)))

    # Generate path points
    path_points = []
    for i in range(n_segments + 1):
        x = (i / n_segments) * length
        # Sinusoidal crimp in Z direction
        z = z_base + crimp_amplitude * np.sin(2 * np.pi * x / crimp_wavelength)
        path_points.append(np.array([x, y_base, z]))

    # Create cylinders between consecutive points
    segments = []
    for i in range(len(path_points) - 1):
        p1 = path_points[i]
        p2 = path_points[i + 1]

        segment = make_segment(p1, p2, radius, resolution)
        if segment.num_vert() > 0:
            segments.append(segment)

    # Union all segments
    if not segments:
        return m3d.Manifold()

    if len(segments) == 1:
        return segments[0]

    result = segments[0]
    for seg in segments[1:]:
        result = result + seg

    return result


def make_segment(p1: np.ndarray, p2: np.ndarray, radius: float, resolution: int) -> m3d.Manifold:
    """
    Create a cylindrical segment between two points (legacy function).

    Args:
        p1: Starting point (x, y, z)
        p2: Ending point (x, y, z)
        radius: Segment radius
        resolution: Number of segments around cylinder

    Returns:
        Manifold representing the segment cylinder
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    length = np.sqrt(dx*dx + dy*dy + dz*dz)

    if length < 1e-6:
        return m3d.Manifold()

    # Create cylinder along Z axis
    segment = m3d.Manifold.cylinder(length, radius, radius, resolution)

    # Calculate rotation to align with direction vector
    h = np.sqrt(dx*dx + dy*dy)
    if h > 0.001 or abs(dz) > 0.001:
        angle_y = np.arctan2(h, dz) * 180 / np.pi
        angle_z = np.arctan2(dy, dx) * 180 / np.pi

        segment = segment.rotate([0, angle_y, 0])
        segment = segment.rotate([0, 0, angle_z])

    return segment.translate([p1[0], p1[1], p1[2]])


def generate_tendon_ligament_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate tendon/ligament from dictionary parameters (for API compatibility).

    Args:
        params: Dictionary with keys matching TendonLigamentParams fields

    Returns:
        Tuple of (manifold, stats_dict)
    """
    return generate_tendon_ligament(TendonLigamentParams(
        # Basic geometry
        length=params.get('length', 30.0),
        width=params.get('width', 8.0),
        thickness=params.get('thickness', 3.0),
        resolution=params.get('resolution', 12),
        seed=params.get('seed', 42),

        # Fascicle organization
        bundle_count=params.get('bundle_count', 5),
        fascicle_diameter=params.get('fascicle_diameter', 1.5),
        fascicle_spacing=params.get('fascicle_spacing', 0.3),
        enable_fascicle_boundaries=params.get('enable_fascicle_boundaries', True),

        # Fiber properties
        fiber_diameter=params.get('fiber_diameter', 0.15),
        fiber_diameter_um=params.get('fiber_diameter_um', 150.0),
        fiber_spacing=params.get('fiber_spacing', 0.4),
        fiber_spacing_um=params.get('fiber_spacing_um', 400.0),
        fibers_per_fascicle=params.get('fibers_per_fascicle', 20),

        # Fibril properties
        fibril_diameter_um=params.get('fibril_diameter_um', 5.0),
        fibril_spacing_um=params.get('fibril_spacing_um', 10.0),
        enable_fibril_detail=params.get('enable_fibril_detail', False),

        # Crimp pattern
        crimp_amplitude=params.get('crimp_amplitude', 0.3),
        crimp_amplitude_um=params.get('crimp_amplitude_um', 300.0),
        crimp_wavelength=params.get('crimp_wavelength', 2.0),
        crimp_wavelength_um=params.get('crimp_wavelength_um', 2000.0),
        crimp_angle_deg=params.get('crimp_angle_deg', 10.0),
        crimp_variance=params.get('crimp_variance', 0.1),

        # Endotenon
        enable_endotenon=params.get('enable_endotenon', True),
        endotenon_thickness=params.get('endotenon_thickness', 0.05),
        endotenon_porosity=params.get('endotenon_porosity', 0.5),

        # Epitenon
        enable_epitenon=params.get('enable_epitenon', True),
        epitenon_thickness=params.get('epitenon_thickness', 0.1),
        epitenon_porosity=params.get('epitenon_porosity', 0.3),

        # Paratenon
        enable_paratenon=params.get('enable_paratenon', False),
        paratenon_thickness=params.get('paratenon_thickness', 0.2),

        # Collagen alignment
        primary_fiber_angle_deg=params.get('primary_fiber_angle_deg', 0.0),
        fiber_angle_variance_deg=params.get('fiber_angle_variance_deg', 5.0),
        enable_cross_links=params.get('enable_cross_links', True),
        cross_link_density=params.get('cross_link_density', 0.1),

        # Vascular features
        enable_vascular_channels=params.get('enable_vascular_channels', True),
        vascular_channel_diameter=params.get('vascular_channel_diameter', 0.08),
        vascular_channel_spacing=params.get('vascular_channel_spacing', 1.5),
        vascular_channel_pattern=params.get('vascular_channel_pattern', 'longitudinal'),

        # Enthesis
        enable_enthesis_transition=params.get('enable_enthesis_transition', False),
        enthesis_length=params.get('enthesis_length', 3.0),
        enthesis_mineralization_gradient=params.get('enthesis_mineralization_gradient', 0.5),

        # Mechanical indicators
        target_stiffness_mpa=params.get('target_stiffness_mpa', 1000.0),
        target_ultimate_stress_mpa=params.get('target_ultimate_stress_mpa', 100.0),

        # Porosity
        porosity=params.get('porosity', 0.15),
        pore_size_um=params.get('pore_size_um', 50.0),

        # Surface
        surface_roughness=params.get('surface_roughness', 0.02),
        enable_surface_texture=params.get('enable_surface_texture', True),

        # Randomization
        position_noise=params.get('position_noise', 0.05),
        diameter_variance=params.get('diameter_variance', 0.1),
        spacing_variance=params.get('spacing_variance', 0.1),
    ))
