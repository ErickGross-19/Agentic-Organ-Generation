"""
Intervertebral disc scaffold generator with annulus fibrosus and nucleus pulposus.

Creates a disc-shaped structure with:
- Annulus fibrosus: 20 concentric lamellae with alternating fiber angles
- Nucleus pulposus: soft gel-like center with high porosity
- Cartilaginous endplates with nutrient diffusion channels
- Transition zone at NP-AF interface
- Optional vascular supply in outer AF
- Optional notochordal remnants and degeneration features
"""

from __future__ import annotations
import manifold3d as m3d
import numpy as np
from dataclasses import dataclass
from ..core import batch_union


@dataclass
class IntervertebralDiscParams:
    """Parameters for intervertebral disc scaffold generation.

    Biologically realistic parameters based on human intervertebral disc:
    - Disc diameter: 30-50 mm (cervical to lumbar)
    - Disc height: 5-15 mm (varies by spinal level)
    - Nucleus pulposus: 30-50% of disc area, highly hydrated (70-90% water)
    - Annulus fibrosus: 15-25 concentric lamellae, alternating fiber angles ±25-60°
    - Fiber angle: increases from inner (~45°) to outer (~65°) AF
    - Cartilaginous endplates: 0.5-1.0 mm thick
    """
    # Basic geometry (mm)
    disc_diameter: float = 40.0  # Total disc diameter
    disc_height: float = 8.5  # Total disc height
    resolution: int = 16
    seed: int = 42

    # Nucleus pulposus (NP) - central gel-like region
    np_diameter: float = 16.0  # NP diameter (40% of disc diameter typical)
    nucleus_percentage: float = 0.40  # NP as fraction of disc area
    np_height_ratio: float = 0.85  # NP height relative to disc height
    np_porosity: float = 0.90  # Very high porosity (gel-like)
    np_water_content: float = 0.80  # Water content indicator
    np_proteoglycan_content: float = 0.8  # Relative proteoglycan density
    np_pore_size: float = 0.3  # Pore size in NP region (mm)

    # Annulus fibrosus (AF) - outer fibrous ring
    af_ring_count: int = 3  # Simplified ring count (legacy)
    num_lamellae: int = 20  # Actual number of concentric lamellae (15-25 typical)
    annulus_percentage: float = 0.60  # AF as fraction of disc area
    af_porosity: float = 0.65  # Lower porosity than NP
    af_water_content: float = 0.60  # Water content indicator

    # Fiber properties
    fiber_diameter: float = 0.2  # Collagen fiber bundle diameter (mm)
    fiber_diameter_um: float = 200.0  # Fiber diameter in micrometers
    fiber_angle: float = 30.0  # Mean fiber angle (legacy)
    af_layer_angle: float = 30.0  # Base alternating angle (legacy alias)
    inner_af_fiber_angle_deg: float = 45.0  # Inner AF fiber angle
    outer_af_fiber_angle_deg: float = 65.0  # Outer AF fiber angle
    fiber_angle_variance_deg: float = 5.0  # Random variance in fiber angle

    # Lamella properties
    lamella_thickness: float = 0.15  # Individual lamella thickness (mm)
    interlaminar_spacing: float = 0.02  # Gap between lamellae (mm)
    enable_interlaminar_connections: bool = True  # Cross-links between lamellae

    # Mechanical property indicators (relative stiffness)
    np_stiffness_kpa: float = 2.0  # NP aggregate modulus (~1-5 kPa)
    af_stiffness_kpa: float = 100.0  # AF tensile modulus (~50-200 kPa)
    stiffness_gradient: bool = True  # Gradual stiffness change from NP to AF

    # Cartilaginous endplates
    enable_endplates: bool = True  # Include cartilaginous endplates
    endplate_thickness: float = 0.7  # Endplate thickness (mm)
    endplate_porosity: float = 0.55  # Endplate porosity
    endplate_pore_size: float = 0.1  # Pore size in endplates

    # Nutrient pathway features
    enable_nutrient_channels: bool = True  # Diffusion channels
    nutrient_channel_diameter: float = 0.05  # Small channel diameter
    nutrient_channel_density: float = 0.3  # Channels per mm²
    enable_endplate_pores: bool = True  # Pores through endplate

    # Transition zone (NP-AF interface)
    transition_zone_width: float = 0.5  # Width of NP-AF transition (mm)
    transition_gradient_type: str = 'linear'  # linear, sigmoid, or step

    # Vascular features (outer AF only)
    enable_outer_vascular: bool = True  # Blood vessels in outer AF
    vascular_channel_diameter: float = 0.1  # Vessel diameter
    vascular_penetration_depth: float = 0.2  # Fraction of AF thickness

    # Notochordal remnants (for developmental accuracy)
    enable_notochordal_cells: bool = False  # Notochordal cell region
    notochordal_region_radius: float = 2.0  # Central notochordal area

    # Age-related changes
    degeneration_level: float = 0.0  # 0 = healthy, 1 = fully degenerated
    enable_fissures: bool = False  # Radial or circumferential tears
    fissure_count: int = 0  # Number of fissures

    # Surface properties
    surface_roughness: float = 0.05  # Surface texture

    # Randomization
    position_noise: float = 0.1  # Random position jitter
    fiber_variance: float = 0.1  # Variation in fiber properties
    lamella_variance: float = 0.05  # Variation in lamella thickness


def generate_intervertebral_disc(params: IntervertebralDiscParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate an intervertebral disc scaffold.

    Creates multiple distinct regions:
    1. Nucleus pulposus (center) - highly porous gel-like structure with optional notochordal remnants
    2. Transition zone - gradual change from NP to AF properties
    3. Annulus fibrosus (outer) - 20 concentric lamellae with angle gradient
    4. Cartilaginous endplates - porous plates at top/bottom
    5. Nutrient channels - vertical microchannels through endplates
    6. Vascular supply - channels in outer 1/3 of AF

    Args:
        params: IntervertebralDiscParams specifying geometry and structure

    Returns:
        Tuple of (manifold, stats_dict)
        - manifold: The 3D geometry
        - stats_dict: Dictionary with triangle_count, volume_mm3, ring_count,
                     af_volume, np_volume, scaffold_type
    """
    rng = np.random.default_rng(params.seed)

    disc_radius = params.disc_diameter / 2
    # Use nucleus_percentage to compute NP radius from disc area
    np_radius = disc_radius * np.sqrt(params.nucleus_percentage)
    np_height = params.disc_height * params.np_height_ratio

    # Use fiber_diameter_um as fallback if fiber_diameter is at default
    # Convert um to mm by dividing by 1000
    if params.fiber_diameter == 0.2 and params.fiber_diameter_um != 200.0:
        fiber_diameter = params.fiber_diameter_um / 1000.0
    else:
        fiber_diameter = params.fiber_diameter
    fiber_radius = fiber_diameter / 2

    all_parts = []

    # Calculate z offset to center the disc
    disc_z_offset = 0.0
    if params.enable_endplates:
        disc_z_offset = params.endplate_thickness

    # Create nucleus pulposus (porous center)
    np_manifold = create_nucleus_pulposus(
        radius=np_radius,
        height=np_height,
        porosity=params.np_porosity,
        pore_size=params.np_pore_size,
        resolution=params.resolution,
        rng=rng,
        water_content=params.np_water_content,
        proteoglycan_content=params.np_proteoglycan_content,
        position_noise=params.position_noise
    )
    # Center NP vertically within disc
    np_z_offset = disc_z_offset + (params.disc_height - np_height) / 2
    np_manifold = np_manifold.translate([0, 0, np_z_offset])
    all_parts.append(np_manifold)

    # Add notochordal remnants if enabled (larger voids in central NP)
    if params.enable_notochordal_cells:
        notochordal = create_notochordal_region(
            radius=params.notochordal_region_radius,
            height=np_height * 0.6,
            resolution=params.resolution,
            rng=rng
        )
        notochordal = notochordal.translate([0, 0, np_z_offset + np_height * 0.2])
        all_parts.append(notochordal)

    # Create transition zone at NP-AF interface
    if params.transition_zone_width > 0:
        transition = create_transition_zone(
            np_radius=np_radius,
            zone_width=params.transition_zone_width,
            height=params.disc_height,
            gradient_type=params.transition_gradient_type,
            np_porosity=params.np_porosity,
            af_porosity=params.af_porosity,
            resolution=params.resolution,
            rng=rng
        )
        transition = transition.translate([0, 0, disc_z_offset])
        all_parts.append(transition)
        af_inner_radius = np_radius + params.transition_zone_width
    else:
        af_inner_radius = np_radius

    # Use legacy fiber_angle/af_layer_angle as fallback when inner/outer angles are at defaults
    # Legacy parameters specify a single angle, used as a base for both
    inner_angle = params.inner_af_fiber_angle_deg
    outer_angle = params.outer_af_fiber_angle_deg

    # If inner/outer are at defaults but legacy angle differs, use legacy as base
    if inner_angle == 45.0 and outer_angle == 65.0:
        if params.fiber_angle != 30.0:
            # Use fiber_angle as base - inner is fiber_angle, outer is fiber_angle + 20
            inner_angle = params.fiber_angle
            outer_angle = params.fiber_angle + 20.0
        elif params.af_layer_angle != 30.0:
            # Use af_layer_angle as base
            inner_angle = params.af_layer_angle
            outer_angle = params.af_layer_angle + 20.0

    # Create annulus fibrosus with proper lamellae
    af_manifolds = create_annulus_fibrosus_lamellae(
        inner_radius=af_inner_radius,
        outer_radius=disc_radius,
        height=params.disc_height,
        num_lamellae=params.num_lamellae,
        inner_angle=inner_angle,
        outer_angle=outer_angle,
        angle_variance=params.fiber_angle_variance_deg,
        lamella_thickness=params.lamella_thickness,
        interlaminar_spacing=params.interlaminar_spacing,
        enable_connections=params.enable_interlaminar_connections,
        fiber_radius=fiber_radius,
        resolution=params.resolution,
        rng=rng,
        water_content=params.af_water_content,
        lamella_variance=params.lamella_variance,
        fiber_variance=params.fiber_variance
    )
    for af in af_manifolds:
        af_translated = af.translate([0, 0, disc_z_offset])
        all_parts.append(af_translated)

    # Add vascular channels in outer AF (only outer 1/3 is vascularized)
    if params.enable_outer_vascular:
        vascular = create_vascular_supply(
            disc_radius=disc_radius,
            af_inner_radius=af_inner_radius,
            penetration_depth=params.vascular_penetration_depth,
            channel_diameter=params.vascular_channel_diameter,
            height=params.disc_height,
            resolution=params.resolution,
            rng=rng
        )
        for v in vascular:
            v_translated = v.translate([0, 0, disc_z_offset])
            all_parts.append(v_translated)

    # Create cartilaginous endplates
    if params.enable_endplates:
        # Bottom endplate
        bottom_endplate = create_endplate(
            radius=disc_radius,
            thickness=params.endplate_thickness,
            porosity=params.endplate_porosity,
            pore_size=params.endplate_pore_size,
            enable_nutrient_channels=params.enable_nutrient_channels,
            nutrient_channel_diameter=params.nutrient_channel_diameter,
            nutrient_channel_density=params.nutrient_channel_density,
            np_radius=np_radius,
            resolution=params.resolution,
            rng=rng,
            enable_endplate_pores=params.enable_endplate_pores
        )
        all_parts.append(bottom_endplate)

        # Top endplate
        top_endplate = create_endplate(
            radius=disc_radius,
            thickness=params.endplate_thickness,
            porosity=params.endplate_porosity,
            pore_size=params.endplate_pore_size,
            enable_nutrient_channels=params.enable_nutrient_channels,
            nutrient_channel_diameter=params.nutrient_channel_diameter,
            nutrient_channel_density=params.nutrient_channel_density,
            np_radius=np_radius,
            resolution=params.resolution,
            rng=rng,
            enable_endplate_pores=params.enable_endplate_pores
        )
        top_endplate = top_endplate.translate([0, 0, disc_z_offset + params.disc_height])
        all_parts.append(top_endplate)

    # Add degeneration features if enabled
    if params.enable_fissures and params.fissure_count > 0 and params.degeneration_level > 0:
        fissures = create_fissures(
            np_radius=np_radius,
            disc_radius=disc_radius,
            height=params.disc_height,
            fissure_count=params.fissure_count,
            degeneration_level=params.degeneration_level,
            resolution=params.resolution,
            rng=rng
        )
        for f in fissures:
            f_translated = f.translate([0, 0, disc_z_offset])
            all_parts.append(f_translated)

    # Add surface roughness to outer rim if enabled
    if params.surface_roughness > 0:
        roughness_features = create_surface_roughness(
            disc_radius=disc_radius,
            disc_height=params.disc_height,
            roughness=params.surface_roughness,
            z_offset=disc_z_offset,
            resolution=params.resolution,
            rng=rng
        )
        all_parts.extend(roughness_features)

    # Combine all parts
    result = batch_union(all_parts)

    # Calculate statistics
    mesh = result.to_mesh()
    volume = result.volume() if hasattr(result, 'volume') else 0

    # Approximate volume distribution
    np_vol_ratio = (np_radius ** 2) / (disc_radius ** 2)
    np_volume = volume * np_vol_ratio
    af_volume = volume * (1 - np_vol_ratio)

    stats = {
        'triangle_count': len(mesh.tri_verts) // 3 if hasattr(mesh, 'tri_verts') else 0,
        'volume_mm3': volume,
        'ring_count': params.num_lamellae,
        'num_lamellae': params.num_lamellae,
        'af_volume_mm3': af_volume,
        'np_volume_mm3': np_volume,
        'fiber_count': len(af_manifolds),
        'endplates_enabled': params.enable_endplates,
        'vascular_enabled': params.enable_outer_vascular,
        'nutrient_channels_enabled': params.enable_nutrient_channels,
        'notochordal_enabled': params.enable_notochordal_cells,
        'fissures_enabled': params.enable_fissures,
        'degeneration_level': params.degeneration_level,
        'scaffold_type': 'intervertebral_disc'
    }

    return result, stats


def create_nucleus_pulposus(
    radius: float,
    height: float,
    porosity: float,
    pore_size: float,
    resolution: int,
    rng: np.random.Generator,
    water_content: float = 0.80,
    proteoglycan_content: float = 0.8,
    position_noise: float = 0.1
) -> m3d.Manifold:
    """
    Create nucleus pulposus with high porosity (gel-like center).

    Args:
        radius: Radius of nucleus pulposus
        height: Height of NP region
        porosity: Target porosity (0.8-0.95)
        pore_size: Size of pores in mm
        resolution: Angular resolution
        rng: Random number generator
        water_content: Water content (0-1), higher = more open structure
        proteoglycan_content: Proteoglycan content (0-1), higher = smaller pores
        position_noise: Random position jitter factor

    Returns:
        Porous nucleus pulposus manifold
    """
    # Create base cylinder
    base = m3d.Manifold.cylinder(height, radius, radius, resolution)

    # Modulate porosity based on water content (higher water = more open structure)
    # Water content 0.80 = baseline, range 0.70-0.95
    water_factor = 0.5 + (water_content / 0.80) * 0.5  # Scale around 1.0 at baseline
    effective_porosity = porosity * water_factor
    effective_porosity = max(0.3, min(0.98, effective_porosity))  # Clamp to reasonable range

    # Modulate pore size based on proteoglycan content (higher PG = smaller pores)
    # Proteoglycans attract water and create a gel matrix with smaller pore spaces
    # Higher proteoglycan = tighter matrix = smaller pores
    pg_factor = 1.0 / (0.5 + proteoglycan_content * 0.625)  # ~1.0 at 0.8, smaller at higher values
    effective_pore_size = pore_size * pg_factor
    effective_pore_size = max(0.05, min(pore_size * 2, effective_pore_size))

    # Use effective values for pore generation
    pore_radius = effective_pore_size / 2
    # Spacing inversely related to porosity (higher porosity = closer spacing)
    pore_spacing = effective_pore_size * (3.0 - effective_porosity * 2.0)

    n_radial = max(2, int(radius / pore_spacing))
    n_angular = max(4, int(2 * np.pi * radius / pore_spacing))
    n_vertical = max(2, int(height / pore_spacing))

    pores = []
    for i in range(n_radial):
        r = (i + 0.5) * (radius / n_radial) * 0.8

        for j in range(n_angular):
            angle = j * 2 * np.pi / n_angular
            # Add slight randomness
            angle += rng.uniform(-0.1, 0.1)

            for k in range(n_vertical):
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                z = (k + 0.5) * (height / n_vertical)

                # Add position noise scaled by position_noise parameter
                noise_scale = effective_pore_size * position_noise
                x += rng.uniform(-noise_scale, noise_scale)
                y += rng.uniform(-noise_scale, noise_scale)
                z += rng.uniform(-noise_scale, noise_scale)
                z = max(pore_radius, min(height - pore_radius, z))

                pore = m3d.Manifold.sphere(pore_radius, max(4, resolution // 2))
                pore = pore.translate([x, y, z])
                pores.append(pore)

    # Subtract pores from base
    result = base
    for pore in pores:
        result = result - pore

    return result


def create_notochordal_region(
    radius: float,
    height: float,
    resolution: int,
    rng: np.random.Generator
) -> m3d.Manifold:
    """
    Create notochordal remnants - larger spherical voids in central NP.

    Notochordal cells are large (25-85um), located in NP center.
    We model these as larger void spaces.

    Args:
        radius: Radius of notochordal region
        height: Height of region
        resolution: Angular resolution
        rng: Random number generator

    Returns:
        Manifold with larger notochordal void spaces
    """
    # Large notochordal cell voids (>30um = 0.03mm)
    void_radius = 0.04  # 40um typical
    void_spacing = void_radius * 4

    n_radial = max(1, int(radius / void_spacing))
    n_angular = max(3, int(2 * np.pi * radius / void_spacing))
    n_vertical = max(2, int(height / void_spacing))

    voids = []
    for i in range(n_radial):
        r = (i + 0.5) * (radius / n_radial) * 0.7

        for j in range(n_angular):
            angle = j * 2 * np.pi / n_angular + rng.uniform(-0.2, 0.2)

            for k in range(n_vertical):
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                z = (k + 0.5) * height / n_vertical

                # Vary void size within biological range
                this_void_radius = void_radius * rng.uniform(0.8, 1.5)

                void = m3d.Manifold.sphere(this_void_radius, max(4, resolution // 2))
                void = void.translate([x, y, z])
                voids.append(void)

    if not voids:
        return m3d.Manifold()

    # Create a shell structure with notochordal voids
    base = m3d.Manifold.cylinder(height, radius * 0.3, radius * 0.3, resolution)
    for void in voids:
        base = base - void

    return base


def create_transition_zone(
    np_radius: float,
    zone_width: float,
    height: float,
    gradient_type: str,
    np_porosity: float,
    af_porosity: float,
    resolution: int,
    rng: np.random.Generator
) -> m3d.Manifold:
    """
    Create transition zone at NP-AF interface with gradual property change.

    Args:
        np_radius: Radius of nucleus pulposus
        zone_width: Width of transition zone
        height: Height of disc
        gradient_type: 'linear', 'sigmoid', or 'step'
        np_porosity: Porosity at inner edge (NP side)
        af_porosity: Porosity at outer edge (AF side)
        resolution: Angular resolution
        rng: Random number generator

    Returns:
        Transition zone manifold
    """
    outer_radius = np_radius + zone_width

    # Create base annular cylinder
    outer = m3d.Manifold.cylinder(height, outer_radius, outer_radius, resolution)
    inner = m3d.Manifold.cylinder(height + 0.1, np_radius, np_radius, resolution)
    inner = inner.translate([0, 0, -0.05])
    base = outer - inner

    # Create pores with gradient
    n_rings = 3  # Number of pore rings in transition
    pores = []

    for ring in range(n_rings):
        t = ring / (n_rings - 1) if n_rings > 1 else 0.5

        # Calculate gradient factor based on type
        if gradient_type == 'sigmoid':
            # Sigmoid transition (smooth S-curve)
            gradient = 1 / (1 + np.exp(-10 * (t - 0.5)))
        elif gradient_type == 'step':
            # Step transition
            gradient = 0.0 if t < 0.5 else 1.0
        else:  # linear
            gradient = t

        # Interpolate porosity
        local_porosity = np_porosity + gradient * (af_porosity - np_porosity)

        # Pore size decreases from NP to AF
        pore_radius = 0.15 * (1 - gradient * 0.5)
        r = np_radius + (ring + 0.5) * zone_width / n_rings

        n_angular = max(6, int(2 * np.pi * r / (pore_radius * 4)))
        n_vertical = max(3, int(height / (pore_radius * 4)))

        for j in range(n_angular):
            angle = j * 2 * np.pi / n_angular + rng.uniform(-0.1, 0.1)

            for k in range(n_vertical):
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                z = (k + 0.5) * height / n_vertical

                pore = m3d.Manifold.sphere(pore_radius, max(4, resolution // 2))
                pore = pore.translate([x, y, z])
                pores.append(pore)

    # Subtract pores
    for pore in pores:
        base = base - pore

    return base


def create_annulus_fibrosus_lamellae(
    inner_radius: float,
    outer_radius: float,
    height: float,
    num_lamellae: int,
    inner_angle: float,
    outer_angle: float,
    angle_variance: float,
    lamella_thickness: float,
    interlaminar_spacing: float,
    enable_connections: bool,
    fiber_radius: float,
    resolution: int,
    rng: np.random.Generator,
    water_content: float = 0.60,
    lamella_variance: float = 0.05,
    fiber_variance: float = 0.1
) -> list[m3d.Manifold]:
    """
    Create annulus fibrosus with proper lamellae structure.

    Creates num_lamellae (typically 20) concentric thin rings with:
    - Alternating +/- fiber orientations
    - Fiber angle gradient from inner (steeper) to outer (shallower)
    - Optional interlaminar connections (elastic fiber bridges)

    Args:
        inner_radius: Inner radius of AF region
        outer_radius: Outer radius of AF
        height: Height of disc
        num_lamellae: Number of lamellae (typically 20)
        inner_angle: Fiber angle at inner AF (degrees from vertical)
        outer_angle: Fiber angle at outer AF (degrees from vertical)
        angle_variance: Random variance in fiber angle
        lamella_thickness: Thickness of each lamella
        interlaminar_spacing: Gap between lamellae
        enable_connections: Whether to add interlaminar connections
        fiber_radius: Radius of individual fibers
        resolution: Angular resolution
        rng: Random number generator
        water_content: AF water content (0-1), higher = larger interlaminar spacing
        lamella_variance: Variance in lamella thickness (0-1)
        fiber_variance: Variance in fiber properties (0-1)

    Returns:
        List of lamella manifolds
    """
    lamellae = []
    total_thickness = (outer_radius - inner_radius)

    # Modulate interlaminar spacing based on water content
    # Higher water content = more hydrated = larger spacing between lamellae
    # Baseline water content is 0.60
    water_factor = water_content / 0.60  # Scale around 1.0 at baseline
    effective_spacing = interlaminar_spacing * water_factor

    # Each lamella has thickness + spacing
    lamella_unit = lamella_thickness + effective_spacing
    actual_num_lamellae = min(num_lamellae, int(total_thickness / lamella_unit))

    if actual_num_lamellae < 1:
        actual_num_lamellae = 1
        lamella_unit = total_thickness / actual_num_lamellae

    for i in range(actual_num_lamellae):
        # Progress from inner to outer (0 to 1)
        t = i / (actual_num_lamellae - 1) if actual_num_lamellae > 1 else 0.5

        # Calculate fiber angle with gradient (inner=65deg, outer=50deg from vertical)
        # Note: biological convention - inner is steeper
        base_angle = inner_angle + t * (outer_angle - inner_angle)

        # Add variance
        angle = base_angle + rng.uniform(-angle_variance, angle_variance)

        # Alternating +/- pattern between lamellae
        sign = 1 if i % 2 == 0 else -1
        fiber_angle = sign * angle

        # Calculate radii for this lamella with variance
        # Apply lamella_variance to thickness
        thickness_variation = 1.0 + rng.uniform(-lamella_variance, lamella_variance)
        this_lamella_thickness = lamella_thickness * thickness_variation

        r_inner = inner_radius + i * lamella_unit
        r_outer = r_inner + this_lamella_thickness

        if r_outer > outer_radius:
            r_outer = outer_radius

        # Apply fiber_variance to fiber radius
        fiber_variation = 1.0 + rng.uniform(-fiber_variance, fiber_variance)
        this_fiber_radius = fiber_radius * fiber_variation

        # Create helical lamella
        lamella = create_helical_lamella(
            r_inner=r_inner,
            r_outer=r_outer,
            height=height,
            fiber_angle=fiber_angle,
            fiber_radius=this_fiber_radius,
            resolution=resolution,
            rng=rng
        )

        if lamella.num_vert() > 0:
            lamellae.append(lamella)

    # Add interlaminar connections if enabled
    if enable_connections and len(lamellae) > 1:
        connections = create_interlaminar_connections(
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            height=height,
            num_lamellae=actual_num_lamellae,
            lamella_unit=lamella_unit,
            resolution=resolution,
            rng=rng
        )
        lamellae.extend(connections)

    return lamellae


def create_helical_lamella(
    r_inner: float,
    r_outer: float,
    height: float,
    fiber_angle: float,
    fiber_radius: float,
    resolution: int,
    rng: np.random.Generator
) -> m3d.Manifold:
    """
    Create a single lamella with helical fiber pattern.

    Args:
        r_inner: Inner radius of lamella
        r_outer: Outer radius of lamella
        height: Height of disc
        fiber_angle: Helix angle in degrees (+ or -)
        fiber_radius: Radius of fibers
        resolution: Angular resolution
        rng: Random number generator

    Returns:
        Lamella manifold with helical fibers
    """
    fibers = []
    r_mid = (r_inner + r_outer) / 2

    # Number of fibers around circumference
    circumference = 2 * np.pi * r_mid
    fiber_spacing = fiber_radius * 4
    n_fibers = max(8, int(circumference / fiber_spacing))

    for i in range(n_fibers):
        angle_start = i * 2 * np.pi / n_fibers

        fiber = create_helical_fiber(
            r_mid=r_mid,
            ring_width=r_outer - r_inner,
            height=height,
            angle_start=angle_start,
            fiber_angle=fiber_angle,
            fiber_radius=fiber_radius,
            resolution=resolution
        )

        if fiber.num_vert() > 0:
            fibers.append(fiber)

    if not fibers:
        return m3d.Manifold()

    return batch_union(fibers)


def create_interlaminar_connections(
    inner_radius: float,
    outer_radius: float,
    height: float,
    num_lamellae: int,
    lamella_unit: float,
    resolution: int,
    rng: np.random.Generator
) -> list[m3d.Manifold]:
    """
    Create elastic fiber bridges between lamellae.

    Args:
        inner_radius: Inner radius of AF
        outer_radius: Outer radius of AF
        height: Height of disc
        num_lamellae: Number of lamellae
        lamella_unit: Thickness + spacing per lamella
        resolution: Angular resolution
        rng: Random number generator

    Returns:
        List of connection manifolds
    """
    connections = []
    connection_radius = 0.03  # Small bridging fibers

    # Connections between adjacent lamellae
    for i in range(num_lamellae - 1):
        r_from = inner_radius + (i + 0.5) * lamella_unit
        r_to = inner_radius + (i + 1.5) * lamella_unit

        if r_to > outer_radius:
            continue

        # Create sparse connections
        n_angular = max(4, int(2 * np.pi * r_from / (lamella_unit * 10)))
        n_vertical = max(2, int(height / (lamella_unit * 10)))

        for j in range(n_angular):
            angle = j * 2 * np.pi / n_angular + rng.uniform(-0.2, 0.2)

            for k in range(n_vertical):
                z = (k + 0.5) * height / n_vertical

                # Create radial bridge
                x1 = r_from * np.cos(angle)
                y1 = r_from * np.sin(angle)
                x2 = r_to * np.cos(angle)
                y2 = r_to * np.sin(angle)

                bridge = make_fiber_segment(
                    np.array([x1, y1, z]),
                    np.array([x2, y2, z]),
                    connection_radius,
                    max(4, resolution // 2)
                )

                if bridge.num_vert() > 0:
                    connections.append(bridge)

    return connections


def create_endplate(
    radius: float,
    thickness: float,
    porosity: float,
    pore_size: float,
    enable_nutrient_channels: bool,
    nutrient_channel_diameter: float,
    nutrient_channel_density: float,
    np_radius: float,
    resolution: int,
    rng: np.random.Generator,
    enable_endplate_pores: bool = True
) -> m3d.Manifold:
    """
    Create cartilaginous endplate with nutrient diffusion channels.

    Args:
        radius: Radius of endplate (disc radius)
        thickness: Endplate thickness (0.6-1mm typical)
        porosity: Endplate porosity (55% typical)
        pore_size: Size of pores for diffusion
        enable_nutrient_channels: Whether to add nutrient channels
        nutrient_channel_diameter: Diameter of nutrient channels
        nutrient_channel_density: Channels per mm^2
        np_radius: Radius of nucleus pulposus (channels concentrated here)
        resolution: Angular resolution
        rng: Random number generator
        enable_endplate_pores: Whether to add distributed pores for nutrient diffusion

    Returns:
        Endplate manifold with pores and channels
    """
    # Create base disc
    base = m3d.Manifold.cylinder(thickness, radius, radius, resolution)

    # Add distributed pores for nutrient diffusion pathways (if enabled)
    if enable_endplate_pores:
        pore_radius = pore_size / 2
        pore_spacing = pore_size * 2.5

        n_radial = max(2, int(radius / pore_spacing))
        n_angular = max(6, int(2 * np.pi * radius / pore_spacing))

        for i in range(n_radial):
            r = (i + 0.5) * radius / n_radial
            this_n_angular = max(4, int(2 * np.pi * r / pore_spacing))

            for j in range(this_n_angular):
                angle = j * 2 * np.pi / this_n_angular + rng.uniform(-0.1, 0.1)

                x = r * np.cos(angle)
                y = r * np.sin(angle)
                z = thickness / 2

                pore = m3d.Manifold.sphere(pore_radius, max(4, resolution // 2))
                pore = pore.translate([x, y, z])
                base = base - pore

        # Add additional through-thickness micropores in the NP region for enhanced diffusion
        # These are smaller than nutrient channels but go completely through
        micropore_radius = pore_radius * 0.5
        micropore_spacing = pore_spacing * 1.5
        n_micropore_radial = max(2, int(np_radius / micropore_spacing))

        for i in range(n_micropore_radial):
            r = (i + 0.5) * np_radius / n_micropore_radial * 0.9
            this_n_angular = max(4, int(2 * np.pi * r / micropore_spacing))

            for j in range(this_n_angular):
                angle = j * 2 * np.pi / this_n_angular + rng.uniform(-0.15, 0.15)

                x = r * np.cos(angle)
                y = r * np.sin(angle)

                # Vertical micropore through entire endplate
                micropore = m3d.Manifold.cylinder(
                    thickness + 0.1,
                    micropore_radius,
                    micropore_radius,
                    max(4, resolution // 2)
                )
                micropore = micropore.translate([x, y, -0.05])
                base = base - micropore

    # Add nutrient channels (vertical through endplate, concentrated over NP)
    if enable_nutrient_channels:
        channel_radius = nutrient_channel_diameter / 2
        # Channels per area
        np_area = np.pi * np_radius ** 2
        num_channels = int(np_area * nutrient_channel_density)
        num_channels = max(3, min(num_channels, 50))  # Reasonable limits

        for _ in range(num_channels):
            # Random position within NP area
            r = np_radius * np.sqrt(rng.uniform(0, 1))
            angle = rng.uniform(0, 2 * np.pi)

            x = r * np.cos(angle)
            y = r * np.sin(angle)

            # Vertical channel through endplate
            channel = m3d.Manifold.cylinder(
                thickness + 0.2,  # Extend slightly
                channel_radius,
                channel_radius,
                max(4, resolution // 2)
            )
            channel = channel.translate([x, y, -0.1])
            base = base - channel

    return base


def create_vascular_supply(
    disc_radius: float,
    af_inner_radius: float,
    penetration_depth: float,
    channel_diameter: float,
    height: float,
    resolution: int,
    rng: np.random.Generator
) -> list[m3d.Manifold]:
    """
    Create vascular supply channels in outer 1/3 of AF.

    Only the outer 33% of AF is vascularized.

    Args:
        disc_radius: Outer radius of disc
        af_inner_radius: Inner radius of AF
        penetration_depth: Fraction of AF thickness with vessels
        channel_diameter: Vessel diameter
        height: Height of disc
        resolution: Angular resolution
        rng: Random number generator

    Returns:
        List of vascular channel manifolds
    """
    channels = []
    channel_radius = channel_diameter / 2

    # Only outer 1/3 of AF is vascularized
    af_thickness = disc_radius - af_inner_radius
    vascular_zone_inner = disc_radius - af_thickness * 0.33  # Outer 33%
    vascular_zone_outer = disc_radius

    # Sparse vascular channels
    n_angular = max(6, int(2 * np.pi * disc_radius / (channel_diameter * 20)))
    n_vertical = max(2, int(height / (channel_diameter * 10)))

    for j in range(n_angular):
        angle = j * 2 * np.pi / n_angular + rng.uniform(-0.2, 0.2)
        r = rng.uniform(vascular_zone_inner, vascular_zone_outer * 0.95)

        for k in range(n_vertical):
            z = (k + 0.5) * height / n_vertical

            x = r * np.cos(angle)
            y = r * np.sin(angle)

            # Small vessel segment (horizontal/radial orientation)
            vessel = m3d.Manifold.sphere(channel_radius * 1.5, max(4, resolution // 2))
            vessel = vessel.translate([x, y, z])
            channels.append(vessel)

    return channels


def create_fissures(
    np_radius: float,
    disc_radius: float,
    height: float,
    fissure_count: int,
    degeneration_level: float,
    resolution: int,
    rng: np.random.Generator
) -> list[m3d.Manifold]:
    """
    Create degeneration fissures (radial/circumferential tears in AF).

    Args:
        np_radius: Radius of nucleus pulposus
        disc_radius: Outer radius of disc
        height: Height of disc
        fissure_count: Number of fissures
        degeneration_level: Severity (0-1)
        resolution: Angular resolution
        rng: Random number generator

    Returns:
        List of fissure manifolds (thin gaps)
    """
    fissures = []
    fissure_thickness = 0.1 * (0.5 + degeneration_level)  # Wider with more degeneration

    for _ in range(fissure_count):
        # Random fissure type: radial or circumferential
        is_radial = rng.random() > 0.5

        angle = rng.uniform(0, 2 * np.pi)
        z_pos = rng.uniform(height * 0.2, height * 0.8)

        if is_radial:
            # Radial fissure - extends from NP outward
            length = (disc_radius - np_radius) * (0.3 + 0.5 * degeneration_level)
            r_start = np_radius * 1.1
            r_end = r_start + length

            # Create thin box for fissure
            fissure_width = fissure_thickness
            fissure_height = height * 0.2 * (0.5 + degeneration_level)

            # Create elongated shape
            fissure = m3d.Manifold.cylinder(length, fissure_width, fissure_width, 4)
            fissure = fissure.rotate([0, 90, 0])  # Rotate to radial
            fissure = fissure.rotate([0, 0, angle * 180 / np.pi])

            x = (r_start + length/2) * np.cos(angle)
            y = (r_start + length/2) * np.sin(angle)
            fissure = fissure.translate([x, y, z_pos])
        else:
            # Circumferential fissure - arc within AF
            r = rng.uniform(np_radius * 1.2, disc_radius * 0.8)
            arc_angle = np.pi * 0.3 * (0.5 + degeneration_level)  # Partial arc

            # Create series of small segments for arc
            n_segments = max(3, int(arc_angle * r / fissure_thickness / 2))
            for i in range(n_segments):
                seg_angle = angle + (i / n_segments) * arc_angle
                x = r * np.cos(seg_angle)
                y = r * np.sin(seg_angle)

                seg = m3d.Manifold.sphere(fissure_thickness, 4)
                seg = seg.translate([x, y, z_pos])
                fissures.append(seg)
            continue

        if fissure.num_vert() > 0:
            fissures.append(fissure)

    return fissures


def create_surface_roughness(
    disc_radius: float,
    disc_height: float,
    roughness: float,
    z_offset: float,
    resolution: int,
    rng: np.random.Generator
) -> list[m3d.Manifold]:
    """
    Create surface roughness features on the outer rim of the disc.

    Adds small bumps and variations to the outer surface to simulate
    natural tissue texture.

    Args:
        disc_radius: Radius of the disc
        disc_height: Height of the disc
        roughness: Roughness factor (0-1), controls bump size
        z_offset: Z offset for the disc
        resolution: Angular resolution
        rng: Random number generator

    Returns:
        List of small bump manifolds on the outer surface
    """
    features = []

    # Bump size based on roughness parameter (0.05 roughness = small bumps)
    bump_radius = roughness * 0.5  # Max bump radius of 0.5mm at roughness=1.0
    bump_radius = max(0.01, min(0.3, bump_radius))  # Clamp to reasonable range

    # Calculate spacing between bumps
    bump_spacing = bump_radius * 4
    circumference = 2 * np.pi * disc_radius

    n_angular = max(8, int(circumference / bump_spacing))
    n_vertical = max(3, int(disc_height / bump_spacing))

    for j in range(n_angular):
        base_angle = j * 2 * np.pi / n_angular

        for k in range(n_vertical):
            # Add randomness to position
            angle = base_angle + rng.uniform(-0.2, 0.2)
            z = z_offset + (k + 0.5) * disc_height / n_vertical
            z += rng.uniform(-bump_spacing * 0.2, bump_spacing * 0.2)

            # Clamp z to valid range
            z = max(z_offset + bump_radius, min(z_offset + disc_height - bump_radius, z))

            # Random bump size variation
            this_bump_radius = bump_radius * rng.uniform(0.5, 1.5)

            # Position on outer surface (slightly inward so bump protrudes)
            r = disc_radius - this_bump_radius * 0.3

            x = r * np.cos(angle)
            y = r * np.sin(angle)

            # Create bump (slightly elongated radially for natural look)
            bump = m3d.Manifold.sphere(this_bump_radius, max(4, resolution // 2))
            # Scale slightly in radial direction
            scale_factor = 1.0 + rng.uniform(0, 0.3)
            bump = bump.scale([scale_factor, scale_factor, 1.0])
            bump = bump.translate([x, y, z])
            features.append(bump)

    return features


def create_helical_fiber(
    r_mid: float,
    ring_width: float,
    height: float,
    angle_start: float,
    fiber_angle: float,
    fiber_radius: float,
    resolution: int
) -> m3d.Manifold:
    """
    Create a helical fiber wrapping around at specified angle.

    Args:
        r_mid: Mid-radius of ring
        ring_width: Width of ring (radial extent)
        height: Height of disc
        angle_start: Starting angular position
        fiber_angle: Helix angle in degrees
        fiber_radius: Radius of fiber
        resolution: Number of segments for smooth curve

    Returns:
        Helical fiber manifold
    """
    # Number of segments for one complete wrap
    n_segments = max(8, int(2 * np.pi * r_mid / (fiber_radius * 3)))

    # Calculate pitch (vertical rise per full rotation)
    pitch = 2 * np.pi * r_mid * np.tan(fiber_angle * np.pi / 180)

    # Number of rotations to cover height
    n_rotations = height / abs(pitch) if pitch != 0 else 0.5

    # Generate path points
    path_points = []
    total_angle = n_rotations * 2 * np.pi

    for i in range(int(n_segments * n_rotations) + 1):
        t = i / (n_segments * n_rotations) if n_rotations > 0 else 0.5
        angle = angle_start + t * total_angle

        # Slight radial variation within ring width
        r = r_mid + (np.sin(i * 0.5) * ring_width * 0.3)

        x = r * np.cos(angle)
        y = r * np.sin(angle)
        z = t * height

        path_points.append(np.array([x, y, z]))

    # Create segments
    segments = []
    for i in range(len(path_points) - 1):
        p1 = path_points[i]
        p2 = path_points[i + 1]

        segment = make_fiber_segment(p1, p2, fiber_radius, resolution)
        if segment.num_vert() > 0:
            segments.append(segment)

    if not segments:
        return m3d.Manifold()

    # Union segments
    result = segments[0]
    for seg in segments[1:]:
        result = result + seg

    return result


def make_fiber_segment(p1: np.ndarray, p2: np.ndarray, radius: float, resolution: int) -> m3d.Manifold:
    """
    Create a cylindrical fiber segment between two points.

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


def generate_intervertebral_disc_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate intervertebral disc from dictionary parameters (for API compatibility).

    Args:
        params: Dictionary with keys matching IntervertebralDiscParams fields

    Returns:
        Tuple of (manifold, stats_dict)
    """
    return generate_intervertebral_disc(IntervertebralDiscParams(
        # Basic geometry
        disc_diameter=params.get('disc_diameter', 40.0),
        disc_height=params.get('disc_height', 8.5),
        resolution=params.get('resolution', 16),
        seed=params.get('seed', 42),

        # Nucleus pulposus
        np_diameter=params.get('np_diameter', 16.0),
        nucleus_percentage=params.get('nucleus_percentage', 0.40),
        np_height_ratio=params.get('np_height_ratio', 0.85),
        np_porosity=params.get('np_porosity', 0.90),
        np_water_content=params.get('np_water_content', 0.80),
        np_proteoglycan_content=params.get('np_proteoglycan_content', 0.8),
        np_pore_size=params.get('np_pore_size', 0.3),

        # Annulus fibrosus
        af_ring_count=params.get('af_ring_count', 3),
        num_lamellae=params.get('num_lamellae', 20),
        annulus_percentage=params.get('annulus_percentage', 0.60),
        af_porosity=params.get('af_porosity', 0.65),
        af_water_content=params.get('af_water_content', 0.60),

        # Fiber properties
        fiber_diameter=params.get('fiber_diameter', 0.2),
        fiber_diameter_um=params.get('fiber_diameter_um', 200.0),
        fiber_angle=params.get('fiber_angle', 30.0),
        af_layer_angle=params.get('af_layer_angle', 30.0),
        inner_af_fiber_angle_deg=params.get('inner_af_fiber_angle_deg', 45.0),
        outer_af_fiber_angle_deg=params.get('outer_af_fiber_angle_deg', 65.0),
        fiber_angle_variance_deg=params.get('fiber_angle_variance_deg', 5.0),

        # Lamella properties
        lamella_thickness=params.get('lamella_thickness', 0.15),
        interlaminar_spacing=params.get('interlaminar_spacing', 0.02),
        enable_interlaminar_connections=params.get('enable_interlaminar_connections', True),

        # Mechanical indicators
        np_stiffness_kpa=params.get('np_stiffness_kpa', 2.0),
        af_stiffness_kpa=params.get('af_stiffness_kpa', 100.0),
        stiffness_gradient=params.get('stiffness_gradient', True),

        # Endplates
        enable_endplates=params.get('enable_endplates', True),
        endplate_thickness=params.get('endplate_thickness', 0.7),
        endplate_porosity=params.get('endplate_porosity', 0.55),
        endplate_pore_size=params.get('endplate_pore_size', 0.1),

        # Nutrient pathways
        enable_nutrient_channels=params.get('enable_nutrient_channels', True),
        nutrient_channel_diameter=params.get('nutrient_channel_diameter', 0.05),
        nutrient_channel_density=params.get('nutrient_channel_density', 0.3),
        enable_endplate_pores=params.get('enable_endplate_pores', True),

        # Transition zone
        transition_zone_width=params.get('transition_zone_width', 0.5),
        transition_gradient_type=params.get('transition_gradient_type', 'linear'),

        # Vascular
        enable_outer_vascular=params.get('enable_outer_vascular', True),
        vascular_channel_diameter=params.get('vascular_channel_diameter', 0.1),
        vascular_penetration_depth=params.get('vascular_penetration_depth', 0.2),

        # Notochordal
        enable_notochordal_cells=params.get('enable_notochordal_cells', False),
        notochordal_region_radius=params.get('notochordal_region_radius', 2.0),

        # Age-related
        degeneration_level=params.get('degeneration_level', 0.0),
        enable_fissures=params.get('enable_fissures', False),
        fissure_count=params.get('fissure_count', 0),

        # Surface
        surface_roughness=params.get('surface_roughness', 0.05),

        # Randomization
        position_noise=params.get('position_noise', 0.1),
        fiber_variance=params.get('fiber_variance', 0.1),
        lamella_variance=params.get('lamella_variance', 0.05),
    ))
