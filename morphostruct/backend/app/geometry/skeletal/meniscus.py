"""
Meniscus scaffold generator with wedge-shaped geometry.

Creates a wedge-shaped structure with radial and circumferential zones
mimicking the fibrocartilaginous structure of knee meniscus.

Biological basis:
- Circumferential fibers: Main load-bearing, 20-50μm bundles (up to 480μm fascicles)
- Radial tie fibers: Prevent splitting, 10μm sheets at 0.6-1mm spacing
- Vascular zones: Outer 10-30% only (red zone), inner 65-75% avascular (white zone)
- Three-layer structure: Superficial (random), lamellar, central
- Cell density: 27,199/mm³ (vascular) to 12,820/mm³ (avascular)
"""

from __future__ import annotations
import manifold3d as m3d
import numpy as np
from dataclasses import dataclass
from ..core import batch_union


@dataclass
class MeniscusParams:
    """Parameters for meniscus scaffold generation.

    Biologically realistic parameters based on human knee meniscus:
    - Outer diameter: 35-45 mm (medial), 32-35 mm (lateral)
    - Inner diameter: 15-25 mm
    - Thickness: 4-8 mm at periphery, thinner toward center
    - Circumferential fibers: 100-200 μm diameter bundles
    - Radial fibers: 10-30 μm diameter (tie fibers)
    - Porosity: ~60-70% in vascular zone, lower in avascular
    """
    # Basic geometry (mm)
    outer_radius: float = 22.5  # Outer edge radius (45mm diameter / 2)
    inner_radius: float = 10.0  # Inner edge radius (20mm diameter / 2)
    outer_diameter: float = 45.0  # Alternative specification (mm)
    inner_diameter: float = 20.0  # Alternative specification (mm)
    thickness: float = 8.0  # Maximum thickness at periphery
    height: float = 8.0  # Legacy alias for thickness
    arc_degrees: float = 300.0  # C-shape arc extent (not full circle)
    resolution: int = 32
    seed: int = 42

    # Wedge geometry
    wedge_angle_deg: float = 20.0  # Wedge slope angle
    inner_edge_height_ratio: float = 0.3  # Inner edge height as fraction of max height

    # Radial zones (from inner to outer)
    zone_count: int = 3  # Number of radial zones
    outer_zone_thickness_ratio: float = 0.33  # Vascular/red zone
    middle_zone_thickness_ratio: float = 0.33  # Red-white zone
    inner_zone_thickness_ratio: float = 0.34  # Avascular/white zone

    # Circumferential fiber properties (main load-bearing fibers)
    # Research: 20-50μm bundles, large fascicles up to 480μm, ~200μm spacing
    circumferential_bundle_diameter_um: float = 100.0  # Fiber bundle diameter
    circumferential_bundle_spacing_um: float = 200.0  # Spacing between bundles
    circumferential_fiber_density: float = 0.8  # Density in outer zone (48% of volume)

    # Radial (tie) fiber properties
    # Research: 10μm sheets, macro-spacing 0.6-1mm, micro-spacing 25-100μm
    radial_bundle_diameter_um: float = 20.0  # Tie fiber diameter
    radial_bundle_spacing_um: float = 100.0  # Spacing between radial fibers
    radial_fiber_density: float = 0.6  # Tie fiber density
    enable_radial_tie_fibers: bool = True  # Include tie fibers

    # General fiber properties
    fiber_diameter: float = 0.2  # Legacy: default fiber diameter (mm)
    fiber_orientation_variance_deg: float = 15.0  # Random variance in fiber angle

    # Porosity per zone (research-based gradient)
    outer_zone_porosity: float = 0.65  # Vascular zone (higher for blood supply)
    middle_zone_porosity: float = 0.55  # Transition zone
    inner_zone_porosity: float = 0.45  # Avascular zone (denser)
    porosity: float = 0.61  # Overall target porosity

    # Vascular features (outer 10-30% is vascular - red zone)
    enable_vascular_channels: bool = True  # Blood vessel channels in outer zone
    vascular_channel_diameter: float = 0.1  # Vessel diameter (mm) ~100μm
    vascular_channel_spacing: float = 0.8  # Spacing between vessels (mm)
    vascular_penetration_depth: float = 0.33  # Fraction of radial depth (outer 1/3)

    # Horn attachments (anterior ~61mm², posterior ~30mm²)
    enable_anterior_horn: bool = False  # Anterior horn attachment region
    enable_posterior_horn: bool = False  # Posterior horn attachment region
    horn_insertion_width: float = 5.0  # Width of horn insertion (mm)
    horn_insertion_depth: float = 3.0  # Depth of horn attachment (mm)

    # Surface properties (femoral curvature 21-32mm radius)
    surface_roughness: float = 0.05  # Surface texture
    enable_femoral_surface: bool = True  # Concave upper surface
    enable_tibial_surface: bool = True  # Flatter lower surface
    femoral_curvature_radius: float = 30.0  # Curvature matching femoral condyle (mm)

    # Collagen organization (3 distinct layers)
    enable_lamellar_structure: bool = True  # Layered fiber organization
    lamella_count: int = 6  # Number of fiber layers
    interlaminar_spacing_um: float = 50.0  # Space between layers

    # Cell lacunae (for biological appearance)
    # Research: 27,199 cells/mm³ vascular, 12,820 cells/mm³ avascular
    enable_cell_lacunae: bool = False  # Chondrocyte spaces
    lacuna_density: float = 500.0  # Lacunae per mm³ (scaffold representation)
    lacuna_diameter_um: float = 15.0  # Lacuna size

    # Randomization
    position_noise: float = 0.1  # Random position jitter
    thickness_variance: float = 0.1  # Variation in local thickness


def generate_meniscus(params: MeniscusParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate a meniscus scaffold with wedge-shaped cross-section.

    Creates a C-shaped wedge with:
    - Circumferential fibers in outer zones (main load-bearing)
    - Radial tie fibers preventing splitting
    - Vascular channels in outer zone only
    - Zone-specific porosity gradient
    - Optional horn attachments
    - Femoral surface curvature
    - Lamellar structure with multiple layers
    - Cell lacunae distributed by zone

    Args:
        params: MeniscusParams specifying geometry and structure

    Returns:
        Tuple of (manifold, stats_dict)
        - manifold: The 3D geometry
        - stats_dict: Dictionary with comprehensive statistics
    """
    np.random.seed(params.seed)

    # Create base wedge shape with femoral curvature
    base = create_wedge_base(
        inner_radius=params.inner_radius,
        outer_radius=params.outer_radius,
        height=params.height,
        inner_edge_height_ratio=params.inner_edge_height_ratio,
        arc_degrees=params.arc_degrees,
        resolution=params.resolution,
        enable_femoral_surface=params.enable_femoral_surface,
        femoral_curvature_radius=params.femoral_curvature_radius,
        enable_tibial_surface=params.enable_tibial_surface,
        wedge_angle_deg=params.wedge_angle_deg
    )

    all_fibers = []
    all_pores = []
    vascular_channels = []
    lacunae = []

    # Zone boundaries (from inner to outer)
    radial_width = params.outer_radius - params.inner_radius
    zone_boundaries = [
        params.inner_radius,
        params.inner_radius + radial_width * params.inner_zone_thickness_ratio,
        params.inner_radius + radial_width * (params.inner_zone_thickness_ratio + params.middle_zone_thickness_ratio),
        params.outer_radius
    ]

    # Zone porosities
    zone_porosities = [
        params.inner_zone_porosity,
        params.middle_zone_porosity,
        params.outer_zone_porosity
    ]

    # Create circumferential fiber bundles (main load-bearing structure)
    circumferential_fibers = create_circumferential_fiber_bundles(
        inner_radius=params.inner_radius,
        outer_radius=params.outer_radius,
        height=params.height,
        inner_edge_height_ratio=params.inner_edge_height_ratio,
        arc_degrees=params.arc_degrees,
        bundle_diameter_um=params.circumferential_bundle_diameter_um,
        bundle_spacing_um=params.circumferential_bundle_spacing_um,
        fiber_density=params.circumferential_fiber_density,
        resolution=params.resolution,
        orientation_variance_deg=params.fiber_orientation_variance_deg
    )
    all_fibers.extend(circumferential_fibers)

    # Create radial tie fibers (prevent splitting under load)
    if params.enable_radial_tie_fibers:
        radial_fibers = create_radial_tie_fibers(
            inner_radius=params.inner_radius,
            outer_radius=params.outer_radius,
            height=params.height,
            inner_edge_height_ratio=params.inner_edge_height_ratio,
            arc_degrees=params.arc_degrees,
            bundle_diameter_um=params.radial_bundle_diameter_um,
            bundle_spacing_um=params.radial_bundle_spacing_um,
            fiber_density=params.radial_fiber_density,
            resolution=params.resolution
        )
        all_fibers.extend(radial_fibers)

    # Create lamellar structure if enabled
    if params.enable_lamellar_structure:
        lamellar_fibers = create_lamellar_structure(
            inner_radius=params.inner_radius,
            outer_radius=params.outer_radius,
            height=params.height,
            inner_edge_height_ratio=params.inner_edge_height_ratio,
            arc_degrees=params.arc_degrees,
            lamella_count=params.lamella_count,
            interlaminar_spacing_um=params.interlaminar_spacing_um,
            fiber_diameter=params.fiber_diameter,
            resolution=params.resolution
        )
        all_fibers.extend(lamellar_fibers)

    # Create vascular channels in outer zone ONLY
    if params.enable_vascular_channels:
        vascular_channels = create_vascular_channels(
            inner_radius=params.inner_radius,
            outer_radius=params.outer_radius,
            height=params.height,
            inner_edge_height_ratio=params.inner_edge_height_ratio,
            arc_degrees=params.arc_degrees,
            vascular_penetration_depth=params.vascular_penetration_depth,
            channel_diameter=params.vascular_channel_diameter,
            channel_spacing=params.vascular_channel_spacing,
            resolution=params.resolution
        )

    # Create zone-specific pores based on porosity gradient
    for zone_idx in range(params.zone_count):
        r_inner = zone_boundaries[zone_idx]
        r_outer = zone_boundaries[zone_idx + 1]
        porosity = zone_porosities[zone_idx]

        zone_pores = create_zone_pores(
            r_inner=r_inner,
            r_outer=r_outer,
            height=params.height,
            inner_edge_height_ratio=params.inner_edge_height_ratio,
            arc_degrees=params.arc_degrees,
            porosity=porosity,
            resolution=params.resolution,
            position_noise=params.position_noise,
            thickness_variance=params.thickness_variance
        )
        all_pores.extend(zone_pores)

    # Create cell lacunae if enabled
    if params.enable_cell_lacunae:
        lacunae = create_cell_lacunae(
            inner_radius=params.inner_radius,
            outer_radius=params.outer_radius,
            height=params.height,
            inner_edge_height_ratio=params.inner_edge_height_ratio,
            arc_degrees=params.arc_degrees,
            lacuna_density=params.lacuna_density,
            lacuna_diameter_um=params.lacuna_diameter_um,
            vascular_penetration_depth=params.vascular_penetration_depth,
            resolution=params.resolution
        )

    # Build the scaffold
    result = base

    # Add fiber network using batch_union for efficiency
    if all_fibers:
        fiber_network = batch_union(all_fibers)
        if fiber_network is not None:
            # Intersect fibers with base shape to clip to wedge
            # Use proper intersection (not XOR) to keep only fibers inside the base
            result = m3d.Manifold.batch_boolean([fiber_network, base], m3d.OpType.Intersect)

    # Subtract pores for zone-specific porosity
    if all_pores:
        pore_network = batch_union(all_pores)
        if pore_network is not None:
            result = result - pore_network

    # Subtract vascular channels (outer zone only)
    if vascular_channels:
        vascular_network = batch_union(vascular_channels)
        if vascular_network is not None:
            result = result - vascular_network

    # Subtract cell lacunae
    if lacunae:
        lacunae_network = batch_union(lacunae)
        if lacunae_network is not None:
            result = result - lacunae_network

    # Apply surface roughness if specified
    if params.surface_roughness > 0:
        roughness_features = create_surface_roughness(
            inner_radius=params.inner_radius,
            outer_radius=params.outer_radius,
            height=params.height,
            inner_edge_height_ratio=params.inner_edge_height_ratio,
            arc_degrees=params.arc_degrees,
            roughness_scale=params.surface_roughness,
            resolution=params.resolution
        )
        if roughness_features:
            roughness_network = batch_union(roughness_features)
            if roughness_network is not None:
                result = result + roughness_network

    # Add horn attachments if enabled
    if params.enable_anterior_horn or params.enable_posterior_horn:
        horns = create_horn_attachments(
            inner_radius=params.inner_radius,
            outer_radius=params.outer_radius,
            height=params.height,
            arc_degrees=params.arc_degrees,
            horn_width=params.horn_insertion_width,
            horn_depth=params.horn_insertion_depth,
            enable_anterior=params.enable_anterior_horn,
            enable_posterior=params.enable_posterior_horn,
            resolution=params.resolution
        )
        if horns:
            horn_mesh = batch_union(horns)
            if horn_mesh is not None:
                result = result + horn_mesh

    # Calculate statistics
    mesh = result.to_mesh()
    volume = result.volume() if hasattr(result, 'volume') else 0

    stats = {
        'triangle_count': len(mesh.tri_verts) // 3 if hasattr(mesh, 'tri_verts') else 0,
        'volume_mm3': volume,
        'fiber_count': len(all_fibers),
        'circumferential_fiber_count': len(circumferential_fibers),
        'radial_fiber_count': len(radial_fibers) if params.enable_radial_tie_fibers else 0,
        'vascular_channel_count': len(vascular_channels),
        'pore_count': len(all_pores),
        'lacuna_count': len(lacunae),
        'zone_count': params.zone_count,
        'zone_porosities': zone_porosities,
        'lamellar_layers': params.lamella_count if params.enable_lamellar_structure else 0,
        'has_anterior_horn': params.enable_anterior_horn,
        'has_posterior_horn': params.enable_posterior_horn,
        'scaffold_type': 'meniscus'
    }

    return result, stats


def create_wedge_base(
    inner_radius: float,
    outer_radius: float,
    height: float,
    inner_edge_height_ratio: float,
    arc_degrees: float,
    resolution: int,
    enable_femoral_surface: bool = True,
    femoral_curvature_radius: float = 30.0,
    enable_tibial_surface: bool = True,
    wedge_angle_deg: float = 20.0
) -> m3d.Manifold:
    """
    Create the base wedge-shaped manifold (C-shaped with triangular cross-section).

    The femoral (top) surface is concave to match the femoral condyle curvature.
    The tibial (bottom) surface is relatively flat.

    Args:
        inner_radius: Inner radius of C-shape
        outer_radius: Outer radius of C-shape
        height: Maximum height (at outer edge)
        inner_edge_height_ratio: Height ratio at inner edge (typically 0.3)
        arc_degrees: Arc extent (e.g., 300 for C-shape)
        resolution: Angular resolution
        enable_femoral_surface: Apply concave curvature to top surface
        femoral_curvature_radius: Radius of femoral surface curvature (21-32mm)
        enable_tibial_surface: Apply slight concavity to bottom surface when True (default flat)
        wedge_angle_deg: Wedge slope angle (affects slope box rotation)

    Returns:
        Wedge-shaped manifold
    """
    # Create outer cylinder
    outer_cyl = m3d.Manifold.cylinder(height, outer_radius, outer_radius, resolution)

    # Create inner cylinder to subtract
    inner_cyl = m3d.Manifold.cylinder(height * 2, inner_radius, inner_radius, resolution)
    inner_cyl = inner_cyl.translate([0, 0, -height * 0.5])

    # Create annular section
    annulus = outer_cyl - inner_cyl

    # Create wedge slope by subtracting a sloped volume
    # The inner edge should be lower than outer edge
    inner_height = height * inner_edge_height_ratio

    # Create a cone-like shape to subtract from top for wedge effect
    # This creates the characteristic triangular cross-section
    wedge_cut_height = height - inner_height
    if wedge_cut_height > 0.01:
        # Create the sloped cut using a box with rotation
        # Use wedge_angle_deg to control slope factor
        slope_factor = np.tan(wedge_angle_deg * np.pi / 180) / np.tan(20.0 * np.pi / 180) * 0.3
        slope_angle = np.arctan2(wedge_cut_height, outer_radius - inner_radius) * 180 / np.pi
        slope_box = m3d.Manifold.cube([outer_radius * 4, outer_radius * 4, height])
        slope_box = slope_box.translate([-outer_radius * 2, -outer_radius * 2, inner_height])
        slope_box = slope_box.rotate([0, -slope_angle * slope_factor, 0])
        slope_box = slope_box.translate([0, 0, wedge_cut_height * slope_factor])

        # Apply the wedge cut to create triangular cross-section
        annulus = annulus - slope_box

    # Apply femoral surface curvature (concave on top)
    if enable_femoral_surface and femoral_curvature_radius > 0:
        # Create a sphere positioned above to create concave top surface
        sphere_offset = femoral_curvature_radius - height * 0.3
        femoral_sphere = m3d.Manifold.sphere(femoral_curvature_radius, resolution)
        femoral_sphere = femoral_sphere.translate([0, 0, height + sphere_offset])

        # Subtract sphere to create concave surface
        annulus = annulus - femoral_sphere

    # Apply tibial surface curvature (slight concavity when enabled)
    if enable_tibial_surface and not enable_femoral_surface:
        # Add very slight concavity to tibial (bottom) surface
        # Use much larger radius for gentler curve (tibial plateau is relatively flat)
        tibial_curvature_radius = femoral_curvature_radius * 3.0  # 90mm for gentle curve
        tibial_sphere = m3d.Manifold.sphere(tibial_curvature_radius, resolution)
        tibial_sphere = tibial_sphere.translate([0, 0, -tibial_curvature_radius + height * 0.1])
        annulus = annulus - tibial_sphere

    # Cut to arc (C-shape) by removing a wedge
    if arc_degrees < 360:
        gap_degrees = 360 - arc_degrees
        gap_rad = gap_degrees * np.pi / 180

        # Create cutting box positioned at the gap
        cut_box_size = outer_radius * 3
        cut_box = m3d.Manifold.cube([cut_box_size, cut_box_size, height * 2])

        # Position cutting box to remove the gap section
        # Center the gap at a specific angle (e.g., negative x direction)
        cut_box = cut_box.translate([0, -cut_box_size / 2, -height * 0.5])

        # For a 300-degree arc, we need to cut out 60 degrees (30 on each side of -X axis)
        half_gap = gap_degrees / 2

        # First cut
        cut1 = cut_box.rotate([0, 0, 180 - half_gap])
        annulus = annulus - cut1

        # Second cut
        cut2 = cut_box.rotate([0, 0, 180 + half_gap])
        annulus = annulus - cut2

    return annulus


def create_circumferential_fiber_bundles(
    inner_radius: float,
    outer_radius: float,
    height: float,
    inner_edge_height_ratio: float,
    arc_degrees: float,
    bundle_diameter_um: float,
    bundle_spacing_um: float,
    fiber_density: float,
    resolution: int,
    orientation_variance_deg: float = 15.0
) -> list[m3d.Manifold]:
    """
    Create circumferential fiber bundles following the meniscus arc.

    These are the main load-bearing fibers (48% of meniscal volume).
    Research: 20-50μm bundles, large fascicles up to 480μm.

    Args:
        inner_radius: Inner radius of meniscus
        outer_radius: Outer radius of meniscus
        height: Maximum height at outer edge
        inner_edge_height_ratio: Height ratio at inner edge
        arc_degrees: Arc extent in degrees
        bundle_diameter_um: Fiber bundle diameter in micrometers
        bundle_spacing_um: Spacing between bundles in micrometers
        fiber_density: Fiber density factor (0-1)
        resolution: Angular resolution
        orientation_variance_deg: Random variance in fiber orientation

    Returns:
        List of fiber manifolds
    """
    fibers = []

    bundle_diameter = bundle_diameter_um / 1000.0  # Convert to mm
    bundle_spacing = bundle_spacing_um / 1000.0  # Convert to mm
    bundle_radius = bundle_diameter / 2

    arc_rad = arc_degrees * np.pi / 180
    radial_width = outer_radius - inner_radius

    # Calculate number of radial layers based on spacing
    n_radial_layers = max(3, int(radial_width / bundle_spacing * fiber_density))

    # Calculate number of height layers
    n_height_layers = max(2, int(height / bundle_spacing * fiber_density))

    for r_layer in range(n_radial_layers):
        # Radial position (bias toward outer region where circumferential fibers dominate)
        r_ratio = (r_layer + 0.5) / n_radial_layers
        r = inner_radius + r_ratio * radial_width

        # Height varies with radial position (wedge shape)
        local_height = height * (inner_edge_height_ratio + (1 - inner_edge_height_ratio) * r_ratio)

        for h_layer in range(n_height_layers):
            z = (h_layer + 0.5) * local_height / n_height_layers

            # Add small random variance to orientation
            variance_rad = np.random.uniform(-1, 1) * orientation_variance_deg * np.pi / 180

            # Create arc segments along circumference
            n_segments = max(16, int(arc_rad * r / (bundle_diameter * 2)))

            for seg in range(n_segments):
                angle1 = seg * arc_rad / n_segments + variance_rad
                angle2 = (seg + 1) * arc_rad / n_segments + variance_rad

                x1 = r * np.cos(angle1)
                y1 = r * np.sin(angle1)
                x2 = r * np.cos(angle2)
                y2 = r * np.sin(angle2)

                fiber = make_fiber(
                    np.array([x1, y1, z]),
                    np.array([x2, y2, z]),
                    bundle_radius,
                    max(8, resolution // 4)
                )
                if fiber.num_vert() > 0:
                    fibers.append(fiber)

    return fibers


def create_radial_tie_fibers(
    inner_radius: float,
    outer_radius: float,
    height: float,
    inner_edge_height_ratio: float,
    arc_degrees: float,
    bundle_diameter_um: float,
    bundle_spacing_um: float,
    fiber_density: float,
    resolution: int
) -> list[m3d.Manifold]:
    """
    Create radial tie fibers running from inner to outer edge.

    These fibers prevent splitting under load.
    Research: 10μm sheets, macro-spacing 0.6-1mm (honeycomb), micro-spacing 25-100μm.

    Args:
        inner_radius: Inner radius of meniscus
        outer_radius: Outer radius of meniscus
        height: Maximum height at outer edge
        inner_edge_height_ratio: Height ratio at inner edge
        arc_degrees: Arc extent in degrees
        bundle_diameter_um: Tie fiber diameter in micrometers
        bundle_spacing_um: Spacing between radial fibers in micrometers
        fiber_density: Fiber density factor (0-1)
        resolution: Angular resolution

    Returns:
        List of radial tie fiber manifolds
    """
    fibers = []

    bundle_diameter = bundle_diameter_um / 1000.0  # Convert to mm
    bundle_spacing = bundle_spacing_um / 1000.0  # Convert to mm
    bundle_radius = bundle_diameter / 2

    arc_rad = arc_degrees * np.pi / 180

    # Number of radial fibers around the arc
    circumference = arc_rad * (inner_radius + outer_radius) / 2
    n_angular = max(8, int(circumference / bundle_spacing * fiber_density))

    # Number of height layers
    n_height_layers = max(2, int(height / (bundle_spacing * 2)))

    for ang_idx in range(n_angular):
        angle = ang_idx * arc_rad / n_angular

        for h_layer in range(n_height_layers):
            # Calculate z positions at inner and outer edges (wedge shape)
            inner_max_height = height * inner_edge_height_ratio
            outer_max_height = height

            z_inner = (h_layer + 0.5) * inner_max_height / n_height_layers
            z_outer = (h_layer + 0.5) * outer_max_height / n_height_layers

            x1 = inner_radius * np.cos(angle)
            y1 = inner_radius * np.sin(angle)
            x2 = outer_radius * np.cos(angle)
            y2 = outer_radius * np.sin(angle)

            fiber = make_fiber(
                np.array([x1, y1, z_inner]),
                np.array([x2, y2, z_outer]),
                bundle_radius,
                max(8, resolution // 4)
            )
            if fiber.num_vert() > 0:
                fibers.append(fiber)

    return fibers


def create_lamellar_structure(
    inner_radius: float,
    outer_radius: float,
    height: float,
    inner_edge_height_ratio: float,
    arc_degrees: float,
    lamella_count: int,
    interlaminar_spacing_um: float,
    fiber_diameter: float,
    resolution: int
) -> list[m3d.Manifold]:
    """
    Create lamellar (layered) fiber structure.

    Research: 3 distinct layers - superficial (150-200μm, random fibers),
    lamellar, and central zone.

    Args:
        inner_radius: Inner radius of meniscus
        outer_radius: Outer radius of meniscus
        height: Maximum height at outer edge
        inner_edge_height_ratio: Height ratio at inner edge
        arc_degrees: Arc extent in degrees
        lamella_count: Number of lamella layers
        interlaminar_spacing_um: Spacing between layers in micrometers
        fiber_diameter: Fiber diameter in mm
        resolution: Angular resolution

    Returns:
        List of lamellar fiber manifolds
    """
    fibers = []

    interlaminar_spacing = interlaminar_spacing_um / 1000.0  # Convert to mm
    fiber_radius = fiber_diameter / 2
    arc_rad = arc_degrees * np.pi / 180
    radial_width = outer_radius - inner_radius

    # Create distinct layers with different fiber orientations
    for layer in range(lamella_count):
        # Z position of this layer
        z_ratio = (layer + 0.5) / lamella_count

        # Layer type determines fiber orientation
        if layer < lamella_count // 3:
            # Superficial zone - random/tangential fibers
            orientation = 'random'
        elif layer < 2 * lamella_count // 3:
            # Middle lamellar zone - oblique fibers
            orientation = 'oblique'
        else:
            # Central zone - mixed orientation
            orientation = 'mixed'

        # Create fibers for this layer
        n_fibers_per_layer = max(4, int(arc_rad * (inner_radius + outer_radius) / 2 / (fiber_diameter * 4)))

        for i in range(n_fibers_per_layer):
            # Base angle
            angle = i * arc_rad / n_fibers_per_layer

            # Random radial position
            r = inner_radius + np.random.uniform(0.2, 0.8) * radial_width

            # Z position accounting for wedge
            r_ratio = (r - inner_radius) / radial_width
            local_height = height * (inner_edge_height_ratio + (1 - inner_edge_height_ratio) * r_ratio)
            z = z_ratio * local_height

            # Fiber direction based on orientation type
            if orientation == 'random':
                # Random short fibers in superficial zone
                rand_angle = np.random.uniform(0, 2 * np.pi)
                length = fiber_diameter * 3
                dx = length * np.cos(rand_angle)
                dy = length * np.sin(rand_angle)
                dz = np.random.uniform(-0.5, 0.5) * fiber_diameter
            elif orientation == 'oblique':
                # Oblique fibers (45-degree pattern)
                length = fiber_diameter * 4
                oblique_angle = angle + np.pi / 4
                dx = length * np.cos(oblique_angle)
                dy = length * np.sin(oblique_angle)
                dz = length * 0.3
            else:
                # Mixed - short circumferential segments
                length = fiber_diameter * 3
                dx = length * np.cos(angle + np.pi / 2)
                dy = length * np.sin(angle + np.pi / 2)
                dz = 0

            x = r * np.cos(angle)
            y = r * np.sin(angle)

            fiber = make_fiber(
                np.array([x - dx / 2, y - dy / 2, z - dz / 2]),
                np.array([x + dx / 2, y + dy / 2, z + dz / 2]),
                fiber_radius,
                max(6, resolution // 4)
            )
            if fiber.num_vert() > 0:
                fibers.append(fiber)

    return fibers


def create_vascular_channels(
    inner_radius: float,
    outer_radius: float,
    height: float,
    inner_edge_height_ratio: float,
    arc_degrees: float,
    vascular_penetration_depth: float,
    channel_diameter: float,
    channel_spacing: float,
    resolution: int
) -> list[m3d.Manifold]:
    """
    Create vascular channels in the outer zone ONLY (red zone).

    Research: Vascular in outer 10-30% only. Channel diameter ~100μm, spacing 0.8mm.
    NO channels in inner 65-75% (white zone - avascular).

    Args:
        inner_radius: Inner radius of meniscus
        outer_radius: Outer radius of meniscus
        height: Maximum height at outer edge
        inner_edge_height_ratio: Height ratio at inner edge
        arc_degrees: Arc extent in degrees
        vascular_penetration_depth: Fraction of radial depth for vascular zone (0.1-0.3)
        channel_diameter: Diameter of vascular channels in mm
        channel_spacing: Spacing between channels in mm
        resolution: Angular resolution

    Returns:
        List of vascular channel manifolds (to be subtracted)
    """
    channels = []

    channel_radius = channel_diameter / 2
    arc_rad = arc_degrees * np.pi / 180
    radial_width = outer_radius - inner_radius

    # Vascular zone is only in outer region
    vascular_inner_radius = outer_radius - (radial_width * vascular_penetration_depth)

    # Number of channels around the arc
    vascular_circumference = arc_rad * (vascular_inner_radius + outer_radius) / 2
    n_angular = max(4, int(vascular_circumference / channel_spacing))

    # Number of radial rows of channels
    vascular_width = outer_radius - vascular_inner_radius
    n_radial = max(2, int(vascular_width / channel_spacing))

    for r_idx in range(n_radial):
        r = vascular_inner_radius + (r_idx + 0.5) * vascular_width / n_radial

        # Local height at this radial position
        r_ratio = (r - inner_radius) / radial_width
        local_height = height * (inner_edge_height_ratio + (1 - inner_edge_height_ratio) * r_ratio)

        for ang_idx in range(n_angular):
            angle = ang_idx * arc_rad / n_angular

            x = r * np.cos(angle)
            y = r * np.sin(angle)

            # Create vertical channel through the vascular zone
            channel = m3d.Manifold.cylinder(
                local_height * 1.1,  # Slightly taller to ensure clean subtraction
                channel_radius,
                channel_radius,
                max(8, resolution // 4)
            )
            channel = channel.translate([x, y, -local_height * 0.05])
            channels.append(channel)

    return channels


def create_zone_pores(
    r_inner: float,
    r_outer: float,
    height: float,
    inner_edge_height_ratio: float,
    arc_degrees: float,
    porosity: float,
    resolution: int,
    position_noise: float = 0.1,
    thickness_variance: float = 0.1
) -> list[m3d.Manifold]:
    """
    Create pores for a specific zone with target porosity.

    Research-based porosity gradient:
    - Outer zone: 65% porosity
    - Middle zone: 55% porosity
    - Inner zone: 45% porosity

    Args:
        r_inner: Inner radius of zone
        r_outer: Outer radius of zone
        height: Maximum height of meniscus
        inner_edge_height_ratio: Height ratio at inner edge
        arc_degrees: Arc extent in degrees
        porosity: Target porosity for this zone (0-1)
        resolution: Angular resolution
        position_noise: Random jitter in pore positions
        thickness_variance: Local thickness variation (±10-15% typical)

    Returns:
        List of pore manifolds (to be subtracted)
    """
    pores = []

    arc_rad = arc_degrees * np.pi / 180
    zone_width = r_outer - r_inner

    # Pore size scales with porosity (higher porosity = larger pores)
    pore_radius = 0.05 + porosity * 0.1  # 0.05-0.15mm range

    # Spacing to achieve target porosity (approximate)
    # porosity ~ (pore_volume / total_volume)
    # For spherical pores in a grid: porosity ~ (4/3 * pi * r^3) / (spacing^3)
    spacing = pore_radius * 2 / (porosity ** 0.33) * 1.5

    # Number of pores in each dimension
    n_radial = max(2, int(zone_width / spacing))
    n_height = max(2, int(height / spacing))

    for r_idx in range(n_radial):
        r_base = r_inner + (r_idx + 0.5) * zone_width / n_radial

        # Calculate local height at this radius (accounting for wedge)
        overall_r_ratio = (r_base - r_inner) / (r_outer - r_inner) if r_outer > r_inner else 0.5
        radial_ratio = (r_inner + overall_r_ratio * zone_width - r_inner) / max(0.001, r_outer - r_inner)
        base_height = height * (inner_edge_height_ratio + (1 - inner_edge_height_ratio) * overall_r_ratio)

        # Apply thickness variance to local height
        thickness_factor = 1.0 + np.random.uniform(-thickness_variance, thickness_variance)
        local_max_height = base_height * thickness_factor

        # Circumference at this radius
        circumference = arc_rad * r_base
        n_angular = max(4, int(circumference / spacing))

        for h_idx in range(n_height):
            z_base = (h_idx + 0.5) * local_max_height / n_height

            for ang_idx in range(n_angular):
                angle = ang_idx * arc_rad / n_angular

                # Add position noise
                r = r_base + np.random.uniform(-1, 1) * position_noise * zone_width / n_radial
                z = z_base + np.random.uniform(-1, 1) * position_noise * local_max_height / n_height
                angle += np.random.uniform(-1, 1) * position_noise * arc_rad / n_angular

                # Clamp to valid range
                r = max(r_inner, min(r_outer, r))
                z = max(0, min(local_max_height, z))

                x = r * np.cos(angle)
                y = r * np.sin(angle)

                # Vary pore size slightly
                actual_pore_radius = pore_radius * np.random.uniform(0.8, 1.2)

                pore = m3d.Manifold.sphere(actual_pore_radius, max(6, resolution // 4))
                pore = pore.translate([x, y, z])
                pores.append(pore)

    return pores


def create_cell_lacunae(
    inner_radius: float,
    outer_radius: float,
    height: float,
    inner_edge_height_ratio: float,
    arc_degrees: float,
    lacuna_density: float,
    lacuna_diameter_um: float,
    vascular_penetration_depth: float,
    resolution: int
) -> list[m3d.Manifold]:
    """
    Create cell lacunae (chondrocyte spaces) distributed by zone.

    Research:
    - Vascular region: 27,199 cells/mm³
    - Avascular region: 12,820 cells/mm³
    - Lacuna diameter: ~15μm

    Args:
        inner_radius: Inner radius of meniscus
        outer_radius: Outer radius of meniscus
        height: Maximum height at outer edge
        inner_edge_height_ratio: Height ratio at inner edge
        arc_degrees: Arc extent in degrees
        lacuna_density: Lacunae per mm³ (scaffold representation, scaled down from biological)
        lacuna_diameter_um: Lacuna diameter in micrometers
        vascular_penetration_depth: Fraction defining vascular zone boundary
        resolution: Angular resolution

    Returns:
        List of lacuna manifolds (to be subtracted)
    """
    lacunae = []

    lacuna_radius = (lacuna_diameter_um / 1000.0) / 2  # Convert to mm
    arc_rad = arc_degrees * np.pi / 180
    radial_width = outer_radius - inner_radius

    # Vascular zone boundary
    vascular_inner_radius = outer_radius - (radial_width * vascular_penetration_depth)

    # Approximate volume of meniscus for density calculation
    avg_radius = (inner_radius + outer_radius) / 2
    avg_height = height * (1 + inner_edge_height_ratio) / 2
    approx_volume = arc_rad * (outer_radius**2 - inner_radius**2) / 2 * avg_height

    # Calculate total lacunae to create (scaled representation)
    total_lacunae = int(approx_volume * lacuna_density)
    total_lacunae = min(total_lacunae, 2000)  # Cap for performance

    # Distribute lacunae with higher density in vascular zone
    vascular_volume_ratio = vascular_penetration_depth
    vascular_lacuna_count = int(total_lacunae * vascular_volume_ratio * 1.5)  # Higher density
    avascular_lacuna_count = total_lacunae - vascular_lacuna_count

    # Create vascular zone lacunae
    for _ in range(vascular_lacuna_count):
        r = np.random.uniform(vascular_inner_radius, outer_radius)
        angle = np.random.uniform(0, arc_rad)

        r_ratio = (r - inner_radius) / radial_width
        local_height = height * (inner_edge_height_ratio + (1 - inner_edge_height_ratio) * r_ratio)
        z = np.random.uniform(0, local_height)

        x = r * np.cos(angle)
        y = r * np.sin(angle)

        lacuna = m3d.Manifold.sphere(lacuna_radius, max(4, resolution // 6))
        lacuna = lacuna.translate([x, y, z])
        lacunae.append(lacuna)

    # Create avascular zone lacunae
    for _ in range(avascular_lacuna_count):
        r = np.random.uniform(inner_radius, vascular_inner_radius)
        angle = np.random.uniform(0, arc_rad)

        r_ratio = (r - inner_radius) / radial_width
        local_height = height * (inner_edge_height_ratio + (1 - inner_edge_height_ratio) * r_ratio)
        z = np.random.uniform(0, local_height)

        x = r * np.cos(angle)
        y = r * np.sin(angle)

        lacuna = m3d.Manifold.sphere(lacuna_radius, max(4, resolution // 6))
        lacuna = lacuna.translate([x, y, z])
        lacunae.append(lacuna)

    return lacunae


def create_surface_roughness(
    inner_radius: float,
    outer_radius: float,
    height: float,
    inner_edge_height_ratio: float,
    arc_degrees: float,
    roughness_scale: float,
    resolution: int
) -> list[m3d.Manifold]:
    """
    Create surface roughness texture for articular surfaces.

    Research: Meniscus articular surfaces have micro-texture for lubrication.
    Roughness ~5-50μm scale.

    Args:
        inner_radius: Inner radius of meniscus
        outer_radius: Outer radius of meniscus
        height: Maximum height at outer edge
        inner_edge_height_ratio: Height ratio at inner edge
        arc_degrees: Arc extent in degrees
        roughness_scale: Surface roughness parameter (0.05 typical)
        resolution: Angular resolution

    Returns:
        List of small bump manifolds for surface texture
    """
    bumps = []

    arc_rad = arc_degrees * np.pi / 180
    radial_width = outer_radius - inner_radius

    # Bump amplitude scales with roughness parameter
    bump_amplitude = roughness_scale * 0.1  # Convert to mm (0.05 → 0.005mm = 5μm)
    bump_radius = bump_amplitude * 1.5  # Slightly larger radius

    # Spacing between bumps (~3x bump size)
    bump_spacing = bump_radius * 6

    # Number of bumps in each dimension
    n_radial = max(3, int(radial_width / bump_spacing))
    n_height = max(3, int(height / bump_spacing))

    for r_idx in range(n_radial):
        r = inner_radius + (r_idx + 0.5) * radial_width / n_radial

        # Local height at this radius
        r_ratio = (r - inner_radius) / radial_width
        local_height = height * (inner_edge_height_ratio + (1 - inner_edge_height_ratio) * r_ratio)

        # Circumference at this radius
        circumference = arc_rad * r
        n_angular = max(4, int(circumference / bump_spacing))

        for h_idx in range(n_height):
            z = (h_idx + 0.5) * local_height / n_height

            for ang_idx in range(n_angular):
                angle = ang_idx * arc_rad / n_angular

                x = r * np.cos(angle)
                y = r * np.sin(angle)

                # Create small spherical bump
                bump = m3d.Manifold.sphere(bump_radius, max(4, resolution // 6))
                bump = bump.translate([x, y, z])
                bumps.append(bump)

    return bumps


def create_horn_attachments(
    inner_radius: float,
    outer_radius: float,
    height: float,
    arc_degrees: float,
    horn_width: float,
    horn_depth: float,
    enable_anterior: bool,
    enable_posterior: bool,
    resolution: int
) -> list[m3d.Manifold]:
    """
    Create horn attachment extensions at the ends of the C-shaped meniscus.

    Research:
    - Anterior horn insertion: ~61mm² area
    - Posterior horn insertion: ~30mm² area

    Args:
        inner_radius: Inner radius of meniscus
        outer_radius: Outer radius of meniscus
        height: Height of meniscus
        arc_degrees: Arc extent in degrees
        horn_width: Width of horn insertion (mm)
        horn_depth: Depth of horn attachment (mm)
        enable_anterior: Create anterior horn
        enable_posterior: Create posterior horn
        resolution: Angular resolution

    Returns:
        List of horn manifolds
    """
    horns = []
    arc_rad = arc_degrees * np.pi / 180

    # Gap angle (where the C-shape is open)
    gap_start_angle = arc_rad
    gap_center_angle = arc_rad + (2 * np.pi - arc_rad) / 2

    # Horn dimensions
    horn_height = height * 0.6  # Horns are shorter than main body

    if enable_anterior:
        # Anterior horn at one end of the arc
        angle = 0
        mid_radius = (inner_radius + outer_radius) / 2

        # Create a protruding cylinder/box for horn attachment
        horn = m3d.Manifold.cylinder(horn_height, horn_width / 2, horn_width / 2, resolution)
        horn = horn.scale([1, horn_depth / horn_width, 1])

        # Position at arc start
        x = mid_radius * np.cos(angle)
        y = mid_radius * np.sin(angle)

        # Rotate to point outward from arc end
        horn = horn.rotate([0, 0, angle * 180 / np.pi - 90])
        horn = horn.translate([x - horn_depth * 0.5, y, 0])

        horns.append(horn)

    if enable_posterior:
        # Posterior horn at other end of the arc
        angle = arc_rad
        mid_radius = (inner_radius + outer_radius) / 2

        # Create horn (smaller than anterior)
        posterior_width = horn_width * 0.8  # Posterior is smaller
        posterior_depth = horn_depth * 0.8

        horn = m3d.Manifold.cylinder(horn_height, posterior_width / 2, posterior_width / 2, resolution)
        horn = horn.scale([1, posterior_depth / posterior_width, 1])

        # Position at arc end
        x = mid_radius * np.cos(angle)
        y = mid_radius * np.sin(angle)

        # Rotate to point outward
        horn = horn.rotate([0, 0, angle * 180 / np.pi + 90])
        horn = horn.translate([x + posterior_depth * 0.5, y, 0])

        horns.append(horn)

    return horns


def make_fiber(p1: np.ndarray, p2: np.ndarray, radius: float, resolution: int) -> m3d.Manifold:
    """
    Create a cylindrical fiber between two points.

    Args:
        p1: Starting point (x, y, z)
        p2: Ending point (x, y, z)
        radius: Fiber radius
        resolution: Number of segments around cylinder

    Returns:
        Manifold representing the fiber cylinder
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    length = np.sqrt(dx*dx + dy*dy + dz*dz)

    if length < 1e-6:
        return m3d.Manifold()

    # Create cylinder along Z axis
    fiber = m3d.Manifold.cylinder(length, radius, radius, resolution)

    # Calculate rotation to align with direction vector
    h = np.sqrt(dx*dx + dy*dy)
    if h > 0.001 or abs(dz) > 0.001:
        angle_y = np.arctan2(h, dz) * 180 / np.pi
        angle_z = np.arctan2(dy, dx) * 180 / np.pi

        fiber = fiber.rotate([0, angle_y, 0])
        fiber = fiber.rotate([0, 0, angle_z])

    return fiber.translate([p1[0], p1[1], p1[2]])


def generate_meniscus_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate meniscus from dictionary parameters (for API compatibility).

    Args:
        params: Dictionary with keys matching MeniscusParams fields

    Returns:
        Tuple of (manifold, stats_dict)
    """
    return generate_meniscus(MeniscusParams(
        # Basic geometry
        outer_radius=params.get('outer_radius', 22.5),
        inner_radius=params.get('inner_radius', 10.0),
        outer_diameter=params.get('outer_diameter', 45.0),
        inner_diameter=params.get('inner_diameter', 20.0),
        thickness=params.get('thickness', 8.0),
        height=params.get('height', 8.0),
        arc_degrees=params.get('arc_degrees', 300.0),
        resolution=params.get('resolution', 32),
        seed=params.get('seed', 42),

        # Wedge geometry
        wedge_angle_deg=params.get('wedge_angle_deg', 20.0),
        inner_edge_height_ratio=params.get('inner_edge_height_ratio', 0.3),

        # Radial zones
        zone_count=params.get('zone_count', 3),
        outer_zone_thickness_ratio=params.get('outer_zone_thickness_ratio', 0.33),
        middle_zone_thickness_ratio=params.get('middle_zone_thickness_ratio', 0.33),
        inner_zone_thickness_ratio=params.get('inner_zone_thickness_ratio', 0.34),

        # Circumferential fibers
        circumferential_bundle_diameter_um=params.get('circumferential_bundle_diameter_um', 100.0),
        circumferential_bundle_spacing_um=params.get('circumferential_bundle_spacing_um', 200.0),
        circumferential_fiber_density=params.get('circumferential_fiber_density', 0.8),

        # Radial fibers
        radial_bundle_diameter_um=params.get('radial_bundle_diameter_um', 20.0),
        radial_bundle_spacing_um=params.get('radial_bundle_spacing_um', 100.0),
        radial_fiber_density=params.get('radial_fiber_density', 0.6),
        enable_radial_tie_fibers=params.get('enable_radial_tie_fibers', True),

        # General fiber properties
        fiber_diameter=params.get('fiber_diameter', 0.2),
        fiber_orientation_variance_deg=params.get('fiber_orientation_variance_deg', 15.0),

        # Porosity
        outer_zone_porosity=params.get('outer_zone_porosity', 0.65),
        middle_zone_porosity=params.get('middle_zone_porosity', 0.55),
        inner_zone_porosity=params.get('inner_zone_porosity', 0.45),
        porosity=params.get('porosity', 0.61),

        # Vascular features
        enable_vascular_channels=params.get('enable_vascular_channels', True),
        vascular_channel_diameter=params.get('vascular_channel_diameter', 0.1),
        vascular_channel_spacing=params.get('vascular_channel_spacing', 0.8),
        vascular_penetration_depth=params.get('vascular_penetration_depth', 0.33),

        # Horn attachments
        enable_anterior_horn=params.get('enable_anterior_horn', False),
        enable_posterior_horn=params.get('enable_posterior_horn', False),
        horn_insertion_width=params.get('horn_insertion_width', 5.0),
        horn_insertion_depth=params.get('horn_insertion_depth', 3.0),

        # Surface properties
        surface_roughness=params.get('surface_roughness', 0.05),
        enable_femoral_surface=params.get('enable_femoral_surface', True),
        enable_tibial_surface=params.get('enable_tibial_surface', True),
        femoral_curvature_radius=params.get('femoral_curvature_radius', 30.0),

        # Collagen organization
        enable_lamellar_structure=params.get('enable_lamellar_structure', True),
        lamella_count=params.get('lamella_count', 6),
        interlaminar_spacing_um=params.get('interlaminar_spacing_um', 50.0),

        # Cell lacunae
        enable_cell_lacunae=params.get('enable_cell_lacunae', False),
        lacuna_density=params.get('lacuna_density', 500.0),
        lacuna_diameter_um=params.get('lacuna_diameter_um', 15.0),

        # Randomization
        position_noise=params.get('position_noise', 0.1),
        thickness_variance=params.get('thickness_variance', 0.1),
    ))
