"""
Multilayer skin scaffold generator with epidermis, dermis, and hypodermis.

Provides parametric generation of skin tissue scaffolds with:
- Epidermis layer (0.1-0.2mm thickness)
- Dermis layer (1-2mm thickness) with higher porosity
- Hypodermis layer (2-5mm thickness) with highest porosity
- Optional vertical vascular channels
- Gradient porosity from outer to inner layers
"""

from __future__ import annotations
import manifold3d as m3d
import numpy as np
from dataclasses import dataclass
from ..core import batch_union


def apply_position_noise(
    x: float,
    y: float,
    noise_factor: float,
    spacing: float,
    rng: np.random.Generator
) -> tuple[float, float]:
    """
    Apply random jitter to x, y coordinates for organic appearance.

    Args:
        x: Original x coordinate
        y: Original y coordinate
        noise_factor: Noise magnitude as fraction of spacing (e.g., 0.15 = ±15%)
        spacing: Base spacing between features
        rng: Random number generator

    Returns:
        Tuple of (x_jittered, y_jittered)
    """
    if noise_factor <= 0:
        return x, y

    offset_magnitude = noise_factor * spacing
    dx = rng.uniform(-offset_magnitude, offset_magnitude)
    dy = rng.uniform(-offset_magnitude, offset_magnitude)

    return x + dx, y + dy


@dataclass
class MultilayerSkinParams:
    """
    Parameters for multilayer skin scaffold generation.

    Biologically realistic parameters based on human skin histology:
    - Epidermis: 50-150 μm thick, contains rete ridges
    - Dermis: 1-4 mm thick, divided into papillary and reticular layers
    - Hypodermis: 1-50 mm thick depending on body location
    """
    # === Basic Geometry ===
    diameter_mm: float = 10.0  # Overall scaffold diameter

    # === Layer Thicknesses (biologically accurate) ===
    epidermis_thickness_um: float = 100.0  # 50-150 μm typical; Varies by body site: 30µm (eyelid) to 600µm (palm/sole)
    dermis_thickness_mm: float = 1.5  # 1-4 mm typical
    papillary_dermis_thickness_um: float = 200.0  # ~20% of dermis, loose connective tissue
    reticular_dermis_thickness_mm: float = 1.3  # ~80% of dermis, dense connective tissue
    hypodermis_thickness_mm: float = 10.0  # 1-50 mm depending on location
    keratinocyte_layers: int = 5  # Number of keratinocyte cell layers in epidermis (typically 4-6). REFERENCE ONLY: Cell-level detail not modeled at scaffold scale (~30μm cells)

    # === Rete Ridges (dermal-epidermal junction) ===
    enable_rete_ridges: bool = True
    rete_ridge_height_um: float = 51.0  # 35-65 μm typical
    rete_ridge_width_um: float = 150.0  # 100-200 μm
    rete_ridge_spacing_um: float = 105.0  # 70-140 μm center-to-center
    rete_ridge_depth_variance: float = 0.2  # Random variation

    # === Dermal Papillae ===
    enable_dermal_papillae: bool = True
    papillae_density_per_mm2: float = 100.0  # 50-200 per mm²
    papillae_height_um: float = 100.0  # 50-200 μm typical
    papillae_diameter_um: float = 75.0  # 50-150 μm typical

    # === Appendage Densities ===
    enable_hair_follicles: bool = False
    hair_follicle_density_per_cm2: float = 130.0  # 100-150/cm² on scalp, 50-70 elsewhere
    hair_follicle_diameter_um: float = 70.0  # 50-100 μm
    hair_follicle_depth_mm: float = 2.5  # Extends into dermis/hypodermis

    enable_sweat_glands: bool = False
    sweat_gland_density_per_cm2: float = 155.0  # 100-200/cm² (eccrine)
    sweat_gland_diameter_um: float = 40.0  # Coiled tubular structure
    sweat_gland_depth_mm: float = 1.5  # In dermis

    # === Sebaceous Glands ===
    enable_sebaceous_glands: bool = False
    sebaceous_gland_density_per_cm2: float = 100.0  # Associated with hair follicles
    sebaceous_gland_diameter_um: float = 100.0  # 50-150 μm typical diameter

    # === Dermal Features ===
    dermal_porosity: float = 0.19  # 0.15-0.25 typical
    pore_diameter_um: float = 219.0  # 150-300 μm for cell migration
    pore_interconnectivity: float = 0.8  # 0-1 scale, connectivity between pores

    # === Pore Gradient (porosity by layer) ===
    pore_gradient: tuple[float, float, float] = (0.15, 0.25, 0.40)  # epidermis, dermis, hypodermis

    # === Vascular Channels ===
    enable_vascular_channels: bool = True
    vascular_channel_diameter_mm: float = 0.15  # 100-200 μm
    vascular_channel_count: int = 4
    vascular_channel_spacing_mm: float = 2.0

    # === Collagen Architecture ===
    enable_collagen_orientation: bool = False
    papillary_collagen_diameter_um: float = 2.0  # Fine fibers
    reticular_collagen_diameter_um: float = 10.0  # Coarse fibers
    collagen_alignment: float = 0.3  # 0=random, 1=fully aligned (Langer lines)

    # === Surface Features ===
    enable_surface_texture: bool = False
    surface_roughness: float = 0.1  # 0-1 scale

    # === Randomization ===
    seed: int = 42
    randomness: float = 0.1  # 0-1 scale for stochastic variation
    position_noise: float = 0.15  # Positional variation for organic appearance

    # === Resolution ===
    resolution: int = 16


def make_rete_ridge_layer(
    diameter: float,
    base_thickness: float,
    z_offset: float,
    ridge_height: float,
    ridge_spacing: float,
    ridge_width: float,
    depth_variance: float,
    porosity: float,
    pore_size: float,
    resolution: int,
    rng: np.random.Generator,
    is_dermis: bool = False,
    position_noise: float = 0.0,
    pore_interconnectivity: float = 0.0
) -> m3d.Manifold:
    """
    Create a layer with rete ridges at the interface.

    For simplicity, use cylinder-based geometry instead of custom mesh.
    The rete ridges are represented by the porous structure rather than
    explicit surface undulations.

    Args:
        diameter: Outer diameter of layer
        base_thickness: Base layer thickness
        z_offset: Z position of layer bottom
        ridge_height: Amplitude of ridges (used for thickness adjustment)
        ridge_spacing: Spacing of ridges (not used in simplified version)
        ridge_width: Width of ridges (not used in simplified version)
        depth_variance: Random variation (not used in simplified version)
        porosity: Fraction of volume to be porous (0-1)
        pore_size: Size of individual pores
        resolution: Circular resolution
        rng: Random number generator
        is_dermis: If True, creates dermis; if False, epidermis
        position_noise: Positional jitter for pores as fraction of spacing
        pore_interconnectivity: Fraction of pore pairs connected by channels (0-1)

    Returns:
        Manifold representing the layer
    """
    radius = diameter / 2

    # Use simple cylinder for base geometry
    # Adjust thickness slightly to account for ridge effect
    effective_thickness = base_thickness + ridge_height * 0.5

    result = m3d.Manifold.cylinder(effective_thickness, radius, radius, resolution)
    result = result.translate([0, 0, z_offset])

    # Add porosity if needed
    if porosity > 0.01:
        pore_radius = pore_size / 2
        spacing = pore_size / np.sqrt(porosity)
        pores = []
        pore_positions = []  # Track positions for interconnectivity
        n_pores = max(1, int(diameter / spacing))

        for i in range(n_pores):
            for jp in range(n_pores):
                x = (i - n_pores / 2 + 0.5) * spacing
                y = (jp - n_pores / 2 + 0.5) * spacing

                # Apply position noise if enabled
                if position_noise > 0:
                    x, y = apply_position_noise(x, y, position_noise, spacing, rng)

                if np.sqrt(x * x + y * y) < radius - pore_radius:
                    pore_height = base_thickness + ridge_height * 2
                    pore = m3d.Manifold.cylinder(
                        pore_height * 1.2,
                        pore_radius,
                        pore_radius,
                        resolution // 2
                    )
                    pore = pore.translate([x, y, z_offset - ridge_height - pore_height * 0.1])
                    pores.append(pore)
                    pore_positions.append((i, jp, x, y))

        # Add interconnecting channels between adjacent pores
        if pore_interconnectivity > 0.01 and len(pore_positions) > 1:
            channel_radius = pore_radius * 0.5  # Smaller connecting passages
            channel_height = base_thickness + ridge_height * 2
            z_center = z_offset + base_thickness / 2

            # Create channels between horizontal neighbors
            for i, jp, x1, y1 in pore_positions:
                # Check right neighbor (i+1, jp)
                for i2, jp2, x2, y2 in pore_positions:
                    if (i2 == i + 1 and jp2 == jp) or (i2 == i and jp2 == jp + 1):
                        # Create connection with probability = pore_interconnectivity
                        if rng.random() < pore_interconnectivity:
                            # Calculate channel length and orientation
                            dx = x2 - x1
                            dy = y2 - y1
                            length = np.sqrt(dx * dx + dy * dy)

                            if length > 0.001:
                                # Create horizontal channel
                                channel = m3d.Manifold.cylinder(
                                    length,
                                    channel_radius,
                                    channel_radius,
                                    max(4, resolution // 4)
                                )

                                # Rotate to align with connection direction
                                angle = np.arctan2(dy, dx) * 180 / np.pi
                                channel = channel.rotate([0, 90, 0])  # Rotate to horizontal
                                channel = channel.rotate([0, 0, angle])  # Rotate to connection angle

                                # Position at midpoint between pores
                                mid_x = (x1 + x2) / 2
                                mid_y = (y1 + y2) / 2
                                channel = channel.translate([mid_x, mid_y, z_center])
                                pores.append(channel)

        if pores:
            pores_union = batch_union(pores)
            result = result - pores_union

    return result


def make_porous_layer(
    diameter: float,
    thickness: float,
    z_offset: float,
    porosity: float,
    pore_size: float,
    resolution: int,
    position_noise: float = 0.0,
    rng: np.random.Generator | None = None,
    pore_interconnectivity: float = 0.0
) -> m3d.Manifold:
    """
    Create a single porous disc layer.

    Args:
        diameter: Outer diameter of layer
        thickness: Layer thickness
        z_offset: Z position of layer bottom
        porosity: Fraction of volume to be porous (0-1)
        pore_size: Size of individual pores
        resolution: Circular resolution
        position_noise: Positional jitter as fraction of spacing (default 0.0)
        rng: Random number generator (required if position_noise > 0)
        pore_interconnectivity: Fraction of pore pairs connected by channels (0-1)

    Returns:
        Manifold representing the porous layer
    """
    # Create base disc
    radius = diameter / 2
    base = m3d.Manifold.cylinder(thickness, radius, radius, resolution)
    base = base.translate([0, 0, z_offset])

    if porosity <= 0.01:
        return base

    # Calculate pore grid spacing from porosity
    # Assuming circular pores in square grid
    pore_radius = pore_size / 2
    spacing = pore_size / np.sqrt(porosity)

    # Create pore grid
    pores = []
    pore_positions = []  # Track positions for interconnectivity
    n_pores = max(1, int(diameter / spacing))

    for i in range(n_pores):
        for j in range(n_pores):
            x = (i - n_pores / 2 + 0.5) * spacing
            y = (j - n_pores / 2 + 0.5) * spacing

            # Apply position noise if enabled
            if position_noise > 0 and rng is not None:
                x, y = apply_position_noise(x, y, position_noise, spacing, rng)

            # Check if pore is within circular boundary
            if np.sqrt(x*x + y*y) < radius - pore_radius:
                pore = m3d.Manifold.cylinder(
                    thickness * 1.1,  # Slightly taller to ensure clean subtraction
                    pore_radius,
                    pore_radius,
                    resolution // 2
                )
                pore = pore.translate([x, y, z_offset - thickness * 0.05])
                pores.append(pore)
                pore_positions.append((i, j, x, y))

    # Add interconnecting channels between adjacent pores
    if pore_interconnectivity > 0.01 and len(pore_positions) > 1 and rng is not None:
        channel_radius = pore_radius * 0.5  # Smaller connecting passages
        z_center = z_offset + thickness / 2

        # Create channels between horizontal neighbors
        for i, j, x1, y1 in pore_positions:
            # Check right neighbor (i+1, j) and bottom neighbor (i, j+1)
            for i2, j2, x2, y2 in pore_positions:
                if (i2 == i + 1 and j2 == j) or (i2 == i and j2 == j + 1):
                    # Create connection with probability = pore_interconnectivity
                    if rng.random() < pore_interconnectivity:
                        # Calculate channel length and orientation
                        dx = x2 - x1
                        dy = y2 - y1
                        length = np.sqrt(dx * dx + dy * dy)

                        if length > 0.001:
                            # Create horizontal channel
                            channel = m3d.Manifold.cylinder(
                                length,
                                channel_radius,
                                channel_radius,
                                max(4, resolution // 4)
                            )

                            # Rotate to align with connection direction
                            angle = np.arctan2(dy, dx) * 180 / np.pi
                            channel = channel.rotate([0, 90, 0])  # Rotate to horizontal
                            channel = channel.rotate([0, 0, angle])  # Rotate to connection angle

                            # Position at midpoint between pores
                            mid_x = (x1 + x2) / 2
                            mid_y = (y1 + y2) / 2
                            channel = channel.translate([mid_x, mid_y, z_center])
                            pores.append(channel)

    if not pores:
        return base

    # Subtract pores from base
    pores_union = batch_union(pores)
    return base - pores_union


def make_vascular_channel(
    diameter: float,
    total_thickness: float,
    channel_diameter: float,
    x_pos: float,
    y_pos: float,
    resolution: int
) -> m3d.Manifold:
    """
    Create a vertical vascular channel through all layers.

    Args:
        diameter: Outer diameter of scaffold
        total_thickness: Total height of all layers
        channel_diameter: Diameter of vascular channel
        x_pos: X position of channel
        y_pos: Y position of channel
        resolution: Circular resolution

    Returns:
        Manifold representing the channel
    """
    channel_radius = channel_diameter / 2
    radius = diameter / 2

    # Check if channel is within boundary
    if np.sqrt(x_pos*x_pos + y_pos*y_pos) > radius - channel_radius:
        return m3d.Manifold()

    channel = m3d.Manifold.cylinder(
        total_thickness * 1.1,
        channel_radius,
        channel_radius,
        resolution // 2
    )
    return channel.translate([x_pos, y_pos, -total_thickness * 0.05])


def make_hair_follicles(
    diameter: float,
    follicle_diameter: float,
    follicle_depth: float,
    density_per_cm2: float,
    top_z: float,
    resolution: int,
    rng: np.random.Generator
) -> list[m3d.Manifold]:
    """
    Create angled hair follicle channels.

    Hair follicles are angled 15-30 degrees from vertical and extend
    through the dermis into the hypodermis.

    Args:
        diameter: Scaffold diameter in mm
        follicle_diameter: Follicle diameter in mm
        follicle_depth: Depth of follicle in mm
        density_per_cm2: Number of follicles per cm²
        top_z: Z coordinate of skin surface
        resolution: Circular resolution
        rng: Random number generator

    Returns:
        List of follicle channel manifolds for Boolean subtraction
    """
    follicles = []
    radius = diameter / 2
    follicle_radius = follicle_diameter / 2

    # Calculate area in cm² and number of follicles
    area_mm2 = np.pi * radius * radius
    area_cm2 = area_mm2 / 100.0  # 1 cm² = 100 mm²
    n_follicles = max(1, int(density_per_cm2 * area_cm2))

    # Distribute follicles randomly within the circular area
    for _ in range(n_follicles):
        # Random position within circle using rejection sampling
        while True:
            x = rng.uniform(-radius + follicle_radius, radius - follicle_radius)
            y = rng.uniform(-radius + follicle_radius, radius - follicle_radius)
            if np.sqrt(x * x + y * y) < radius - follicle_radius:
                break

        # Random angle from vertical (15-30 degrees)
        tilt_angle = rng.uniform(15, 30) * np.pi / 180

        # Random azimuthal direction
        azimuth = rng.uniform(0, 2 * np.pi)

        # Create follicle as angled cylinder
        # Use slightly longer cylinder to ensure clean subtraction
        follicle_length = follicle_depth / np.cos(tilt_angle) * 1.1

        follicle = m3d.Manifold.cylinder(
            follicle_length,
            follicle_radius,
            follicle_radius,
            max(8, resolution // 2)
        )

        # Rotate to tilt angle
        # First rotate around X axis to tilt
        follicle = follicle.rotate([tilt_angle * 180 / np.pi, 0, 0])
        # Then rotate around Z axis for azimuthal direction
        follicle = follicle.rotate([0, 0, azimuth * 180 / np.pi])

        # Position at skin surface, pointing downward
        # Start slightly above surface to ensure clean cut
        follicle = follicle.translate([x, y, top_z + 0.05])
        # Rotate 180 degrees to point downward
        follicle = follicle.rotate([180, 0, 0])
        follicle = follicle.translate([0, 0, top_z + 0.05])

        # Simpler approach: create cylinder pointing down
        follicle = m3d.Manifold.cylinder(
            follicle_length,
            follicle_radius,
            follicle_radius,
            max(8, resolution // 2)
        )

        # Calculate end position with tilt
        dx = follicle_depth * np.sin(tilt_angle) * np.cos(azimuth)
        dy = follicle_depth * np.sin(tilt_angle) * np.sin(azimuth)
        dz = -follicle_depth  # Going down

        # Position the cylinder
        # We'll create a simple tilted cylinder by translation
        follicle = follicle.translate([x, y, top_z - follicle_depth])

        # For proper tilted cylinder, we need to rotate appropriately
        follicle = m3d.Manifold.cylinder(
            follicle_length,
            follicle_radius,
            follicle_radius * 0.9,  # Slight taper
            max(8, resolution // 2)
        )

        # Rotate: first tilt, then azimuth
        follicle = follicle.rotate([tilt_angle * 180 / np.pi, 0, azimuth * 180 / np.pi])

        # Position so top is at surface
        follicle = follicle.translate([x, y, top_z - follicle_depth / 2])

        follicles.append(follicle)

    return follicles




def make_sebaceous_glands(
    diameter: float,
    density_per_cm2: float,
    dermis_z_start: float,
    dermis_thickness: float,
    gland_diameter_um: float,
    resolution: int,
    rng: np.random.Generator
) -> list[m3d.Manifold]:
    """
    Create sebaceous gland structures in mid-dermis.

    Sebaceous glands are lobular structures that produce sebum.
    They're located in the mid-dermis (50-70% depth) and are associated
    with hair follicles. Each gland consists of 2-4 overlapping lobules.

    Args:
        diameter: Scaffold diameter in mm
        density_per_cm2: Number of glands per cm²
        dermis_z_start: Z coordinate where dermis begins
        dermis_thickness: Thickness of dermis layer in mm
        gland_diameter_um: Diameter of gland lobules in μm (50-150 typical)
        resolution: Circular resolution
        rng: Random number generator

    Returns:
        List of sebaceous gland manifolds for Boolean subtraction
    """
    glands = []
    radius = diameter / 2

    # Convert μm to mm
    gland_diameter_mm = gland_diameter_um / 1000.0
    gland_radius = gland_diameter_mm / 2

    # Position glands in mid-dermis (50-70% depth)
    mid_dermis_start = dermis_z_start + dermis_thickness * 0.5
    mid_dermis_end = dermis_z_start + dermis_thickness * 0.7
    mid_dermis_range = mid_dermis_end - mid_dermis_start

    # Calculate area and number of glands
    area_mm2 = np.pi * radius * radius
    area_cm2 = area_mm2 / 100.0
    n_glands = max(1, int(density_per_cm2 * area_cm2))

    # Distribute glands randomly within the circular area
    for _ in range(n_glands):
        # Random position within circle using rejection sampling
        while True:
            x = rng.uniform(-radius + gland_radius, radius - gland_radius)
            y = rng.uniform(-radius + gland_radius, radius - gland_radius)
            if np.sqrt(x * x + y * y) < radius - gland_radius:
                break

        # Random z position in mid-dermis
        z = mid_dermis_start + rng.uniform(0, mid_dermis_range)

        # Random size variation (±20%)
        size_variation = rng.uniform(0.8, 1.2)
        lobule_radius = gland_radius * size_variation

        # Create gland as 2-4 overlapping spheres (lobules)
        n_lobules = rng.choice([2, 3, 4])
        gland_parts = []

        for i in range(n_lobules):
            # Arrange lobules in a cluster
            if n_lobules == 2:
                offset_x = lobule_radius * 0.4 * (i - 0.5)
                offset_y = 0
            elif n_lobules == 3:
                angle = i * 2 * np.pi / 3
                offset_x = lobule_radius * 0.4 * np.cos(angle)
                offset_y = lobule_radius * 0.4 * np.sin(angle)
            else:  # 4 lobules
                angle = i * 2 * np.pi / 4
                offset_x = lobule_radius * 0.4 * np.cos(angle)
                offset_y = lobule_radius * 0.4 * np.sin(angle)

            lobule = m3d.Manifold.sphere(
                lobule_radius,
                max(6, resolution // 2)
            )
            lobule = lobule.translate([x + offset_x, y + offset_y, z])
            gland_parts.append(lobule)

        # Create short duct (small cylinder leading upward)
        duct_length = dermis_thickness * 0.2  # Duct extends 20% toward surface
        duct_radius = lobule_radius * 0.3  # Narrow duct

        duct = m3d.Manifold.cylinder(
            duct_length,
            duct_radius,
            duct_radius,
            max(4, resolution // 4)
        )
        duct = duct.translate([x, y, z])
        gland_parts.append(duct)

        # Union all parts to create complete gland
        if gland_parts:
            gland = batch_union(gland_parts)
            glands.append(gland)

    return glands



def make_dermal_papillae(
    diameter: float,
    density_per_mm2: float,
    dermis_top_z: float,
    papillae_height_um: float,
    papillae_diameter_um: float,
    resolution: int,
    rng: np.random.Generator
) -> list[m3d.Manifold]:
    """
    Create dermal papillae structures projecting upward from dermis into epidermis.

    Dermal papillae are small, nipple-like projections that contain capillary loops
    and anchor the dermal-epidermal junction. They increase surface area for
    nutrient exchange between dermis and epidermis.

    Args:
        diameter: Scaffold diameter in mm
        density_per_mm2: Number of papillae per mm²
        dermis_top_z: Z coordinate where dermis meets epidermis
        papillae_height_um: Height of papillae in μm (50-200 typical)
        papillae_diameter_um: Diameter of papillae in μm (50-150 typical)
        resolution: Circular resolution
        rng: Random number generator

    Returns:
        List of papillae manifolds to union with dermis
    """
    papillae = []
    radius = diameter / 2

    # Convert μm to mm
    papillae_height_mm = papillae_height_um / 1000.0
    papillae_diameter_mm = papillae_diameter_um / 1000.0
    papillae_radius = papillae_diameter_mm / 2

    # Calculate area and number of papillae
    area_mm2 = np.pi * radius * radius
    n_papillae = max(1, int(density_per_mm2 * area_mm2))

    # Distribute papillae randomly within the circular area
    for _ in range(n_papillae):
        # Random position within circle using rejection sampling
        while True:
            x = rng.uniform(-radius + papillae_radius, radius - papillae_radius)
            y = rng.uniform(-radius + papillae_radius, radius - papillae_radius)
            if np.sqrt(x * x + y * y) < radius - papillae_radius:
                break

        # Random height variation (±20%)
        height = papillae_height_mm * rng.uniform(0.8, 1.2)

        # Create papilla as tapered cylinder with dome top
        # Cylinder body (tapered)
        cylinder_height = height * 0.7  # 70% cylinder
        dome_height = height * 0.3      # 30% dome

        cylinder = m3d.Manifold.cylinder(
            cylinder_height,
            papillae_radius,
            papillae_radius * 0.6,  # Taper to 60% at top
            max(6, resolution // 2)
        )

        # Position cylinder base at dermis-epidermis junction
        cylinder = cylinder.translate([x, y, dermis_top_z])

        # Create dome top as scaled sphere
        dome = m3d.Manifold.sphere(
            papillae_radius * 0.7,  # Slightly smaller radius for smooth taper
            max(6, resolution // 2)
        )

        # Scale dome to be ellipsoidal (taller than wide)
        dome = dome.scale([1.0, 1.0, dome_height / (papillae_radius * 0.7)])

        # Position dome at top of cylinder
        dome = dome.translate([x, y, dermis_top_z + cylinder_height + dome_height * 0.5])

        # Union cylinder and dome to create complete papilla
        papilla = cylinder + dome
        papillae.append(papilla)

    return papillae



def make_sweat_glands(
    diameter: float,
    gland_tube_diameter: float,
    gland_depth: float,
    density_per_cm2: float,
    top_z: float,
    resolution: int,
    rng: np.random.Generator
) -> list[m3d.Manifold]:
    """
    Create sweat gland structures with coiled base and straight duct.

    Sweat glands consist of a coiled secretory portion at the base
    and a straight duct leading to the skin surface.

    Args:
        diameter: Scaffold diameter in mm
        gland_tube_diameter: Tube diameter in mm
        gland_depth: Depth of gland in mm
        density_per_cm2: Number of glands per cm²
        top_z: Z coordinate of skin surface
        resolution: Circular resolution
        rng: Random number generator

    Returns:
        List of sweat gland manifolds for Boolean subtraction
    """
    glands = []
    radius = diameter / 2
    tube_radius = gland_tube_diameter / 2

    # Coil parameters
    coil_radius = 0.25  # 250 μm coil radius
    coil_turns = 3
    coil_height = 0.3  # 300 μm coil height

    # Calculate area and number of glands
    area_mm2 = np.pi * radius * radius
    area_cm2 = area_mm2 / 100.0
    n_glands = max(1, int(density_per_cm2 * area_cm2))

    for _ in range(n_glands):
        # Random position within circle
        while True:
            x = rng.uniform(-radius + coil_radius + tube_radius, radius - coil_radius - tube_radius)
            y = rng.uniform(-radius + coil_radius + tube_radius, radius - coil_radius - tube_radius)
            if np.sqrt(x * x + y * y) < radius - coil_radius - tube_radius:
                break

        gland_parts = []

        # Create straight duct from surface to coil
        duct_length = gland_depth - coil_height
        if duct_length > 0:
            duct = m3d.Manifold.cylinder(
                duct_length,
                tube_radius,
                tube_radius,
                max(6, resolution // 2)
            )
            duct = duct.translate([x, y, top_z - duct_length])
            gland_parts.append(duct)

        # Create coiled portion using series of spheres/cylinders
        # Approximate helix with connected spheres
        coil_z_base = top_z - gland_depth
        n_segments = max(12, coil_turns * 8)

        for i in range(n_segments):
            t = i / n_segments
            angle = t * coil_turns * 2 * np.pi
            z = coil_z_base + t * coil_height

            cx = x + coil_radius * np.cos(angle)
            cy = y + coil_radius * np.sin(angle)

            # Create small sphere at each point
            sphere = m3d.Manifold.sphere(tube_radius * 1.2, max(6, resolution // 2))
            sphere = sphere.translate([cx, cy, z])
            gland_parts.append(sphere)

            # Connect to next point with cylinder
            if i < n_segments - 1:
                t_next = (i + 1) / n_segments
                angle_next = t_next * coil_turns * 2 * np.pi
                z_next = coil_z_base + t_next * coil_height

                cx_next = x + coil_radius * np.cos(angle_next)
                cy_next = y + coil_radius * np.sin(angle_next)

                # Create connecting cylinder
                dx = cx_next - cx
                dy = cy_next - cy
                dz = z_next - z
                seg_length = np.sqrt(dx * dx + dy * dy + dz * dz)

                if seg_length > 0.001:
                    seg = m3d.Manifold.cylinder(
                        seg_length,
                        tube_radius,
                        tube_radius,
                        max(4, resolution // 4)
                    )

                    # Calculate rotation angles
                    # Rotate from Z-axis to direction vector
                    horiz_dist = np.sqrt(dx * dx + dy * dy)
                    if horiz_dist > 0.0001:
                        pitch = np.arctan2(horiz_dist, dz) * 180 / np.pi
                        yaw = np.arctan2(dy, dx) * 180 / np.pi
                        seg = seg.rotate([pitch, 0, yaw])

                    seg = seg.translate([cx, cy, z])
                    gland_parts.append(seg)

        if gland_parts:
            gland = batch_union(gland_parts)
            glands.append(gland)

    return glands


def make_collagen_fibers(
    diameter: float,
    z_start: float,
    z_end: float,
    fiber_diameter: float,
    alignment: float,
    is_papillary: bool,
    resolution: int,
    rng: np.random.Generator
) -> list[m3d.Manifold]:
    """
    Create collagen fiber bundles within dermis layer.

    Papillary dermis has fine, randomly oriented fibers.
    Reticular dermis has coarse fibers aligned to Langer's lines.

    Args:
        diameter: Scaffold diameter in mm
        z_start: Bottom Z coordinate of layer
        z_end: Top Z coordinate of layer
        fiber_diameter: Diameter of fiber bundles in mm
        alignment: Alignment factor (0=random, 1=fully aligned)
        is_papillary: True for papillary dermis (random), False for reticular
        resolution: Circular resolution
        rng: Random number generator

    Returns:
        List of fiber manifolds
    """
    fibers = []
    radius = diameter / 2
    fiber_radius = fiber_diameter / 2
    layer_thickness = z_end - z_start

    # Fiber density based on layer type
    # Fewer, larger fibers for visualization
    fiber_density = 50 if is_papillary else 30  # fibers per mm²

    # Calculate number of fibers
    area_mm2 = np.pi * radius * radius
    n_fibers = max(5, int(fiber_density * area_mm2 / 100))  # Reduced for performance

    # Langer's line direction (simplified as horizontal for this model)
    langer_direction = 0.0  # Angle in radians (horizontal)

    for _ in range(n_fibers):
        # Random position within circle
        while True:
            x = rng.uniform(-radius * 0.8, radius * 0.8)
            y = rng.uniform(-radius * 0.8, radius * 0.8)
            if np.sqrt(x * x + y * y) < radius * 0.8:
                break

        z = rng.uniform(z_start + fiber_radius, z_end - fiber_radius)

        # Fiber length (shorter for papillary, longer for reticular)
        fiber_length = rng.uniform(0.3, 0.8) if is_papillary else rng.uniform(0.8, 1.5)
        fiber_length = min(fiber_length, diameter * 0.4)  # Cap at 40% of diameter

        # Calculate orientation
        if is_papillary or alignment < 0.1:
            # Random orientation
            azimuth = rng.uniform(0, 2 * np.pi)
            elevation = rng.uniform(-20, 20) * np.pi / 180  # Mostly horizontal
        else:
            # Aligned to Langer's lines with some variation
            deviation = (1 - alignment) * np.pi / 2
            azimuth = langer_direction + rng.uniform(-deviation, deviation)
            elevation = rng.uniform(-10, 10) * np.pi / 180

        # Create fiber as cylinder
        fiber = m3d.Manifold.cylinder(
            fiber_length,
            fiber_radius,
            fiber_radius,
            max(4, resolution // 4)
        )

        # Rotate to orientation (elevation then azimuth)
        fiber = fiber.rotate([elevation * 180 / np.pi, 0, 0])
        fiber = fiber.rotate([0, 0, azimuth * 180 / np.pi])

        # Center the fiber at position
        fiber = fiber.translate([x - fiber_length / 2 * np.cos(azimuth), y - fiber_length / 2 * np.sin(azimuth), z])

        fibers.append(fiber)

    return fibers


def apply_surface_texture(
    base_manifold: m3d.Manifold,
    diameter: float,
    top_z: float,
    roughness: float,
    resolution: int,
    rng: np.random.Generator
) -> m3d.Manifold:
    """
    Apply surface texture to the top of the skin scaffold.

    Creates multi-scale texture simulating skin furrows and ridges.

    Args:
        base_manifold: The base skin scaffold
        diameter: Scaffold diameter in mm
        top_z: Z coordinate of top surface
        roughness: Roughness factor (0-1)
        resolution: Mesh resolution
        rng: Random number generator

    Returns:
        Manifold with textured surface
    """
    if roughness < 0.01:
        return base_manifold

    radius = diameter / 2

    # Create texture using overlapping sinusoidal patterns
    # We'll add small bumps/ridges on the surface

    # Main furrow scale: 50-100 μm depth, 200-500 μm spacing
    furrow_depth = roughness * 0.05  # Up to 50 μm
    furrow_spacing = 0.3  # 300 μm

    # Create a thin textured layer to union with top surface
    texture_elements = []

    # Create ridge pattern using small elongated bumps
    n_ridges_x = int(diameter / furrow_spacing)
    n_ridges_y = int(diameter / furrow_spacing)

    for i in range(n_ridges_x):
        for j in range(n_ridges_y):
            x = (i - n_ridges_x / 2 + 0.5) * furrow_spacing
            y = (j - n_ridges_y / 2 + 0.5) * furrow_spacing

            # Add randomness
            x += rng.uniform(-0.1, 0.1) * furrow_spacing
            y += rng.uniform(-0.1, 0.1) * furrow_spacing

            # Check if within boundary
            if np.sqrt(x * x + y * y) > radius * 0.9:
                continue

            # Create small bump
            bump_height = furrow_depth * rng.uniform(0.5, 1.0)
            bump_radius = furrow_spacing * 0.3

            bump = m3d.Manifold.cylinder(
                bump_height,
                bump_radius,
                bump_radius * 0.5,  # Tapered top
                max(4, resolution // 4)
            )
            bump = bump.translate([x, y, top_z - bump_height * 0.1])
            texture_elements.append(bump)

    if texture_elements:
        texture = batch_union(texture_elements)
        return base_manifold + texture

    return base_manifold


def generate_multilayer_skin(params: MultilayerSkinParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate a multilayer skin scaffold with biologically realistic parameters.

    Args:
        params: MultilayerSkinParams specifying geometry and structure

    Returns:
        Tuple of (manifold, stats_dict)
        - manifold: The 3D geometry
        - stats_dict: Dictionary with triangle_count, volume_mm3, layer_thicknesses,
                     porosity_gradient, scaffold_type

    Raises:
        ValueError: If parameters are invalid
    """
    # Set random seed for reproducibility
    np.random.seed(params.seed)
    rng = np.random.default_rng(params.seed)

    # Convert epidermis thickness from um to mm
    epidermis_thickness_mm = params.epidermis_thickness_um / 1000.0

    if epidermis_thickness_mm <= 0 or params.dermis_thickness_mm <= 0 or params.hypodermis_thickness_mm <= 0:
        raise ValueError("Layer thicknesses must be positive")

    # Convert rete ridge parameters from um to mm
    rete_ridge_height_mm = params.rete_ridge_height_um / 1000.0
    rete_ridge_spacing_mm = params.rete_ridge_spacing_um / 1000.0
    rete_ridge_width_mm = params.rete_ridge_width_um / 1000.0

    total_thickness = (
        epidermis_thickness_mm +
        params.dermis_thickness_mm +
        params.hypodermis_thickness_mm
    )

    # Pore size from parameters (convert um to mm)
    pore_size_mm = params.pore_diameter_um / 1000.0

    # Scale pore sizes for each layer based on biological gradients
    epidermis_pore_size = pore_size_mm * 0.5  # Smaller pores for epidermis
    dermis_pore_size = pore_size_mm           # Standard pores for dermis
    hypodermis_pore_size = pore_size_mm * 1.5 # Larger pores for hypodermis

    # Create layers from bottom to top
    layers = []

    # Hypodermis (bottom layer) - always flat, no rete ridges
    z_hypo = 0
    hypo = make_porous_layer(
        params.diameter_mm,
        params.hypodermis_thickness_mm,
        z_hypo,
        params.pore_gradient[2],
        hypodermis_pore_size,
        params.resolution,
        params.position_noise,
        rng,
        params.pore_interconnectivity
    )
    layers.append(hypo)

    # Dermis (middle layer) - ridged top if rete ridges enabled
    z_dermis = z_hypo + params.hypodermis_thickness_mm
    if params.enable_rete_ridges:
        dermis = make_rete_ridge_layer(
            params.diameter_mm,
            params.dermis_thickness_mm,
            z_dermis,
            rete_ridge_height_mm,
            rete_ridge_spacing_mm,
            rete_ridge_width_mm,
            params.rete_ridge_depth_variance,
            params.pore_gradient[1],
            dermis_pore_size,
            params.resolution,
            rng,
            is_dermis=True,
            position_noise=params.position_noise,
            pore_interconnectivity=params.pore_interconnectivity
        )
    else:
        dermis = make_porous_layer(
            params.diameter_mm,
            params.dermis_thickness_mm,
            z_dermis,
            params.pore_gradient[1],
            dermis_pore_size,
            params.resolution,
            params.position_noise,
            rng,
            params.pore_interconnectivity
        )
    layers.append(dermis)

    # Epidermis (top layer) - ridged bottom if rete ridges enabled
    z_epi = z_dermis + params.dermis_thickness_mm
    if params.enable_rete_ridges:
        epi = make_rete_ridge_layer(
            params.diameter_mm,
            epidermis_thickness_mm,
            z_epi,
            rete_ridge_height_mm,
            rete_ridge_spacing_mm,
            rete_ridge_width_mm,
            params.rete_ridge_depth_variance,
            params.pore_gradient[0],
            epidermis_pore_size,
            params.resolution,
            rng,
            is_dermis=False,
            position_noise=params.position_noise,
            pore_interconnectivity=params.pore_interconnectivity
        )
    else:
        epi = make_porous_layer(
            params.diameter_mm,
            epidermis_thickness_mm,
            z_epi,
            params.pore_gradient[0],
            epidermis_pore_size,
            params.resolution,
            params.position_noise,
            rng,
            params.pore_interconnectivity
        )
    layers.append(epi)

    # Union all layers
    result = batch_union(layers)

    # Track elements for Boolean subtraction
    elements_to_subtract = []

    # Create vascular channels if enabled
    if params.enable_vascular_channels and params.vascular_channel_count > 0 and params.vascular_channel_diameter_mm > 0:
        # Grid-based layout using vascular_channel_spacing_mm
        spacing = params.vascular_channel_spacing_mm
        radius = params.diameter_mm / 2

        # Calculate grid bounds
        grid_min = -radius
        grid_max = radius

        # Generate grid positions
        channel_count = 0
        i = 0
        while grid_min + i * spacing <= grid_max:
            j = 0
            while grid_min + j * spacing <= grid_max:
                x = grid_min + i * spacing
                y = grid_min + j * spacing

                # Add randomness if enabled
                if params.randomness > 0:
                    x += rng.uniform(-spacing * 0.2, spacing * 0.2) * params.randomness
                    y += rng.uniform(-spacing * 0.2, spacing * 0.2) * params.randomness

                # Only include channels within scaffold radius
                if np.sqrt(x**2 + y**2) <= radius * 0.9:  # 90% of radius for safety margin
                    channel = make_vascular_channel(
                        params.diameter_mm,
                        total_thickness,
                        params.vascular_channel_diameter_mm,
                        x, y,
                        params.resolution
                    )
                    if channel.num_vert() > 0:
                        elements_to_subtract.append(channel)
                        channel_count += 1

                        # Stop if we've reached the max count
                        if channel_count >= params.vascular_channel_count:
                            break

                j += 1

            if channel_count >= params.vascular_channel_count:
                break
            i += 1

    # Create hair follicles if enabled
    top_z = z_epi + epidermis_thickness_mm
    if params.enable_hair_follicles and params.hair_follicle_density_per_cm2 > 0:
        follicle_diameter_mm = params.hair_follicle_diameter_um / 1000.0
        follicles = make_hair_follicles(
            params.diameter_mm,
            follicle_diameter_mm,
            params.hair_follicle_depth_mm,
            params.hair_follicle_density_per_cm2,
            top_z,
            params.resolution,
            rng
        )
        elements_to_subtract.extend(follicles)

    # Create sweat glands if enabled
    if params.enable_sweat_glands and params.sweat_gland_density_per_cm2 > 0:
        gland_diameter_mm = params.sweat_gland_diameter_um / 1000.0
        sweat_glands = make_sweat_glands(
            params.diameter_mm,
            gland_diameter_mm,
            params.sweat_gland_depth_mm,
            params.sweat_gland_density_per_cm2,
            top_z,
            params.resolution,
            rng
        )
        elements_to_subtract.extend(sweat_glands)

    # Create sebaceous glands if enabled
    if params.enable_sebaceous_glands and params.sebaceous_gland_density_per_cm2 > 0:
        sebaceous_glands = make_sebaceous_glands(
            params.diameter_mm,
            params.sebaceous_gland_density_per_cm2,
            z_dermis,
            params.dermis_thickness_mm,
            params.sebaceous_gland_diameter_um,
            params.resolution,
            rng
        )
        elements_to_subtract.extend(sebaceous_glands)

    # Subtract all channels/follicles/glands from result
    if elements_to_subtract:
        subtract_union = batch_union(elements_to_subtract)
        result = result - subtract_union

    # Track elements for Boolean addition (union)
    elements_to_add = []

    # Create dermal papillae if enabled
    if params.enable_dermal_papillae and params.papillae_density_per_mm2 > 0:
        # Dermis top is at z_dermis + dermis_thickness_mm
        dermis_top_z = z_dermis + params.dermis_thickness_mm
        papillae = make_dermal_papillae(
            params.diameter_mm,
            params.papillae_density_per_mm2,
            dermis_top_z,
            params.papillae_height_um,
            params.papillae_diameter_um,
            params.resolution,
            rng
        )
        elements_to_add.extend(papillae)

    # Add collagen fibers if enabled
    if params.enable_collagen_orientation:
        # Papillary dermis collagen (fine, random)
        papillary_z_start = z_dermis + params.dermis_thickness_mm - params.papillary_dermis_thickness_um / 1000.0
        papillary_z_end = z_dermis + params.dermis_thickness_mm
        papillary_fiber_diameter_mm = params.papillary_collagen_diameter_um / 1000.0

        papillary_fibers = make_collagen_fibers(
            params.diameter_mm,
            papillary_z_start,
            papillary_z_end,
            papillary_fiber_diameter_mm,
            0.0,  # Random orientation for papillary
            is_papillary=True,
            resolution=params.resolution,
            rng=rng
        )
        elements_to_add.extend(papillary_fibers)

        # Reticular dermis collagen (coarse, aligned)
        reticular_z_start = z_dermis
        reticular_z_end = papillary_z_start
        reticular_fiber_diameter_mm = params.reticular_collagen_diameter_um / 1000.0

        reticular_fibers = make_collagen_fibers(
            params.diameter_mm,
            reticular_z_start,
            reticular_z_end,
            reticular_fiber_diameter_mm,
            params.collagen_alignment,
            is_papillary=False,
            resolution=params.resolution,
            rng=rng
        )
        elements_to_add.extend(reticular_fibers)

    # Union all additive elements (papillae, collagen) with result
    if elements_to_add:
        add_union = batch_union(elements_to_add)
        result = result + add_union

    # Apply surface texture if enabled
    if params.enable_surface_texture and params.surface_roughness > 0:
        result = apply_surface_texture(
            result,
            params.diameter_mm,
            top_z,
            params.surface_roughness,
            params.resolution,
            rng
        )

    # Calculate statistics
    mesh = result.to_mesh()
    volume = result.volume() if hasattr(result, 'volume') else 0

    stats = {
        'triangle_count': len(mesh.tri_verts) // 3 if hasattr(mesh, 'tri_verts') else 0,
        'volume_mm3': volume,
        'layer_thicknesses_mm': {
            'epidermis': epidermis_thickness_mm,
            'papillary_dermis': params.papillary_dermis_thickness_um / 1000.0,
            'reticular_dermis': params.reticular_dermis_thickness_mm,
            'dermis_total': params.dermis_thickness_mm,
            'hypodermis': params.hypodermis_thickness_mm,
            'total': total_thickness
        },
        'porosity_gradient': {
            'epidermis': params.pore_gradient[0],
            'dermis': params.pore_gradient[1],
            'hypodermis': params.pore_gradient[2]
        },
        'dermal_porosity': params.dermal_porosity,
        'pore_diameter_um': params.pore_diameter_um,
        'pore_interconnectivity': params.pore_interconnectivity,
        'keratinocyte_layers': params.keratinocyte_layers,  # Reference only, not geometrically modeled
        'rete_ridge_params': {
            'enabled': params.enable_rete_ridges,
            'height_um': params.rete_ridge_height_um,
            'width_um': params.rete_ridge_width_um,
            'spacing_um': params.rete_ridge_spacing_um,
            'depth_variance': params.rete_ridge_depth_variance
        },
        'appendages': {
            'hair_follicles_enabled': params.enable_hair_follicles,
            'hair_follicle_density_per_cm2': params.hair_follicle_density_per_cm2,
            'hair_follicle_diameter_um': params.hair_follicle_diameter_um,
            'hair_follicle_depth_mm': params.hair_follicle_depth_mm,
            'sweat_glands_enabled': params.enable_sweat_glands,
            'sweat_gland_density_per_cm2': params.sweat_gland_density_per_cm2,
            'sweat_gland_diameter_um': params.sweat_gland_diameter_um,
            'sweat_gland_depth_mm': params.sweat_gland_depth_mm,
            'sebaceous_glands_enabled': params.enable_sebaceous_glands,
            'sebaceous_gland_density_per_cm2': params.sebaceous_gland_density_per_cm2,
            'sebaceous_gland_diameter_um': params.sebaceous_gland_diameter_um
        },
        'collagen': {
            'enabled': params.enable_collagen_orientation,
            'papillary_diameter_um': params.papillary_collagen_diameter_um,
            'reticular_diameter_um': params.reticular_collagen_diameter_um,
            'alignment': params.collagen_alignment
        },
        'surface_texture': {
            'enabled': params.enable_surface_texture,
            'roughness': params.surface_roughness
        },
        'vascular_channels': params.vascular_channel_count if params.enable_vascular_channels else 0,
        'scaffold_type': 'multilayer_skin'
    }

    return result, stats


def generate_multilayer_skin_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate multilayer skin scaffold from dictionary parameters (for API compatibility).

    Args:
        params: Dictionary with keys matching MultilayerSkinParams fields

    Returns:
        Tuple of (manifold, stats_dict)
    """
    # Handle pore_gradient as array or dict
    pore_gradient = params.get('pore_gradient', (0.15, 0.25, 0.40))
    if isinstance(pore_gradient, dict):
        pore_gradient = (
            pore_gradient.get('epidermis', 0.15),
            pore_gradient.get('dermis', 0.25),
            pore_gradient.get('hypodermis', 0.40)
        )
    elif isinstance(pore_gradient, list):
        pore_gradient = tuple(pore_gradient)

    # Support legacy parameter names
    epidermis_thickness_um = params.get('epidermis_thickness_um', 100.0)
    if 'epidermis_thickness_mm' in params:
        epidermis_thickness_um = params['epidermis_thickness_mm'] * 1000  # Convert mm to um

    return generate_multilayer_skin(MultilayerSkinParams(
        # Basic geometry
        diameter_mm=params.get('diameter_mm', 10.0),

        # Layer thicknesses
        epidermis_thickness_um=epidermis_thickness_um,
        dermis_thickness_mm=params.get('dermis_thickness_mm', 1.5),
        papillary_dermis_thickness_um=params.get('papillary_dermis_thickness_um', 200.0),
        reticular_dermis_thickness_mm=params.get('reticular_dermis_thickness_mm', 1.3),
        hypodermis_thickness_mm=params.get('hypodermis_thickness_mm', 10.0),
        keratinocyte_layers=params.get('keratinocyte_layers', 5),

        # Rete ridges
        enable_rete_ridges=params.get('enable_rete_ridges', True),
        rete_ridge_height_um=params.get('rete_ridge_height_um', 51.0),
        rete_ridge_width_um=params.get('rete_ridge_width_um', 150.0),
        rete_ridge_spacing_um=params.get('rete_ridge_spacing_um', 105.0),
        rete_ridge_depth_variance=params.get('rete_ridge_depth_variance', 0.2),

        # Dermal papillae
        enable_dermal_papillae=params.get('enable_dermal_papillae', True),
        papillae_density_per_mm2=params.get('papillae_density_per_mm2', 100.0),
        papillae_height_um=params.get('papillae_height_um', 100.0),
        papillae_diameter_um=params.get('papillae_diameter_um', 75.0),

        # Hair follicles
        enable_hair_follicles=params.get('enable_hair_follicles', False),
        hair_follicle_density_per_cm2=params.get('hair_follicle_density_per_cm2', 130.0),
        hair_follicle_diameter_um=params.get('hair_follicle_diameter_um', 70.0),
        hair_follicle_depth_mm=params.get('hair_follicle_depth_mm', 2.5),

        # Sweat glands
        enable_sweat_glands=params.get('enable_sweat_glands', False),
        sweat_gland_density_per_cm2=params.get('sweat_gland_density_per_cm2', 155.0),
        sweat_gland_diameter_um=params.get('sweat_gland_diameter_um', 40.0),
        sweat_gland_depth_mm=params.get('sweat_gland_depth_mm', 1.5),

        # Sebaceous glands
        enable_sebaceous_glands=params.get('enable_sebaceous_glands', False),
        sebaceous_gland_density_per_cm2=params.get('sebaceous_gland_density_per_cm2', 100.0),
        sebaceous_gland_diameter_um=params.get('sebaceous_gland_diameter_um', 100.0),

        # Dermal features
        dermal_porosity=params.get('dermal_porosity', 0.19),
        pore_diameter_um=params.get('pore_diameter_um', 219.0),
        pore_interconnectivity=params.get('pore_interconnectivity', 0.8),
        pore_gradient=pore_gradient,

        # Vascular channels
        enable_vascular_channels=params.get('enable_vascular_channels', True),
        vascular_channel_diameter_mm=params.get('vascular_channel_diameter_mm', 0.15),
        vascular_channel_count=params.get('vascular_channel_count', 4),
        vascular_channel_spacing_mm=params.get('vascular_channel_spacing_mm', 2.0),

        # Collagen architecture
        enable_collagen_orientation=params.get('enable_collagen_orientation', False),
        papillary_collagen_diameter_um=params.get('papillary_collagen_diameter_um', 2.0),
        reticular_collagen_diameter_um=params.get('reticular_collagen_diameter_um', 10.0),
        collagen_alignment=params.get('collagen_alignment', 0.3),

        # Surface features
        enable_surface_texture=params.get('enable_surface_texture', False),
        surface_roughness=params.get('surface_roughness', 0.1),

        # Randomization
        seed=params.get('seed', 42),
        randomness=params.get('randomness', 0.1),
        position_noise=params.get('position_noise', 0.15),

        # Resolution
        resolution=params.get('resolution', 16)
    ))
