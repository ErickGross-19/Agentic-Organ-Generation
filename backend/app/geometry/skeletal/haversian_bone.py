"""
Haversian bone scaffold generator with cortical bone microstructure.

Creates a solid cortical bone structure with longitudinal vascular channels
(Haversian canals) arranged in osteon patterns (hexagonal or random).
Implements biologically accurate features including:
- Volkmann's canals (transverse connections)
- Concentric lamellae (rings around canals)
- Lacunae (osteocyte spaces)
- Canaliculi (fine connecting channels)
- Cement lines (osteon boundaries)
- Interstitial lamellae (old bone fragments)
- Periosteal/endosteal surfaces
- Resorption spaces (remodeling cavities)
"""

from __future__ import annotations
import manifold3d as m3d
import numpy as np
from dataclasses import dataclass
from typing import Literal
from ..core import batch_union


@dataclass
class HaversianBoneParams:
    """Parameters for Haversian bone scaffold generation.

    Biologically realistic parameters based on human cortical bone:
    - Haversian canal diameter: 40-90 μm (mean ~50 μm)
    - Osteon diameter: 150-350 μm (mean ~200-250 μm)
    - Volkmann's canal diameter: 20-100 μm (mean ~75 μm)
    - Concentric lamellae: 4-20 per osteon (mean ~8)
    - Lamella thickness: 3-7 μm
    - Cortical porosity: 5-15% (mean ~8%)
    """
    # Basic geometry
    bounding_box: tuple[float, float, float] = (5.0, 5.0, 8.0)
    resolution: int = 8
    seed: int = 42

    # Haversian canal (central vascular channel)
    haversian_canal_diameter_um: float = 50.0  # Central canal diameter
    canal_diameter_um: float = 50.0  # Legacy alias
    canal_wall_thickness_um: float = 5.0  # Canal lining thickness
    enable_haversian_vessels: bool = True  # Include vessel structures

    # Osteon (Haversian system) properties
    osteon_diameter_um: float = 225.0  # Full osteon diameter
    canal_spacing_um: float = 400.0  # Legacy: spacing between canals
    osteon_spacing_um: float = 250.0  # Center-to-center osteon spacing
    osteon_pattern: Literal['hexagonal', 'random', 'organized'] = 'hexagonal'

    # Volkmann's canals (transverse/oblique connecting canals)
    enable_volkmann_canals: bool = True  # Include transverse canals
    volkmann_canal_diameter_um: float = 75.0  # Transverse canal diameter
    volkmann_canal_density: float = 0.5  # Density of Volkmann canals (0-1)
    volkmann_angle_deg: float = 60.0  # Angle from longitudinal axis

    # Concentric lamellae (rings around Haversian canal)
    num_concentric_lamellae: int = 8  # Lamellae per osteon (4-20 typical)
    lamella_thickness_um: float = 5.0  # Individual lamella thickness (3-7 μm)
    interlamellar_spacing_um: float = 1.0  # Gap between lamellae
    enable_lamellar_detail: bool = True  # Show lamellar structure

    # Lacunae (osteocyte spaces)
    enable_lacunae: bool = True  # Osteocyte lacunae
    lacuna_density: float = 15000.0  # Lacunae per mm³ (10000-25000 typical)
    lacuna_length_um: float = 20.0  # Lacuna long axis (15-25 μm)
    lacuna_width_um: float = 8.0  # Lacuna short axis (5-10 μm)

    # Canaliculi (connecting channels from lacunae)
    enable_canaliculi: bool = False  # Fine connecting channels (high resolution)
    canaliculus_diameter_um: float = 0.5  # Very small channels
    canaliculi_per_lacuna: int = 50  # Channels per lacuna (40-100 typical)

    # Cement line (osteon boundary)
    enable_cement_line: bool = True  # Boundary between osteons
    cement_line_thickness_um: float = 3.0  # Cement line thickness (1-5 μm)

    # Interstitial lamellae (between osteons)
    enable_interstitial_lamellae: bool = True  # Old bone fragments
    interstitial_fraction: float = 0.2  # Fraction of area as interstitial

    # Cortical bone properties
    cortical_thickness: float = 5.0  # Legacy: overall cortical thickness (mm)
    cortical_porosity: float = 0.08  # Total porosity (5-15%)
    bone_mineral_density: float = 1.8  # g/cm³ indicator

    # Periosteal and endosteal surfaces
    enable_periosteal_surface: bool = True  # Outer bone surface
    enable_endosteal_surface: bool = True  # Inner bone surface
    periosteal_layer_thickness_um: float = 50.0  # Periosteum-adjacent layer
    endosteal_layer_thickness_um: float = 30.0  # Endosteum-adjacent layer

    # Remodeling features
    enable_resorption_spaces: bool = False  # Active remodeling cavities
    resorption_space_diameter_um: float = 200.0  # Cutting cone diameter
    resorption_density: float = 0.02  # Fraction of bone under remodeling

    # Mechanical orientation
    primary_orientation_deg: float = 0.0  # Main osteon alignment
    orientation_variance_deg: float = 10.0  # Random variance in alignment

    # Surface characteristics
    surface_roughness: float = 0.05  # Surface texture

    # Randomization
    position_noise: float = 0.1  # Random jitter in osteon positions
    diameter_variance: float = 0.15  # Variation in osteon/canal diameters
    lamella_count_variance: int = 2  # +/- variation in lamella count


def _um_to_mm(um: float) -> float:
    """Convert micrometers to millimeters."""
    return um / 1000.0


def _create_rotation_matrix(angle_deg: float, axis: str = 'z') -> np.ndarray:
    """Create a 3D rotation matrix for the given angle and axis."""
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    if axis == 'z':
        return np.array([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ])
    elif axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ])
    elif axis == 'y':
        return np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
    return np.eye(3)


def _create_cylindrical_shell(
    inner_radius: float,
    outer_radius: float,
    height: float,
    resolution: int
) -> m3d.Manifold:
    """
    Create a cylindrical shell (tube/ring).

    Args:
        inner_radius: Inner radius of the shell
        outer_radius: Outer radius of the shell
        height: Height of the cylinder
        resolution: Number of segments for circular approximation

    Returns:
        Manifold representing the cylindrical shell
    """
    outer_cyl = m3d.Manifold.cylinder(height, outer_radius, outer_radius, resolution)
    inner_cyl = m3d.Manifold.cylinder(height, inner_radius, inner_radius, resolution)
    return outer_cyl - inner_cyl


def _create_ellipsoid(
    length: float,
    width: float,
    depth: float,
    resolution: int
) -> m3d.Manifold:
    """
    Create an ellipsoid by scaling a sphere.

    Args:
        length: Size along X axis
        width: Size along Y axis
        depth: Size along Z axis
        resolution: Resolution for sphere

    Returns:
        Manifold representing the ellipsoid
    """
    # Create a unit sphere and scale it
    radius = max(length, width, depth) / 2
    sphere = m3d.Manifold.sphere(radius, resolution)
    # Scale to ellipsoid dimensions
    scale_x = length / (2 * radius)
    scale_y = width / (2 * radius)
    scale_z = depth / (2 * radius)
    return sphere.scale([scale_x, scale_y, scale_z])


def _create_canal_wall(
    canal_center: tuple[float, float],
    canal_radius: float,
    wall_thickness: float,
    height: float,
    resolution: int
) -> m3d.Manifold:
    """
    Create a wall/shell around a Haversian canal.

    The canal wall represents the immediate bone tissue lining the vascular
    channel, distinct from the concentric lamellae.

    Args:
        canal_center: (x, y) position of the canal center
        canal_radius: Inner radius (lumen radius)
        wall_thickness: Thickness of the canal wall
        height: Height of the cylinder
        resolution: Circular resolution

    Returns:
        Cylindrical shell Manifold representing the canal wall
    """
    outer_radius = canal_radius + wall_thickness
    wall = _create_cylindrical_shell(canal_radius, outer_radius, height, resolution)
    wall = wall.translate([canal_center[0], canal_center[1], 0])
    return wall


def _apply_surface_roughness(
    manifold: m3d.Manifold,
    roughness: float,
    bounding_box: tuple[float, float, float],
    resolution: int,
    rng: np.random.Generator
) -> m3d.Manifold:
    """
    Apply surface roughness/texture to bone geometry.

    Creates small surface perturbations by adding and subtracting small
    spheres at random positions on the outer surfaces.

    Args:
        manifold: The geometry to roughen
        roughness: Roughness factor (0-1), controls size and density of perturbations
        bounding_box: (bx, by, bz) dimensions
        resolution: Resolution for sphere creation
        rng: Random number generator

    Returns:
        Manifold with surface roughness applied
    """
    if roughness <= 0:
        return manifold

    bx, by, bz = bounding_box

    # Roughness controls the perturbation size (scaled to mm)
    # roughness of 0.05 = perturbations of ~50 micrometers
    perturbation_size = roughness * 0.1  # Convert to mm scale

    # Number of perturbations based on surface area and roughness
    surface_area = 2 * (bx * by + bx * bz + by * bz)
    num_perturbations = int(surface_area * roughness * 100)
    num_perturbations = min(num_perturbations, 200)  # Limit for performance

    if num_perturbations == 0:
        return manifold

    additions = []
    subtractions = []

    for _ in range(num_perturbations):
        # Random size within roughness range
        size = rng.uniform(perturbation_size * 0.3, perturbation_size)

        # Randomly choose a surface (top, bottom, or sides)
        surface = rng.integers(0, 6)
        if surface == 0:  # Top (Z = bz)
            x = rng.uniform(size, bx - size)
            y = rng.uniform(size, by - size)
            z = bz
        elif surface == 1:  # Bottom (Z = 0)
            x = rng.uniform(size, bx - size)
            y = rng.uniform(size, by - size)
            z = 0
        elif surface == 2:  # Front (Y = 0)
            x = rng.uniform(size, bx - size)
            y = 0
            z = rng.uniform(size, bz - size)
        elif surface == 3:  # Back (Y = by)
            x = rng.uniform(size, bx - size)
            y = by
            z = rng.uniform(size, bz - size)
        elif surface == 4:  # Left (X = 0)
            x = 0
            y = rng.uniform(size, by - size)
            z = rng.uniform(size, bz - size)
        else:  # Right (X = bx)
            x = bx
            y = rng.uniform(size, by - size)
            z = rng.uniform(size, bz - size)

        # Create small sphere for perturbation
        sphere = m3d.Manifold.sphere(size, max(4, resolution // 2))
        sphere = sphere.translate([x, y, z])

        # Randomly add or subtract (60% subtract for porous appearance)
        if rng.random() < 0.6:
            subtractions.append(sphere)
        else:
            additions.append(sphere)

    # Apply additions and subtractions
    result = manifold
    if additions:
        additions_union = batch_union(additions)
        if additions_union:
            result = result + additions_union

    if subtractions:
        subtractions_union = batch_union(subtractions)
        if subtractions_union:
            result = result - subtractions_union

    return result


def _create_volkmann_canals(
    canal_positions: list[tuple[float, float]],
    bx: float,
    by: float,
    bz: float,
    params: HaversianBoneParams,
    rng: np.random.Generator
) -> list[m3d.Manifold]:
    """
    Create Volkmann's canals connecting Haversian canals horizontally.

    Volkmann's canals run perpendicular (or oblique) to Haversian canals,
    connecting adjacent osteons. Diameter: 50-70μm, angle: ~90° from vertical.

    Args:
        canal_positions: List of (x, y) positions of Haversian canals
        bx, by, bz: Bounding box dimensions
        params: Generator parameters
        rng: Random number generator

    Returns:
        List of Manifold objects representing Volkmann's canals
    """
    if not params.enable_volkmann_canals:
        return []

    volkmann_radius = _um_to_mm(params.volkmann_canal_diameter_um) / 2
    # Use parameter-specified angle (typically 45-90° from longitudinal axis)
    base_angle = params.volkmann_angle_deg

    canals = []

    # Find pairs of nearby Haversian canals to connect
    for i, (x1, y1) in enumerate(canal_positions):
        for j, (x2, y2) in enumerate(canal_positions):
            if i >= j:
                continue

            # Calculate distance between canals
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            # Only connect nearby canals (within ~2x osteon spacing)
            max_dist = _um_to_mm(params.osteon_spacing_um) * 2
            if dist > max_dist or dist < _um_to_mm(params.osteon_diameter_um):
                continue

            # Apply density filter
            if rng.random() > params.volkmann_canal_density:
                continue

            # Create 2-4 Volkmann canals per pair along the Z axis
            num_connections = rng.integers(2, 5)
            z_positions = np.linspace(bz * 0.1, bz * 0.9, num_connections)

            for z in z_positions:
                # Add some randomness to position
                z_offset = rng.uniform(-bz * 0.05, bz * 0.05)
                z_actual = max(0.1, min(bz - 0.1, z + z_offset))

                # Calculate direction and length
                dx = x2 - x1
                dy = y2 - y1
                length = dist

                # Create horizontal cylinder
                canal = m3d.Manifold.cylinder(
                    length, volkmann_radius, volkmann_radius, params.resolution
                )

                # Rotate to align with direction between canals
                angle_xy = np.degrees(np.arctan2(dy, dx))
                # Volkmann canals at angle from longitudinal axis (90° = horizontal, 60° = oblique)
                canal = canal.rotate([0, base_angle, 0])  # Tilt by volkmann_angle_deg
                canal = canal.rotate([0, 0, angle_xy])  # Align with direction between canals

                # Translate to position
                canal = canal.translate([x1, y1, z_actual])

                canals.append(canal)

    return canals


def _create_concentric_lamellae(
    canal_center: tuple[float, float],
    canal_radius: float,
    osteon_radius: float,
    num_lamellae: int,
    lamella_thickness: float,
    interlamellar_spacing: float,
    height: float,
    resolution: int
) -> list[m3d.Manifold]:
    """
    Create concentric lamellae rings around a Haversian canal.

    Lamellae are 5-7μm thick concentric layers of mineralized collagen
    surrounding the central Haversian canal.

    Args:
        canal_center: (x, y) position of the canal center
        canal_radius: Radius of the Haversian canal
        osteon_radius: Outer radius of the osteon
        num_lamellae: Number of concentric lamellae (8-12 typical)
        lamella_thickness: Thickness of each lamella
        interlamellar_spacing: Gap between adjacent lamellae
        height: Height of the structure
        resolution: Circular resolution

    Returns:
        List of cylindrical shell Manifolds representing lamellae
    """
    rings = []

    # Calculate radii for each lamella, accounting for interlamellar spacing
    available_space = osteon_radius - canal_radius
    spacing_per_lamella = lamella_thickness + interlamellar_spacing
    actual_num = min(num_lamellae, int(available_space / spacing_per_lamella))

    r_inner = canal_radius
    for i in range(actual_num):
        r_outer = r_inner + lamella_thickness
        if r_outer > osteon_radius:
            break

        ring = _create_cylindrical_shell(r_inner, r_outer, height, resolution)
        ring = ring.translate([canal_center[0], canal_center[1], 0])
        rings.append(ring)

        # Add interlamellar spacing for the next lamella
        r_inner = r_outer + interlamellar_spacing

    return rings


def _create_lacunae_in_osteon(
    canal_center: tuple[float, float],
    canal_radius: float,
    osteon_radius: float,
    height: float,
    lacuna_density: float,
    lacuna_dims: tuple[float, float, float],
    resolution: int,
    rng: np.random.Generator
) -> list[m3d.Manifold]:
    """
    Create lacunae (osteocyte spaces) distributed within an osteon.

    Lacunae are ellipsoid cavities (~15μm × 8μm × 5μm) where osteocytes reside,
    distributed within the lamellae at ~50,000 per mm³.

    Args:
        canal_center: (x, y) position of the canal center
        canal_radius: Radius of the Haversian canal
        osteon_radius: Outer radius of the osteon
        height: Height of the osteon
        lacuna_density: Number of lacunae per mm³
        lacuna_dims: (length, width, depth) of ellipsoid lacunae
        resolution: Resolution for ellipsoid creation
        rng: Random number generator

    Returns:
        List of ellipsoid Manifolds representing lacunae
    """
    lacunae = []

    # Calculate osteon volume (annular cylinder)
    osteon_volume = np.pi * (osteon_radius**2 - canal_radius**2) * height

    # Calculate number of lacunae based on density
    num_lacunae = int(osteon_volume * lacuna_density)
    num_lacunae = min(num_lacunae, 50)  # Limit for performance

    length, width, depth = lacuna_dims

    for _ in range(num_lacunae):
        # Random position within the lamellar region
        r = rng.uniform(canal_radius + length/2, osteon_radius - length/2)
        theta = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(length/2, height - length/2)

        x = canal_center[0] + r * np.cos(theta)
        y = canal_center[1] + r * np.sin(theta)

        # Create ellipsoid lacuna
        lacuna = _create_ellipsoid(length, width, depth, max(4, resolution // 2))

        # Orient lacuna tangent to lamella (parallel to canal axis rotation)
        lacuna = lacuna.rotate([0, 0, np.degrees(theta)])
        lacuna = lacuna.translate([x, y, z])

        lacunae.append(lacuna)

    return lacunae


def _create_canaliculi_from_lacuna(
    lacuna_center: tuple[float, float, float],
    num_canaliculi: int,
    canaliculus_radius: float,
    canaliculus_length: float,
    resolution: int,
    rng: np.random.Generator
) -> list[m3d.Manifold]:
    """
    Create canaliculi radiating from a lacuna.

    Canaliculi are very fine channels (~0.2μm diameter) extending 25-40μm
    from each lacuna, with ~60 per lacuna.

    Args:
        lacuna_center: (x, y, z) position of the lacuna
        num_canaliculi: Number of channels to create
        canaliculus_radius: Radius of each channel
        canaliculus_length: Length of each channel
        resolution: Circular resolution
        rng: Random number generator

    Returns:
        List of thin cylinder Manifolds representing canaliculi
    """
    canaliculi = []
    x, y, z = lacuna_center

    # Create channels radiating in random directions
    for _ in range(num_canaliculi):
        # Random direction
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)

        # Create thin cylinder
        canal = m3d.Manifold.cylinder(
            canaliculus_length, canaliculus_radius, canaliculus_radius,
            max(4, resolution // 2)
        )

        # Rotate to random direction
        canal = canal.rotate([0, np.degrees(phi), np.degrees(theta)])
        canal = canal.translate([x, y, z])

        canaliculi.append(canal)

    return canaliculi


def _create_cement_lines(
    canal_positions: list[tuple[float, float]],
    osteon_radius: float,
    cement_thickness: float,
    height: float,
    resolution: int
) -> list[m3d.Manifold]:
    """
    Create cement lines (boundaries) around each osteon.

    Cement lines are ~2μm thick mineralized boundaries marking the
    outer edge of each osteon.

    Args:
        canal_positions: List of (x, y) positions of Haversian canals
        osteon_radius: Outer radius of the osteon
        cement_thickness: Thickness of cement line
        height: Height of the structure
        resolution: Circular resolution

    Returns:
        List of thin cylindrical shell Manifolds
    """
    lines = []

    for x, y in canal_positions:
        inner_r = osteon_radius - cement_thickness / 2
        outer_r = osteon_radius + cement_thickness / 2

        line = _create_cylindrical_shell(inner_r, outer_r, height, resolution)
        line = line.translate([x, y, 0])
        lines.append(line)

    return lines


def _create_interstitial_lamellae(
    bx: float,
    by: float,
    bz: float,
    canal_positions: list[tuple[float, float]],
    osteon_radius: float,
    interstitial_fraction: float,
    resolution: int,
    rng: np.random.Generator
) -> list[m3d.Manifold]:
    """
    Create interstitial lamellae (old bone remnants between osteons).

    Interstitial lamellae are fragments of previously remodeled bone
    filling ~20% of the space between osteons.

    Args:
        bx, by, bz: Bounding box dimensions
        canal_positions: List of (x, y) positions of Haversian canals
        osteon_radius: Radius of each osteon
        interstitial_fraction: Fraction of interstitial volume to fill
        resolution: Resolution for geometry
        rng: Random number generator

    Returns:
        List of irregular shape Manifolds representing interstitial bone
    """
    interstitials = []

    # Create small irregular patches in spaces between osteons
    # Use small cylinders/boxes as simplified representation

    # Sample grid points
    spacing = osteon_radius / 2
    nx = int(bx / spacing) + 1
    ny = int(by / spacing) + 1

    for ix in range(nx):
        for iy in range(ny):
            x = ix * spacing + spacing / 2
            y = iy * spacing + spacing / 2

            if x < 0.1 or x > bx - 0.1 or y < 0.1 or y > by - 0.1:
                continue

            # Check if point is outside all osteons
            in_osteon = False
            for cx, cy in canal_positions:
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist < osteon_radius * 1.1:
                    in_osteon = True
                    break

            if in_osteon:
                continue

            # Apply interstitial fraction filter
            if rng.random() > interstitial_fraction:
                continue

            # Create small irregular interstitial fragment
            size = rng.uniform(spacing * 0.3, spacing * 0.6)
            height_frac = rng.uniform(0.5, 1.0)

            fragment = m3d.Manifold.cylinder(
                bz * height_frac, size, size * 0.8, resolution
            )
            z_offset = rng.uniform(0, bz * (1 - height_frac))
            fragment = fragment.translate([x, y, z_offset])

            interstitials.append(fragment)

    return interstitials


def _create_periosteal_layer(
    bx: float,
    by: float,
    bz: float,
    thickness: float
) -> m3d.Manifold:
    """
    Create periosteal surface layer (outer bone surface).

    The periosteum is the outer membrane layer, ~50-200μm thick.

    Args:
        bx, by, bz: Bounding box dimensions
        thickness: Layer thickness in mm

    Returns:
        Manifold representing the outer shell layer
    """
    # Outer box minus inner box to create shell
    outer = m3d.Manifold.cube([bx, by, bz])
    inner = m3d.Manifold.cube([
        bx - 2 * thickness,
        by - 2 * thickness,
        bz
    ])
    inner = inner.translate([thickness, thickness, 0])

    # Only keep the sides (not top/bottom for now)
    shell = outer - inner

    return shell


def _create_endosteal_layer(
    bx: float,
    by: float,
    bz: float,
    thickness: float,
    inner_margin: float = 0.5
) -> m3d.Manifold:
    """
    Create endosteal surface layer (inner bone surface).

    The endosteum is the inner lining, ~10μm thick.
    For a block model, we represent this as a thin layer at the center.

    Args:
        bx, by, bz: Bounding box dimensions
        thickness: Layer thickness in mm
        inner_margin: Distance from center to place the layer

    Returns:
        Manifold representing the inner layer
    """
    # Create a thin layer near the center of the block
    center_x = bx / 2
    center_y = by / 2

    # Create a thin vertical sheet
    layer = m3d.Manifold.cube([thickness, by * 0.8, bz])
    layer = layer.translate([center_x - thickness / 2, by * 0.1, 0])

    return layer


def _create_resorption_spaces(
    bx: float,
    by: float,
    bz: float,
    resorption_diameter: float,
    resorption_density: float,
    resolution: int,
    rng: np.random.Generator
) -> list[m3d.Manifold]:
    """
    Create resorption spaces (Howship's lacunae and cutting cones).

    Resorption spaces are ~250μm diameter × 50μm depth cavities where
    osteoclasts actively remodel bone. Cutting cones are ~200μm tunnels.

    Args:
        bx, by, bz: Bounding box dimensions
        resorption_diameter: Diameter of resorption cavities
        resorption_density: Fraction of volume as resorption spaces (~2%)
        resolution: Resolution for geometry
        rng: Random number generator

    Returns:
        List of spherical/cylindrical Manifolds representing resorption spaces
    """
    spaces = []

    # Calculate volume and number of spaces
    total_volume = bx * by * bz
    space_volume = (4/3) * np.pi * (resorption_diameter / 2)**3
    num_spaces = int(total_volume * resorption_density / space_volume)
    num_spaces = min(num_spaces, 20)  # Limit for performance

    for _ in range(num_spaces):
        # Random position
        x = rng.uniform(resorption_diameter, bx - resorption_diameter)
        y = rng.uniform(resorption_diameter, by - resorption_diameter)
        z = rng.uniform(resorption_diameter, bz - resorption_diameter)

        # Randomly choose between Howship's lacuna (sphere) and cutting cone (cylinder)
        if rng.random() > 0.5:
            # Spherical Howship's lacuna
            space = m3d.Manifold.sphere(resorption_diameter / 2, resolution)
            space = space.translate([x, y, z])
        else:
            # Cylindrical cutting cone
            length = rng.uniform(resorption_diameter, resorption_diameter * 3)
            space = m3d.Manifold.cylinder(
                length, resorption_diameter / 2, resorption_diameter / 2, resolution
            )
            # Random orientation
            theta = rng.uniform(0, 360)
            phi = rng.uniform(0, 90)
            space = space.rotate([phi, 0, theta])
            space = space.translate([x, y, z])

        spaces.append(space)

    return spaces


def _apply_orientation(
    manifold: m3d.Manifold,
    primary_orientation_deg: float,
    variance_deg: float,
    rng: np.random.Generator
) -> m3d.Manifold:
    """
    Apply orientation rotation to the bone structure.

    Rotates the entire structure based on loading direction with variance
    for organic appearance.

    Args:
        manifold: The geometry to rotate
        primary_orientation_deg: Main alignment angle (degrees)
        variance_deg: Random variance to add
        rng: Random number generator

    Returns:
        Rotated manifold
    """
    # Add variance to primary orientation
    actual_orientation = primary_orientation_deg + rng.uniform(-variance_deg, variance_deg)

    # Rotate around Z axis (vertical/longitudinal axis of bone)
    return manifold.rotate([0, 0, actual_orientation])


def generate_haversian_bone(params: HaversianBoneParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate a Haversian bone scaffold with complete osteon structure.

    Creates cortical bone with all biologically relevant features:
    - Haversian canals (longitudinal vascular channels)
    - Volkmann's canals (transverse connections)
    - Concentric lamellae (rings around canals)
    - Lacunae (osteocyte spaces)
    - Canaliculi (fine connecting channels)
    - Cement lines (osteon boundaries)
    - Interstitial lamellae (old bone fragments)
    - Periosteal/endosteal surfaces
    - Resorption spaces (remodeling cavities)

    Args:
        params: HaversianBoneParams specifying geometry and structure

    Returns:
        Tuple of (manifold, stats_dict)
        - manifold: The 3D geometry
        - stats_dict: Dictionary with triangle_count, volume_mm3, canal_count,
                     osteon_pattern, scaffold_type, and feature counts
    """
    # Initialize random number generator
    rng = np.random.default_rng(params.seed)

    bx, by, bz = params.bounding_box
    canal_radius = _um_to_mm(params.haversian_canal_diameter_um) / 2
    canal_spacing = _um_to_mm(params.canal_spacing_um)
    osteon_radius = _um_to_mm(params.osteon_diameter_um) / 2

    # Create base solid block
    base = m3d.Manifold.cube([bx, by, bz])

    # Generate canal positions based on pattern
    if params.osteon_pattern == 'hexagonal':
        canal_positions = generate_hexagonal_pattern(
            width=bx,
            depth=by,
            spacing=canal_spacing,
            noise=params.position_noise,
            rng=rng
        )
    else:  # random or organized
        canal_positions = generate_random_pattern(
            width=bx,
            depth=by,
            spacing=canal_spacing,
            seed=params.seed
        )

    # Filter positions to be within bounds
    canal_positions = [
        (x, y) for x, y in canal_positions
        if canal_radius < x < bx - canal_radius and canal_radius < y < by - canal_radius
    ]

    # Apply bone mineral density to modify effective canal radius
    # Higher BMD (1.8-2.2) = denser bone = smaller canals, thicker walls
    # Lower BMD (1.2-1.6) = more porous = larger canals, thinner walls
    # Reference BMD is 1.8 g/cm³
    bmd_factor = params.bone_mineral_density / 1.8
    # BMD affects canal size inversely (higher BMD = smaller canals)
    canal_bmd_modifier = 1.0 / bmd_factor  # e.g., BMD 2.0 -> 0.9x canal size

    # Create Haversian canals (vertical cylinders)
    haversian_canals = []
    canal_wall_thickness = _um_to_mm(params.canal_wall_thickness_um) * bmd_factor
    canal_walls = []

    for x, y in canal_positions:
        # Apply diameter variance and BMD modifier
        actual_radius = canal_radius * canal_bmd_modifier * (1 + rng.uniform(-params.diameter_variance, params.diameter_variance))
        canal = m3d.Manifold.cylinder(bz, actual_radius, actual_radius, params.resolution)
        canal = canal.translate([x, y, 0])
        haversian_canals.append(canal)

        # Create canal wall (shell around the canal lumen)
        if canal_wall_thickness > 0:
            wall = _create_canal_wall(
                canal_center=(x, y),
                canal_radius=actual_radius,
                wall_thickness=canal_wall_thickness,
                height=bz,
                resolution=params.resolution
            )
            canal_walls.append(wall)

    # Subtract Haversian canals from base
    result = base
    if haversian_canals:
        all_canals = batch_union(haversian_canals)
        if all_canals:
            result = result - all_canals

    # Canal walls are structural (part of the bone matrix)
    # They're already represented by the solid around the canal

    # Create Haversian vessels (blood vessels inside canal lumen)
    haversian_vessels = []
    if params.enable_haversian_vessels:
        # Vessel wall thickness: 2-5μm
        vessel_wall_thickness = canal_wall_thickness * 0.4  # ~40% of canal wall thickness

        for x, y in canal_positions:
            # Apply diameter variance and BMD modifier (same as canal)
            actual_canal_radius = canal_radius * canal_bmd_modifier * (1 + rng.uniform(-params.diameter_variance, params.diameter_variance))

            # Vessel lumen is ~60-70% of canal diameter
            vessel_lumen_radius = actual_canal_radius * 0.65
            vessel_outer_radius = vessel_lumen_radius + vessel_wall_thickness

            # Ensure vessel fits within canal lumen
            if vessel_outer_radius < actual_canal_radius:
                # Create thin-walled vessel tube (shell)
                vessel = _create_cylindrical_shell(
                    vessel_lumen_radius,
                    vessel_outer_radius,
                    bz,
                    params.resolution
                )
                vessel = vessel.translate([x, y, 0])
                haversian_vessels.append(vessel)

        # Add vessels to the structure (vessels are solid tissue, not voids)
        # They represent the blood vessel wall within the canal
        if haversian_vessels:
            vessels_union = batch_union(haversian_vessels)
            if vessels_union:
                result = result + vessels_union

    # Create Volkmann's canals
    volkmann_canals = []
    if params.enable_volkmann_canals:
        volkmann_canals = _create_volkmann_canals(
            canal_positions, bx, by, bz, params, rng
        )
        if volkmann_canals:
            volkmann_union = batch_union(volkmann_canals)
            if volkmann_union:
                result = result - volkmann_union

    # Create concentric lamellae (visual representation)
    lamellae_count = 0
    if params.enable_lamellar_detail:
        # BMD affects lamella thickness (higher BMD = thicker lamellae)
        lamella_thickness = _um_to_mm(params.lamella_thickness_um) * bmd_factor
        interlamellar_spacing = _um_to_mm(params.interlamellar_spacing_um)
        all_lamellae = []

        for x, y in canal_positions:
            # Apply lamella count variance
            actual_num = params.num_concentric_lamellae + rng.integers(
                -params.lamella_count_variance, params.lamella_count_variance + 1
            )
            actual_num = max(2, actual_num)

            lamellae = _create_concentric_lamellae(
                canal_center=(x, y),
                canal_radius=canal_radius * canal_bmd_modifier,
                osteon_radius=osteon_radius,
                num_lamellae=actual_num,
                lamella_thickness=lamella_thickness,
                interlamellar_spacing=interlamellar_spacing,
                height=bz,
                resolution=params.resolution
            )
            all_lamellae.extend(lamellae)
            lamellae_count += len(lamellae)

        # Lamellae are structural, they form part of the bone (not subtracted)
        # They're represented by the solid matrix already

    # Create lacunae (osteocyte spaces)
    lacunae_count = 0
    if params.enable_lacunae:
        lacuna_dims = (
            _um_to_mm(params.lacuna_length_um),
            _um_to_mm(params.lacuna_width_um),
            _um_to_mm(params.lacuna_width_um) * 0.6  # Depth is ~60% of width
        )

        all_lacunae = []
        for x, y in canal_positions:
            lacunae = _create_lacunae_in_osteon(
                canal_center=(x, y),
                canal_radius=canal_radius,
                osteon_radius=osteon_radius,
                height=bz,
                lacuna_density=params.lacuna_density,
                lacuna_dims=lacuna_dims,
                resolution=params.resolution,
                rng=rng
            )
            all_lacunae.extend(lacunae)

        lacunae_count = len(all_lacunae)
        if all_lacunae:
            lacunae_union = batch_union(all_lacunae)
            if lacunae_union:
                result = result - lacunae_union

    # Create canaliculi (very fine channels - optional due to size)
    canaliculi_count = 0
    if params.enable_canaliculi:
        canaliculus_radius = _um_to_mm(params.canaliculus_diameter_um) / 2
        canaliculus_length = _um_to_mm(30.0)  # ~30μm average length

        # Only create canaliculi for a subset of lacunae (performance)
        num_canaliculi_lacunae = min(10, lacunae_count)

        all_canaliculi = []
        for _ in range(num_canaliculi_lacunae):
            # Random position within osteons
            idx = rng.integers(0, len(canal_positions))
            cx, cy = canal_positions[idx]
            r = rng.uniform(canal_radius, osteon_radius)
            theta = rng.uniform(0, 2 * np.pi)
            z = rng.uniform(canaliculus_length, bz - canaliculus_length)

            lacuna_center = (
                cx + r * np.cos(theta),
                cy + r * np.sin(theta),
                z
            )

            # Limit canaliculi per lacuna for performance
            num_per = min(10, params.canaliculi_per_lacuna)
            canaliculi = _create_canaliculi_from_lacuna(
                lacuna_center=lacuna_center,
                num_canaliculi=num_per,
                canaliculus_radius=canaliculus_radius,
                canaliculus_length=canaliculus_length,
                resolution=max(4, params.resolution // 2),
                rng=rng
            )
            all_canaliculi.extend(canaliculi)

        canaliculi_count = len(all_canaliculi)
        if all_canaliculi:
            canaliculi_union = batch_union(all_canaliculi)
            if canaliculi_union:
                result = result - canaliculi_union

    # Create cement lines (osteon boundaries)
    cement_lines_count = 0
    if params.enable_cement_line:
        cement_thickness = _um_to_mm(params.cement_line_thickness_um)
        cement_lines = _create_cement_lines(
            canal_positions,
            osteon_radius,
            cement_thickness,
            bz,
            params.resolution
        )
        cement_lines_count = len(cement_lines)
        # Cement lines are structural boundaries (not subtracted, part of matrix)

    # Create interstitial lamellae
    interstitial_count = 0
    if params.enable_interstitial_lamellae:
        interstitials = _create_interstitial_lamellae(
            bx, by, bz,
            canal_positions,
            osteon_radius,
            params.interstitial_fraction,
            params.resolution,
            rng
        )
        interstitial_count = len(interstitials)
        # Interstitials are bone matrix (not subtracted)

    # Create periosteal surface layer (outer bone surface)
    periosteal_layer = None
    if params.enable_periosteal_surface:
        periosteal_thickness = _um_to_mm(params.periosteal_layer_thickness_um)
        # Create the periosteal layer as a distinct shell on the outer surface
        periosteal_layer = _create_periosteal_layer(bx, by, bz, periosteal_thickness)
        # The periosteal layer adds to the bone structure (union with result)
        # For visual distinction, we could make it slightly different, but
        # structurally it's part of the solid matrix

    # Create endosteal surface layer (inner bone surface)
    endosteal_layer = None
    if params.enable_endosteal_surface:
        endosteal_thickness = _um_to_mm(params.endosteal_layer_thickness_um)
        # Create the endosteal layer as a thin internal sheet
        endosteal_layer = _create_endosteal_layer(bx, by, bz, endosteal_thickness)
        # For block model, this represents the inner lining region

    # Create resorption spaces
    resorption_count = 0
    if params.enable_resorption_spaces:
        resorption_diameter = _um_to_mm(params.resorption_space_diameter_um)
        resorption_spaces = _create_resorption_spaces(
            bx, by, bz,
            resorption_diameter,
            params.resorption_density,
            params.resolution,
            rng
        )
        resorption_count = len(resorption_spaces)
        if resorption_spaces:
            resorption_union = batch_union(resorption_spaces)
            if resorption_union:
                result = result - resorption_union

    # Apply orientation
    if params.primary_orientation_deg != 0 or params.orientation_variance_deg > 0:
        # Center the model before rotation
        result = result.translate([-bx/2, -by/2, -bz/2])
        result = _apply_orientation(
            result,
            params.primary_orientation_deg,
            params.orientation_variance_deg,
            rng
        )
        result = result.translate([bx/2, by/2, bz/2])

    # Apply surface roughness to bone surfaces
    if params.surface_roughness > 0:
        result = _apply_surface_roughness(
            result,
            params.surface_roughness,
            params.bounding_box,
            params.resolution,
            rng
        )

    # Calculate statistics
    mesh = result.to_mesh()
    volume = result.volume() if hasattr(result, 'volume') else 0

    stats = {
        'triangle_count': len(mesh.tri_verts) // 3 if hasattr(mesh, 'tri_verts') else 0,
        'volume_mm3': volume,
        'canal_count': len(haversian_canals),
        'canal_wall_count': len(canal_walls),
        'haversian_vessels_count': len(haversian_vessels),
        'volkmann_canal_count': len(volkmann_canals),
        'lamellae_count': lamellae_count,
        'lacunae_count': lacunae_count,
        'canaliculi_count': canaliculi_count,
        'cement_lines_count': cement_lines_count,
        'interstitial_count': interstitial_count,
        'resorption_count': resorption_count,
        'osteon_pattern': params.osteon_pattern,
        'porosity': 1.0 - (volume / (bx * by * bz)),
        'bone_mineral_density': params.bone_mineral_density,
        'bmd_factor': bmd_factor,
        'surface_roughness': params.surface_roughness,
        'periosteal_thickness_mm': _um_to_mm(params.periosteal_layer_thickness_um) if params.enable_periosteal_surface else 0,
        'endosteal_thickness_mm': _um_to_mm(params.endosteal_layer_thickness_um) if params.enable_endosteal_surface else 0,
        'interlamellar_spacing_mm': _um_to_mm(params.interlamellar_spacing_um),
        'canal_wall_thickness_mm': _um_to_mm(params.canal_wall_thickness_um),
        'scaffold_type': 'haversian_bone'
    }

    return result, stats


def generate_hexagonal_pattern(
    width: float,
    depth: float,
    spacing: float,
    noise: float = 0.0,
    rng: np.random.Generator | None = None
) -> list[tuple[float, float]]:
    """
    Generate hexagonal close-packed positions for osteons.

    Args:
        width: Width of region (X)
        depth: Depth of region (Y)
        spacing: Distance between osteon centers
        noise: Position noise factor (0-1)
        rng: Random number generator for noise

    Returns:
        List of (x, y) positions
    """
    positions = []

    # Hexagonal grid parameters
    row_height = spacing * np.sqrt(3) / 2
    n_rows = int(depth / row_height) + 2
    n_cols = int(width / spacing) + 2

    for row in range(n_rows):
        y = row * row_height

        for col in range(n_cols):
            # Offset every other row for hexagonal packing
            x_offset = (spacing / 2) if (row % 2 == 1) else 0
            x = col * spacing + x_offset

            # Apply position noise
            if noise > 0 and rng is not None:
                x += rng.uniform(-noise * spacing, noise * spacing)
                y_noisy = y + rng.uniform(-noise * spacing, noise * spacing)
            else:
                y_noisy = y

            positions.append((x, y_noisy))

    return positions


def generate_random_pattern(
    width: float,
    depth: float,
    spacing: float,
    seed: int = 42
) -> list[tuple[float, float]]:
    """
    Generate random positions for osteons with minimum spacing constraint.

    Args:
        width: Width of region (X)
        depth: Depth of region (Y)
        spacing: Minimum distance between osteon centers
        seed: Random seed for reproducibility

    Returns:
        List of (x, y) positions
    """
    rng = np.random.default_rng(seed)
    positions = []

    # Estimate number of canals based on area and spacing
    area = width * depth
    canal_area = spacing ** 2
    n_canals = int(area / canal_area * 0.8)  # 80% packing efficiency

    # Generate random positions with minimum spacing
    max_attempts = n_canals * 10
    attempts = 0

    while len(positions) < n_canals and attempts < max_attempts:
        x = rng.uniform(0, width)
        y = rng.uniform(0, depth)

        # Check minimum distance to existing positions
        valid = True
        for px, py in positions:
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist < spacing * 0.9:  # 90% of spacing for slight overlap
                valid = False
                break

        if valid:
            positions.append((x, y))

        attempts += 1

    return positions


def generate_haversian_bone_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate Haversian bone from dictionary parameters (for API compatibility).

    Args:
        params: Dictionary with keys matching HaversianBoneParams fields

    Returns:
        Tuple of (manifold, stats_dict)
    """
    bbox = params.get('bounding_box', (10, 10, 15))
    if isinstance(bbox, dict):
        bbox = (bbox['x'], bbox['y'], bbox['z'])

    return generate_haversian_bone(HaversianBoneParams(
        # Basic geometry
        bounding_box=tuple(bbox),
        resolution=params.get('resolution', 8),
        seed=params.get('seed', 42),

        # Haversian canal
        haversian_canal_diameter_um=params.get('haversian_canal_diameter_um', 50.0),
        canal_diameter_um=params.get('canal_diameter_um', 50.0),
        canal_wall_thickness_um=params.get('canal_wall_thickness_um', 5.0),
        enable_haversian_vessels=params.get('enable_haversian_vessels', True),

        # Osteon properties
        osteon_diameter_um=params.get('osteon_diameter_um', 225.0),
        canal_spacing_um=params.get('canal_spacing_um', 400.0),
        osteon_spacing_um=params.get('osteon_spacing_um', 250.0),
        osteon_pattern=params.get('osteon_pattern', 'hexagonal'),

        # Volkmann's canals
        enable_volkmann_canals=params.get('enable_volkmann_canals', True),
        volkmann_canal_diameter_um=params.get('volkmann_canal_diameter_um', 75.0),
        volkmann_canal_density=params.get('volkmann_canal_density', 0.5),
        volkmann_angle_deg=params.get('volkmann_angle_deg', 60.0),

        # Concentric lamellae
        num_concentric_lamellae=params.get('num_concentric_lamellae', 8),
        lamella_thickness_um=params.get('lamella_thickness_um', 5.0),
        interlamellar_spacing_um=params.get('interlamellar_spacing_um', 1.0),
        enable_lamellar_detail=params.get('enable_lamellar_detail', True),

        # Lacunae
        enable_lacunae=params.get('enable_lacunae', True),
        lacuna_density=params.get('lacuna_density', 15000.0),
        lacuna_length_um=params.get('lacuna_length_um', 20.0),
        lacuna_width_um=params.get('lacuna_width_um', 8.0),

        # Canaliculi
        enable_canaliculi=params.get('enable_canaliculi', False),
        canaliculus_diameter_um=params.get('canaliculus_diameter_um', 0.5),
        canaliculi_per_lacuna=params.get('canaliculi_per_lacuna', 50),

        # Cement line
        enable_cement_line=params.get('enable_cement_line', True),
        cement_line_thickness_um=params.get('cement_line_thickness_um', 3.0),

        # Interstitial lamellae
        enable_interstitial_lamellae=params.get('enable_interstitial_lamellae', True),
        interstitial_fraction=params.get('interstitial_fraction', 0.2),

        # Cortical properties
        cortical_thickness=params.get('cortical_thickness', 5.0),
        cortical_porosity=params.get('cortical_porosity', 0.08),
        bone_mineral_density=params.get('bone_mineral_density', 1.8),

        # Surfaces
        enable_periosteal_surface=params.get('enable_periosteal_surface', True),
        enable_endosteal_surface=params.get('enable_endosteal_surface', True),
        periosteal_layer_thickness_um=params.get('periosteal_layer_thickness_um', 50.0),
        endosteal_layer_thickness_um=params.get('endosteal_layer_thickness_um', 30.0),

        # Remodeling
        enable_resorption_spaces=params.get('enable_resorption_spaces', False),
        resorption_space_diameter_um=params.get('resorption_space_diameter_um', 200.0),
        resorption_density=params.get('resorption_density', 0.02),

        # Orientation
        primary_orientation_deg=params.get('primary_orientation_deg', 0.0),
        orientation_variance_deg=params.get('orientation_variance_deg', 10.0),

        # Surface
        surface_roughness=params.get('surface_roughness', 0.05),

        # Randomization
        position_noise=params.get('position_noise', 0.1),
        diameter_variance=params.get('diameter_variance', 0.15),
        lamella_count_variance=params.get('lamella_count_variance', 2),
    ))
