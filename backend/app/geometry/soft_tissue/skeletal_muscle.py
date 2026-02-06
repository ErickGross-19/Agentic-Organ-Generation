"""
Skeletal muscle scaffold generator with aligned myofiber channels.

Provides parametric generation of muscle tissue scaffolds with:
- Parallel, unipennate, or bipennate fiber architectures
- Aligned myofiber channels (10-100um diameter)
- Configurable pennation angle (0-30 deg)
- Multiple fascicle groupings with connective tissue layers
- Customizable fiber spacing and density
- Vascular network with capillaries, arterioles, and venules
- Motor endplate markers
- Fiber curvature/waviness
"""

from __future__ import annotations
import manifold3d as m3d
import numpy as np
from dataclasses import dataclass
from typing import Literal, List, Tuple, Optional
from ..core import batch_union


@dataclass
class SkeletalMuscleParams:
    """
    Parameters for skeletal muscle scaffold generation.

    Biologically realistic parameters based on human skeletal muscle histology:
    - Muscle fibers: 10-100 um diameter (type I: 10-80 um, type II: 50-100 um)
    - Fascicles: 1-10 mm diameter bundles of fibers
    - Sarcomere: 2.0-2.5 um length (functional contractile unit)
    - Connective tissue layers: endomysium, perimysium, epimysium
    """
    # === Basic Geometry ===
    length_mm: float = 30.0  # Scaffold length (longer for proper fiber alignment)
    width_mm: float = 20.0   # Scaffold width (accommodates multiple fascicles)
    height_mm: float = 15.0  # Scaffold height (accommodates multiple fascicles)

    # === Muscle Fiber Parameters ===
    fiber_diameter_um: float = 50.0  # 10-100 um typical (type I: 10-80, type II: 50-100)
    fiber_spacing_um: float = 100.0  # Center-to-center spacing
    fiber_length_mm: float = 15.0    # Individual fiber length (can span entire muscle)

    # === Fascicle Parameters ===
    fascicle_diameter_mm: float = 2.0  # 1-10 mm bundles (default 2mm to fit in 5mm height)
    fascicle_count: int = 4            # Number of fascicles
    fascicle_spacing_mm: float = 0.5   # Gap between fascicles

    # === Sarcomere Parameters (for textural accuracy) ===
    sarcomere_length_um: float = 2.5  # 2.0-2.5 um at rest
    sarcomere_resolution: int = 4  # STATS-ONLY: Not used in scaffold geometry.
    # Sarcomere length (2 µm) is below 3D printing resolution (50-200 µm min feature).
    # This parameter is retained for simulation coupling (FEA) and reference only.

    # === Fiber Architecture ===
    architecture_type: Literal['parallel', 'unipennate', 'bipennate', 'multipennate', 'circular', 'convergent'] = 'parallel'
    pennation_angle_deg: float = 15.0  # 0-30 deg typical (unipennate/bipennate)
    fiber_alignment: float = 0.6       # 0=random, 1=perfectly aligned
    fiber_curvature: float = 0.0       # 0=straight, 1=curved (for curved fibers)

    # === Connective Tissue Layers ===
    # Endomysium (around individual fibers)
    # STATS-ONLY: Endomysium at 0.5 µm is too thin to model in scaffold geometry.
    # Function create_endomysium_shell exists but returns None (see line ~490).
    # These parameters are retained for biological reference and stats output.
    endomysium_thickness_um: float = 0.5  # ~0.2-1.0 µm typical thickness
    endomysium_porosity: float = 0.3  # 0.35-0.55 typical, used for reference only

    # Perimysium (around fascicles)
    perimysium_thickness_um: float = 20.0  # 10-50 um
    perimysium_porosity: float = 0.25
    enable_perimysium: bool = True

    # Epimysium (outer layer around whole muscle)
    epimysium_thickness_um: float = 100.0  # 50-200 um
    enable_epimysium: bool = True

    # === Mechanical Properties (for reference) ===
    contraction_force_uN: float = 443.0  # uN per fiber typical

    # === Vascular Network ===
    enable_vascular_channels: bool = True
    capillary_diameter_um: float = 8.0     # ~8 um
    capillary_density_per_mm2: float = 300  # 200-500/mm2 typical
    arteriole_diameter_um: float = 50.0
    venule_diameter_um: float = 60.0

    # === Neuromuscular Features ===
    enable_motor_endplates: bool = False
    motor_endplate_density_per_mm2: float = 0.5  # Sparse distribution

    # === Randomization ===
    seed: int = 42
    position_noise: float = 0.1  # Positional variation
    diameter_variance: float = 0.1  # Fiber diameter variation

    # === Resolution ===
    resolution: int = 8


def make_fiber_channel(
    start_point: np.ndarray,
    end_point: np.ndarray,
    radius: float,
    resolution: int
) -> m3d.Manifold:
    """
    Create a cylindrical fiber channel between two points.

    Args:
        start_point: Starting point (x, y, z)
        end_point: Ending point (x, y, z)
        radius: Fiber radius
        resolution: Number of segments around cylinder

    Returns:
        Manifold representing the fiber channel
    """
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    dz = end_point[2] - start_point[2]
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

    return fiber.translate([start_point[0], start_point[1], start_point[2]])


def make_curved_fiber_channel(
    start_point: np.ndarray,
    end_point: np.ndarray,
    radius: float,
    curvature: float,
    resolution: int,
    rng: np.random.Generator
) -> Tuple[m3d.Manifold, np.ndarray]:
    """
    Create a curved fiber channel with sinusoidal waviness.

    Args:
        start_point: Starting point (x, y, z)
        end_point: Ending point (x, y, z)
        radius: Fiber radius
        curvature: Curvature factor (0-1), 0=straight, 1=wavy
        resolution: Number of segments around cylinder
        rng: Random number generator

    Returns:
        Tuple of (Manifold representing the curved fiber, midpoint array)
    """
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]
    dz = end_point[2] - start_point[2]
    length = np.sqrt(dx*dx + dy*dy + dz*dz)

    if length < 1e-6 or curvature < 0.01:
        # Fall back to straight fiber
        fiber = make_fiber_channel(start_point, end_point, radius, resolution)
        midpoint = (start_point + end_point) / 2
        return fiber, midpoint

    # Calculate waviness parameters
    # Amplitude = curvature * fiber_diameter * 0.15 (using radius * 2)
    amplitude = curvature * (radius * 2) * 0.15
    # Wavelength = 500-2000 um, pick randomly within range
    wavelength_mm = rng.uniform(0.5, 2.0)  # Convert um to mm

    # Number of segments based on length and wavelength
    num_segments = max(8, int(length / (wavelength_mm / 4)))

    # Direction vector (normalized)
    direction = np.array([dx, dy, dz]) / length

    # Create perpendicular vectors for displacement
    if abs(direction[2]) < 0.9:
        perp1 = np.cross(direction, np.array([0, 0, 1]))
    else:
        perp1 = np.cross(direction, np.array([1, 0, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)

    # Random phase offset
    phase = rng.uniform(0, 2 * np.pi)

    # Generate points along the curved path
    segments = []
    points = []
    for i in range(num_segments + 1):
        t = i / num_segments
        # Base position along the straight line
        base_pos = start_point + t * length * direction

        # Sinusoidal displacement
        wave_pos = t * length / wavelength_mm * 2 * np.pi + phase
        displacement = amplitude * np.sin(wave_pos)

        # Apply displacement perpendicular to fiber direction
        pos = base_pos + displacement * perp1
        points.append(pos)

    # Create cylinder segments between consecutive points
    for i in range(len(points) - 1):
        seg = make_fiber_channel(points[i], points[i + 1], radius, resolution)
        if seg.num_vert() > 0:
            segments.append(seg)
        # Add junction spheres for smooth connections
        if i > 0:
            sphere = m3d.Manifold.sphere(radius * 0.99, resolution)
            sphere = sphere.translate([points[i][0], points[i][1], points[i][2]])
            segments.append(sphere)

    if not segments:
        fiber = make_fiber_channel(start_point, end_point, radius, resolution)
        midpoint = (start_point + end_point) / 2
        return fiber, midpoint

    # Union all segments
    result = batch_union(segments)
    midpoint = points[len(points) // 2]

    return result, midpoint


def get_parallel_fiber_positions(
    width: float,
    height: float,
    spacing: float
) -> list[tuple[float, float]]:
    """
    Get grid positions for parallel fibers.

    Args:
        width: Width of muscle
        height: Height of muscle
        spacing: Spacing between fibers

    Returns:
        List of (y, z) positions for fiber centers
    """
    positions = []
    n_y = max(1, int(width / spacing))
    n_z = max(1, int(height / spacing))

    for i in range(n_y):
        for j in range(n_z):
            y = (i - n_y / 2 + 0.5) * spacing
            z = (j - n_z / 2 + 0.5) * spacing
            positions.append((y, z))

    return positions


def get_fascicle_positions(
    width: float,
    height: float,
    fascicle_count: int,
    fascicle_diameter: float,
    fascicle_spacing: float
) -> List[Tuple[float, float]]:
    """
    Calculate fascicle center positions arranged in a grid/hexagonal pattern.

    Args:
        width: Muscle width in mm
        height: Muscle height in mm
        fascicle_count: Number of fascicles to place
        fascicle_diameter: Diameter of each fascicle in mm
        fascicle_spacing: Spacing between fascicles in mm

    Returns:
        List of (y, z) positions for fascicle centers
    """
    positions = []

    # Calculate how many fascicles fit in each dimension
    effective_diameter = fascicle_diameter + fascicle_spacing
    n_y = max(1, int(width / effective_diameter))
    n_z = max(1, int(height / effective_diameter))

    # Limit to requested count
    count = 0
    for i in range(n_z):
        for j in range(n_y):
            if count >= fascicle_count:
                break
            y = (j - (n_y - 1) / 2) * effective_diameter
            z = (i - (n_z - 1) / 2) * effective_diameter

            # Ensure fascicle fits within bounds
            if abs(y) + fascicle_diameter / 2 <= width / 2 and \
               abs(z) + fascicle_diameter / 2 <= height / 2:
                positions.append((y, z))
                count += 1
        if count >= fascicle_count:
            break

    # If we couldn't place all fascicles in grid, use circular arrangement with bounds check
    if len(positions) < fascicle_count:
        positions = []
        # Calculate max radius that keeps fascicles within bounds
        max_r = min(width / 2 - fascicle_diameter / 2, height / 2 - fascicle_diameter / 2)
        max_r = max(0, max_r)  # Ensure non-negative

        # If fascicles are too big to fit, place them at center
        if max_r < 0.1:
            # Single fascicle at center or scaled arrangement
            r = 0
        else:
            r = max_r * 0.8  # Use 80% of max radius for some margin

        for i in range(fascicle_count):
            angle = 2 * np.pi * i / fascicle_count
            y = r * np.cos(angle)
            z = r * np.sin(angle)
            positions.append((y, z))

    return positions[:fascicle_count]


def get_fibers_in_fascicle(
    fascicle_center_y: float,
    fascicle_center_z: float,
    fascicle_diameter: float,
    fiber_spacing: float,
    min_fibers: int = 20,
    max_fibers: int = 80,
    rng: Optional[np.random.Generator] = None
) -> List[Tuple[float, float]]:
    """
    Get fiber positions within a single fascicle.

    Args:
        fascicle_center_y: Y coordinate of fascicle center
        fascicle_center_z: Z coordinate of fascicle center
        fascicle_diameter: Diameter of fascicle in mm
        fiber_spacing: Spacing between fibers in mm
        min_fibers: Minimum fibers per fascicle
        max_fibers: Maximum fibers per fascicle
        rng: Random number generator

    Returns:
        List of (y, z) positions for fiber centers
    """
    positions = []
    fascicle_radius = fascicle_diameter / 2

    # Calculate grid of fibers within circular fascicle
    n_across = max(1, int(fascicle_diameter / fiber_spacing))
    start_offset = -fascicle_radius + fiber_spacing / 2

    for i in range(n_across):
        for j in range(n_across):
            local_y = start_offset + i * fiber_spacing
            local_z = start_offset + j * fiber_spacing

            # Check if within circular boundary
            dist = np.sqrt(local_y**2 + local_z**2)
            if dist < fascicle_radius * 0.9:  # 90% of radius to leave edge space
                y = fascicle_center_y + local_y
                z = fascicle_center_z + local_z
                positions.append((y, z))

    # Ensure we have between min_fibers and max_fibers
    if len(positions) < min_fibers and rng is not None:
        # Add more fibers randomly
        while len(positions) < min_fibers:
            angle = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(0, fascicle_radius * 0.85)
            y = fascicle_center_y + r * np.cos(angle)
            z = fascicle_center_z + r * np.sin(angle)
            # Check it's not too close to existing fibers
            too_close = False
            for py, pz in positions:
                if np.sqrt((y - py)**2 + (z - pz)**2) < fiber_spacing * 0.5:
                    too_close = True
                    break
            if not too_close:
                positions.append((y, z))

    if len(positions) > max_fibers:
        # Randomly select max_fibers
        if rng is not None:
            indices = rng.choice(len(positions), max_fibers, replace=False)
            positions = [positions[i] for i in indices]
        else:
            positions = positions[:max_fibers]

    return positions


def get_pennate_fiber_endpoints(
    fiber_y: float,
    fiber_z: float,
    length: float,
    pennation_angle: float,
    architecture: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate fiber endpoints for pennate architecture.

    Args:
        fiber_y: Y position of fiber center
        fiber_z: Z position of fiber center
        length: Muscle length
        pennation_angle: Pennation angle in radians
        architecture: 'parallel', 'unipennate', or 'bipennate'

    Returns:
        Tuple of (start_point, end_point) as numpy arrays
    """
    if architecture == 'parallel':
        # Simple parallel fibers along X axis
        start = np.array([0, fiber_y, fiber_z])
        end = np.array([length, fiber_y, fiber_z])
    elif architecture == 'unipennate':
        # Fibers angled relative to central axis
        dx = length * np.cos(pennation_angle)
        dy = length * np.sin(pennation_angle)
        start = np.array([0, fiber_y - dy/2, fiber_z])
        end = np.array([dx, fiber_y + dy/2, fiber_z])
    else:  # bipennate
        # Alternating fiber angles creating V-pattern
        # Use fiber_y to determine which side
        sign = 1 if fiber_y >= 0 else -1
        dx = length * np.cos(pennation_angle)
        dy = sign * length * np.sin(pennation_angle)
        start = np.array([0, fiber_y - dy/2, fiber_z])
        end = np.array([dx, fiber_y + dy/2, fiber_z])

    return start, end


def apply_alignment_variance(
    start: np.ndarray,
    end: np.ndarray,
    fiber_alignment: float,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply alignment variance to fiber endpoints based on alignment parameter.

    Args:
        start: Starting point of fiber
        end: Ending point of fiber
        fiber_alignment: 0.0 = random orientation (±90°), 1.0 = perfect alignment (0°)
        rng: Random number generator

    Returns:
        Tuple of (rotated_start, rotated_end) as numpy arrays
    """
    if fiber_alignment >= 0.99:
        return start, end  # Perfect alignment, no change

    # Max variance: 90° at alignment=0, 0° at alignment=1
    max_variance_rad = np.radians(90.0 * (1.0 - fiber_alignment))

    # Random rotation angle
    rotation_angle = rng.uniform(-max_variance_rad, max_variance_rad)

    # Calculate midpoint and rotate endpoints around it
    midpoint = (start + end) / 2
    cos_a, sin_a = np.cos(rotation_angle), np.sin(rotation_angle)

    # Rotate in XY plane (horizontal variance)
    def rotate_point(p, center):
        rel = p - center
        rotated = np.array([
            rel[0] * cos_a - rel[1] * sin_a,
            rel[0] * sin_a + rel[1] * cos_a,
            rel[2]
        ])
        return center + rotated

    return rotate_point(start, midpoint), rotate_point(end, midpoint)


def create_endomysium_shell(
    fiber_manifold: m3d.Manifold,
    fiber_radius: float,
    thickness_mm: float,
    resolution: int
) -> Optional[m3d.Manifold]:
    """
    Create endomysium (thin connective tissue shell around a fiber).

    NOTE: This function intentionally returns None because endomysium
    is too thin (~0.5 µm) to practically model in 3D printed scaffolds
    (minimum feature size 50-200 µm). The endomysium parameters are
    retained in SkeletalMuscleParams for biological reference and stats
    output only.

    In native tissue, endomysium:
    - Provides structural support and maintains fiber spacing
    - Regulates transport of nutrients, oxygen, and myokines
    - Has porosity of 0.35-0.55 for fluid transport

    Args:
        fiber_manifold: The fiber geometry to wrap
        fiber_radius: Radius of the fiber in mm
        thickness_mm: Thickness of endomysium in mm
        resolution: Mesh resolution

    Returns:
        None (see note above). Would return thin shell if modeled.
    """
    if thickness_mm < 0.0001:  # Skip if too thin to model
        return None

    # Create outer shell by scaling the fiber slightly
    # This is a simplified approach; true offset would require more complex operations
    outer_radius = fiber_radius + thickness_mm
    scale_factor = outer_radius / fiber_radius if fiber_radius > 0 else 1.0

    # For now, we return None as endomysium is too thin to practically model
    # In production, this could be represented in stats or as a texture
    return None


def create_perimysium_shell(
    fascicle_center_y: float,
    fascicle_center_z: float,
    fascicle_diameter: float,
    muscle_length: float,
    thickness_mm: float,
    porosity: float,
    resolution: int,
    rng: np.random.Generator
) -> m3d.Manifold:
    """
    Create perimysium shell around a fascicle.

    Args:
        fascicle_center_y: Y position of fascicle center
        fascicle_center_z: Z position of fascicle center
        fascicle_diameter: Diameter of the fascicle in mm
        muscle_length: Length of the muscle in mm
        thickness_mm: Thickness of perimysium in mm
        porosity: Porosity value (0-1) controlling pore density
        resolution: Mesh resolution
        rng: Random number generator for pore placement

    Returns:
        Perimysium shell manifold (hollow cylinder with pores)
    """
    outer_radius = fascicle_diameter / 2 + thickness_mm
    inner_radius = fascicle_diameter / 2

    # Create outer cylinder
    outer = m3d.Manifold.cylinder(muscle_length, outer_radius, outer_radius, resolution * 2)

    # Create inner cylinder (slightly longer for clean subtraction)
    inner = m3d.Manifold.cylinder(muscle_length + 0.2, inner_radius, inner_radius, resolution * 2)
    inner = inner.translate([0, 0, -0.1])

    # Create shell
    shell = outer - inner

    # Add pores to perimysium based on porosity
    if porosity > 0.05:
        pores = []
        # Calculate pore size (5-20 um as per research)
        pore_diameter_mm = 0.015  # 15 um average
        pore_radius = pore_diameter_mm / 2

        # Calculate number of pores based on porosity and surface area
        # Surface area of cylinder ≈ 2 * pi * radius * length
        surface_area = 2 * np.pi * (fascicle_diameter / 2) * muscle_length
        pore_area = np.pi * pore_radius * pore_radius

        # Number of pores = porosity * surface_area / pore_area (with some randomness)
        n_pores = int(porosity * surface_area / pore_area * 0.3)  # 0.3 factor for practical limits
        n_pores = min(n_pores, 200)  # Cap for performance

        for _ in range(n_pores):
            # Random position along cylinder
            angle = rng.uniform(0, 2 * np.pi)
            z_pos = rng.uniform(0, muscle_length)

            # Pore at surface of perimysium
            x_offset = (fascicle_diameter / 2 + thickness_mm / 2) * np.cos(angle)
            y_offset = (fascicle_diameter / 2 + thickness_mm / 2) * np.sin(angle)

            # Create pore as small cylinder perpendicular to surface (radial direction)
            pore = m3d.Manifold.cylinder(
                thickness_mm * 1.5,  # Slightly longer to ensure clean subtraction
                pore_radius,
                pore_radius,
                6
            )

            # Rotate to point radially outward
            pore = pore.rotate([0, 0, np.degrees(angle)])
            pore = pore.rotate([0, 90, 0])

            # Position at pore location
            pore = pore.translate([x_offset, y_offset, z_pos])
            pores.append(pore)

        if pores:
            pores_union = batch_union(pores)
            shell = shell - pores_union

    # Rotate to align with X axis (fibers run along X)
    shell = shell.rotate([0, 90, 0])

    # Translate to fascicle center - shell runs from x=0 to x=length
    shell = shell.translate([0, fascicle_center_y, fascicle_center_z])

    return shell


def create_epimysium_shell(
    width: float,
    height: float,
    length: float,
    thickness_mm: float,
    resolution: int
) -> m3d.Manifold:
    """
    Create epimysium shell around entire muscle.

    Args:
        width: Muscle width in mm
        height: Muscle height in mm
        length: Muscle length in mm
        thickness_mm: Thickness of epimysium in mm
        resolution: Mesh resolution

    Returns:
        Epimysium shell manifold (outer shell of the muscle)
    """
    # Outer box - centered in Y and Z, from -thickness to length+thickness in X
    outer = m3d.Manifold.cube([
        length + 2 * thickness_mm,
        width + 2 * thickness_mm,
        height + 2 * thickness_mm
    ])
    outer = outer.translate([
        -thickness_mm,
        -(width + 2 * thickness_mm) / 2,
        -(height + 2 * thickness_mm) / 2
    ])

    # Inner box (the muscle volume itself) - centered in Y and Z
    inner = m3d.Manifold.cube([
        length + 0.2,
        width,
        height
    ])
    inner = inner.translate([
        -0.1,
        -width / 2,
        -height / 2
    ])

    # Create shell
    shell = outer - inner

    return shell


def create_capillaries_for_fiber(
    start_point: np.ndarray,
    end_point: np.ndarray,
    fiber_radius: float,
    capillary_radius: float,
    num_capillaries: int,
    resolution: int,
    rng: np.random.Generator
) -> List[m3d.Manifold]:
    """
    Create capillary channels running parallel to a muscle fiber.

    Args:
        start_point: Fiber start point
        end_point: Fiber end point
        fiber_radius: Radius of the muscle fiber
        capillary_radius: Radius of capillaries
        num_capillaries: Number of capillaries around this fiber
        resolution: Mesh resolution
        rng: Random number generator

    Returns:
        List of capillary manifolds
    """
    capillaries = []

    # Capillaries run parallel to fiber, offset by fiber radius + small gap
    offset_distance = fiber_radius + capillary_radius * 2

    # Direction vector
    direction = end_point - start_point
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return capillaries

    direction = direction / length

    # Create perpendicular vectors
    if abs(direction[2]) < 0.9:
        perp1 = np.cross(direction, np.array([0, 0, 1]))
    else:
        perp1 = np.cross(direction, np.array([1, 0, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)

    # Place capillaries around the fiber
    for i in range(num_capillaries):
        angle = 2 * np.pi * i / num_capillaries + rng.uniform(0, 0.3)

        # Offset position
        offset = offset_distance * (np.cos(angle) * perp1 + np.sin(angle) * perp2)

        cap_start = start_point + offset
        cap_end = end_point + offset

        # Create capillary channel
        cap = make_fiber_channel(cap_start, cap_end, capillary_radius, max(6, resolution // 2))
        if cap.num_vert() > 0:
            capillaries.append(cap)

    return capillaries


def create_arteriole_venule_pair(
    fascicle_center_y: float,
    fascicle_center_z: float,
    fascicle_diameter: float,
    muscle_length: float,
    arteriole_radius: float,
    venule_radius: float,
    resolution: int,
    rng: np.random.Generator
) -> Tuple[m3d.Manifold, m3d.Manifold]:
    """
    Create arteriole and venule pair in the perimysium space around a fascicle.

    Args:
        fascicle_center_y: Y position of fascicle center
        fascicle_center_z: Z position of fascicle center
        fascicle_diameter: Diameter of the fascicle
        muscle_length: Length of the muscle
        arteriole_radius: Radius of arteriole
        venule_radius: Radius of venule
        resolution: Mesh resolution
        rng: Random number generator

    Returns:
        Tuple of (arteriole manifold, venule manifold)
    """
    # Place arteriole and venule in perimysium space (just outside fascicle)
    offset = fascicle_diameter / 2 + max(arteriole_radius, venule_radius) * 2

    # Random angles for placement
    art_angle = rng.uniform(0, np.pi)
    ven_angle = art_angle + np.pi  # Opposite side

    # Arteriole position
    art_y = fascicle_center_y + offset * np.cos(art_angle)
    art_z = fascicle_center_z + offset * np.sin(art_angle)

    # Venule position
    ven_y = fascicle_center_y + offset * np.cos(ven_angle)
    ven_z = fascicle_center_z + offset * np.sin(ven_angle)

    # Create vessels running along muscle length
    art_start = np.array([0, art_y, art_z])
    art_end = np.array([muscle_length, art_y, art_z])

    ven_start = np.array([0, ven_y, ven_z])
    ven_end = np.array([muscle_length, ven_y, ven_z])

    arteriole = make_fiber_channel(art_start, art_end, arteriole_radius, resolution)
    venule = make_fiber_channel(ven_start, ven_end, venule_radius, resolution)

    return arteriole, venule


def create_motor_endplate(
    position: np.ndarray,
    diameter_um: float,
    resolution: int
) -> m3d.Manifold:
    """
    Create a motor endplate marker as a small ellipsoid.

    Args:
        position: (x, y, z) position of the motor endplate
        diameter_um: Diameter of endplate in micrometers
        resolution: Mesh resolution

    Returns:
        Ellipsoid manifold representing the motor endplate
    """
    # Convert um to mm
    radius_mm = diameter_um / 2000.0

    # Create sphere and scale to ellipsoid (flatter in one direction)
    sphere = m3d.Manifold.sphere(radius_mm, max(6, resolution))

    # Scale to make it ellipsoidal (pancake shape typical of motor endplates)
    # Motor endplates are flattened structures
    sphere = sphere.scale([1.5, 1.5, 0.5])

    # Translate to position
    sphere = sphere.translate([position[0], position[1], position[2]])

    return sphere


def generate_skeletal_muscle(params: SkeletalMuscleParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate a skeletal muscle scaffold with biologically realistic parameters.

    Args:
        params: SkeletalMuscleParams specifying geometry and structure

    Returns:
        Tuple of (manifold, stats_dict)
        - manifold: The 3D geometry
        - stats_dict: Dictionary with triangle_count, volume_mm3, fiber_count,
                     architecture_type, pennation_angle, scaffold_type

    Raises:
        ValueError: If parameters are invalid
    """
    # Set random seed for reproducibility
    rng = np.random.default_rng(params.seed)

    if params.fiber_diameter_um <= 0 or params.fiber_spacing_um <= 0:
        raise ValueError("Fiber diameter and spacing must be positive")
    if not 0 <= params.pennation_angle_deg <= 45:
        raise ValueError("Pennation angle must be between 0 and 45 degrees")

    # Convert micrometers to millimeters
    fiber_radius_mm = (params.fiber_diameter_um / 1000.0) / 2.0
    fiber_spacing_mm = params.fiber_spacing_um / 1000.0
    capillary_radius_mm = params.capillary_diameter_um / 2000.0
    arteriole_radius_mm = params.arteriole_diameter_um / 2000.0
    venule_radius_mm = params.venule_diameter_um / 2000.0
    perimysium_thickness_mm = params.perimysium_thickness_um / 1000.0
    epimysium_thickness_mm = params.epimysium_thickness_um / 1000.0

    pennation_angle_rad = np.radians(params.pennation_angle_deg)

    # Create bounding box (will be subtracted from to create channels)
    # Center the box at origin in Y and Z, extend from 0 to length in X
    bounding_box = m3d.Manifold.cube([
        params.length_mm,
        params.width_mm,
        params.height_mm
    ])
    bounding_box = bounding_box.translate([
        0,
        -params.width_mm / 2,
        -params.height_mm / 2
    ])

    # ===== FASCICLE ORGANIZATION =====
    fascicle_positions = get_fascicle_positions(
        params.width_mm,
        params.height_mm,
        params.fascicle_count,
        params.fascicle_diameter_mm,
        params.fascicle_spacing_mm
    )

    # Collect all geometry components
    fibers = []
    capillaries = []
    arterioles_venules = []
    perimysium_shells = []
    motor_endplates = []
    fiber_midpoints = []

    # Generate fibers grouped by fascicle
    total_fiber_count = 0
    for fascicle_idx, (fasc_y, fasc_z) in enumerate(fascicle_positions):
        # Get fiber positions within this fascicle
        fiber_positions = get_fibers_in_fascicle(
            fasc_y,
            fasc_z,
            params.fascicle_diameter_mm,
            fiber_spacing_mm,
            min_fibers=20,
            max_fibers=80,
            rng=rng
        )

        fascicle_fiber_count = 0

        for y_pos, z_pos in fiber_positions:
            # Add position noise if enabled
            actual_y = y_pos
            actual_z = z_pos
            if params.position_noise > 0:
                actual_y += rng.uniform(-1, 1) * params.position_noise * fiber_spacing_mm
                actual_z += rng.uniform(-1, 1) * params.position_noise * fiber_spacing_mm

            # Add diameter variance if enabled
            actual_radius = fiber_radius_mm
            if params.diameter_variance > 0:
                actual_radius *= (1 + rng.uniform(-1, 1) * params.diameter_variance)

            start, end = get_pennate_fiber_endpoints(
                actual_y,
                actual_z,
                params.length_mm,
                pennation_angle_rad,
                params.architecture_type
            )

            # Apply alignment variance
            start, end = apply_alignment_variance(
                start, end, params.fiber_alignment, rng
            )

            # Create fiber with curvature if enabled
            if params.fiber_curvature > 0.01:
                fiber, midpoint = make_curved_fiber_channel(
                    start, end, actual_radius,
                    params.fiber_curvature,
                    params.resolution, rng
                )
            else:
                fiber = make_fiber_channel(start, end, actual_radius, params.resolution)
                midpoint = (start + end) / 2

            if fiber.num_vert() > 0:
                fibers.append(fiber)
                fiber_midpoints.append(midpoint)
                fascicle_fiber_count += 1
                total_fiber_count += 1

                # ===== CAPILLARIES PER FIBER =====
                if params.enable_vascular_channels:
                    # Calculate capillaries per fiber based on density
                    # capillary_density_per_mm2 is capillaries per mm² cross-section
                    # fiber_spacing_mm is the spacing between fiber centers

                    # Each fiber "owns" an area of approximately fiber_spacing² mm²
                    area_per_fiber_mm2 = fiber_spacing_mm * fiber_spacing_mm

                    # Number of capillaries for this fiber's area
                    target_caps_per_fiber = params.capillary_density_per_mm2 * area_per_fiber_mm2

                    # Add some randomness but stay close to target
                    num_caps = max(1, int(rng.normal(target_caps_per_fiber, target_caps_per_fiber * 0.2)))
                    num_caps = min(num_caps, 8)  # Cap at 8 for performance

                    caps = create_capillaries_for_fiber(
                        start, end, actual_radius,
                        capillary_radius_mm, num_caps,
                        params.resolution, rng
                    )
                    capillaries.extend(caps)

        # ===== PERIMYSIUM AROUND FASCICLE =====
        if params.enable_perimysium and fascicle_fiber_count > 0:
            peri_shell = create_perimysium_shell(
                fasc_y, fasc_z,
                params.fascicle_diameter_mm,
                params.length_mm,
                perimysium_thickness_mm,
                params.perimysium_porosity,
                params.resolution,
                rng
            )
            perimysium_shells.append(peri_shell)

        # ===== ARTERIOLE/VENULE PER FASCICLE =====
        if params.enable_vascular_channels and fascicle_fiber_count > 0:
            arteriole, venule = create_arteriole_venule_pair(
                fasc_y, fasc_z,
                params.fascicle_diameter_mm,
                params.length_mm,
                arteriole_radius_mm,
                venule_radius_mm,
                params.resolution,
                rng
            )
            if arteriole.num_vert() > 0:
                arterioles_venules.append(arteriole)
            if venule.num_vert() > 0:
                arterioles_venules.append(venule)

    if not fibers:
        raise ValueError("No fibers generated")

    # ===== MOTOR ENDPLATES =====
    if params.enable_motor_endplates and fiber_midpoints:
        # Approximately one motor endplate per fiber (at midpoint)
        # motor_endplate_density_per_mm2 controls how many we actually create
        cross_section_area = params.width_mm * params.height_mm
        target_endplates = int(params.motor_endplate_density_per_mm2 * cross_section_area)
        target_endplates = max(1, min(target_endplates, len(fiber_midpoints)))

        # Select random fibers to have motor endplates
        endplate_indices = rng.choice(
            len(fiber_midpoints),
            size=min(target_endplates, len(fiber_midpoints)),
            replace=False
        )

        for idx in endplate_indices:
            midpoint = fiber_midpoints[idx]
            # Motor endplate diameter 30-40 um
            diameter = rng.uniform(30, 40)
            endplate = create_motor_endplate(midpoint, diameter, params.resolution)
            if endplate.num_vert() > 0:
                motor_endplates.append(endplate)

    # Union all fibers
    fibers_union = batch_union(fibers)

    # Subtract fibers from bounding box to create channels
    result = bounding_box - fibers_union

    # ===== ADD VASCULAR CHANNELS =====
    if capillaries:
        capillaries_union = batch_union(capillaries)
        result = result - capillaries_union

    if arterioles_venules:
        av_union = batch_union(arterioles_venules)
        result = result - av_union

    # ===== ADD CONNECTIVE TISSUE LAYERS =====
    # Perimysium shells are added to the scaffold
    if perimysium_shells:
        peri_union = batch_union(perimysium_shells)
        result = result + peri_union

    # ===== EPIMYSIUM (outer shell) =====
    if params.enable_epimysium:
        epi_shell = create_epimysium_shell(
            params.width_mm,
            params.height_mm,
            params.length_mm,
            epimysium_thickness_mm,
            params.resolution
        )
        result = result + epi_shell

    # ===== MOTOR ENDPLATES (subtracted as markers/features) =====
    if motor_endplates:
        endplates_union = batch_union(motor_endplates)
        # Motor endplates could be added or subtracted depending on visualization needs
        # Here we subtract them to create visible marker indentations
        result = result - endplates_union

    # Calculate statistics
    mesh = result.to_mesh()
    volume = result.volume() if hasattr(result, 'volume') else 0
    solid_volume = params.length_mm * params.width_mm * params.height_mm
    porosity = 1 - (volume / solid_volume) if solid_volume > 0 else 0

    # Count sarcomeres per fiber (for stats)
    fiber_length_um = params.fiber_length_mm * 1000
    sarcomeres_per_fiber = int(fiber_length_um / params.sarcomere_length_um)

    stats = {
        'triangle_count': len(mesh.tri_verts) // 3 if hasattr(mesh, 'tri_verts') else 0,
        'volume_mm3': volume,
        'porosity': porosity,
        'fiber_count': total_fiber_count,
        'fiber_parameters': {
            'diameter_um': params.fiber_diameter_um,
            'spacing_um': params.fiber_spacing_um,
            'length_mm': params.fiber_length_mm,
            'alignment': params.fiber_alignment,
            'curvature': params.fiber_curvature
        },
        'fascicle_parameters': {
            'count': len(fascicle_positions),
            'diameter_mm': params.fascicle_diameter_mm,
            'spacing_mm': params.fascicle_spacing_mm,
            'fibers_per_fascicle': total_fiber_count // max(1, len(fascicle_positions))
        },
        'sarcomere_length_um': params.sarcomere_length_um,
        'sarcomeres_per_fiber': sarcomeres_per_fiber,
        'architecture_type': params.architecture_type,
        'pennation_angle_deg': params.pennation_angle_deg,
        'connective_tissue': {
            'endomysium_thickness_um': params.endomysium_thickness_um,
            'perimysium_thickness_um': params.perimysium_thickness_um,
            'perimysium_enabled': params.enable_perimysium,
            'perimysium_count': len(perimysium_shells),
            'epimysium_thickness_um': params.epimysium_thickness_um,
            'epimysium_enabled': params.enable_epimysium
        },
        'vascular': {
            'enabled': params.enable_vascular_channels,
            'capillary_diameter_um': params.capillary_diameter_um,
            'capillary_density_per_mm2': params.capillary_density_per_mm2,  # Actively controls capillary generation
            'capillary_count': len(capillaries),
            'arteriole_diameter_um': params.arteriole_diameter_um,
            'venule_diameter_um': params.venule_diameter_um,
            'arteriole_venule_pairs': len(arterioles_venules) // 2
        },
        'neuromuscular': {
            'motor_endplates_enabled': params.enable_motor_endplates,
            'motor_endplate_count': len(motor_endplates),
            'motor_endplate_density_per_mm2': params.motor_endplate_density_per_mm2
        },
        'contraction_force_uN': params.contraction_force_uN,
        'scaffold_type': 'skeletal_muscle'
    }

    return result, stats


def generate_skeletal_muscle_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate skeletal muscle scaffold from dictionary parameters (for API compatibility).

    Args:
        params: Dictionary with keys matching SkeletalMuscleParams fields

    Returns:
        Tuple of (manifold, stats_dict)
    """
    return generate_skeletal_muscle(SkeletalMuscleParams(
        # Basic geometry
        length_mm=params.get('length_mm', 20.0),
        width_mm=params.get('width_mm', 10.0),
        height_mm=params.get('height_mm', 5.0),

        # Muscle fiber parameters
        fiber_diameter_um=params.get('fiber_diameter_um', 50.0),
        fiber_spacing_um=params.get('fiber_spacing_um', 100.0),
        fiber_length_mm=params.get('fiber_length_mm', 15.0),

        # Fascicle parameters
        fascicle_diameter_mm=params.get('fascicle_diameter_mm', 2.0),
        fascicle_count=params.get('fascicle_count', 4),
        fascicle_spacing_mm=params.get('fascicle_spacing_mm', 0.5),

        # Sarcomere parameters
        sarcomere_length_um=params.get('sarcomere_length_um', 2.5),
        sarcomere_resolution=params.get('sarcomere_resolution', 4),

        # Fiber architecture
        architecture_type=params.get('architecture_type', 'parallel'),
        pennation_angle_deg=params.get('pennation_angle_deg', 15.0),
        fiber_alignment=params.get('fiber_alignment', 0.6),
        fiber_curvature=params.get('fiber_curvature', 0.0),

        # Connective tissue layers
        endomysium_thickness_um=params.get('endomysium_thickness_um', 0.5),
        endomysium_porosity=params.get('endomysium_porosity', 0.3),
        perimysium_thickness_um=params.get('perimysium_thickness_um', 20.0),
        perimysium_porosity=params.get('perimysium_porosity', 0.25),
        enable_perimysium=params.get('enable_perimysium', True),
        epimysium_thickness_um=params.get('epimysium_thickness_um', 100.0),
        enable_epimysium=params.get('enable_epimysium', True),

        # Mechanical properties
        contraction_force_uN=params.get('contraction_force_uN', 443.0),

        # Vascular network
        enable_vascular_channels=params.get('enable_vascular_channels', True),
        capillary_diameter_um=params.get('capillary_diameter_um', 8.0),
        capillary_density_per_mm2=params.get('capillary_density_per_mm2', 300),
        arteriole_diameter_um=params.get('arteriole_diameter_um', 50.0),
        venule_diameter_um=params.get('venule_diameter_um', 60.0),

        # Neuromuscular features
        enable_motor_endplates=params.get('enable_motor_endplates', False),
        motor_endplate_density_per_mm2=params.get('motor_endplate_density_per_mm2', 0.5),

        # Randomization
        seed=params.get('seed', 42),
        position_noise=params.get('position_noise', 0.1),
        diameter_variance=params.get('diameter_variance', 0.1),

        # Resolution
        resolution=params.get('resolution', 8)
    ))
