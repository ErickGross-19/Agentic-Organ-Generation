"""
Liver sinusoid scaffold generator.

Creates core-shell tubular structures mimicking hepatic sinusoidal channels:
- Hollow tube with fenestrations (pores) in wall
- Fenestrated endothelium (50-150nm pores, scaled for scaffold)
- Space of Disse representation
- Hepatocyte-scale features surrounding sinusoid
- Perfusable lumen for blood flow simulation
- Kupffer cell and stellate cell zones
"""

from __future__ import annotations
import logging
import manifold3d as m3d
import numpy as np
from dataclasses import dataclass
from ..core import batch_union

logger = logging.getLogger(__name__)


@dataclass
class LiverSinusoidParams:
    """
    Parameters for liver sinusoid scaffold generation.

    Based on native hepatic sinusoid anatomy:
    - Sinusoid diameter: 7-15 um
    - Sinusoid length: 200-500 um
    - Fenestration diameter: 50-150 nm (scaled for printability)
    - Fenestration density: 9-13 per um^2
    - Space of Disse: 0.2-0.5 um
    - Hepatocyte diameter: 20-30 um
    """
    # === Sinusoid Geometry ===
    sinusoid_diameter: float = 12.0  # um (7-15 um native)
    sinusoid_length: float = 250.0  # um (200-500 um native)
    sinusoid_wall_thickness: float = 0.5  # um (endothelial cell layer)

    # === Fenestration Parameters ===
    fenestration_diameter: float = 150.0  # nm (50-150 nm, scaled to um for scaffold)
    fenestration_density: float = 10.0  # per um^2 (9-13 native)
    fenestration_pattern: str = 'clustered'  # clustered (sieve plates), random
    sieve_plate_count: int = 8  # number of sieve plate regions
    fenestration_porosity: float = 0.06  # ~6-8% of endothelial surface

    # === Space of Disse ===
    enable_space_of_disse: bool = True
    space_of_disse_thickness: float = 0.5  # um (0.2-0.5 um native)
    space_of_disse_porosity: float = 0.8  # high porosity for protein exchange

    # === Hepatocyte Zone ===
    enable_hepatocyte_zone: bool = False  # surrounding hepatocyte markers
    hepatocyte_diameter: float = 25.0  # um (20-30 um native)
    hepatocyte_spacing: float = 5.0  # um (gap between hepatocytes)
    hepatocytes_per_sinusoid: int = 12  # around circumference

    # === Cell Type Zones ===
    enable_kupffer_cell_zones: bool = False  # macrophage attachment sites
    kupffer_cell_density: float = 0.15  # ~15% of sinusoid wall
    enable_stellate_cell_zones: bool = False  # HSC in space of Disse
    stellate_cell_spacing: float = 50.0  # um

    # === Sinusoid Network ===
    sinusoid_count: int = 1  # number of parallel sinusoids
    sinusoid_spacing: float = 50.0  # um between sinusoid centers
    network_pattern: str = 'parallel'  # parallel, branching, anastomosing
    enable_central_vein_connection: bool = False
    central_vein_diameter: float = 30.0  # um

    # === Scaffold Structure ===
    scaffold_length: float = 2.0  # mm total scaffold length
    enable_scaffold_shell: bool = False  # outer structural shell
    shell_thickness: float = 50.0  # um
    shell_porosity: float = 0.5

    # === Bile Canaliculi (optional) ===
    enable_bile_canaliculi: bool = False
    canaliculus_diameter: float = 1.0  # um (0.5-2.0 um native)
    canaliculus_spacing: float = 25.0  # um

    # === ECM Components ===
    enable_ecm_fibers: bool = False  # collagen fibers in space of Disse
    ecm_fiber_diameter: float = 0.12  # um (100-150 nm native reticular fibers)
    ecm_fiber_density: float = 0.2  # fraction of space of Disse

    # === Stochastic Variation ===
    diameter_variance: float = 0.0  # 0-1 variation in sinusoid diameter
    fenestration_variance: float = 0.0  # 0-1 variation in fenestration size
    position_noise: float = 0.0  # 0-1 position jitter
    seed: int = 42

    # === Resolution ===
    resolution: int = 12  # segments around cylinder


def create_hollow_tube(length: float, outer_radius: float, inner_radius: float, resolution: int) -> m3d.Manifold:
    """
    Create a hollow tube along Z axis.

    Args:
        length: Tube length
        outer_radius: Outer radius
        inner_radius: Inner radius (lumen)
        resolution: Number of segments around cylinder

    Returns:
        Hollow tube manifold
    """
    outer = m3d.Manifold.cylinder(length, outer_radius, outer_radius, resolution)
    inner = m3d.Manifold.cylinder(length + 0.02, inner_radius, inner_radius, resolution)
    inner = inner.translate([0, 0, -0.01])
    return outer - inner


def create_fenestrations_clustered(length: float, radius: float, fenestration_radius: float,
                                   sieve_plate_count: int, porosity: float,
                                   resolution: int, seed: int) -> list[m3d.Manifold]:
    """
    Create fenestration pores in clustered sieve plate pattern.

    Args:
        length: Tube length
        radius: Tube radius (where pores are placed)
        fenestration_radius: Radius of each pore (mm)
        sieve_plate_count: Number of sieve plate regions
        porosity: Target porosity fraction
        resolution: Pore resolution
        seed: Random seed

    Returns:
        List of fenestration sphere manifolds
    """
    np.random.seed(seed)

    if porosity <= 0:
        return []

    fenestrations = []

    # Calculate number of pores per sieve plate
    plate_length = length / sieve_plate_count
    plate_surface = 2 * np.pi * radius * plate_length
    pore_area = np.pi * fenestration_radius**2
    pores_per_plate = max(3, int(plate_surface / pore_area * porosity * 0.5))

    for plate_idx in range(sieve_plate_count):
        plate_z_start = plate_idx * plate_length
        plate_z_end = plate_z_start + plate_length * 0.6  # Sieve plates don't cover full length

        # Cluster pores within plate region
        for _ in range(pores_per_plate):
            z = np.random.uniform(plate_z_start + 0.1 * plate_length, plate_z_end)
            angle = np.random.uniform(0, 2 * np.pi)

            x = radius * np.cos(angle)
            y = radius * np.sin(angle)

            pore = m3d.Manifold.sphere(fenestration_radius, 6)
            pore = pore.translate([x, y, z])
            fenestrations.append(pore)

    return fenestrations


def create_fenestrations_random(length: float, radius: float, fenestration_radius: float,
                                density: float, resolution: int) -> list[m3d.Manifold]:
    """
    Create fenestration pores distributed randomly on tube surface.

    Args:
        length: Tube length
        radius: Tube radius (where pores are placed)
        fenestration_radius: Radius of each pore
        density: Pore density (pores per mm^2)
        resolution: Pore resolution

    Returns:
        List of fenestration sphere manifolds
    """
    if density <= 0:
        return []

    # Calculate number of pores based on surface area and density
    surface_area = 2 * np.pi * radius * length
    num_pores = max(5, int(surface_area * density))

    fenestrations = []

    # Distribute pores evenly in cylindrical coordinates
    num_circumferential = max(4, int(np.sqrt(num_pores * 2)))
    num_axial = max(2, num_pores // num_circumferential)

    for i in range(num_axial):
        z = (i + 0.5) * length / num_axial

        # Alternate offset for each row (brick pattern)
        angle_offset = (i % 2) * np.pi / num_circumferential

        for j in range(num_circumferential):
            angle = 2 * np.pi * j / num_circumferential + angle_offset

            # Position on tube surface
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)

            # Create small sphere for fenestration
            pore = m3d.Manifold.sphere(fenestration_radius, 6)
            pore = pore.translate([x, y, z])
            fenestrations.append(pore)

    return fenestrations


def create_space_of_disse_layer(length: float, inner_radius: float, thickness: float,
                                porosity: float, pore_radius: float,
                                resolution: int) -> m3d.Manifold:
    """
    Create the space of Disse layer (perisinusoidal space).

    Args:
        length: Tube length
        inner_radius: Inner radius (at sinusoid wall)
        thickness: Thickness of space of Disse
        porosity: Target porosity
        pore_radius: Pore radius for the porous structure
        resolution: Cylinder resolution

    Returns:
        Porous layer manifold
    """
    outer_radius = inner_radius + thickness
    layer = create_hollow_tube(length, outer_radius, inner_radius, resolution)

    if porosity > 0:
        # Add pores to represent protein exchange space
        surface_area = 2 * np.pi * outer_radius * length
        pore_area = np.pi * pore_radius**2
        pore_count = max(6, int(surface_area / pore_area * porosity * 0.3))

        pores = []
        for i in range(pore_count):
            z = np.random.uniform(0, length)
            angle = np.random.uniform(0, 2 * np.pi)
            x = outer_radius * np.cos(angle)
            y = outer_radius * np.sin(angle)

            pore = m3d.Manifold.sphere(pore_radius, 6)
            pore = pore.translate([x, y, z])
            pores.append(pore)

        if pores:
            pore_union = batch_union(pores)
            layer = layer - pore_union

    return layer


def create_hepatocyte_markers(length: float, sinusoid_radius: float,
                              hepatocyte_radius: float, count: int,
                              spacing: float, resolution: int) -> list[m3d.Manifold]:
    """
    Create spherical markers representing hepatocytes around sinusoid.

    Args:
        length: Sinusoid length
        sinusoid_radius: Radius of sinusoid
        hepatocyte_radius: Radius of hepatocyte markers
        count: Number of hepatocytes around circumference
        spacing: Gap between hepatocytes (mm)
        resolution: Sphere resolution

    Returns:
        List of hepatocyte marker manifolds
    """
    markers = []
    # Use spacing parameter for ring distance (hepatocyte diameter + gap)
    ring_spacing = hepatocyte_radius * 2 + spacing
    num_rings = max(1, int(length / ring_spacing))
    placement_radius = sinusoid_radius + hepatocyte_radius * 1.2

    for ring_idx in range(num_rings):
        z = ring_spacing * (ring_idx + 0.5)

        for i in range(count):
            angle = 2 * np.pi * i / count + (ring_idx % 2) * np.pi / count
            x = placement_radius * np.cos(angle)
            y = placement_radius * np.sin(angle)

            marker = m3d.Manifold.sphere(hepatocyte_radius, resolution)
            marker = marker.translate([x, y, z])
            markers.append(marker)

    return markers


def create_kupffer_cell_zones(length: float, inner_radius: float,
                              density: float, seed: int,
                              resolution: int) -> list[m3d.Manifold]:
    """
    Create Kupffer cell protrusions on sinusoid inner wall.

    Kupffer cells are resident liver macrophages (~10-15um) that attach to
    sinusoidal endothelium and protrude into the lumen.

    Args:
        length: Sinusoid length (mm)
        inner_radius: Inner radius of sinusoid (mm)
        density: Fraction of wall surface covered by Kupffer cells (0-1)
        seed: Random seed
        resolution: Sphere resolution

    Returns:
        List of Kupffer cell manifolds (spherical protrusions)
    """
    np.random.seed(seed + 100)  # Offset seed for independent randomness

    if density <= 0:
        return []

    # Kupffer cell size: 10-15um, use 12.5um average = 0.0125mm diameter
    kupffer_radius = 0.00625  # 6.25um radius = 12.5um diameter

    # Calculate number of cells based on wall surface area and density
    wall_surface = 2 * np.pi * inner_radius * length
    cell_area = np.pi * kupffer_radius**2
    num_cells = max(1, int(wall_surface * density / cell_area))

    cells = []
    for _ in range(num_cells):
        z = np.random.uniform(kupffer_radius, length - kupffer_radius)
        angle = np.random.uniform(0, 2 * np.pi)

        # Position on inner wall, protruding into lumen
        x = inner_radius * np.cos(angle)
        y = inner_radius * np.sin(angle)

        cell = m3d.Manifold.sphere(kupffer_radius, resolution)
        cell = cell.translate([x, y, z])
        cells.append(cell)

    return cells


def create_stellate_cell_zones(length: float, sinusoid_outer_radius: float,
                               space_of_disse_thickness: float,
                               spacing: float, seed: int,
                               resolution: int) -> list[m3d.Manifold]:
    """
    Create stellate cell markers in the space of Disse.

    Hepatic stellate cells (HSC) are star-shaped cells that store vitamin A
    and reside in the space of Disse between sinusoid and hepatocytes.

    Args:
        length: Sinusoid length (mm)
        sinusoid_outer_radius: Outer radius of sinusoid wall (mm)
        space_of_disse_thickness: Thickness of space of Disse (mm)
        spacing: Distance between stellate cells (mm)
        seed: Random seed
        resolution: Sphere resolution

    Returns:
        List of stellate cell manifolds
    """
    np.random.seed(seed + 200)

    if spacing <= 0:
        return []

    # Stellate cell body: 5-10um, use 7.5um = 0.0075mm diameter
    stellate_radius = 0.00375  # 3.75um radius

    # Position in middle of space of Disse
    placement_radius = sinusoid_outer_radius + space_of_disse_thickness / 2

    # Calculate number of cells along length
    num_along_length = max(1, int(length / (spacing / 1000.0)))  # spacing is in um
    spacing_mm = spacing / 1000.0

    # Place around circumference too (every 60 degrees = 6 cells around)
    num_around = 6

    cells = []
    for i in range(num_along_length):
        z = (i + 0.5) * spacing_mm
        if z > length:
            break

        for j in range(num_around):
            angle = 2 * np.pi * j / num_around + (i % 2) * np.pi / num_around

            x = placement_radius * np.cos(angle)
            y = placement_radius * np.sin(angle)

            cell = m3d.Manifold.sphere(stellate_radius, resolution)
            cell = cell.translate([x, y, z])
            cells.append(cell)

    return cells


def create_central_vein(sinusoid_length: float, vein_diameter: float,
                        sinusoid_positions: list, sinusoid_outer_radius: float,
                        resolution: int) -> m3d.Manifold:
    """
    Create central vein with connections to sinusoids.

    The central vein receives blood from multiple sinusoids (15-20 native).
    Creates a vertical tube at z=length end with connecting channels.

    Args:
        sinusoid_length: Length of sinusoids (mm)
        vein_diameter: Diameter of central vein (um)
        sinusoid_positions: List of sinusoid center positions
        sinusoid_outer_radius: Outer radius of sinusoids (mm)
        resolution: Cylinder resolution

    Returns:
        Central vein manifold with connections
    """
    vein_radius = vein_diameter / 2000.0  # um to mm
    vein_length = vein_diameter / 1000.0  # Make vein length equal to diameter

    # Calculate center position (average of all sinusoid positions)
    if len(sinusoid_positions) > 1:
        center_x = np.mean([p[0] for p in sinusoid_positions])
        center_y = np.mean([p[1] for p in sinusoid_positions])
    else:
        center_x = 0.0
        center_y = 0.0

    # Create main vein tube (hollow)
    vein_wall_thickness = vein_radius * 0.1  # 10% wall thickness
    vein_inner_radius = vein_radius - vein_wall_thickness

    outer_vein = m3d.Manifold.cylinder(vein_length, vein_radius, vein_radius, resolution)
    inner_vein = m3d.Manifold.cylinder(vein_length + 0.002, vein_inner_radius, vein_inner_radius, resolution)
    inner_vein = inner_vein.translate([0, 0, -0.001])
    central_vein = outer_vein - inner_vein

    # Position at end of sinusoids
    central_vein = central_vein.translate([center_x, center_y, sinusoid_length])

    # Create connecting channels from each sinusoid to central vein
    parts = [central_vein]

    for pos in sinusoid_positions:
        # Connection tube from sinusoid end to central vein
        dx = center_x - pos[0]
        dy = center_y - pos[1]
        dist = np.sqrt(dx**2 + dy**2)

        if dist > 0.001:  # Only create connection if sinusoid is not at center
            # Create angled tube connecting sinusoid to vein
            connection_radius = sinusoid_outer_radius * 0.8
            connection = m3d.Manifold.cylinder(dist + vein_radius, connection_radius, connection_radius, 8)

            # Rotate to point toward center
            angle = np.arctan2(dy, dx)
            connection = connection.rotate([0, 90, 0])
            connection = connection.rotate([0, 0, np.degrees(angle)])
            connection = connection.translate([pos[0], pos[1], sinusoid_length])
            parts.append(connection)

    return batch_union(parts)


def create_scaffold_shell(sinusoid_positions: list, scaffold_length_mm: float,
                          sinusoid_outer_radius: float, space_of_disse: float,
                          hepatocyte_radius: float, shell_thickness: float,
                          shell_porosity: float, resolution: int,
                          seed: int) -> m3d.Manifold:
    """
    Create outer structural shell around entire sinusoid network.

    Args:
        sinusoid_positions: List of sinusoid center positions
        scaffold_length_mm: Total scaffold length (mm) - Z-dimension of shell
        sinusoid_outer_radius: Outer radius of sinusoids (mm)
        space_of_disse: Space of Disse thickness (mm)
        hepatocyte_radius: Hepatocyte radius if zone enabled (mm)
        shell_thickness: Thickness of shell (um)
        shell_porosity: Porosity of shell (0-1)
        resolution: Cylinder resolution
        seed: Random seed

    Returns:
        Porous shell manifold
    """
    np.random.seed(seed + 300)

    shell_thickness_mm = shell_thickness / 1000.0

    # Calculate bounding dimensions
    if len(sinusoid_positions) > 1:
        xs = [p[0] for p in sinusoid_positions]
        ys = [p[1] for p in sinusoid_positions]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
    else:
        min_x = max_x = sinusoid_positions[0][0] if sinusoid_positions else 0
        min_y = max_y = sinusoid_positions[0][1] if sinusoid_positions else 0

    # Content radius (sinusoid + space of Disse + hepatocyte margin)
    content_radius = sinusoid_outer_radius + space_of_disse + hepatocyte_radius * 2.5

    # Shell inner dimensions
    inner_radius = max(max_x - min_x, max_y - min_y) / 2 + content_radius
    outer_radius = inner_radius + shell_thickness_mm

    # Create cylindrical shell using scaffold_length for Z-dimension
    outer_shell = m3d.Manifold.cylinder(scaffold_length_mm, outer_radius, outer_radius, resolution)
    inner_shell = m3d.Manifold.cylinder(scaffold_length_mm + 0.002, inner_radius, inner_radius, resolution)
    inner_shell = inner_shell.translate([0, 0, -0.001])
    shell = outer_shell - inner_shell

    # Center the shell
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    shell = shell.translate([center_x, center_y, 0])

    # Add pores if porosity > 0
    if shell_porosity > 0:
        pore_radius = shell_thickness_mm * 0.3
        surface_area = 2 * np.pi * outer_radius * scaffold_length_mm
        pore_area = np.pi * pore_radius**2
        num_pores = max(6, int(surface_area * shell_porosity / pore_area * 0.5))

        pores = []
        for _ in range(num_pores):
            z = np.random.uniform(pore_radius, scaffold_length_mm - pore_radius)
            angle = np.random.uniform(0, 2 * np.pi)

            x = center_x + outer_radius * np.cos(angle)
            y = center_y + outer_radius * np.sin(angle)

            pore = m3d.Manifold.sphere(pore_radius, 6)
            pore = pore.translate([x, y, z])
            pores.append(pore)

        if pores:
            shell = shell - batch_union(pores)

    return shell


def create_bile_canaliculi(length: float, sinusoid_outer_radius: float,
                           space_of_disse: float, hepatocyte_radius: float,
                           canaliculus_diameter: float, canaliculus_spacing: float,
                           resolution: int) -> list[m3d.Manifold]:
    """
    Create bile canaliculi channels running between hepatocytes.

    Bile canaliculi are tiny channels (0.5-2.0um) sealed by tight junctions
    between adjacent hepatocytes, draining bile toward bile ducts.

    Args:
        length: Sinusoid length (mm)
        sinusoid_outer_radius: Outer radius of sinusoid (mm)
        space_of_disse: Space of Disse thickness (mm)
        hepatocyte_radius: Hepatocyte radius (mm)
        canaliculus_diameter: Diameter of canaliculi (um)
        canaliculus_spacing: Spacing between canaliculi (um)
        resolution: Cylinder resolution

    Returns:
        List of canaliculus manifolds
    """
    canaliculus_radius = canaliculus_diameter / 2000.0  # um to mm
    spacing_mm = canaliculus_spacing / 1000.0

    # Position canaliculi in hepatocyte zone (between hepatocytes)
    canaliculus_placement_radius = sinusoid_outer_radius + space_of_disse + hepatocyte_radius * 1.5

    # Number of canaliculi along length
    num_along_length = max(1, int(length / spacing_mm))

    # Canaliculi run perpendicular to sinusoid (radially outward)
    # Create short tubes representing cross-sections visible at scaffold surface
    canaliculi = []

    for i in range(num_along_length):
        z = (i + 0.5) * spacing_mm
        if z > length:
            break

        # Place canaliculi around circumference (between hepatocytes)
        num_around = 6  # Between pairs of hepatocytes
        for j in range(num_around):
            angle = 2 * np.pi * j / num_around + (i % 2) * np.pi / num_around

            x = canaliculus_placement_radius * np.cos(angle)
            y = canaliculus_placement_radius * np.sin(angle)

            # Create short tube running parallel to sinusoid axis
            canaliculus = m3d.Manifold.cylinder(spacing_mm * 0.8, canaliculus_radius, canaliculus_radius, 6)
            canaliculus = canaliculus.translate([x, y, z - spacing_mm * 0.4])
            canaliculi.append(canaliculus)

    return canaliculi


def create_ecm_fibers(length: float, sinusoid_outer_radius: float,
                      space_of_disse: float, fiber_diameter: float,
                      fiber_density: float, seed: int,
                      resolution: int) -> list[m3d.Manifold]:
    """
    Create ECM collagen fibers in space of Disse.

    Reticular fibers (collagen III) support the sinusoidal structure
    and run along the sinusoid axis in the space of Disse.

    Args:
        length: Sinusoid length (mm)
        sinusoid_outer_radius: Outer radius of sinusoid (mm)
        space_of_disse: Space of Disse thickness (mm)
        fiber_diameter: Diameter of fibers (um)
        fiber_density: Fraction of space occupied by fibers (0-1)
        seed: Random seed
        resolution: Cylinder resolution

    Returns:
        List of fiber manifolds
    """
    np.random.seed(seed + 400)

    if fiber_density <= 0:
        return []

    fiber_radius = fiber_diameter / 2000.0  # um to mm

    # Place fibers in middle of space of Disse
    placement_radius = sinusoid_outer_radius + space_of_disse / 2

    # Calculate number of fibers based on density
    # Cross-sectional area of space of Disse ring
    disse_area = np.pi * ((sinusoid_outer_radius + space_of_disse)**2 - sinusoid_outer_radius**2)
    fiber_area = np.pi * fiber_radius**2
    num_fibers = max(3, int(disse_area * fiber_density / fiber_area))

    fibers = []
    for i in range(num_fibers):
        angle = 2 * np.pi * i / num_fibers

        # Add slight radial variation
        r_variation = space_of_disse * 0.3 * np.random.uniform(-1, 1)
        r = placement_radius + r_variation

        x = r * np.cos(angle)
        y = r * np.sin(angle)

        # Create fiber running along sinusoid length
        fiber = m3d.Manifold.cylinder(length, fiber_radius, fiber_radius, 6)
        fiber = fiber.translate([x, y, 0])
        fibers.append(fiber)

    return fibers


def generate_liver_sinusoid(params: LiverSinusoidParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate a liver sinusoid scaffold with fenestrations.

    Args:
        params: LiverSinusoidParams specifying geometry

    Returns:
        Tuple of (manifold, stats_dict)
        - manifold: The 3D geometry
        - stats_dict: Dictionary with triangle_count, volume_mm3, fenestration_count, scaffold_type

    Raises:
        ValueError: If no geometry is generated
    """
    np.random.seed(params.seed)

    # Convert um to mm
    outer_radius = params.sinusoid_diameter / 2000.0
    wall_thickness = params.sinusoid_wall_thickness / 1000.0
    fenestration_radius_base = params.fenestration_diameter / 1000.0 / 2000.0  # nm to mm
    space_of_disse_mm = params.space_of_disse_thickness / 1000.0
    hepatocyte_radius_mm = params.hepatocyte_diameter / 2000.0
    hepatocyte_spacing_mm = params.hepatocyte_spacing / 1000.0
    sinusoid_length_mm = params.sinusoid_length / 1000.0  # um to mm (individual sinusoid)
    scaffold_length_mm = params.scaffold_length  # mm (overall scaffold Z-dimension)

    # Validate scaffold_length vs sinusoid_length
    # scaffold_length defines the bounding shell Z-dimension
    # If scaffold_length < sinusoid_length, warn and use sinusoid_length
    if scaffold_length_mm < sinusoid_length_mm:
        logger.warning(
            f"scaffold_length ({scaffold_length_mm:.3f} mm) is less than sinusoid_length "
            f"({sinusoid_length_mm:.3f} mm). Using sinusoid_length as scaffold_length."
        )
        scaffold_length_mm = sinusoid_length_mm

    # Ensure minimum fenestration size for printability
    fenestration_radius_base = max(fenestration_radius_base, 0.001)  # min 1um radius

    inner_radius = outer_radius - wall_thickness
    if inner_radius <= 0:
        inner_radius = outer_radius * 0.5

    all_parts = []
    all_fenestrations = []
    all_kupffer_cells = []
    all_stellate_cells = []
    all_bile_canaliculi = []
    all_ecm_fibers = []

    # Generate sinusoids based on count and pattern
    sinusoid_positions = []
    spacing_mm = params.sinusoid_spacing / 1000.0

    # Calculate Z-offset for distributing multiple sinusoids within scaffold_length
    # When sinusoid_count > 1, distribute them along Z-axis within scaffold_length
    # Each sinusoid has length sinusoid_length_mm; we center them within scaffold_length_mm
    if params.sinusoid_count == 1:
        # Single sinusoid: center it in the scaffold
        z_offset = (scaffold_length_mm - sinusoid_length_mm) / 2.0
        sinusoid_positions.append(np.array([0.0, 0.0, z_offset]))
    else:
        # Multiple sinusoids: distribute both in XY and along Z within scaffold_length
        # Calculate how many sinusoids can fit end-to-end in scaffold_length
        # If they fit, arrange them sequentially; otherwise overlay them at different XY positions
        total_length_needed = params.sinusoid_count * sinusoid_length_mm

        for i in range(params.sinusoid_count):
            # Apply position noise if enabled
            noise_x = 0.0
            noise_y = 0.0
            if params.position_noise > 0:
                noise_x = np.random.uniform(-1, 1) * params.position_noise * spacing_mm * 0.2
                noise_y = np.random.uniform(-1, 1) * params.position_noise * spacing_mm * 0.2

            if params.network_pattern == 'parallel':
                x_offset = (i - params.sinusoid_count / 2) * spacing_mm + noise_x
                # Distribute along Z within scaffold_length
                # Space sinusoids evenly, considering their length
                available_z = scaffold_length_mm - sinusoid_length_mm
                if available_z > 0 and params.sinusoid_count > 1:
                    z_offset = (available_z / (params.sinusoid_count - 1)) * i
                else:
                    z_offset = 0.0
                sinusoid_positions.append(np.array([x_offset, noise_y, z_offset]))
            else:  # branching or anastomosing
                angle = 2 * np.pi * i / params.sinusoid_count
                x = spacing_mm * np.cos(angle) + noise_x
                y = spacing_mm * np.sin(angle) + noise_y
                # For radial patterns, also distribute along Z
                available_z = scaffold_length_mm - sinusoid_length_mm
                if available_z > 0 and params.sinusoid_count > 1:
                    z_offset = (available_z / (params.sinusoid_count - 1)) * i
                else:
                    z_offset = 0.0
                sinusoid_positions.append(np.array([x, y, z_offset]))

    for pos_idx, pos in enumerate(sinusoid_positions):
        # Apply diameter variance
        if params.diameter_variance > 0:
            variance = 1 + np.random.uniform(-1, 1) * params.diameter_variance * 0.2
            sin_outer = outer_radius * variance
            sin_inner = inner_radius * variance
        else:
            sin_outer = outer_radius
            sin_inner = inner_radius

        # Create hollow tube for sinusoid
        tube = create_hollow_tube(sinusoid_length_mm, sin_outer, sin_inner, params.resolution)
        tube = tube.translate([pos[0], pos[1], pos[2]])

        # Create fenestrations with variance
        if params.fenestration_pattern == 'clustered':
            # Apply fenestration variance by creating fenestrations with varied sizes
            if params.fenestration_variance > 0:
                # Create fenestrations with varied sizes
                np.random.seed(params.seed + pos_idx * 1000)
                base_fenestrations = create_fenestrations_clustered(
                    sinusoid_length_mm, sin_outer, fenestration_radius_base,
                    params.sieve_plate_count, params.fenestration_porosity,
                    8, params.seed + pos_idx
                )
                # Vary sizes by recreating with different radii
                fenestrations = []
                for fen in base_fenestrations:
                    # Get center from bounding box
                    bbox = fen.bounding_box()
                    center = [(bbox.min[i] + bbox.max[i]) / 2 for i in range(3)]
                    # Create new fenestration with varied size
                    size_factor = 1 + np.random.uniform(-1, 1) * params.fenestration_variance * 0.3
                    varied_radius = fenestration_radius_base * size_factor
                    varied_fen = m3d.Manifold.sphere(varied_radius, 6)
                    varied_fen = varied_fen.translate(center)
                    fenestrations.append(varied_fen)
            else:
                fenestrations = create_fenestrations_clustered(
                    sinusoid_length_mm, sin_outer, fenestration_radius_base,
                    params.sieve_plate_count, params.fenestration_porosity,
                    8, params.seed + pos_idx
                )
        else:
            fenestrations = create_fenestrations_random(
                sinusoid_length_mm, sin_outer, fenestration_radius_base,
                params.fenestration_density, 8
            )
            # Apply variance to random fenestrations too
            if params.fenestration_variance > 0:
                varied_fenestrations = []
                for fen in fenestrations:
                    bbox = fen.bounding_box()
                    center = [(bbox.min[i] + bbox.max[i]) / 2 for i in range(3)]
                    size_factor = 1 + np.random.uniform(-1, 1) * params.fenestration_variance * 0.3
                    varied_radius = fenestration_radius_base * size_factor
                    varied_fen = m3d.Manifold.sphere(varied_radius, 6)
                    varied_fen = varied_fen.translate(center)
                    varied_fenestrations.append(varied_fen)
                fenestrations = varied_fenestrations

        # Translate fenestrations to sinusoid position
        for fen in fenestrations:
            fen = fen.translate([pos[0], pos[1], pos[2]])
            all_fenestrations.append(fen)

        all_parts.append(tube)

        # Add Kupffer cell zones if enabled
        if params.enable_kupffer_cell_zones:
            kupffer_cells = create_kupffer_cell_zones(
                sinusoid_length_mm, sin_inner, params.kupffer_cell_density,
                params.seed + pos_idx * 100, max(6, params.resolution - 4)
            )
            for cell in kupffer_cells:
                cell = cell.translate([pos[0], pos[1], pos[2]])
                all_kupffer_cells.append(cell)

        # Add space of Disse if enabled
        if params.enable_space_of_disse:
            disse_layer = create_space_of_disse_layer(
                sinusoid_length_mm, sin_outer, space_of_disse_mm,
                params.space_of_disse_porosity, fenestration_radius_base * 2,
                params.resolution
            )
            disse_layer = disse_layer.translate([pos[0], pos[1], pos[2]])
            all_parts.append(disse_layer)

            # Add stellate cells in space of Disse if enabled
            if params.enable_stellate_cell_zones:
                stellate_cells = create_stellate_cell_zones(
                    sinusoid_length_mm, sin_outer, space_of_disse_mm,
                    params.stellate_cell_spacing, params.seed + pos_idx * 200,
                    max(6, params.resolution - 4)
                )
                for cell in stellate_cells:
                    cell = cell.translate([pos[0], pos[1], pos[2]])
                    all_stellate_cells.append(cell)

            # Add ECM fibers in space of Disse if enabled
            if params.enable_ecm_fibers:
                ecm_fibers = create_ecm_fibers(
                    sinusoid_length_mm, sin_outer, space_of_disse_mm,
                    params.ecm_fiber_diameter, params.ecm_fiber_density,
                    params.seed + pos_idx * 400, 6
                )
                for fiber in ecm_fibers:
                    fiber = fiber.translate([pos[0], pos[1], pos[2]])
                    all_ecm_fibers.append(fiber)

        # Add hepatocyte markers if enabled
        if params.enable_hepatocyte_zone:
            markers = create_hepatocyte_markers(
                sinusoid_length_mm, sin_outer + space_of_disse_mm,
                hepatocyte_radius_mm, params.hepatocytes_per_sinusoid,
                hepatocyte_spacing_mm, max(8, params.resolution - 2)
            )
            for marker in markers:
                marker = marker.translate([pos[0], pos[1], pos[2]])
                all_parts.append(marker)

            # Add bile canaliculi if enabled (requires hepatocyte zone)
            if params.enable_bile_canaliculi:
                canaliculi = create_bile_canaliculi(
                    sinusoid_length_mm, sin_outer, space_of_disse_mm, hepatocyte_radius_mm,
                    params.canaliculus_diameter, params.canaliculus_spacing,
                    6
                )
                for canaliculus in canaliculi:
                    canaliculus = canaliculus.translate([pos[0], pos[1], pos[2]])
                    all_bile_canaliculi.append(canaliculus)
        elif params.enable_bile_canaliculi:
            # Bile canaliculi alongside sinusoid even without hepatocyte zone
            canaliculi = create_bile_canaliculi(
                sinusoid_length_mm, sin_outer, space_of_disse_mm, hepatocyte_radius_mm,
                params.canaliculus_diameter, params.canaliculus_spacing,
                6
            )
            for canaliculus in canaliculi:
                canaliculus = canaliculus.translate([pos[0], pos[1], pos[2]])
                all_bile_canaliculi.append(canaliculus)

    if not all_parts:
        raise ValueError("No geometry generated")

    # Add Kupffer cells to parts (they protrude into lumen, so add them)
    all_parts.extend(all_kupffer_cells)

    # Add stellate cells to parts
    all_parts.extend(all_stellate_cells)

    # Add ECM fibers to parts
    all_parts.extend(all_ecm_fibers)

    # Add bile canaliculi to parts
    all_parts.extend(all_bile_canaliculi)

    # Add central vein connection if enabled
    # Central vein connects at the end of sinusoids, positioned based on scaffold_length
    if params.enable_central_vein_connection and len(sinusoid_positions) > 0:
        # Find the maximum Z extent of all sinusoids
        max_z_extent = max(pos[2] + sinusoid_length_mm for pos in sinusoid_positions)
        central_vein = create_central_vein(
            max_z_extent, params.central_vein_diameter,
            sinusoid_positions, outer_radius,
            params.resolution
        )
        all_parts.append(central_vein)

    # Add scaffold shell if enabled
    # Shell uses scaffold_length for Z-dimension to encompass all sinusoids
    if params.enable_scaffold_shell:
        shell = create_scaffold_shell(
            sinusoid_positions, scaffold_length_mm, outer_radius,
            space_of_disse_mm if params.enable_space_of_disse else 0,
            hepatocyte_radius_mm if params.enable_hepatocyte_zone else 0,
            params.shell_thickness, params.shell_porosity,
            params.resolution, params.seed
        )
        all_parts.append(shell)

    result = batch_union(all_parts)

    # Subtract fenestrations from sinusoid walls
    if all_fenestrations:
        fenestration_union = batch_union(all_fenestrations)
        result = result - fenestration_union

    if result.num_vert() == 0:
        raise ValueError("No geometry generated after fenestration subtraction")

    # Calculate statistics
    mesh = result.to_mesh()
    volume = result.volume() if hasattr(result, 'volume') else 0

    stats = {
        'triangle_count': len(mesh.tri_verts) if hasattr(mesh, 'tri_verts') else 0,
        'volume_mm3': volume,
        'fenestration_count': len(all_fenestrations),
        'sinusoid_count': params.sinusoid_count,
        'sinusoid_length_mm': sinusoid_length_mm,
        'scaffold_length_mm': scaffold_length_mm,
        'has_space_of_disse': params.enable_space_of_disse,
        'kupffer_cell_count': len(all_kupffer_cells),
        'stellate_cell_count': len(all_stellate_cells),
        'bile_canaliculi_count': len(all_bile_canaliculi),
        'ecm_fiber_count': len(all_ecm_fibers),
        'has_central_vein': params.enable_central_vein_connection,
        'has_scaffold_shell': params.enable_scaffold_shell,
        'scaffold_type': 'liver_sinusoid'
    }

    return result, stats


def generate_liver_sinusoid_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate liver sinusoid from dictionary parameters (for API compatibility).

    Args:
        params: Dictionary with keys matching LiverSinusoidParams fields

    Returns:
        Tuple of (manifold, stats_dict)
    """
    # Handle legacy parameter names
    length = params.get('length', params.get('sinusoid_length', 250.0))
    if length > 10:  # Assume it's in um if > 10
        length_um = length
    else:  # Assume it's in mm, convert to um
        length_um = length * 1000.0

    fenestration_size = params.get('fenestration_size', params.get('fenestration_diameter', 150.0))
    fenestration_density = params.get('fenestration_density', 10.0)

    return generate_liver_sinusoid(LiverSinusoidParams(
        # Sinusoid geometry
        sinusoid_diameter=params.get('sinusoid_diameter', 12.0),
        sinusoid_length=length_um,
        sinusoid_wall_thickness=params.get('sinusoid_wall_thickness', 0.5),

        # Fenestration parameters
        fenestration_diameter=fenestration_size,
        fenestration_density=fenestration_density,
        fenestration_pattern=params.get('fenestration_pattern', 'clustered'),
        sieve_plate_count=params.get('sieve_plate_count', 8),
        fenestration_porosity=params.get('fenestration_porosity', 0.06),

        # Space of Disse
        enable_space_of_disse=params.get('enable_space_of_disse', True),
        space_of_disse_thickness=params.get('space_of_disse_thickness', 0.5),
        space_of_disse_porosity=params.get('space_of_disse_porosity', 0.8),

        # Hepatocyte zone
        enable_hepatocyte_zone=params.get('enable_hepatocyte_zone', False),
        hepatocyte_diameter=params.get('hepatocyte_diameter', 25.0),
        hepatocyte_spacing=params.get('hepatocyte_spacing', 5.0),
        hepatocytes_per_sinusoid=params.get('hepatocytes_per_sinusoid', 12),

        # Cell type zones
        enable_kupffer_cell_zones=params.get('enable_kupffer_cell_zones', False),
        kupffer_cell_density=params.get('kupffer_cell_density', 0.15),
        enable_stellate_cell_zones=params.get('enable_stellate_cell_zones', False),
        stellate_cell_spacing=params.get('stellate_cell_spacing', 50.0),

        # Sinusoid network
        sinusoid_count=params.get('sinusoid_count', 1),
        sinusoid_spacing=params.get('sinusoid_spacing', 50.0),
        network_pattern=params.get('network_pattern', 'parallel'),
        enable_central_vein_connection=params.get('enable_central_vein_connection', False),
        central_vein_diameter=params.get('central_vein_diameter', 30.0),

        # Scaffold structure
        scaffold_length=params.get('scaffold_length', 2.0),
        enable_scaffold_shell=params.get('enable_scaffold_shell', False),
        shell_thickness=params.get('shell_thickness', 50.0),
        shell_porosity=params.get('shell_porosity', 0.5),

        # Bile canaliculi
        enable_bile_canaliculi=params.get('enable_bile_canaliculi', False),
        canaliculus_diameter=params.get('canaliculus_diameter', 1.0),
        canaliculus_spacing=params.get('canaliculus_spacing', 25.0),

        # ECM components
        enable_ecm_fibers=params.get('enable_ecm_fibers', False),
        ecm_fiber_diameter=params.get('ecm_fiber_diameter', 0.5),
        ecm_fiber_density=params.get('ecm_fiber_density', 0.2),

        # Stochastic variation
        diameter_variance=params.get('diameter_variance', 0.0),
        fenestration_variance=params.get('fenestration_variance', 0.0),
        position_noise=params.get('position_noise', 0.0),
        seed=params.get('seed', 42),

        # Resolution
        resolution=params.get('resolution', 12)
    ))
