"""
Lung alveoli scaffold generator.

Creates branching airway structures terminating in alveolar sacs:
- Recursive branching tree (dichotomous branching) up to 23 generations
- Airways narrow with each generation following Murray's law
- Spherical alveoli at terminal branches (~200um diameter)
- Thin alveolar walls (0.2-0.6um native, scaled for scaffold)
- Blood-air barrier representation
- Capillary network integration for gas exchange
"""

from __future__ import annotations
import manifold3d as m3d
import numpy as np
from dataclasses import dataclass
from ..core import batch_union


@dataclass
class LungAlveoliParams:
    """
    Parameters for lung alveoli scaffold generation.

    Based on native pulmonary anatomy:
    - Total airway generations: 23 (trachea to alveolar sacs)
    - Alveolar diameter: 200-300 um
    - Alveolar wall thickness: 0.2-0.6 um (blood-air barrier)
    - Terminal bronchiole: ~600-1000 um diameter
    - Total alveolar surface area: ~70 m^2
    """
    # === Alveolar Geometry ===
    alveolar_diameter: float = 200.0  # um (200-300 um typical)
    alveolar_wall_thickness: float = 0.6  # um (0.2-0.6 um native, scaled for scaffold)
    alveolar_depth: float = 150.0  # um (depth of alveolar sac)
    alveoli_per_duct: int = 6  # number of alveoli around each alveolar duct

    # === Airway Tree ===
    total_branching_generations: int = 23  # native has 23 generations
    scaffold_generations: int = 3  # generations to actually model (max 5 for performance)
    terminal_bronchiole_diameter: float = 750.0  # um (600-1000 um native)
    airway_diameter: float = 1.5  # mm (starting airway, scaled from trachea)
    branch_angle: float = 35.0  # degrees (typically 30-45 deg)
    branch_length_ratio: float = 0.8  # child length / parent length

    # === Murray's Law Branching ===
    branching_ratio: float = 0.79  # Murray's law: r_child = r_parent * 0.79
    enable_asymmetric_branching: bool = False  # different angles for daughter branches
    asymmetry_factor: float = 0.1  # 0-1 difference between branches

    # === Porosity and Structure ===
    porosity: float = 0.85  # high porosity for gas exchange
    pore_interconnectivity: float = 0.95  # connection between alveoli
    enable_pores_of_kohn: bool = True  # inter-alveolar pores
    pore_of_kohn_diameter: float = 10.0  # um (8-12 um native)
    pores_per_alveolus: int = 3  # average pores of Kohn per alveolus

    # === Blood-Air Barrier ===
    enable_blood_air_barrier: bool = False  # explicit barrier layer
    type_1_cell_coverage: float = 0.95  # Type I pneumocyte coverage
    type_2_cell_coverage: float = 0.06  # Type II pneumocyte (surfactant, 5-7% lit range)
    type_2_cell_diameter: float = 9.0  # um (cuboidal cells ~9-10 um)

    # === Capillary Network ===
    enable_capillary_network: bool = False
    capillary_diameter: float = 8.0  # um (5-10 um)
    capillary_density: float = 2800.0  # per mm^2 (very high in alveoli)
    capillary_spacing: float = 15.0  # um

    # === Alveolar Duct Features ===
    enable_alveolar_ducts: bool = True  # transition zone
    alveolar_duct_length: float = 1.0  # mm (1-2 mm native)
    alveolar_duct_diameter: float = 400.0  # um

    # === Scaffold Bounding ===
    bounding_box: tuple[float, float, float] = (8.0, 8.0, 8.0)  # mm

    # === Surfactant Layer (conceptual) ===
    surfactant_layer_thickness: float = 0.15  # um (100-200 nm native hypophase)

    # === Stochastic Variation ===
    size_variance: float = 0.0  # 0-1 variation in alveolar size
    position_noise: float = 0.0  # 0-1 variation in positions
    angle_noise: float = 0.0  # 0-1 variation in branch angles
    seed: int = 42

    # === Resolution ===
    resolution: int = 10  # sphere/cylinder segments


class BranchNode:
    """Represents a node in the branching tree."""
    def __init__(self, position: np.ndarray, radius: float, generation: int):
        self.position = position
        self.radius = radius
        self.generation = generation
        self.children = []


def create_airway_segment(start: np.ndarray, end: np.ndarray, radius: float, resolution: int) -> m3d.Manifold:
    """
    Create an airway segment (cylinder) between two points.

    Args:
        start: Start position (x, y, z)
        end: End position (x, y, z)
        radius: Airway radius
        resolution: Number of segments around cylinder

    Returns:
        Airway cylinder manifold
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dz = end[2] - start[2]
    length = np.sqrt(dx*dx + dy*dy + dz*dz)

    if length < 1e-6:
        return m3d.Manifold()

    cyl = m3d.Manifold.cylinder(length, radius, radius, resolution)

    h = np.sqrt(dx*dx + dy*dy)
    if h > 0.001 or abs(dz) > 0.001:
        tilt = np.arctan2(h, dz) * 180 / np.pi
        azim = np.arctan2(dy, dx) * 180 / np.pi
        cyl = cyl.rotate([0, tilt, 0]).rotate([0, 0, azim])

    return cyl.translate([start[0], start[1], start[2]])


def create_alveolus(position: np.ndarray, radius: float, wall_thickness: float,
                    resolution: int, enable_hollow: bool = True,
                    depth_ratio: float = 1.0,
                    surfactant_thickness: float = 0.0) -> m3d.Manifold:
    """
    Create a spherical alveolus, optionally hollow with depth modulation and surfactant layer.

    Args:
        position: Center position (x, y, z)
        radius: Alveolus outer radius
        wall_thickness: Wall thickness (mm)
        resolution: Sphere resolution
        enable_hollow: Create hollow alveolus
        depth_ratio: Ratio of alveolar_depth to alveolar_diameter (0.5-1.0 typical)
                     Values < 1.0 create shallower, more dimple-like alveoli
        surfactant_thickness: Thickness of surfactant layer inside alveolus (mm)

    Returns:
        Sphere manifold (hollow or solid) with optional depth modulation
    """
    # Apply depth ratio - creates shallower alveoli when depth < diameter
    # A ratio of 0.75 (150um depth / 200um diameter) creates a dimpled shape
    if depth_ratio < 1.0 and depth_ratio > 0.0:
        # Create ellipsoid-like shape with reduced depth (z-axis)
        # Use scaling to create the dimple effect
        outer = m3d.Manifold.sphere(radius, resolution)
        outer = outer.scale([1.0, 1.0, depth_ratio])
    else:
        outer = m3d.Manifold.sphere(radius, resolution)

    if enable_hollow and wall_thickness > 0:
        inner_radius = radius - wall_thickness
        if inner_radius > 0:
            inner = m3d.Manifold.sphere(inner_radius, resolution)
            if depth_ratio < 1.0 and depth_ratio > 0.0:
                inner = inner.scale([1.0, 1.0, depth_ratio])
            result = outer - inner

            # Add surfactant layer if specified (thin shell at air-tissue interface)
            if surfactant_thickness > 0 and inner_radius > surfactant_thickness:
                surfactant_outer_radius = inner_radius
                surfactant_inner_radius = inner_radius - surfactant_thickness
                if surfactant_inner_radius > 0:
                    surf_outer = m3d.Manifold.sphere(surfactant_outer_radius, resolution)
                    surf_inner = m3d.Manifold.sphere(surfactant_inner_radius, resolution)
                    if depth_ratio < 1.0 and depth_ratio > 0.0:
                        surf_outer = surf_outer.scale([1.0, 1.0, depth_ratio])
                        surf_inner = surf_inner.scale([1.0, 1.0, depth_ratio])
                    surfactant_layer = surf_outer - surf_inner
                    result = result + surfactant_layer
        else:
            result = outer
    else:
        result = outer

    return result.translate([position[0], position[1], position[2]])


def create_pore_of_kohn(pos1: np.ndarray, pos2: np.ndarray,
                        pore_diameter: float, resolution: int) -> m3d.Manifold:
    """
    Create a Pore of Kohn (inter-alveolar pore) connecting two alveoli.

    Pores of Kohn are small cylindrical channels (8-12um native) that connect
    adjacent alveoli, enabling collateral ventilation.

    Args:
        pos1: Center of first alveolus
        pos2: Center of second alveolus
        pore_diameter: Diameter of the pore (mm)
        resolution: Cylinder resolution

    Returns:
        Cylindrical pore manifold
    """
    direction = pos2 - pos1
    distance = np.linalg.norm(direction)

    if distance < 1e-6:
        return m3d.Manifold()

    pore_radius = pore_diameter / 2.0

    # Create cylinder along z-axis, then orient
    cyl = m3d.Manifold.cylinder(distance, pore_radius, pore_radius, resolution)

    # Calculate rotation to align with direction
    dx, dy, dz = direction
    h = np.sqrt(dx*dx + dy*dy)

    if h > 0.001 or abs(dz) > 0.001:
        tilt = np.arctan2(h, dz) * 180 / np.pi
        azim = np.arctan2(dy, dx) * 180 / np.pi
        cyl = cyl.rotate([0, tilt, 0]).rotate([0, 0, azim])

    return cyl.translate([pos1[0], pos1[1], pos1[2]])


def generate_pores_of_kohn(alveoli_positions: list[np.ndarray],
                           alveolar_radius: float,
                           pore_diameter: float,
                           pores_per_alveolus: int,
                           resolution: int,
                           seed: int) -> list[m3d.Manifold]:
    """
    Generate Pores of Kohn between adjacent alveoli.

    Finds nearby alveoli pairs and creates small connecting channels.
    Native pores are 8-12um diameter, with 2-4 pores per alveolus.

    Args:
        alveoli_positions: List of alveoli center positions
        alveolar_radius: Radius of each alveolus (mm)
        pore_diameter: Diameter of pores (mm)
        pores_per_alveolus: Target number of pores per alveolus
        resolution: Cylinder resolution
        seed: Random seed

    Returns:
        List of pore manifolds
    """
    np.random.seed(seed)
    pores = []

    if len(alveoli_positions) < 2:
        return pores

    # Maximum distance for adjacency (touching or slightly overlapping)
    max_adjacent_dist = alveolar_radius * 3.5

    # Track connections to limit pores per alveolus
    connection_count = {i: 0 for i in range(len(alveoli_positions))}
    connected_pairs = set()

    # Find all adjacent pairs
    adjacent_pairs = []
    for i, pos1 in enumerate(alveoli_positions):
        for j, pos2 in enumerate(alveoli_positions):
            if i >= j:
                continue
            dist = np.linalg.norm(pos2 - pos1)
            if dist < max_adjacent_dist:
                adjacent_pairs.append((i, j, dist))

    # Sort by distance (prioritize closer pairs)
    adjacent_pairs.sort(key=lambda x: x[2])

    # Create pores for adjacent pairs
    for i, j, dist in adjacent_pairs:
        # Check if either alveolus has reached pore limit
        if connection_count[i] >= pores_per_alveolus or connection_count[j] >= pores_per_alveolus:
            continue

        pair_key = (min(i, j), max(i, j))
        if pair_key in connected_pairs:
            continue

        pos1 = alveoli_positions[i]
        pos2 = alveoli_positions[j]

        # Calculate pore position on alveolar walls
        direction = pos2 - pos1
        direction_norm = direction / np.linalg.norm(direction)

        # Pore starts at surface of alveolus 1, ends at surface of alveolus 2
        pore_start = pos1 + direction_norm * alveolar_radius * 0.9
        pore_end = pos2 - direction_norm * alveolar_radius * 0.9

        pore = create_pore_of_kohn(pore_start, pore_end, pore_diameter, resolution)
        if pore.num_vert() > 0:
            pores.append(pore)
            connection_count[i] += 1
            connection_count[j] += 1
            connected_pairs.add(pair_key)

    return pores


def create_wall_porosity_pores(position: np.ndarray, alveolar_radius: float,
                                wall_thickness: float, porosity: float,
                                resolution: int, seed: int) -> list[m3d.Manifold]:
    """
    Create small pores in the alveolar wall to achieve target porosity.

    Wall porosity enables better gas and nutrient diffusion through the scaffold.
    The pores are distributed across the alveolar surface.

    Args:
        position: Center of the alveolus
        alveolar_radius: Outer radius of the alveolus (mm)
        wall_thickness: Wall thickness (mm)
        porosity: Target porosity (0.0-1.0, e.g., 0.85 = 85% void)
        resolution: Resolution for pore geometry
        seed: Random seed

    Returns:
        List of small cylindrical pore manifolds to subtract from walls
    """
    np.random.seed(seed)
    pores = []

    # Only create wall pores if porosity > 0
    if porosity <= 0:
        return pores

    # Calculate pore parameters based on porosity
    # Higher porosity = more and/or larger pores
    # Use porosity to determine pore density (number per surface area)
    surface_area = 4 * np.pi * alveolar_radius * alveolar_radius  # mm^2

    # Pore diameter scales with wall thickness
    pore_diameter = wall_thickness * 0.6  # 60% of wall thickness
    pore_radius = pore_diameter / 2.0

    # Number of pores based on porosity target (more porosity = more pores)
    # Target: each pore removes area ~ pi*r^2 from wall
    pore_area = np.pi * pore_radius * pore_radius
    # Scale number of pores by porosity fraction (0.85 porosity = many pores)
    num_pores = max(3, int(surface_area * porosity * 0.5 / pore_area))
    num_pores = min(num_pores, 30)  # Cap for performance

    # Distribute pores on alveolar surface using Fibonacci sphere
    golden_ratio = (1 + np.sqrt(5)) / 2
    angle_increment = 2 * np.pi * golden_ratio

    for i in range(num_pores):
        # Fibonacci distribution with slight randomness
        t = (i + np.random.uniform(0.1, 0.9)) / num_pores
        inclination = np.arccos(1 - 2 * t)
        azimuth = angle_increment * i + np.random.uniform(-0.3, 0.3)

        # Calculate position on surface
        x = alveolar_radius * np.sin(inclination) * np.cos(azimuth)
        y = alveolar_radius * np.sin(inclination) * np.sin(azimuth)
        z = alveolar_radius * np.cos(inclination)

        surface_pos = position + np.array([x, y, z])

        # Direction pointing inward (toward center)
        direction = position - surface_pos
        direction = direction / (np.linalg.norm(direction) + 1e-10)

        # Create cylindrical pore through the wall
        pore_length = wall_thickness * 1.2  # Slightly longer than wall
        cyl = m3d.Manifold.cylinder(pore_length, pore_radius, pore_radius, max(4, resolution // 2))

        # Orient pore to point toward center
        dx, dy, dz = direction
        h = np.sqrt(dx*dx + dy*dy)
        if h > 0.001 or abs(dz) > 0.001:
            tilt = np.arctan2(h, dz) * 180 / np.pi
            azim_rot = np.arctan2(dy, dx) * 180 / np.pi
            cyl = cyl.rotate([0, tilt, 0]).rotate([0, 0, azim_rot])

        # Position at outer surface
        pore_start = surface_pos + direction * (pore_length * 0.1)
        cyl = cyl.translate([pore_start[0], pore_start[1], pore_start[2]])

        if cyl.num_vert() > 0:
            pores.append(cyl)

    return pores


def create_interconnectivity_channels(alveoli_positions: list[np.ndarray],
                                       alveolar_radius: float,
                                       interconnectivity: float,
                                       channel_diameter: float,
                                       resolution: int,
                                       seed: int) -> list[m3d.Manifold]:
    """
    Create interconnecting channels between adjacent alveoli based on interconnectivity fraction.

    Unlike Pores of Kohn (which are small anatomical pores), these represent
    general scaffold interconnectivity for nutrient/cell transport.

    Args:
        alveoli_positions: List of alveoli center positions
        alveolar_radius: Radius of each alveolus (mm)
        interconnectivity: Fraction of possible connections to create (0.0-1.0)
        channel_diameter: Diameter of interconnecting channels (mm)
        resolution: Cylinder resolution
        seed: Random seed

    Returns:
        List of channel manifolds
    """
    np.random.seed(seed)
    channels = []

    if len(alveoli_positions) < 2 or interconnectivity <= 0:
        return channels

    channel_radius = channel_diameter / 2.0

    # Maximum distance for potential connection (adjacent or nearby alveoli)
    max_connect_dist = alveolar_radius * 4.0

    # Find all possible connection pairs
    possible_pairs = []
    for i, pos1 in enumerate(alveoli_positions):
        for j, pos2 in enumerate(alveoli_positions):
            if i >= j:
                continue
            dist = np.linalg.norm(pos2 - pos1)
            if dist < max_connect_dist:
                possible_pairs.append((i, j, dist))

    # Sort by distance (connect closer pairs first)
    possible_pairs.sort(key=lambda x: x[2])

    # Create channels for fraction of pairs based on interconnectivity
    num_channels = max(1, int(len(possible_pairs) * interconnectivity))

    for idx in range(min(num_channels, len(possible_pairs))):
        i, j, dist = possible_pairs[idx]
        pos1 = alveoli_positions[i]
        pos2 = alveoli_positions[j]

        # Calculate channel endpoints at alveolar surfaces
        direction = pos2 - pos1
        direction_norm = direction / (np.linalg.norm(direction) + 1e-10)

        # Start and end just inside alveolar surfaces
        channel_start = pos1 + direction_norm * alveolar_radius * 0.85
        channel_end = pos2 - direction_norm * alveolar_radius * 0.85

        channel_length = np.linalg.norm(channel_end - channel_start)
        if channel_length < 1e-6:
            continue

        # Create channel cylinder
        cyl = m3d.Manifold.cylinder(channel_length, channel_radius, channel_radius, max(4, resolution // 2))

        # Orient cylinder
        dx, dy, dz = direction_norm
        h = np.sqrt(dx*dx + dy*dy)
        if h > 0.001 or abs(dz) > 0.001:
            tilt = np.arctan2(h, dz) * 180 / np.pi
            azim = np.arctan2(dy, dx) * 180 / np.pi
            cyl = cyl.rotate([0, tilt, 0]).rotate([0, 0, azim])

        cyl = cyl.translate([channel_start[0], channel_start[1], channel_start[2]])

        if cyl.num_vert() > 0:
            channels.append(cyl)

    return channels


def create_blood_air_barrier(position: np.ndarray, alveolar_radius: float,
                              barrier_thickness: float, type_1_coverage: float,
                              type_2_coverage: float, type_2_cell_diameter: float,
                              resolution: int, seed: int) -> tuple[m3d.Manifold, list[m3d.Manifold]]:
    """
    Create the blood-air barrier layer lining the alveolus interior.

    The blood-air barrier is the thinnest membrane in the body (0.2-0.6um native),
    composed of:
    - Type I pneumocytes: thin squamous cells covering 95% of surface (gas exchange)
    - Type II pneumocytes: cuboidal cells covering 5% of surface (surfactant production)

    Args:
        position: Center of the alveolus
        alveolar_radius: Outer radius of the alveolus (mm)
        barrier_thickness: Thickness of the barrier layer (mm)
        type_1_coverage: Fraction of surface covered by Type I cells (0-1, typically 0.95)
                         Used to determine Type II marker density (1 - type_1_coverage)
        type_2_coverage: Fraction of surface covered by Type II cells (0-1, typically 0.05)
        type_2_cell_diameter: Diameter of Type II cells (um)
        resolution: Sphere resolution
        seed: Random seed for Type II cell distribution

    Returns:
        Tuple of (barrier_shell, type_2_bumps)
        - barrier_shell: Thin shell representing the main barrier
        - type_2_bumps: List of small protrusions for Type II cell locations
    """
    np.random.seed(seed)

    # Create thin shell for the barrier (just inside the alveolar wall)
    inner_radius = alveolar_radius - barrier_thickness
    barrier_radius = inner_radius - barrier_thickness

    if barrier_radius <= 0 or inner_radius <= 0:
        return m3d.Manifold(), []

    outer_shell = m3d.Manifold.sphere(inner_radius, resolution)
    inner_shell = m3d.Manifold.sphere(barrier_radius, resolution)
    barrier = outer_shell - inner_shell

    # Create Type II cell bumps (cuboidal cells appearing as small protrusions)
    type_2_bumps = []

    # Use type_1_cell_coverage to determine Type II density
    # If type_1_coverage is high (0.95), fewer Type II cells
    # If type_1_coverage is lower, more Type II cells
    effective_type_2_coverage = min(type_2_coverage, 1.0 - type_1_coverage)

    # Number of Type II cells based on coverage and surface area
    surface_area = 4 * np.pi * inner_radius * inner_radius
    type_2_cell_size = type_2_cell_diameter / 1000.0  # Convert um to mm
    cells_per_area = effective_type_2_coverage / (type_2_cell_size * type_2_cell_size)
    num_type_2_cells = max(1, int(surface_area * cells_per_area * 10))  # Scale for visibility
    num_type_2_cells = min(num_type_2_cells, 20)  # Cap for performance

    # Distribute Type II cells randomly on inner surface
    golden_ratio = (1 + np.sqrt(5)) / 2
    angle_increment = 2 * np.pi * golden_ratio

    for i in range(num_type_2_cells):
        # Fibonacci sphere distribution with randomness
        t = (i + np.random.uniform(0, 0.5)) / num_type_2_cells
        inclination = np.arccos(1 - 2 * t)
        azimuth = angle_increment * i + np.random.uniform(-0.2, 0.2)

        # Position on inner surface
        x = barrier_radius * np.sin(inclination) * np.cos(azimuth)
        y = barrier_radius * np.sin(inclination) * np.sin(azimuth)
        z = barrier_radius * np.cos(inclination)

        # Create small bump for Type II cell
        bump_radius = type_2_cell_size
        bump = m3d.Manifold.sphere(bump_radius, max(4, resolution // 2))

        bump_pos = position + np.array([x, y, z])
        bump = bump.translate([bump_pos[0], bump_pos[1], bump_pos[2]])

        if bump.num_vert() > 0:
            type_2_bumps.append(bump)

    barrier = barrier.translate([position[0], position[1], position[2]])

    return barrier, type_2_bumps


def generate_blood_air_barriers(alveoli_data: list[tuple],
                                 barrier_thickness: float,
                                 type_1_coverage: float,
                                 type_2_coverage: float,
                                 type_2_cell_diameter: float,
                                 resolution: int,
                                 seed: int) -> tuple[list[m3d.Manifold], list[m3d.Manifold]]:
    """
    Generate blood-air barriers for all alveoli.

    Args:
        alveoli_data: List of (position, radius, wall_thickness) tuples
        barrier_thickness: Thickness of barrier layer (mm)
        type_1_coverage: Fraction of Type I cell coverage (0-1, typically 0.95)
        type_2_coverage: Fraction of Type II cell coverage (0-1, typically 0.05)
        type_2_cell_diameter: Diameter of Type II cells (um)
        resolution: Sphere resolution
        seed: Random seed

    Returns:
        Tuple of (barrier_shells, type_2_bumps_all)
    """
    barriers = []
    all_type_2_bumps = []

    for idx, (pos, radius, _wall_thick) in enumerate(alveoli_data):
        barrier, type_2_bumps = create_blood_air_barrier(
            pos, radius, barrier_thickness, type_1_coverage, type_2_coverage,
            type_2_cell_diameter, resolution, seed + idx
        )
        if barrier.num_vert() > 0:
            barriers.append(barrier)
        all_type_2_bumps.extend(type_2_bumps)

    return barriers, all_type_2_bumps


def create_capillary_segment(start: np.ndarray, end: np.ndarray,
                              capillary_radius: float, resolution: int) -> m3d.Manifold:
    """
    Create a single capillary segment (small tube).

    Args:
        start: Start position
        end: End position
        capillary_radius: Radius of capillary (mm)
        resolution: Cylinder resolution

    Returns:
        Capillary cylinder manifold
    """
    direction = end - start
    length = np.linalg.norm(direction)

    if length < 1e-6:
        return m3d.Manifold()

    cyl = m3d.Manifold.cylinder(length, capillary_radius, capillary_radius, max(4, resolution // 2))

    dx, dy, dz = direction
    h = np.sqrt(dx*dx + dy*dy)

    if h > 0.001 or abs(dz) > 0.001:
        tilt = np.arctan2(h, dz) * 180 / np.pi
        azim = np.arctan2(dy, dx) * 180 / np.pi
        cyl = cyl.rotate([0, tilt, 0]).rotate([0, 0, azim])

    return cyl.translate([start[0], start[1], start[2]])


def create_capillary_network_for_alveolus(position: np.ndarray, alveolar_radius: float,
                                           capillary_diameter: float, capillary_spacing: float,
                                           capillary_density: float,
                                           resolution: int, seed: int) -> list[m3d.Manifold]:
    """
    Create a capillary network mesh wrapping around a single alveolus.

    The pulmonary capillary network has the highest density in the body (~2800/mm2),
    forming a dense mesh on the exterior of alveoli. Capillaries are positioned
    in a single layer between adjacent alveoli to maximize gas exchange efficiency.

    Args:
        position: Center of the alveolus
        alveolar_radius: Radius of the alveolus (mm)
        capillary_diameter: Diameter of capillaries (mm)
        capillary_spacing: Base spacing between capillaries (mm)
        capillary_density: Target capillary density per mm^2 (e.g., 2800)
        resolution: Mesh resolution
        seed: Random seed

    Returns:
        List of capillary segment manifolds
    """
    np.random.seed(seed)
    capillaries = []

    capillary_radius = capillary_diameter / 2.0
    network_radius = alveolar_radius * 1.05  # Just outside alveolar surface

    # Calculate surface area of the alveolus (mm^2)
    surface_area = 4 * np.pi * alveolar_radius * alveolar_radius

    # Calculate effective spacing from capillary_density
    # capillary_density is capillaries per mm^2
    # Each capillary segment spans an arc, roughly covering area = spacing^2
    # Total capillaries = density * surface_area
    # Grid points ~ sqrt(total_capillaries) per dimension
    if capillary_density > 0:
        total_capillaries_target = capillary_density * surface_area
        # Each grid cell creates ~2 capillary segments (lat + lon)
        grid_points_total = total_capillaries_target / 2
        grid_points_per_dim = max(3, int(np.sqrt(grid_points_total)))
        # Derive effective spacing from grid count
        circumference = 2 * np.pi * network_radius
        effective_spacing = circumference / grid_points_per_dim
    else:
        effective_spacing = capillary_spacing

    # Calculate number of latitude/longitude rings based on effective spacing
    circumference = 2 * np.pi * network_radius
    num_rings = max(3, int(circumference / effective_spacing))
    points_per_ring = max(4, int(circumference / effective_spacing))

    # Cap for performance
    num_rings = min(num_rings, 20)
    points_per_ring = min(points_per_ring, 20)

    # Generate grid points on sphere surface
    grid_points = []
    for lat_idx in range(num_rings):
        phi = np.pi * (lat_idx + 0.5) / num_rings  # Latitude angle

        for lon_idx in range(points_per_ring):
            theta = 2 * np.pi * lon_idx / points_per_ring  # Longitude angle

            # Spherical to Cartesian
            x = network_radius * np.sin(phi) * np.cos(theta)
            y = network_radius * np.sin(phi) * np.sin(theta)
            z = network_radius * np.cos(phi)

            grid_points.append((lat_idx, lon_idx, position + np.array([x, y, z])))

    # Create capillary segments connecting adjacent grid points
    # Latitude connections (horizontal rings)
    for lat_idx in range(num_rings):
        ring_points = [p for p in grid_points if p[0] == lat_idx]
        for i in range(len(ring_points)):
            p1 = ring_points[i][2]
            p2 = ring_points[(i + 1) % len(ring_points)][2]

            cap = create_capillary_segment(p1, p2, capillary_radius, resolution)
            if cap.num_vert() > 0:
                capillaries.append(cap)

    # Longitude connections (vertical arcs) - only create some to avoid overcrowding
    for lon_idx in range(0, points_per_ring, 2):  # Every other longitude
        lon_points = [p for p in grid_points if p[1] == lon_idx]
        lon_points.sort(key=lambda x: x[0])

        for i in range(len(lon_points) - 1):
            p1 = lon_points[i][2]
            p2 = lon_points[i + 1][2]

            cap = create_capillary_segment(p1, p2, capillary_radius, resolution)
            if cap.num_vert() > 0:
                capillaries.append(cap)

    return capillaries


def generate_capillary_network(alveoli_data: list[tuple],
                                capillary_diameter: float,
                                capillary_spacing: float,
                                capillary_density: float,
                                resolution: int,
                                seed: int) -> list[m3d.Manifold]:
    """
    Generate capillary network for all alveoli.

    Creates a mesh of small tubes wrapping around the exterior of each alveolus.
    Native pulmonary capillary density is ~2800/mm2 (highest in body).

    Args:
        alveoli_data: List of (position, radius, wall_thickness) tuples
        capillary_diameter: Diameter of capillaries (mm)
        capillary_spacing: Base spacing between capillaries (mm)
        capillary_density: Target capillary density per mm^2 (e.g., 2800)
        resolution: Mesh resolution
        seed: Random seed

    Returns:
        List of capillary manifolds
    """
    all_capillaries = []

    # Reduce resolution for capillary network (performance)
    cap_resolution = max(4, resolution // 2)

    for idx, (pos, radius, _wall_thick) in enumerate(alveoli_data):
        caps = create_capillary_network_for_alveolus(
            pos, radius, capillary_diameter, capillary_spacing,
            capillary_density, cap_resolution, seed + idx * 1000
        )
        all_capillaries.extend(caps)

    return all_capillaries


def create_alveolar_duct(position: np.ndarray, direction: np.ndarray,
                          duct_length: float, duct_diameter: float,
                          alveolar_radius: float, wall_thickness: float,
                          alveoli_per_duct: int, resolution: int,
                          seed: int) -> tuple[m3d.Manifold, list[tuple], list[np.ndarray]]:
    """
    Create an alveolar duct with alveoli opening off the walls.

    Alveolar ducts are the transition zone between respiratory bronchioles and
    alveolar sacs. Native ducts are 400-600um diameter, 1-2mm long, with walls
    completely covered by alveolar openings.

    Args:
        position: Start position of the duct
        direction: Direction vector of the duct
        duct_length: Length of the duct (mm)
        duct_diameter: Diameter of the duct (mm)
        alveolar_radius: Radius of individual alveoli (mm)
        wall_thickness: Wall thickness for alveoli (mm)
        alveoli_per_duct: Number of alveoli around the duct
        resolution: Cylinder resolution
        seed: Random seed

    Returns:
        Tuple of (duct_manifold, alveoli_data, alveoli_positions)
        - duct_manifold: The cylindrical duct
        - alveoli_data: List of (position, radius, wall_thickness) for alveoli
        - alveoli_positions: List of alveoli center positions (for pore generation)
    """
    np.random.seed(seed)

    duct_radius = duct_diameter / 2000.0  # um to mm
    direction = direction / (np.linalg.norm(direction) + 1e-10)

    # Create the duct cylinder
    duct = m3d.Manifold.cylinder(duct_length, duct_radius, duct_radius, resolution)

    # Orient the duct along the direction
    dx, dy, dz = direction
    h = np.sqrt(dx*dx + dy*dy)

    if h > 0.001 or abs(dz) > 0.001:
        tilt = np.arctan2(h, dz) * 180 / np.pi
        azim = np.arctan2(dy, dx) * 180 / np.pi
        duct = duct.rotate([0, tilt, 0]).rotate([0, 0, azim])

    duct = duct.translate([position[0], position[1], position[2]])

    # Find perpendicular vectors to direction
    if abs(direction[2]) < 0.9:
        perp1 = np.array([0, 0, 1])
    else:
        perp1 = np.array([1, 0, 0])

    perp1 = perp1 - np.dot(perp1, direction) * direction
    perp1 = perp1 / (np.linalg.norm(perp1) + 1e-10)
    perp2 = np.cross(direction, perp1)

    # Generate alveoli positions along the duct
    alveoli_data = []
    alveoli_positions = []

    # Rows along the duct
    num_rows = max(2, int(duct_length / (alveolar_radius * 2.5)))
    alveoli_per_row = max(3, alveoli_per_duct // num_rows)

    for row in range(num_rows):
        # Position along duct
        t = (row + 0.5) / num_rows
        duct_pos = position + direction * duct_length * t

        # Distribute alveoli around the duct circumference
        for i in range(alveoli_per_row):
            angle = 2 * np.pi * i / alveoli_per_row
            angle += row * np.pi / alveoli_per_row  # Stagger rows

            # Position on duct surface, pointing outward
            radial_dir = perp1 * np.cos(angle) + perp2 * np.sin(angle)
            alv_center = duct_pos + radial_dir * (duct_radius + alveolar_radius * 0.8)

            alveoli_data.append((alv_center, alveolar_radius, wall_thickness))
            alveoli_positions.append(alv_center)

    return duct, alveoli_data, alveoli_positions


def generate_alveolar_ducts(terminal_positions: list[tuple],
                             params: LungAlveoliParams) -> tuple[list[m3d.Manifold], list[tuple], list[np.ndarray]]:
    """
    Generate alveolar ducts at terminal branch positions.

    Args:
        terminal_positions: List of (position, direction, radius) for terminal branches
        params: LungAlveoliParams with duct settings

    Returns:
        Tuple of (duct_manifolds, all_alveoli_data, all_alveoli_positions)
    """
    ducts = []
    all_alveoli_data = []
    all_alveoli_positions = []

    duct_length = params.alveolar_duct_length
    duct_diameter = params.alveolar_duct_diameter
    alveolar_radius = params.alveolar_diameter / 2000.0
    wall_thickness = params.alveolar_wall_thickness / 1000.0

    for idx, (pos, direction, _radius) in enumerate(terminal_positions):
        duct, alveoli_data, alveoli_positions = create_alveolar_duct(
            pos, direction, duct_length, duct_diameter,
            alveolar_radius, wall_thickness,
            params.alveoli_per_duct, params.resolution, params.seed + idx
        )

        if duct.num_vert() > 0:
            ducts.append(duct)

        all_alveoli_data.extend(alveoli_data)
        all_alveoli_positions.extend(alveoli_positions)

    return ducts, all_alveoli_data, all_alveoli_positions


def create_alveolar_cluster(center: np.ndarray, alveolar_radius: float,
                            wall_thickness: float, count: int,
                            resolution: int, seed: int) -> list[m3d.Manifold]:
    """
    Create a cluster of alveoli around a central point.

    Args:
        center: Center of the cluster
        alveolar_radius: Radius of each alveolus (mm)
        wall_thickness: Wall thickness (mm)
        count: Number of alveoli in cluster
        resolution: Sphere resolution
        seed: Random seed

    Returns:
        List of alveolus manifolds
    """
    np.random.seed(seed)
    alveoli = []

    # Distribute alveoli on sphere surface around center
    golden_ratio = (1 + np.sqrt(5)) / 2
    angle_increment = 2 * np.pi * golden_ratio

    cluster_radius = alveolar_radius * 1.5

    for i in range(count):
        # Fibonacci sphere distribution
        t = i / count
        inclination = np.arccos(1 - 2 * t)
        azimuth = angle_increment * i

        # Position on sphere surface
        x = cluster_radius * np.sin(inclination) * np.cos(azimuth)
        y = cluster_radius * np.sin(inclination) * np.sin(azimuth)
        z = cluster_radius * np.cos(inclination)

        pos = center + np.array([x, y, z])
        alv = create_alveolus(pos, alveolar_radius, wall_thickness, resolution)
        if alv.num_vert() > 0:
            alveoli.append(alv)

    return alveoli


def generate_branching_tree(root_pos: np.ndarray, root_radius: float, max_generations: int,
                            branch_angle: float, branching_ratio: float,
                            direction: np.ndarray, params: LungAlveoliParams) -> BranchNode:
    """
    Generate a recursive branching tree structure.

    Args:
        root_pos: Starting position
        root_radius: Starting radius
        max_generations: Maximum depth of branching
        branch_angle: Angle between branches (degrees)
        branching_ratio: Murray's law ratio
        direction: Initial direction vector
        params: Full parameters for variation settings

    Returns:
        Root node of branching tree
    """
    np.random.seed(params.seed)
    root = BranchNode(root_pos, root_radius, 0)

    def build_branch(node: BranchNode, parent_dir: np.ndarray):
        if node.generation >= max_generations:
            return

        # Branch length decreases with generation
        length = root_radius * 4.0 * (params.branch_length_ratio ** node.generation)

        # Child radius follows Murray's law
        child_radius = node.radius * branching_ratio

        # Apply angle noise
        effective_angle = branch_angle
        if params.angle_noise > 0:
            effective_angle += np.random.uniform(-1, 1) * params.angle_noise * 10

        angle_rad = np.radians(effective_angle)

        # Normalize parent direction
        parent_dir = parent_dir / (np.linalg.norm(parent_dir) + 1e-10)

        # Find perpendicular vector
        if abs(parent_dir[2]) < 0.9:
            perp = np.array([0, 0, 1])
        else:
            perp = np.array([1, 0, 0])

        right = np.cross(parent_dir, perp)
        right = right / (np.linalg.norm(right) + 1e-10)

        # Create two branches by rotating around 'right' axis
        for idx, sign in enumerate([-1, 1]):
            # Apply asymmetry if enabled
            if params.enable_asymmetric_branching:
                asymm = 1 + sign * params.asymmetry_factor * 0.5
                effective_child_radius = child_radius * asymm
            else:
                effective_child_radius = child_radius

            # Rotate parent_dir around right axis by +/-branch_angle
            child_dir = parent_dir * np.cos(angle_rad)
            child_dir += np.cross(right, parent_dir) * np.sin(angle_rad) * sign
            child_dir += right * np.dot(right, parent_dir) * (1 - np.cos(angle_rad))

            child_dir = child_dir / (np.linalg.norm(child_dir) + 1e-10)

            # Apply position noise
            if params.position_noise > 0:
                noise = np.random.uniform(-1, 1, 3) * params.position_noise * length * 0.1
                child_dir = child_dir + noise
                child_dir = child_dir / np.linalg.norm(child_dir)

            child_pos = node.position + child_dir * length
            child = BranchNode(child_pos, effective_child_radius, node.generation + 1)
            node.children.append(child)

            build_branch(child, child_dir)

    build_branch(root, direction)
    return root


def collect_airways_and_alveoli(root: BranchNode, max_gen: int,
                                params: LungAlveoliParams) -> tuple[list, list, list]:
    """
    Collect all airways, alveoli, and terminal positions from branching tree.

    Args:
        root: Root node of tree
        max_gen: Maximum generation (for terminal detection)
        params: Parameters for alveoli generation

    Returns:
        Tuple of (airway_segments, alveoli_data, terminal_positions)
        - airway_segments: List of (start, end, radius) tuples
        - alveoli_data: List of (position, radius, wall_thickness) tuples for terminal alveoli
        - terminal_positions: List of (position, direction, radius) tuples for duct generation
    """
    airways = []
    alveoli = []
    terminal_positions = []

    alveolar_radius_mm = params.alveolar_diameter / 2000.0
    wall_thickness_mm = params.alveolar_wall_thickness / 1000.0

    def traverse(node: BranchNode, parent_pos: np.ndarray = None):
        for child in node.children:
            # Add airway segment from node to child
            airways.append((node.position, child.position, child.radius))
            traverse(child, node.position)

        # Terminal node - add alveolus cluster
        if node.generation == max_gen:
            # Apply size variance
            if params.size_variance > 0:
                variance = 1 + np.random.uniform(-1, 1) * params.size_variance * 0.2
                alv_radius = alveolar_radius_mm * variance
            else:
                alv_radius = alveolar_radius_mm

            alveoli.append((node.position, alv_radius, wall_thickness_mm))

            # Calculate direction from parent (for duct orientation)
            if parent_pos is not None:
                direction = node.position - parent_pos
                direction = direction / (np.linalg.norm(direction) + 1e-10)
            else:
                direction = np.array([0.0, 0.0, 1.0])

            terminal_positions.append((node.position, direction, node.radius))

    traverse(root)
    return airways, alveoli, terminal_positions


def generate_lung_alveoli(params: LungAlveoliParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate a lung alveoli scaffold with branching airways.

    Features implemented based on parameters:
    - Branching airway tree with Murray's law diameter reduction
    - Alveolar clusters at terminal branches
    - Alveolar ducts (when enable_alveolar_ducts=True)
    - Pores of Kohn for collateral ventilation (when enable_pores_of_kohn=True)
    - Blood-air barrier with Type I/II pneumocytes (when enable_blood_air_barrier=True)
    - Capillary network mesh (when enable_capillary_network=True)

    Args:
        params: LungAlveoliParams specifying geometry

    Returns:
        Tuple of (manifold, stats_dict)
        - manifold: The 3D geometry
        - stats_dict: Dictionary with component counts and statistics

    Raises:
        ValueError: If no geometry is generated
    """
    np.random.seed(params.seed)

    # Convert um to mm
    alveoli_radius = params.alveolar_diameter / 2000.0
    wall_thickness_mm = params.alveolar_wall_thickness / 1000.0
    airway_radius = params.airway_diameter / 2.0
    pore_diameter_mm = params.pore_of_kohn_diameter / 1000.0
    capillary_diameter_mm = params.capillary_diameter / 1000.0
    capillary_spacing_mm = params.capillary_spacing / 1000.0
    barrier_thickness_mm = params.alveolar_wall_thickness / 1000.0 * 0.5  # Barrier is thinner than wall
    surfactant_thickness_mm = params.surfactant_layer_thickness / 1000.0  # Convert um to mm

    # Calculate depth ratio for alveolar shape (alveolar_depth / alveolar_diameter)
    # Default 150um depth / 200um diameter = 0.75 creates slightly flattened alveoli
    depth_ratio = params.alveolar_depth / params.alveolar_diameter if params.alveolar_diameter > 0 else 1.0
    depth_ratio = max(0.3, min(1.0, depth_ratio))  # Clamp to reasonable range

    # Use terminal_bronchiole_diameter to set the starting radius at terminal generation
    # Terminal bronchioles are smaller than main airways
    terminal_bronchiole_radius_mm = params.terminal_bronchiole_diameter / 2000.0  # um to mm

    # Use scaffold_generations (capped for performance)
    effective_generations = min(params.scaffold_generations, 5)

    # Generate branching tree
    # Use terminal_bronchiole_diameter for the root if it's smaller than airway_diameter
    # This makes the scaffold start at terminal bronchiole size (physiologically accurate for alveolar region)
    if terminal_bronchiole_radius_mm < airway_radius:
        # Use terminal bronchiole diameter as starting point (more accurate for alveolar scaffolds)
        effective_start_radius = terminal_bronchiole_radius_mm
    else:
        effective_start_radius = airway_radius

    root_pos = np.array([0.0, 0.0, 0.0])
    initial_dir = np.array([0.0, 0.0, 1.0])

    root = generate_branching_tree(
        root_pos, effective_start_radius, effective_generations,
        params.branch_angle, params.branching_ratio, initial_dir, params
    )

    # Collect airways, alveoli, and terminal positions
    airway_segments, alveoli_data, terminal_positions = collect_airways_and_alveoli(
        root, effective_generations, params
    )

    # Initialize manifold lists
    airway_manifolds = []
    alveoli_manifolds = []
    duct_manifolds = []
    pore_manifolds = []
    barrier_manifolds = []
    type_2_manifolds = []
    capillary_manifolds = []

    # Track all alveoli positions for pore generation
    all_alveoli_positions = []

    # Create airway manifolds
    for start, end, radius in airway_segments:
        seg = create_airway_segment(start, end, radius, params.resolution)
        if seg.num_vert() > 0:
            airway_manifolds.append(seg)

    # === ALVEOLAR DUCTS (when enabled) ===
    final_alveoli_data = []

    if params.enable_alveolar_ducts and terminal_positions:
        # Generate alveolar ducts at terminal branches
        ducts, duct_alveoli_data, duct_alveoli_positions = generate_alveolar_ducts(
            terminal_positions, params
        )
        duct_manifolds.extend(ducts)
        final_alveoli_data.extend(duct_alveoli_data)
        all_alveoli_positions.extend(duct_alveoli_positions)
    else:
        # Use standard alveoli data without ducts
        final_alveoli_data = alveoli_data
        all_alveoli_positions = [pos for pos, _, _ in alveoli_data]

    # Create alveoli manifolds with depth_ratio and surfactant_layer
    wall_porosity_pores = []  # Track wall pores for porosity

    for pos, radius, wall_thick in final_alveoli_data:
        # Create single alveolus or cluster
        if params.alveoli_per_duct > 1 and not params.enable_alveolar_ducts:
            # Only use clusters if ducts are disabled (ducts handle their own alveoli)
            cluster = create_alveolar_cluster(
                pos, radius, wall_thick,
                params.alveoli_per_duct, params.resolution, params.seed
            )
            alveoli_manifolds.extend(cluster)
            # Add cluster positions for pore generation
            cluster_radius = radius * 1.5
            golden_ratio = (1 + np.sqrt(5)) / 2
            for i in range(params.alveoli_per_duct):
                t = i / params.alveoli_per_duct
                inclination = np.arccos(1 - 2 * t)
                azimuth = 2 * np.pi * golden_ratio * i
                x = cluster_radius * np.sin(inclination) * np.cos(azimuth)
                y = cluster_radius * np.sin(inclination) * np.sin(azimuth)
                z = cluster_radius * np.cos(inclination)
                all_alveoli_positions.append(pos + np.array([x, y, z]))
        else:
            # Create alveolus with depth_ratio (affects shape) and surfactant layer
            alv = create_alveolus(
                pos, radius, wall_thick, params.resolution, True,
                depth_ratio=depth_ratio,
                surfactant_thickness=surfactant_thickness_mm
            )
            if alv.num_vert() > 0:
                alveoli_manifolds.append(alv)

            # Create wall porosity pores if porosity > 0
            if params.porosity > 0 and params.porosity < 1.0:
                pores = create_wall_porosity_pores(
                    pos, radius, wall_thick, params.porosity,
                    params.resolution, params.seed + len(wall_porosity_pores)
                )
                wall_porosity_pores.extend(pores)

    # === PORES OF KOHN (when enabled) ===
    if params.enable_pores_of_kohn and len(all_alveoli_positions) >= 2:
        pore_manifolds = generate_pores_of_kohn(
            all_alveoli_positions,
            alveoli_radius,
            pore_diameter_mm,
            params.pores_per_alveolus,
            max(4, params.resolution // 2),
            params.seed
        )

    # === PORE INTERCONNECTIVITY (additional channels based on interconnectivity parameter) ===
    interconnectivity_manifolds = []
    if params.pore_interconnectivity > 0 and len(all_alveoli_positions) >= 2:
        # Create interconnecting channels based on pore_interconnectivity fraction
        # Channel diameter is larger than pores of kohn for nutrient/cell transport
        interconnect_channel_diameter = pore_diameter_mm * 1.5  # Larger than Kohn pores
        interconnectivity_manifolds = create_interconnectivity_channels(
            all_alveoli_positions,
            alveoli_radius,
            params.pore_interconnectivity,
            interconnect_channel_diameter,
            max(4, params.resolution // 2),
            params.seed + 5000
        )

    # === BLOOD-AIR BARRIER (when enabled) ===
    if params.enable_blood_air_barrier and final_alveoli_data:
        barriers, type_2_bumps = generate_blood_air_barriers(
            final_alveoli_data,
            barrier_thickness_mm,
            params.type_1_cell_coverage,  # Pass type_1_coverage to control Type II density
            params.type_2_cell_coverage,
            params.type_2_cell_diameter,
            params.resolution,
            params.seed
        )
        barrier_manifolds.extend(barriers)
        type_2_manifolds.extend(type_2_bumps)

    # === CAPILLARY NETWORK (when enabled) ===
    if params.enable_capillary_network and final_alveoli_data:
        capillary_manifolds = generate_capillary_network(
            final_alveoli_data,
            capillary_diameter_mm,
            capillary_spacing_mm,
            params.capillary_density,  # Pass density to calculate network from density
            params.resolution,
            params.seed
        )

    # Combine all parts (union first, then subtract wall pores for porosity)
    all_parts = (
        airway_manifolds +
        duct_manifolds +
        alveoli_manifolds +
        pore_manifolds +
        interconnectivity_manifolds +
        barrier_manifolds +
        type_2_manifolds +
        capillary_manifolds
    )

    if not all_parts:
        raise ValueError("No geometry generated")

    result = batch_union(all_parts)

    # Subtract wall porosity pores from the result
    if wall_porosity_pores:
        porosity_subtract = batch_union(wall_porosity_pores)
        result = result - porosity_subtract

    # Clip to bounding box
    bx, by, bz = params.bounding_box
    bbox = m3d.Manifold.cube([bx, by, bz]).translate([-bx/2, -by/2, 0])
    result = result ^ bbox

    # Calculate statistics
    mesh = result.to_mesh()
    volume = result.volume() if hasattr(result, 'volume') else 0

    # Calculate total alveoli count
    if params.enable_alveolar_ducts:
        total_alveoli = len(final_alveoli_data)
    else:
        total_alveoli = len(final_alveoli_data) * max(1, params.alveoli_per_duct)

    stats = {
        'triangle_count': len(mesh.tri_verts) if hasattr(mesh, 'tri_verts') else 0,
        'volume_mm3': volume,
        'branch_count': len(airway_segments),
        'alveoli_count': total_alveoli,
        'generations': effective_generations,
        'porosity_target': params.porosity,
        'scaffold_type': 'lung_alveoli',
        # New feature counts
        'duct_count': len(duct_manifolds),
        'pores_of_kohn_count': len(pore_manifolds),
        'interconnectivity_channel_count': len(interconnectivity_manifolds),
        'wall_porosity_pore_count': len(wall_porosity_pores),
        'barrier_elements_count': len(barrier_manifolds),
        'type_2_cell_count': len(type_2_manifolds),
        'capillary_segment_count': len(capillary_manifolds),
        # New parameter values in stats
        'alveolar_depth_ratio': depth_ratio,
        'terminal_bronchiole_radius_mm': terminal_bronchiole_radius_mm,
        'surfactant_layer_thickness_mm': surfactant_thickness_mm,
        'capillary_density_per_mm2': params.capillary_density,
        'pore_interconnectivity': params.pore_interconnectivity,
        'type_1_cell_coverage': params.type_1_cell_coverage,
        # Feature flags
        'enable_alveolar_ducts': params.enable_alveolar_ducts,
        'enable_pores_of_kohn': params.enable_pores_of_kohn,
        'enable_blood_air_barrier': params.enable_blood_air_barrier,
        'enable_capillary_network': params.enable_capillary_network,
    }

    return result, stats


def generate_lung_alveoli_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate lung alveoli from dictionary parameters (for API compatibility).

    Args:
        params: Dictionary with keys matching LungAlveoliParams fields

    Returns:
        Tuple of (manifold, stats_dict)
    """
    # Handle bounding box variations
    bbox = params.get('bounding_box', (8.0, 8.0, 8.0))
    if isinstance(bbox, dict):
        bbox = (bbox.get('x', 8.0), bbox.get('y', 8.0), bbox.get('z', 8.0))

    # Handle legacy parameter names
    branch_generations = params.get('branch_generations', params.get('scaffold_generations', 3))
    alveoli_diameter = params.get('alveoli_diameter', params.get('alveolar_diameter', 200.0))

    return generate_lung_alveoli(LungAlveoliParams(
        # Alveolar geometry
        alveolar_diameter=alveoli_diameter,
        alveolar_wall_thickness=params.get('alveolar_wall_thickness', 0.6),
        alveolar_depth=params.get('alveolar_depth', 150.0),
        alveoli_per_duct=params.get('alveoli_per_duct', 6),

        # Airway tree
        total_branching_generations=params.get('total_branching_generations', 23),
        scaffold_generations=branch_generations,
        terminal_bronchiole_diameter=params.get('terminal_bronchiole_diameter', 750.0),
        airway_diameter=params.get('airway_diameter', 1.5),
        branch_angle=params.get('branch_angle', 35.0),
        branch_length_ratio=params.get('branch_length_ratio', 0.8),

        # Murray's law
        branching_ratio=params.get('branching_ratio', 0.79),
        enable_asymmetric_branching=params.get('enable_asymmetric_branching', False),
        asymmetry_factor=params.get('asymmetry_factor', 0.1),

        # Porosity
        porosity=params.get('porosity', 0.85),
        pore_interconnectivity=params.get('pore_interconnectivity', 0.95),
        enable_pores_of_kohn=params.get('enable_pores_of_kohn', True),
        pore_of_kohn_diameter=params.get('pore_of_kohn_diameter', 10.0),
        pores_per_alveolus=params.get('pores_per_alveolus', 3),

        # Blood-air barrier
        enable_blood_air_barrier=params.get('enable_blood_air_barrier', False),
        type_1_cell_coverage=params.get('type_1_cell_coverage', 0.95),
        type_2_cell_coverage=params.get('type_2_cell_coverage', 0.06),
        type_2_cell_diameter=params.get('type_2_cell_diameter', 9.0),

        # Capillary network
        enable_capillary_network=params.get('enable_capillary_network', False),
        capillary_diameter=params.get('capillary_diameter', 8.0),
        capillary_density=params.get('capillary_density', 2800.0),
        capillary_spacing=params.get('capillary_spacing', 15.0),

        # Alveolar duct
        enable_alveolar_ducts=params.get('enable_alveolar_ducts', True),
        alveolar_duct_length=params.get('alveolar_duct_length', 0.5),
        alveolar_duct_diameter=params.get('alveolar_duct_diameter', 400.0),

        # Bounding box
        bounding_box=tuple(bbox),

        # Surfactant
        surfactant_layer_thickness=params.get('surfactant_layer_thickness', 0.02),

        # Stochastic variation
        size_variance=params.get('size_variance', 0.0),
        position_noise=params.get('position_noise', 0.0),
        angle_noise=params.get('angle_noise', 0.0),
        seed=params.get('seed', 42),

        # Resolution
        resolution=params.get('resolution', 10)
    ))
