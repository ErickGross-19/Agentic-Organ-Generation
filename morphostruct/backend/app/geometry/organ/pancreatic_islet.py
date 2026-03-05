"""
Pancreatic islet scaffold generator.

Creates spherical cluster structures with core-shell architecture:
- Porous spheres mimicking islet of Langerhans (~150um diameter)
- Cell type distribution (beta, alpha, delta, PP cells)
- Encapsulation capsule for immunoprotection
- Vascular network for nutrient/oxygen delivery
- Controlled pore size and density for diffusion
"""

from __future__ import annotations
import manifold3d as m3d
import numpy as np
from dataclasses import dataclass
from ..core import batch_union


@dataclass
class PancreaticIsletParams:
    """
    Parameters for pancreatic islet scaffold generation.

    Based on native islet of Langerhans anatomy:
    - Islet diameter: 50-500 um (average ~150 um)
    - Beta cells: 50-80% of islet (insulin)
    - Alpha cells: 15-20% (glucagon)
    - Delta cells: 3-10% (somatostatin)
    - PP cells: 1-2% (pancreatic polypeptide)
    """
    # === Islet Geometry ===
    islet_diameter: float = 150.0  # um (50-500 um, average ~150)
    islet_count: int = 3  # number of islets in scaffold
    islet_spacing: float = 300.0  # um between islet centers

    # === Cell Type Fractions ===
    beta_cell_fraction: float = 0.50  # insulin-producing (human: 40-54%, rodent: 60-80%)
    alpha_cell_fraction: float = 0.35  # glucagon-producing (human: 35%, rodent: 15-20%)
    delta_cell_fraction: float = 0.07  # somatostatin-producing (3-10% native)
    pp_cell_fraction: float = 0.02  # pancreatic polypeptide (1-2% native)
    enable_cell_distribution_markers: bool = False  # visual markers for cell zones

    # === Encapsulation Capsule ===
    enable_capsule: bool = True
    capsule_thickness: float = 100.0  # um (immunoprotection layer)
    capsule_diameter: float = 500.0  # um (outer diameter including islets)
    capsule_porosity: float = 0.4  # allow nutrients, block immune cells
    capsule_pore_size: float = 12.0  # um (10-15 um for immunoisolation)

    # === Islet Shell Structure ===
    shell_thickness: float = 40.0  # um (outer shell of islet)
    core_porosity: float = 0.6  # inner core porosity
    shell_porosity: float = 0.5  # outer shell porosity
    enable_core_shell_architecture: bool = True

    # === Porosity and Diffusion ===
    pore_size: float = 20.0  # um (for nutrient diffusion)
    pore_density: float = 0.1  # pores per surface area
    oxygen_diffusion_coefficient: float = 2.0e-9  # m^2/s (reference value)
    glucose_diffusion_coefficient: float = 6.7e-10  # m^2/s (reference value)

    # === Vascularization ===
    enable_vascular_channels: bool = False
    vascular_channel_diameter: float = 20.0  # um
    vascular_channel_count: int = 4  # channels per islet
    vascular_channel_pattern: str = 'radial'  # radial, parallel

    # === Viability Support ===
    islet_viability: float = 0.85  # target viability (fraction of live cells)
    max_diffusion_distance: float = 150.0  # um (oxygen diffusion limit)

    # === ECM Components (conceptual) ===
    enable_ecm_coating: bool = False
    ecm_thickness: float = 5.0  # um (collagen/laminin layer)
    ecm_type: str = 'collagen'  # collagen, laminin, matrigel

    # === Multi-islet Configuration ===
    cluster_pattern: str = 'hexagonal'  # hexagonal, random, linear
    enable_inter_islet_connections: bool = False
    connection_diameter: float = 30.0  # um

    # === Stochastic Variation ===
    size_variance: float = 0.0  # 0-1 variation in islet size
    position_noise: float = 0.0  # 0-1 variation in position
    seed: int = 42

    # === Resolution ===
    resolution: int = 12  # sphere segments


def create_cell_distribution_markers(center: np.ndarray, islet_radius: float,
                                      beta_fraction: float, alpha_fraction: float,
                                      delta_fraction: float, pp_fraction: float,
                                      marker_size: float, total_markers: int,
                                      resolution: int, seed: int) -> tuple[list[m3d.Manifold], dict]:
    """
    Create visual markers showing cell type distribution within an islet.

    Native islet architecture:
    - Beta cells: 50-80%, concentrated in core
    - Alpha cells: 15-20%, peripheral mantle
    - Delta cells: 3-10%, scattered throughout
    - PP cells: 1-2%, scattered in periphery

    Args:
        center: Islet center position (x, y, z)
        islet_radius: Radius of the islet (mm)
        beta_fraction: Fraction of beta cells (0-1)
        alpha_fraction: Fraction of alpha cells (0-1)
        delta_fraction: Fraction of delta cells (0-1)
        pp_fraction: Fraction of pp cells (0-1)
        marker_size: Size of marker spheres (mm)
        total_markers: Total number of markers to create
        resolution: Sphere resolution
        seed: Random seed

    Returns:
        Tuple of (list of marker manifolds, cell count dict)
    """
    np.random.seed(seed)
    markers = []

    # Normalize fractions
    total_fraction = beta_fraction + alpha_fraction + delta_fraction + pp_fraction
    if total_fraction <= 0:
        return [], {}

    beta_fraction /= total_fraction
    alpha_fraction /= total_fraction
    delta_fraction /= total_fraction
    pp_fraction /= total_fraction

    # Calculate marker counts per cell type
    beta_count = int(total_markers * beta_fraction)
    alpha_count = int(total_markers * alpha_fraction)
    delta_count = int(total_markers * delta_fraction)
    pp_count = total_markers - beta_count - alpha_count - delta_count

    cell_counts = {
        'beta_markers': beta_count,
        'alpha_markers': alpha_count,
        'delta_markers': delta_count,
        'pp_markers': pp_count
    }

    # Core region (0 to 60% of radius) - primarily beta cells
    # Mantle region (60% to 100% of radius) - alpha, delta, PP cells
    core_boundary = 0.6 * islet_radius

    def fibonacci_sphere_points(n: int, r_min: float, r_max: float) -> list[np.ndarray]:
        """Generate points in a spherical shell using Fibonacci distribution."""
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        angle_increment = 2 * np.pi * golden_ratio

        for i in range(n):
            # Fibonacci sphere for angular distribution
            t = i / max(n, 1)
            inclination = np.arccos(1 - 2 * t)
            azimuth = angle_increment * i

            # Random radius within the shell
            r = np.random.uniform(r_min, r_max)

            x = r * np.sin(inclination) * np.cos(azimuth)
            y = r * np.sin(inclination) * np.sin(azimuth)
            z = r * np.cos(inclination)

            points.append(np.array([x, y, z]))
        return points

    # Beta cells in core (some scattered in mantle too for realism)
    beta_core_count = int(beta_count * 0.85)
    beta_mantle_count = beta_count - beta_core_count

    # Core beta cells
    if beta_core_count > 0:
        positions = fibonacci_sphere_points(beta_core_count, 0, core_boundary)
        for pos in positions:
            marker = m3d.Manifold.sphere(marker_size, max(4, resolution // 2))
            marker = marker.translate([
                center[0] + pos[0],
                center[1] + pos[1],
                center[2] + pos[2]
            ])
            if marker.num_vert() > 0:
                markers.append(marker)

    # Scattered beta cells in mantle
    if beta_mantle_count > 0:
        positions = fibonacci_sphere_points(beta_mantle_count, core_boundary, islet_radius * 0.9)
        for pos in positions:
            marker = m3d.Manifold.sphere(marker_size, max(4, resolution // 2))
            marker = marker.translate([
                center[0] + pos[0],
                center[1] + pos[1],
                center[2] + pos[2]
            ])
            if marker.num_vert() > 0:
                markers.append(marker)

    # Alpha cells in peripheral mantle
    if alpha_count > 0:
        positions = fibonacci_sphere_points(alpha_count, core_boundary * 1.1, islet_radius * 0.95)
        for pos in positions:
            marker = m3d.Manifold.sphere(marker_size * 0.9, max(4, resolution // 2))
            marker = marker.translate([
                center[0] + pos[0],
                center[1] + pos[1],
                center[2] + pos[2]
            ])
            if marker.num_vert() > 0:
                markers.append(marker)

    # Delta cells scattered throughout (slightly smaller markers)
    if delta_count > 0:
        positions = fibonacci_sphere_points(delta_count, core_boundary * 0.5, islet_radius * 0.9)
        for pos in positions:
            marker = m3d.Manifold.sphere(marker_size * 0.8, max(4, resolution // 2))
            marker = marker.translate([
                center[0] + pos[0],
                center[1] + pos[1],
                center[2] + pos[2]
            ])
            if marker.num_vert() > 0:
                markers.append(marker)

    # PP cells in periphery (smallest markers)
    if pp_count > 0:
        positions = fibonacci_sphere_points(pp_count, core_boundary, islet_radius * 0.95)
        for pos in positions:
            marker = m3d.Manifold.sphere(marker_size * 0.7, max(4, resolution // 2))
            marker = marker.translate([
                center[0] + pos[0],
                center[1] + pos[1],
                center[2] + pos[2]
            ])
            if marker.num_vert() > 0:
                markers.append(marker)

    return markers, cell_counts


def create_ecm_coating(center: np.ndarray, islet_radius: float, ecm_thickness: float,
                       ecm_type: str, resolution: int) -> m3d.Manifold:
    """
    Create ECM (extracellular matrix) coating on islet surface.

    ECM provides structural support and cell signaling. Types:
    - collagen: denser, smaller pores
    - laminin: medium porosity
    - matrigel: higher porosity, larger pores

    Args:
        center: Islet center position (x, y, z)
        islet_radius: Radius of the islet (mm)
        ecm_thickness: Thickness of ECM layer (mm)
        ecm_type: Type of ECM (collagen, laminin, matrigel)
        resolution: Sphere resolution

    Returns:
        ECM shell manifold with pores
    """
    outer_radius = islet_radius + ecm_thickness
    inner_radius = islet_radius

    # ECM type affects pore characteristics
    if ecm_type == 'collagen':
        pore_size_factor = 0.15  # smaller pores
        pore_count_factor = 0.08  # fewer pores
    elif ecm_type == 'matrigel':
        pore_size_factor = 0.35  # larger pores
        pore_count_factor = 0.15  # more pores
    else:  # laminin (default)
        pore_size_factor = 0.25  # medium pores
        pore_count_factor = 0.12

    # Create ECM shell
    outer_sphere = m3d.Manifold.sphere(outer_radius, resolution)
    inner_sphere = m3d.Manifold.sphere(inner_radius, resolution)
    ecm_shell = outer_sphere - inner_sphere

    # Add pores for diffusion
    pore_radius = ecm_thickness * pore_size_factor
    surface_area = 4 * np.pi * outer_radius**2
    pore_count = max(4, int(surface_area / (np.pi * pore_radius**2) * pore_count_factor))

    # Fibonacci distribution for pores
    pores = []
    golden_ratio = (1 + np.sqrt(5)) / 2
    angle_increment = 2 * np.pi * golden_ratio

    for i in range(pore_count):
        t = i / pore_count
        inclination = np.arccos(1 - 2 * t)
        azimuth = angle_increment * i

        x = outer_radius * np.sin(inclination) * np.cos(azimuth)
        y = outer_radius * np.sin(inclination) * np.sin(azimuth)
        z = outer_radius * np.cos(inclination)

        pore = m3d.Manifold.sphere(pore_radius, max(4, resolution // 3))
        pore = pore.translate([x, y, z])
        pores.append(pore)

    if pores:
        pore_union = batch_union(pores)
        ecm_shell = ecm_shell - pore_union

    # Translate to islet center
    return ecm_shell.translate([center[0], center[1], center[2]])


def create_inter_islet_connection(pos1: np.ndarray, pos2: np.ndarray,
                                   connection_diameter: float, resolution: int) -> m3d.Manifold:
    """
    Create a cylindrical connection between two islets.

    Args:
        pos1: First islet center position
        pos2: Second islet center position
        connection_diameter: Diameter of connection channel (mm)
        resolution: Cylinder resolution

    Returns:
        Cylindrical connection manifold
    """
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    dz = pos2[2] - pos1[2]
    length = np.sqrt(dx*dx + dy*dy + dz*dz)

    if length < 1e-6:
        return m3d.Manifold()

    radius = connection_diameter / 2.0
    cyl = m3d.Manifold.cylinder(length, radius, radius, resolution)

    # Rotate to align with connection vector
    h = np.sqrt(dx*dx + dy*dy)
    if h > 0.001 or abs(dz) > 0.001:
        tilt = np.arctan2(h, dz) * 180 / np.pi
        azim = np.arctan2(dy, dx) * 180 / np.pi
        cyl = cyl.rotate([0, tilt, 0]).rotate([0, 0, azim])

    return cyl.translate([pos1[0], pos1[1], pos1[2]])


def find_connectable_islet_pairs(positions: list[np.ndarray], islet_radius: float,
                                  max_connection_distance: float) -> list[tuple[int, int]]:
    """
    Find pairs of islets that are close enough to connect.

    Args:
        positions: List of islet center positions
        islet_radius: Radius of islets (mm)
        max_connection_distance: Maximum distance for connection (mm)

    Returns:
        List of (index1, index2) tuples for connectable pairs
    """
    pairs = []
    n = len(positions)

    # Minimum distance: islets must not overlap
    min_dist = islet_radius * 2.1  # slight gap

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if min_dist < dist < max_connection_distance:
                pairs.append((i, j))

    return pairs


def create_porous_sphere_dual_porosity(center: np.ndarray, outer_radius: float, inner_radius: float,
                                        pore_radius: float, core_porosity: float, shell_porosity: float,
                                        shell_thickness: float, resolution: int,
                                        pore_density: float = 0.1) -> m3d.Manifold:
    """
    Create a porous spherical shell with different core and shell porosities.

    Core region (inner) has higher porosity for cell migration/nutrients.
    Shell region (outer) has lower porosity for structural support.

    Args:
        center: Center position (x, y, z)
        outer_radius: Outer radius of sphere
        inner_radius: Inner radius (creates shell)
        pore_radius: Radius of pores
        core_porosity: Porosity for inner core region (0-1)
        shell_porosity: Porosity for outer shell region (0-1)
        shell_thickness: Thickness of outer shell region (mm)
        resolution: Sphere resolution

    Returns:
        Porous sphere manifold with dual porosity
    """
    # Create outer and inner spheres
    outer = m3d.Manifold.sphere(outer_radius, resolution)
    inner = m3d.Manifold.sphere(inner_radius, resolution)
    shell = outer - inner

    # Define core-shell boundary
    shell_boundary = outer_radius - shell_thickness
    if shell_boundary <= inner_radius:
        shell_boundary = (outer_radius + inner_radius) / 2

    # Calculate pore counts for each region using pore_density
    # pore_density represents fraction of surface area covered by pores
    # Core region: between inner_radius and shell_boundary
    core_mid_radius = (inner_radius + shell_boundary) / 2
    core_surface_area = 4 * np.pi * core_mid_radius**2
    core_pore_area = np.pi * pore_radius**2
    # Use pore_density to calculate pore count: surface_area * pore_density / pore_area
    # Multiply by core_porosity to scale for the region's target porosity
    core_pore_count = max(3, int(core_surface_area * pore_density * core_porosity / core_pore_area))

    # Shell region: between shell_boundary and outer_radius
    shell_mid_radius = (shell_boundary + outer_radius) / 2
    shell_surface_area = 4 * np.pi * shell_mid_radius**2
    shell_pore_area = np.pi * pore_radius**2
    # Use pore_density to calculate pore count, scaled by shell_porosity
    shell_pore_count = max(3, int(shell_surface_area * pore_density * shell_porosity / shell_pore_area))

    pores = []
    golden_ratio = (1 + np.sqrt(5)) / 2
    angle_increment = 2 * np.pi * golden_ratio

    # Core region pores (placed at core mid-radius)
    for i in range(core_pore_count):
        t = i / core_pore_count
        inclination = np.arccos(1 - 2 * t)
        azimuth = angle_increment * i

        x = core_mid_radius * np.sin(inclination) * np.cos(azimuth)
        y = core_mid_radius * np.sin(inclination) * np.sin(azimuth)
        z = core_mid_radius * np.cos(inclination)

        # Slightly larger pores in core for better nutrient flow
        pore = m3d.Manifold.sphere(pore_radius * 1.2, 8)
        pore = pore.translate([x, y, z])
        pores.append(pore)

    # Shell region pores (placed at outer_radius for surface access)
    for i in range(shell_pore_count):
        t = i / shell_pore_count
        inclination = np.arccos(1 - 2 * t)
        azimuth = angle_increment * i + np.pi / 4  # offset from core pores

        x = outer_radius * np.sin(inclination) * np.cos(azimuth)
        y = outer_radius * np.sin(inclination) * np.sin(azimuth)
        z = outer_radius * np.cos(inclination)

        pore = m3d.Manifold.sphere(pore_radius, 8)
        pore = pore.translate([x, y, z])
        pores.append(pore)

    # Subtract pores from shell
    if pores:
        pore_union = batch_union(pores)
        shell = shell - pore_union

    return shell.translate([center[0], center[1], center[2]])


def create_viability_channel(center: np.ndarray, islet_radius: float,
                              max_diffusion_distance: float, channel_diameter: float,
                              resolution: int) -> m3d.Manifold:
    """
    Create central channel for oxygen delivery in large islets.

    When islet radius exceeds max_diffusion_distance, oxygen cannot reach
    the core, leading to cell death. A central channel provides oxygen access.

    Args:
        center: Islet center position (x, y, z)
        islet_radius: Radius of islet (mm)
        max_diffusion_distance: Maximum oxygen diffusion distance (mm)
        channel_diameter: Diameter of central channel (mm)
        resolution: Cylinder resolution

    Returns:
        Central channel manifold (to be subtracted from islet)
    """
    # Channel runs through the islet
    channel_length = islet_radius * 2.2  # slightly longer than diameter
    channel_radius = channel_diameter / 2.0

    cyl = m3d.Manifold.cylinder(channel_length, channel_radius, channel_radius, resolution)
    # Center the channel vertically
    cyl = cyl.translate([center[0], center[1], center[2] - channel_length / 2])

    return cyl


def check_viability(islet_radius_um: float, max_diffusion_distance_um: float) -> tuple[bool, str]:
    """
    Check if islet geometry supports cell viability.

    Args:
        islet_radius_um: Islet radius in micrometers
        max_diffusion_distance_um: Maximum diffusion distance in micrometers

    Returns:
        Tuple of (is_viable, warning_message)
    """
    if islet_radius_um <= max_diffusion_distance_um:
        return True, ""
    else:
        ratio = islet_radius_um / max_diffusion_distance_um
        return False, f"Islet radius ({islet_radius_um:.0f}um) exceeds max diffusion distance ({max_diffusion_distance_um:.0f}um) by {ratio:.1f}x. Central necrosis risk."


def calculate_viability_vascular_channels(islet_diameter_um: float, max_diffusion_distance_um: float,
                                           target_viability: float) -> int:
    """
    Calculate number of vascular channels needed to achieve target viability.

    When islet diameter exceeds 2 * max_diffusion_distance, the core becomes hypoxic.
    Vascular channels provide oxygen delivery to the core. More channels = better coverage.

    The calculation is based on:
    - Each vascular channel serves a cylindrical volume with radius = max_diffusion_distance
    - Higher viability targets require more complete coverage
    - Minimum 1 channel (central), scaling up with size and viability target

    Args:
        islet_diameter_um: Islet diameter in micrometers
        max_diffusion_distance_um: Maximum oxygen diffusion distance in micrometers
        target_viability: Target cell viability fraction (0-1)

    Returns:
        Number of vascular channels needed (0 if islet is small enough)
    """
    islet_radius_um = islet_diameter_um / 2.0

    # If islet is within diffusion limit, no channels needed
    if islet_radius_um <= max_diffusion_distance_um:
        return 0

    # Calculate the "at-risk" core volume (beyond diffusion reach from surface)
    # Volume at risk = sphere volume - shell volume (where shell = diffusible region)
    at_risk_radius = islet_radius_um - max_diffusion_distance_um
    total_volume = (4/3) * np.pi * islet_radius_um**3
    at_risk_volume = (4/3) * np.pi * at_risk_radius**3
    at_risk_fraction = at_risk_volume / total_volume

    # Each vascular channel serves a cylindrical region of radius max_diffusion_distance
    # Cross-sectional area served by one channel
    channel_service_area = np.pi * max_diffusion_distance_um**2
    # Cross-sectional area of at-risk core
    at_risk_cross_section = np.pi * at_risk_radius**2

    # Base channels needed to cover at-risk area
    base_channels = max(1, int(np.ceil(at_risk_cross_section / channel_service_area)))

    # Scale by viability target: higher viability = more redundancy
    # At viability=0.5, use base channels; at viability=1.0, use 2x base channels
    viability_scale = 1.0 + (target_viability - 0.5)
    scaled_channels = int(np.ceil(base_channels * viability_scale))

    # Minimum 1 channel (central), maximum reasonable limit based on size
    max_channels = max(1, int(islet_diameter_um / (max_diffusion_distance_um * 0.5)))
    return min(max(1, scaled_channels), max_channels)


def create_porous_sphere(center: np.ndarray, outer_radius: float, inner_radius: float,
                         pore_radius: float, pore_count: int, resolution: int,
                         pore_density: float = None) -> m3d.Manifold:
    """
    Create a porous spherical shell with evenly distributed pores.

    Args:
        center: Center position (x, y, z)
        outer_radius: Outer radius of sphere
        inner_radius: Inner radius (creates shell)
        pore_radius: Radius of pores
        pore_count: Number of pores
        resolution: Sphere resolution

    Returns:
        Porous sphere manifold
    """
    # Create outer and inner spheres for shell
    outer = m3d.Manifold.sphere(outer_radius, resolution)
    inner = m3d.Manifold.sphere(inner_radius, resolution)

    shell = outer - inner

    # If pore_density is provided, recalculate pore_count based on surface area
    # pore_density = fraction of surface area covered by pores
    if pore_density is not None and pore_density > 0:
        surface_area = 4 * np.pi * outer_radius**2
        pore_area = np.pi * pore_radius**2
        pore_count = max(3, int(surface_area * pore_density / pore_area))

    # Create pores using Fibonacci sphere distribution
    pores = []
    golden_ratio = (1 + np.sqrt(5)) / 2
    angle_increment = 2 * np.pi * golden_ratio

    for i in range(pore_count):
        # Fibonacci sphere points
        t = i / pore_count
        inclination = np.arccos(1 - 2 * t)
        azimuth = angle_increment * i

        # Position on sphere surface
        x = outer_radius * np.sin(inclination) * np.cos(azimuth)
        y = outer_radius * np.sin(inclination) * np.sin(azimuth)
        z = outer_radius * np.cos(inclination)

        # Create pore as small sphere cutting through shell
        pore = m3d.Manifold.sphere(pore_radius, 8)
        pore = pore.translate([x, y, z])
        pores.append(pore)

    # Subtract pores from shell
    if pores:
        pore_union = batch_union(pores)
        shell = shell - pore_union

    # Translate to center position
    return shell.translate([center[0], center[1], center[2]])


def create_solid_sphere(center: np.ndarray, radius: float, resolution: int) -> m3d.Manifold:
    """
    Create a solid sphere.

    Args:
        center: Center position (x, y, z)
        radius: Sphere radius
        resolution: Sphere resolution

    Returns:
        Sphere manifold
    """
    sphere = m3d.Manifold.sphere(radius, resolution)
    return sphere.translate([center[0], center[1], center[2]])


def create_capsule_shell(center: np.ndarray, outer_radius: float, thickness: float,
                         pore_radius: float, porosity: float, resolution: int) -> m3d.Manifold:
    """
    Create encapsulation capsule with controlled porosity.

    Args:
        center: Center position
        outer_radius: Outer radius of capsule
        thickness: Wall thickness
        pore_radius: Pore radius for diffusion
        porosity: Target porosity (0-1)
        resolution: Sphere resolution

    Returns:
        Porous capsule manifold
    """
    inner_radius = outer_radius - thickness
    if inner_radius <= 0:
        inner_radius = outer_radius * 0.8

    # Calculate pore count from porosity
    surface_area = 4 * np.pi * outer_radius**2
    pore_area = np.pi * pore_radius**2
    pore_count = max(6, int(surface_area / pore_area * porosity * 0.5))

    return create_porous_sphere(center, outer_radius, inner_radius,
                                pore_radius, pore_count, resolution)


def generate_cluster_positions(count: int, spacing: float, pattern: str,
                               seed: int) -> list[np.ndarray]:
    """
    Generate positions for islets in a cluster arrangement.

    Args:
        count: Number of islets
        spacing: Distance between centers (mm)
        pattern: 'hexagonal', 'random', or 'linear'
        seed: Random seed

    Returns:
        List of position vectors
    """
    np.random.seed(seed)

    if count <= 0:
        return []
    if count == 1:
        return [np.array([0.0, 0.0, 0.0])]

    positions = [np.array([0.0, 0.0, 0.0])]

    if pattern == 'linear':
        for i in range(1, count):
            positions.append(np.array([i * spacing, 0.0, 0.0]))

    elif pattern == 'random':
        for i in range(1, count):
            angle = np.random.uniform(0, 2 * np.pi)
            distance = spacing * (0.5 + np.random.uniform(0, 0.5))
            z_offset = np.random.uniform(-spacing/3, spacing/3)
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            positions.append(np.array([x, y, z_offset]))

    else:  # hexagonal (default)
        # Layer 1: 6 around center in XY plane
        if count > 1:
            for i in range(min(6, count - 1)):
                angle = i * np.pi / 3
                x = spacing * np.cos(angle)
                y = spacing * np.sin(angle)
                positions.append(np.array([x, y, 0.0]))

        # Layer 2: Above and below
        if count > 7:
            remaining = count - 7
            layer_z = spacing * 0.866  # sqrt(3)/2

            # Above layer
            for i in range(min(3, remaining)):
                angle = i * 2 * np.pi / 3
                x = spacing * 0.5 * np.cos(angle)
                y = spacing * 0.5 * np.sin(angle)
                positions.append(np.array([x, y, layer_z]))

            # Below layer
            if remaining > 3:
                for i in range(min(3, remaining - 3)):
                    angle = i * 2 * np.pi / 3 + np.pi / 3
                    x = spacing * 0.5 * np.cos(angle)
                    y = spacing * 0.5 * np.sin(angle)
                    positions.append(np.array([x, y, -layer_z]))

    return positions[:count]


def create_vascular_channels(center: np.ndarray, islet_radius: float,
                             channel_radius: float, channel_count: int,
                             pattern: str, resolution: int) -> list[m3d.Manifold]:
    """
    Create vascular channels through/around islet.

    Args:
        center: Islet center
        islet_radius: Radius of islet (mm)
        channel_radius: Channel radius (mm)
        channel_count: Number of channels
        pattern: 'radial' or 'parallel'
        resolution: Cylinder resolution

    Returns:
        List of channel manifolds
    """
    channels = []

    if pattern == 'radial':
        # Channels radiating from center
        for i in range(channel_count):
            angle = 2 * np.pi * i / channel_count
            start = center
            end = center + np.array([
                islet_radius * 1.2 * np.cos(angle),
                islet_radius * 1.2 * np.sin(angle),
                0
            ])

            length = np.linalg.norm(end - start)
            cyl = m3d.Manifold.cylinder(length, channel_radius, channel_radius, resolution)

            # Rotate to align
            direction = (end - start) / length
            h = np.sqrt(direction[0]**2 + direction[1]**2)
            if h > 0.001 or abs(direction[2]) > 0.001:
                tilt = np.arctan2(h, direction[2]) * 180 / np.pi
                azim = np.arctan2(direction[1], direction[0]) * 180 / np.pi
                cyl = cyl.rotate([0, tilt, 0]).rotate([0, 0, azim])

            cyl = cyl.translate([start[0], start[1], start[2]])
            channels.append(cyl)

    else:  # parallel
        # Parallel channels through islet
        spacing = islet_radius * 2 / (channel_count + 1)
        for i in range(channel_count):
            x_offset = -islet_radius + (i + 1) * spacing
            start = center + np.array([x_offset, 0, -islet_radius])
            end = center + np.array([x_offset, 0, islet_radius])

            length = np.linalg.norm(end - start)
            cyl = m3d.Manifold.cylinder(length, channel_radius, channel_radius, resolution)
            cyl = cyl.translate([start[0], start[1], start[2]])
            channels.append(cyl)

    return channels


def generate_pancreatic_islet(params: PancreaticIsletParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate a pancreatic islet scaffold with porous sphere clusters.

    Features:
    - Cell distribution markers (beta, alpha, delta, PP cells)
    - ECM coating on islet surfaces
    - Inter-islet connections
    - Differential core/shell porosity
    - Viability validation with optional central channels

    Args:
        params: PancreaticIsletParams specifying geometry

    Returns:
        Tuple of (manifold, stats_dict)
        - manifold: The 3D geometry
        - stats_dict: Dictionary with triangle_count, volume_mm3, islet_count, scaffold_type

    Raises:
        ValueError: If no islets are generated
    """
    np.random.seed(params.seed)

    # Convert um to mm
    outer_radius = params.islet_diameter / 2000.0
    shell_thickness_mm = params.shell_thickness / 1000.0
    inner_radius = outer_radius - shell_thickness_mm
    pore_radius = params.pore_size / 2000.0
    spacing_mm = params.islet_spacing / 1000.0
    capsule_radius_mm = params.capsule_diameter / 2000.0
    capsule_thickness_mm = params.capsule_thickness / 1000.0
    capsule_pore_radius_mm = params.capsule_pore_size / 2000.0
    vascular_radius_mm = params.vascular_channel_diameter / 2000.0
    ecm_thickness_mm = params.ecm_thickness / 1000.0
    connection_diameter_mm = params.connection_diameter / 1000.0
    max_diffusion_mm = params.max_diffusion_distance / 1000.0

    if inner_radius <= 0:
        inner_radius = outer_radius * 0.5

    # Check viability
    islet_radius_um = params.islet_diameter / 2.0
    is_viable, viability_warning = check_viability(islet_radius_um, params.max_diffusion_distance)

    # Calculate viability-driven vascular channels if needed
    # islet_viability parameter drives automatic channel creation for large islets
    viability_driven_channels = calculate_viability_vascular_channels(
        params.islet_diameter, params.max_diffusion_distance, params.islet_viability
    )
    auto_enable_vascular = viability_driven_channels > 0 and not params.enable_vascular_channels

    # Generate cluster positions
    islet_count = params.islet_count
    positions = generate_cluster_positions(islet_count, spacing_mm,
                                           params.cluster_pattern, params.seed)

    # Track actual positions after noise (needed for connections)
    actual_positions = []

    # Create islets
    islets = []
    all_channels = []
    all_cell_markers = []
    all_ecm_coatings = []
    viability_channels = []
    total_cell_counts = {'beta_markers': 0, 'alpha_markers': 0, 'delta_markers': 0, 'pp_markers': 0}

    for idx, pos in enumerate(positions):
        # Apply size variance
        if params.size_variance > 0:
            variance = 1 + np.random.uniform(-1, 1) * params.size_variance * 0.2
            islet_outer = outer_radius * variance
            islet_inner = inner_radius * variance
        else:
            islet_outer = outer_radius
            islet_inner = inner_radius

        # Apply position noise
        if params.position_noise > 0:
            noise = np.random.uniform(-1, 1, 3) * params.position_noise * spacing_mm * 0.2
            pos = pos + noise

        actual_positions.append(pos.copy())

        # Create porous islet sphere with differential core/shell porosity
        # pore_density now controls the number of pores based on surface area
        if params.enable_core_shell_architecture:
            islet = create_porous_sphere_dual_porosity(
                pos, islet_outer, islet_inner,
                pore_radius, params.core_porosity, params.shell_porosity,
                shell_thickness_mm, params.resolution,
                pore_density=params.pore_density
            )
        else:
            # Solid islet with surface pores only
            islet = create_solid_sphere(pos, islet_outer, params.resolution)

        if islet.num_vert() > 0:
            islets.append(islet)

        # Create viability channel if islet exceeds diffusion limit
        if not is_viable and islet_outer > max_diffusion_mm:
            viability_channel = create_viability_channel(
                pos, islet_outer, max_diffusion_mm,
                pore_radius * 4,  # Channel diameter scaled to pore size
                max(6, params.resolution - 2)
            )
            if viability_channel.num_vert() > 0:
                viability_channels.append(viability_channel)

        # Create cell distribution markers if enabled
        if params.enable_cell_distribution_markers:
            # Marker size proportional to islet size, ~5-10% of radius
            marker_size = islet_outer * 0.05
            # Total markers based on islet volume (roughly 20-40 per islet)
            total_markers = max(15, int(30 * (islet_outer / outer_radius)**3))

            markers, cell_counts = create_cell_distribution_markers(
                pos, islet_outer * 0.85,  # Place markers inside the shell
                params.beta_cell_fraction, params.alpha_cell_fraction,
                params.delta_cell_fraction, params.pp_cell_fraction,
                marker_size, total_markers,
                params.resolution, params.seed + idx
            )
            all_cell_markers.extend(markers)
            for key in total_cell_counts:
                total_cell_counts[key] += cell_counts.get(key, 0)

        # Create ECM coating if enabled
        if params.enable_ecm_coating:
            ecm = create_ecm_coating(
                pos, islet_outer, ecm_thickness_mm,
                params.ecm_type, params.resolution
            )
            if ecm.num_vert() > 0:
                all_ecm_coatings.append(ecm)

        # Create vascular channels if enabled OR if viability requires them
        # islet_viability parameter drives automatic channel creation for large islets
        if params.enable_vascular_channels or auto_enable_vascular:
            # Use viability-calculated channel count if auto-enabled, otherwise user-specified
            channel_count = viability_driven_channels if auto_enable_vascular else params.vascular_channel_count
            channels = create_vascular_channels(
                pos, islet_outer, vascular_radius_mm,
                channel_count,
                params.vascular_channel_pattern,
                max(6, params.resolution - 2)
            )
            all_channels.extend(channels)

    if not islets:
        raise ValueError("No islets generated")

    result = batch_union(islets)

    # Add cell markers if present
    if all_cell_markers:
        marker_union = batch_union(all_cell_markers)
        result = result + marker_union

    # Add ECM coatings if present
    if all_ecm_coatings:
        ecm_union = batch_union(all_ecm_coatings)
        result = result + ecm_union

    # Create inter-islet connections if enabled
    inter_islet_connections = []
    if params.enable_inter_islet_connections and len(actual_positions) > 1:
        # Maximum connection distance: 2.5x spacing (connects nearby islets)
        max_connection_dist = spacing_mm * 2.5
        connectable_pairs = find_connectable_islet_pairs(
            actual_positions, outer_radius, max_connection_dist
        )

        for i, j in connectable_pairs:
            connection = create_inter_islet_connection(
                actual_positions[i], actual_positions[j],
                connection_diameter_mm, max(6, params.resolution - 2)
            )
            if connection.num_vert() > 0:
                inter_islet_connections.append(connection)

    if inter_islet_connections:
        connection_union = batch_union(inter_islet_connections)
        result = result + connection_union

    # Add encapsulation capsule if enabled
    if params.enable_capsule:
        capsule_center = np.array([0.0, 0.0, 0.0])
        capsule = create_capsule_shell(
            capsule_center, capsule_radius_mm, capsule_thickness_mm,
            capsule_pore_radius_mm, params.capsule_porosity,
            params.resolution
        )
        if capsule.num_vert() > 0:
            result = result + capsule

    # Subtract vascular channels (hollow)
    if all_channels:
        channel_union = batch_union(all_channels)
        result = result - channel_union

    # Subtract viability channels (hollow central channels for large islets)
    if viability_channels:
        viability_union = batch_union(viability_channels)
        result = result - viability_union

    # Calculate statistics
    mesh = result.to_mesh()
    volume = result.volume() if hasattr(result, 'volume') else 0

    stats = {
        'triangle_count': len(mesh.tri_verts) if hasattr(mesh, 'tri_verts') else 0,
        'volume_mm3': volume,
        'islet_count': len(islets),
        'has_capsule': params.enable_capsule,
        'beta_cell_fraction': params.beta_cell_fraction,
        'alpha_cell_fraction': params.alpha_cell_fraction,
        'delta_cell_fraction': params.delta_cell_fraction,
        'pp_cell_fraction': params.pp_cell_fraction,
        'scaffold_type': 'pancreatic_islet',
        # New feature stats
        'has_cell_markers': params.enable_cell_distribution_markers,
        'cell_marker_counts': total_cell_counts if params.enable_cell_distribution_markers else None,
        'has_ecm_coating': params.enable_ecm_coating,
        'ecm_type': params.ecm_type if params.enable_ecm_coating else None,
        'inter_islet_connection_count': len(inter_islet_connections),
        'core_porosity': params.core_porosity,
        'shell_porosity': params.shell_porosity,
        'is_viable': is_viable,
        'viability_warning': viability_warning if viability_warning else None,
        'viability_channels_added': len(viability_channels),
        # pore_density and islet_viability parameter tracking
        'pore_density': params.pore_density,
        'target_viability': params.islet_viability,
        'viability_auto_vascular_enabled': auto_enable_vascular,
        'viability_driven_channel_count': viability_driven_channels if auto_enable_vascular else 0,
        'total_vascular_channels': len(all_channels)
    }

    return result, stats


def generate_pancreatic_islet_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate pancreatic islet from dictionary parameters (for API compatibility).

    Args:
        params: Dictionary with keys matching PancreaticIsletParams fields

    Returns:
        Tuple of (manifold, stats_dict)
    """
    # Handle legacy parameter names
    cluster_count = params.get('cluster_count', params.get('islet_count', 3))
    spacing = params.get('spacing', params.get('islet_spacing', 300.0))

    return generate_pancreatic_islet(PancreaticIsletParams(
        # Islet geometry
        islet_diameter=params.get('islet_diameter', 150.0),
        islet_count=cluster_count,
        islet_spacing=spacing,

        # Cell type fractions
        beta_cell_fraction=params.get('beta_cell_fraction', 0.50),
        alpha_cell_fraction=params.get('alpha_cell_fraction', 0.35),
        delta_cell_fraction=params.get('delta_cell_fraction', 0.07),
        pp_cell_fraction=params.get('pp_cell_fraction', 0.02),
        enable_cell_distribution_markers=params.get('enable_cell_distribution_markers', False),

        # Encapsulation capsule
        enable_capsule=params.get('enable_capsule', True),
        capsule_thickness=params.get('capsule_thickness', 100.0),
        capsule_diameter=params.get('capsule_diameter', 500.0),
        capsule_porosity=params.get('capsule_porosity', 0.4),
        capsule_pore_size=params.get('capsule_pore_size', 12.0),

        # Shell structure
        shell_thickness=params.get('shell_thickness', 40.0),
        core_porosity=params.get('core_porosity', 0.6),
        shell_porosity=params.get('shell_porosity', 0.5),
        enable_core_shell_architecture=params.get('enable_core_shell_architecture', True),

        # Porosity and diffusion
        pore_size=params.get('pore_size', 20.0),
        pore_density=params.get('pore_density', 0.1),
        oxygen_diffusion_coefficient=params.get('oxygen_diffusion_coefficient', 2.0e-9),
        glucose_diffusion_coefficient=params.get('glucose_diffusion_coefficient', 6.7e-10),

        # Vascularization
        enable_vascular_channels=params.get('enable_vascular_channels', False),
        vascular_channel_diameter=params.get('vascular_channel_diameter', 20.0),
        vascular_channel_count=params.get('vascular_channel_count', 4),
        vascular_channel_pattern=params.get('vascular_channel_pattern', 'radial'),

        # Viability support
        islet_viability=params.get('islet_viability', 0.85),
        max_diffusion_distance=params.get('max_diffusion_distance', 150.0),

        # ECM components
        enable_ecm_coating=params.get('enable_ecm_coating', False),
        ecm_thickness=params.get('ecm_thickness', 5.0),
        ecm_type=params.get('ecm_type', 'collagen'),

        # Multi-islet configuration
        cluster_pattern=params.get('cluster_pattern', 'hexagonal'),
        enable_inter_islet_connections=params.get('enable_inter_islet_connections', False),
        connection_diameter=params.get('connection_diameter', 30.0),

        # Stochastic variation
        size_variance=params.get('size_variance', 0.0),
        position_noise=params.get('position_noise', 0.0),
        seed=params.get('seed', 42),

        # Resolution
        resolution=params.get('resolution', 12)
    ))
