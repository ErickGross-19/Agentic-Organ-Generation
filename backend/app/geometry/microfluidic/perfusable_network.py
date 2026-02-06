"""
Perfusable vascular network generator for thick tissue engineering.

Creates branching vascular trees following Murray's law for optimal perfusion.
Designed for thick tissue constructs requiring internal vascularization.

Features:
- Murray's law branching: r_parent^3 = sum(r_child^3)
- Multi-scale vessel hierarchy (large vessels to capillaries)
- Configurable branch generations and topology
- Arterial, venous, and capillary network types
- Capillary bed generation with tortuosity
- Anastomosis cross-connections
- Multiple network topologies (hierarchical, anastomosing, parallel, loop)
- Optimized for tissue perfusion applications

Based on physiological vascular architecture:
- Maximum cell-to-vessel distance: 150-200 um (oxygen diffusion limit)
- Capillary spacing ensures no cell is >200 um from a vessel
- Vessel hierarchy follows PMC vascularization studies
"""

from __future__ import annotations
import manifold3d as m3d
import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional, List, Tuple, Dict, NamedTuple
from ..core import batch_union


class VesselEndpoint(NamedTuple):
    """Represents a terminal vessel endpoint for capillary bed connection."""
    position: np.ndarray
    direction: np.ndarray
    radius: float
    vessel_type: str  # 'artery', 'arteriole', 'capillary', 'venule', 'vein'
    generation: int


@dataclass
class VesselStats:
    """Tracks vessel counts by type during generation."""
    artery_count: int = 0
    arteriole_count: int = 0
    capillary_count: int = 0
    venule_count: int = 0
    vein_count: int = 0
    anastomosis_count: int = 0
    total_length_mm: float = 0.0


@dataclass
class PerfusableNetworkParams:
    """
    Parameters for perfusable vascular network generation.

    Based on physiological vascular architecture and Murray's law
    for optimal blood flow distribution.
    """
    # === Multi-scale Vessel Hierarchy ===
    large_vessel_diameter_mm: float = 5.0  # 1-10 mm for main feeding vessels
    arteriole_diameter_um: float = 100.0  # 50-300 um arterioles
    capillary_diameter_um: float = 8.0  # 5-10 um capillaries
    venule_diameter_um: float = 50.0  # 30-200 um venules

    # === Branching Parameters ===
    num_branching_generations: int = 7  # 3-12 typical
    murrays_law_ratio: float = 1.26  # Cube root of 2 (~1.26) for equal splitting
    bifurcation_angle_deg: float = 70.0  # 60-75 deg physiological
    bifurcation_asymmetry: float = 0.0  # 0 = symmetric, 1 = max asymmetry
    enable_trifurcation: bool = False  # Allow 3-way branching

    # === Capillary Network ===
    capillary_density_per_mm2: float = 1500.0  # 500-3000 per mm^2 tissue-dependent
    max_cell_distance_um: float = 200.0  # Maximum distance from any point to vessel
    enable_capillary_bed: bool = False
    capillary_tortuosity: float = 0.2  # 0-1 waviness

    # === Flow Parameters ===
    design_flow_rate_ml_min: float = 1.0  # 0.1-10 mL/min typical
    target_wall_shear_stress_pa: float = 1.5  # 0.5-5 Pa physiological

    # === Vessel Wall ===
    vessel_wall_thickness_ratio: float = 0.1  # Wall thickness as ratio of radius
    enable_hollow_vessels: bool = False
    min_wall_thickness_um: float = 40.0  # um (40+ um for bioprinting feasibility)

    # === Network Topology ===
    network_topology: Literal['hierarchical', 'anastomosing', 'parallel', 'loop'] = 'hierarchical'
    enable_arterio_venous_separation: bool = False
    anastomosis_density: float = 0.0  # 0-1 for cross-connections

    # === Scaffold Geometry ===
    bounding_box_mm: tuple[float, float, float] = (10.0, 10.0, 10.0)
    network_type: Literal['arterial', 'venous', 'capillary', 'mixed'] = 'arterial'
    inlet_position: Literal['top', 'bottom', 'side'] = 'top'
    outlet_position: Literal['top', 'bottom', 'side', 'distributed'] = 'bottom'

    # === Organic Variation ===
    enable_organic_variation: bool = False
    position_noise: float = 0.1  # Random position jitter
    angle_noise: float = 0.1  # Random angle variation
    diameter_variance: float = 0.1  # Random diameter variation

    # === Quality/Resolution ===
    resolution: int = 12
    seed: int = 42


def classify_vessel_type(
    radius_mm: float,
    arteriole_radius_mm: float,
    capillary_radius_mm: float,
    venule_radius_mm: float,
    is_venous: bool = False
) -> str:
    """
    Classify vessel type based on radius thresholds.

    Args:
        radius_mm: Current vessel radius in mm
        arteriole_radius_mm: Arteriole threshold radius in mm
        capillary_radius_mm: Capillary threshold radius in mm
        venule_radius_mm: Venule threshold radius in mm
        is_venous: True if this is part of venous return tree

    Returns:
        Vessel type string: 'artery', 'arteriole', 'capillary', 'venule', or 'vein'
    """
    if is_venous:
        if radius_mm > venule_radius_mm * 2:
            return 'vein'
        elif radius_mm > capillary_radius_mm:
            return 'venule'
        else:
            return 'capillary'
    else:
        if radius_mm > arteriole_radius_mm:
            return 'artery'
        elif radius_mm > capillary_radius_mm:
            return 'arteriole'
        else:
            return 'capillary'


def make_vessel_segment(
    p1: np.ndarray,
    p2: np.ndarray,
    r1: float,
    r2: float,
    resolution: int,
    wall_thickness_ratio: float = 0.0,
    enable_hollow: bool = False,
    min_wall_thickness_mm: float = 0.01
) -> m3d.Manifold:
    """
    Create a tapered cylindrical vessel segment.

    Args:
        p1: Starting point (x, y, z)
        p2: Ending point (x, y, z)
        r1: Radius at start point
        r2: Radius at end point
        resolution: Number of segments around cylinder
        wall_thickness_ratio: Wall thickness as ratio of radius
        enable_hollow: Create hollow vessel
        min_wall_thickness_mm: Minimum wall thickness in mm (enforced when hollow)

    Returns:
        Manifold representing the vessel segment
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    length = np.sqrt(dx*dx + dy*dy + dz*dz)

    if length < 1e-6:
        return m3d.Manifold()

    # Create cylinder along Z axis
    vessel = m3d.Manifold.cylinder(length, r1, r2, resolution)

    # Create hollow vessel if requested
    if enable_hollow and wall_thickness_ratio > 0:
        # Calculate wall thickness, enforcing minimum
        wall_t1 = max(r1 * wall_thickness_ratio, min_wall_thickness_mm)
        wall_t2 = max(r2 * wall_thickness_ratio, min_wall_thickness_mm)

        inner_r1 = r1 - wall_t1
        inner_r2 = r2 - wall_t2

        # Only hollow if inner radius is positive and meaningful
        if inner_r1 > 0.001 and inner_r2 > 0.001:
            inner = m3d.Manifold.cylinder(length + 0.01, inner_r1, inner_r2, resolution)
            vessel = vessel - inner

    # Calculate rotation to align with direction vector
    h = np.sqrt(dx*dx + dy*dy)  # horizontal distance

    if h > 0.001 or abs(dz) > 0.001:
        angle_y = np.arctan2(h, dz) * 180 / np.pi
        angle_z = np.arctan2(dy, dx) * 180 / np.pi

        vessel = vessel.rotate([0, angle_y, 0])
        vessel = vessel.rotate([0, 0, angle_z])

    return vessel.translate([p1[0], p1[1], p1[2]])


def calculate_child_radius(
    parent_radius: float,
    num_children: int,
    murray_ratio: float,
    asymmetry: float = 0.0,
    child_index: int = 0
) -> float:
    """
    Calculate child vessel radius using Murray's law.

    Murray's law: r_parent^3 = sum(r_child^3)
    For symmetric bifurcation: r_child = r_parent / (n^(1/3))

    Args:
        parent_radius: Radius of parent vessel
        num_children: Number of child branches
        murray_ratio: Murray's law ratio (typically cube root of 2)
        asymmetry: Asymmetry factor (0 = symmetric)
        child_index: Index of this child (for asymmetric branching)

    Returns:
        Radius for this child vessel
    """
    base_radius = parent_radius / murray_ratio

    if asymmetry > 0 and num_children == 2:
        # Asymmetric branching
        if child_index == 0:
            return base_radius * (1 + asymmetry * 0.3)
        else:
            return base_radius * (1 - asymmetry * 0.3)

    return base_radius


def apply_position_noise(
    position: np.ndarray,
    noise_factor: float,
    segment_length: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Apply random position noise to a point.

    Args:
        position: Original position
        noise_factor: Noise magnitude (0-1)
        segment_length: Reference length for scaling noise
        rng: Random number generator

    Returns:
        Position with noise applied
    """
    if noise_factor <= 0:
        return position.copy()

    noise = rng.uniform(-1, 1, 3) * noise_factor * segment_length * 0.3
    return position + noise


def create_branch_recursive(
    position: np.ndarray,
    direction: np.ndarray,
    radius: float,
    generation: int,
    max_generations: int,
    params: PerfusableNetworkParams,
    bounding_box: tuple[float, float, float],
    segments: List[m3d.Manifold],
    endpoints: List[VesselEndpoint],
    stats: VesselStats,
    rng: np.random.Generator,
    is_venous: bool = False
) -> None:
    """
    Recursively generate branching vascular tree.

    Args:
        position: Current branch start position
        direction: Current branch direction (normalized)
        radius: Current branch radius
        generation: Current generation level
        max_generations: Maximum generation depth
        params: Network parameters
        bounding_box: Bounding box dimensions
        segments: List to append vessel segments to
        endpoints: List to collect terminal endpoints for capillary bed
        stats: Vessel statistics tracker
        rng: Random number generator
        is_venous: True if generating venous return tree
    """
    # Convert diameters to radii in mm
    arteriole_radius = params.arteriole_diameter_um / 1000.0 / 2.0
    capillary_radius = params.capillary_diameter_um / 1000.0 / 2.0
    venule_radius = params.venule_diameter_um / 1000.0 / 2.0
    min_wall_mm = params.min_wall_thickness_um / 1000.0

    # Classify current vessel type
    vessel_type = classify_vessel_type(
        radius, arteriole_radius, capillary_radius, venule_radius, is_venous
    )

    # Termination conditions
    if generation >= max_generations or radius < capillary_radius:
        # Record endpoint for capillary bed connection
        endpoints.append(VesselEndpoint(
            position=position.copy(),
            direction=direction.copy(),
            radius=radius,
            vessel_type=vessel_type,
            generation=generation
        ))
        return

    # Calculate segment length based on generation
    base_length = bounding_box[2] / (max_generations * 1.2)
    segment_length = base_length * (1.5 - generation / max_generations * 0.8)

    # Add organic variation
    if params.enable_organic_variation and generation > 0:
        segment_length *= rng.uniform(0.7, 1.3)

    # Calculate end position
    end_position = position + direction * segment_length

    # Apply position noise if enabled
    if params.enable_organic_variation and params.position_noise > 0 and generation > 0:
        end_position = apply_position_noise(
            end_position, params.position_noise, segment_length, rng
        )

    # Clamp to bounding box
    bbox_x, bbox_y, bbox_z = bounding_box
    end_position = np.clip(end_position, [-bbox_x/2, -bbox_y/2, -bbox_z/2],
                          [bbox_x/2, bbox_y/2, bbox_z/2])

    # Update vessel statistics
    segment_actual_length = np.linalg.norm(end_position - position)
    stats.total_length_mm += segment_actual_length

    if vessel_type == 'artery':
        stats.artery_count += 1
    elif vessel_type == 'arteriole':
        stats.arteriole_count += 1
    elif vessel_type == 'capillary':
        stats.capillary_count += 1
    elif vessel_type == 'venule':
        stats.venule_count += 1
    elif vessel_type == 'vein':
        stats.vein_count += 1

    # Determine number of children
    num_children = 3 if params.enable_trifurcation and rng.random() < 0.2 else 2

    # Calculate child radius using Murray's law
    child_radius = calculate_child_radius(
        radius, num_children, params.murrays_law_ratio,
        params.bifurcation_asymmetry, 0
    )

    # Add diameter variance
    if params.enable_organic_variation:
        child_radius *= (1 + rng.uniform(-params.diameter_variance, params.diameter_variance))

    # Create vessel segment
    segment = make_vessel_segment(
        position,
        end_position,
        radius,
        child_radius,
        params.resolution,
        params.vessel_wall_thickness_ratio,
        params.enable_hollow_vessels,
        min_wall_mm
    )

    if segment.num_vert() > 0:
        segments.append(segment)

        # Add junction sphere for smooth connection
        junction = m3d.Manifold.sphere(radius * 1.05, params.resolution)
        junction = junction.translate([end_position[0], end_position[1], end_position[2]])
        segments.append(junction)

    # Generate child branches if not at last generation
    if generation < max_generations - 1:
        branching_angle = params.bifurcation_angle_deg * np.pi / 180

        for i in range(num_children):
            # Calculate angle offset for each child
            if num_children == 2:
                angle_offset = branching_angle if i == 0 else -branching_angle
            else:  # Trifurcation
                angle_offset = (i - 1) * branching_angle

            # Add angle noise
            if params.enable_organic_variation:
                angle_offset += rng.uniform(-params.angle_noise, params.angle_noise) * np.pi / 4

            # Random rotation axis perpendicular to current direction
            perp_axis = np.array([
                rng.uniform(-1, 1),
                rng.uniform(-1, 1),
                0.2 * rng.uniform(-1, 1)
            ])
            perp_axis = perp_axis / (np.linalg.norm(perp_axis) + 1e-6)

            # Rodrigues rotation formula
            k = perp_axis
            cos_theta = np.cos(angle_offset)
            sin_theta = np.sin(angle_offset)

            child_direction = (
                direction * cos_theta +
                np.cross(k, direction) * sin_theta +
                k * np.dot(k, direction) * (1 - cos_theta)
            )

            # Normalize
            child_direction = child_direction / (np.linalg.norm(child_direction) + 1e-6)

            # Calculate asymmetric child radius
            asymm_child_radius = calculate_child_radius(
                radius, num_children, params.murrays_law_ratio,
                params.bifurcation_asymmetry, i
            )

            # Recurse for child branch
            create_branch_recursive(
                end_position,
                child_direction,
                asymm_child_radius,
                generation + 1,
                max_generations,
                params,
                bounding_box,
                segments,
                endpoints,
                stats,
                rng,
                is_venous
            )
    else:
        # At last generation - record terminal endpoint for capillary bed
        # Classify the terminal vessel type based on child radius
        terminal_type = classify_vessel_type(
            child_radius, arteriole_radius, capillary_radius, venule_radius, is_venous
        )
        endpoints.append(VesselEndpoint(
            position=end_position.copy(),
            direction=direction.copy(),
            radius=child_radius,
            vessel_type=terminal_type,
            generation=generation
        ))


def create_capillary_bed(
    arterial_endpoints: List[VesselEndpoint],
    venous_endpoints: Optional[List[VesselEndpoint]],
    bounding_box: tuple[float, float, float],
    params: PerfusableNetworkParams,
    stats: VesselStats,
    rng: np.random.Generator
) -> List[m3d.Manifold]:
    """
    Create dense capillary network connecting arteriole terminals.

    Based on PMC vascularization studies:
    - Capillary density: 500-3000 per mm^2
    - Maximum cell-to-vessel distance: 150-200 um

    Args:
        arterial_endpoints: Terminal points of arteriole tree
        venous_endpoints: Terminal points of venule tree (if arterio-venous separation)
        bounding_box: Scaffold dimensions
        params: Network parameters
        stats: Vessel statistics tracker
        rng: Random number generator

    Returns:
        List of capillary segment manifolds
    """
    capillaries: List[m3d.Manifold] = []

    if not arterial_endpoints:
        return capillaries

    bbox_x, bbox_y, bbox_z = bounding_box
    capillary_radius = params.capillary_diameter_um / 1000.0 / 2.0
    max_cell_dist_mm = params.max_cell_distance_um / 1000.0
    min_wall_mm = params.min_wall_thickness_um / 1000.0

    # Calculate number of capillaries needed
    area_mm2 = bbox_x * bbox_y
    num_capillaries = int(params.capillary_density_per_mm2 * area_mm2)

    # Limit to reasonable number for performance
    num_capillaries = min(num_capillaries, 5000)

    # Get arteriole terminal positions
    arteriole_positions = [ep.position for ep in arterial_endpoints if ep.vessel_type == 'arteriole']

    # If no arterioles, use all endpoints
    if not arteriole_positions:
        arteriole_positions = [ep.position for ep in arterial_endpoints]

    # Get venule positions for connection targets
    if venous_endpoints:
        venule_positions = [ep.position for ep in venous_endpoints]
    else:
        # Create synthetic venule targets offset from arterioles
        venule_positions = []
        for pos in arteriole_positions:
            offset = rng.uniform(-1, 1, 3) * max_cell_dist_mm * 2
            venule_positions.append(pos + offset)

    # Create meandering capillaries connecting arterioles to venules
    for i in range(num_capillaries):
        # Pick random arteriole endpoint or generate point near one
        if arteriole_positions and rng.random() < 0.7:
            start_idx = rng.integers(0, len(arteriole_positions))
            start = arteriole_positions[start_idx].copy()
            # Add small offset
            start += rng.uniform(-1, 1, 3) * max_cell_dist_mm * 0.5
        else:
            # Random start within bounding box
            start = np.array([
                rng.uniform(-bbox_x/2 * 0.8, bbox_x/2 * 0.8),
                rng.uniform(-bbox_y/2 * 0.8, bbox_y/2 * 0.8),
                rng.uniform(-bbox_z/4, bbox_z/4)  # Near middle
            ])

        # Determine endpoint (toward venule if available)
        if venule_positions and rng.random() < 0.6:
            # Find nearest venule
            distances = [np.linalg.norm(start - vp) for vp in venule_positions]
            nearest_idx = np.argmin(distances)
            target = venule_positions[nearest_idx]
        else:
            # Random direction
            target = start + rng.uniform(-1, 1, 3) * max_cell_dist_mm * 3

        # Create wavy path if tortuosity > 0
        length = np.linalg.norm(target - start)
        if length < 0.01:  # Skip very short capillaries
            continue

        points = [start]
        current = start.copy()
        base_direction = (target - start) / length

        num_segments = max(3, int(length / 0.05))  # ~50um segments
        segment_len = length / num_segments

        for j in range(num_segments):
            # Direction with tortuosity-based deviation
            if params.capillary_tortuosity > 0:
                deviation = rng.uniform(-1, 1, 3) * params.capillary_tortuosity
                direction = base_direction + deviation
                direction = direction / (np.linalg.norm(direction) + 1e-6)
            else:
                direction = base_direction

            current = current + direction * segment_len
            points.append(current.copy())

        # Create capillary segments
        for j in range(len(points) - 1):
            cap = make_vessel_segment(
                points[j], points[j+1],
                capillary_radius, capillary_radius,
                max(6, params.resolution // 2),
                params.vessel_wall_thickness_ratio,
                params.enable_hollow_vessels,
                min_wall_mm
            )
            if cap.num_vert() > 0:
                capillaries.append(cap)
                stats.capillary_count += 1
                stats.total_length_mm += np.linalg.norm(points[j+1] - points[j])

    return capillaries


def create_anastomoses(
    endpoints: List[VesselEndpoint],
    density: float,
    params: PerfusableNetworkParams,
    stats: VesselStats,
    rng: np.random.Generator
) -> List[m3d.Manifold]:
    """
    Create cross-connections between nearby vessel endpoints.

    Anastomoses are cross-connections that improve perfusion redundancy.
    Higher density in well-vascularized organs (liver, kidney).

    Args:
        endpoints: List of vessel endpoints
        density: Connection density (0-1)
        params: Network parameters
        stats: Vessel statistics tracker
        rng: Random number generator

    Returns:
        List of anastomotic connection manifolds
    """
    anastomoses: List[m3d.Manifold] = []

    if density <= 0 or len(endpoints) < 2:
        return anastomoses

    min_wall_mm = params.min_wall_thickness_um / 1000.0
    max_dist = 2.0  # mm, max distance for anastomosis

    # Find pairs of nearby endpoints
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            pos_i = endpoints[i].position
            pos_j = endpoints[j].position
            r_i = endpoints[i].radius
            r_j = endpoints[j].radius

            dist = np.linalg.norm(pos_i - pos_j)

            # Create anastomosis based on distance and density
            if dist < max_dist and dist > 0.1 and rng.random() < density:
                # Anastomotic connection uses smaller radius
                conn_radius = min(r_i, r_j) * 0.5

                conn = make_vessel_segment(
                    pos_i, pos_j,
                    conn_radius, conn_radius,
                    max(6, params.resolution // 2),
                    params.vessel_wall_thickness_ratio,
                    params.enable_hollow_vessels,
                    min_wall_mm
                )

                if conn.num_vert() > 0:
                    anastomoses.append(conn)
                    stats.anastomosis_count += 1
                    stats.total_length_mm += dist

    return anastomoses


def generate_parallel_network(
    params: PerfusableNetworkParams,
    stats: VesselStats,
    rng: np.random.Generator
) -> Tuple[List[m3d.Manifold], List[VesselEndpoint]]:
    """
    Generate parallel vessel network topology.

    Multiple parallel main vessels with limited cross-connections.

    Args:
        params: Network parameters
        stats: Vessel statistics tracker
        rng: Random number generator

    Returns:
        Tuple of (segments list, endpoints list)
    """
    segments: List[m3d.Manifold] = []
    endpoints: List[VesselEndpoint] = []

    bbox_x, bbox_y, bbox_z = params.bounding_box_mm
    inlet_radius = params.large_vessel_diameter_mm / 2
    min_wall_mm = params.min_wall_thickness_um / 1000.0

    # Number of parallel main vessels
    num_parallel = max(2, min(5, int(bbox_x / 3)))

    # Spacing between parallel vessels
    spacing = bbox_x / (num_parallel + 1)

    for p in range(num_parallel):
        x_pos = -bbox_x/2 + spacing * (p + 1)

        # Determine start based on inlet position
        if params.inlet_position == 'top':
            start = np.array([x_pos, 0.0, bbox_z/2])
            direction = np.array([0.0, 0.0, -1.0])
        elif params.inlet_position == 'bottom':
            start = np.array([x_pos, 0.0, -bbox_z/2])
            direction = np.array([0.0, 0.0, 1.0])
        else:  # side
            start = np.array([-bbox_x/2, x_pos - bbox_x/2, 0.0])
            direction = np.array([1.0, 0.0, 0.0])

        # Create main vessel
        end = start + direction * bbox_z * 0.8
        vessel = make_vessel_segment(
            start, end,
            inlet_radius * 0.7, inlet_radius * 0.5,
            params.resolution,
            params.vessel_wall_thickness_ratio,
            params.enable_hollow_vessels,
            min_wall_mm
        )

        if vessel.num_vert() > 0:
            segments.append(vessel)
            stats.artery_count += 1
            stats.total_length_mm += np.linalg.norm(end - start)

        # Add branching from main vessel
        num_branches = params.num_branching_generations
        for b in range(num_branches):
            branch_pos = start + direction * (bbox_z * 0.8 * (b + 1) / (num_branches + 1))
            branch_dir = np.array([
                rng.uniform(-0.5, 0.5),
                rng.uniform(-0.5, 0.5),
                direction[2] * 0.5
            ])
            branch_dir = branch_dir / (np.linalg.norm(branch_dir) + 1e-6)

            branch_radius = inlet_radius * 0.3 * (1 - b / num_branches * 0.5)

            create_branch_recursive(
                branch_pos,
                branch_dir,
                branch_radius,
                generation=2,
                max_generations=params.num_branching_generations,
                params=params,
                bounding_box=params.bounding_box_mm,
                segments=segments,
                endpoints=endpoints,
                stats=stats,
                rng=rng,
                is_venous=False
            )

        # Record endpoint
        endpoints.append(VesselEndpoint(
            position=end,
            direction=direction,
            radius=inlet_radius * 0.5,
            vessel_type='arteriole',
            generation=1
        ))

    return segments, endpoints


def generate_loop_network(
    params: PerfusableNetworkParams,
    stats: VesselStats,
    rng: np.random.Generator
) -> Tuple[List[m3d.Manifold], List[VesselEndpoint]]:
    """
    Generate looping/circular vessel network topology.

    Vessels form loops for redundant perfusion pathways.

    Args:
        params: Network parameters
        stats: Vessel statistics tracker
        rng: Random number generator

    Returns:
        Tuple of (segments list, endpoints list)
    """
    segments: List[m3d.Manifold] = []
    endpoints: List[VesselEndpoint] = []

    bbox_x, bbox_y, bbox_z = params.bounding_box_mm
    inlet_radius = params.large_vessel_diameter_mm / 2
    min_wall_mm = params.min_wall_thickness_um / 1000.0

    # Create concentric loops at different z-levels
    num_loops = max(2, min(5, params.num_branching_generations // 2))
    z_spacing = bbox_z * 0.7 / num_loops

    for loop_idx in range(num_loops):
        z_level = bbox_z/2 - z_spacing * (loop_idx + 0.5)
        loop_radius = min(bbox_x, bbox_y) / 2 * (0.9 - loop_idx * 0.15)
        vessel_radius = inlet_radius * (0.6 - loop_idx * 0.1)

        # Create loop as series of segments
        num_segments = max(8, params.resolution)
        prev_point = None

        for s in range(num_segments + 1):
            angle = 2 * np.pi * s / num_segments
            point = np.array([
                loop_radius * np.cos(angle),
                loop_radius * np.sin(angle),
                z_level
            ])

            if prev_point is not None:
                seg = make_vessel_segment(
                    prev_point, point,
                    vessel_radius, vessel_radius,
                    params.resolution,
                    params.vessel_wall_thickness_ratio,
                    params.enable_hollow_vessels,
                    min_wall_mm
                )
                if seg.num_vert() > 0:
                    segments.append(seg)
                    stats.artery_count += 1
                    stats.total_length_mm += np.linalg.norm(point - prev_point)

            # Add inward branches
            if s % 2 == 0 and s < num_segments:
                branch_dir = -np.array([np.cos(angle), np.sin(angle), rng.uniform(-0.2, 0.2)])
                branch_dir = branch_dir / (np.linalg.norm(branch_dir) + 1e-6)

                create_branch_recursive(
                    point,
                    branch_dir,
                    vessel_radius * 0.5,
                    generation=loop_idx + 2,
                    max_generations=params.num_branching_generations,
                    params=params,
                    bounding_box=params.bounding_box_mm,
                    segments=segments,
                    endpoints=endpoints,
                    stats=stats,
                    rng=rng,
                    is_venous=False
                )

            prev_point = point

    # Create vertical connections between loops
    if num_loops > 1:
        for angle_idx in range(4):
            angle = np.pi / 2 * angle_idx
            for loop_idx in range(num_loops - 1):
                z1 = bbox_z/2 - z_spacing * (loop_idx + 0.5)
                z2 = bbox_z/2 - z_spacing * (loop_idx + 1.5)
                r1 = min(bbox_x, bbox_y) / 2 * (0.9 - loop_idx * 0.15)
                r2 = min(bbox_x, bbox_y) / 2 * (0.9 - (loop_idx + 1) * 0.15)

                p1 = np.array([r1 * np.cos(angle), r1 * np.sin(angle), z1])
                p2 = np.array([r2 * np.cos(angle), r2 * np.sin(angle), z2])

                conn_radius = inlet_radius * 0.4
                conn = make_vessel_segment(
                    p1, p2,
                    conn_radius, conn_radius,
                    params.resolution,
                    params.vessel_wall_thickness_ratio,
                    params.enable_hollow_vessels,
                    min_wall_mm
                )
                if conn.num_vert() > 0:
                    segments.append(conn)
                    stats.total_length_mm += np.linalg.norm(p2 - p1)

    return segments, endpoints


def create_inlet_outlet_connections(
    segments: List[m3d.Manifold],
    endpoints: List[VesselEndpoint],
    params: PerfusableNetworkParams,
    stats: VesselStats,
    rng: np.random.Generator
) -> List[m3d.Manifold]:
    """
    Create connections to outlet based on outlet_position parameter.

    Args:
        segments: Existing vessel segments
        endpoints: Terminal vessel endpoints
        params: Network parameters
        stats: Vessel statistics tracker
        rng: Random number generator

    Returns:
        List of outlet connection manifolds
    """
    outlet_segments: List[m3d.Manifold] = []
    bbox_x, bbox_y, bbox_z = params.bounding_box_mm
    min_wall_mm = params.min_wall_thickness_um / 1000.0

    # Determine outlet position and direction
    if params.outlet_position == 'top':
        outlet_pos = np.array([0.0, 0.0, bbox_z/2])
        outlet_dir = np.array([0.0, 0.0, 1.0])
    elif params.outlet_position == 'bottom':
        outlet_pos = np.array([0.0, 0.0, -bbox_z/2])
        outlet_dir = np.array([0.0, 0.0, -1.0])
    elif params.outlet_position == 'side':
        outlet_pos = np.array([bbox_x/2, 0.0, 0.0])
        outlet_dir = np.array([1.0, 0.0, 0.0])
    else:  # 'distributed'
        # Multiple outlets at terminal points - no central collection
        return outlet_segments

    # For non-distributed outlets, create venous return if separation enabled
    if params.enable_arterio_venous_separation and endpoints:
        # Create collection vessel toward outlet
        venule_radius = params.venule_diameter_um / 1000.0 / 2.0

        # Find endpoints far from inlet that should drain to outlet
        inlet_pos = np.array([0.0, 0.0, bbox_z/2 if params.inlet_position == 'top' else -bbox_z/2])

        for ep in endpoints:
            dist_to_inlet = np.linalg.norm(ep.position - inlet_pos)
            dist_to_outlet = np.linalg.norm(ep.position - outlet_pos)

            # Connect terminals that are closer to outlet than inlet
            if dist_to_outlet < dist_to_inlet and rng.random() < 0.3:
                conn = make_vessel_segment(
                    ep.position, outlet_pos,
                    ep.radius * 0.5, venule_radius,
                    max(6, params.resolution // 2),
                    params.vessel_wall_thickness_ratio,
                    params.enable_hollow_vessels,
                    min_wall_mm
                )
                if conn.num_vert() > 0:
                    outlet_segments.append(conn)
                    stats.venule_count += 1
                    stats.total_length_mm += np.linalg.norm(outlet_pos - ep.position)

    return outlet_segments


def generate_perfusable_network(params: PerfusableNetworkParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate a perfusable vascular network.

    Creates a branching tree structure following Murray's law for optimal
    fluid flow characteristics. Suitable for thick tissue engineering scaffolds.

    Supports multiple network topologies:
    - hierarchical: Traditional tree-like branching (default)
    - anastomosing: Tree with cross-connections for redundancy
    - parallel: Multiple parallel main vessels
    - loop: Circular/looping patterns for redundant perfusion

    Args:
        params: PerfusableNetworkParams specifying network structure

    Returns:
        Tuple of (manifold, stats_dict)
        - manifold: The 3D vascular network geometry
        - stats_dict: Dictionary with comprehensive vessel statistics

    Raises:
        ValueError: If no vessel segments are generated
    """
    rng = np.random.default_rng(params.seed)
    stats = VesselStats()

    # Calculate inlet radius from large vessel diameter
    inlet_radius = params.large_vessel_diameter_mm / 2

    bbox_x, bbox_y, bbox_z = params.bounding_box_mm

    # Initialize segment and endpoint lists
    segments: List[m3d.Manifold] = []
    arterial_endpoints: List[VesselEndpoint] = []
    venous_endpoints: List[VesselEndpoint] = []

    # Determine starting position based on inlet_position
    if params.inlet_position == 'top':
        start_position = np.array([0.0, 0.0, bbox_z/2])
        start_direction = np.array([0.0, 0.0, -1.0])
    elif params.inlet_position == 'bottom':
        start_position = np.array([0.0, 0.0, -bbox_z/2])
        start_direction = np.array([0.0, 0.0, 1.0])
    else:  # side
        start_position = np.array([-bbox_x/2, 0.0, 0.0])
        start_direction = np.array([1.0, 0.0, 0.0])

    # Create inlet sphere
    inlet_sphere = m3d.Manifold.sphere(inlet_radius, params.resolution)
    inlet_sphere = inlet_sphere.translate([start_position[0], start_position[1], start_position[2]])
    segments.append(inlet_sphere)

    # Generate network based on topology
    if params.network_topology == 'hierarchical':
        # Standard tree-based branching
        create_branch_recursive(
            start_position,
            start_direction,
            inlet_radius,
            generation=0,
            max_generations=params.num_branching_generations,
            params=params,
            bounding_box=params.bounding_box_mm,
            segments=segments,
            endpoints=arterial_endpoints,
            stats=stats,
            rng=rng,
            is_venous=False
        )

    elif params.network_topology == 'anastomosing':
        # Tree with cross-connections
        create_branch_recursive(
            start_position,
            start_direction,
            inlet_radius,
            generation=0,
            max_generations=params.num_branching_generations,
            params=params,
            bounding_box=params.bounding_box_mm,
            segments=segments,
            endpoints=arterial_endpoints,
            stats=stats,
            rng=rng,
            is_venous=False
        )

        # Add anastomoses (use parameter or default 0.3 for this topology)
        anastomosis_density = params.anastomosis_density if params.anastomosis_density > 0 else 0.3
        anastomoses = create_anastomoses(
            arterial_endpoints,
            anastomosis_density,
            params,
            stats,
            rng
        )
        segments.extend(anastomoses)

    elif params.network_topology == 'parallel':
        # Multiple parallel main vessels
        parallel_segments, arterial_endpoints = generate_parallel_network(
            params, stats, rng
        )
        segments.extend(parallel_segments)

    elif params.network_topology == 'loop':
        # Circular/looping patterns
        loop_segments, arterial_endpoints = generate_loop_network(
            params, stats, rng
        )
        segments.extend(loop_segments)

    # Generate venous return tree if arterio-venous separation enabled
    if params.enable_arterio_venous_separation:
        # Determine venous inlet (opposite of arterial outlet)
        if params.outlet_position == 'top':
            venous_start = np.array([0.0, 0.0, bbox_z/2])
            venous_dir = np.array([0.0, 0.0, -1.0])
        elif params.outlet_position == 'bottom':
            venous_start = np.array([0.0, 0.0, -bbox_z/2])
            venous_dir = np.array([0.0, 0.0, 1.0])
        elif params.outlet_position == 'side':
            venous_start = np.array([bbox_x/2, 0.0, 0.0])
            venous_dir = np.array([-1.0, 0.0, 0.0])
        else:  # distributed - place venous return at opposite of inlet
            if params.inlet_position == 'top':
                venous_start = np.array([0.0, 0.0, -bbox_z/2])
                venous_dir = np.array([0.0, 0.0, 1.0])
            elif params.inlet_position == 'bottom':
                venous_start = np.array([0.0, 0.0, bbox_z/2])
                venous_dir = np.array([0.0, 0.0, -1.0])
            else:
                venous_start = np.array([bbox_x/2, 0.0, 0.0])
                venous_dir = np.array([-1.0, 0.0, 0.0])

        venous_radius = params.venule_diameter_um / 1000.0 * 2  # Larger collecting veins

        # Create venous inlet sphere
        venous_sphere = m3d.Manifold.sphere(venous_radius, params.resolution)
        venous_sphere = venous_sphere.translate([venous_start[0], venous_start[1], venous_start[2]])
        segments.append(venous_sphere)

        # Generate venous tree (fewer generations, converging)
        create_branch_recursive(
            venous_start,
            venous_dir,
            venous_radius,
            generation=0,
            max_generations=max(3, params.num_branching_generations - 2),
            params=params,
            bounding_box=params.bounding_box_mm,
            segments=segments,
            endpoints=venous_endpoints,
            stats=stats,
            rng=rng,
            is_venous=True
        )

    # Add capillary bed if enabled
    if params.enable_capillary_bed:
        capillary_segments = create_capillary_bed(
            arterial_endpoints,
            venous_endpoints if params.enable_arterio_venous_separation else None,
            params.bounding_box_mm,
            params,
            stats,
            rng
        )
        segments.extend(capillary_segments)

    # Add anastomoses if density > 0 and not already done for anastomosing topology
    if params.anastomosis_density > 0 and params.network_topology != 'anastomosing':
        all_endpoints = arterial_endpoints + venous_endpoints
        anastomoses = create_anastomoses(
            all_endpoints,
            params.anastomosis_density,
            params,
            stats,
            rng
        )
        segments.extend(anastomoses)

    # Create outlet connections
    outlet_segments = create_inlet_outlet_connections(
        segments,
        arterial_endpoints + venous_endpoints,
        params,
        stats,
        rng
    )
    segments.extend(outlet_segments)

    if not segments:
        raise ValueError("No vessel segments generated")

    # Union all segments
    result = batch_union(segments)

    # Calculate statistics
    mesh = result.to_mesh()
    volume = result.volume() if hasattr(result, 'volume') else 0

    # Compile comprehensive stats
    stats_dict = {
        'triangle_count': len(mesh.tri_verts) // 3 if hasattr(mesh, 'tri_verts') else 0,
        'volume_mm3': volume,
        'segment_count': len(segments),
        'generation_count': params.num_branching_generations,

        # Network configuration
        'network_type': params.network_type,
        'network_topology': params.network_topology,
        'inlet_position': params.inlet_position,
        'outlet_position': params.outlet_position,

        # Vessel dimensions
        'large_vessel_diameter_mm': params.large_vessel_diameter_mm,
        'arteriole_diameter_um': params.arteriole_diameter_um,
        'capillary_diameter_um': params.capillary_diameter_um,
        'venule_diameter_um': params.venule_diameter_um,

        # Branching parameters
        'murrays_law_ratio': params.murrays_law_ratio,
        'bifurcation_angle_deg': params.bifurcation_angle_deg,

        # Vessel counts by type
        'artery_count': stats.artery_count,
        'arteriole_count': stats.arteriole_count,
        'capillary_count': stats.capillary_count,
        'venule_count': stats.venule_count,
        'vein_count': stats.vein_count,
        'anastomosis_count': stats.anastomosis_count,

        # Network metrics
        'total_vessel_length_mm': stats.total_length_mm,
        'enable_capillary_bed': params.enable_capillary_bed,
        'enable_arterio_venous_separation': params.enable_arterio_venous_separation,
        'anastomosis_density': params.anastomosis_density,

        # Metadata
        'scaffold_type': 'perfusable_network'
    }

    return result, stats_dict


def generate_perfusable_network_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate perfusable network from dictionary parameters (for API compatibility).

    Args:
        params: Dictionary with keys matching PerfusableNetworkParams fields

    Returns:
        Tuple of (manifold, stats_dict)
    """
    # Handle bounding_box variations
    bbox = params.get('bounding_box_mm', params.get('bounding_box', (10.0, 10.0, 10.0)))
    if isinstance(bbox, dict):
        bbox = (bbox.get('x', 10.0), bbox.get('y', 10.0), bbox.get('z', 10.0))

    p = PerfusableNetworkParams(
        # Multi-scale vessel hierarchy
        large_vessel_diameter_mm=params.get('large_vessel_diameter_mm', params.get('large_vessel_diameter', params.get('inlet_diameter_mm', params.get('inlet_diameter', 5.0)))),
        arteriole_diameter_um=params.get('arteriole_diameter_um', params.get('arteriole_diameter', 100.0)),
        capillary_diameter_um=params.get('capillary_diameter_um', params.get('capillary_diameter', 8.0)),
        venule_diameter_um=params.get('venule_diameter_um', params.get('venule_diameter', 50.0)),

        # Branching parameters
        num_branching_generations=params.get('num_branching_generations', params.get('branch_generations', params.get('branching_generations', params.get('generations', 7)))),
        murrays_law_ratio=params.get('murrays_law_ratio', params.get('murray_ratio', 1.26)),
        bifurcation_angle_deg=params.get('bifurcation_angle_deg', params.get('bifurcation_angle', params.get('branching_angle_deg', params.get('branching_angle', 70.0)))),
        bifurcation_asymmetry=params.get('bifurcation_asymmetry', 0.0),
        enable_trifurcation=params.get('enable_trifurcation', False),

        # Capillary network
        capillary_density_per_mm2=params.get('capillary_density_per_mm2', params.get('capillary_density', 1500.0)),
        max_cell_distance_um=params.get('max_cell_distance_um', params.get('max_cell_distance', 200.0)),
        enable_capillary_bed=params.get('enable_capillary_bed', False),
        capillary_tortuosity=params.get('capillary_tortuosity', 0.2),

        # Flow parameters
        design_flow_rate_ml_min=params.get('design_flow_rate_ml_min', params.get('flow_rate', 1.0)),
        target_wall_shear_stress_pa=params.get('target_wall_shear_stress_pa', params.get('wall_shear_stress', 1.5)),

        # Vessel wall
        vessel_wall_thickness_ratio=params.get('vessel_wall_thickness_ratio', params.get('wall_thickness_ratio', 0.1)),
        enable_hollow_vessels=params.get('enable_hollow_vessels', False),
        min_wall_thickness_um=params.get('min_wall_thickness_um', params.get('min_wall_thickness', 40.0)),  # um (40+ um for bioprinting feasibility)

        # Network topology
        network_topology=params.get('network_topology', 'hierarchical'),
        enable_arterio_venous_separation=params.get('enable_arterio_venous_separation', False),
        anastomosis_density=params.get('anastomosis_density', 0.0),

        # Scaffold geometry
        bounding_box_mm=tuple(bbox),
        network_type=params.get('network_type', 'arterial'),
        inlet_position=params.get('inlet_position', 'top'),
        outlet_position=params.get('outlet_position', 'bottom'),

        # Organic variation
        enable_organic_variation=params.get('enable_organic_variation', False),
        position_noise=params.get('position_noise', 0.1),
        angle_noise=params.get('angle_noise', 0.1),
        diameter_variance=params.get('diameter_variance', 0.1),

        # Quality/Resolution
        resolution=params.get('resolution', 12),
        seed=params.get('seed', params.get('random_seed', 42)),
    )

    return generate_perfusable_network(p)
