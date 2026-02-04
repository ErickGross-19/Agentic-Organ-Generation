"""
Cardiac patch scaffold generator.

Creates aligned microfibrous scaffolds for cardiomyocyte culture with:
- Parallel fibers in grid pattern mimicking native myocardium
- Angular offset between layers for mechanical reinforcement
- Controlled fiber spacing and diameter based on cardiomyocyte dimensions
- Multi-layer architecture with epicardial helix angle variation
- Capillary channel integration for vascularization
- Porous structure for nutrient diffusion
"""

from __future__ import annotations
import manifold3d as m3d
import numpy as np
from dataclasses import dataclass
from typing import Optional
from ..core import batch_union


class SpatialGrid:
    """Fast spatial lookup using grid-based hashing for O(1) neighbor queries."""

    def __init__(self, cell_size: float, bounds: tuple[float, float, float, float]):
        """
        Args:
            cell_size: Size of each grid cell (should be >= min_spacing)
            bounds: (min_x, max_x, min_y, max_y)
        """
        self.cell_size = cell_size
        self.min_x, self.max_x, self.min_y, self.max_y = bounds
        self.nx = max(1, int((self.max_x - self.min_x) / cell_size) + 1)
        self.ny = max(1, int((self.max_y - self.min_y) / cell_size) + 1)
        self.grid: dict[tuple[int, int], list[np.ndarray]] = {}

    def _cell_idx(self, x: float, y: float) -> tuple[int, int]:
        """Get grid cell indices for a point."""
        ix = int((x - self.min_x) / self.cell_size)
        iy = int((y - self.min_y) / self.cell_size)
        return (max(0, min(ix, self.nx - 1)), max(0, min(iy, self.ny - 1)))

    def insert(self, pos: np.ndarray) -> None:
        """Insert a position into the grid."""
        cell = self._cell_idx(pos[0], pos[1])
        if cell not in self.grid:
            self.grid[cell] = []
        self.grid[cell].append(pos)

    def check_min_distance(self, pos: np.ndarray, min_dist: float) -> bool:
        """Check if position is at least min_dist from all existing points. O(1) average."""
        cx, cy = self._cell_idx(pos[0], pos[1])
        # Check 3x3 neighborhood of cells
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                cell = (cx + dx, cy + dy)
                if cell in self.grid:
                    for existing in self.grid[cell]:
                        dist_sq = (pos[0] - existing[0])**2 + (pos[1] - existing[1])**2
                        if dist_sq < min_dist * min_dist:
                            return False
        return True

    def get_all_positions(self) -> list[np.ndarray]:
        """Get all stored positions."""
        result = []
        for positions in self.grid.values():
            result.extend(positions)
        return result


@dataclass
class CardiacPatchParams:
    """
    Parameters for cardiac patch scaffold generation.

    Based on native cardiac tissue architecture:
    - Cardiomyocytes: 100-150 um long, 10-35 um diameter
    - Sarcomere: 1.8-2.2 um at resting length
    - Fiber bundles: 50-100 um diameter
    - Capillary density: 2500-3500/mm^2
    """
    # === Cardiomyocyte Dimensions ===
    cardiomyocyte_length: float = 100.0  # um (80-150 um typical)
    cardiomyocyte_diameter: float = 25.0  # um (10-35 um typical)
    sarcomere_length: float = 1.8  # um (1.8-2.2 um at rest)

    # === Fiber Architecture ===
    fiber_diameter: float = 80.0  # um (50-100 um for fiber bundles)
    fiber_spacing: float = 150.0  # um (based on cardiomyocyte + ECM)
    fiber_alignment_degree: float = 0.9  # 0-1 (1 = perfectly aligned)

    # === Layer Configuration ===
    # Transmural fiber rotation: epicardium (outer) has negative helix angle (left-handed),
    # endocardium (inner) has positive helix angle (right-handed)
    # Native cardiac tissue exhibits ~120° total rotation from epi to endo
    epicardial_helix_angle: float = -60.0  # degrees (typically -30° to -70°, negative = left-handed helix)
    endocardial_helix_angle: float = 60.0  # degrees (typically +30° to +80°, positive = right-handed helix)
    layer_count: int = 3  # number of fiber layers
    enable_helix_gradient: bool = True  # gradient from endo to epicardial

    # === Patch Dimensions ===
    patch_length: float = 10.0  # mm
    patch_width: float = 10.0  # mm
    patch_thickness: float = 2.0  # mm (native: 10-15mm, scaffold: 0.5-3mm)

    # === Porosity ===
    porosity: float = 0.75  # target void fraction (70-85% typical)
    pore_size: float = 50.0  # um (20-100 um optimal for cardiac cells)
    enable_interconnected_pores: bool = True
    pore_interconnectivity: float = 0.85  # 0-1 connectivity ratio

    # === Vascularization Features ===
    enable_capillary_channels: bool = True
    capillary_diameter: float = 8.0  # um (5-10 um typical)
    capillary_density: float = 3000.0  # per mm^2 (2500-3500 typical)
    capillary_spacing: float = 20.0  # um (based on oxygen diffusion ~200um max)

    # === Mechanical Features ===
    enable_cross_fibers: bool = True  # perpendicular fiber reinforcement
    cross_fiber_ratio: float = 0.3  # ratio of cross to main fibers
    enable_fiber_crimping: bool = False  # wavy fiber pattern
    crimp_amplitude: float = 5.0  # um
    crimp_wavelength: float = 50.0  # um

    # === Electrical Guidance ===
    enable_conduction_channels: bool = False  # channels for electrical propagation
    conduction_channel_diameter: float = 100.0  # um
    conduction_channel_spacing: float = 500.0  # um

    # === Stochastic Variation ===
    position_noise: float = 0.0  # 0-1 random position jitter
    diameter_variance: float = 0.0  # 0-1 random diameter variation
    angle_noise: float = 0.0  # 0-1 random angle variation
    seed: int = 42  # random seed for reproducibility

    # === Resolution ===
    resolution: int = 8  # cylinder segments


def make_fiber(start: np.ndarray, end: np.ndarray, radius: float, resolution: int) -> m3d.Manifold:
    """
    Create a cylindrical fiber between two points.

    Args:
        start: Starting point (x, y, z)
        end: Ending point (x, y, z)
        radius: Fiber radius
        resolution: Number of segments around cylinder

    Returns:
        Fiber manifold
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dz = end[2] - start[2]
    length = np.sqrt(dx*dx + dy*dy + dz*dz)

    if length < 1e-6:
        return m3d.Manifold()

    fiber = m3d.Manifold.cylinder(length, radius, radius, resolution)

    h = np.sqrt(dx*dx + dy*dy)
    if h > 0.001 or abs(dz) > 0.001:
        tilt = np.arctan2(h, dz) * 180 / np.pi
        azim = np.arctan2(dy, dx) * 180 / np.pi
        fiber = fiber.rotate([0, tilt, 0]).rotate([0, 0, azim])

    return fiber.translate([start[0], start[1], start[2]])


def make_crimped_fiber(start: np.ndarray, end: np.ndarray, radius: float,
                       amplitude: float, wavelength: float, resolution: int) -> m3d.Manifold:
    """
    Create a crimped (wavy) fiber between two points.

    Args:
        start: Starting point
        end: Ending point
        radius: Fiber radius
        amplitude: Crimp amplitude in mm
        wavelength: Crimp wavelength in mm
        resolution: Cylinder resolution

    Returns:
        Crimped fiber manifold
    """
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return m3d.Manifold()

    direction = direction / length

    # Find perpendicular vector for crimp direction
    if abs(direction[2]) < 0.9:
        perp = np.cross(direction, np.array([0, 0, 1]))
    else:
        perp = np.cross(direction, np.array([1, 0, 0]))
    perp = perp / np.linalg.norm(perp)

    # Create segments along crimped path
    num_segments = max(10, int(length / (wavelength / 4)))
    segments = []

    for i in range(num_segments):
        t0 = i / num_segments
        t1 = (i + 1) / num_segments

        # Position along main axis
        p0 = start + direction * length * t0
        p1 = start + direction * length * t1

        # Add sinusoidal offset
        offset0 = amplitude * np.sin(2 * np.pi * t0 * length / wavelength)
        offset1 = amplitude * np.sin(2 * np.pi * t1 * length / wavelength)

        p0 = p0 + perp * offset0
        p1 = p1 + perp * offset1

        seg = make_fiber(p0, p1, radius, resolution)
        if seg.num_vert() > 0:
            segments.append(seg)

    if not segments:
        return m3d.Manifold()

    return batch_union(segments)


def generate_cardiac_patch(params: CardiacPatchParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate a cardiac patch scaffold with aligned microfibers.

    Args:
        params: CardiacPatchParams specifying geometry

    Returns:
        Tuple of (manifold, stats_dict)
        - manifold: The 3D geometry
        - stats_dict: Dictionary with triangle_count, volume_mm3, fiber_count, scaffold_type

    Raises:
        ValueError: If no fibers are generated
    """
    np.random.seed(params.seed)

    # Convert um to mm
    fiber_radius_mm = params.fiber_diameter / 2000.0
    fiber_spacing_mm = params.fiber_spacing / 1000.0
    capillary_radius_mm = params.capillary_diameter / 2000.0
    pore_radius_mm = params.pore_size / 2000.0
    crimp_amp_mm = params.crimp_amplitude / 1000.0

    # Convert cardiomyocyte dimensions to mm
    cardiomyocyte_length_mm = params.cardiomyocyte_length / 1000.0
    cardiomyocyte_diameter_mm = params.cardiomyocyte_diameter / 1000.0
    sarcomere_length_mm = params.sarcomere_length / 1000.0
    capillary_spacing_mm = params.capillary_spacing / 1000.0

    # IMPLEMENTATION: cardiomyocyte_diameter affects minimum fiber spacing
    # Native cardiomyocytes need sufficient space between fibers for attachment and alignment
    # Enforce minimum fiber spacing based on cardiomyocyte diameter (cells need ~1.5x their diameter)
    min_fiber_spacing_mm = cardiomyocyte_diameter_mm * 1.5
    fiber_spacing_mm = max(fiber_spacing_mm, min_fiber_spacing_mm)

    # IMPLEMENTATION: sarcomere_length influences crimp_wavelength when crimping is enabled
    # Sarcomeres are the basic contractile units; crimp wavelength should be a multiple of sarcomere length
    # Native cardiac fibers have crimp wavelength of ~20-30 sarcomere lengths
    if params.enable_fiber_crimping:
        # Calculate sarcomere-based wavelength for physiological relevance (25x sarcomere length)
        sarcomere_based_wavelength_mm = sarcomere_length_mm * 25.0
        user_specified_wavelength_mm = params.crimp_wavelength / 1000.0
        # Blend user-specified wavelength with sarcomere-based value using scaling factor
        # The sarcomere length scales the user's specified wavelength to maintain physiological relevance
        sarcomere_scale_factor = sarcomere_based_wavelength_mm / (1.8 / 1000.0 * 25.0)  # Normalize to default sarcomere
        crimp_wave_mm = user_specified_wavelength_mm * sarcomere_scale_factor
    else:
        crimp_wave_mm = params.crimp_wavelength / 1000.0

    px, py, pz = params.patch_length, params.patch_width, params.patch_thickness
    layer_height = pz / params.layer_count

    all_fibers = []
    all_channels = []
    total_fiber_count = 0

    for layer_idx in range(params.layer_count):
        z_center = (layer_idx + 0.5) * layer_height

        # Calculate layer helix angle (gradient from endo to epicardial)
        if params.enable_helix_gradient and params.layer_count > 1:
            t = layer_idx / (params.layer_count - 1)
            layer_angle = params.endocardial_helix_angle + t * (
                params.epicardial_helix_angle - params.endocardial_helix_angle
            )
        else:
            layer_angle = params.epicardial_helix_angle * (1 if layer_idx % 2 == 0 else -1)

        # Add angle noise
        if params.angle_noise > 0:
            layer_angle += np.random.uniform(-1, 1) * params.angle_noise * 10

        angle_rad = np.radians(layer_angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # Generate main fibers along the helical direction
        num_fibers = int(py / fiber_spacing_mm) + 1
        for i in range(num_fibers):
            y_base = -py/2 + i * fiber_spacing_mm

            # Apply position noise
            if params.position_noise > 0:
                y_pos = y_base + np.random.uniform(-1, 1) * params.position_noise * fiber_spacing_mm * 0.3
            else:
                y_pos = y_base

            # Fiber runs across entire X dimension
            start_local = np.array([-px/2, y_pos, 0])
            end_local = np.array([px/2, y_pos, 0])

            # Rotate fiber by layer helix angle
            start_rot = np.array([
                start_local[0] * cos_a - start_local[1] * sin_a,
                start_local[0] * sin_a + start_local[1] * cos_a,
                z_center
            ])
            end_rot = np.array([
                end_local[0] * cos_a - end_local[1] * sin_a,
                end_local[0] * sin_a + end_local[1] * cos_a,
                z_center
            ])

            # Apply diameter variance
            if params.diameter_variance > 0:
                radius = fiber_radius_mm * (1 + np.random.uniform(-1, 1) * params.diameter_variance * 0.2)
            else:
                radius = fiber_radius_mm

            # Apply fiber alignment degree - adds angular deviation per fiber
            # 1.0 = perfectly aligned, 0.5 = up to ±45° deviation
            if params.fiber_alignment_degree < 1.0:
                alignment_deviation = np.random.uniform(-1, 1) * (1 - params.fiber_alignment_degree) * 90
                deviation_rad = np.radians(alignment_deviation)
                cos_d, sin_d = np.cos(deviation_rad), np.sin(deviation_rad)
                # Apply additional rotation around z-axis to both start and end
                start_rot = np.array([
                    start_rot[0] * cos_d - start_rot[1] * sin_d,
                    start_rot[0] * sin_d + start_rot[1] * cos_d,
                    start_rot[2]
                ])
                end_rot = np.array([
                    end_rot[0] * cos_d - end_rot[1] * sin_d,
                    end_rot[0] * sin_d + end_rot[1] * cos_d,
                    end_rot[2]
                ])

            # Clip to bounding box
            if abs(start_rot[0]) <= px/2 * 1.1 and abs(start_rot[1]) <= py/2 * 1.1:
                if abs(end_rot[0]) <= px/2 * 1.1 and abs(end_rot[1]) <= py/2 * 1.1:
                    if params.enable_fiber_crimping:
                        fiber = make_crimped_fiber(
                            start_rot, end_rot, radius,
                            crimp_amp_mm, crimp_wave_mm, params.resolution
                        )
                    else:
                        fiber = make_fiber(start_rot, end_rot, radius, params.resolution)

                    if fiber.num_vert() > 0:
                        all_fibers.append(fiber)
                        total_fiber_count += 1

                    # IMPLEMENTATION: cardiomyocyte_length defines cross-connection spacing
                    # Add cross-connection points along the fiber at intervals based on cardiomyocyte length
                    # This mimics the intercalated disc spacing in native cardiac tissue
                    if params.enable_cross_fibers:
                        fiber_direction = end_rot - start_rot
                        fiber_length = np.linalg.norm(fiber_direction)
                        if fiber_length > cardiomyocyte_length_mm:
                            fiber_direction_norm = fiber_direction / fiber_length
                            # Create small perpendicular cross-connections at cardiomyocyte-length intervals
                            num_connections = int(fiber_length / cardiomyocyte_length_mm)
                            cross_connection_radius = radius * 0.4  # Smaller than main fiber
                            cross_connection_length = cardiomyocyte_diameter_mm * 0.8  # Short stubs

                            for conn_idx in range(1, num_connections):
                                # Position along the fiber
                                conn_pos = start_rot + fiber_direction_norm * (conn_idx * cardiomyocyte_length_mm)
                                # Perpendicular direction (in XY plane)
                                perp_dir = np.array([-fiber_direction_norm[1], fiber_direction_norm[0], 0])
                                perp_dir = perp_dir / (np.linalg.norm(perp_dir) + 1e-9)

                                # Create short cross-connection stubs
                                conn_start = conn_pos - perp_dir * cross_connection_length / 2
                                conn_end = conn_pos + perp_dir * cross_connection_length / 2

                                conn_fiber = make_fiber(conn_start, conn_end, cross_connection_radius, params.resolution)
                                if conn_fiber.num_vert() > 0:
                                    all_fibers.append(conn_fiber)

        # Generate cross fibers if enabled
        if params.enable_cross_fibers:
            cross_angle_rad = angle_rad + np.pi / 2
            cos_c, sin_c = np.cos(cross_angle_rad), np.sin(cross_angle_rad)

            num_cross = int(num_fibers * params.cross_fiber_ratio)
            for i in range(num_cross):
                x_pos = -px/2 + (i + 0.5) * (px / num_cross)

                start_local = np.array([x_pos, -py/2, 0])
                end_local = np.array([x_pos, py/2, 0])

                start_rot = np.array([
                    start_local[0] * cos_c - start_local[1] * sin_c,
                    start_local[0] * sin_c + start_local[1] * cos_c,
                    z_center
                ])
                end_rot = np.array([
                    end_local[0] * cos_c - end_local[1] * sin_c,
                    end_local[0] * sin_c + end_local[1] * cos_c,
                    z_center
                ])

                fiber = make_fiber(start_rot, end_rot, fiber_radius_mm * 0.7, params.resolution)
                if fiber.num_vert() > 0:
                    all_fibers.append(fiber)
                    total_fiber_count += 1

    # Generate capillary channels using density-based distribution
    # Native cardiac: 2500-3500 capillaries/mm^2
    capillary_count = 0
    if params.enable_capillary_channels:
        # Calculate number of capillaries based on density
        patch_area_mm2 = px * py
        num_capillaries = int(params.capillary_density * patch_area_mm2)

        # Limit to reasonable number to prevent performance issues
        num_capillaries = min(num_capillaries, 10000)

        # Use spatial grid for O(1) neighbor lookups instead of O(n)
        spatial_grid = SpatialGrid(
            cell_size=capillary_spacing_mm,
            bounds=(-px/2, px/2, -py/2, py/2)
        )

        max_attempts_per_capillary = 20  # Reduced since grid is much faster

        for _ in range(num_capillaries):
            # Try to place capillary with minimum spacing constraint
            placed = False
            for attempt in range(max_attempts_per_capillary):
                x = np.random.uniform(-px/2, px/2)
                y = np.random.uniform(-py/2, py/2)
                candidate_pos = np.array([x, y])

                # O(1) distance check using spatial grid
                if spatial_grid.check_min_distance(candidate_pos, capillary_spacing_mm):
                    spatial_grid.insert(candidate_pos)
                    placed = True

                    # Create vertical capillary channel
                    start = np.array([x, y, 0])
                    end = np.array([x, y, pz])

                    channel = make_fiber(start, end, capillary_radius_mm, max(6, params.resolution - 2))
                    if channel.num_vert() > 0:
                        all_channels.append(channel)
                        capillary_count += 1
                    break

            # If not placed after max attempts, skip (area saturated)

    # Generate conduction channels if enabled
    # Longitudinal channels aligned with fiber direction for electrical signal propagation
    conduction_channels = []
    conduction_channel_count = 0
    if params.enable_conduction_channels:
        conduction_radius_mm = params.conduction_channel_diameter / 2000.0
        conduction_spacing_mm = params.conduction_channel_spacing / 1000.0

        # Calculate average fiber direction (use epicardial angle as reference)
        avg_angle_rad = np.radians(params.epicardial_helix_angle)
        cos_cond, sin_cond = np.cos(avg_angle_rad), np.sin(avg_angle_rad)

        # Place conduction channels parallel to main fiber direction at regular spacing
        num_cond_y = max(1, int(py / conduction_spacing_mm))
        num_cond_z = max(1, int(pz / conduction_spacing_mm))

        for j in range(num_cond_y):
            for k in range(num_cond_z):
                y_pos = -py/2 + (j + 0.5) * (py / num_cond_y)
                z_pos = (k + 0.5) * (pz / num_cond_z)

                # Channel runs along the angled fiber direction
                # Start and end points at patch boundaries
                half_diag = np.sqrt(px*px + py*py) / 2
                start_local = np.array([-half_diag, y_pos, z_pos])
                end_local = np.array([half_diag, y_pos, z_pos])

                # Rotate by fiber angle
                start_rot = np.array([
                    start_local[0] * cos_cond - (start_local[1] - y_pos) * sin_cond,
                    start_local[0] * sin_cond + (start_local[1] - y_pos) * cos_cond + y_pos,
                    start_local[2]
                ])
                end_rot = np.array([
                    end_local[0] * cos_cond - (end_local[1] - y_pos) * sin_cond,
                    end_local[0] * sin_cond + (end_local[1] - y_pos) * cos_cond + y_pos,
                    end_local[2]
                ])

                channel = make_fiber(start_rot, end_rot, conduction_radius_mm, params.resolution)
                if channel.num_vert() > 0:
                    conduction_channels.append(channel)
                    conduction_channel_count += 1

    if not all_fibers:
        raise ValueError("No fibers generated")

    result = batch_union(all_fibers)

    # Clip to exact bounding box
    bbox = m3d.Manifold.cube([px, py, pz]).translate([-px/2, -py/2, 0])
    result = result ^ bbox

    # Generate pores for porosity
    # Spherical pores distributed throughout the fiber matrix
    pore_count = 0
    all_pores = []
    if params.porosity > 0 and pore_radius_mm > 0:
        # Calculate pore count based on porosity and volume
        patch_volume_mm3 = px * py * pz
        single_pore_volume = (4/3) * np.pi * (pore_radius_mm ** 3)
        target_pore_volume = patch_volume_mm3 * params.porosity
        pore_count = int(target_pore_volume / single_pore_volume)

        # Limit to reasonable number to prevent performance issues
        pore_count = min(pore_count, 5000)

        # Generate pores using vectorized numpy operations
        x_coords = np.random.uniform(-px/2 + pore_radius_mm, px/2 - pore_radius_mm, pore_count)
        y_coords = np.random.uniform(-py/2 + pore_radius_mm, py/2 - pore_radius_mm, pore_count)
        z_coords = np.random.uniform(pore_radius_mm, pz - pore_radius_mm, pore_count)

        # Pre-create sphere template once and reuse
        sphere_template = m3d.Manifold.sphere(pore_radius_mm, max(4, params.resolution - 2))

        pore_positions = np.column_stack([x_coords, y_coords, z_coords])

        for i in range(pore_count):
            pore = sphere_template.translate([x_coords[i], y_coords[i], z_coords[i]])
            all_pores.append(pore)

        # Generate interconnecting channels using KD-tree style approach
        # Only connect nearby pores - use grid-based bucketing
        if params.enable_interconnected_pores and params.pore_interconnectivity > 0 and pore_count > 1:
            interconnect_radius = pore_radius_mm * 0.3
            max_connect_dist = pore_radius_mm * 5

            # Build spatial buckets for O(1) neighbor lookup
            bucket_size = max_connect_dist
            buckets: dict[tuple[int, int, int], list[int]] = {}

            for i in range(pore_count):
                bx = int(pore_positions[i, 0] / bucket_size)
                by = int(pore_positions[i, 1] / bucket_size)
                bz = int(pore_positions[i, 2] / bucket_size)
                key = (bx, by, bz)
                if key not in buckets:
                    buckets[key] = []
                buckets[key].append(i)

            # Connect each pore to nearest neighbor in same/adjacent buckets
            num_interconnects = int(pore_count * params.pore_interconnectivity)
            connected = set()

            for i in range(min(num_interconnects, pore_count)):
                if i in connected:
                    continue

                bx = int(pore_positions[i, 0] / bucket_size)
                by = int(pore_positions[i, 1] / bucket_size)
                bz = int(pore_positions[i, 2] / bucket_size)

                nearest_idx = -1
                min_dist_sq = max_connect_dist * max_connect_dist

                # Check only adjacent buckets (3x3x3 = 27 buckets max)
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        for dz in range(-1, 2):
                            key = (bx + dx, by + dy, bz + dz)
                            if key in buckets:
                                for j in buckets[key]:
                                    if j > i and j not in connected:
                                        diff = pore_positions[j] - pore_positions[i]
                                        dist_sq = diff[0]**2 + diff[1]**2 + diff[2]**2
                                        if dist_sq < min_dist_sq:
                                            min_dist_sq = dist_sq
                                            nearest_idx = j

                if nearest_idx >= 0:
                    p1 = pore_positions[i]
                    p2 = pore_positions[nearest_idx]
                    interconnect = make_fiber(p1, p2, interconnect_radius, max(4, params.resolution - 2))
                    if interconnect.num_vert() > 0:
                        all_pores.append(interconnect)
                    connected.add(i)
                    connected.add(nearest_idx)

        # Subtract pores from fiber structure
        if all_pores:
            pore_union = batch_union(all_pores)
            result = result - pore_union

    # Subtract capillary channels (they become hollow)
    if all_channels:
        channel_union = batch_union(all_channels)
        result = result - channel_union

    # Subtract conduction channels (hollow channels for electrical propagation)
    if conduction_channels:
        conduction_union = batch_union(conduction_channels)
        result = result - conduction_union

    # Calculate statistics
    mesh = result.to_mesh()
    volume = result.volume() if hasattr(result, 'volume') else 0

    stats = {
        'triangle_count': len(mesh.tri_verts) if hasattr(mesh, 'tri_verts') else 0,
        'volume_mm3': volume,
        'fiber_count': total_fiber_count,
        'capillary_count': capillary_count,
        'pore_count': pore_count,
        'conduction_channel_count': conduction_channel_count,
        'layer_count': params.layer_count,
        'scaffold_type': 'cardiac_patch'
    }

    return result, stats


def generate_cardiac_patch_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate cardiac patch from dictionary parameters (for API compatibility).

    Args:
        params: Dictionary with keys matching CardiacPatchParams fields

    Returns:
        Tuple of (manifold, stats_dict)
    """
    # Handle legacy patch_size parameter
    if 'patch_size' in params:
        patch_size = params['patch_size']
        if isinstance(patch_size, dict):
            patch_length = patch_size.get('x', 10.0)
            patch_width = patch_size.get('y', 10.0)
            patch_thickness = patch_size.get('z', 2.0)
        elif isinstance(patch_size, (list, tuple)):
            patch_length = patch_size[0] if len(patch_size) > 0 else 10.0
            patch_width = patch_size[1] if len(patch_size) > 1 else 10.0
            patch_thickness = patch_size[2] if len(patch_size) > 2 else 2.0
        else:
            patch_length = params.get('patch_length', 10.0)
            patch_width = params.get('patch_width', 10.0)
            patch_thickness = params.get('patch_thickness', 2.0)
    else:
        patch_length = params.get('patch_length', 10.0)
        patch_width = params.get('patch_width', 10.0)
        patch_thickness = params.get('patch_thickness', 2.0)

    # Map legacy parameters
    fiber_spacing = params.get('fiber_spacing', 150.0)
    fiber_diameter = params.get('fiber_diameter', 80.0)
    alignment_angle = params.get('alignment_angle', params.get('epicardial_helix_angle', -60.0))

    return generate_cardiac_patch(CardiacPatchParams(
        # Cardiomyocyte dimensions
        cardiomyocyte_length=params.get('cardiomyocyte_length', 100.0),
        cardiomyocyte_diameter=params.get('cardiomyocyte_diameter', 25.0),
        sarcomere_length=params.get('sarcomere_length', 1.8),

        # Fiber architecture
        fiber_diameter=fiber_diameter,
        fiber_spacing=fiber_spacing,
        fiber_alignment_degree=params.get('fiber_alignment_degree', 0.9),

        # Layer configuration
        epicardial_helix_angle=alignment_angle,
        endocardial_helix_angle=params.get('endocardial_helix_angle', 60.0),
        layer_count=params.get('layer_count', 3),
        enable_helix_gradient=params.get('enable_helix_gradient', True),

        # Patch dimensions
        patch_length=patch_length,
        patch_width=patch_width,
        patch_thickness=patch_thickness,

        # Porosity
        porosity=params.get('porosity', 0.75),
        pore_size=params.get('pore_size', 50.0),
        enable_interconnected_pores=params.get('enable_interconnected_pores', True),
        pore_interconnectivity=params.get('pore_interconnectivity', 0.85),

        # Vascularization
        enable_capillary_channels=params.get('enable_capillary_channels', True),
        capillary_diameter=params.get('capillary_diameter', 8.0),
        capillary_density=params.get('capillary_density', 3000.0),
        capillary_spacing=params.get('capillary_spacing', 20.0),

        # Mechanical features
        enable_cross_fibers=params.get('enable_cross_fibers', True),
        cross_fiber_ratio=params.get('cross_fiber_ratio', 0.3),
        enable_fiber_crimping=params.get('enable_fiber_crimping', False),
        crimp_amplitude=params.get('crimp_amplitude', 5.0),
        crimp_wavelength=params.get('crimp_wavelength', 50.0),

        # Electrical guidance
        enable_conduction_channels=params.get('enable_conduction_channels', False),
        conduction_channel_diameter=params.get('conduction_channel_diameter', 100.0),
        conduction_channel_spacing=params.get('conduction_channel_spacing', 500.0),

        # Stochastic variation
        position_noise=params.get('position_noise', 0.0),
        diameter_variance=params.get('diameter_variance', 0.0),
        angle_noise=params.get('angle_noise', 0.0),
        seed=params.get('seed', 42),

        # Resolution
        resolution=params.get('resolution', 8)
    ))
