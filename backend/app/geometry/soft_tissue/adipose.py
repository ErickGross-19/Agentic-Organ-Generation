"""
Adipose tissue scaffold generator with honeycomb structure.

Provides parametric generation of adipose (fat) tissue scaffolds with:
- Hexagonal honeycomb pattern for adipocyte housing
- Configurable cell size (100-200um)
- Thin walls between cells
- Vertical vascular channels for perfusion
- Cylindrical overall shape
- Lobule organization with Voronoi-based clustering
- Hierarchical vascular network (capillaries, arterioles, venules)
- SVF channels for stem cell migration
- ECM fiber networks
- Surface texture for organic appearance
"""

from __future__ import annotations
import manifold3d as m3d
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from ..core import batch_union


@dataclass
class AdiposeTissueParams:
    """
    Parameters for adipose tissue scaffold generation.

    Biologically realistic parameters based on human adipose tissue histology:
    - Adipocytes: 50-200 um diameter (typically ~90 um in lean adults)
    - Organized in lobules separated by connective tissue septa
    - Highly vascularized with capillary network
    - Contains stromal vascular fraction (SVF) with stem cells
    """
    # === Basic Geometry ===
    diameter_mm: float = 15.0  # Overall scaffold diameter
    height_mm: float = 10.0    # Scaffold height

    # === Adipocyte Parameters ===
    adipocyte_diameter_um: float = 90.0  # 50-200 um (avg ~90 in lean, ~120 in obese)
    adipocyte_size_variance: float = 0.15  # Size variability
    cell_density_per_mL: float = 40e6  # 20-60 million cells/mL

    # === Lobule Structure ===
    lobule_size_mm: float = 2.0  # 1-3 mm typical lobule diameter
    enable_lobules: bool = True
    lobules_per_cm2: float = 4.0  # ~4-8 lobules per cm2

    # === Septum/Connective Tissue ===
    septum_thickness_um: float = 50.0  # 30-100 um interlobular septa
    septum_porosity: float = 0.3  # Loose connective tissue

    # === Vascular Network ===
    enable_vascular_channels: bool = True
    capillary_diameter_um: float = 7.0  # 5-10 um typical
    capillary_density_per_mm2: float = 400  # High vascular density
    vascular_channel_diameter_mm: float = 0.4  # Larger perfusion channels
    vascular_channel_spacing_mm: float = 2.0

    # === Arteriole/Venule ===
    arteriole_diameter_um: float = 30.0  # 20-50 um
    venule_diameter_um: float = 40.0  # 30-60 um

    # === Stromal Vascular Fraction ===
    enable_svf_channels: bool = False  # Space for SVF cells
    svf_channel_diameter_um: float = 50.0

    # === Pore Architecture ===
    porosity: float = 0.8  # High porosity for cell infiltration
    pore_size_um: float = 200.0  # 150-300 um for adipogenic differentiation
    pore_interconnectivity: float = 0.9  # 0-1 scale

    # === ECM Features ===
    enable_ecm_fibers: bool = False
    collagen_fiber_density: float = 0.3  # Lower than other tissues
    elastin_fiber_density: float = 0.1

    # === Basement Membrane ===
    enable_basement_membrane: bool = False
    basement_membrane_thickness_um: float = 0.1  # ~100 nm around adipocytes

    # === Surface Features ===
    enable_surface_texture: bool = False
    surface_roughness: float = 0.2

    # === Mechanical Properties (for reference) ===
    target_stiffness_kPa: float = 2.0  # 1-4 kPa for adipose tissue. REFERENCE ONLY: Mechanical property for FEA simulation coupling, no geometric representation

    # === Randomization ===
    seed: int = 42
    position_noise: float = 0.2  # Positional variation for organic look

    # === Resolution ===
    resolution: int = 8


def make_hexagonal_prism(
    side_length: float,
    height: float,
    center_x: float,
    center_y: float,
    resolution: int = 6
) -> m3d.Manifold:
    """
    Create a hexagonal prism (cell chamber).

    Args:
        side_length: Length of hexagon side
        height: Height of prism
        center_x: X position of center
        center_y: Y position of center
        resolution: Must be 6 for hexagon

    Returns:
        Manifold representing the hexagonal prism
    """
    # Create hexagon by using cylinder with 6 segments
    # For a hexagon inscribed in a circle of radius R:
    # side_length = R
    radius = side_length
    hexagon = m3d.Manifold.cylinder(height, radius, radius, 6)

    return hexagon.translate([center_x, center_y, height / 2])


def get_hexagonal_grid_positions(
    diameter: float,
    cell_size: float
) -> list[tuple[float, float]]:
    """
    Get positions for hexagonal grid pattern.

    Uses offset grid to create honeycomb pattern.

    Args:
        diameter: Overall diameter of scaffold
        cell_size: Size of each hexagonal cell

    Returns:
        List of (x, y) positions for cell centers
    """
    positions = []
    radius = diameter / 2

    # Hexagonal grid spacing
    # For hexagons with side length s:
    # Horizontal spacing: 1.5 * s
    # Vertical spacing: sqrt(3) * s
    h_spacing = 1.5 * cell_size
    v_spacing = np.sqrt(3) * cell_size

    # Calculate grid bounds
    n_cols = max(1, int(diameter / h_spacing))
    n_rows = max(1, int(diameter / v_spacing))

    for row in range(n_rows):
        for col in range(n_cols):
            # Offset every other row for honeycomb pattern
            x_offset = (cell_size * 0.75) if row % 2 == 1 else 0
            x = (col - n_cols / 2 + 0.5) * h_spacing + x_offset
            y = (row - n_rows / 2 + 0.5) * v_spacing

            # Check if within circular boundary
            if np.sqrt(x*x + y*y) < radius - cell_size:
                positions.append((x, y))

    return positions


def make_vascular_channel(
    height: float,
    channel_diameter: float,
    x_pos: float,
    y_pos: float,
    resolution: int
) -> m3d.Manifold:
    """
    Create a vertical vascular channel.

    Args:
        height: Height of scaffold
        channel_diameter: Diameter of channel
        x_pos: X position
        y_pos: Y position
        resolution: Circular resolution

    Returns:
        Manifold representing the channel
    """
    channel_radius = channel_diameter / 2
    channel = m3d.Manifold.cylinder(
        height * 1.1,
        channel_radius,
        channel_radius,
        resolution
    )
    return channel.translate([x_pos, y_pos, height / 2])


def generate_lobule_centers(
    diameter_mm: float,
    lobule_size_mm: float,
    lobules_per_cm2: float,
    rng: np.random.Generator,
    position_noise: float = 0.2
) -> List[Tuple[float, float]]:
    """
    Generate lobule center positions using a grid-based approach with jitter.

    Args:
        diameter_mm: Scaffold diameter in mm
        lobule_size_mm: Target lobule diameter in mm
        lobules_per_cm2: Target lobule density per cm2
        rng: Random number generator
        position_noise: Amount of position noise (0-1)

    Returns:
        List of (x, y) positions for lobule centers
    """
    radius = diameter_mm / 2

    # Calculate spacing from lobule density
    # lobules_per_cm2 = 4 means spacing ~ sqrt(100 / 4) = 5mm per lobule
    area_cm2 = np.pi * (diameter_mm / 10) ** 2  # Convert to cm
    target_lobules = int(area_cm2 * lobules_per_cm2)

    # Use lobule_size_mm as approximate spacing
    spacing = lobule_size_mm

    centers = []
    n_grid = max(1, int(diameter_mm / spacing) + 1)

    for i in range(n_grid):
        for j in range(n_grid):
            # Grid position centered on scaffold
            x = (i - n_grid / 2 + 0.5) * spacing
            y = (j - n_grid / 2 + 0.5) * spacing

            # Add jitter for organic look
            if position_noise > 0:
                jitter = spacing * 0.3 * position_noise
                x += rng.uniform(-jitter, jitter)
                y += rng.uniform(-jitter, jitter)

            # Check if within circular boundary (with margin)
            dist = np.sqrt(x*x + y*y)
            if dist < radius - spacing * 0.3:
                centers.append((x, y))

    return centers


def assign_cells_to_lobules(
    cell_positions: List[Tuple[float, float]],
    lobule_centers: List[Tuple[float, float]]
) -> List[int]:
    """
    Assign each cell to its nearest lobule (Voronoi-like assignment).

    Args:
        cell_positions: List of (x, y) cell positions
        lobule_centers: List of (x, y) lobule center positions

    Returns:
        List of lobule indices for each cell
    """
    if not lobule_centers:
        return [0] * len(cell_positions)

    assignments = []
    lobule_array = np.array(lobule_centers)

    for cx, cy in cell_positions:
        # Calculate distances to all lobule centers
        distances = np.sqrt((lobule_array[:, 0] - cx)**2 + (lobule_array[:, 1] - cy)**2)
        nearest_lobule = int(np.argmin(distances))
        assignments.append(nearest_lobule)

    return assignments


def find_voronoi_edges(
    lobule_centers: List[Tuple[float, float]],
    diameter_mm: float
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Find edges between neighboring lobule regions (approximated).

    Uses a simple approach: for each pair of nearby lobule centers,
    compute the midpoint and perpendicular direction for the edge.

    Args:
        lobule_centers: List of (x, y) lobule centers
        diameter_mm: Scaffold diameter

    Returns:
        List of edge segments as ((x1,y1), (x2,y2)) tuples
    """
    if len(lobule_centers) < 2:
        return []

    edges = []
    radius = diameter_mm / 2
    centers = np.array(lobule_centers)

    # Find edges between neighboring lobules
    for i, (cx1, cy1) in enumerate(lobule_centers):
        for j, (cx2, cy2) in enumerate(lobule_centers):
            if j <= i:
                continue

            # Distance between centers
            dist = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)

            # Only consider nearby lobules (within 1.5x typical spacing)
            avg_spacing = diameter_mm / np.sqrt(len(lobule_centers))
            if dist < avg_spacing * 1.8:
                # Midpoint
                mx, my = (cx1 + cx2) / 2, (cy1 + cy2) / 2

                # Perpendicular direction (edge runs perpendicular to line between centers)
                dx, dy = cx2 - cx1, cy2 - cy1
                length = np.sqrt(dx*dx + dy*dy)
                if length > 0.01:
                    # Perpendicular unit vector
                    px, py = -dy / length, dx / length

                    # Edge length (roughly half the distance between centers)
                    edge_len = dist * 0.6

                    # Edge endpoints
                    x1 = mx + px * edge_len / 2
                    y1 = my + py * edge_len / 2
                    x2 = mx - px * edge_len / 2
                    y2 = my - py * edge_len / 2

                    # Clip to scaffold boundary
                    for pt in [(x1, y1), (x2, y2)]:
                        d = np.sqrt(pt[0]**2 + pt[1]**2)
                        if d < radius:
                            edges.append(((x1, y1), (x2, y2)))
                            break

    return edges


def create_septum_walls(
    lobule_edges: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    height_mm: float,
    septum_thickness_um: float,
    septum_porosity: float,
    resolution: int,
    rng: np.random.Generator
) -> List[m3d.Manifold]:
    """
    Create septum walls between lobules.

    Args:
        lobule_edges: List of edge segments
        height_mm: Scaffold height
        septum_thickness_um: Septum thickness in um
        septum_porosity: Porosity of septa (0-1)
        resolution: Mesh resolution
        rng: Random number generator

    Returns:
        List of manifolds representing septum walls
    """
    if not lobule_edges:
        return []

    thickness_mm = septum_thickness_um / 1000.0
    septa = []

    for (x1, y1), (x2, y2) in lobule_edges:
        # Length of the septum wall
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length < 0.1:
            continue

        # Create a box for the wall
        # Use a thin cylinder oriented along the edge
        wall = m3d.Manifold.cylinder(
            length,
            thickness_mm,
            thickness_mm,
            max(4, resolution // 2)
        )

        # Rotate to align with edge direction
        angle = np.arctan2(y2 - y1, x2 - x1)
        wall = wall.rotate([90, 0, np.degrees(angle)])

        # Scale to make it a flat wall instead of cylinder
        wall = wall.scale([1, 0.3, height_mm / length])

        # Position at edge midpoint
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        wall = wall.translate([mx, my, height_mm / 2])

        # Add porosity by creating holes if septum_porosity > 0
        if septum_porosity > 0.1:
            # Simplified: just add to septa (full porosity implementation would add holes)
            septa.append(wall)
        else:
            septa.append(wall)

    return septa


def create_vascular_hierarchy(
    lobule_centers: List[Tuple[float, float]],
    lobule_edges: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    height_mm: float,
    params: AdiposeTissueParams,
    rng: np.random.Generator
) -> List[m3d.Manifold]:
    """
    Create hierarchical vascular network with arterioles, venules, and capillaries.

    Uses Murray's law for branching: r_child = r_parent * 0.79

    Args:
        lobule_centers: Lobule center positions
        lobule_edges: Edges between lobules (septum locations)
        height_mm: Scaffold height
        params: Adipose tissue parameters
        rng: Random number generator

    Returns:
        List of manifolds representing vascular channels
    """
    vessels = []

    # Convert um to mm
    arteriole_r_mm = params.arteriole_diameter_um / 2000.0
    venule_r_mm = params.venule_diameter_um / 2000.0
    capillary_r_mm = params.capillary_diameter_um / 2000.0

    # Murray's law ratio
    murray_ratio = 0.79

    # Calculate area covered by each main vessel
    area_per_vessel = np.pi * (params.lobule_size_mm / 2) ** 2

    # Create main arteriole/venule channels along septa
    for i, ((x1, y1), (x2, y2)) in enumerate(lobule_edges):
        if i >= len(lobule_edges) // 2:
            # Alternate between arterioles and venules
            vessel_r = venule_r_mm
        else:
            vessel_r = arteriole_r_mm

        # Main vessel along septum (vertical channel)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        main_vessel = m3d.Manifold.cylinder(
            height_mm * 1.1,
            vessel_r,
            vessel_r,
            params.resolution
        )
        main_vessel = main_vessel.translate([mx, my, height_mm / 2])
        vessels.append(main_vessel)

        # Branch capillaries from the main vessel
        # Calculate number of branches based on capillary density
        target_capillaries = int(params.capillary_density_per_mm2 * area_per_vessel)
        n_branches = max(2, min(target_capillaries // max(1, len(lobule_edges)), 8))
        for branch in range(n_branches):
            # Vertical position of branch
            z_branch = (branch + 1) * height_mm / (n_branches + 1)

            # Branch direction (radial)
            angle = rng.uniform(0, 2 * np.pi)
            branch_length = params.lobule_size_mm * 0.3

            # Capillary endpoint
            ex = mx + branch_length * np.cos(angle)
            ey = my + branch_length * np.sin(angle)

            # First level branch (child of main vessel)
            branch1_r = vessel_r * murray_ratio
            cap1 = make_branching_vessel(
                mx, my, z_branch,
                ex, ey, z_branch,
                branch1_r,
                params.resolution // 2
            )
            if cap1 is not None:
                vessels.append(cap1)

            # Second level branch (capillary)
            cap_r = branch1_r * murray_ratio
            if cap_r >= capillary_r_mm:
                ex2 = ex + branch_length * 0.5 * np.cos(angle + rng.uniform(-0.5, 0.5))
                ey2 = ey + branch_length * 0.5 * np.sin(angle + rng.uniform(-0.5, 0.5))
                cap2 = make_branching_vessel(
                    ex, ey, z_branch,
                    ex2, ey2, z_branch,
                    max(cap_r, capillary_r_mm),
                    max(4, params.resolution // 4)
                )
                if cap2 is not None:
                    vessels.append(cap2)

    return vessels


def make_branching_vessel(
    x1: float, y1: float, z1: float,
    x2: float, y2: float, z2: float,
    radius: float,
    resolution: int
) -> Optional[m3d.Manifold]:
    """
    Create a vessel segment between two points.

    Args:
        x1, y1, z1: Start point
        x2, y2, z2: End point
        radius: Vessel radius
        resolution: Mesh resolution

    Returns:
        Manifold representing the vessel, or None if too short
    """
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    length = np.sqrt(dx*dx + dy*dy + dz*dz)

    if length < 0.01 or radius < 0.001:
        return None

    # Create cylinder along z-axis
    cyl = m3d.Manifold.cylinder(length, radius, radius, max(4, resolution))

    # Calculate rotation angles
    h = np.sqrt(dx*dx + dy*dy)
    if h > 0.001 or abs(dz) > 0.001:
        # Rotate to align with direction vector
        cyl = cyl.rotate([0, np.arctan2(h, dz) * 180 / np.pi, 0])
        if h > 0.001:
            cyl = cyl.rotate([0, 0, np.arctan2(dy, dx) * 180 / np.pi])

    return cyl.translate([x1, y1, z1])


def create_svf_channels(
    lobule_edges: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    height_mm: float,
    svf_channel_diameter_um: float,
    resolution: int,
    rng: np.random.Generator
) -> List[m3d.Manifold]:
    """
    Create SVF (stromal vascular fraction) channels through septa.

    These channels provide pathways for stem cell migration between lobules.

    Args:
        lobule_edges: Edges between lobules
        height_mm: Scaffold height
        svf_channel_diameter_um: Channel diameter in um
        resolution: Mesh resolution
        rng: Random number generator

    Returns:
        List of manifolds representing SVF channels
    """
    channels = []
    channel_r_mm = svf_channel_diameter_um / 2000.0

    for (x1, y1), (x2, y2) in lobule_edges:
        # Create horizontal channel along the septum
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length < 0.1:
            continue

        # Multiple channels at different heights
        n_channels = max(1, int(height_mm / 2))
        for i in range(n_channels):
            z = (i + 0.5) * height_mm / n_channels

            # Channel along the edge
            channel = make_branching_vessel(
                x1, y1, z,
                x2, y2, z,
                channel_r_mm,
                max(4, resolution // 2)
            )
            if channel is not None:
                channels.append(channel)

            # Add junction sphere at midpoint
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            junction = m3d.Manifold.sphere(channel_r_mm * 1.2, max(4, resolution // 2))
            junction = junction.translate([mx, my, z])
            channels.append(junction)

    return channels


def create_ecm_fibers(
    cell_positions: List[Tuple[float, float]],
    lobule_edges: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    height_mm: float,
    cell_size_mm: float,
    collagen_density: float,
    elastin_density: float,
    resolution: int,
    rng: np.random.Generator
) -> List[m3d.Manifold]:
    """
    Create ECM fiber network around adipocytes.

    Creates collagen and elastin fibers with:
    - Radial orientation around each cell
    - Higher density in septa regions

    Args:
        cell_positions: Adipocyte positions
        lobule_edges: Septum edge positions
        height_mm: Scaffold height
        cell_size_mm: Adipocyte diameter in mm
        collagen_density: Collagen fiber density (0-1)
        elastin_density: Elastin fiber density (0-1)
        resolution: Mesh resolution
        rng: Random number generator

    Returns:
        List of manifolds representing ECM fibers
    """
    fibers = []

    # Fiber thickness (collagen ~1-10um, elastin ~1-3um)
    collagen_r_mm = 0.005  # 5um radius
    elastin_r_mm = 0.002   # 2um radius

    # Sample a subset of cells for fiber generation (for performance)
    n_cells_to_process = min(len(cell_positions), int(len(cell_positions) * (collagen_density + elastin_density)))
    selected_indices = rng.choice(len(cell_positions), size=n_cells_to_process, replace=False)

    for idx in selected_indices:
        cx, cy = cell_positions[idx]

        # Generate radial fibers around the cell
        n_fibers = rng.integers(4, 8)
        for f in range(n_fibers):
            angle = f * 2 * np.pi / n_fibers + rng.uniform(-0.3, 0.3)

            # Fiber starts at cell surface, extends outward
            start_r = cell_size_mm / 2
            end_r = cell_size_mm * 0.8

            fx1 = cx + start_r * np.cos(angle)
            fy1 = cy + start_r * np.sin(angle)
            fx2 = cx + end_r * np.cos(angle)
            fy2 = cy + end_r * np.sin(angle)

            # Random z position
            z = rng.uniform(cell_size_mm, height_mm - cell_size_mm)

            # Decide collagen vs elastin based on densities
            if rng.random() < collagen_density / (collagen_density + elastin_density):
                fiber_r = collagen_r_mm
            else:
                fiber_r = elastin_r_mm

            fiber = make_branching_vessel(
                fx1, fy1, z,
                fx2, fy2, z,
                fiber_r,
                4
            )
            if fiber is not None:
                fibers.append(fiber)

    return fibers


def create_basement_membrane(
    cell_positions: List[Tuple[float, float]],
    cell_sizes: List[float],
    height_mm: float,
    membrane_thickness_um: float,
    resolution: int
) -> List[m3d.Manifold]:
    """
    Create thin basement membrane shells around adipocytes.

    Since real basement membrane is ~100nm thick (too thin for geometry),
    we create a visual representation with a minimum practical thickness.

    Args:
        cell_positions: Adipocyte (x, y) positions
        cell_sizes: Diameter of each cell in mm
        height_mm: Scaffold height
        membrane_thickness_um: Target membrane thickness (visual representation)
        resolution: Mesh resolution

    Returns:
        List of manifolds representing basement membrane shells
    """
    membranes = []

    # Use a minimum practical thickness for visualization
    # Real membrane is 0.1um, we use 1-2um for visibility
    visual_thickness_mm = max(membrane_thickness_um, 2.0) / 1000.0

    for (cx, cy), size in zip(cell_positions, cell_sizes):
        outer_r = size / 2
        inner_r = outer_r - visual_thickness_mm

        if inner_r <= 0:
            continue

        # Create spherical shell at center of scaffold height
        z_center = height_mm / 2

        outer = m3d.Manifold.sphere(outer_r, resolution)
        inner = m3d.Manifold.sphere(inner_r, resolution)
        shell = outer - inner

        shell = shell.translate([cx, cy, z_center])
        membranes.append(shell)

    return membranes


def apply_surface_texture(
    outer_cylinder: m3d.Manifold,
    lobule_centers: List[Tuple[float, float]],
    diameter_mm: float,
    height_mm: float,
    surface_roughness: float,
    resolution: int,
    rng: np.random.Generator
) -> m3d.Manifold:
    """
    Apply lobular surface texture to the outer surface.

    Creates bumpy appearance by adding/subtracting small features.

    Args:
        outer_cylinder: Base outer cylinder
        lobule_centers: Lobule center positions (for bump placement)
        diameter_mm: Scaffold diameter
        height_mm: Scaffold height
        surface_roughness: Roughness scale (0-1)
        resolution: Mesh resolution
        rng: Random number generator

    Returns:
        Manifold with textured surface
    """
    if surface_roughness < 0.01 or not lobule_centers:
        return outer_cylinder

    radius = diameter_mm / 2
    bumps = []

    # Create bumps at lobule centers projected onto surface
    for lx, ly in lobule_centers:
        # Project lobule center to surface
        dist = np.sqrt(lx*lx + ly*ly)
        if dist < 0.01:
            continue

        # Normalized direction
        nx, ny = lx / dist, ly / dist

        # Surface point
        sx, sy = nx * radius, ny * radius

        # Bump size based on roughness
        bump_size = surface_roughness * 0.5  # 0-0.5mm bump
        bump_height = surface_roughness * 0.3

        # Create outward bump at multiple heights
        n_bumps = max(1, int(height_mm / 2))
        for i in range(n_bumps):
            z = (i + 0.5) * height_mm / n_bumps

            bump = m3d.Manifold.sphere(bump_size, max(4, resolution // 2))
            # Move bump slightly outside surface
            bump = bump.translate([sx + nx * bump_height, sy + ny * bump_height, z])
            bumps.append(bump)

    if bumps:
        # Add bumps to surface
        bumps_union = batch_union(bumps)
        return outer_cylinder + bumps_union

    return outer_cylinder


def calculate_porosity_adjustment(
    current_volume: float,
    total_volume: float,
    target_porosity: float,
    pore_size_um: float,
    pore_interconnectivity: float
) -> Tuple[float, int]:
    """
    Calculate adjustments needed to achieve target porosity.

    Args:
        current_volume: Current solid volume
        total_volume: Total scaffold volume
        target_porosity: Desired porosity (0-1)
        pore_size_um: Target pore size
        pore_interconnectivity: Pore connectivity (0-1)

    Returns:
        Tuple of (additional_pore_volume_needed, estimated_pores_to_add)
    """
    current_porosity = 1 - (current_volume / total_volume) if total_volume > 0 else 0

    if current_porosity >= target_porosity:
        return 0.0, 0

    # Volume of solid that needs to be removed
    target_solid_volume = total_volume * (1 - target_porosity)
    volume_to_remove = current_volume - target_solid_volume

    if volume_to_remove <= 0:
        return 0.0, 0

    # Calculate how many pores needed
    pore_size_mm = pore_size_um / 1000.0
    pore_radius_mm = pore_size_mm / 2
    single_pore_volume = (4/3) * np.pi * pore_radius_mm**3

    # Adjust for interconnectivity (overlapping pores reduce effective volume)
    effective_pore_volume = single_pore_volume * (1 - pore_interconnectivity * 0.3)

    n_pores = int(volume_to_remove / effective_pore_volume) if effective_pore_volume > 0 else 0

    return volume_to_remove, n_pores


def create_additional_pores(
    diameter_mm: float,
    height_mm: float,
    pore_size_um: float,
    n_pores: int,
    resolution: int,
    rng: np.random.Generator
) -> List[m3d.Manifold]:
    """
    Create additional spherical pores to achieve target porosity.

    Args:
        diameter_mm: Scaffold diameter
        height_mm: Scaffold height
        pore_size_um: Pore diameter in um
        n_pores: Number of pores to create
        resolution: Mesh resolution
        rng: Random number generator

    Returns:
        List of pore manifolds
    """
    if n_pores <= 0:
        return []

    pores = []
    radius = diameter_mm / 2 * 0.9  # Keep pores within boundary
    pore_r_mm = pore_size_um / 2000.0

    for _ in range(min(n_pores, 500)):  # Cap at 500 for performance
        # Random position within scaffold
        r = radius * np.sqrt(rng.random())
        theta = rng.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = rng.uniform(pore_r_mm, height_mm - pore_r_mm)

        pore = m3d.Manifold.sphere(pore_r_mm, max(4, resolution // 2))
        pore = pore.translate([x, y, z])
        pores.append(pore)

    return pores


def generate_adipose_tissue(params: AdiposeTissueParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate an adipose tissue scaffold with biologically realistic parameters.

    Args:
        params: AdiposeTissueParams specifying geometry and structure

    Returns:
        Tuple of (manifold, stats_dict)
        - manifold: The 3D geometry
        - stats_dict: Dictionary with triangle_count, volume_mm3, cell_count,
                     cell_size_um, porosity, scaffold_type

    Raises:
        ValueError: If parameters are invalid
    """
    # Set random seed for reproducibility
    rng = np.random.default_rng(params.seed)
    np.random.seed(params.seed)

    if params.adipocyte_diameter_um <= 0 or params.septum_thickness_um <= 0:
        raise ValueError("Adipocyte diameter and septum thickness must be positive")
    if params.septum_thickness_um >= params.adipocyte_diameter_um:
        raise ValueError("Septum thickness must be less than adipocyte diameter")

    # Convert micrometers to millimeters
    cell_size_mm = params.adipocyte_diameter_um / 1000.0
    wall_thickness_mm = params.septum_thickness_um / 1000.0

    # Create outer cylinder
    outer_radius = params.diameter_mm / 2
    outer_cylinder = m3d.Manifold.cylinder(
        params.height_mm,
        outer_radius,
        outer_radius,
        params.resolution * 4  # Higher resolution for outer wall
    )
    outer_cylinder = outer_cylinder.translate([0, 0, params.height_mm / 2])

    # Generate lobule centers if lobules enabled
    lobule_centers = []
    lobule_edges = []
    if params.enable_lobules:
        lobule_centers = generate_lobule_centers(
            params.diameter_mm,
            params.lobule_size_mm,
            params.lobules_per_cm2,
            rng,
            params.position_noise
        )
        lobule_edges = find_voronoi_edges(lobule_centers, params.diameter_mm)

    # Get hexagonal cell positions
    cell_positions = get_hexagonal_grid_positions(
        params.diameter_mm - wall_thickness_mm * 2,  # Leave room for outer wall
        cell_size_mm
    )

    # Apply cell density constraint by subsampling if needed
    # Calculate scaffold volume in mL (1 mL = 1000 mm³)
    scaffold_volume_mm3 = np.pi * (params.diameter_mm/2)**2 * params.height_mm
    scaffold_volume_mL = scaffold_volume_mm3 / 1000.0

    # Target cell count from density
    target_cells = int(params.cell_density_per_mL * scaffold_volume_mL)

    # If we have more cells than target, subsample
    if len(cell_positions) > target_cells and target_cells > 0:
        indices = rng.choice(len(cell_positions), size=target_cells, replace=False)
        cell_positions = [cell_positions[i] for i in sorted(indices)]

    # Assign cells to lobules
    cell_lobule_assignments = []
    if params.enable_lobules and lobule_centers:
        cell_lobule_assignments = assign_cells_to_lobules(cell_positions, lobule_centers)

    # Create hexagonal cell chambers with optional position noise
    cells = []
    cell_sizes = []
    for i, (x, y) in enumerate(cell_positions):
        # Add position noise for organic look
        if params.position_noise > 0:
            x += np.random.uniform(-1, 1) * params.position_noise * cell_size_mm
            y += np.random.uniform(-1, 1) * params.position_noise * cell_size_mm

        # Add size variance
        actual_size = cell_size_mm
        if params.adipocyte_size_variance > 0:
            actual_size *= (1 + np.random.uniform(-1, 1) * params.adipocyte_size_variance)

        cell_sizes.append(actual_size)
        cell = make_hexagonal_prism(
            actual_size / 2,  # Hexagon radius
            params.height_mm,
            x, y,
            6  # Hexagon has 6 sides
        )
        cells.append(cell)

    if not cells:
        raise ValueError("No cells generated")

    # Union all cells
    cells_union = batch_union(cells)

    # Create honeycomb by subtracting cells from outer cylinder
    # This leaves the walls between cells
    honeycomb = outer_cylinder - cells_union

    # Create septum walls between lobules if enabled
    septa_manifolds = []
    if params.enable_lobules and lobule_edges:
        septa_manifolds = create_septum_walls(
            lobule_edges,
            params.height_mm,
            params.septum_thickness_um,
            params.septum_porosity,
            params.resolution,
            rng
        )
        if septa_manifolds:
            septa_union = batch_union(septa_manifolds)
            honeycomb = honeycomb + septa_union

    # Add vascular channels if enabled
    channels = []
    vascular_hierarchy = []
    if params.enable_vascular_channels:
        # Original simple vascular channels
        if params.vascular_channel_spacing_mm > 0 and params.vascular_channel_diameter_mm > 0:
            n_channels = max(1, int(params.diameter_mm / params.vascular_channel_spacing_mm))

            for i in range(n_channels):
                for j in range(n_channels):
                    x = (i - n_channels / 2 + 0.5) * params.vascular_channel_spacing_mm
                    y = (j - n_channels / 2 + 0.5) * params.vascular_channel_spacing_mm

                    # Add position noise
                    if params.position_noise > 0:
                        x += np.random.uniform(-0.5, 0.5) * params.position_noise * params.vascular_channel_spacing_mm
                        y += np.random.uniform(-0.5, 0.5) * params.position_noise * params.vascular_channel_spacing_mm

                    # Check if channel is within boundary
                    if np.sqrt(x*x + y*y) < outer_radius - params.vascular_channel_diameter_mm:
                        channel = make_vascular_channel(
                            params.height_mm,
                            params.vascular_channel_diameter_mm,
                            x, y,
                            params.resolution * 2
                        )
                        channels.append(channel)

        # Create hierarchical vascular network
        if lobule_centers and lobule_edges:
            vascular_hierarchy = create_vascular_hierarchy(
                lobule_centers,
                lobule_edges,
                params.height_mm,
                params,
                rng
            )
            channels.extend(vascular_hierarchy)

    if channels:
        channels_union = batch_union(channels)
        result = honeycomb - channels_union
    else:
        result = honeycomb

    # Create SVF channels if enabled
    svf_channels = []
    if params.enable_svf_channels and lobule_edges:
        svf_channels = create_svf_channels(
            lobule_edges,
            params.height_mm,
            params.svf_channel_diameter_um,
            params.resolution,
            rng
        )
        if svf_channels:
            svf_union = batch_union(svf_channels)
            result = result - svf_union

    # Create ECM fibers if enabled
    ecm_fibers = []
    if params.enable_ecm_fibers and cell_positions:
        ecm_fibers = create_ecm_fibers(
            cell_positions,
            lobule_edges,
            params.height_mm,
            cell_size_mm,
            params.collagen_fiber_density,
            params.elastin_fiber_density,
            params.resolution,
            rng
        )
        if ecm_fibers:
            ecm_union = batch_union(ecm_fibers)
            result = result + ecm_union

    # Create basement membrane if enabled
    basement_membranes = []
    if params.enable_basement_membrane:
        basement_membranes = create_basement_membrane(
            cell_positions,
            cell_sizes,
            params.height_mm,
            params.basement_membrane_thickness_um,
            params.resolution
        )
        if basement_membranes:
            membrane_union = batch_union(basement_membranes)
            result = result + membrane_union

    # Apply surface texture if enabled
    if params.enable_surface_texture and lobule_centers:
        # Get current outer shape for texturing
        result = apply_surface_texture(
            result,
            lobule_centers,
            params.diameter_mm,
            params.height_mm,
            params.surface_roughness,
            params.resolution,
            rng
        )

    # Check and adjust porosity if needed
    mesh = result.to_mesh()
    current_volume = result.volume() if hasattr(result, 'volume') else 0
    solid_volume = np.pi * outer_radius * outer_radius * params.height_mm
    actual_porosity = 1 - (current_volume / solid_volume) if solid_volume > 0 else 0

    # If porosity is below target, add more pores
    if actual_porosity < params.porosity * 0.9:  # Within 90% of target
        volume_to_remove, n_pores = calculate_porosity_adjustment(
            current_volume,
            solid_volume,
            params.porosity,
            params.pore_size_um,
            params.pore_interconnectivity
        )
        if n_pores > 0:
            additional_pores = create_additional_pores(
                params.diameter_mm,
                params.height_mm,
                params.pore_size_um,
                n_pores,
                params.resolution,
                rng
            )
            if additional_pores:
                pores_union = batch_union(additional_pores)
                result = result - pores_union

    # Recalculate final statistics
    mesh = result.to_mesh()
    volume = result.volume() if hasattr(result, 'volume') else 0
    actual_porosity = 1 - (volume / solid_volume) if solid_volume > 0 else 0

    stats = {
        'triangle_count': len(mesh.tri_verts) // 3 if hasattr(mesh, 'tri_verts') else 0,
        'volume_mm3': volume,
        'porosity': actual_porosity,
        'target_porosity': params.porosity,
        'adipocyte_parameters': {
            'diameter_um': params.adipocyte_diameter_um,
            'size_variance': params.adipocyte_size_variance,
            'cell_count': len(cells),
            'cell_density_per_mL': params.cell_density_per_mL
        },
        'lobule_parameters': {
            'enabled': params.enable_lobules,
            'size_mm': params.lobule_size_mm,
            'density_per_cm2': params.lobules_per_cm2,
            'lobule_count': len(lobule_centers),
            'septum_count': len(lobule_edges)
        },
        'septum_parameters': {
            'thickness_um': params.septum_thickness_um,
            'porosity': params.septum_porosity,
            'wall_count': len(septa_manifolds)
        },
        'vascular_network': {
            'enabled': params.enable_vascular_channels,
            'channel_count': len(channels),
            'channel_diameter_mm': params.vascular_channel_diameter_mm,
            'capillary_diameter_um': params.capillary_diameter_um,
            'capillary_density_per_mm2': params.capillary_density_per_mm2,
            'arteriole_diameter_um': params.arteriole_diameter_um,
            'venule_diameter_um': params.venule_diameter_um,
            'hierarchy_vessel_count': len(vascular_hierarchy)
        },
        'svf_channels': {
            'enabled': params.enable_svf_channels,
            'channel_count': len(svf_channels),
            'diameter_um': params.svf_channel_diameter_um
        },
        'ecm_fibers': {
            'enabled': params.enable_ecm_fibers,
            'collagen_density': params.collagen_fiber_density,
            'elastin_density': params.elastin_fiber_density,
            'fiber_count': len(ecm_fibers)
        },
        'basement_membrane': {
            'enabled': params.enable_basement_membrane,
            'thickness_um': params.basement_membrane_thickness_um,
            'membrane_count': len(basement_membranes)
        },
        'surface_texture': {
            'enabled': params.enable_surface_texture,
            'roughness': params.surface_roughness
        },
        'pore_architecture': {
            'pore_size_um': params.pore_size_um,
            'interconnectivity': params.pore_interconnectivity
        },
        'mechanical': {
            'target_stiffness_kPa': params.target_stiffness_kPa
        },
        'scaffold_type': 'adipose_tissue'
    }

    return result, stats


def generate_adipose_tissue_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate adipose tissue scaffold from dictionary parameters (for API compatibility).

    Args:
        params: Dictionary with keys matching AdiposeTissueParams fields

    Returns:
        Tuple of (manifold, stats_dict)
    """
    # Support legacy parameter names
    adipocyte_diameter_um = params.get('adipocyte_diameter_um', 90.0)
    if 'cell_size_um' in params:
        adipocyte_diameter_um = params['cell_size_um']

    septum_thickness_um = params.get('septum_thickness_um', 50.0)
    if 'wall_thickness_um' in params:
        septum_thickness_um = params['wall_thickness_um']

    return generate_adipose_tissue(AdiposeTissueParams(
        # Basic geometry
        diameter_mm=params.get('diameter_mm', 15.0),
        height_mm=params.get('height_mm', 10.0),

        # Adipocyte parameters
        adipocyte_diameter_um=adipocyte_diameter_um,
        adipocyte_size_variance=params.get('adipocyte_size_variance', 0.15),
        cell_density_per_mL=params.get('cell_density_per_mL', 40e6),

        # Lobule structure
        lobule_size_mm=params.get('lobule_size_mm', 2.0),
        enable_lobules=params.get('enable_lobules', True),
        lobules_per_cm2=params.get('lobules_per_cm2', 4.0),

        # Septum/connective tissue
        septum_thickness_um=septum_thickness_um,
        septum_porosity=params.get('septum_porosity', 0.3),

        # Vascular network
        enable_vascular_channels=params.get('enable_vascular_channels', True),
        capillary_diameter_um=params.get('capillary_diameter_um', 7.0),
        capillary_density_per_mm2=params.get('capillary_density_per_mm2', 400),
        vascular_channel_diameter_mm=params.get('vascular_channel_diameter_mm', 0.4),
        vascular_channel_spacing_mm=params.get('vascular_channel_spacing_mm', 2.0),

        # Arteriole/venule
        arteriole_diameter_um=params.get('arteriole_diameter_um', 30.0),
        venule_diameter_um=params.get('venule_diameter_um', 40.0),

        # Stromal vascular fraction
        enable_svf_channels=params.get('enable_svf_channels', False),
        svf_channel_diameter_um=params.get('svf_channel_diameter_um', 50.0),

        # Pore architecture
        porosity=params.get('porosity', 0.8),
        pore_size_um=params.get('pore_size_um', 200.0),
        pore_interconnectivity=params.get('pore_interconnectivity', 0.9),

        # ECM features
        enable_ecm_fibers=params.get('enable_ecm_fibers', False),
        collagen_fiber_density=params.get('collagen_fiber_density', 0.3),
        elastin_fiber_density=params.get('elastin_fiber_density', 0.1),

        # Basement membrane
        enable_basement_membrane=params.get('enable_basement_membrane', False),
        basement_membrane_thickness_um=params.get('basement_membrane_thickness_um', 0.1),

        # Surface features
        enable_surface_texture=params.get('enable_surface_texture', False),
        surface_roughness=params.get('surface_roughness', 0.2),

        # Mechanical properties
        target_stiffness_kPa=params.get('target_stiffness_kPa', 2.0),

        # Randomization
        seed=params.get('seed', 42),
        position_noise=params.get('position_noise', 0.2),

        # Resolution
        resolution=params.get('resolution', 8)
    ))
