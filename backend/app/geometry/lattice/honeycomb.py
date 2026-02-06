"""
Honeycomb lattice scaffold generator.

Creates hexagonal honeycomb structures optimized for load-bearing applications.
The honeycomb pattern provides excellent strength-to-weight ratio and is
commonly used in structural applications.

Properties:
- High in-plane stiffness and strength
- Efficient material usage
- Uniform pore distribution
- Good for directional loading (perpendicular to honeycomb plane)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass

try:
    import manifold3d as m3d
    HAS_MANIFOLD = True
except ImportError:
    m3d = None
    HAS_MANIFOLD = False

from ..core import batch_union


@dataclass
class HoneycombParams:
    """
    Parameters for honeycomb lattice scaffold generation with biologically realistic defaults.

    Honeycomb structures provide excellent in-plane stiffness and strength with
    efficient material usage. The uniform pore distribution is ideal for
    applications requiring directional loading perpendicular to the honeycomb plane.

    Literature references:
    - Wall thickness: 0.5-2.0mm for structural integrity
    - Cell inner length: 2-5mm for cell infiltration
    - Cell angle: 120 degrees for regular hexagons
    - Porosity: 80-90% for high permeability

    Attributes:
        # === Basic Geometry ===
        bounding_box_x_mm: Scaffold X dimension in millimeters
        bounding_box_y_mm: Scaffold Y dimension in millimeters
        bounding_box_z_mm: Scaffold Z dimension in millimeters
        cell_height_mm: Extrusion height (if None, uses bounding_box z)

        # === Cell Geometry ===
        wall_thickness_mm: Wall thickness (0.5-2.0mm)
        cell_inner_length_mm: Inner side length of hexagon cell (2-5mm)
        cell_angle_deg: Cell angle (120 for regular hexagon)
        cell_orientation: Flat-top or pointy-top orientation

        # === Porosity ===
        porosity: Target void fraction (0.8-0.9)
        num_cells_x: Number of cells in X (auto-calculated if None)
        num_cells_y: Number of cells in Y (auto-calculated if None)

        # === Wall Features ===
        enable_gradient_wall_thickness: Gradient wall thickness
        wall_thickness_start_mm: Wall thickness at gradient start
        wall_thickness_end_mm: Wall thickness at gradient end
        gradient_axis: Axis for wall thickness gradient

        # === Surface Properties ===
        enable_wall_texture: Add surface texture to walls
        texture_depth_um: Texture depth for cell attachment
        enable_wall_perforations: Add perforations to walls
        perforation_diameter_um: Diameter of wall perforations
        perforation_spacing_mm: Spacing between perforations

        # === Quality & Generation ===
        resolution: Segments per hexagon side (1 = sharp edges)
        seed: Random seed for stochastic features
    """
    # Basic Geometry
    bounding_box_x_mm: float = 10.0
    bounding_box_y_mm: float = 10.0
    bounding_box_z_mm: float = 5.0
    cell_height_mm: float | None = None

    # Cell Geometry
    wall_thickness_mm: float = 1.0  # 0.5-2.0mm for structural integrity
    cell_inner_length_mm: float = 3.0  # Side length of hexagon (2-5mm)
    cell_angle_deg: float = 120.0  # Regular hexagon
    cell_orientation: str = 'flat_top'  # 'flat_top' or 'pointy_top'

    # Porosity
    porosity: float = 0.85  # 80-90% for high permeability
    num_cells_x: int | None = None  # Auto-calculated if None
    num_cells_y: int | None = None  # Auto-calculated if None

    # Wall Features
    enable_gradient_wall_thickness: bool = False
    wall_thickness_start_mm: float = 0.8
    wall_thickness_end_mm: float = 1.5
    gradient_axis: str = 'x'

    # Surface Properties
    enable_wall_texture: bool = False
    texture_depth_um: float = 20.0
    enable_wall_perforations: bool = False
    perforation_diameter_um: float = 200.0
    perforation_spacing_mm: float = 0.5

    # Quality & Generation
    resolution: int = 1
    seed: int | None = None

    # Legacy compatibility
    bounding_box_mm: tuple[float, float, float] | None = None
    cell_size_mm: float | None = None  # Maps to cell_inner_length_mm
    height_mm: float | None = None  # Maps to cell_height_mm


def _get_hexagon_vertices(
    center: tuple[float, float],
    radius: float,
    orientation: str = 'flat_top'
) -> list[tuple[float, float]]:
    """
    Get hexagon vertices for given center and radius.

    Args:
        center: (x, y) center of hexagon
        radius: Circumradius (center to vertex)
        orientation: 'flat_top' or 'pointy_top'

    Returns:
        List of 6 vertex coordinates
    """
    cx, cy = center
    # For flat_top: first vertex at 0 degrees (right side)
    # For pointy_top: first vertex at 30 degrees (top-right)
    angle_offset = 0 if orientation == 'flat_top' else np.pi / 6

    angles = [angle_offset + i * np.pi / 3 for i in range(6)]
    vertices = []
    for angle in angles:
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        vertices.append((x, y))

    return vertices


def create_hexagon_prism(
    center: tuple[float, float],
    radius: float,
    height: float,
    wall_thickness: float,
    orientation: str = 'flat_top'
) -> m3d.Manifold:
    """
    Create a hollow hexagonal prism (one honeycomb cell wall).

    The hexagon is created as outer - inner to form walls.

    Args:
        center: (x, y) center of hexagon
        radius: Circumradius (center to vertex)
        height: Extrusion height
        wall_thickness: Wall thickness
        orientation: 'flat_top' or 'pointy_top'

    Returns:
        Manifold representing the hollow hexagon prism
    """
    cx, cy = center

    # Get outer hexagon vertices
    outer_verts = [[x, y] for x, y in _get_hexagon_vertices(center, radius, orientation)]

    # Inner hexagon vertices (smaller radius for wall thickness)
    inner_radius = radius - wall_thickness / np.cos(np.pi / 6)  # Adjust for flat-to-flat
    if inner_radius <= 0:
        # Wall thickness too large, return solid hexagon
        inner_radius = radius * 0.1

    inner_verts = [[x, y] for x, y in _get_hexagon_vertices(center, inner_radius, orientation)]

    # Create outer prism
    outer_cross_section = m3d.CrossSection([outer_verts])
    outer_prism = m3d.Manifold.extrude(outer_cross_section, height)

    # Create inner prism (to subtract)
    inner_cross_section = m3d.CrossSection([inner_verts])
    inner_prism = m3d.Manifold.extrude(inner_cross_section, height + 0.1)
    inner_prism = inner_prism.translate([0, 0, -0.05])

    # Hollow hexagon = outer - inner
    return outer_prism - inner_prism


def create_hexagon_wall_segment(
    p1: tuple[float, float],
    p2: tuple[float, float],
    height: float,
    thickness: float,
    resolution: int = 1
) -> m3d.Manifold:
    """
    Create a wall segment between two points.

    Args:
        p1: Start point (x, y)
        p2: End point (x, y)
        height: Wall height
        thickness: Wall thickness
        resolution: Number of segments (for curved edges, not used for straight walls)

    Returns:
        Manifold representing the wall segment
    """
    x1, y1 = p1
    x2, y2 = p2

    # Calculate wall direction and normal
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx*dx + dy*dy)

    if length < 1e-6:
        return m3d.Manifold()

    # Normal vector (perpendicular to wall)
    nx = -dy / length
    ny = dx / length

    # Half thickness offset
    ht = thickness / 2

    # Wall corners (rectangle)
    corners = [
        [x1 - nx * ht, y1 - ny * ht],
        [x2 - nx * ht, y2 - ny * ht],
        [x2 + nx * ht, y2 + ny * ht],
        [x1 + nx * ht, y1 + ny * ht],
    ]

    cross_section = m3d.CrossSection([corners])
    return m3d.Manifold.extrude(cross_section, height)


def create_wall_with_gradient_thickness(
    p1: tuple[float, float],
    p2: tuple[float, float],
    height: float,
    thickness_start: float,
    thickness_end: float,
    gradient_axis: str,
    bbox: tuple[float, float, float]
) -> m3d.Manifold:
    """
    Create a wall segment with gradient thickness along specified axis.

    The wall is divided into segments with varying thickness based on position
    along the gradient axis.

    Args:
        p1: Start point (x, y)
        p2: End point (x, y)
        height: Wall height
        thickness_start: Thickness at gradient start (0 position)
        thickness_end: Thickness at gradient end (max position)
        gradient_axis: 'x', 'y', or 'z' for gradient direction
        bbox: Bounding box dimensions for normalizing position

    Returns:
        Manifold representing the wall with gradient thickness
    """
    x1, y1 = p1
    x2, y2 = p2

    # Calculate wall center position
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # Determine normalized position along gradient axis
    if gradient_axis == 'x':
        t = cx / bbox[0] if bbox[0] > 0 else 0.5
    elif gradient_axis == 'y':
        t = cy / bbox[1] if bbox[1] > 0 else 0.5
    else:  # 'z' - handled differently, use average
        t = 0.5  # For z-gradient, we'll slice the wall vertically

    t = np.clip(t, 0, 1)

    if gradient_axis == 'z':
        # For z-axis gradient, create stacked wall segments
        num_slices = max(3, int(height / 0.5))  # At least 3 slices
        slice_height = height / num_slices
        walls = []

        for i in range(num_slices):
            z_t = (i + 0.5) / num_slices  # Normalized z position
            thickness = thickness_start + (thickness_end - thickness_start) * z_t

            wall = create_hexagon_wall_segment(p1, p2, slice_height, thickness)
            wall = wall.translate([0, 0, i * slice_height])
            walls.append(wall)

        return batch_union(walls) if walls else m3d.Manifold()
    else:
        # For x or y gradient, use interpolated thickness
        thickness = thickness_start + (thickness_end - thickness_start) * t
        return create_hexagon_wall_segment(p1, p2, height, thickness)


def create_wall_perforations(
    p1: tuple[float, float],
    p2: tuple[float, float],
    height: float,
    wall_thickness: float,
    perforation_diameter_mm: float,
    spacing_mm: float,
    resolution: int = 16
) -> list[m3d.Manifold]:
    """
    Create perforation cylinders to subtract from a wall.

    Perforations are cylinders oriented perpendicular to the wall surface,
    spaced evenly along the wall length and height.

    Args:
        p1: Wall start point (x, y)
        p2: Wall end point (x, y)
        height: Wall height
        wall_thickness: Wall thickness (cylinder length)
        perforation_diameter_mm: Diameter of each perforation hole
        spacing_mm: Spacing between perforation centers
        resolution: Number of segments for cylinder

    Returns:
        List of cylinder manifolds to subtract from wall
    """
    x1, y1 = p1
    x2, y2 = p2

    dx = x2 - x1
    dy = y2 - y1
    wall_length = np.sqrt(dx*dx + dy*dy)

    if wall_length < spacing_mm:
        return []

    # Wall direction and normal
    dir_x = dx / wall_length
    dir_y = dy / wall_length
    norm_x = -dir_y
    norm_y = dir_x

    radius = perforation_diameter_mm / 2
    perforations = []

    # Calculate number of perforations along length and height
    num_along_length = max(1, int(wall_length / spacing_mm))
    num_along_height = max(1, int(height / spacing_mm))

    # Create perforation cylinders
    for i in range(num_along_length):
        t = (i + 0.5) / num_along_length
        px = x1 + t * dx
        py = y1 + t * dy

        for j in range(num_along_height):
            z = (j + 0.5) * spacing_mm
            if z >= height:
                continue

            # Create cylinder oriented along wall normal
            # Cylinder is created along Z, then rotated to align with wall normal
            cylinder = m3d.Manifold.cylinder(
                height=wall_thickness * 1.5,  # Slightly longer to ensure complete cut
                radius=radius,
                circular_segments=resolution
            )

            # Calculate rotation to align cylinder with wall normal
            # Default cylinder is along Z-axis, we need it along the XY normal
            angle = np.arctan2(norm_y, norm_x)

            # Rotate cylinder to lie along wall normal (in XY plane)
            cylinder = cylinder.rotate([90, 0, np.degrees(angle)])

            # Position cylinder at perforation location
            offset_x = px - norm_x * wall_thickness * 0.75
            offset_y = py - norm_y * wall_thickness * 0.75
            cylinder = cylinder.translate([offset_x, offset_y, z])

            perforations.append(cylinder)

    return perforations


def generate_honeycomb_walls(
    bbox: tuple[float, float, float],
    cell_size: float,
    wall_thickness: float,
    height: float,
    orientation: str = 'flat_top',
    num_cells_x: int | None = None,
    num_cells_y: int | None = None,
    enable_gradient: bool = False,
    thickness_start: float = 0.5,
    thickness_end: float = 1.0,
    gradient_axis: str = 'x',
    enable_perforations: bool = False,
    perforation_diameter_mm: float = 0.2,
    perforation_spacing_mm: float = 0.5,
    resolution: int = 1
) -> tuple[list[m3d.Manifold], list[m3d.Manifold], int]:
    """
    Generate all wall segments for a honeycomb lattice.

    Supports both flat-top and pointy-top orientations, gradient wall thickness,
    and wall perforations for enhanced cell migration.

    Args:
        bbox: Bounding box (x, y, z)
        cell_size: Flat-to-flat hexagon size
        wall_thickness: Wall thickness (used if gradient disabled)
        height: Extrusion height
        orientation: 'flat_top' or 'pointy_top'
        num_cells_x: Override automatic cell count in X
        num_cells_y: Override automatic cell count in Y
        enable_gradient: Enable gradient wall thickness
        thickness_start: Wall thickness at gradient start
        thickness_end: Wall thickness at gradient end
        gradient_axis: Axis for gradient ('x', 'y', or 'z')
        enable_perforations: Add perforations to walls
        perforation_diameter_mm: Perforation hole diameter
        perforation_spacing_mm: Spacing between perforations
        resolution: Mesh resolution for curved features

    Returns:
        Tuple of (wall_manifolds, perforation_manifolds, estimated_cell_count)
    """
    bx, by, bz = bbox

    # Hexagon geometry
    # cell_size = flat-to-flat distance = 2 * apothem
    apothem = cell_size / 2  # Distance from center to flat edge
    circumradius = apothem / np.cos(np.pi / 6)  # Distance from center to vertex

    # Spacing depends on orientation
    if orientation == 'pointy_top':
        # For pointy-top: vertices point up/down
        # Horizontal spacing is flat-to-flat
        # Vertical spacing involves the circumradius
        h_spacing = cell_size  # flat-to-flat
        v_spacing = 1.5 * circumradius
    else:
        # For flat-top: flat edges top/bottom
        # Horizontal spacing involves circumradius
        # Vertical spacing is flat-to-flat
        h_spacing = 1.5 * circumradius
        v_spacing = cell_size

    # Calculate or use specified cell counts
    if num_cells_x is not None:
        nx = num_cells_x + 1
    else:
        nx = int(bx / h_spacing) + 2

    if num_cells_y is not None:
        ny = num_cells_y + 1
    else:
        ny = int(by / v_spacing) + 2

    # Track unique wall segments to avoid duplicates
    wall_segments = set()

    # Generate hexagon centers in a grid pattern
    for row in range(ny + 1):
        y_offset = row * v_spacing

        # Offset every other row for proper tessellation
        if orientation == 'pointy_top':
            x_offset = (cell_size * 0.5) if (row % 2) else 0
        else:
            x_offset = (circumradius * 0.75) if (row % 2) else 0

        for col in range(nx + 1):
            cx = col * h_spacing + x_offset
            cy = y_offset

            # Skip if center is too far outside bounds
            if cx > bx + cell_size or cy > by + cell_size:
                continue
            if cx < -cell_size or cy < -cell_size:
                continue

            # Generate 6 vertices of this hexagon
            vertices = _get_hexagon_vertices((cx, cy), circumradius, orientation)
            vertices = [(round(x, 4), round(y, 4)) for x, y in vertices]

            # Add wall segments (edges of hexagon)
            for i in range(6):
                p1 = vertices[i]
                p2 = vertices[(i + 1) % 6]
                # Normalize segment direction for deduplication
                segment = tuple(sorted([p1, p2]))
                wall_segments.add(segment)

    # Create wall manifolds
    walls = []
    all_perforations = []

    for (p1, p2) in wall_segments:
        # Only create walls that are at least partially within bounds
        if (max(p1[0], p2[0]) < 0 or min(p1[0], p2[0]) > bx or
            max(p1[1], p2[1]) < 0 or min(p1[1], p2[1]) > by):
            continue

        # Create wall with or without gradient
        if enable_gradient:
            wall = create_wall_with_gradient_thickness(
                p1, p2, height,
                thickness_start, thickness_end,
                gradient_axis, bbox
            )
        else:
            wall = create_hexagon_wall_segment(p1, p2, height, wall_thickness, resolution)

        if wall.num_vert() > 0:
            walls.append(wall)

            # Generate perforations for this wall if enabled
            if enable_perforations:
                perf_thickness = wall_thickness
                if enable_gradient:
                    # Use average thickness for perforation depth
                    perf_thickness = (thickness_start + thickness_end) / 2

                perfs = create_wall_perforations(
                    p1, p2, height, perf_thickness,
                    perforation_diameter_mm, perforation_spacing_mm,
                    resolution=max(8, resolution * 8)
                )
                all_perforations.extend(perfs)

    # Estimate cell count
    cells_x = num_cells_x if num_cells_x is not None else int(bx / h_spacing)
    cells_y = num_cells_y if num_cells_y is not None else int(by / v_spacing)
    estimated_cells = max(1, cells_x) * max(1, cells_y)

    return walls, all_perforations, estimated_cells


def generate_honeycomb(params: HoneycombParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate a honeycomb lattice scaffold.

    Creates a hexagonal honeycomb pattern extruded to the specified height.
    The walls form a continuous interconnected structure ideal for
    load-bearing applications.

    Supports:
    - Flat-top or pointy-top hexagon orientation
    - Manual cell count override (num_cells_x, num_cells_y)
    - Gradient wall thickness along any axis
    - Wall perforations for nutrient transport and cell migration
    - Surface texture (recorded in stats for manufacturing)
    - Configurable mesh resolution

    Args:
        params: HoneycombParams configuration object

    Returns:
        Tuple of (manifold, statistics_dict)

        Statistics include:
            - triangle_count: Number of triangles in mesh
            - volume_mm3: Scaffold volume
            - relative_density: Volume fraction
            - wall_count: Number of wall segments
            - scaffold_type: 'honeycomb'
            - cell_orientation: Hexagon orientation used
            - perforations_enabled: Whether perforations were added
            - texture_enabled: Whether texture is specified

    Raises:
        ImportError: If manifold3d is not installed
        ValueError: If no walls are generated
    """
    if not HAS_MANIFOLD:
        raise ImportError(
            "manifold3d is required for geometry operations. "
            "Install with: pip install manifold3d"
        )

    # Handle both legacy tuple format and new individual params
    if params.bounding_box_mm is not None:
        bx, by, bz = params.bounding_box_mm
    else:
        bx = params.bounding_box_x_mm
        by = params.bounding_box_y_mm
        bz = params.bounding_box_z_mm

    # Handle height - priority: height_mm > cell_height_mm > bounding_box z
    if params.height_mm is not None:
        height = params.height_mm
    elif params.cell_height_mm is not None:
        height = params.cell_height_mm
    else:
        height = bz

    # Handle cell size - legacy cell_size_mm or new cell_inner_length_mm
    cell_size = params.cell_size_mm if params.cell_size_mm is not None else params.cell_inner_length_mm

    # Normalize orientation parameter
    orientation = params.cell_orientation.lower().replace('-', '_').replace(' ', '_')
    if orientation in ('flat', 'flat_top', 'flattop'):
        orientation = 'flat_top'
    elif orientation in ('pointy', 'pointy_top', 'pointytop'):
        orientation = 'pointy_top'
    else:
        orientation = 'flat_top'  # Default

    # Convert perforation diameter from μm to mm
    perforation_diameter_mm = params.perforation_diameter_um / 1000.0

    # Generate all wall segments with full parameter support
    walls, perforations, estimated_cells = generate_honeycomb_walls(
        bbox=(bx, by, bz),
        cell_size=cell_size,
        wall_thickness=params.wall_thickness_mm,
        height=height,
        orientation=orientation,
        num_cells_x=params.num_cells_x,
        num_cells_y=params.num_cells_y,
        enable_gradient=params.enable_gradient_wall_thickness,
        thickness_start=params.wall_thickness_start_mm,
        thickness_end=params.wall_thickness_end_mm,
        gradient_axis=params.gradient_axis.lower(),
        enable_perforations=params.enable_wall_perforations,
        perforation_diameter_mm=perforation_diameter_mm,
        perforation_spacing_mm=params.perforation_spacing_mm,
        resolution=params.resolution
    )

    if not walls:
        raise ValueError("No honeycomb walls generated")

    # Union all walls
    result = batch_union(walls)

    # Subtract perforations if any
    if perforations:
        perforation_union = batch_union(perforations)
        result = result - perforation_union

    # Clip to bounding box (intersection, not XOR)
    clip_box = m3d.Manifold.cube([bx, by, height])
    result = result & clip_box

    # Calculate statistics
    mesh = result.to_mesh()
    volume = result.volume() if hasattr(result, 'volume') else 0
    solid_volume = bx * by * height
    relative_density = volume / solid_volume if solid_volume > 0 else 0
    porosity = 1.0 - relative_density

    # Calculate actual cell counts
    apothem = cell_size / 2
    circumradius = apothem / np.cos(np.pi / 6)

    if orientation == 'pointy_top':
        h_spacing = cell_size
        v_spacing = 1.5 * circumradius
    else:
        h_spacing = 1.5 * circumradius
        v_spacing = cell_size

    cells_x = params.num_cells_x if params.num_cells_x is not None else int(bx / h_spacing)
    cells_y = params.num_cells_y if params.num_cells_y is not None else int(by / v_spacing)

    # Calculate effective wall thickness for stats
    if params.enable_gradient_wall_thickness:
        effective_wall_thickness = (params.wall_thickness_start_mm + params.wall_thickness_end_mm) / 2
    else:
        effective_wall_thickness = params.wall_thickness_mm

    stats = {
        'triangle_count': len(mesh.tri_verts) // 3 if hasattr(mesh, 'tri_verts') else 0,
        'vertex_count': len(mesh.vert_properties) if hasattr(mesh, 'vert_properties') else 0,
        'volume_mm3': volume,
        'relative_density': relative_density,
        'porosity': porosity,
        'target_porosity': params.porosity,
        'wall_count': len(walls),
        'estimated_cell_count': estimated_cells,
        'cells_x': cells_x,
        'cells_y': cells_y,
        'wall_thickness_mm': effective_wall_thickness,
        'cell_inner_length_mm': cell_size,
        'cell_angle_deg': params.cell_angle_deg,
        'cell_height_mm': height,
        'cell_orientation': orientation,

        # Gradient wall thickness
        'gradient_wall_enabled': params.enable_gradient_wall_thickness,
        'gradient_axis': params.gradient_axis if params.enable_gradient_wall_thickness else None,
        'wall_thickness_start_mm': params.wall_thickness_start_mm if params.enable_gradient_wall_thickness else None,
        'wall_thickness_end_mm': params.wall_thickness_end_mm if params.enable_gradient_wall_thickness else None,

        # Perforations
        'perforations_enabled': params.enable_wall_perforations,
        'perforation_count': len(perforations) if params.enable_wall_perforations else 0,
        'perforation_diameter_um': params.perforation_diameter_um if params.enable_wall_perforations else None,
        'perforation_spacing_mm': params.perforation_spacing_mm if params.enable_wall_perforations else None,

        # Surface texture (recorded for manufacturing, not geometrically modeled)
        'texture_enabled': params.enable_wall_texture,
        'texture_depth_um': params.texture_depth_um if params.enable_wall_texture else None,

        # Quality
        'resolution': params.resolution,
        'seed': params.seed,
        'scaffold_type': 'honeycomb'
    }

    return result, stats


def generate_honeycomb_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate honeycomb scaffold from dictionary parameters.

    Supports both new biological parameters and legacy parameter names
    for backward compatibility.

    Args:
        params: Dictionary with keys matching HoneycombParams fields

    Returns:
        Tuple of (manifold, statistics_dict)
    """
    # Handle bounding box - support multiple formats
    bbox = params.get('bounding_box_mm', params.get('bounding_box'))
    if isinstance(bbox, dict):
        bx = bbox.get('x', 10.0)
        by = bbox.get('y', 10.0)
        bz = bbox.get('z', 5.0)
    elif isinstance(bbox, (list, tuple)):
        bx, by, bz = bbox[0], bbox[1], bbox[2]
    else:
        bx = params.get('bounding_box_x_mm', 10.0)
        by = params.get('bounding_box_y_mm', 10.0)
        bz = params.get('bounding_box_z_mm', 5.0)

    # Handle cell size - legacy and new
    cell_inner = params.get('cell_inner_length_mm', params.get('cell_size_mm', 3.0))

    # Handle height - legacy and new
    cell_height = params.get('cell_height_mm', params.get('height_mm'))

    return generate_honeycomb(HoneycombParams(
        # Basic Geometry
        bounding_box_x_mm=bx,
        bounding_box_y_mm=by,
        bounding_box_z_mm=bz,
        cell_height_mm=cell_height,

        # Cell Geometry
        wall_thickness_mm=params.get('wall_thickness_mm', 1.0),
        cell_inner_length_mm=cell_inner,
        cell_angle_deg=params.get('cell_angle_deg', 120.0),
        cell_orientation=params.get('cell_orientation', 'flat_top'),

        # Porosity
        porosity=params.get('porosity', 0.85),
        num_cells_x=params.get('num_cells_x'),
        num_cells_y=params.get('num_cells_y'),

        # Wall Features
        enable_gradient_wall_thickness=params.get('enable_gradient_wall_thickness', False),
        wall_thickness_start_mm=params.get('wall_thickness_start_mm', 0.8),
        wall_thickness_end_mm=params.get('wall_thickness_end_mm', 1.5),
        gradient_axis=params.get('gradient_axis', 'x'),

        # Surface Properties
        enable_wall_texture=params.get('enable_wall_texture', False),
        texture_depth_um=params.get('texture_depth_um', 20.0),
        enable_wall_perforations=params.get('enable_wall_perforations', False),
        perforation_diameter_um=params.get('perforation_diameter_um', 200.0),
        perforation_spacing_mm=params.get('perforation_spacing_mm', 0.5),

        # Quality & Generation
        resolution=params.get('resolution', 1),
        seed=params.get('seed'),
    ))
