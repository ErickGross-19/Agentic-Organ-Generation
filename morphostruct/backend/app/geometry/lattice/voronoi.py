"""
Voronoi lattice scaffold generator.

Creates organic, randomized cellular structures based on Voronoi tessellation.
The structure mimics natural trabecular bone architecture.

Properties:
- Randomized but controllable structure (via seed)
- Interconnected pore network
- Biomimetic appearance similar to cancellous bone
- Tunable cell density and strut thickness
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

try:
    from scipy.spatial import Voronoi
    HAS_SCIPY = True
except ImportError:
    Voronoi = None
    HAS_SCIPY = False

from ..core import batch_union


@dataclass
class VoronoiParams:
    """
    Parameters for Voronoi lattice scaffold generation with biologically realistic defaults.

    Voronoi scaffolds create organic, randomized cellular structures that mimic
    natural trabecular bone architecture. The controlled randomness provides
    excellent cell infiltration and nutrient transport while maintaining
    structural integrity.

    Literature references:
    - Target pore radius: 0.3-0.9mm for bone scaffolds
    - Target porosity: 60-80% for trabecular bone-like structures
    - Strut diameter: 0.2-0.5mm for balance of strength and porosity

    Attributes:
        # === Basic Geometry ===
        bounding_box_x_mm: Scaffold X dimension in millimeters
        bounding_box_y_mm: Scaffold Y dimension in millimeters
        bounding_box_z_mm: Scaffold Z dimension in millimeters

        # === Pore Structure ===
        target_pore_radius_mm: Target pore radius (0.3-0.9mm)
        target_porosity: Target void fraction (0.6-0.8)
        seed_count: Number of Voronoi seed points (auto-calculated if None)

        # === Strut Properties ===
        strut_diameter_mm: Strut diameter (0.2-0.5mm)
        strut_taper: Taper ratio from center to joints
        enable_strut_roughness: Add surface roughness

        # === Randomization Control ===
        random_coefficient: Overall randomness (0=regular, 1=fully random)
        irregularity: Shape irregularity factor (0-1)
        seed: Random seed for reproducibility

        # === Gradient Features ===
        enable_gradient: Enable density gradient
        gradient_direction: Direction vector or axis
        density_gradient_start: Density at gradient start
        density_gradient_end: Density at gradient end

        # === Quality & Generation ===
        resolution: Cylinder segments (6-12)
        margin_factor: Margin around bbox for seed points (0.1-0.5)
        min_strut_length_mm: Minimum strut length filter
    """
    # Basic Geometry
    bounding_box_x_mm: float = 10.0
    bounding_box_y_mm: float = 10.0
    bounding_box_z_mm: float = 10.0

    # Pore Structure
    target_pore_radius_mm: float = 0.6  # 0.3-0.9mm range
    target_porosity: float = 0.70  # 60-80% for trabecular-like
    seed_count: int | None = None  # Auto-calculated if None

    # Strut Properties
    strut_diameter_mm: float = 0.3  # 0.2-0.5mm for bone scaffolds
    strut_taper: float = 0.0
    enable_strut_roughness: bool = False
    strut_roughness_amplitude: float = 0.02  # mm

    # Randomization Control
    random_coefficient: float = 0.5  # Balance of regularity and randomness
    irregularity: float = 0.5  # Shape irregularity
    seed: int | None = None

    # Gradient Features
    enable_gradient: bool = False
    gradient_direction: str = 'z'  # 'x', 'y', 'z' or custom vector
    density_gradient_start: float = 0.5
    density_gradient_end: float = 0.9

    # Quality & Generation
    resolution: int = 8
    margin_factor: float = 0.2
    min_strut_length_mm: float = 0.1

    # Legacy compatibility
    bounding_box_mm: tuple[float, float, float] | None = None
    cell_count: int | None = None  # Maps to seed_count


def make_strut(
    p1: np.ndarray,
    p2: np.ndarray,
    radius: float,
    resolution: int,
    taper: float = 0.0,
    roughness_enabled: bool = False,
    roughness_amplitude: float = 0.0,
    rng: np.random.Generator | None = None
) -> m3d.Manifold:
    """
    Create a cylindrical strut between two points with optional taper and roughness.

    Args:
        p1: Starting point (x, y, z)
        p2: Ending point (x, y, z)
        radius: Strut radius at nodes (ends)
        resolution: Number of segments around cylinder
        taper: Taper ratio (0-1). 0=uniform, 1=midpoint radius is 0.
               Formula: r(t) = r_node - taper * r_node * (1 - |2t - 1|)
               At t=0 or t=1 (nodes): r = r_node
               At t=0.5 (midpoint): r = r_node * (1 - taper)
        roughness_enabled: Whether to add surface roughness
        roughness_amplitude: Roughness amplitude in mm (Ra value)
        rng: Random number generator for roughness

    Returns:
        Manifold representing the strut (cylinder or tapered/rough variant)
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    length = np.sqrt(dx*dx + dy*dy + dz*dz)

    if length < 1e-6:
        return m3d.Manifold()

    # Determine if we need segmented construction (taper or roughness)
    needs_segments = (taper > 0.01) or (roughness_enabled and roughness_amplitude > 0)

    if not needs_segments:
        # Simple uniform cylinder
        strut = m3d.Manifold.cylinder(length, radius, radius, resolution)
    else:
        # Build strut from multiple segments for taper/roughness
        # Number of segments along length - more for roughness
        n_segments = 8 if roughness_enabled else 5

        segment_length = length / n_segments
        segments = []

        for i in range(n_segments):
            # Parameter t along strut (0 to 1)
            t_start = i / n_segments
            t_end = (i + 1) / n_segments
            t_mid = (t_start + t_end) / 2

            # Calculate radius at start and end of this segment
            # Taper formula: r(t) = r_node * (1 - taper * (1 - |2t - 1|))
            # At t=0,1: factor = 1 - taper * 0 = 1 (full radius)
            # At t=0.5: factor = 1 - taper * 1 = 1 - taper (minimum radius)
            def taper_factor(t: float) -> float:
                return 1.0 - taper * (1.0 - abs(2.0 * t - 1.0))

            r_start = radius * taper_factor(t_start)
            r_end = radius * taper_factor(t_end)

            # Add roughness as random variation
            if roughness_enabled and roughness_amplitude > 0 and rng is not None:
                # Roughness as ±amplitude variation
                r_start += rng.uniform(-roughness_amplitude, roughness_amplitude)
                r_end += rng.uniform(-roughness_amplitude, roughness_amplitude)
                # Ensure positive radius
                r_start = max(r_start, radius * 0.3)
                r_end = max(r_end, radius * 0.3)

            # Create tapered cylinder segment (cone frustum)
            z_start = t_start * length
            seg = m3d.Manifold.cylinder(segment_length, r_start, r_end, resolution)
            seg = seg.translate([0, 0, z_start])
            segments.append(seg)

        # Union all segments
        if segments:
            from ..core import batch_union
            strut = batch_union(segments)
        else:
            strut = m3d.Manifold.cylinder(length, radius, radius, resolution)

    # Rotate to align with direction vector
    h = np.sqrt(dx*dx + dy*dy)
    if h > 0.001 or abs(dz) > 0.001:
        angle_y = np.arctan2(h, dz) * 180 / np.pi
        angle_z = np.arctan2(dy, dx) * 180 / np.pi
        strut = strut.rotate([0, angle_y, 0])
        strut = strut.rotate([0, 0, angle_z])

    return strut.translate([p1[0], p1[1], p1[2]])


def generate_seed_points(
    bbox: tuple[float, float, float],
    count: int,
    margin: float,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Generate random seed points for Voronoi tessellation.

    Points are generated within the bounding box plus a margin to ensure
    good edge coverage. Mirror points are added outside to handle boundaries.

    Args:
        bbox: Bounding box dimensions (x, y, z)
        count: Number of interior seed points
        margin: Margin around bbox for extended seed region
        rng: Numpy random generator

    Returns:
        Array of seed points (N, 3)
    """
    bx, by, bz = bbox

    # Generate interior points
    interior = rng.uniform(
        low=[0, 0, 0],
        high=[bx, by, bz],
        size=(count, 3)
    )

    # Add mirror points outside each face for better boundary behavior
    # This prevents infinite Voronoi ridges at boundaries
    mirror_points = []

    for point in interior:
        # Mirror across each face if point is close to that face
        x, y, z = point
        threshold = max(bx, by, bz) * 0.3

        if x < threshold:
            mirror_points.append([-x, y, z])
        if x > bx - threshold:
            mirror_points.append([2*bx - x, y, z])
        if y < threshold:
            mirror_points.append([x, -y, z])
        if y > by - threshold:
            mirror_points.append([x, 2*by - y, z])
        if z < threshold:
            mirror_points.append([x, y, -z])
        if z > bz - threshold:
            mirror_points.append([x, y, 2*bz - z])

    if mirror_points:
        all_points = np.vstack([interior, np.array(mirror_points)])
    else:
        all_points = interior

    return all_points


def get_voronoi_edges(
    vor: Voronoi,
    bbox: tuple[float, float, float],
    margin: float = 0.1,
    min_strut_length: float = 0.1
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Extract Voronoi edges that lie within or near the bounding box.

    Args:
        vor: Scipy Voronoi object
        bbox: Bounding box dimensions (x, y, z)
        margin: Margin for including edges near boundaries
        min_strut_length: Minimum strut length to include (filters short struts)

    Returns:
        List of (p1, p2) tuples defining edge endpoints
    """
    bx, by, bz = bbox
    edges = []

    for ridge_vertices in vor.ridge_vertices:
        if -1 in ridge_vertices:
            # Ridge goes to infinity - skip
            continue

        v1 = vor.vertices[ridge_vertices[0]]
        v2 = vor.vertices[ridge_vertices[1]]

        # Check if edge is within bounds (with margin)
        margin_x = bx * margin
        margin_y = by * margin
        margin_z = bz * margin

        # Check if both points are reasonably within bounds
        in_bounds_1 = (
            -margin_x <= v1[0] <= bx + margin_x and
            -margin_y <= v1[1] <= by + margin_y and
            -margin_z <= v1[2] <= bz + margin_z
        )
        in_bounds_2 = (
            -margin_x <= v2[0] <= bx + margin_x and
            -margin_y <= v2[1] <= by + margin_y and
            -margin_z <= v2[2] <= bz + margin_z
        )

        if in_bounds_1 and in_bounds_2:
            # Clip points to bounding box
            p1 = np.clip(v1, [0, 0, 0], [bx, by, bz])
            p2 = np.clip(v2, [0, 0, 0], [bx, by, bz])

            # Filter by minimum strut length
            strut_length = np.linalg.norm(p2 - p1)
            if strut_length >= min_strut_length:
                edges.append((p1, p2))

    return edges


def calculate_gradient_factor(
    p1: np.ndarray,
    p2: np.ndarray,
    bbox: tuple[float, float, float],
    direction: str,
    density_start: float,
    density_end: float
) -> float:
    """
    Calculate the density gradient factor for a strut based on its position.

    The gradient affects strut diameter: higher density = thicker struts, smaller pores.
    Lower density = thinner struts, larger pores.

    Args:
        p1: Start point of strut
        p2: End point of strut
        bbox: Bounding box dimensions (x, y, z)
        direction: Gradient direction ('x', 'y', 'z', or 'radial')
        density_start: Density factor at start of gradient (0-1)
        density_end: Density factor at end of gradient (0-1)

    Returns:
        Density factor (0-1) at strut midpoint position
    """
    bx, by, bz = bbox
    # Use strut midpoint for gradient calculation
    midpoint = (p1 + p2) / 2

    if direction == 'x':
        # Gradient along X axis
        t = midpoint[0] / bx if bx > 0 else 0.5
    elif direction == 'y':
        # Gradient along Y axis
        t = midpoint[1] / by if by > 0 else 0.5
    elif direction == 'z':
        # Gradient along Z axis (most common for layered scaffolds)
        t = midpoint[2] / bz if bz > 0 else 0.5
    elif direction == 'radial':
        # Radial gradient from center
        center = np.array([bx / 2, by / 2, bz / 2])
        max_dist = np.sqrt((bx/2)**2 + (by/2)**2 + (bz/2)**2)
        dist = np.linalg.norm(midpoint - center)
        t = dist / max_dist if max_dist > 0 else 0.5
    else:
        # Default to Z gradient
        t = midpoint[2] / bz if bz > 0 else 0.5

    # Clamp t to [0, 1]
    t = np.clip(t, 0.0, 1.0)

    # Linear interpolation between start and end density
    density = density_start + t * (density_end - density_start)

    return density


def generate_voronoi(params: VoronoiParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate a Voronoi lattice scaffold.

    Algorithm:
    1. Generate random seed points (with reproducible seed)
    2. Compute 3D Voronoi tessellation
    3. Extract edges that lie within bounding box
    4. Create cylindrical struts along each edge
    5. Union all struts and clip to bounding box

    Args:
        params: VoronoiParams configuration object

    Returns:
        Tuple of (manifold, statistics_dict)

        Statistics include:
            - triangle_count: Number of triangles in mesh
            - volume_mm3: Scaffold volume
            - relative_density: Volume fraction
            - cell_count: Number of Voronoi cells
            - strut_count: Number of struts created
            - scaffold_type: 'voronoi'

    Raises:
        ImportError: If required libraries are not installed
        ValueError: If no struts are generated
    """
    if not HAS_MANIFOLD:
        raise ImportError(
            "manifold3d is required for geometry operations. "
            "Install with: pip install manifold3d"
        )
    if not HAS_SCIPY:
        raise ImportError(
            "scipy is required for Voronoi tessellation. "
            "Install with: pip install scipy"
        )

    # Handle both legacy tuple format and new individual params
    if params.bounding_box_mm is not None:
        bx, by, bz = params.bounding_box_mm
    else:
        bx = params.bounding_box_x_mm
        by = params.bounding_box_y_mm
        bz = params.bounding_box_z_mm
    bbox = (bx, by, bz)
    radius = params.strut_diameter_mm / 2

    # Initialize random generator
    rng = np.random.default_rng(params.seed)

    # Determine seed count - use explicit value, legacy cell_count, or auto-calculate
    if params.seed_count is not None:
        actual_seed_count = params.seed_count
    elif params.cell_count is not None:
        actual_seed_count = params.cell_count
    else:
        # Auto-calculate based on target pore size and volume
        volume = bx * by * bz
        cell_volume = (4/3) * np.pi * (params.target_pore_radius_mm ** 3)
        actual_seed_count = max(10, int(volume / cell_volume / 2))

    # Generate seed points with mirror boundary handling
    points = generate_seed_points(
        bbox,
        actual_seed_count,
        params.margin_factor,
        rng
    )

    # Compute Voronoi tessellation
    vor = Voronoi(points)

    # Extract edges within bounding box, applying min_strut_length filter
    edges = get_voronoi_edges(
        vor,
        bbox,
        margin=0.05,
        min_strut_length=params.min_strut_length_mm
    )

    if not edges:
        raise ValueError("No Voronoi edges generated within bounding box")

    # Create strut manifolds with taper, gradient, and roughness
    strut_manifolds = []
    struts_filtered_count = 0

    for p1, p2 in edges:
        # Calculate effective radius for this strut
        effective_radius = radius

        # Apply density gradient if enabled
        if params.enable_gradient:
            density_factor = calculate_gradient_factor(
                p1, p2, bbox,
                params.gradient_direction,
                params.density_gradient_start,
                params.density_gradient_end
            )
            # Scale radius by density factor
            # Higher density = thicker struts (larger radius)
            # density_factor typically 0.3-0.9, so scale to reasonable range
            effective_radius = radius * (0.5 + density_factor)

        # Create strut with taper and optional roughness
        strut = make_strut(
            p1, p2,
            effective_radius,
            params.resolution,
            taper=params.strut_taper,
            roughness_enabled=params.enable_strut_roughness,
            roughness_amplitude=params.strut_roughness_amplitude,
            rng=rng
        )

        if strut.num_vert() > 0:
            strut_manifolds.append(strut)
        else:
            struts_filtered_count += 1

    if not strut_manifolds:
        raise ValueError("No valid struts created for Voronoi lattice")

    # Union all struts
    result = batch_union(strut_manifolds)

    # Clip to bounding box (intersection, not XOR)
    clip_box = m3d.Manifold.cube([bx, by, bz])
    result = result & clip_box

    # Calculate statistics
    mesh = result.to_mesh()
    scaffold_volume = result.volume() if hasattr(result, 'volume') else 0
    solid_volume = bx * by * bz
    relative_density = scaffold_volume / solid_volume if solid_volume > 0 else 0
    porosity = 1.0 - relative_density

    # Calculate average strut length
    total_length = sum(np.linalg.norm(np.array(p2) - np.array(p1)) for p1, p2 in edges)
    avg_strut_length = total_length / len(edges) if edges else 0

    stats = {
        'triangle_count': len(mesh.tri_verts) // 3 if hasattr(mesh, 'tri_verts') else 0,
        'vertex_count': len(mesh.vert_properties) if hasattr(mesh, 'vert_properties') else 0,
        'volume_mm3': scaffold_volume,
        'relative_density': relative_density,
        'porosity': porosity,
        'target_porosity': params.target_porosity,
        'seed_count': actual_seed_count,
        'strut_count': len(edges),
        'avg_strut_length_mm': avg_strut_length,
        'strut_diameter_mm': params.strut_diameter_mm,
        'target_pore_radius_mm': params.target_pore_radius_mm,
        'random_coefficient': params.random_coefficient,
        'irregularity': params.irregularity,
        # Strut taper info
        'strut_taper': params.strut_taper,
        'strut_taper_enabled': params.strut_taper > 0.01,
        # Roughness info
        'roughness_enabled': params.enable_strut_roughness,
        'strut_roughness_amplitude_mm': params.strut_roughness_amplitude if params.enable_strut_roughness else 0.0,
        # Gradient info
        'gradient_enabled': params.enable_gradient,
        'gradient_direction': params.gradient_direction if params.enable_gradient else None,
        'density_gradient_start': params.density_gradient_start if params.enable_gradient else None,
        'density_gradient_end': params.density_gradient_end if params.enable_gradient else None,
        # Filter info
        'min_strut_length_mm': params.min_strut_length_mm,
        'seed': params.seed,
        'scaffold_type': 'voronoi'
    }

    return result, stats


def generate_voronoi_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate Voronoi scaffold from dictionary parameters.

    Supports both new biological parameters and legacy parameter names
    for backward compatibility.

    Args:
        params: Dictionary with keys matching VoronoiParams fields

    Returns:
        Tuple of (manifold, statistics_dict)
    """
    # Handle bounding box - support multiple formats
    bbox = params.get('bounding_box_mm', params.get('bounding_box'))
    if isinstance(bbox, dict):
        bx = bbox.get('x', 10.0)
        by = bbox.get('y', 10.0)
        bz = bbox.get('z', 10.0)
    elif isinstance(bbox, (list, tuple)):
        bx, by, bz = bbox[0], bbox[1], bbox[2]
    else:
        bx = params.get('bounding_box_x_mm', 10.0)
        by = params.get('bounding_box_y_mm', 10.0)
        bz = params.get('bounding_box_z_mm', 10.0)

    # Handle seed_count vs legacy cell_count
    seed_count = params.get('seed_count', params.get('cell_count'))

    return generate_voronoi(VoronoiParams(
        # Basic Geometry
        bounding_box_x_mm=bx,
        bounding_box_y_mm=by,
        bounding_box_z_mm=bz,

        # Pore Structure
        target_pore_radius_mm=params.get('target_pore_radius_mm', 0.6),
        target_porosity=params.get('target_porosity', 0.70),
        seed_count=seed_count,

        # Strut Properties
        strut_diameter_mm=params.get('strut_diameter_mm', 0.3),
        strut_taper=params.get('strut_taper', 0.0),
        enable_strut_roughness=params.get('enable_strut_roughness', False),
        strut_roughness_amplitude=params.get('strut_roughness_amplitude', 0.02),

        # Randomization Control
        random_coefficient=params.get('random_coefficient', 0.5),
        irregularity=params.get('irregularity', 0.5),
        seed=params.get('seed'),

        # Gradient Features
        enable_gradient=params.get('enable_gradient', False),
        gradient_direction=params.get('gradient_direction', 'z'),
        density_gradient_start=params.get('density_gradient_start', 0.5),
        density_gradient_end=params.get('density_gradient_end', 0.9),

        # Quality & Generation
        resolution=params.get('resolution', 8),
        margin_factor=params.get('margin_factor', 0.2),
        min_strut_length_mm=params.get('min_strut_length_mm', 0.1),
    ))
