"""
Octet truss lattice scaffold generator.

The octet truss is a stretch-dominated lattice with exceptional strength-to-weight
ratio. Each unit cell contains an octahedron surrounded by tetrahedra, creating
a highly efficient load-bearing structure.

Unit cell composition:
- 12 edge struts (cube edges)
- 24 face diagonal struts (6 faces x 4 diagonals each, but shared)
- 6 internal struts connecting face centers to body center

The structure is commonly used in aerospace and biomedical applications.
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
class OctetTrussParams:
    """
    Parameters for octet truss lattice generation with biologically realistic defaults.

    The octet truss is a stretch-dominated lattice with exceptional strength-to-weight
    ratio, commonly used for load-bearing bone scaffolds. The FCC-based structure
    provides isotropic mechanical properties.

    Literature references:
    - Optimal strut diameter: 0.8-1.5mm for bone scaffolds (Taniguchi et al., 2016)
    - Unit cell edge: 5-10mm for sufficient pore interconnectivity
    - Relative density: 0.3-0.6 for bone-mimicking properties
    - Pore size: 200-800 μm for bone ingrowth

    Attributes:
        # === Basic Geometry ===
        bounding_box_x_mm: Scaffold X dimension in millimeters
        bounding_box_y_mm: Scaffold Y dimension in millimeters
        bounding_box_z_mm: Scaffold Z dimension in millimeters
        unit_cell_edge_mm: Unit cell edge length (5-10mm for bone)

        # === Strut Properties ===
        strut_diameter_mm: Strut diameter (0.8-1.5mm typical)
        strut_length_mm: Derived from unit cell (read-only in stats)
        strut_taper: Taper ratio from center to joints (0-0.3)
        strut_surface_roughness: Surface roughness factor (0-1)

        # === Porosity & Pore Control ===
        relative_density: Solid volume fraction (0.3-0.6 for bone)
        pore_size_um: Target pore size (200-800 μm for bone ingrowth)
        d_over_pore_ratio: Strut diameter to pore size ratio

        # === Mechanical Properties (Target) ===
        normalized_modulus: E/E_s ratio (0.1-0.5 typical)
        yield_strength_target_mpa: Target yield strength
        elastic_modulus_target_gpa: Target elastic modulus

        # === Node Features ===
        node_fillet_radius_mm: Fillet radius at strut joints
        enable_node_spheres: Add spherical nodes at joints
        node_sphere_factor: Node sphere size relative to strut

        # === Gradient Features ===
        enable_gradient: Enable density gradient
        gradient_axis: Axis for gradient ('x', 'y', or 'z')
        gradient_start_density: Relative density at start
        gradient_end_density: Relative density at end

        # === Quality & Generation ===
        resolution: Cylinder segments (6-16)
        seed: Random seed for any stochastic features
    """
    # Basic Geometry
    bounding_box_x_mm: float = 10.0
    bounding_box_y_mm: float = 10.0
    bounding_box_z_mm: float = 10.0
    unit_cell_edge_mm: float = 7.5  # Optimal for bone scaffolds

    # Strut Properties
    strut_diameter_mm: float = 1.2  # 0.8-1.5mm for bone
    strut_taper: float = 0.0  # No taper by default
    strut_surface_roughness: float = 0.0  # Smooth

    # Porosity & Pore Control
    relative_density: float = 0.48  # Mid-range for bone
    pore_size_um: float = 500.0  # 200-800 μm range
    d_over_pore_ratio: float = 2.4  # strut_diameter / pore_size

    # Mechanical Properties (Target)
    normalized_modulus: float = 0.25  # E/E_s
    yield_strength_target_mpa: float = 50.0
    elastic_modulus_target_gpa: float = 10.0

    # Node Features
    node_fillet_radius_mm: float = 0.0  # No fillet by default
    enable_node_spheres: bool = False
    node_sphere_factor: float = 1.2  # Slightly larger than strut

    # Gradient Features
    enable_gradient: bool = False
    gradient_axis: str = 'z'
    gradient_start_density: float = 0.3
    gradient_end_density: float = 0.6

    # Quality & Generation
    resolution: int = 8
    seed: int | None = None

    # Legacy compatibility
    bounding_box_mm: tuple[float, float, float] | None = None
    cell_size_mm: float | None = None


def make_strut(p1: np.ndarray, p2: np.ndarray, radius: float, resolution: int) -> m3d.Manifold:
    """
    Create a cylindrical strut between two points.

    Args:
        p1: Starting point (x, y, z)
        p2: Ending point (x, y, z)
        radius: Strut radius
        resolution: Number of segments around cylinder

    Returns:
        Manifold representing the strut cylinder
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    length = np.sqrt(dx*dx + dy*dy + dz*dz)

    if length < 1e-6:
        return m3d.Manifold()

    # Create cylinder along Z axis
    strut = m3d.Manifold.cylinder(length, radius, radius, resolution)

    # Calculate rotation to align with direction vector
    h = np.sqrt(dx*dx + dy*dy)
    if h > 0.001 or abs(dz) > 0.001:
        angle_y = np.arctan2(h, dz) * 180 / np.pi
        angle_z = np.arctan2(dy, dx) * 180 / np.pi
        strut = strut.rotate([0, angle_y, 0])
        strut = strut.rotate([0, 0, angle_z])

    return strut.translate([p1[0], p1[1], p1[2]])


def make_tapered_strut(
    p1: np.ndarray,
    p2: np.ndarray,
    radius_max: float,
    taper: float,
    resolution: int,
    surface_roughness: float = 0.0,
    rng: np.random.Generator | None = None
) -> m3d.Manifold:
    """
    Create a tapered strut between two points.

    The strut is thicker at the endpoints (nodes) and thinner at the midpoint.
    This mimics natural bone strut geometry and reduces stress concentrations.

    Taper formula: r(t) = r_mid + (r_max - r_mid) * |2t - 1|
    where t ∈ [0, 1] is the position along the strut.
    At endpoints (t=0, t=1): r = r_max
    At midpoint (t=0.5): r = r_mid = r_max * (1 - taper)

    Args:
        p1: Starting point (x, y, z)
        p2: Ending point (x, y, z)
        radius_max: Maximum radius at endpoints
        taper: Taper ratio (0=uniform, 1=fully tapered to zero at midpoint)
        resolution: Number of segments around cylinder
        surface_roughness: Surface roughness Ra in μm (adds micro-variation)
        rng: Random number generator for roughness

    Returns:
        Manifold representing the tapered strut
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    length = np.sqrt(dx*dx + dy*dy + dz*dz)

    if length < 1e-6:
        return m3d.Manifold()

    # Clamp taper to reasonable range
    taper = max(0.0, min(0.95, taper))

    # Calculate minimum radius at midpoint
    radius_mid = radius_max * (1.0 - taper)

    # Number of segments for tapering (more segments = smoother taper)
    n_segments = max(3, int(8 * taper) + 3)  # 3-11 segments based on taper

    # Create segments
    segments = []
    segment_length = length / n_segments

    for i in range(n_segments):
        # Position along strut (0 to 1)
        t_start = i / n_segments
        t_end = (i + 1) / n_segments
        t_mid = (t_start + t_end) / 2

        # Radius at this segment: r(t) = r_mid + (r_max - r_mid) * |2t - 1|
        # At t=0: |2*0-1| = 1 → r = r_mid + (r_max - r_mid) = r_max
        # At t=0.5: |2*0.5-1| = 0 → r = r_mid
        # At t=1: |2*1-1| = 1 → r = r_max
        r_start = radius_mid + (radius_max - radius_mid) * abs(2 * t_start - 1)
        r_end = radius_mid + (radius_max - radius_mid) * abs(2 * t_end - 1)

        # Apply surface roughness as slight random variation
        if surface_roughness > 0 and rng is not None:
            # Convert roughness from μm to mm and apply as percentage variation
            roughness_mm = surface_roughness / 1000.0
            r_start *= (1.0 + rng.uniform(-roughness_mm, roughness_mm) / radius_max)
            r_end *= (1.0 + rng.uniform(-roughness_mm, roughness_mm) / radius_max)

        # Create tapered cylinder segment
        seg = m3d.Manifold.cylinder(segment_length, r_start, r_end, resolution)
        seg = seg.translate([0, 0, i * segment_length])
        segments.append(seg)

    # Union all segments
    if len(segments) == 1:
        strut = segments[0]
    else:
        strut = segments[0]
        for seg in segments[1:]:
            strut = strut + seg

    # Rotate to align with direction vector
    h = np.sqrt(dx*dx + dy*dy)
    if h > 0.001 or abs(dz) > 0.001:
        angle_y = np.arctan2(h, dz) * 180 / np.pi
        angle_z = np.arctan2(dy, dx) * 180 / np.pi
        strut = strut.rotate([0, angle_y, 0])
        strut = strut.rotate([0, 0, angle_z])

    return strut.translate([p1[0], p1[1], p1[2]])


def make_node_sphere(center: np.ndarray, radius: float, resolution: int) -> m3d.Manifold:
    """
    Create a sphere at a lattice node.

    Args:
        center: Center point (x, y, z)
        radius: Sphere radius
        resolution: Number of segments (affects smoothness)

    Returns:
        Manifold representing the sphere
    """
    sphere = m3d.Manifold.sphere(radius, resolution * 2)
    return sphere.translate([center[0], center[1], center[2]])


def get_gradient_density_factor(
    position: np.ndarray,
    bbox: tuple[float, float, float],
    axis: str,
    start_density: float,
    end_density: float
) -> float:
    """
    Calculate density factor based on position along gradient axis.

    Args:
        position: 3D position (typically strut midpoint)
        bbox: Bounding box dimensions (x, y, z)
        axis: Gradient axis ('x', 'y', or 'z')
        start_density: Density at axis start (0)
        end_density: Density at axis end (max)

    Returns:
        Density factor (interpolated between start and end density)
    """
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    axis_idx = axis_map.get(axis.lower(), 2)  # Default to z

    # Get position along axis normalized to [0, 1]
    axis_max = bbox[axis_idx]
    if axis_max <= 0:
        return (start_density + end_density) / 2

    t = np.clip(position[axis_idx] / axis_max, 0.0, 1.0)

    # Linear interpolation between start and end density
    return start_density + t * (end_density - start_density)


def get_octet_truss_struts(cell_origin: np.ndarray, cell_size: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Get strut endpoints for an octet truss unit cell.

    The octet truss consists of:
    - 12 edge struts (cube edges)
    - Face diagonals connecting face centers to corners
    - Internal octahedron struts

    This implementation uses the FCC (face-centered cubic) node arrangement:
    - 8 corner nodes
    - 6 face center nodes

    Args:
        cell_origin: Origin point (x, y, z) of the unit cell
        cell_size: Size of the unit cell

    Returns:
        List of (p1, p2) tuples defining strut endpoints
    """
    L = cell_size
    o = cell_origin

    # 8 corner nodes
    corners = []
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                corners.append(o + np.array([i, j, k]) * L)

    # 6 face center nodes (FCC positions)
    face_centers = [
        o + np.array([0.5, 0.5, 0]) * L,    # bottom face
        o + np.array([0.5, 0.5, 1]) * L,    # top face
        o + np.array([0.5, 0, 0.5]) * L,    # front face
        o + np.array([0.5, 1, 0.5]) * L,    # back face
        o + np.array([0, 0.5, 0.5]) * L,    # left face
        o + np.array([1, 0.5, 0.5]) * L,    # right face
    ]

    struts = []

    # Edge struts (12 edges of the cube)
    # X-direction edges
    for j in [0, 1]:
        for k in [0, 1]:
            p1 = o + np.array([0, j, k]) * L
            p2 = o + np.array([1, j, k]) * L
            struts.append((p1, p2))

    # Y-direction edges
    for i in [0, 1]:
        for k in [0, 1]:
            p1 = o + np.array([i, 0, k]) * L
            p2 = o + np.array([i, 1, k]) * L
            struts.append((p1, p2))

    # Z-direction edges
    for i in [0, 1]:
        for j in [0, 1]:
            p1 = o + np.array([i, j, 0]) * L
            p2 = o + np.array([i, j, 1]) * L
            struts.append((p1, p2))

    # Face diagonal struts: connect each face center to its 4 corners
    # Bottom face (z=0)
    fc = o + np.array([0.5, 0.5, 0]) * L
    for i in [0, 1]:
        for j in [0, 1]:
            corner = o + np.array([i, j, 0]) * L
            struts.append((fc, corner))

    # Top face (z=1)
    fc = o + np.array([0.5, 0.5, 1]) * L
    for i in [0, 1]:
        for j in [0, 1]:
            corner = o + np.array([i, j, 1]) * L
            struts.append((fc, corner))

    # Front face (y=0)
    fc = o + np.array([0.5, 0, 0.5]) * L
    for i in [0, 1]:
        for k in [0, 1]:
            corner = o + np.array([i, 0, k]) * L
            struts.append((fc, corner))

    # Back face (y=1)
    fc = o + np.array([0.5, 1, 0.5]) * L
    for i in [0, 1]:
        for k in [0, 1]:
            corner = o + np.array([i, 1, k]) * L
            struts.append((fc, corner))

    # Left face (x=0)
    fc = o + np.array([0, 0.5, 0.5]) * L
    for j in [0, 1]:
        for k in [0, 1]:
            corner = o + np.array([0, j, k]) * L
            struts.append((fc, corner))

    # Right face (x=1)
    fc = o + np.array([1, 0.5, 0.5]) * L
    for j in [0, 1]:
        for k in [0, 1]:
            corner = o + np.array([1, j, k]) * L
            struts.append((fc, corner))

    # Internal octahedron struts: connect adjacent face centers
    # Each face center connects to 4 adjacent face centers
    # Bottom-to-sides
    struts.append((face_centers[0], face_centers[2]))  # bottom to front
    struts.append((face_centers[0], face_centers[3]))  # bottom to back
    struts.append((face_centers[0], face_centers[4]))  # bottom to left
    struts.append((face_centers[0], face_centers[5]))  # bottom to right

    # Top-to-sides
    struts.append((face_centers[1], face_centers[2]))  # top to front
    struts.append((face_centers[1], face_centers[3]))  # top to back
    struts.append((face_centers[1], face_centers[4]))  # top to left
    struts.append((face_centers[1], face_centers[5]))  # top to right

    # Horizontal ring (front-left-back-right-front)
    struts.append((face_centers[2], face_centers[4]))  # front to left
    struts.append((face_centers[4], face_centers[3]))  # left to back
    struts.append((face_centers[3], face_centers[5]))  # back to right
    struts.append((face_centers[5], face_centers[2]))  # right to front

    return struts


def generate_octet_truss(params: OctetTrussParams) -> tuple[m3d.Manifold, dict]:
    """
    Generate an octet truss lattice scaffold.

    Supports advanced features:
    - Strut tapering: Struts thicker at nodes, thinner at midpoint
    - Surface roughness: Micro-texture for cell adhesion
    - Node fillets: Smooth blend at strut junctions (reduces stress concentration)
    - Node spheres: Spherical reinforcement at lattice nodes
    - Density gradient: Varying strut thickness along an axis

    Args:
        params: OctetTrussParams configuration object

    Returns:
        Tuple of (manifold, statistics_dict)

        Statistics include:
            - triangle_count: Number of triangles in mesh
            - volume_mm3: Scaffold volume
            - relative_density: Volume fraction (solid/total)
            - cell_count: Number of unit cells
            - strut_count: Number of unique struts
            - node_count: Number of lattice nodes (if spheres enabled)
            - scaffold_type: 'octet_truss'

    Raises:
        ImportError: If manifold3d is not installed
        ValueError: If no struts are generated
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
    bbox = (bx, by, bz)

    # Handle legacy cell_size_mm or new unit_cell_edge_mm
    cell = params.cell_size_mm if params.cell_size_mm is not None else params.unit_cell_edge_mm
    base_radius = params.strut_diameter_mm / 2

    # Initialize random generator for surface roughness
    rng = np.random.default_rng(params.seed) if params.seed is not None else np.random.default_rng()

    # Calculate number of cells
    nx = max(1, int(bx / cell))
    ny = max(1, int(by / cell))
    nz = max(1, int(bz / cell))

    # Collect unique struts and nodes across all cells
    all_strut_endpoints = set()
    all_nodes = set()

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                origin = np.array([i, j, k], dtype=float) * cell
                struts = get_octet_truss_struts(origin, cell)
                for p1, p2 in struts:
                    p1_key = tuple(p1.round(6))
                    p2_key = tuple(p2.round(6))
                    key = tuple(sorted([p1_key, p2_key]))
                    all_strut_endpoints.add(key)
                    # Track unique node positions
                    all_nodes.add(p1_key)
                    all_nodes.add(p2_key)

    # Determine if we need advanced strut features
    use_taper = params.strut_taper > 0.01
    use_roughness = params.strut_surface_roughness > 0.01
    use_gradient = params.enable_gradient

    # Create strut manifolds
    strut_manifolds = []
    for (p1_tuple, p2_tuple) in all_strut_endpoints:
        p1 = np.array(p1_tuple)
        p2 = np.array(p2_tuple)

        # Calculate effective radius based on gradient
        effective_radius = base_radius
        if use_gradient:
            # Use strut midpoint for gradient calculation
            midpoint = (p1 + p2) / 2
            density_factor = get_gradient_density_factor(
                midpoint, bbox,
                params.gradient_axis,
                params.gradient_start_density,
                params.gradient_end_density
            )
            # Scale radius by density factor (higher density = thicker struts)
            # density_factor is relative density, so radius scales as sqrt(density)
            effective_radius = base_radius * np.sqrt(density_factor / 0.5)

        # Create strut with appropriate method
        if use_taper or use_roughness:
            strut = make_tapered_strut(
                p1, p2,
                effective_radius,
                params.strut_taper if use_taper else 0.0,
                params.resolution,
                params.strut_surface_roughness if use_roughness else 0.0,
                rng if use_roughness else None
            )
        else:
            strut = make_strut(p1, p2, effective_radius, params.resolution)

        if strut.num_vert() > 0:
            strut_manifolds.append(strut)

    if not strut_manifolds:
        raise ValueError("No struts generated for octet truss")

    # Add node spheres if enabled
    node_manifolds = []
    if params.enable_node_spheres:
        sphere_radius = base_radius * params.node_sphere_factor
        for node_tuple in all_nodes:
            node_pos = np.array(node_tuple)
            # Only add spheres within bounding box (with small tolerance)
            if (0 - sphere_radius <= node_pos[0] <= bx + sphere_radius and
                0 - sphere_radius <= node_pos[1] <= by + sphere_radius and
                0 - sphere_radius <= node_pos[2] <= bz + sphere_radius):

                # Apply gradient to sphere size if enabled
                if use_gradient:
                    density_factor = get_gradient_density_factor(
                        node_pos, bbox,
                        params.gradient_axis,
                        params.gradient_start_density,
                        params.gradient_end_density
                    )
                    local_sphere_radius = sphere_radius * np.sqrt(density_factor / 0.5)
                else:
                    local_sphere_radius = sphere_radius

                sphere = make_node_sphere(node_pos, local_sphere_radius, params.resolution)
                if sphere.num_vert() > 0:
                    node_manifolds.append(sphere)

    # Add fillets at node junctions if enabled
    fillet_manifolds = []
    if params.node_fillet_radius_mm > 0.01:
        # Fillets are approximated using small spheres at each node
        # This provides smooth stress distribution similar to true fillets
        fillet_radius = params.node_fillet_radius_mm
        for node_tuple in all_nodes:
            node_pos = np.array(node_tuple)
            # Only add fillets within bounding box
            if (0 <= node_pos[0] <= bx and
                0 <= node_pos[1] <= by and
                0 <= node_pos[2] <= bz):

                # Create fillet sphere (slightly larger than strut for blending)
                fillet_sphere_radius = base_radius + fillet_radius * 0.5
                if use_gradient:
                    density_factor = get_gradient_density_factor(
                        node_pos, bbox,
                        params.gradient_axis,
                        params.gradient_start_density,
                        params.gradient_end_density
                    )
                    fillet_sphere_radius *= np.sqrt(density_factor / 0.5)

                fillet = make_node_sphere(node_pos, fillet_sphere_radius, params.resolution)
                if fillet.num_vert() > 0:
                    fillet_manifolds.append(fillet)

    # Combine all manifolds
    all_manifolds = strut_manifolds + node_manifolds + fillet_manifolds

    # Union all parts
    result = batch_union(all_manifolds)

    # Clip to bounding box (intersection, not XOR)
    clip_box = m3d.Manifold.cube([bx, by, bz])
    result = result & clip_box

    # Calculate statistics
    mesh = result.to_mesh()
    volume = result.volume() if hasattr(result, 'volume') else 0
    solid_volume = bx * by * bz
    relative_density = volume / solid_volume if solid_volume > 0 else 0

    # Calculate derived strut length (face diagonal for octet)
    strut_length_mm = cell * np.sqrt(2) / 2  # Half of face diagonal

    # Estimate pore size based on geometry
    estimated_pore_size_um = (cell - params.strut_diameter_mm) * 1000 / 2

    stats = {
        'triangle_count': len(mesh.tri_verts) // 3 if hasattr(mesh, 'tri_verts') else 0,
        'vertex_count': len(mesh.vert_properties) if hasattr(mesh, 'vert_properties') else 0,
        'volume_mm3': volume,
        'relative_density': relative_density,
        'target_relative_density': params.relative_density,
        'cell_count': nx * ny * nz,
        'strut_count': len(all_strut_endpoints),
        'node_count': len(all_nodes),
        'unit_cell_edge_mm': cell,
        'strut_diameter_mm': params.strut_diameter_mm,
        'strut_length_mm': strut_length_mm,
        'strut_taper': params.strut_taper,
        'strut_surface_roughness': params.strut_surface_roughness,
        'pore_size_um': params.pore_size_um,
        'estimated_pore_size_um': estimated_pore_size_um,
        'd_over_pore_ratio': params.d_over_pore_ratio,
        'normalized_modulus': params.normalized_modulus,
        'yield_strength_target_mpa': params.yield_strength_target_mpa,
        'elastic_modulus_target_gpa': params.elastic_modulus_target_gpa,
        'node_fillet_radius_mm': params.node_fillet_radius_mm,
        'enable_node_spheres': params.enable_node_spheres,
        'node_sphere_factor': params.node_sphere_factor,
        'gradient_enabled': params.enable_gradient,
        'gradient_axis': params.gradient_axis if params.enable_gradient else None,
        'gradient_start_density': params.gradient_start_density if params.enable_gradient else None,
        'gradient_end_density': params.gradient_end_density if params.enable_gradient else None,
        'seed': params.seed,
        'scaffold_type': 'octet_truss'
    }

    return result, stats


def generate_octet_truss_from_dict(params: dict) -> tuple[m3d.Manifold, dict]:
    """
    Generate octet truss scaffold from dictionary parameters.

    Supports both new biological parameters and legacy parameter names
    for backward compatibility.

    Args:
        params: Dictionary with keys matching OctetTrussParams fields

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

    # Handle unit cell - legacy and new
    unit_cell = params.get('unit_cell_edge_mm', params.get('cell_size_mm', 7.5))

    return generate_octet_truss(OctetTrussParams(
        # Basic Geometry
        bounding_box_x_mm=bx,
        bounding_box_y_mm=by,
        bounding_box_z_mm=bz,
        unit_cell_edge_mm=unit_cell,

        # Strut Properties
        strut_diameter_mm=params.get('strut_diameter_mm', 1.2),
        strut_taper=params.get('strut_taper', 0.0),
        strut_surface_roughness=params.get('strut_surface_roughness', 0.0),

        # Porosity & Pore Control
        relative_density=params.get('relative_density', 0.48),
        pore_size_um=params.get('pore_size_um', 500.0),
        d_over_pore_ratio=params.get('d_over_pore_ratio', 2.4),

        # Mechanical Properties
        normalized_modulus=params.get('normalized_modulus', 0.25),
        yield_strength_target_mpa=params.get('yield_strength_target_mpa', 50.0),
        elastic_modulus_target_gpa=params.get('elastic_modulus_target_gpa', 10.0),

        # Node Features
        node_fillet_radius_mm=params.get('node_fillet_radius_mm', 0.0),
        enable_node_spheres=params.get('enable_node_spheres', False),
        node_sphere_factor=params.get('node_sphere_factor', 1.2),

        # Gradient Features
        enable_gradient=params.get('enable_gradient', False),
        gradient_axis=params.get('gradient_axis', 'z'),
        gradient_start_density=params.get('gradient_start_density', 0.3),
        gradient_end_density=params.get('gradient_end_density', 0.6),

        # Quality & Generation
        resolution=params.get('resolution', 8),
        seed=params.get('seed'),
    ))
