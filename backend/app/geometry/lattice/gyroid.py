"""
Gyroid triply periodic minimal surface (TPMS) scaffold generator.

The gyroid is a TPMS discovered by Alan Schoen in 1970. It has excellent
properties for tissue engineering scaffolds:
- High surface area to volume ratio
- Fully interconnected pore network
- Smooth curvature (no sharp corners)
- Self-supporting for 3D printing

Implicit function: sin(2pi*x/L)cos(2pi*y/L) + sin(2pi*y/L)cos(2pi*z/L) + sin(2pi*z/L)cos(2pi*x/L) = isovalue
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Any

try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except ImportError:
    marching_cubes = None
    HAS_SKIMAGE = False


@dataclass
class GyroidParams:
    """
    Parameters for gyroid TPMS scaffold generation with biologically realistic defaults.

    The gyroid TPMS is widely used in bone tissue engineering due to its
    excellent mechanical properties, high surface area, and interconnected
    pore network ideal for cell infiltration and nutrient transport.

    Literature references:
    - Optimal pore size for bone ingrowth: 500-800 μm (Taniguchi et al., 2016)
    - Porosity range for trabecular bone: 50-62% (Gibson & Ashby, 1997)
    - Wall thickness for structural integrity: 200-300 μm

    Attributes:
        # === Basic Geometry ===
        bounding_box_x_mm: Scaffold X dimension in millimeters
        bounding_box_y_mm: Scaffold Y dimension in millimeters
        bounding_box_z_mm: Scaffold Z dimension in millimeters
        unit_cell_size_mm: Unit cell period (1.0-3.0mm for bone scaffolds)

        # === Porosity Control ===
        pore_size_um: Target pore diameter in micrometers (500-800 optimal for bone)
        porosity: Target void fraction (0.5-0.62 for trabecular bone-like)
        wall_thickness_um: Wall thickness in micrometers (200-300 typical)
        isovalue: Implicit surface level (-1 to 1, primary porosity control)

        # === Mechanical Properties (Target) ===
        elastic_modulus_target_gpa: Target elastic modulus (10-20 GPa for cortical bone)
        stress_distribution_uniformity: Target stress uniformity (0-1, 1=uniform)
        yield_strength_target_mpa: Target yield strength (optional optimization)

        # === Surface Properties ===
        surface_area_ratio: Target surface area to volume ratio
        enable_surface_texture: Add micro-texture for cell attachment
        texture_amplitude_um: Surface texture height (5-20 μm for osteoblasts)

        # === Gradient Features ===
        enable_gradient: Enable porosity/density gradient
        gradient_axis: Axis for gradient ('x', 'y', or 'z')
        gradient_start_porosity: Porosity at gradient start
        gradient_end_porosity: Porosity at gradient end

        # === Quality & Generation ===
        samples_per_cell: Grid resolution per unit cell (15-30)
        mesh_density: Output mesh density factor (1.0 = standard)
        seed: Random seed for reproducibility
        resolution: Overall resolution multiplier
    """
    # Basic Geometry
    bounding_box_x_mm: float = 10.0
    bounding_box_y_mm: float = 10.0
    bounding_box_z_mm: float = 10.0
    unit_cell_size_mm: float = 1.5  # Optimal for bone scaffolds

    # Porosity Control
    pore_size_um: float = 700.0  # Optimal for osteogenic differentiation
    porosity: float = 0.56  # Middle of 50-62% range
    wall_thickness_um: float = 250.0  # 200-300 μm for structural integrity
    isovalue: float = 0.0  # Controls actual surface position

    # Mechanical Properties (Target)
    elastic_modulus_target_gpa: float = 15.0  # Between trabecular (0.1-5) and cortical (10-30)
    stress_distribution_uniformity: float = 0.8  # High uniformity target
    yield_strength_target_mpa: float = 100.0  # Target yield strength

    # Surface Properties
    surface_area_ratio: float = 2.5  # mm²/mm³, high for cell attachment
    enable_surface_texture: bool = False
    texture_amplitude_um: float = 10.0  # Micro-roughness for osteoblasts

    # Gradient Features
    enable_gradient: bool = False
    gradient_axis: str = 'z'
    gradient_start_porosity: float = 0.5
    gradient_end_porosity: float = 0.7

    # Quality & Generation
    samples_per_cell: int = 20
    mesh_density: float = 1.0
    seed: int | None = None
    resolution: int = 15

    # Legacy compatibility
    bounding_box_mm: tuple[float, float, float] | None = None
    cell_size_mm: float | None = None
    wall_thickness_mm: float | None = None


def _apply_surface_texture(
    vertices: np.ndarray,
    faces: np.ndarray,
    amplitude_um: float,
    seed: int | None = None
) -> np.ndarray:
    """
    Apply micro-texture to mesh surface using noise-based vertex displacement.

    This creates surface roughness beneficial for cell attachment. The texture
    is applied by displacing vertices along their estimated normals.

    Args:
        vertices: Vertex array (N, 3) from marching cubes
        faces: Face array (M, 3) with vertex indices (used for normal estimation)
        amplitude_um: Texture amplitude in micrometers (5-20 μm typical for osteoblasts)
        seed: Random seed for reproducibility

    Returns:
        Modified vertex array with texture applied
    """
    if amplitude_um <= 0:
        return vertices

    # Convert amplitude from micrometers to millimeters
    amplitude_mm = amplitude_um / 1000.0

    # Set random seed for reproducibility
    rng = np.random.default_rng(seed)

    # Estimate vertex normals from adjacent faces
    # Initialize normal accumulator
    vertex_normals = np.zeros_like(vertices)

    # Calculate face normals and accumulate to vertices
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Face normals via cross product
    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normals = np.cross(edge1, edge2)

    # Normalize face normals
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1.0)  # Avoid division by zero
    face_normals = face_normals / norms

    # Accumulate face normals to vertices
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_normals)

    # Normalize vertex normals
    vertex_norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    vertex_norms = np.where(vertex_norms > 1e-10, vertex_norms, 1.0)
    vertex_normals = vertex_normals / vertex_norms

    # Generate pseudo-random displacement based on vertex position
    # Use simple noise: combine position-based randomness with smooth variation
    n_verts = len(vertices)

    # Create position-dependent noise (simulates Perlin-like behavior)
    # Hash vertex positions to create consistent noise
    pos_scale = 10.0  # Frequency of texture features
    noise_x = np.sin(vertices[:, 0] * pos_scale + vertices[:, 1] * 0.7)
    noise_y = np.sin(vertices[:, 1] * pos_scale + vertices[:, 2] * 0.7)
    noise_z = np.sin(vertices[:, 2] * pos_scale + vertices[:, 0] * 0.7)
    position_noise = (noise_x + noise_y + noise_z) / 3.0

    # Add random component for micro-roughness
    random_noise = rng.uniform(-1, 1, n_verts)

    # Combine: 70% position-based (smooth), 30% random (rough)
    displacement = (0.7 * position_noise + 0.3 * random_noise) * amplitude_mm

    # Apply displacement along normals
    textured_vertices = vertices + vertex_normals * displacement[:, np.newaxis]

    return textured_vertices


def _compute_gradient_field(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    L: float,
    gradient_axis: str,
    start_porosity: float,
    end_porosity: float,
    base_isovalue: float,
    bounding_box: tuple[float, float, float]
) -> np.ndarray:
    """
    Compute gyroid field with spatially varying isovalue for porosity gradient.

    The porosity gradient is achieved by varying the isovalue along the
    specified axis. Higher isovalue magnitude = lower porosity (more solid).

    Args:
        X, Y, Z: 3D coordinate arrays
        L: Unit cell period
        gradient_axis: 'x', 'y', or 'z'
        start_porosity: Porosity at axis start (0.0)
        end_porosity: Porosity at axis end
        base_isovalue: Base isovalue at neutral porosity
        bounding_box: (bx, by, bz) dimensions

    Returns:
        3D array of field values with gradient applied
    """
    # Compute base gyroid field
    k = 2 * np.pi / L
    base_field = (
        np.sin(k * X) * np.cos(k * Y) +
        np.sin(k * Y) * np.cos(k * Z) +
        np.sin(k * Z) * np.cos(k * X)
    )

    # Map porosity to isovalue offset
    # Porosity ~0.5 at isovalue=0 for gyroid
    # Lower porosity (more solid) = positive isovalue
    # Higher porosity (more void) = negative isovalue
    # Approximate relationship: porosity = 0.5 - isovalue * 0.15
    # Therefore: isovalue = (0.5 - porosity) / 0.15

    start_isovalue = (0.5 - start_porosity) / 0.15
    end_isovalue = (0.5 - end_porosity) / 0.15

    # Select coordinate array for gradient axis
    bx, by, bz = bounding_box
    if gradient_axis.lower() == 'x':
        coord = X
        max_coord = bx
    elif gradient_axis.lower() == 'y':
        coord = Y
        max_coord = by
    else:  # 'z' default
        coord = Z
        max_coord = bz

    # Normalize position along axis [0, 1]
    normalized_pos = coord / max_coord if max_coord > 0 else np.zeros_like(coord)
    normalized_pos = np.clip(normalized_pos, 0, 1)

    # Interpolate isovalue along gradient axis
    spatially_varying_isovalue = start_isovalue + (end_isovalue - start_isovalue) * normalized_pos

    # Subtract varying isovalue from field (shifts the zero-crossing)
    gradient_field = base_field - spatially_varying_isovalue

    return gradient_field


def gyroid_function(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, L: float) -> np.ndarray:
    """
    Evaluate the gyroid implicit function.

    F(x,y,z) = sin(kx)cos(ky) + sin(ky)cos(kz) + sin(kz)cos(kx)
    where k = 2*pi/L (wave number for period L)

    Args:
        X, Y, Z: 3D coordinate arrays (meshgrid output)
        L: Unit cell period (cell size)

    Returns:
        3D array of function values
    """
    k = 2 * np.pi / L
    return (
        np.sin(k * X) * np.cos(k * Y) +
        np.sin(k * Y) * np.cos(k * Z) +
        np.sin(k * Z) * np.cos(k * X)
    )


class _SimpleMesh:
    """
    Simple mesh wrapper that provides the interface expected by stl_export.

    Mimics the manifold3d Mesh interface with:
    - vert_properties: 2D array of vertex positions (N, 3)
    - tri_verts: 2D array of triangle vertex indices (M, 3)
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        """
        Initialize mesh from marching cubes output.

        Args:
            vertices: Vertex array (N, 3) from marching cubes
            faces: Face array (M, 3) with vertex indices
        """
        # Store as 2D arrays - stl_export expects vert_properties[:, :3]
        self.vert_properties = vertices.astype(np.float32)
        # Store faces as 2D array - stl_export expects tri_verts as (M, 3)
        self.tri_verts = faces.astype(np.uint32)


class _MarchingCubesMeshWrapper:
    """
    Wrapper class that provides a manifold3d-like interface for marching cubes output.

    This allows the API to use marching cubes meshes directly without requiring
    manifold3d boolean operations, which fail on non-manifold TPMS surfaces.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        """
        Initialize wrapper from marching cubes output.

        Args:
            vertices: Vertex array (N, 3)
            faces: Face array (M, 3) with vertex indices
        """
        self._vertices = vertices
        self._faces = faces
        self._mesh = _SimpleMesh(vertices, faces)

    def to_mesh(self) -> _SimpleMesh:
        """Return the mesh object."""
        return self._mesh

    def volume(self) -> float:
        """
        Return volume (0 for TPMS surfaces since they're not closed solids).
        """
        return 0.0


def generate_gyroid(params: GyroidParams) -> tuple[Any, dict]:
    """
    Generate a gyroid TPMS scaffold using marching cubes.

    The algorithm:
    1. Create a 3D grid covering the bounding box
    2. Evaluate the gyroid implicit function at each grid point
    3. Use marching cubes to extract the isosurface
    4. Return the mesh directly (no manifold3d boolean operations)

    Args:
        params: GyroidParams configuration object

    Returns:
        Tuple of (mesh_wrapper, statistics_dict)

        Statistics include:
            - triangle_count: Number of triangles in mesh
            - volume_mm3: Scaffold volume (0 for surfaces)
            - porosity: Void fraction (approximate)
            - cell_count: Number of unit cells in each dimension
            - scaffold_type: 'gyroid'

    Raises:
        ImportError: If required libraries are not installed
        ValueError: If surface extraction fails
    """
    if not HAS_SKIMAGE:
        raise ImportError(
            "scikit-image is required for TPMS generation. "
            "Install with: pip install scikit-image"
        )

    # Handle both legacy tuple format and new individual params
    if params.bounding_box_mm is not None:
        bx, by, bz = params.bounding_box_mm
    else:
        bx = params.bounding_box_x_mm
        by = params.bounding_box_y_mm
        bz = params.bounding_box_z_mm

    # Handle legacy cell_size_mm or new unit_cell_size_mm
    L = params.cell_size_mm if params.cell_size_mm is not None else params.unit_cell_size_mm

    # Apply mesh_density and resolution multipliers to grid resolution
    # resolution parameter scales the overall mesh quality (typical range: 5-30)
    effective_samples = int(params.samples_per_cell * params.mesh_density * (params.resolution / 15.0))
    effective_samples = max(10, effective_samples)  # Minimum 10 samples per cell

    # Calculate grid resolution with mesh_density applied
    nx = max(10, int(bx / L * effective_samples))
    ny = max(10, int(by / L * effective_samples))
    nz = max(10, int(bz / L * effective_samples))

    # Create coordinate arrays
    x = np.linspace(0, bx, nx)
    y = np.linspace(0, by, ny)
    z = np.linspace(0, bz, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Spacing for marching cubes
    spacing = (x[1] - x[0], y[1] - y[0], z[1] - z[0])

    # Evaluate gyroid function - with or without gradient
    if params.enable_gradient:
        # Use gradient field with spatially varying isovalue
        F = _compute_gradient_field(
            X, Y, Z, L,
            gradient_axis=params.gradient_axis,
            start_porosity=params.gradient_start_porosity,
            end_porosity=params.gradient_end_porosity,
            base_isovalue=params.isovalue,
            bounding_box=(bx, by, bz)
        )
        # For gradient field, extract at level=0 since isovalue is baked into field
        extraction_level = 0.0
    else:
        # Standard uniform gyroid
        F = gyroid_function(X, Y, Z, L)
        extraction_level = params.isovalue

    # Extract single isosurface using marching cubes
    try:
        verts, faces, _, _ = marching_cubes(F, level=extraction_level, spacing=spacing)
    except ValueError as e:
        raise ValueError(f"Failed to extract gyroid surface: {e}")

    if len(verts) == 0 or len(faces) == 0:
        raise ValueError("Marching cubes produced empty mesh for gyroid surface")

    # Apply surface texture if enabled
    if params.enable_surface_texture and params.texture_amplitude_um > 0:
        verts = _apply_surface_texture(
            verts, faces,
            amplitude_um=params.texture_amplitude_um,
            seed=params.seed
        )

    # Create wrapper with manifold3d-like interface
    result = _MarchingCubesMeshWrapper(verts, faces)

    # Calculate statistics
    cell_count = (int(bx / L), int(by / L), int(bz / L))
    total_cells = cell_count[0] * cell_count[1] * cell_count[2]

    # Estimate porosity based on isovalue
    # Gyroid porosity is approximately 0.5 at isovalue=0, varies with isovalue
    estimated_porosity = 0.5 - (params.isovalue * 0.15)
    estimated_porosity = max(0.3, min(0.8, estimated_porosity))

    stats = {
        'triangle_count': len(faces),
        'vertex_count': len(verts),
        'volume_mm3': 0.0,  # TPMS surfaces are not closed solids
        'porosity': estimated_porosity,
        'target_porosity': params.porosity,
        'cell_count': cell_count,
        'total_cells': total_cells,
        'unit_cell_size_mm': L,
        'pore_size_um': params.pore_size_um,
        'wall_thickness_um': params.wall_thickness_um if hasattr(params, 'wall_thickness_um') else params.wall_thickness_mm * 1000 if params.wall_thickness_mm else 250.0,
        'elastic_modulus_target_gpa': params.elastic_modulus_target_gpa if hasattr(params, 'elastic_modulus_target_gpa') else 15.0,
        'surface_area_ratio': params.surface_area_ratio if hasattr(params, 'surface_area_ratio') else 2.5,
        'gradient_enabled': params.enable_gradient if hasattr(params, 'enable_gradient') else False,
        'seed': params.seed if hasattr(params, 'seed') else None,
        'scaffold_type': 'gyroid',
        # Mechanical target parameters (for FEA optimization reference)
        'stress_distribution_uniformity': params.stress_distribution_uniformity,
        'yield_strength_target_mpa': params.yield_strength_target_mpa,
        # Surface texture parameters
        'surface_texture_enabled': params.enable_surface_texture,
        'texture_amplitude_um': params.texture_amplitude_um if params.enable_surface_texture else 0.0,
        # Gradient parameters (detailed)
        'gradient_axis': params.gradient_axis if params.enable_gradient else None,
        'gradient_start_porosity': params.gradient_start_porosity if params.enable_gradient else None,
        'gradient_end_porosity': params.gradient_end_porosity if params.enable_gradient else None,
        # Mesh generation parameters
        'mesh_density': params.mesh_density,
        'effective_samples_per_cell': effective_samples,
    }

    return result, stats


def generate_gyroid_from_dict(params: dict) -> tuple[Any, dict]:
    """
    Generate gyroid scaffold from dictionary parameters.

    Supports both new biological parameters and legacy parameter names
    for backward compatibility.

    Args:
        params: Dictionary with keys matching GyroidParams fields

    Returns:
        Tuple of (mesh_wrapper, statistics_dict)
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

    # Handle unit cell size - legacy and new
    unit_cell = params.get('unit_cell_size_mm', params.get('cell_size_mm', 1.5))

    # Handle wall thickness - um or mm
    wall_um = params.get('wall_thickness_um', 250.0)
    wall_mm = params.get('wall_thickness_mm')
    if wall_mm is not None:
        wall_um = wall_mm * 1000.0

    return generate_gyroid(GyroidParams(
        # Basic Geometry
        bounding_box_x_mm=bx,
        bounding_box_y_mm=by,
        bounding_box_z_mm=bz,
        unit_cell_size_mm=unit_cell,

        # Porosity Control
        pore_size_um=params.get('pore_size_um', 700.0),
        porosity=params.get('porosity', 0.56),
        wall_thickness_um=wall_um,
        isovalue=params.get('isovalue', 0.0),

        # Mechanical Properties
        elastic_modulus_target_gpa=params.get('elastic_modulus_target_gpa', 15.0),
        stress_distribution_uniformity=params.get('stress_distribution_uniformity', 0.8),
        yield_strength_target_mpa=params.get('yield_strength_target_mpa', 100.0),

        # Surface Properties
        surface_area_ratio=params.get('surface_area_ratio', 2.5),
        enable_surface_texture=params.get('enable_surface_texture', False),
        texture_amplitude_um=params.get('texture_amplitude_um', 10.0),

        # Gradient Features
        enable_gradient=params.get('enable_gradient', False),
        gradient_axis=params.get('gradient_axis', 'z'),
        gradient_start_porosity=params.get('gradient_start_porosity', 0.5),
        gradient_end_porosity=params.get('gradient_end_porosity', 0.7),

        # Quality & Generation
        samples_per_cell=params.get('samples_per_cell', 20),
        mesh_density=params.get('mesh_density', 1.0),
        seed=params.get('seed'),
        resolution=params.get('resolution', 15),
    ))
