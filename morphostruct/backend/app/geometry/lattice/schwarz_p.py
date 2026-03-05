"""
Schwarz P (Primitive) triply periodic minimal surface scaffold generator.

The Schwarz P surface is one of the simplest TPMS, discovered by Hermann Schwarz
in 1865. It has cubic symmetry with circular openings along each axis.

Properties:
- Simpler topology than gyroid
- Higher connectivity in orthogonal directions
- Good for applications requiring directional flow
- Self-supporting for 3D printing

Implicit function: cos(2pi*x/L) + cos(2pi*y/L) + cos(2pi*z/L) = isovalue
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
class SchwarzPParams:
    """
    Parameters for Schwarz P TPMS scaffold generation with biologically realistic defaults.

    The Schwarz P surface has a simpler topology with higher connectivity
    in orthogonal directions, making it suitable for applications requiring
    directional flow and nutrient transport.

    Literature references:
    - Bi-modal pore distribution: small pores 100-200 μm, large pores 300-500 μm
    - Target porosity for tissue ingrowth: 70-80%
    - Wall thickness for structural integrity: 150-250 μm

    Attributes:
        # === Basic Geometry ===
        bounding_box_x_mm: Scaffold X dimension in millimeters
        bounding_box_y_mm: Scaffold Y dimension in millimeters
        bounding_box_z_mm: Scaffold Z dimension in millimeters
        unit_cell_size_mm: Unit cell period (1.0-3.0mm typical)

        # === Porosity Control ===
        porosity: Target void fraction (0.70-0.80 for high permeability)
        small_pore_size_um: Small pore diameter (100-200 μm for cell attachment)
        large_pore_size_um: Large pore diameter (300-500 μm for vascularization)
        wall_thickness_um: Wall thickness in micrometers (150-250 typical)
        isovalue: Implicit surface level (-3 to 3, primary porosity control)

        # === TPMS Shape Parameters ===
        k_parameter: Wave number modifier for surface shape
        s_parameter: Surface stretching factor

        # === Transport Properties ===
        fluid_permeability_target: Target permeability (Darcy units)
        diffusion_coefficient_ratio: Effective/bulk diffusion ratio

        # === Mechanical Properties ===
        mechanical_stability: Target stability factor (0-1)
        elastic_modulus_target_gpa: Target elastic modulus

        # === Gradient Features ===
        enable_gradient: Enable porosity/density gradient
        gradient_axis: Axis for gradient ('x', 'y', or 'z')
        gradient_start_porosity: Porosity at gradient start
        gradient_end_porosity: Porosity at gradient end

        # === Quality & Generation ===
        samples_per_cell: Grid resolution per unit cell (15-30)
        mesh_density: Output mesh density factor
        seed: Random seed (reserved for future stochastic features, currently unused)
        resolution: Overall resolution multiplier (affects mesh quality via sampling)
    """
    # Basic Geometry
    bounding_box_x_mm: float = 10.0
    bounding_box_y_mm: float = 10.0
    bounding_box_z_mm: float = 10.0
    unit_cell_size_mm: float = 2.0

    # Porosity Control
    porosity: float = 0.75  # High porosity for nutrient transport
    small_pore_size_um: float = 200.0  # For cell attachment
    large_pore_size_um: float = 400.0  # For vascularization
    wall_thickness_um: float = 200.0  # Structural integrity
    isovalue: float = 0.0  # Controls actual surface position

    # TPMS Shape Parameters
    k_parameter: float = 1.0  # Wave number modifier
    s_parameter: float = 1.0  # Surface stretching

    # Transport Properties
    fluid_permeability_target: float = 1e-9  # m² (Darcy)
    diffusion_coefficient_ratio: float = 0.3  # Effective/bulk

    # Mechanical Properties
    mechanical_stability: float = 0.7  # Stability factor
    elastic_modulus_target_gpa: float = 5.0  # Softer for cartilage-like

    # Gradient Features
    enable_gradient: bool = False
    gradient_axis: str = 'z'
    gradient_start_porosity: float = 0.6
    gradient_end_porosity: float = 0.85

    # Quality & Generation
    samples_per_cell: int = 20
    mesh_density: float = 1.0
    seed: int | None = None
    resolution: int = 15

    # Legacy compatibility
    bounding_box_mm: tuple[float, float, float] | None = None
    cell_size_mm: float | None = None
    wall_thickness_mm: float | None = None


def schwarz_p_function(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    L: float,
    k_param: float = 1.0,
    s_param: float = 1.0
) -> np.ndarray:
    """
    Evaluate the Schwarz P implicit function with shape modifiers.

    F(x,y,z) = s * (cos(k*kx) + cos(k*ky) + cos(k*kz))
    where k = 2*pi/L (base wave number for period L)

    Shape modifiers:
    - k_param: Wave number multiplier (>1 = higher frequency, more cells per period)
    - s_param: Surface stretching/scaling factor (affects isovalue sensitivity)

    The base function ranges from -3 to +3:
    - At corners of unit cell (0,0,0): cos(0)+cos(0)+cos(0) = 3
    - At center of unit cell (L/2,L/2,L/2): cos(pi)+cos(pi)+cos(pi) = -3

    With s_param, the range becomes -3*s to +3*s.

    Args:
        X, Y, Z: 3D coordinate arrays (meshgrid output)
        L: Unit cell period (cell size)
        k_param: Wave number multiplier (default 1.0, higher = more oscillations)
        s_param: Surface stretching factor (default 1.0, scales output amplitude)

    Returns:
        3D array of function values
    """
    k = 2 * np.pi / L * k_param
    return s_param * (np.cos(k * X) + np.cos(k * Y) + np.cos(k * Z))


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


def _porosity_to_isovalue(porosity: float) -> float:
    """
    Convert target porosity (0-1) to Schwarz P isovalue.

    For Schwarz P surface:
    - isovalue = 0: ~50% porosity
    - isovalue > 0: lower porosity (more solid)
    - isovalue < 0: higher porosity (more void)

    Empirical relationship: porosity ≈ 0.5 - 0.15 * isovalue
    Inverted: isovalue ≈ (0.5 - porosity) / 0.15

    Args:
        porosity: Target porosity (0.2 to 0.9 typical)

    Returns:
        Isovalue for marching cubes (-2.5 to 2.5 range, clamped)
    """
    # Clamp porosity to valid range
    porosity = max(0.2, min(0.9, porosity))
    # Invert the porosity-isovalue relationship
    isovalue = (0.5 - porosity) / 0.15
    # Clamp isovalue to safe range (avoid extreme values)
    return max(-2.5, min(2.5, isovalue))


def _gibson_ashby_porosity(target_modulus_gpa: float, solid_modulus_gpa: float = 20.0) -> float:
    """
    Calculate optimal porosity for target elastic modulus using Gibson-Ashby relationship.

    Gibson-Ashby equation for open-cell foams:
    E/E_s = C * (rho/rho_s)^n
    where C ≈ 1.0 and n ≈ 2 for open-cell structures

    For porosity phi: rho/rho_s = (1 - phi)
    So: E = E_s * (1 - phi)^2
    Inverted: phi = 1 - sqrt(E / E_s)

    Args:
        target_modulus_gpa: Target elastic modulus in GPa
        solid_modulus_gpa: Modulus of solid material (default 20 GPa for polymers)

    Returns:
        Suggested porosity (0.2 to 0.9 range)
    """
    if target_modulus_gpa >= solid_modulus_gpa:
        return 0.2  # Minimum porosity for very stiff targets
    if target_modulus_gpa <= 0:
        return 0.9  # Maximum porosity

    ratio = target_modulus_gpa / solid_modulus_gpa
    porosity = 1.0 - np.sqrt(ratio)
    return max(0.2, min(0.9, porosity))


def generate_schwarz_p(params: SchwarzPParams) -> tuple[Any, dict]:
    """
    Generate a Schwarz P TPMS scaffold using marching cubes.

    The algorithm:
    1. Create a 3D grid covering the bounding box
    2. Evaluate the Schwarz P implicit function at each grid point
    3. Apply gradient porosity if enabled (spatially-varying isovalue)
    4. Adjust for diffusion and mechanical targets
    5. Use marching cubes to extract the isosurface
    6. Return the mesh directly (no manifold3d boolean operations)

    Args:
        params: SchwarzPParams configuration object

    Returns:
        Tuple of (mesh_wrapper, statistics_dict)

        Statistics include:
            - triangle_count: Number of triangles in mesh
            - volume_mm3: Scaffold volume (0 for surfaces)
            - porosity: Void fraction (approximate)
            - cell_count: Number of unit cells in each dimension
            - scaffold_type: 'schwarz_p'
            - gradient_enabled: Whether gradient porosity was applied
            - diffusion_coefficient_ratio: Effective/free diffusion ratio used
            - elastic_modulus_target_gpa: Target modulus used
            - gibson_ashby_suggested_porosity: Porosity suggested by Gibson-Ashby

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

    # === mesh_density: Scale grid resolution ===
    # Higher mesh_density = finer mesh, more triangles
    mesh_density = params.mesh_density if params.mesh_density > 0 else 1.0

    # Calculate grid resolution with mesh_density and resolution multipliers
    # resolution parameter (default 15) affects overall mesh quality
    base_samples = params.samples_per_cell
    resolution_factor = params.resolution / 15.0  # Normalize around default of 15
    effective_samples = int(base_samples * mesh_density * resolution_factor)
    effective_samples = max(10, effective_samples)  # Ensure minimum viable resolution
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

    # Evaluate Schwarz P function with shape parameters
    F = schwarz_p_function(X, Y, Z, L, k_param=params.k_parameter, s_param=params.s_parameter)

    # === diffusion_coefficient_ratio: Adjust connectivity ===
    # Higher diffusion ratio requires better pore connectivity
    # We can modulate the base isovalue slightly to improve connectivity
    # Range: 0.1 (low connectivity) to 1.0 (high connectivity)
    diffusion_ratio = max(0.1, min(1.0, params.diffusion_coefficient_ratio))
    # Higher diffusion ratio -> slightly lower isovalue -> more open pores
    diffusion_isovalue_offset = (0.6 - diffusion_ratio) * 0.3  # Range: -0.12 to +0.15

    # === elastic_modulus_target_gpa: Gibson-Ashby guidance ===
    # Calculate suggested porosity based on target modulus
    gibson_ashby_porosity = _gibson_ashby_porosity(params.elastic_modulus_target_gpa)

    # === Determine effective isovalue ===
    if params.enable_gradient:
        # === Gradient porosity: spatially-varying isovalue ===
        # Convert start/end porosity to isovalues
        start_porosity = max(0.2, min(0.9, params.gradient_start_porosity))
        end_porosity = max(0.2, min(0.9, params.gradient_end_porosity))

        isovalue_start = _porosity_to_isovalue(start_porosity)
        isovalue_end = _porosity_to_isovalue(end_porosity)

        # Apply diffusion offset to both ends
        isovalue_start += diffusion_isovalue_offset
        isovalue_end += diffusion_isovalue_offset

        # Create position-dependent isovalue field based on gradient_axis
        axis = params.gradient_axis.lower()
        if axis == 'x':
            # Gradient along X: normalize X position to [0, 1]
            normalized_pos = X / bx
            axis_length = bx
        elif axis == 'y':
            # Gradient along Y: normalize Y position to [0, 1]
            normalized_pos = Y / by
            axis_length = by
        else:  # Default to 'z'
            # Gradient along Z: normalize Z position to [0, 1]
            normalized_pos = Z / bz
            axis_length = bz

        # Linear interpolation of isovalue along gradient axis
        # isovalue(pos) = isovalue_start + (isovalue_end - isovalue_start) * normalized_pos
        isovalue_field = isovalue_start + (isovalue_end - isovalue_start) * normalized_pos

        # For gradient case, we need to threshold F against the spatially-varying isovalue
        # Create a modified field: F_modified = F - isovalue_field
        # Then extract isosurface at level=0
        F_modified = F - isovalue_field

        # Extract isosurface at level 0 (where F = isovalue_field)
        try:
            verts, faces, _, _ = marching_cubes(F_modified, level=0.0, spacing=spacing)
        except ValueError as e:
            raise ValueError(f"Failed to extract Schwarz P gradient surface: {e}")

        # Calculate average porosity for stats
        avg_isovalue = (isovalue_start + isovalue_end) / 2
        estimated_porosity = 0.5 - (avg_isovalue * 0.15)
        estimated_porosity = max(0.2, min(0.9, estimated_porosity))

    else:
        # === Uniform porosity: single isovalue ===
        # Apply diffusion offset to base isovalue
        effective_isovalue = params.isovalue + diffusion_isovalue_offset

        # Extract single isosurface using marching cubes
        try:
            verts, faces, _, _ = marching_cubes(F, level=effective_isovalue, spacing=spacing)
        except ValueError as e:
            raise ValueError(f"Failed to extract Schwarz P surface: {e}")

        # Estimate porosity based on effective isovalue
        estimated_porosity = 0.5 - (effective_isovalue * 0.15)
        estimated_porosity = max(0.2, min(0.9, estimated_porosity))

    if len(verts) == 0 or len(faces) == 0:
        raise ValueError("Marching cubes produced empty mesh for Schwarz P surface")

    # Create wrapper with manifold3d-like interface
    result = _MarchingCubesMeshWrapper(verts, faces)

    # Calculate statistics
    cell_count = (int(bx / L), int(by / L), int(bz / L))
    total_cells = cell_count[0] * cell_count[1] * cell_count[2]

    stats = {
        'triangle_count': len(faces),
        'vertex_count': len(verts),
        'volume_mm3': 0.0,  # TPMS surfaces are not closed solids
        'porosity': estimated_porosity,
        'target_porosity': params.porosity,
        'cell_count': cell_count,
        'total_cells': total_cells,
        'unit_cell_size_mm': L,
        'small_pore_size_um': params.small_pore_size_um,
        'large_pore_size_um': params.large_pore_size_um,
        'wall_thickness_um': params.wall_thickness_um if hasattr(params, 'wall_thickness_um') else params.wall_thickness_mm * 1000 if params.wall_thickness_mm else 200.0,
        'k_parameter': params.k_parameter if hasattr(params, 'k_parameter') else 1.0,
        's_parameter': params.s_parameter if hasattr(params, 's_parameter') else 1.0,
        'fluid_permeability_target': params.fluid_permeability_target if hasattr(params, 'fluid_permeability_target') else 1e-9,
        'mechanical_stability': params.mechanical_stability if hasattr(params, 'mechanical_stability') else 0.7,
        'scaffold_type': 'schwarz_p',
        # === New parameters in stats ===
        'gradient_enabled': params.enable_gradient,
        'gradient_axis': params.gradient_axis if params.enable_gradient else None,
        'gradient_start_porosity': params.gradient_start_porosity if params.enable_gradient else None,
        'gradient_end_porosity': params.gradient_end_porosity if params.enable_gradient else None,
        'diffusion_coefficient_ratio': params.diffusion_coefficient_ratio,
        'diffusion_isovalue_offset': diffusion_isovalue_offset,
        'elastic_modulus_target_gpa': params.elastic_modulus_target_gpa,
        'gibson_ashby_suggested_porosity': gibson_ashby_porosity,
        'mesh_density': mesh_density,
        'resolution': params.resolution,
        'effective_samples_per_cell': effective_samples,
        'grid_resolution': (nx, ny, nz),
        'seed': params.seed,  # Reserved for future stochastic features
    }

    return result, stats


def generate_schwarz_p_from_dict(params: dict) -> tuple[Any, dict]:
    """
    Generate Schwarz P scaffold from dictionary parameters.

    Supports both new biological parameters and legacy parameter names
    for backward compatibility.

    Args:
        params: Dictionary with keys matching SchwarzPParams fields

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
    unit_cell = params.get('unit_cell_size_mm', params.get('cell_size_mm', 2.0))

    # Handle wall thickness - um or mm
    wall_um = params.get('wall_thickness_um', 200.0)
    wall_mm = params.get('wall_thickness_mm')
    if wall_mm is not None:
        wall_um = wall_mm * 1000.0

    return generate_schwarz_p(SchwarzPParams(
        # Basic Geometry
        bounding_box_x_mm=bx,
        bounding_box_y_mm=by,
        bounding_box_z_mm=bz,
        unit_cell_size_mm=unit_cell,

        # Porosity Control
        porosity=params.get('porosity', 0.75),
        small_pore_size_um=params.get('small_pore_size_um', 200.0),
        large_pore_size_um=params.get('large_pore_size_um', 400.0),
        wall_thickness_um=wall_um,
        isovalue=params.get('isovalue', 0.0),

        # TPMS Shape Parameters
        k_parameter=params.get('k_parameter', 1.0),
        s_parameter=params.get('s_parameter', 1.0),

        # Transport Properties
        fluid_permeability_target=params.get('fluid_permeability_target', 1e-9),
        diffusion_coefficient_ratio=params.get('diffusion_coefficient_ratio', 0.3),

        # Mechanical Properties
        mechanical_stability=params.get('mechanical_stability', 0.7),
        elastic_modulus_target_gpa=params.get('elastic_modulus_target_gpa', 5.0),

        # Gradient Features
        enable_gradient=params.get('enable_gradient', False),
        gradient_axis=params.get('gradient_axis', 'z'),
        gradient_start_porosity=params.get('gradient_start_porosity', 0.6),
        gradient_end_porosity=params.get('gradient_end_porosity', 0.85),

        # Quality & Generation
        samples_per_cell=params.get('samples_per_cell', 20),
        mesh_density=params.get('mesh_density', 1.0),
        seed=params.get('seed'),
        resolution=params.get('resolution', 15),
    ))
