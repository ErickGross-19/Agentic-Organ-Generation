"""
Mesh feature primitives for adding structural elements to domain meshes.

This module provides reusable geometry primitives like raised ridges that can
be applied to domain faces (cylinder tops, box faces, etc.) using robust
voxel-based union operations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, Union
import numpy as np
import trimesh

from ..core.domain import DomainSpec, BoxDomain, CylinderDomain


class FaceId(Enum):
    """Identifier for faces of a domain."""
    X_MIN = "x_min"
    X_MAX = "x_max"
    Y_MIN = "y_min"
    Y_MAX = "y_max"
    Z_MIN = "z_min"
    Z_MAX = "z_max"
    TOP = "z_max"
    BOTTOM = "z_min"


@dataclass
class RidgeSpec:
    """
    Specification for a raised ridge feature.
    
    Parameters
    ----------
    height : float
        Height of the ridge above the base surface
    thickness : float
        Width/thickness of the ridge (ring width for cylinders, frame width for boxes)
    overlap : float, optional
        Amount the ridge overlaps into the base mesh to ensure proper union.
        Defaults to 0.5 * height if not specified.
    inset : float
        Distance to inset the ridge from the outer boundary (default: 0)
    resolution : int
        Number of points around the ring/rectangle for mesh generation (default: 64)
    voxel_pitch : float, optional
        Voxel pitch for union operation. If None, auto-computed from thickness.
    """
    height: float
    thickness: float
    overlap: Optional[float] = None
    inset: float = 0.0
    resolution: int = 64
    voxel_pitch: Optional[float] = None
    
    def __post_init__(self):
        if self.overlap is None:
            self.overlap = 0.5 * self.height
        if self.voxel_pitch is None:
            self.voxel_pitch = min(self.thickness / 4, self.height / 4)


def create_annular_ridge(
    outer_radius: float,
    inner_radius: float,
    height: float,
    z_base: float,
    center_xy: Tuple[float, float] = (0.0, 0.0),
    resolution: int = 64,
) -> trimesh.Trimesh:
    """
    Create a watertight annular ring (ridge) mesh via direct vertex/face construction.
    
    No booleans required. The ring is a closed solid with:
    - outer wall
    - inner wall  
    - top annulus
    - bottom annulus
    
    Parameters
    ----------
    outer_radius : float
        Outer radius of the annular ring
    inner_radius : float
        Inner radius of the annular ring (hole radius)
    height : float
        Height of the ridge
    z_base : float
        Z coordinate of the bottom of the ridge
    center_xy : tuple
        (x, y) center of the ring (default: (0, 0))
    resolution : int
        Number of points around the circumference (default: 64)
        
    Returns
    -------
    trimesh.Trimesh
        Watertight annular ring mesh
    """
    cx, cy = center_xy
    n = max(resolution, 16)
    
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    c = np.cos(angles)
    s = np.sin(angles)
    
    z0 = float(z_base)
    z1 = float(z_base + height)
    
    outer_bottom = np.column_stack([cx + outer_radius * c, cy + outer_radius * s, np.full(n, z0)])
    outer_top = np.column_stack([cx + outer_radius * c, cy + outer_radius * s, np.full(n, z1)])
    inner_bottom = np.column_stack([cx + inner_radius * c, cy + inner_radius * s, np.full(n, z0)])
    inner_top = np.column_stack([cx + inner_radius * c, cy + inner_radius * s, np.full(n, z1)])
    
    vertices = np.vstack([outer_bottom, outer_top, inner_bottom, inner_top]).astype(np.float64)
    
    faces = []
    
    def idx_outer_bottom(i):
        return i
    
    def idx_outer_top(i):
        return n + i
    
    def idx_inner_bottom(i):
        return 2 * n + i
    
    def idx_inner_top(i):
        return 3 * n + i
    
    for i in range(n):
        j = (i + 1) % n
        
        a = idx_outer_bottom(i)
        b = idx_outer_bottom(j)
        c1 = idx_outer_top(i)
        d = idx_outer_top(j)
        faces.append([a, c1, b])
        faces.append([b, c1, d])
        
        a = idx_inner_bottom(i)
        b = idx_inner_bottom(j)
        c1 = idx_inner_top(i)
        d = idx_inner_top(j)
        faces.append([a, b, c1])
        faces.append([b, d, c1])
        
        a = idx_outer_top(i)
        b = idx_outer_top(j)
        c1 = idx_inner_top(i)
        d = idx_inner_top(j)
        faces.append([a, b, c1])
        faces.append([b, d, c1])
        
        a = idx_outer_bottom(i)
        b = idx_outer_bottom(j)
        c1 = idx_inner_bottom(i)
        d = idx_inner_bottom(j)
        faces.append([a, c1, b])
        faces.append([b, c1, d])
    
    ring = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int64), process=True)
    
    ring.remove_unreferenced_vertices()
    ring.merge_vertices()
    
    if ring.volume < 0:
        ring.invert()
    trimesh.repair.fix_normals(ring)
    trimesh.repair.fill_holes(ring)
    
    if not ring.is_watertight:
        ring.merge_vertices()
        trimesh.repair.fill_holes(ring)
        trimesh.repair.fix_normals(ring)
    
    return ring


def create_frame_ridge(
    width: float,
    depth: float,
    height: float,
    thickness: float,
    z_base: float,
    center_xy: Tuple[float, float] = (0.0, 0.0),
) -> trimesh.Trimesh:
    """
    Create a rectangular frame ridge for box domains.
    
    Creates a hollow rectangular frame (like a picture frame) that can be
    placed on a box face.
    
    Parameters
    ----------
    width : float
        Outer width of the frame (X dimension)
    depth : float
        Outer depth of the frame (Y dimension)
    height : float
        Height of the ridge
    thickness : float
        Thickness of the frame walls
    z_base : float
        Z coordinate of the bottom of the ridge
    center_xy : tuple
        (x, y) center of the frame (default: (0, 0))
        
    Returns
    -------
    trimesh.Trimesh
        Watertight frame ridge mesh
    """
    cx, cy = center_xy
    
    outer_box = trimesh.creation.box(
        extents=[width, depth, height],
        transform=trimesh.transformations.translation_matrix([cx, cy, z_base + height / 2])
    )
    
    inner_width = width - 2 * thickness
    inner_depth = depth - 2 * thickness
    
    if inner_width <= 0 or inner_depth <= 0:
        return outer_box
    
    inner_box = trimesh.creation.box(
        extents=[inner_width, inner_depth, height + 0.001],
        transform=trimesh.transformations.translation_matrix([cx, cy, z_base + height / 2])
    )
    
    try:
        frame = outer_box.difference(inner_box)
        if frame.is_empty:
            return outer_box
        return frame
    except Exception:
        return outer_box


def add_raised_ridge(
    base_mesh: trimesh.Trimesh,
    domain: Optional[DomainSpec] = None,
    face: FaceId = FaceId.TOP,
    ridge_spec: Optional[RidgeSpec] = None,
    height: Optional[float] = None,
    thickness: Optional[float] = None,
    overlap: Optional[float] = None,
    inset: float = 0.0,
    resolution: int = 64,
    voxel_pitch: Optional[float] = None,
) -> trimesh.Trimesh:
    """
    Add a raised ridge to a mesh face using voxel-based union.
    
    This is the main API for adding ridge features. It automatically determines
    the appropriate ridge geometry based on the domain type and face.
    
    Parameters
    ----------
    base_mesh : trimesh.Trimesh
        The base mesh to add the ridge to
    domain : DomainSpec, optional
        Domain specification for determining ridge geometry.
        If None, geometry is inferred from base_mesh bounds.
    face : FaceId
        Which face to add the ridge to (default: TOP/Z_MAX)
    ridge_spec : RidgeSpec, optional
        Full ridge specification. If provided, overrides individual parameters.
    height : float, optional
        Ridge height (required if ridge_spec not provided)
    thickness : float, optional
        Ridge thickness (required if ridge_spec not provided)
    overlap : float, optional
        Ridge overlap into base (default: 0.5 * height)
    inset : float
        Distance to inset ridge from boundary (default: 0)
    resolution : int
        Mesh resolution for ridge (default: 64)
    voxel_pitch : float, optional
        Voxel pitch for union (auto-computed if not provided)
        
    Returns
    -------
    trimesh.Trimesh
        The base mesh with ridge added via voxel union
        
    Raises
    ------
    ValueError
        If required parameters are missing
    """
    from validity.mesh.voxel_utils import voxel_union_meshes
    
    if ridge_spec is not None:
        height = ridge_spec.height
        thickness = ridge_spec.thickness
        overlap = ridge_spec.overlap
        inset = ridge_spec.inset
        resolution = ridge_spec.resolution
        voxel_pitch = ridge_spec.voxel_pitch
    
    if height is None or thickness is None:
        raise ValueError("height and thickness are required (either directly or via ridge_spec)")
    
    if overlap is None:
        overlap = 0.5 * height
    
    if voxel_pitch is None:
        voxel_pitch = min(thickness / 4, height / 4)
    
    bounds = base_mesh.bounds
    mesh_min = bounds[0]
    mesh_max = bounds[1]
    
    center_x = (mesh_min[0] + mesh_max[0]) / 2
    center_y = (mesh_min[1] + mesh_max[1]) / 2
    
    if isinstance(domain, CylinderDomain):
        radius = domain.radius
        center_x = domain.center.x
        center_y = domain.center.y
        
        if face in (FaceId.TOP, FaceId.Z_MAX):
            z_base = domain.center.z + domain.height / 2 - overlap
        else:
            z_base = domain.center.z - domain.height / 2 - height + overlap
        
        outer_radius = radius - inset
        inner_radius = outer_radius - thickness
        
        ridge = create_annular_ridge(
            outer_radius=outer_radius,
            inner_radius=inner_radius,
            height=height + overlap,
            z_base=z_base,
            center_xy=(center_x, center_y),
            resolution=resolution,
        )
        
    elif isinstance(domain, BoxDomain):
        width = domain.x_max - domain.x_min
        depth = domain.y_max - domain.y_min
        center_x = (domain.x_min + domain.x_max) / 2
        center_y = (domain.y_min + domain.y_max) / 2
        
        if face in (FaceId.TOP, FaceId.Z_MAX):
            z_base = domain.z_max - overlap
        else:
            z_base = domain.z_min - height + overlap
        
        ridge = create_frame_ridge(
            width=width - 2 * inset,
            depth=depth - 2 * inset,
            height=height + overlap,
            thickness=thickness,
            z_base=z_base,
            center_xy=(center_x, center_y),
        )
        
    else:
        width = mesh_max[0] - mesh_min[0]
        depth = mesh_max[1] - mesh_min[1]
        
        is_cylindrical = abs(width - depth) < 0.01 * max(width, depth)
        
        if is_cylindrical:
            radius = width / 2
            
            if face in (FaceId.TOP, FaceId.Z_MAX):
                z_base = mesh_max[2] - overlap
            else:
                z_base = mesh_min[2] - height + overlap
            
            outer_radius = radius - inset
            inner_radius = outer_radius - thickness
            
            ridge = create_annular_ridge(
                outer_radius=outer_radius,
                inner_radius=inner_radius,
                height=height + overlap,
                z_base=z_base,
                center_xy=(center_x, center_y),
                resolution=resolution,
            )
        else:
            if face in (FaceId.TOP, FaceId.Z_MAX):
                z_base = mesh_max[2] - overlap
            else:
                z_base = mesh_min[2] - height + overlap
            
            ridge = create_frame_ridge(
                width=width - 2 * inset,
                depth=depth - 2 * inset,
                height=height + overlap,
                thickness=thickness,
                z_base=z_base,
                center_xy=(center_x, center_y),
            )
    
    result = voxel_union_meshes(
        [base_mesh, ridge],
        pitch=voxel_pitch,
        fill=True,
        log_prefix="[add_raised_ridge] ",
    )
    
    return result
