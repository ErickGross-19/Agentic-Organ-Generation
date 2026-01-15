"""
Helper functions for creating ridge geometries.

These functions create the actual mesh geometry for ridge features
on domain faces.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from typing import Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    import trimesh


def create_annular_ridge(
    outer_radius: float,
    inner_radius: float,
    height: float,
    z_base: float = 0.0,
    center_xy: Tuple[float, float] = (0.0, 0.0),
    resolution: int = 64,
) -> "trimesh.Trimesh":
    """
    Create an annular (ring-shaped) ridge mesh.
    
    This creates a hollow cylinder (tube) that can be used as a ridge
    around the perimeter of a circular face.
    
    Parameters
    ----------
    outer_radius : float
        Outer radius of the ridge (meters)
    inner_radius : float
        Inner radius of the ridge (meters)
    height : float
        Height of the ridge (meters)
    z_base : float
        Z coordinate of the base of the ridge
    center_xy : tuple
        (x, y) center coordinates
    resolution : int
        Number of segments around the circumference
        
    Returns
    -------
    trimesh.Trimesh
        Ridge mesh
    """
    import trimesh
    
    # Create outer and inner cylinders
    outer_cylinder = trimesh.creation.cylinder(
        radius=outer_radius,
        height=height,
        sections=resolution,
    )
    
    inner_cylinder = trimesh.creation.cylinder(
        radius=inner_radius,
        height=height,
        sections=resolution,
    )
    
    # Position cylinders
    z_center = z_base + height / 2
    outer_cylinder.apply_translation([center_xy[0], center_xy[1], z_center])
    inner_cylinder.apply_translation([center_xy[0], center_xy[1], z_center])
    
    # Boolean difference to create hollow ring
    try:
        ridge = outer_cylinder.difference(inner_cylinder)
        
        if ridge is None or len(ridge.vertices) == 0:
            # Fallback: create manually
            ridge = _create_annular_ridge_manual(
                outer_radius, inner_radius, height, z_base, center_xy, resolution
            )
    except Exception:
        # Fallback: create manually
        ridge = _create_annular_ridge_manual(
            outer_radius, inner_radius, height, z_base, center_xy, resolution
        )
    
    return ridge


def _create_annular_ridge_manual(
    outer_radius: float,
    inner_radius: float,
    height: float,
    z_base: float,
    center_xy: Tuple[float, float],
    resolution: int,
) -> "trimesh.Trimesh":
    """Create annular ridge manually without boolean operations."""
    import trimesh
    
    angles = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
    
    # Create vertices for outer and inner circles at top and bottom
    vertices = []
    
    # Bottom outer circle
    for angle in angles:
        x = center_xy[0] + outer_radius * np.cos(angle)
        y = center_xy[1] + outer_radius * np.sin(angle)
        vertices.append([x, y, z_base])
    
    # Bottom inner circle
    for angle in angles:
        x = center_xy[0] + inner_radius * np.cos(angle)
        y = center_xy[1] + inner_radius * np.sin(angle)
        vertices.append([x, y, z_base])
    
    # Top outer circle
    for angle in angles:
        x = center_xy[0] + outer_radius * np.cos(angle)
        y = center_xy[1] + outer_radius * np.sin(angle)
        vertices.append([x, y, z_base + height])
    
    # Top inner circle
    for angle in angles:
        x = center_xy[0] + inner_radius * np.cos(angle)
        y = center_xy[1] + inner_radius * np.sin(angle)
        vertices.append([x, y, z_base + height])
    
    vertices = np.array(vertices)
    
    # Create faces
    faces = []
    n = resolution
    
    # Outer wall faces
    for i in range(n):
        i_next = (i + 1) % n
        # Bottom outer, top outer
        faces.append([i, i_next, 2 * n + i_next])
        faces.append([i, 2 * n + i_next, 2 * n + i])
    
    # Inner wall faces (reversed winding)
    for i in range(n):
        i_next = (i + 1) % n
        # Bottom inner, top inner
        faces.append([n + i, 3 * n + i, 3 * n + i_next])
        faces.append([n + i, 3 * n + i_next, n + i_next])
    
    # Bottom face (annular)
    for i in range(n):
        i_next = (i + 1) % n
        faces.append([i, n + i, n + i_next])
        faces.append([i, n + i_next, i_next])
    
    # Top face (annular)
    for i in range(n):
        i_next = (i + 1) % n
        faces.append([2 * n + i, 2 * n + i_next, 3 * n + i_next])
        faces.append([2 * n + i, 3 * n + i_next, 3 * n + i])
    
    faces = np.array(faces)
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.merge_vertices()
    trimesh.repair.fix_normals(mesh)
    
    return mesh


def create_frame_ridge(
    width: float,
    depth: float,
    height: float,
    thickness: float,
    z_base: float = 0.0,
    center_xy: Tuple[float, float] = (0.0, 0.0),
) -> "trimesh.Trimesh":
    """
    Create a rectangular frame ridge mesh.
    
    This creates a hollow rectangular frame that can be used as a ridge
    around the perimeter of a rectangular face.
    
    Parameters
    ----------
    width : float
        Total width of the frame (x-direction, meters)
    depth : float
        Total depth of the frame (y-direction, meters)
    height : float
        Height of the ridge (meters)
    thickness : float
        Thickness of the frame walls (meters)
    z_base : float
        Z coordinate of the base of the ridge
    center_xy : tuple
        (x, y) center coordinates
        
    Returns
    -------
    trimesh.Trimesh
        Ridge mesh
    """
    import trimesh
    
    # Create outer and inner boxes
    outer_box = trimesh.creation.box(
        extents=[width, depth, height],
    )
    
    inner_width = width - 2 * thickness
    inner_depth = depth - 2 * thickness
    
    if inner_width <= 0 or inner_depth <= 0:
        # Frame is too thick, return solid box
        outer_box.apply_translation([
            center_xy[0],
            center_xy[1],
            z_base + height / 2,
        ])
        return outer_box
    
    inner_box = trimesh.creation.box(
        extents=[inner_width, inner_depth, height + 0.001],  # Slightly taller for clean boolean
    )
    
    # Position boxes
    z_center = z_base + height / 2
    outer_box.apply_translation([center_xy[0], center_xy[1], z_center])
    inner_box.apply_translation([center_xy[0], center_xy[1], z_center])
    
    # Boolean difference to create hollow frame
    try:
        ridge = outer_box.difference(inner_box)
        
        if ridge is None or len(ridge.vertices) == 0:
            # Fallback: create manually
            ridge = _create_frame_ridge_manual(
                width, depth, height, thickness, z_base, center_xy
            )
    except Exception:
        # Fallback: create manually
        ridge = _create_frame_ridge_manual(
            width, depth, height, thickness, z_base, center_xy
        )
    
    return ridge


def _create_frame_ridge_manual(
    width: float,
    depth: float,
    height: float,
    thickness: float,
    z_base: float,
    center_xy: Tuple[float, float],
) -> "trimesh.Trimesh":
    """Create frame ridge manually by combining four wall segments."""
    import trimesh
    
    cx, cy = center_xy
    hw = width / 2
    hd = depth / 2
    
    # Create four wall segments
    walls = []
    
    # Front wall (positive Y)
    front = trimesh.creation.box(extents=[width, thickness, height])
    front.apply_translation([cx, cy + hd - thickness / 2, z_base + height / 2])
    walls.append(front)
    
    # Back wall (negative Y)
    back = trimesh.creation.box(extents=[width, thickness, height])
    back.apply_translation([cx, cy - hd + thickness / 2, z_base + height / 2])
    walls.append(back)
    
    # Left wall (negative X)
    inner_depth = depth - 2 * thickness
    if inner_depth > 0:
        left = trimesh.creation.box(extents=[thickness, inner_depth, height])
        left.apply_translation([cx - hw + thickness / 2, cy, z_base + height / 2])
        walls.append(left)
    
    # Right wall (positive X)
    if inner_depth > 0:
        right = trimesh.creation.box(extents=[thickness, inner_depth, height])
        right.apply_translation([cx + hw - thickness / 2, cy, z_base + height / 2])
        walls.append(right)
    
    # Concatenate all walls
    ridge = trimesh.util.concatenate(walls)
    ridge.merge_vertices()
    
    return ridge


__all__ = [
    "create_annular_ridge",
    "create_frame_ridge",
]
