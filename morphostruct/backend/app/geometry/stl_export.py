"""
STL export utilities for manifold3d objects.

Provides conversion from manifold meshes to STL binary/ASCII formats
and helper functions for mesh data extraction.
"""

import struct
import base64
import numpy as np
from typing import Any, Dict, Tuple, Optional


def manifold_to_stl_binary(manifold: Any) -> bytes:
    """
    Convert manifold to binary STL format.

    Binary STL format:
    - 80 bytes: header (unused)
    - 4 bytes: uint32 triangle count
    - For each triangle:
        - 12 bytes: float32[3] normal vector
        - 36 bytes: float32[9] vertices (3 vertices x 3 coords)
        - 2 bytes: uint16 attribute byte count (unused, set to 0)

    Args:
        manifold: manifold3d Manifold object

    Returns:
        Binary STL data as bytes
    """
    mesh = manifold.to_mesh()
    # tri_verts contains indices into vert_properties, not actual coordinates
    verts = np.array(mesh.vert_properties)[:, :3]  # Get vertex positions (first 3 columns)
    tri_indices = np.array(mesh.tri_verts)  # Triangle vertex indices (Nx3)
    tri_verts = verts[tri_indices]  # Index into vertices to get triangle coords (Nx3x3)

    # Calculate normals (cross product of edges)
    v0, v1, v2 = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
    edge1 = v1 - v0
    edge2 = v2 - v0
    normals = np.cross(edge1, edge2)

    # Normalize (handle degenerate triangles)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normals = normals / norms

    # Build STL binary
    num_triangles = len(tri_verts)
    header = b'\x00' * 80  # 80-byte header
    stl_data = bytearray(header)
    stl_data += struct.pack('<I', num_triangles)

    for i in range(num_triangles):
        # Normal (3 floats)
        stl_data += struct.pack('<fff', *normals[i])
        # 3 vertices (9 floats total)
        for v in tri_verts[i]:
            stl_data += struct.pack('<fff', *v)
        # Attribute byte count (unused)
        stl_data += struct.pack('<H', 0)

    return bytes(stl_data)


def manifold_to_stl_ascii(manifold: Any) -> str:
    """
    Convert manifold to ASCII STL format.

    ASCII STL format:
    solid name
      facet normal ni nj nk
        outer loop
          vertex v1x v1y v1z
          vertex v2x v2y v2z
          vertex v3x v3y v3z
        endloop
      endfacet
    endsolid name

    Args:
        manifold: manifold3d Manifold object

    Returns:
        ASCII STL string
    """
    mesh = manifold.to_mesh()
    # tri_verts contains indices into vert_properties, not actual coordinates
    verts = np.array(mesh.vert_properties)[:, :3]  # Get vertex positions (first 3 columns)
    tri_indices = np.array(mesh.tri_verts)  # Triangle vertex indices (Nx3)
    tri_verts = verts[tri_indices]  # Index into vertices to get triangle coords (Nx3x3)

    # Calculate normals
    v0, v1, v2 = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
    edge1 = v1 - v0
    edge2 = v2 - v0
    normals = np.cross(edge1, edge2)

    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals = normals / norms

    # Build ASCII STL
    lines = ["solid scaffold"]
    for i in range(len(tri_verts)):
        n = normals[i]
        lines.append(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}")
        lines.append("    outer loop")
        for v in tri_verts[i]:
            lines.append(f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid scaffold")

    return "\n".join(lines)


def manifold_to_mesh_dict(manifold: Any) -> Dict[str, Any]:
    """
    Convert manifold to dict suitable for JSON response.

    The mesh data is flattened for efficient transfer:
    - vertices: [x0, y0, z0, x1, y1, z1, ...]
    - normals: [nx0, ny0, nz0, nx1, ny1, nz1, ...]
    - indices: [i0, i1, i2, i3, i4, i5, ...] (triangle indices)

    Args:
        manifold: manifold3d Manifold object

    Returns:
        Dictionary with vertices, indices, normals, vertex_count, triangle_count
    """
    mesh = manifold.to_mesh()

    # tri_verts contains indices into vert_properties, not actual coordinates
    verts = np.array(mesh.vert_properties)[:, :3]  # Get vertex positions (first 3 columns)
    tri_indices = np.array(mesh.tri_verts)  # Triangle vertex indices (Nx3)

    if len(tri_indices) == 0:
        return {
            'vertices': [],
            'normals': [],
            'indices': [],
            'vertex_count': 0,
            'triangle_count': 0
        }

    # Index into vertices to get triangle coords (Nx3x3)
    tri_verts = verts[tri_indices]
    num_triangles = len(tri_verts)

    # Calculate per-triangle normals
    v0, v1, v2 = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normals = np.cross(edge1, edge2)

    # Normalize
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    face_normals = face_normals / norms

    # Expand normals to per-vertex (same normal for all 3 vertices of each triangle)
    # Shape: [N, 3, 3] - N triangles, 3 vertices, 3 normal components
    vertex_normals = np.repeat(face_normals[:, np.newaxis, :], 3, axis=1)

    # Flatten vertices and normals
    vertices = tri_verts.reshape(-1).tolist()
    normals = vertex_normals.reshape(-1).tolist()

    # Create indices (0, 1, 2, 3, 4, 5, ...)
    indices = list(range(num_triangles * 3))

    return {
        'vertices': vertices,
        'normals': normals,
        'indices': indices,
        'vertex_count': num_triangles * 3,
        'triangle_count': num_triangles
    }


def get_bounding_box(manifold: Any) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Get bounding box of a manifold.

    Args:
        manifold: manifold3d Manifold object

    Returns:
        Tuple of (min_point, max_point) where each point is (x, y, z)
    """
    mesh = manifold.to_mesh()

    # vert_properties contains actual vertex coordinates, tri_verts contains indices
    verts = np.array(mesh.vert_properties)[:, :3]  # Get vertex positions (first 3 columns)

    if len(verts) == 0:
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    min_pt = tuple(verts.min(axis=0).tolist())
    max_pt = tuple(verts.max(axis=0).tolist())

    return (min_pt, max_pt)


def stl_to_base64(stl_bytes: bytes) -> str:
    """
    Encode STL bytes to base64 string.

    Args:
        stl_bytes: Binary STL data

    Returns:
        Base64-encoded string
    """
    return base64.b64encode(stl_bytes).decode('ascii')
