"""
Oriented Bounding Box (OBB) utilities for mesh face detection.

This module provides OBB computation using PCA for automatic face detection
on mesh domains. When user-defined faces are not provided, OBB can be used
to define implicit faces like "top", "bottom", "+x", "-x", etc.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
import numpy as np


@dataclass
class OBB:
    """
    Oriented Bounding Box computed from mesh vertices.
    
    Attributes
    ----------
    center : np.ndarray
        Center of the OBB in world coordinates.
    axes : np.ndarray
        3x3 matrix where each row is a principal axis direction.
        axes[0] is the primary axis (largest extent).
    extents : np.ndarray
        Half-extents along each axis.
    """
    
    center: np.ndarray
    axes: np.ndarray
    extents: np.ndarray
    
    @classmethod
    def from_vertices(cls, vertices: np.ndarray) -> "OBB":
        """
        Compute OBB from mesh vertices using PCA.
        
        Parameters
        ----------
        vertices : np.ndarray
            Nx3 array of vertex positions.
        
        Returns
        -------
        OBB
            Oriented bounding box.
        """
        center = np.mean(vertices, axis=0)
        
        centered = vertices - center
        
        cov = np.cov(centered.T)
        
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        
        axes = eigenvectors.T
        
        projected = centered @ eigenvectors
        
        min_proj = np.min(projected, axis=0)
        max_proj = np.max(projected, axis=0)
        
        extents = (max_proj - min_proj) / 2
        
        center_offset = (max_proj + min_proj) / 2
        center = center + eigenvectors @ center_offset
        
        return cls(center=center, axes=axes, extents=extents)
    
    def get_face_frame(self, face: str) -> Dict[str, Any]:
        """
        Get coordinate frame for a named face.
        
        Parameters
        ----------
        face : str
            Face name: "top", "bottom", "+x", "-x", "+y", "-y", "+z", "-z",
            or OBB-relative: "+axis0", "-axis0", "+axis1", "-axis1", "+axis2", "-axis2".
        
        Returns
        -------
        dict
            Face frame with keys: origin, normal, u, v.
        """
        axis_map = {
            "+x": (0, 1),
            "-x": (0, -1),
            "+y": (1, 1),
            "-y": (1, -1),
            "+z": (2, 1),
            "-z": (2, -1),
            "top": (2, 1),
            "bottom": (2, -1),
            "+axis0": (0, 1),
            "-axis0": (0, -1),
            "+axis1": (1, 1),
            "-axis1": (1, -1),
            "+axis2": (2, 1),
            "-axis2": (2, -1),
        }
        
        if face not in axis_map:
            raise ValueError(f"Unknown face '{face}'. Valid faces: {list(axis_map.keys())}")
        
        axis_idx, sign = axis_map[face]
        
        normal = sign * self.axes[axis_idx]
        
        u_idx = (axis_idx + 1) % 3
        v_idx = (axis_idx + 2) % 3
        u = self.axes[u_idx]
        v = self.axes[v_idx]
        
        origin = self.center + sign * self.extents[axis_idx] * self.axes[axis_idx]
        
        return {
            "origin": origin.tolist(),
            "normal": normal.tolist(),
            "u": u.tolist(),
            "v": v.tolist(),
            "extent_u": float(self.extents[u_idx]),
            "extent_v": float(self.extents[v_idx]),
        }
    
    def get_all_face_frames(self) -> Dict[str, Dict[str, Any]]:
        """
        Get coordinate frames for all six faces.
        
        Returns
        -------
        dict
            Dictionary mapping face names to face frames.
        """
        faces = ["+axis0", "-axis0", "+axis1", "-axis1", "+axis2", "-axis2"]
        return {face: self.get_face_frame(face) for face in faces}
    
    def project_to_face(
        self,
        point: np.ndarray,
        face: str,
        mesh=None,
    ) -> Tuple[np.ndarray, bool]:
        """
        Project a point onto a face.
        
        Parameters
        ----------
        point : np.ndarray
            Point to project.
        face : str
            Face name.
        mesh : trimesh.Trimesh, optional
            If provided, ray cast to mesh surface for more accurate projection.
        
        Returns
        -------
        tuple
            (projected_point, success)
        """
        frame = self.get_face_frame(face)
        origin = np.array(frame["origin"])
        normal = np.array(frame["normal"])
        
        if mesh is not None:
            try:
                ray_origins = point.reshape(1, 3)
                ray_directions = (-normal).reshape(1, 3)
                
                locations, index_ray, index_tri = mesh.ray.intersects_location(
                    ray_origins=ray_origins,
                    ray_directions=ray_directions,
                )
                
                if len(locations) > 0:
                    return locations[0], True
            except Exception:
                pass
        
        t = np.dot(origin - point, normal) / np.dot(normal, normal)
        projected = point + t * normal
        
        return projected, True
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "center": self.center.tolist(),
            "axes": self.axes.tolist(),
            "extents": self.extents.tolist(),
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "OBB":
        """Create from dictionary."""
        return cls(
            center=np.array(d["center"]),
            axes=np.array(d["axes"]),
            extents=np.array(d["extents"]),
        )


def compute_mesh_obb(mesh) -> OBB:
    """
    Compute OBB for a trimesh mesh.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to compute OBB for.
    
    Returns
    -------
    OBB
        Oriented bounding box.
    """
    return OBB.from_vertices(mesh.vertices)


def get_mesh_face_frame(
    mesh,
    face: str,
    user_faces: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Get face frame for a mesh, using user-defined faces if available,
    otherwise falling back to OBB implicit faces.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh.
    face : str
        Face name.
    user_faces : dict, optional
        User-defined face planes.
    
    Returns
    -------
    dict
        Face frame with keys: origin, normal, u, v.
    """
    if user_faces is not None and face in user_faces:
        return user_faces[face]
    
    obb = compute_mesh_obb(mesh)
    
    import warnings
    warnings.warn(
        f"Using OBB implicit face approximation for face '{face}'. "
        "For precise control, provide user-defined faces."
    )
    
    return obb.get_face_frame(face)


__all__ = [
    "OBB",
    "compute_mesh_obb",
    "get_mesh_face_frame",
]
