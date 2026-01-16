"""
TransformDomain - Domain wrapper for coordinate transformations.

This module provides a TransformDomain class that wraps any Domain and applies
coordinate transformations (rotation, translation, scaling) to map points
between world and local coordinates.

Example use case: A rotated cylinder where the top face placement still
clamps to the disk correctly in the rotated coordinate system.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import numpy as np

from .domain import DomainSpec
from .types import Point3D


def _rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Create a rotation matrix from axis-angle representation.
    
    Parameters
    ----------
    axis : np.ndarray
        Unit vector representing the rotation axis.
    angle : float
        Rotation angle in radians.
    
    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    axis = axis / np.linalg.norm(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = axis
    
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c],
    ])


def _rotation_matrix_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Create a rotation matrix from Euler angles (ZYX convention).
    
    Parameters
    ----------
    roll : float
        Rotation around X axis in radians.
    pitch : float
        Rotation around Y axis in radians.
    yaw : float
        Rotation around Z axis in radians.
    
    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr],
    ])


@dataclass
class TransformDomain(DomainSpec):
    """
    Domain wrapper that applies coordinate transformations.
    
    Implements the Domain interface by mapping points into base domain coordinates.
    Supports rotation, translation, and uniform scaling.
    
    Parameters
    ----------
    base_domain : DomainSpec
        The underlying domain to transform.
    translation : np.ndarray, optional
        Translation vector [tx, ty, tz]. Default [0, 0, 0].
    rotation : np.ndarray, optional
        3x3 rotation matrix. Default identity.
    scale : float, optional
        Uniform scale factor. Default 1.0.
    
    Coordinate transformation:
        world_point = scale * (rotation @ local_point) + translation
        local_point = rotation.T @ ((world_point - translation) / scale)
    
    Example
    -------
    >>> from generation.core.domain import CylinderDomain
    >>> cylinder = CylinderDomain(radius=0.005, height=0.003)
    >>> # Rotate cylinder 45 degrees around X axis
    >>> rotation = _rotation_matrix_from_euler(np.pi/4, 0, 0)
    >>> transformed = TransformDomain(cylinder, rotation=rotation)
    >>> # Top face placement still clamps to disk correctly in rotated frame
    """
    
    base_domain: DomainSpec
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    scale: float = 1.0
    
    def __post_init__(self):
        self.translation = np.array(self.translation, dtype=float)
        self.rotation = np.array(self.rotation, dtype=float)
        if self.rotation.shape != (3, 3):
            raise ValueError(f"rotation must be 3x3 matrix, got shape {self.rotation.shape}")
        if self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")
    
    def _world_to_local(self, point: np.ndarray) -> np.ndarray:
        """Transform point from world coordinates to local (base domain) coordinates."""
        return self.rotation.T @ ((point - self.translation) / self.scale)
    
    def _local_to_world(self, point: np.ndarray) -> np.ndarray:
        """Transform point from local (base domain) coordinates to world coordinates."""
        return self.scale * (self.rotation @ point) + self.translation
    
    def _point3d_to_array(self, point: Point3D) -> np.ndarray:
        """Convert Point3D to numpy array."""
        return np.array([point.x, point.y, point.z])
    
    def _array_to_point3d(self, arr: np.ndarray) -> Point3D:
        """Convert numpy array to Point3D."""
        return Point3D(float(arr[0]), float(arr[1]), float(arr[2]))
    
    def contains(self, point: Point3D) -> bool:
        """Check if a point is inside the transformed domain."""
        world_arr = self._point3d_to_array(point)
        local_arr = self._world_to_local(world_arr)
        local_point = self._array_to_point3d(local_arr)
        return self.base_domain.contains(local_point)
    
    def project_inside(self, point: Point3D) -> Point3D:
        """Project a point to the nearest point inside the transformed domain."""
        world_arr = self._point3d_to_array(point)
        local_arr = self._world_to_local(world_arr)
        local_point = self._array_to_point3d(local_arr)
        
        projected_local = self.base_domain.project_inside(local_point)
        
        projected_local_arr = self._point3d_to_array(projected_local)
        projected_world_arr = self._local_to_world(projected_local_arr)
        
        return self._array_to_point3d(projected_world_arr)
    
    def distance_to_boundary(self, point: Point3D) -> float:
        """Compute distance from point to transformed domain boundary."""
        world_arr = self._point3d_to_array(point)
        local_arr = self._world_to_local(world_arr)
        local_point = self._array_to_point3d(local_arr)
        
        local_distance = self.base_domain.distance_to_boundary(local_point)
        
        return local_distance * self.scale
    
    def sample_points(self, n_points: int, seed: Optional[int] = None) -> np.ndarray:
        """Sample random points inside the transformed domain."""
        local_points = self.base_domain.sample_points(n_points, seed)
        
        world_points = np.zeros_like(local_points)
        for i in range(len(local_points)):
            world_points[i] = self._local_to_world(local_points[i])
        
        return world_points
    
    def get_bounds(self) -> tuple:
        """Get bounding box of the transformed domain."""
        local_bounds = self.base_domain.get_bounds()
        
        local_corners = np.array([
            [local_bounds[0], local_bounds[2], local_bounds[4]],
            [local_bounds[0], local_bounds[2], local_bounds[5]],
            [local_bounds[0], local_bounds[3], local_bounds[4]],
            [local_bounds[0], local_bounds[3], local_bounds[5]],
            [local_bounds[1], local_bounds[2], local_bounds[4]],
            [local_bounds[1], local_bounds[2], local_bounds[5]],
            [local_bounds[1], local_bounds[3], local_bounds[4]],
            [local_bounds[1], local_bounds[3], local_bounds[5]],
        ])
        
        world_corners = np.array([self._local_to_world(c) for c in local_corners])
        
        min_corner = world_corners.min(axis=0)
        max_corner = world_corners.max(axis=0)
        
        return (
            min_corner[0], max_corner[0],
            min_corner[1], max_corner[1],
            min_corner[2], max_corner[2],
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "type": "transform",
            "base_domain": self.base_domain.to_dict(),
            "translation": self.translation.tolist(),
            "rotation": self.rotation.tolist(),
            "scale": self.scale,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "TransformDomain":
        """Create from dictionary."""
        from .domain import domain_from_dict
        
        return cls(
            base_domain=domain_from_dict(d["base_domain"]),
            translation=np.array(d.get("translation", [0, 0, 0])),
            rotation=np.array(d.get("rotation", np.eye(3).tolist())),
            scale=d.get("scale", 1.0),
        )
    
    def get_face_frame(self, face: str) -> Dict[str, Any]:
        """
        Get the coordinate frame for a named face in world coordinates.
        
        Parameters
        ----------
        face : str
            Face name (e.g., "top", "bottom", "+x", "-x", "+y", "-y", "+z", "-z").
        
        Returns
        -------
        dict
            Face frame with keys: origin, normal, u, v (all in world coordinates).
        """
        if not hasattr(self.base_domain, 'get_face_frame'):
            raise NotImplementedError(
                f"Base domain {type(self.base_domain).__name__} does not support get_face_frame"
            )
        
        local_frame = self.base_domain.get_face_frame(face)
        
        local_origin = np.array(local_frame["origin"])
        local_normal = np.array(local_frame["normal"])
        local_u = np.array(local_frame["u"])
        local_v = np.array(local_frame["v"])
        
        world_origin = self._local_to_world(local_origin)
        world_normal = self.rotation @ local_normal
        world_u = self.rotation @ local_u
        world_v = self.rotation @ local_v
        
        world_normal = world_normal / np.linalg.norm(world_normal)
        world_u = world_u / np.linalg.norm(world_u)
        world_v = world_v / np.linalg.norm(world_v)
        
        return {
            "origin": world_origin.tolist(),
            "normal": world_normal.tolist(),
            "u": world_u.tolist(),
            "v": world_v.tolist(),
        }
    
    @classmethod
    def from_axis_angle(
        cls,
        base_domain: DomainSpec,
        axis: Tuple[float, float, float],
        angle: float,
        translation: Tuple[float, float, float] = (0, 0, 0),
        scale: float = 1.0,
    ) -> "TransformDomain":
        """
        Create a TransformDomain from axis-angle rotation.
        
        Parameters
        ----------
        base_domain : DomainSpec
            The underlying domain to transform.
        axis : tuple
            Rotation axis as (x, y, z).
        angle : float
            Rotation angle in radians.
        translation : tuple, optional
            Translation vector. Default (0, 0, 0).
        scale : float, optional
            Uniform scale factor. Default 1.0.
        
        Returns
        -------
        TransformDomain
            The transformed domain.
        """
        rotation = _rotation_matrix_from_axis_angle(np.array(axis), angle)
        return cls(
            base_domain=base_domain,
            translation=np.array(translation),
            rotation=rotation,
            scale=scale,
        )
    
    @classmethod
    def from_euler(
        cls,
        base_domain: DomainSpec,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
        translation: Tuple[float, float, float] = (0, 0, 0),
        scale: float = 1.0,
    ) -> "TransformDomain":
        """
        Create a TransformDomain from Euler angles.
        
        Parameters
        ----------
        base_domain : DomainSpec
            The underlying domain to transform.
        roll : float, optional
            Rotation around X axis in radians. Default 0.
        pitch : float, optional
            Rotation around Y axis in radians. Default 0.
        yaw : float, optional
            Rotation around Z axis in radians. Default 0.
        translation : tuple, optional
            Translation vector. Default (0, 0, 0).
        scale : float, optional
            Uniform scale factor. Default 1.0.
        
        Returns
        -------
        TransformDomain
            The transformed domain.
        """
        rotation = _rotation_matrix_from_euler(roll, pitch, yaw)
        return cls(
            base_domain=base_domain,
            translation=np.array(translation),
            rotation=rotation,
            scale=scale,
        )


__all__ = [
    "TransformDomain",
    "_rotation_matrix_from_axis_angle",
    "_rotation_matrix_from_euler",
]
