"""
Geometric utilities for liver domain and spatial queries.
"""

import numpy as np
from typing import Tuple
from .config import LiverGeometryConfig


class LiverDomain:
    """Represents the liver as an ellipsoidal domain."""
    
    def __init__(self, config: LiverGeometryConfig):
        self.config = config
        self.a = config.semi_axis_a
        self.b = config.semi_axis_b
        self.c = config.semi_axis_c
        
        self.meeting_shell_min = config.meeting_shell_center_x - config.meeting_shell_thickness / 2
        self.meeting_shell_max = config.meeting_shell_center_x + config.meeting_shell_thickness / 2
    
    def inside_liver(self, point: np.ndarray) -> bool:
        """Check if a point is inside the ellipsoidal liver domain."""
        x, y, z = point
        return (x**2 / self.a**2 + y**2 / self.b**2 + z**2 / self.c**2) <= 1.0
    
    def distance_to_boundary(self, point: np.ndarray) -> float:
        """
        Approximate distance from point to liver boundary.
        
        For an ellipsoid, this is approximate but sufficient for growth bias.
        """
        x, y, z = point
        
        normalized_dist = np.sqrt(x**2 / self.a**2 + y**2 / self.b**2 + z**2 / self.c**2)
        
        if normalized_dist >= 1.0:
            return 0.0  # Outside or on boundary
        
        r_local = np.sqrt(x**2 + y**2 + z**2)
        if r_local < 1e-10:
            return min(self.a, self.b, self.c)
        
        direction = point / r_local
        
        dx, dy, dz = direction
        t_surface = 1.0 / np.sqrt(dx**2 / self.a**2 + dy**2 / self.b**2 + dz**2 / self.c**2)
        surface_point = t_surface * direction
        
        return float(np.linalg.norm(surface_point - point))
    
    def get_root_position(self, tree_type: str) -> np.ndarray:
        """Get the root position for arterial or venous tree."""
        if tree_type == "arterial":
            frac = self.config.arterial_root_position
        elif tree_type == "venous":
            frac = self.config.venous_root_position
        else:
            raise ValueError(f"Unknown tree type: {tree_type}")
        
        return np.array([
            frac[0] * self.a,
            frac[1] * self.b,
            frac[2] * self.c,
        ])
    
    def get_initial_direction(self, tree_type: str) -> np.ndarray:
        """Get initial growth direction for a tree."""
        if tree_type == "arterial":
            direction = np.array([1.0, 0.0, 0.0])
        elif tree_type == "venous":
            direction = np.array([-1.0, 0.0, 0.0])
        else:
            raise ValueError(f"Unknown tree type: {tree_type}")
        
        return direction
    
    def in_meeting_shell(self, point: np.ndarray) -> bool:
        """Check if point is in the meeting shell region."""
        x = point[0]
        return self.meeting_shell_min <= x <= self.meeting_shell_max
    
    def beyond_meeting_shell(self, point: np.ndarray, tree_type: str) -> bool:
        """Check if point has crossed beyond the meeting shell for this tree."""
        x = point[0]
        
        if tree_type == "arterial":
            return x > self.meeting_shell_max
        elif tree_type == "venous":
            return x < self.meeting_shell_min
        else:
            raise ValueError(f"Unknown tree type: {tree_type}")
    
    def center_direction(self, point: np.ndarray) -> np.ndarray:
        """Get unit vector pointing towards liver center from point."""
        if np.linalg.norm(point) < 1e-10:
            return np.array([0.0, 0.0, 0.0])
        
        direction = -point / np.linalg.norm(point)
        return direction


def rotate_vector(vector: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """
    Rotate a vector around an axis by angle (Rodrigues' rotation formula).
    
    Parameters
    ----------
    vector : (3,) array
        Vector to rotate
    axis : (3,) array
        Rotation axis (will be normalized)
    angle_rad : float
        Rotation angle in radians
    
    Returns
    -------
    rotated : (3,) array
        Rotated vector
    """
    axis = axis / np.linalg.norm(axis)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    rotated = (
        vector * cos_angle +
        np.cross(axis, vector) * sin_angle +
        axis * np.dot(axis, vector) * (1 - cos_angle)
    )
    
    return rotated


def get_perpendicular_vector(vector: np.ndarray) -> np.ndarray:
    """Get a vector perpendicular to the input vector."""
    vector = vector / np.linalg.norm(vector)
    
    if abs(vector[0]) < 0.9:
        other = np.array([1.0, 0.0, 0.0])
    else:
        other = np.array([0.0, 1.0, 0.0])
    
    perp = np.cross(vector, other)
    perp = perp / np.linalg.norm(perp)
    
    return perp
