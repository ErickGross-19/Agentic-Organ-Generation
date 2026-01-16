"""
Domain primitives for vascular network generation.

This module provides additional geometric domain primitives:
- SphereDomain: Spherical domain
- CapsuleDomain: Capsule (cylinder with hemispherical caps)
- FrustumDomain: Truncated cone (frustum)

Each primitive supports:
- contains: Check if point is inside
- distance_to_boundary: Compute distance to surface
- sample_points: Random sampling inside domain
- get_face_frame: Get coordinate frame for named faces (top/bottom)
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np

from .domain import DomainSpec
from .types import Point3D


@dataclass
class SphereDomain(DomainSpec):
    """
    Spherical domain.
    
    Parameters
    ----------
    center : Point3D or tuple
        Center of the sphere.
    radius : float
        Radius of the sphere in meters.
    
    Faces:
        - "top": +Z pole
        - "bottom": -Z pole
        - "+x", "-x", "+y", "-y": Equatorial points
    """
    
    radius: float
    center: Point3D = None
    
    def __post_init__(self):
        if self.center is None:
            self.center = Point3D(0.0, 0.0, 0.0)
        elif isinstance(self.center, (tuple, list)):
            self.center = Point3D(self.center[0], self.center[1], self.center[2])
        if self.radius <= 0:
            raise ValueError(f"radius ({self.radius}) must be positive")
    
    def contains(self, point: Point3D) -> bool:
        """Check if point is inside sphere."""
        dx = point.x - self.center.x
        dy = point.y - self.center.y
        dz = point.z - self.center.z
        return dx*dx + dy*dy + dz*dz <= self.radius * self.radius
    
    def project_inside(self, point: Point3D) -> Point3D:
        """Project point to nearest point inside sphere."""
        if self.contains(point):
            return point
        
        dx = point.x - self.center.x
        dy = point.y - self.center.y
        dz = point.z - self.center.z
        
        r = np.sqrt(dx*dx + dy*dy + dz*dz)
        if r < 1e-10:
            return self.center
        
        margin = 0.001
        scale = (self.radius - margin) / r
        
        return Point3D(
            self.center.x + dx * scale,
            self.center.y + dy * scale,
            self.center.z + dz * scale,
        )
    
    def distance_to_boundary(self, point: Point3D) -> float:
        """Compute distance to sphere surface."""
        dx = point.x - self.center.x
        dy = point.y - self.center.y
        dz = point.z - self.center.z
        r = np.sqrt(dx*dx + dy*dy + dz*dz)
        return abs(self.radius - r)
    
    def sample_points(self, n_points: int, seed: Optional[int] = None) -> np.ndarray:
        """Sample random points uniformly inside sphere."""
        rng = np.random.default_rng(seed)
        
        u = rng.uniform(0, 1, n_points)
        r = self.radius * np.cbrt(u)
        
        theta = rng.uniform(0, 2 * np.pi, n_points)
        phi = np.arccos(2 * rng.uniform(0, 1, n_points) - 1)
        
        x = self.center.x + r * np.sin(phi) * np.cos(theta)
        y = self.center.y + r * np.sin(phi) * np.sin(theta)
        z = self.center.z + r * np.cos(phi)
        
        return np.column_stack([x, y, z])
    
    def get_bounds(self) -> tuple:
        """Get bounding box."""
        return (
            self.center.x - self.radius,
            self.center.x + self.radius,
            self.center.y - self.radius,
            self.center.y + self.radius,
            self.center.z - self.radius,
            self.center.z + self.radius,
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "type": "sphere",
            "radius": self.radius,
            "center": self.center.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "SphereDomain":
        """Create from dictionary."""
        return cls(
            radius=d["radius"],
            center=Point3D.from_dict(d["center"]) if "center" in d else None,
        )
    
    def get_face_frame(self, face: str) -> Dict[str, Any]:
        """Get coordinate frame for a named face."""
        face_map = {
            "top": (np.array([0, 0, 1]), np.array([1, 0, 0]), np.array([0, 1, 0])),
            "bottom": (np.array([0, 0, -1]), np.array([1, 0, 0]), np.array([0, -1, 0])),
            "+x": (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])),
            "-x": (np.array([-1, 0, 0]), np.array([0, -1, 0]), np.array([0, 0, 1])),
            "+y": (np.array([0, 1, 0]), np.array([-1, 0, 0]), np.array([0, 0, 1])),
            "-y": (np.array([0, -1, 0]), np.array([1, 0, 0]), np.array([0, 0, 1])),
            "+z": (np.array([0, 0, 1]), np.array([1, 0, 0]), np.array([0, 1, 0])),
            "-z": (np.array([0, 0, -1]), np.array([1, 0, 0]), np.array([0, -1, 0])),
        }
        
        if face not in face_map:
            raise ValueError(f"Unknown face '{face}'. Valid faces: {list(face_map.keys())}")
        
        normal, u, v = face_map[face]
        origin = np.array([self.center.x, self.center.y, self.center.z]) + self.radius * normal
        
        return {
            "origin": origin.tolist(),
            "normal": normal.tolist(),
            "u": u.tolist(),
            "v": v.tolist(),
        }


@dataclass
class CapsuleDomain(DomainSpec):
    """
    Capsule domain (cylinder with hemispherical caps).
    
    The capsule is aligned with a specified axis direction.
    
    Parameters
    ----------
    center : Point3D or tuple
        Center of the capsule.
    axis : tuple
        Unit vector for capsule axis direction. Default (0, 0, 1) for Z-aligned.
    radius : float
        Radius of the capsule in meters.
    length : float
        Length of the cylindrical portion (not including caps) in meters.
    
    Faces:
        - "top": Hemisphere cap in +axis direction
        - "bottom": Hemisphere cap in -axis direction
    """
    
    radius: float
    length: float
    center: Point3D = None
    axis: tuple = (0, 0, 1)
    
    def __post_init__(self):
        if self.center is None:
            self.center = Point3D(0.0, 0.0, 0.0)
        elif isinstance(self.center, (tuple, list)):
            self.center = Point3D(self.center[0], self.center[1], self.center[2])
        
        self.axis = np.array(self.axis, dtype=float)
        self.axis = self.axis / np.linalg.norm(self.axis)
        
        if self.radius <= 0:
            raise ValueError(f"radius ({self.radius}) must be positive")
        if self.length < 0:
            raise ValueError(f"length ({self.length}) must be non-negative")
    
    def _point_to_array(self, point: Point3D) -> np.ndarray:
        return np.array([point.x, point.y, point.z])
    
    def _distance_to_segment(self, point: np.ndarray) -> float:
        """Compute distance from point to the central line segment."""
        center = self._point_to_array(self.center)
        half_length = self.length / 2
        
        p1 = center - half_length * self.axis
        p2 = center + half_length * self.axis
        
        v = p2 - p1
        w = point - p1
        
        c1 = np.dot(w, v)
        if c1 <= 0:
            return np.linalg.norm(point - p1)
        
        c2 = np.dot(v, v)
        if c2 <= c1:
            return np.linalg.norm(point - p2)
        
        b = c1 / c2
        closest = p1 + b * v
        return np.linalg.norm(point - closest)
    
    def contains(self, point: Point3D) -> bool:
        """Check if point is inside capsule."""
        point_arr = self._point_to_array(point)
        dist = self._distance_to_segment(point_arr)
        return dist <= self.radius
    
    def project_inside(self, point: Point3D) -> Point3D:
        """Project point to nearest point inside capsule."""
        if self.contains(point):
            return point
        
        point_arr = self._point_to_array(point)
        center = self._point_to_array(self.center)
        half_length = self.length / 2
        
        p1 = center - half_length * self.axis
        p2 = center + half_length * self.axis
        
        v = p2 - p1
        w = point_arr - p1
        
        c1 = np.dot(w, v)
        c2 = np.dot(v, v)
        
        if c1 <= 0:
            closest_on_axis = p1
        elif c2 <= c1:
            closest_on_axis = p2
        else:
            b = c1 / c2
            closest_on_axis = p1 + b * v
        
        direction = point_arr - closest_on_axis
        dist = np.linalg.norm(direction)
        
        if dist < 1e-10:
            return Point3D(closest_on_axis[0], closest_on_axis[1], closest_on_axis[2])
        
        margin = 0.001
        direction = direction / dist
        inside_point = closest_on_axis + (self.radius - margin) * direction
        
        return Point3D(inside_point[0], inside_point[1], inside_point[2])
    
    def distance_to_boundary(self, point: Point3D) -> float:
        """Compute distance to capsule surface."""
        point_arr = self._point_to_array(point)
        dist_to_axis = self._distance_to_segment(point_arr)
        return abs(self.radius - dist_to_axis)
    
    def sample_points(self, n_points: int, seed: Optional[int] = None) -> np.ndarray:
        """Sample random points uniformly inside capsule."""
        rng = np.random.default_rng(seed)
        
        cylinder_volume = np.pi * self.radius**2 * self.length
        sphere_volume = (4/3) * np.pi * self.radius**3
        total_volume = cylinder_volume + sphere_volume
        
        p_cylinder = cylinder_volume / total_volume if total_volume > 0 else 0.5
        
        center = self._point_to_array(self.center)
        half_length = self.length / 2
        
        perp1 = np.array([1, 0, 0]) if abs(self.axis[0]) < 0.9 else np.array([0, 1, 0])
        perp1 = perp1 - np.dot(perp1, self.axis) * self.axis
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(self.axis, perp1)
        
        points = []
        while len(points) < n_points:
            if rng.random() < p_cylinder:
                t = rng.uniform(-half_length, half_length)
                r = self.radius * np.sqrt(rng.random())
                theta = rng.uniform(0, 2 * np.pi)
                
                point = center + t * self.axis + r * (np.cos(theta) * perp1 + np.sin(theta) * perp2)
            else:
                u = rng.uniform(0, 1)
                r = self.radius * np.cbrt(u)
                theta = rng.uniform(0, 2 * np.pi)
                phi = np.arccos(2 * rng.random() - 1)
                
                local_x = r * np.sin(phi) * np.cos(theta)
                local_y = r * np.sin(phi) * np.sin(theta)
                local_z = r * np.cos(phi)
                
                if local_z >= 0:
                    cap_center = center + half_length * self.axis
                else:
                    cap_center = center - half_length * self.axis
                    local_z = -local_z
                
                point = cap_center + local_z * self.axis + local_x * perp1 + local_y * perp2
            
            points.append(point)
        
        return np.array(points)
    
    def get_bounds(self) -> tuple:
        """Get bounding box."""
        center = self._point_to_array(self.center)
        half_length = self.length / 2
        
        p1 = center - half_length * self.axis
        p2 = center + half_length * self.axis
        
        min_corner = np.minimum(p1, p2) - self.radius
        max_corner = np.maximum(p1, p2) + self.radius
        
        return (
            min_corner[0], max_corner[0],
            min_corner[1], max_corner[1],
            min_corner[2], max_corner[2],
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "type": "capsule",
            "radius": self.radius,
            "length": self.length,
            "center": self.center.to_dict(),
            "axis": self.axis.tolist(),
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "CapsuleDomain":
        """Create from dictionary."""
        return cls(
            radius=d["radius"],
            length=d["length"],
            center=Point3D.from_dict(d["center"]) if "center" in d else None,
            axis=tuple(d.get("axis", (0, 0, 1))),
        )
    
    def get_face_frame(self, face: str) -> Dict[str, Any]:
        """Get coordinate frame for a named face."""
        center = self._point_to_array(self.center)
        half_length = self.length / 2
        
        perp1 = np.array([1, 0, 0]) if abs(self.axis[0]) < 0.9 else np.array([0, 1, 0])
        perp1 = perp1 - np.dot(perp1, self.axis) * self.axis
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(self.axis, perp1)
        
        if face == "top":
            origin = center + (half_length + self.radius) * self.axis
            normal = self.axis.copy()
        elif face == "bottom":
            origin = center - (half_length + self.radius) * self.axis
            normal = -self.axis.copy()
        else:
            raise ValueError(f"Unknown face '{face}'. Valid faces: ['top', 'bottom']")
        
        return {
            "origin": origin.tolist(),
            "normal": normal.tolist(),
            "u": perp1.tolist(),
            "v": perp2.tolist(),
        }


@dataclass
class FrustumDomain(DomainSpec):
    """
    Frustum domain (truncated cone).
    
    The frustum is aligned with a specified axis direction.
    
    Parameters
    ----------
    center : Point3D or tuple
        Center of the frustum (midpoint along axis).
    axis : tuple
        Unit vector for frustum axis direction. Default (0, 0, 1) for Z-aligned.
    radius_top : float
        Radius at the top (+axis direction) in meters.
    radius_bottom : float
        Radius at the bottom (-axis direction) in meters.
    height : float
        Height of the frustum in meters.
    
    Faces:
        - "top": Circular face at +axis end
        - "bottom": Circular face at -axis end
    """
    
    radius_top: float
    radius_bottom: float
    height: float
    center: Point3D = None
    axis: tuple = (0, 0, 1)
    
    def __post_init__(self):
        if self.center is None:
            self.center = Point3D(0.0, 0.0, 0.0)
        elif isinstance(self.center, (tuple, list)):
            self.center = Point3D(self.center[0], self.center[1], self.center[2])
        
        self.axis = np.array(self.axis, dtype=float)
        self.axis = self.axis / np.linalg.norm(self.axis)
        
        if self.radius_top < 0:
            raise ValueError(f"radius_top ({self.radius_top}) must be non-negative")
        if self.radius_bottom < 0:
            raise ValueError(f"radius_bottom ({self.radius_bottom}) must be non-negative")
        if self.height <= 0:
            raise ValueError(f"height ({self.height}) must be positive")
    
    def _point_to_array(self, point: Point3D) -> np.ndarray:
        return np.array([point.x, point.y, point.z])
    
    def _radius_at_height(self, t: float) -> float:
        """Get radius at normalized height t (0=bottom, 1=top)."""
        return self.radius_bottom + t * (self.radius_top - self.radius_bottom)
    
    def contains(self, point: Point3D) -> bool:
        """Check if point is inside frustum."""
        point_arr = self._point_to_array(point)
        center = self._point_to_array(self.center)
        half_height = self.height / 2
        
        bottom = center - half_height * self.axis
        
        v = point_arr - bottom
        t = np.dot(v, self.axis) / self.height
        
        if t < 0 or t > 1:
            return False
        
        radial = v - t * self.height * self.axis
        radial_dist = np.linalg.norm(radial)
        
        radius_at_t = self._radius_at_height(t)
        
        return radial_dist <= radius_at_t
    
    def project_inside(self, point: Point3D) -> Point3D:
        """Project point to nearest point inside frustum."""
        if self.contains(point):
            return point
        
        point_arr = self._point_to_array(point)
        center = self._point_to_array(self.center)
        half_height = self.height / 2
        
        bottom = center - half_height * self.axis
        
        v = point_arr - bottom
        t = np.dot(v, self.axis) / self.height
        
        margin = 0.001
        t = np.clip(t, margin / self.height, 1 - margin / self.height)
        
        radial = v - t * self.height * self.axis
        radial_dist = np.linalg.norm(radial)
        
        radius_at_t = self._radius_at_height(t)
        
        if radial_dist > radius_at_t:
            if radial_dist > 1e-10:
                radial = radial / radial_dist * (radius_at_t - margin)
            else:
                radial = np.zeros(3)
        
        inside_point = bottom + t * self.height * self.axis + radial
        
        return Point3D(inside_point[0], inside_point[1], inside_point[2])
    
    def distance_to_boundary(self, point: Point3D) -> float:
        """Compute distance to frustum surface (approximate)."""
        point_arr = self._point_to_array(point)
        center = self._point_to_array(self.center)
        half_height = self.height / 2
        
        bottom = center - half_height * self.axis
        top = center + half_height * self.axis
        
        v = point_arr - bottom
        t = np.dot(v, self.axis) / self.height
        
        dist_to_bottom = np.dot(point_arr - bottom, self.axis)
        dist_to_top = np.dot(top - point_arr, self.axis)
        
        t_clamped = np.clip(t, 0, 1)
        radial = v - t_clamped * self.height * self.axis
        radial_dist = np.linalg.norm(radial)
        radius_at_t = self._radius_at_height(t_clamped)
        dist_to_side = radius_at_t - radial_dist
        
        return float(min(abs(dist_to_bottom), abs(dist_to_top), abs(dist_to_side)))
    
    def sample_points(self, n_points: int, seed: Optional[int] = None) -> np.ndarray:
        """Sample random points uniformly inside frustum."""
        rng = np.random.default_rng(seed)
        
        center = self._point_to_array(self.center)
        half_height = self.height / 2
        bottom = center - half_height * self.axis
        
        perp1 = np.array([1, 0, 0]) if abs(self.axis[0]) < 0.9 else np.array([0, 1, 0])
        perp1 = perp1 - np.dot(perp1, self.axis) * self.axis
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(self.axis, perp1)
        
        points = []
        max_radius = max(self.radius_top, self.radius_bottom)
        
        while len(points) < n_points:
            t = rng.uniform(0, 1)
            r = max_radius * np.sqrt(rng.random())
            theta = rng.uniform(0, 2 * np.pi)
            
            radius_at_t = self._radius_at_height(t)
            
            if r <= radius_at_t:
                point = bottom + t * self.height * self.axis + r * (np.cos(theta) * perp1 + np.sin(theta) * perp2)
                points.append(point)
        
        return np.array(points)
    
    def get_bounds(self) -> tuple:
        """Get bounding box."""
        center = self._point_to_array(self.center)
        half_height = self.height / 2
        
        bottom = center - half_height * self.axis
        top = center + half_height * self.axis
        
        max_radius = max(self.radius_top, self.radius_bottom)
        
        min_corner = np.minimum(bottom, top) - max_radius
        max_corner = np.maximum(bottom, top) + max_radius
        
        return (
            min_corner[0], max_corner[0],
            min_corner[1], max_corner[1],
            min_corner[2], max_corner[2],
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "type": "frustum",
            "radius_top": self.radius_top,
            "radius_bottom": self.radius_bottom,
            "height": self.height,
            "center": self.center.to_dict(),
            "axis": self.axis.tolist(),
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "FrustumDomain":
        """Create from dictionary."""
        return cls(
            radius_top=d["radius_top"],
            radius_bottom=d["radius_bottom"],
            height=d["height"],
            center=Point3D.from_dict(d["center"]) if "center" in d else None,
            axis=tuple(d.get("axis", (0, 0, 1))),
        )
    
    def get_face_frame(self, face: str) -> Dict[str, Any]:
        """Get coordinate frame for a named face."""
        center = self._point_to_array(self.center)
        half_height = self.height / 2
        
        perp1 = np.array([1, 0, 0]) if abs(self.axis[0]) < 0.9 else np.array([0, 1, 0])
        perp1 = perp1 - np.dot(perp1, self.axis) * self.axis
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(self.axis, perp1)
        
        if face == "top":
            origin = center + half_height * self.axis
            normal = self.axis.copy()
        elif face == "bottom":
            origin = center - half_height * self.axis
            normal = -self.axis.copy()
        else:
            raise ValueError(f"Unknown face '{face}'. Valid faces: ['top', 'bottom']")
        
        return {
            "origin": origin.tolist(),
            "normal": normal.tolist(),
            "u": perp1.tolist(),
            "v": perp2.tolist(),
        }


__all__ = [
    "SphereDomain",
    "CapsuleDomain",
    "FrustumDomain",
]
