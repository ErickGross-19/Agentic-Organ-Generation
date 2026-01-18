"""
Geometric primitive types for vascular networks.
"""

from dataclasses import dataclass, field
from typing import Tuple
import numpy as np


@dataclass
class Point3D:
    """3D point in space."""
    
    x: float
    y: float
    z: float
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Point3D":
        """Create from numpy array."""
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple."""
        return (self.x, self.y, self.z)
    
    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float]) -> "Point3D":
        """Create from tuple."""
        return cls(t[0], t[1], t[2])
    
    def distance_to(self, other: "Point3D") -> float:
        """Compute Euclidean distance to another point."""
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return float(np.sqrt(dx**2 + dy**2 + dz**2))
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {"x": self.x, "y": self.y, "z": self.z}
    
    @classmethod
    def from_dict(cls, d) -> "Point3D":
        """Create from dictionary or list.
        
        Accepts either:
        - dict: {"x": float, "y": float, "z": float}
        - list/tuple: [x, y, z]
        """
        if isinstance(d, (list, tuple)):
            return cls(float(d[0]), float(d[1]), float(d[2]))
        return cls(float(d["x"]), float(d["y"]), float(d["z"]))


@dataclass
class Direction3D:
    """3D direction vector (unit vector)."""
    
    dx: float
    dy: float
    dz: float
    
    def __post_init__(self):
        """Normalize on creation."""
        self.normalize()
    
    def normalize(self) -> None:
        """Normalize to unit length."""
        length = np.sqrt(self.dx**2 + self.dy**2 + self.dz**2)
        if length < 1e-10:
            raise ValueError("Cannot normalize zero-length vector")
        self.dx /= length
        self.dy /= length
        self.dz /= length
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.dx, self.dy, self.dz])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> "Direction3D":
        """Create from numpy array (will be normalized)."""
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))
    
    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to tuple."""
        return (self.dx, self.dy, self.dz)
    
    @classmethod
    def from_tuple(cls, t: Tuple[float, float, float]) -> "Direction3D":
        """Create from tuple (will be normalized)."""
        return cls(t[0], t[1], t[2])
    
    def dot(self, other: "Direction3D") -> float:
        """Compute dot product with another direction."""
        return self.dx * other.dx + self.dy * other.dy + self.dz * other.dz
    
    def cross(self, other: "Direction3D") -> "Direction3D":
        """Compute cross product with another direction."""
        cx = self.dy * other.dz - self.dz * other.dy
        cy = self.dz * other.dx - self.dx * other.dz
        cz = self.dx * other.dy - self.dy * other.dx
        return Direction3D(cx, cy, cz)
    
    def angle_to(self, other: "Direction3D") -> float:
        """Compute angle to another direction in radians."""
        dot_product = np.clip(self.dot(other), -1.0, 1.0)
        return float(np.arccos(dot_product))
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {"dx": self.dx, "dy": self.dy, "dz": self.dz}
    
    @classmethod
    def from_dict(cls, d: dict) -> "Direction3D":
        """Create from dictionary."""
        return cls(d["dx"], d["dy"], d["dz"])


@dataclass
class TubeGeometry:
    """Geometry of a tubular vessel segment."""
    
    start: Point3D
    end: Point3D
    radius_start: float
    radius_end: float
    centerline_points: list = field(default_factory=list)
    
    def length(self) -> float:
        """Compute segment length."""
        if self.centerline_points:
            total = 0.0
            points = [self.start] + self.centerline_points + [self.end]
            for i in range(len(points) - 1):
                total += points[i].distance_to(points[i + 1])
            return total
        else:
            return self.start.distance_to(self.end)
    
    def direction(self) -> Direction3D:
        """Get overall direction from start to end."""
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        dz = self.end.z - self.start.z
        return Direction3D(dx, dy, dz)
    
    def mean_radius(self) -> float:
        """Get mean radius."""
        return (self.radius_start + self.radius_end) / 2.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
            "radius_start": self.radius_start,
            "radius_end": self.radius_end,
            "centerline_points": [p.to_dict() for p in self.centerline_points],
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "TubeGeometry":
        """Create from dictionary."""
        return cls(
            start=Point3D.from_dict(d["start"]),
            end=Point3D.from_dict(d["end"]),
            radius_start=d["radius_start"],
            radius_end=d["radius_end"],
            centerline_points=[Point3D.from_dict(p) for p in d.get("centerline_points", [])],
        )
