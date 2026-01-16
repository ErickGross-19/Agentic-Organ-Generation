"""
Geometric domain specifications for vascular networks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np
from .types import Point3D


class DomainSpec(ABC):
    """Abstract base class for geometric domains."""

    @abstractmethod
    def contains(self, point: Point3D) -> bool:
        """Check if a point is inside the domain."""
        pass

    @abstractmethod
    def project_inside(self, point: Point3D) -> Point3D:
        """Project a point to the nearest point inside the domain."""
        pass

    @abstractmethod
    def distance_to_boundary(self, point: Point3D) -> float:
        """Compute distance from point to domain boundary (unsigned)."""
        pass

    def signed_distance(self, point: Point3D) -> float:
        """
        Compute signed distance from point to domain boundary.
        
        Returns negative values for points inside the domain,
        zero on the boundary, and positive values outside.
        
        Default implementation uses contains() and distance_to_boundary().
        Subclasses may override for more efficient implementations.
        
        Parameters
        ----------
        point : Point3D
            Point to check
            
        Returns
        -------
        float
            Signed distance (negative inside, positive outside)
        """
        dist = self.distance_to_boundary(point)
        if self.contains(point):
            return -dist
        return dist

    @abstractmethod
    def sample_points(self, n_points: int, seed: Optional[int] = None) -> np.ndarray:
        """Sample random points inside the domain."""
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        pass

    @abstractmethod
    def get_bounds(self) -> tuple:
        """Get bounding box (min_x, max_x, min_y, max_y, min_z, max_z)."""
        pass


@dataclass
class EllipsoidDomain(DomainSpec):
    """Ellipsoidal domain (e.g., for liver)."""

    semi_axis_a: float  # x-axis
    semi_axis_b: float  # y-axis
    semi_axis_c: float  # z-axis
    center: Point3D = None

    def __post_init__(self):
        if self.center is None:
            self.center = Point3D(0.0, 0.0, 0.0)

    def contains(self, point: Point3D) -> bool:
        """Check if point is inside ellipsoid."""
        dx = point.x - self.center.x
        dy = point.y - self.center.y
        dz = point.z - self.center.z

        normalized = (
            (dx / self.semi_axis_a) ** 2 +
            (dy / self.semi_axis_b) ** 2 +
            (dz / self.semi_axis_c) ** 2
        )
        return normalized <= 1.0

    def project_inside(self, point: Point3D, margin: Optional[float] = None) -> Point3D:
        """
        Project point to nearest point inside ellipsoid.
        
        H1 FIX: Uses margin parameter instead of hardcoded 0.99 factor.
        
        Parameters
        ----------
        point : Point3D
            Point to project
        margin : float, optional
            Margin from boundary. If None, uses 0.1% of smallest semi-axis.
        """
        if self.contains(point):
            return point

        dx = point.x - self.center.x
        dy = point.y - self.center.y
        dz = point.z - self.center.z

        r = np.sqrt(dx**2 + dy**2 + dz**2)
        if r < 1e-10:
            return self.center

        direction = np.array([dx / r, dy / r, dz / r])

        t = 1.0 / np.sqrt(
            (direction[0] / self.semi_axis_a) ** 2 +
            (direction[1] / self.semi_axis_b) ** 2 +
            (direction[2] / self.semi_axis_c) ** 2
        )

        if margin is None:
            smallest_axis = min(self.semi_axis_a, self.semi_axis_b, self.semi_axis_c)
            margin = smallest_axis * 0.001
        
        margin_factor = 1.0 - (margin / min(self.semi_axis_a, self.semi_axis_b, self.semi_axis_c))
        t *= max(0.9, margin_factor)

        return Point3D(
            self.center.x + t * direction[0],
            self.center.y + t * direction[1],
            self.center.z + t * direction[2],
        )

    def distance_to_boundary(self, point: Point3D) -> float:
        """Approximate distance to boundary."""
        dx = point.x - self.center.x
        dy = point.y - self.center.y
        dz = point.z - self.center.z

        r_local = np.sqrt(dx**2 + dy**2 + dz**2)
        if r_local < 1e-10:
            return min(self.semi_axis_a, self.semi_axis_b, self.semi_axis_c)

        direction = np.array([dx / r_local, dy / r_local, dz / r_local])

        t_surface = 1.0 / np.sqrt(
            (direction[0] / self.semi_axis_a) ** 2 +
            (direction[1] / self.semi_axis_b) ** 2 +
            (direction[2] / self.semi_axis_c) ** 2
        )

        surface_point = np.array([
            self.center.x + t_surface * direction[0],
            self.center.y + t_surface * direction[1],
            self.center.z + t_surface * direction[2],
        ])

        point_arr = point.to_array()
        return float(np.linalg.norm(surface_point - point_arr))

    def sample_points(self, n_points: int, seed: Optional[int] = None) -> np.ndarray:
        """Sample random points uniformly inside ellipsoid."""
        rng = np.random.default_rng(seed)

        points = []
        while len(points) < n_points:
            x = rng.uniform(-self.semi_axis_a, self.semi_axis_a)
            y = rng.uniform(-self.semi_axis_b, self.semi_axis_b)
            z = rng.uniform(-self.semi_axis_c, self.semi_axis_c)

            point = Point3D(
                self.center.x + x,
                self.center.y + y,
                self.center.z + z,
            )

            if self.contains(point):
                points.append([point.x, point.y, point.z])

        return np.array(points)

    def get_bounds(self) -> tuple:
        """Get bounding box."""
        return (
            self.center.x - self.semi_axis_a,
            self.center.x + self.semi_axis_a,
            self.center.y - self.semi_axis_b,
            self.center.y + self.semi_axis_b,
            self.center.z - self.semi_axis_c,
            self.center.z + self.semi_axis_c,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "type": "ellipsoid",
            "semi_axis_a": self.semi_axis_a,
            "semi_axis_b": self.semi_axis_b,
            "semi_axis_c": self.semi_axis_c,
            "center": self.center.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EllipsoidDomain":
        """Create from dictionary."""
        return cls(
            semi_axis_a=d["semi_axis_a"],
            semi_axis_b=d["semi_axis_b"],
            semi_axis_c=d["semi_axis_c"],
            center=Point3D.from_dict(d["center"]),
        )


@dataclass
class BoxDomain(DomainSpec):
    """Rectangular box domain."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def __post_init__(self):
        """Validate box dimensions."""
        if self.x_min >= self.x_max:
            raise ValueError(f"x_min ({self.x_min}) must be less than x_max ({self.x_max})")
        if self.y_min >= self.y_max:
            raise ValueError(f"y_min ({self.y_min}) must be less than y_max ({self.y_max})")
        if self.z_min >= self.z_max:
            raise ValueError(f"z_min ({self.z_min}) must be less than z_max ({self.z_max})")

    def contains(self, point: Point3D) -> bool:
        """Check if point is inside box."""
        return (
            self.x_min <= point.x <= self.x_max and
            self.y_min <= point.y <= self.y_max and
            self.z_min <= point.z <= self.z_max
        )

    def project_inside(self, point: Point3D, margin: Optional[float] = None) -> Point3D:
        """
        Project point to nearest point inside box.
        
        Parameters
        ----------
        point : Point3D
            Point to project
        margin : float, optional
            Margin from boundary. If None, uses 0.1% of smallest dimension.
        """
        if self.contains(point):
            return point

        if margin is None:
            # Use 0.1% of smallest dimension instead of hardcoded 1mm
            smallest_dim = min(
                self.x_max - self.x_min,
                self.y_max - self.y_min,
                self.z_max - self.z_min,
            )
            margin = smallest_dim * 0.001
        
        x = np.clip(point.x, self.x_min + margin, self.x_max - margin)
        y = np.clip(point.y, self.y_min + margin, self.y_max - margin)
        z = np.clip(point.z, self.z_min + margin, self.z_max - margin)

        return Point3D(x, y, z)

    def distance_to_boundary(self, point: Point3D) -> float:
        """Compute distance to nearest box face."""
        dist_x_min = abs(point.x - self.x_min)
        dist_x_max = abs(point.x - self.x_max)
        dist_y_min = abs(point.y - self.y_min)
        dist_y_max = abs(point.y - self.y_max)
        dist_z_min = abs(point.z - self.z_min)
        dist_z_max = abs(point.z - self.z_max)

        return float(min(dist_x_min, dist_x_max, dist_y_min, dist_y_max, dist_z_min, dist_z_max))

    def sample_points(self, n_points: int, seed: Optional[int] = None) -> np.ndarray:
        """Sample random points uniformly inside box."""
        rng = np.random.default_rng(seed)

        x = rng.uniform(self.x_min, self.x_max, n_points)
        y = rng.uniform(self.y_min, self.y_max, n_points)
        z = rng.uniform(self.z_min, self.z_max, n_points)

        return np.column_stack([x, y, z])

    def get_bounds(self) -> tuple:
        """Get bounding box (same as box itself)."""
        return (
            self.x_min, self.x_max,
            self.y_min, self.y_max,
            self.z_min, self.z_max,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "type": "box",
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "z_min": self.z_min,
            "z_max": self.z_max,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BoxDomain":
        """Create from dictionary."""
        return cls(
            x_min=d["x_min"],
            x_max=d["x_max"],
            y_min=d["y_min"],
            y_max=d["y_max"],
            z_min=d["z_min"],
            z_max=d["z_max"],
        )

    @classmethod
    def from_center_and_size(cls, center, width: float, height: float, depth: float) -> "BoxDomain":
        """
        Create box from center point and dimensions.

        Parameters
        ----------
        center : Point3D or tuple or list
            Center point as Point3D, tuple (x, y, z), or list [x, y, z]
        width : float
            Width in x direction
        height : float
            Height in y direction
        depth : float
            Depth in z direction
        """
        # Normalize center to Point3D
        if isinstance(center, (tuple, list)) and len(center) >= 3:
            cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        elif hasattr(center, 'x') and hasattr(center, 'y') and hasattr(center, 'z'):
            cx, cy, cz = center.x, center.y, center.z
        else:
            raise TypeError(f"center must be Point3D, tuple, or list, got {type(center)}")

        return cls(
            x_min=cx - width / 2,
            x_max=cx + width / 2,
            y_min=cy - height / 2,
            y_max=cy + height / 2,
            z_min=cz - depth / 2,
            z_max=cz + depth / 2,
        )


@dataclass
class CylinderDomain(DomainSpec):
    """Cylindrical domain aligned with Z-axis."""

    radius: float
    height: float
    center: Point3D = None

    def __post_init__(self):
        if self.center is None:
            self.center = Point3D(0.0, 0.0, 0.0)
        if self.radius <= 0:
            raise ValueError(f"radius ({self.radius}) must be positive")
        if self.height <= 0:
            raise ValueError(f"height ({self.height}) must be positive")

    def contains(self, point: Point3D) -> bool:
        """Check if point is inside cylinder."""
        dx = point.x - self.center.x
        dy = point.y - self.center.y
        dz = point.z - self.center.z

        r_xy = np.sqrt(dx**2 + dy**2)
        half_height = self.height / 2

        return r_xy <= self.radius and abs(dz) <= half_height

    def project_inside(self, point: Point3D, margin: Optional[float] = None) -> Point3D:
        """
        Project point to nearest point inside cylinder.
        
        Parameters
        ----------
        point : Point3D
            Point to project
        margin : float, optional
            Margin from boundary. If None, uses 0.1% of smallest dimension.
        """
        if self.contains(point):
            return point

        dx = point.x - self.center.x
        dy = point.y - self.center.y
        dz = point.z - self.center.z

        r_xy = np.sqrt(dx**2 + dy**2)
        half_height = self.height / 2

        if margin is None:
            # Use 0.1% of smallest dimension instead of hardcoded 1mm
            smallest_dim = min(self.radius, self.height)
            margin = smallest_dim * 0.001

        if r_xy > self.radius:
            scale = (self.radius - margin) / r_xy
            dx *= scale
            dy *= scale

        dz = np.clip(dz, -half_height + margin, half_height - margin)

        return Point3D(
            self.center.x + dx,
            self.center.y + dy,
            self.center.z + dz,
        )

    def distance_to_boundary(self, point: Point3D) -> float:
        """
        Compute unsigned distance to nearest cylinder surface.
        
        For a capped cylinder, we need to handle different regions:
        - Inside: minimum distance to any surface (side, top, or bottom)
        - Outside radially only: distance to side surface
        - Outside axially only: distance to cap
        - Outside both: distance to cap edge (corner)
        """
        dx = point.x - self.center.x
        dy = point.y - self.center.y
        dz = point.z - self.center.z

        r_xy = np.sqrt(dx**2 + dy**2)
        half_height = self.height / 2

        # Signed distances (positive = inside, negative = outside)
        dist_radial = self.radius - r_xy  # positive if inside radially
        dist_axial = half_height - abs(dz)  # positive if inside axially

        if dist_radial >= 0 and dist_axial >= 0:
            # Point is inside: return minimum distance to any surface
            return float(min(dist_radial, dist_axial))
        elif dist_radial < 0 and dist_axial >= 0:
            # Outside radially, inside axially: distance to side surface
            return float(-dist_radial)
        elif dist_radial >= 0 and dist_axial < 0:
            # Inside radially, outside axially: distance to cap
            return float(-dist_axial)
        else:
            # Outside both radially and axially: distance to cap edge
            return float(np.sqrt(dist_radial**2 + dist_axial**2))

    def sample_points(self, n_points: int, seed: Optional[int] = None) -> np.ndarray:
        """Sample random points uniformly inside cylinder."""
        rng = np.random.default_rng(seed)

        half_height = self.height / 2

        r = self.radius * np.sqrt(rng.uniform(0, 1, n_points))
        theta = rng.uniform(0, 2 * np.pi, n_points)
        z = rng.uniform(-half_height, half_height, n_points)

        x = self.center.x + r * np.cos(theta)
        y = self.center.y + r * np.sin(theta)
        z = self.center.z + z

        return np.column_stack([x, y, z])

    def get_bounds(self) -> tuple:
        """Get bounding box."""
        half_height = self.height / 2
        return (
            self.center.x - self.radius,
            self.center.x + self.radius,
            self.center.y - self.radius,
            self.center.y + self.radius,
            self.center.z - half_height,
            self.center.z + half_height,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "type": "cylinder",
            "radius": self.radius,
            "height": self.height,
            "center": self.center.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CylinderDomain":
        """Create from dictionary."""
        return cls(
            radius=d["radius"],
            height=d["height"],
            center=Point3D.from_dict(d["center"]),
        )


@dataclass
class MeshDomain(DomainSpec):
    """
    Mesh-based domain from STL file.

    Supports user-defined face planes for precise port placement, with OBB
    (Oriented Bounding Box) implicit faces as fallback.

    Parameters
    ----------
    mesh_path : str
        Path to the mesh file (STL, OBJ, etc.).
    faces : dict, optional
        User-defined face planes. Each face is a dict with keys:
        - origin: [x, y, z] - Face center point
        - normal: [x, y, z] - Face normal (outward)
        - u: [x, y, z] - Face U tangent direction
        - v: [x, y, z] - Face V tangent direction
        - region: dict, optional - Face region constraint:
            - type: "disk" | "rect"
            - radius: float (for disk)
            - width, height: float (for rect)

        Example:
        {
            "top": {
                "origin": [0, 0, 0.01],
                "normal": [0, 0, 1],
                "u": [1, 0, 0],
                "v": [0, 1, 0],
                "region": {"type": "disk", "radius": 0.005}
            }
        }

    If faces is None or a face is not found, OBB implicit faces are used
    as fallback (with a warning).
    """

    mesh_path: str
    faces: Optional[dict] = None
    _mesh: Optional[object] = None
    _obb: Optional[object] = None

    def __post_init__(self):
        """Load mesh on initialization."""
        try:
            import trimesh
            self._mesh = trimesh.load(self.mesh_path)
        except ImportError:
            raise ImportError("trimesh is required for MeshDomain. Install with: pip install trimesh")
        except Exception as e:
            raise ValueError(f"Failed to load mesh from {self.mesh_path}: {e}")

    def contains(self, point: Point3D) -> bool:
        """Check if point is inside mesh."""
        point_arr = point.to_array().reshape(1, 3)
        return bool(self._mesh.contains(point_arr)[0])

    def project_inside(self, point: Point3D, margin: Optional[float] = None) -> Point3D:
        """
        Project point to nearest point inside mesh.
        
        H1 FIX: Uses margin parameter instead of hardcoded 0.001.
        
        Parameters
        ----------
        point : Point3D
            Point to project
        margin : float, optional
            Margin from boundary. If None, uses 0.1% of smallest bounding box dimension.
        """
        if self.contains(point):
            return point

        point_arr = point.to_array().reshape(1, 3)
        closest, distance, triangle_id = self._mesh.nearest.on_surface(point_arr)

        normal = self._mesh.face_normals[triangle_id[0]]

        if margin is None:
            bounds = self._mesh.bounds
            smallest_dim = min(
                bounds[1][0] - bounds[0][0],
                bounds[1][1] - bounds[0][1],
                bounds[1][2] - bounds[0][2],
            )
            margin = smallest_dim * 0.001
        
        offset = -margin * normal
        inside_point = closest[0] + offset

        return Point3D.from_array(inside_point)

    def distance_to_boundary(self, point: Point3D) -> float:
        """Compute distance to mesh boundary."""
        point_arr = point.to_array().reshape(1, 3)
        closest, distance, triangle_id = self._mesh.nearest.on_surface(point_arr)
        return float(distance[0])

    def sample_points(self, n_points: int, seed: Optional[int] = None) -> np.ndarray:
        """Sample random points inside mesh."""
        rng = np.random.default_rng(seed)

        bounds = self._mesh.bounds
        min_bound = bounds[0]
        max_bound = bounds[1]

        points = []
        max_attempts = n_points * 100
        attempts = 0

        while len(points) < n_points and attempts < max_attempts:
            x = rng.uniform(min_bound[0], max_bound[0])
            y = rng.uniform(min_bound[1], max_bound[1])
            z = rng.uniform(min_bound[2], max_bound[2])

            point = Point3D(x, y, z)

            if self.contains(point):
                points.append([x, y, z])

            attempts += 1

        if len(points) < n_points:
            raise ValueError(f"Could only sample {len(points)} points after {max_attempts} attempts")

        return np.array(points)

    def get_bounds(self) -> tuple:
        """Get bounding box."""
        bounds = self._mesh.bounds
        return (
            bounds[0][0], bounds[1][0],
            bounds[0][1], bounds[1][1],
            bounds[0][2], bounds[1][2],
        )

    def _get_obb(self):
        """Get or compute OBB for implicit face detection."""
        if self._obb is None:
            from ..geometry.obb import compute_mesh_obb
            self._obb = compute_mesh_obb(self._mesh)
        return self._obb

    def get_face_frame(self, face: str) -> dict:
        """
        Get coordinate frame for a named face.

        Uses user-defined faces if available, otherwise falls back to OBB
        implicit faces (with a warning).

        Parameters
        ----------
        face : str
            Face name (e.g., "top", "bottom", "+x", "-x", "+y", "-y").

        Returns
        -------
        dict
            Face frame with keys: origin, normal, u, v.
        """
        if self.faces is not None and face in self.faces:
            return self.faces[face]

        import warnings
        warnings.warn(
            f"Using OBB implicit face approximation for face '{face}'. "
            "For precise control, provide user-defined faces."
        )

        obb = self._get_obb()
        return obb.get_face_frame(face)

    def get_face_region(self, face: str) -> Optional[dict]:
        """
        Get the region constraint for a face.

        Parameters
        ----------
        face : str
            Face name.

        Returns
        -------
        dict or None
            Region constraint with keys like type, radius (for disk),
            or width/height (for rect). None if no region defined.
        """
        if self.faces is not None and face in self.faces:
            return self.faces[face].get("region")

        obb = self._get_obb()
        frame = obb.get_face_frame(face)
        return {
            "type": "rect",
            "width": frame.get("extent_u", 0.01) * 2,
            "height": frame.get("extent_v", 0.01) * 2,
        }

    def project_to_face(self, point: Point3D, face: str) -> Point3D:
        """
        Project a point onto a face.

        First projects along the face normal, then finds the nearest point
        on the mesh surface.

        Parameters
        ----------
        point : Point3D
            Point to project.
        face : str
            Face name.

        Returns
        -------
        Point3D
            Projected point on the face.
        """
        frame = self.get_face_frame(face)
        origin = np.array(frame["origin"])
        normal = np.array(frame["normal"])
        point_arr = point.to_array()

        t = np.dot(origin - point_arr, normal) / np.dot(normal, normal)
        plane_projected = point_arr + t * normal

        try:
            ray_origins = plane_projected.reshape(1, 3)
            ray_directions = (-normal).reshape(1, 3)

            locations, index_ray, index_tri = self._mesh.ray.intersects_location(
                ray_origins=ray_origins,
                ray_directions=ray_directions,
            )

            if len(locations) > 0:
                return Point3D.from_array(locations[0])
        except Exception:
            pass

        closest, distance, triangle_id = self._mesh.nearest.on_surface(
            plane_projected.reshape(1, 3)
        )
        return Point3D.from_array(closest[0])

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        result = {
            "type": "mesh",
            "mesh_path": self.mesh_path,
        }
        if self.faces is not None:
            result["faces"] = self.faces
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "MeshDomain":
        """Create from dictionary."""
        return cls(
            mesh_path=d["mesh_path"],
            faces=d.get("faces"),
        )


def _convert_to_meters(value, input_units: str):
    """Convert a numeric value or list/tuple of values from input_units to meters.
    
    Handles:
    - Scalar numbers (int, float)
    - Lists/tuples of numbers
    - Dicts with x/y/z keys (Point3D-like)
    """
    if input_units == "m":
        return value
    elif input_units == "mm":
        scale = 1e-3  # justified: unit conversion factor (mm to meters)
    elif input_units == "um":
        scale = 1e-6  # justified: unit conversion factor (um to meters)
    else:
        return value  # Unknown units, return as-is
    
    if isinstance(value, dict):
        # Handle Point3D-like dicts with x/y/z keys
        result = dict(value)
        for key in ("x", "y", "z"):
            if key in result and isinstance(result[key], (int, float)):
                result[key] = result[key] * scale
        return result
    elif isinstance(value, (list, tuple)):
        return type(value)(v * scale if isinstance(v, (int, float)) else v for v in value)
    elif isinstance(value, (int, float)):
        return value * scale
    return value


def _normalize_dict_to_meters(d: dict, input_units: str, length_keys: set) -> dict:
    """Normalize length-valued keys in a dict from input_units to meters."""
    result = dict(d)
    for key in length_keys:
        if key in result:
            result[key] = _convert_to_meters(result[key], input_units)
    return result


def domain_from_dict(d: dict, input_units: str = "m") -> DomainSpec:
    """
    Create domain from dictionary based on type.
    
    Supports all domain types:
    - ellipsoid: EllipsoidDomain
    - box: BoxDomain
    - cylinder: CylinderDomain
    - mesh: MeshDomain
    - transform: TransformDomain (row-major 4x4 or rotation+translation)
    - composite: CompositeDomain (union/intersect/diff)
    - implicit: ImplicitDomain (JSON AST SDF)
    - sphere: SphereDomain
    - capsule: CapsuleDomain
    - frustum: FrustumDomain
    
    Parameters
    ----------
    d : dict
        Dictionary with "type" key and type-specific parameters.
    input_units : str
        Units of numeric parameters in the dict. One of "m", "mm", "um".
        Default is "m" (meters). All values are normalized to meters internally.
        
    Returns
    -------
    DomainSpec
        Compiled domain object.
        
    Raises
    ------
    ValueError
        If domain type is not recognized.
    """
    # Define which keys are length values for each domain type
    length_keys = {
        "center", "radii", "radius", "height", "min_corner", "max_corner",
        "size", "start", "end", "top_radius", "bottom_radius", "translation",
        # Box domain keys
        "x_min", "x_max", "y_min", "y_max", "z_min", "z_max",
        # Capsule/frustum keys
        "length", "radius_top", "radius_bottom",
        # Ellipsoid keys
        "semi_axis_a", "semi_axis_b", "semi_axis_c",
    }
    
    # Normalize the dict to meters
    d_normalized = _normalize_dict_to_meters(d, input_units, length_keys)
    
    domain_type = d_normalized.get("type")

    if domain_type == "ellipsoid":
        return EllipsoidDomain.from_dict(d_normalized)
    elif domain_type == "box":
        return BoxDomain.from_dict(d_normalized)
    elif domain_type == "cylinder":
        return CylinderDomain.from_dict(d_normalized)
    elif domain_type == "mesh":
        return MeshDomain.from_dict(d_normalized)
    elif domain_type == "transform":
        from .domain_transform import TransformDomain
        # For transform domains, recursively convert the base domain using original dict
        # (base_domain is nested and needs its own unit conversion)
        base_domain_dict = d.get("base_domain", {})
        base_domain = domain_from_dict(base_domain_dict, input_units)
        # Build the transform with normalized translation (but keep rotation/scale as-is)
        translation = _convert_to_meters(d.get("translation", [0, 0, 0]), input_units)
        rotation = d.get("rotation", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        scale = d.get("scale", 1.0)
        # Handle 4x4 matrix format - translation is embedded in the matrix
        if "matrix" in d:
            import numpy as np
            matrix = np.array(d["matrix"], dtype=float)
            # Scale the translation part of the matrix (column 3, rows 0-2)
            if input_units == "mm":
                matrix[:3, 3] *= 1e-3  # justified: unit conversion factor (mm to meters)
            elif input_units == "um":
                matrix[:3, 3] *= 1e-6  # justified: unit conversion factor (um to meters)
            return TransformDomain.from_matrix_4x4(base_domain, matrix)
        return TransformDomain(
            base_domain=base_domain,
            translation=translation,
            rotation=rotation,
            scale=scale,
        )
    elif domain_type == "composite":
        from .domain_composite import CompositeDomain
        # For composite domains, recursively convert children using original dict
        # (children are nested and need their own unit conversion)
        children_dicts = d.get("children", [])
        children = [domain_from_dict(child, input_units) for child in children_dicts]
        operation = d_normalized.get("operation", "union")
        return CompositeDomain(children=children, operation=operation)
    elif domain_type == "implicit":
        from .domain_implicit import ImplicitDomain
        return ImplicitDomain.from_dict(d_normalized)
    elif domain_type == "sphere":
        from .domain_primitives import SphereDomain
        return SphereDomain.from_dict(d_normalized)
    elif domain_type == "capsule":
        from .domain_primitives import CapsuleDomain
        return CapsuleDomain.from_dict(d_normalized)
    elif domain_type == "frustum":
        from .domain_primitives import FrustumDomain
        return FrustumDomain.from_dict(d_normalized)
    else:
        raise ValueError(f"Unknown domain type: {domain_type}")
