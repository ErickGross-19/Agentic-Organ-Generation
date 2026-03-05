"""
Unified tube sweep primitive for creating tubular meshes along paths.

This module provides a single, unified function for creating tube meshes
along arbitrary paths with configurable radius schedules. It replaces
and generalizes the existing channel mesh builders.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
import logging

if TYPE_CHECKING:
    import trimesh

logger = logging.getLogger(__name__)


class RadiusScheduleType(str, Enum):
    """Types of radius schedules for tube sweeps."""
    CONSTANT = "constant"
    TAPER = "taper"
    PIECEWISE = "piecewise"
    MURRAY = "murray"


@dataclass
class RadiusSchedule:
    """
    Specification for how radius varies along a path.
    
    Supports constant, linear taper, piecewise, and Murray's Law split.
    
    JSON Schema:
    {
        "type": "constant" | "taper" | "piecewise" | "murray",
        "start_radius": float (meters),
        "end_radius": float (meters),
        "radii": [float, ...] (for piecewise),
        "positions": [float, ...] (for piecewise, 0-1 normalized),
        "murray_exponent": float (for murray),
        "branch_count": int (for murray)
    }
    """
    type: str = "constant"
    start_radius: float = 0.001  # 1mm
    end_radius: Optional[float] = None
    radii: Optional[List[float]] = None
    positions: Optional[List[float]] = None
    murray_exponent: float = 3.0
    branch_count: int = 2
    
    def __post_init__(self):
        if self.end_radius is None:
            self.end_radius = self.start_radius
    
    def get_radius_at(self, t: float) -> float:
        """
        Get radius at normalized position t (0 to 1).
        
        Parameters
        ----------
        t : float
            Normalized position along path (0 = start, 1 = end)
            
        Returns
        -------
        float
            Radius at position t
        """
        t = np.clip(t, 0.0, 1.0)
        
        if self.type == RadiusScheduleType.CONSTANT.value:
            return self.start_radius
        
        elif self.type == RadiusScheduleType.TAPER.value:
            return self.start_radius + t * (self.end_radius - self.start_radius)
        
        elif self.type == RadiusScheduleType.PIECEWISE.value:
            if self.radii is None or self.positions is None:
                return self.start_radius
            
            if len(self.radii) != len(self.positions):
                return self.start_radius
            
            # Find surrounding control points
            for i in range(len(self.positions) - 1):
                if self.positions[i] <= t <= self.positions[i + 1]:
                    # Linear interpolation between control points
                    local_t = (t - self.positions[i]) / (
                        self.positions[i + 1] - self.positions[i]
                    )
                    return self.radii[i] + local_t * (
                        self.radii[i + 1] - self.radii[i]
                    )
            
            # Outside range, use nearest
            if t < self.positions[0]:
                return self.radii[0]
            return self.radii[-1]
        
        elif self.type == RadiusScheduleType.MURRAY.value:
            # Murray's Law: r_parent^n = sum(r_child^n)
            # For uniform split: r_child = r_parent / k^(1/n)
            child_radius = self.start_radius / (
                self.branch_count ** (1.0 / self.murray_exponent)
            )
            return self.start_radius + t * (child_radius - self.start_radius)
        
        return self.start_radius
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "start_radius": self.start_radius,
            "end_radius": self.end_radius,
            "radii": self.radii,
            "positions": self.positions,
            "murray_exponent": self.murray_exponent,
            "branch_count": self.branch_count,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RadiusSchedule":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def constant(cls, radius: float) -> "RadiusSchedule":
        """Create a constant radius schedule."""
        return cls(type="constant", start_radius=radius)
    
    @classmethod
    def taper(cls, start_radius: float, end_radius: float) -> "RadiusSchedule":
        """Create a linear taper radius schedule."""
        return cls(type="taper", start_radius=start_radius, end_radius=end_radius)
    
    @classmethod
    def piecewise(
        cls,
        radii: List[float],
        positions: Optional[List[float]] = None,
    ) -> "RadiusSchedule":
        """Create a piecewise radius schedule."""
        if positions is None:
            # Distribute evenly
            positions = [i / (len(radii) - 1) for i in range(len(radii))]
        return cls(
            type="piecewise",
            start_radius=radii[0],
            end_radius=radii[-1],
            radii=radii,
            positions=positions,
        )
    
    @classmethod
    def murray(
        cls,
        parent_radius: float,
        branch_count: int = 2,
        exponent: float = 3.0,
    ) -> "RadiusSchedule":
        """Create a Murray's Law radius schedule for bifurcation."""
        child_radius = parent_radius / (branch_count ** (1.0 / exponent))
        return cls(
            type="murray",
            start_radius=parent_radius,
            end_radius=child_radius,
            murray_exponent=exponent,
            branch_count=branch_count,
        )


@dataclass
class MeshPolicy:
    """
    Policy for tube mesh generation.
    
    Controls mesh resolution and end cap behavior.
    
    JSON Schema:
    {
        "radial_sections": int,
        "path_samples": int | "auto",
        "cap_ends": bool,
        "open_at_ports": bool,
        "smooth_normals": bool
    }
    """
    radial_sections: int = 16
    path_samples: Union[int, str] = "auto"
    cap_ends: bool = True
    open_at_ports: bool = False
    smooth_normals: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "radial_sections": self.radial_sections,
            "path_samples": self.path_samples,
            "cap_ends": self.cap_ends,
            "open_at_ports": self.open_at_ports,
            "smooth_normals": self.smooth_normals,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MeshPolicy":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SweepReport:
    """Report from a tube sweep operation."""
    success: bool
    vertex_count: int = 0
    face_count: int = 0
    path_length: float = 0.0
    min_radius: float = 0.0
    max_radius: float = 0.0
    is_watertight: bool = False
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "path_length": self.path_length,
            "min_radius": self.min_radius,
            "max_radius": self.max_radius,
            "is_watertight": self.is_watertight,
            "warnings": self.warnings,
            "errors": self.errors,
            "metadata": self.metadata,
        }


def sweep_tube_along_path(
    path_pts: List[np.ndarray],
    radius_schedule: Optional[RadiusSchedule] = None,
    mesh_policy: Optional[MeshPolicy] = None,
) -> Tuple["trimesh.Trimesh", SweepReport]:
    """
    Create a tube mesh by sweeping a circular cross-section along a path.
    
    This is the unified tube sweep primitive that replaces and generalizes
    the existing channel mesh builders (straight, tapered, fang_hook).
    
    Parameters
    ----------
    path_pts : list of np.ndarray
        List of 3D points defining the path centerline
    radius_schedule : RadiusSchedule, optional
        How radius varies along the path (default: constant 1mm)
    mesh_policy : MeshPolicy, optional
        Mesh generation parameters
        
    Returns
    -------
    mesh : trimesh.Trimesh
        Generated tube mesh
    report : SweepReport
        Report with mesh statistics and any warnings
    """
    import trimesh
    
    if radius_schedule is None:
        radius_schedule = RadiusSchedule()
    if mesh_policy is None:
        mesh_policy = MeshPolicy()
    
    warnings = []
    errors = []
    
    # Validate input
    if len(path_pts) < 2:
        return trimesh.Trimesh(), SweepReport(
            success=False,
            errors=["Path must have at least 2 points"],
        )
    
    # Convert to numpy arrays
    path_pts = [np.array(p) for p in path_pts]
    
    # Compute path length and cumulative distances
    segment_lengths = []
    for i in range(len(path_pts) - 1):
        segment_lengths.append(np.linalg.norm(path_pts[i + 1] - path_pts[i]))
    
    total_length = sum(segment_lengths)
    if total_length < 1e-10:
        return trimesh.Trimesh(), SweepReport(
            success=False,
            errors=["Path has zero length"],
        )
    
    cumulative_lengths = [0.0]
    for length in segment_lengths:
        cumulative_lengths.append(cumulative_lengths[-1] + length)
    
    # Determine number of path samples
    if mesh_policy.path_samples == "auto":
        # Use approximately one sample per segment, minimum 2
        n_samples = max(len(path_pts), 2)
    else:
        n_samples = max(int(mesh_policy.path_samples), 2)
    
    # Resample path if needed
    if n_samples != len(path_pts):
        resampled_pts = _resample_path(path_pts, cumulative_lengths, n_samples)
    else:
        resampled_pts = path_pts
    
    # Compute tangent frames along path
    frames = _compute_path_frames(resampled_pts)
    
    # Generate vertices
    n_radial = mesh_policy.radial_sections
    vertices = []
    radii_used = []
    
    for i, (pt, frame) in enumerate(zip(resampled_pts, frames)):
        t = i / (len(resampled_pts) - 1) if len(resampled_pts) > 1 else 0.0
        radius = radius_schedule.get_radius_at(t)
        radii_used.append(radius)
        
        tangent, normal, binormal = frame
        
        # Generate ring of vertices
        for j in range(n_radial):
            angle = 2 * np.pi * j / n_radial
            offset = radius * (np.cos(angle) * normal + np.sin(angle) * binormal)
            vertices.append(pt + offset)
    
    vertices = np.array(vertices)
    
    # Generate faces
    faces = []
    n_rings = len(resampled_pts)
    
    for i in range(n_rings - 1):
        for j in range(n_radial):
            # Current ring indices
            v0 = i * n_radial + j
            v1 = i * n_radial + (j + 1) % n_radial
            # Next ring indices
            v2 = (i + 1) * n_radial + j
            v3 = (i + 1) * n_radial + (j + 1) % n_radial
            
            # Two triangles per quad
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])
    
    # Add end caps if requested
    if mesh_policy.cap_ends:
        # Start cap
        if not mesh_policy.open_at_ports:
            start_center_idx = len(vertices)
            vertices = np.vstack([vertices, resampled_pts[0]])
            for j in range(n_radial):
                v0 = j
                v1 = (j + 1) % n_radial
                faces.append([start_center_idx, v1, v0])
        
        # End cap
        if not mesh_policy.open_at_ports:
            end_center_idx = len(vertices)
            vertices = np.vstack([vertices, resampled_pts[-1]])
            end_ring_start = (n_rings - 1) * n_radial
            for j in range(n_radial):
                v0 = end_ring_start + j
                v1 = end_ring_start + (j + 1) % n_radial
                faces.append([end_center_idx, v0, v1])
    
    # Create mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))
    
    # Fix mesh
    mesh.merge_vertices()
    mesh.remove_unreferenced_vertices()
    
    if mesh.volume < 0:
        mesh.invert()
    
    trimesh.repair.fix_normals(mesh)
    
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        if not mesh.is_watertight:
            warnings.append("Mesh is not watertight after repair")
    
    # Build report
    report = SweepReport(
        success=True,
        vertex_count=len(mesh.vertices),
        face_count=len(mesh.faces),
        path_length=total_length,
        min_radius=min(radii_used),
        max_radius=max(radii_used),
        is_watertight=mesh.is_watertight,
        warnings=warnings,
        errors=errors,
        metadata={
            "path_points": len(path_pts),
            "resampled_points": len(resampled_pts),
            "radial_sections": n_radial,
            "radius_schedule_type": radius_schedule.type,
        },
    )
    
    return mesh, report


def _resample_path(
    path_pts: List[np.ndarray],
    cumulative_lengths: List[float],
    n_samples: int,
) -> List[np.ndarray]:
    """Resample path to have n_samples evenly spaced points."""
    total_length = cumulative_lengths[-1]
    
    resampled = []
    for i in range(n_samples):
        t = i / (n_samples - 1) if n_samples > 1 else 0.0
        target_length = t * total_length
        
        # Find segment containing target length
        for j in range(len(cumulative_lengths) - 1):
            if cumulative_lengths[j] <= target_length <= cumulative_lengths[j + 1]:
                # Interpolate within segment
                segment_length = cumulative_lengths[j + 1] - cumulative_lengths[j]
                if segment_length > 1e-10:
                    local_t = (target_length - cumulative_lengths[j]) / segment_length
                else:
                    local_t = 0.0
                
                pt = path_pts[j] + local_t * (path_pts[j + 1] - path_pts[j])
                resampled.append(pt)
                break
        else:
            # Use last point
            resampled.append(path_pts[-1].copy())
    
    return resampled


def _compute_path_frames(
    path_pts: List[np.ndarray],
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute tangent frames (tangent, normal, binormal) along path.
    
    Uses parallel transport to minimize twisting.
    """
    n = len(path_pts)
    frames = []
    
    # Compute tangents
    tangents = []
    for i in range(n):
        if i == 0:
            tangent = path_pts[1] - path_pts[0]
        elif i == n - 1:
            tangent = path_pts[-1] - path_pts[-2]
        else:
            tangent = path_pts[i + 1] - path_pts[i - 1]
        
        norm = np.linalg.norm(tangent)
        if norm > 1e-10:
            tangent = tangent / norm
        else:
            tangent = np.array([0, 0, 1])
        tangents.append(tangent)
    
    # Initialize first frame
    t0 = tangents[0]
    
    # Find initial normal perpendicular to tangent
    if abs(t0[2]) < 0.9:
        n0 = np.cross(t0, np.array([0, 0, 1]))
    else:
        n0 = np.cross(t0, np.array([1, 0, 0]))
    n0 = n0 / np.linalg.norm(n0)
    
    b0 = np.cross(t0, n0)
    frames.append((t0, n0, b0))
    
    # Propagate frame using parallel transport
    for i in range(1, n):
        t_prev = tangents[i - 1]
        t_curr = tangents[i]
        n_prev = frames[-1][1]
        b_prev = frames[-1][2]
        
        # Rotation axis and angle
        axis = np.cross(t_prev, t_curr)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm > 1e-10:
            axis = axis / axis_norm
            angle = np.arccos(np.clip(np.dot(t_prev, t_curr), -1, 1))
            
            # Rotate normal and binormal
            n_curr = _rotate_vector(n_prev, axis, angle)
            b_curr = _rotate_vector(b_prev, axis, angle)
        else:
            n_curr = n_prev.copy()
            b_curr = b_prev.copy()
        
        # Ensure orthonormality
        n_curr = n_curr - np.dot(n_curr, t_curr) * t_curr
        n_norm = np.linalg.norm(n_curr)
        if n_norm > 1e-10:
            n_curr = n_curr / n_norm
        
        b_curr = np.cross(t_curr, n_curr)
        
        frames.append((t_curr, n_curr, b_curr))
    
    return frames


def _rotate_vector(
    v: np.ndarray,
    axis: np.ndarray,
    angle: float,
) -> np.ndarray:
    """Rotate vector v around axis by angle (Rodrigues' formula)."""
    c = np.cos(angle)
    s = np.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)


def create_straight_channel(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    radius: float,
    segments: int = 16,
) -> "trimesh.Trimesh":
    """
    Create a straight cylindrical channel using sweep_tube_along_path.
    
    This is a convenience wrapper that maintains backward compatibility
    with the existing channel API.
    """
    path_pts = [np.array(start), np.array(end)]
    radius_schedule = RadiusSchedule.constant(radius)
    mesh_policy = MeshPolicy(radial_sections=segments)
    
    mesh, _ = sweep_tube_along_path(path_pts, radius_schedule, mesh_policy)
    return mesh


def create_tapered_channel(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    start_radius: float,
    end_radius: float,
    segments: int = 16,
    length_segments: int = 8,
) -> "trimesh.Trimesh":
    """
    Create a tapered (conical) channel using sweep_tube_along_path.
    
    This is a convenience wrapper that maintains backward compatibility
    with the existing channel API.
    """
    path_pts = [np.array(start), np.array(end)]
    radius_schedule = RadiusSchedule.taper(start_radius, end_radius)
    mesh_policy = MeshPolicy(
        radial_sections=segments,
        path_samples=length_segments + 1,
    )
    
    mesh, _ = sweep_tube_along_path(path_pts, radius_schedule, mesh_policy)
    return mesh


def create_curved_channel(
    path_pts: List[Tuple[float, float, float]],
    radius: float,
    segments: int = 16,
) -> "trimesh.Trimesh":
    """
    Create a curved channel along a path using sweep_tube_along_path.
    
    This is a convenience wrapper for creating channels along arbitrary paths.
    """
    pts = [np.array(p) for p in path_pts]
    radius_schedule = RadiusSchedule.constant(radius)
    mesh_policy = MeshPolicy(radial_sections=segments)
    
    mesh, _ = sweep_tube_along_path(pts, radius_schedule, mesh_policy)
    return mesh


__all__ = [
    "sweep_tube_along_path",
    "RadiusSchedule",
    "RadiusScheduleType",
    "MeshPolicy",
    "SweepReport",
    "create_straight_channel",
    "create_tapered_channel",
    "create_curved_channel",
]
