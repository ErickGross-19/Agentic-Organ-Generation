"""
Channel primitive creation for vascular network voids.

This module provides functions for creating channel geometries including
straight cylinders, tapered channels, and curved fang hooks.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, TYPE_CHECKING
import numpy as np
import logging

from ...policies import ChannelPolicy, OperationReport

if TYPE_CHECKING:
    import trimesh

logger = logging.getLogger(__name__)


def create_straight_channel(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    radius: float,
    segments: int = 16,
) -> "trimesh.Trimesh":
    """
    Create a straight cylindrical channel.
    
    Parameters
    ----------
    start : tuple
        Start position (x, y, z) in meters
    end : tuple
        End position (x, y, z) in meters
    radius : float
        Channel radius in meters
    segments : int
        Number of segments around circumference
        
    Returns
    -------
    trimesh.Trimesh
        Cylinder mesh
    """
    import trimesh
    
    start = np.array(start)
    end = np.array(end)
    
    # Calculate length and direction
    direction = end - start
    length = np.linalg.norm(direction)
    
    if length < 1e-9:
        raise ValueError("Channel length is too small")
    
    # Create cylinder along Z axis
    cylinder = trimesh.creation.cylinder(
        radius=radius,
        height=length,
        sections=segments,
    )
    
    # Calculate rotation to align with direction
    z_axis = np.array([0, 0, 1])
    direction_norm = direction / length
    
    # Rotation axis and angle
    rotation_axis = np.cross(z_axis, direction_norm)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    
    if rotation_axis_norm > 1e-9:
        rotation_axis = rotation_axis / rotation_axis_norm
        angle = np.arccos(np.clip(np.dot(z_axis, direction_norm), -1, 1))
        
        # Create rotation matrix using Rodrigues' formula
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
        # Apply rotation
        transform = np.eye(4)
        transform[:3, :3] = R
        cylinder.apply_transform(transform)
    elif np.dot(z_axis, direction_norm) < 0:
        # 180 degree rotation
        cylinder.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    
    # Translate to midpoint
    midpoint = (start + end) / 2
    cylinder.apply_translation(midpoint)
    
    return cylinder


def create_tapered_channel(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    start_radius: float,
    end_radius: float,
    segments: int = 16,
    length_segments: int = 8,
) -> "trimesh.Trimesh":
    """
    Create a tapered (conical) channel.
    
    Parameters
    ----------
    start : tuple
        Start position (x, y, z) in meters
    end : tuple
        End position (x, y, z) in meters
    start_radius : float
        Radius at start in meters
    end_radius : float
        Radius at end in meters
    segments : int
        Number of segments around circumference
    length_segments : int
        Number of segments along length
        
    Returns
    -------
    trimesh.Trimesh
        Tapered cylinder mesh
    """
    import trimesh
    
    start = np.array(start)
    end = np.array(end)
    
    direction = end - start
    length = np.linalg.norm(direction)
    
    if length < 1e-9:
        raise ValueError("Channel length is too small")
    
    direction_norm = direction / length
    
    # Create vertices for tapered cylinder
    vertices = []
    faces = []
    
    # Generate rings along the length
    for i in range(length_segments + 1):
        t = i / length_segments
        pos = start + direction * t
        radius = start_radius + (end_radius - start_radius) * t
        
        for j in range(segments):
            angle = 2 * np.pi * j / segments
            
            # Find perpendicular vectors
            if abs(direction_norm[2]) < 0.9:
                perp1 = np.cross(direction_norm, np.array([0, 0, 1]))
            else:
                perp1 = np.cross(direction_norm, np.array([1, 0, 0]))
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(direction_norm, perp1)
            
            # Calculate vertex position
            offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            vertices.append(pos + offset)
    
    vertices = np.array(vertices)
    
    # Generate faces
    for i in range(length_segments):
        for j in range(segments):
            v0 = i * segments + j
            v1 = i * segments + (j + 1) % segments
            v2 = (i + 1) * segments + j
            v3 = (i + 1) * segments + (j + 1) % segments
            
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])
    
    # Add end caps
    # Start cap
    start_center_idx = len(vertices)
    vertices = np.vstack([vertices, start])
    for j in range(segments):
        v0 = j
        v1 = (j + 1) % segments
        faces.append([start_center_idx, v1, v0])
    
    # End cap
    end_center_idx = len(vertices)
    vertices = np.vstack([vertices, end])
    end_ring_start = length_segments * segments
    for j in range(segments):
        v0 = end_ring_start + j
        v1 = end_ring_start + (j + 1) % segments
        faces.append([end_center_idx, v0, v1])
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))
    mesh.fix_normals()
    
    return mesh


def create_fang_hook(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    radius: float,
    hook_depth: float,
    hook_angle_deg: float = 90.0,
    segments_per_curve: int = 16,
    effective_radius: Optional[float] = None,
    policy: Optional[ChannelPolicy] = None,
) -> Tuple["trimesh.Trimesh", Dict[str, Any]]:
    """
    Create a curved fang hook channel.
    
    The fang hook is a curved channel that bends inward from the surface,
    useful for creating channels that don't go straight through.
    
    Parameters
    ----------
    start : tuple
        Start position (x, y, z) in meters (typically on domain surface)
    end : tuple
        End position (x, y, z) in meters (inside domain)
    radius : float
        Channel radius in meters
    hook_depth : float
        Depth of the hook curve in meters
    hook_angle_deg : float
        Angle of the hook in degrees (default: 90)
    segments_per_curve : int
        Number of segments for the curved portion
    effective_radius : float, optional
        Maximum effective radius for constraint enforcement
    policy : ChannelPolicy, optional
        Policy for additional constraints
        
    Returns
    -------
    mesh : trimesh.Trimesh
        Fang hook mesh
    meta : dict
        Metadata including any constraint modifications
    """
    import trimesh
    
    start = np.array(start)
    end = np.array(end)
    
    meta = {
        "requested_hook_depth": hook_depth,
        "actual_hook_depth": hook_depth,
        "constraint_modified": False,
    }
    
    # Check effective radius constraint
    if effective_radius is not None and policy is not None and policy.enforce_effective_radius:
        max_hook_depth = effective_radius - radius
        if hook_depth > max_hook_depth:
            meta["actual_hook_depth"] = max_hook_depth
            meta["constraint_modified"] = True
            meta["constraint_warning"] = (
                f"Hook depth reduced from {hook_depth:.6f} to {max_hook_depth:.6f} "
                f"due to effective radius constraint"
            )
            hook_depth = max_hook_depth
            logger.warning(meta["constraint_warning"])
    
    # Calculate the curve
    direction = end - start
    length = np.linalg.norm(direction)
    direction_norm = direction / length if length > 1e-9 else np.array([0, 0, -1])
    
    # Find perpendicular for hook direction
    if abs(direction_norm[2]) < 0.9:
        hook_perp = np.cross(direction_norm, np.array([0, 0, 1]))
    else:
        hook_perp = np.cross(direction_norm, np.array([1, 0, 0]))
    hook_perp = hook_perp / np.linalg.norm(hook_perp)
    
    # Generate centerline points for the hook
    hook_angle_rad = np.radians(hook_angle_deg)
    centerline_points = []
    
    # Straight section at start
    straight_length = length * 0.3
    for i in range(segments_per_curve // 3):
        t = i / (segments_per_curve // 3)
        pos = start + direction_norm * straight_length * t
        centerline_points.append(pos)
    
    # Curved section
    curve_start = start + direction_norm * straight_length
    curve_length = length * 0.4
    
    for i in range(segments_per_curve // 3):
        t = i / (segments_per_curve // 3)
        angle = t * hook_angle_rad
        
        # Position along curve
        along = curve_length * t
        lateral = hook_depth * np.sin(angle)
        
        pos = curve_start + direction_norm * along + hook_perp * lateral
        centerline_points.append(pos)
    
    # Straight section at end
    curve_end = curve_start + direction_norm * curve_length + hook_perp * hook_depth * np.sin(hook_angle_rad)
    remaining = end - curve_end
    
    for i in range(segments_per_curve // 3 + 1):
        t = i / (segments_per_curve // 3)
        pos = curve_end + remaining * t
        centerline_points.append(pos)
    
    centerline_points = np.array(centerline_points)
    
    # Create tube mesh along centerline
    mesh = _create_tube_along_centerline(centerline_points, radius, segments_per_curve)
    
    return mesh, meta


def _create_tube_along_centerline(
    centerline: np.ndarray,
    radius: float,
    segments: int = 16,
) -> "trimesh.Trimesh":
    """Create a tube mesh along a centerline."""
    import trimesh
    
    n_points = len(centerline)
    if n_points < 2:
        raise ValueError("Centerline must have at least 2 points")
    
    vertices = []
    faces = []
    
    for i in range(n_points):
        # Calculate tangent
        if i == 0:
            tangent = centerline[1] - centerline[0]
        elif i == n_points - 1:
            tangent = centerline[-1] - centerline[-2]
        else:
            tangent = centerline[i + 1] - centerline[i - 1]
        
        tangent = tangent / np.linalg.norm(tangent)
        
        # Find perpendicular vectors
        if abs(tangent[2]) < 0.9:
            perp1 = np.cross(tangent, np.array([0, 0, 1]))
        else:
            perp1 = np.cross(tangent, np.array([1, 0, 0]))
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(tangent, perp1)
        
        # Create ring of vertices
        for j in range(segments):
            angle = 2 * np.pi * j / segments
            offset = radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            vertices.append(centerline[i] + offset)
    
    vertices = np.array(vertices)
    
    # Generate faces
    for i in range(n_points - 1):
        for j in range(segments):
            v0 = i * segments + j
            v1 = i * segments + (j + 1) % segments
            v2 = (i + 1) * segments + j
            v3 = (i + 1) * segments + (j + 1) % segments
            
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])
    
    # Add end caps
    start_center_idx = len(vertices)
    vertices = np.vstack([vertices, centerline[0]])
    for j in range(segments):
        v0 = j
        v1 = (j + 1) % segments
        faces.append([start_center_idx, v1, v0])
    
    end_center_idx = len(vertices)
    vertices = np.vstack([vertices, centerline[-1]])
    end_ring_start = (n_points - 1) * segments
    for j in range(segments):
        v0 = end_ring_start + j
        v1 = end_ring_start + (j + 1) % segments
        faces.append([end_center_idx, v0, v1])
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces))
    mesh.fix_normals()
    
    return mesh


def create_channels_from_ports(
    domain: "DomainSpec",
    ports: Dict[str, Any],
    policy: Optional[ChannelPolicy] = None,
) -> Tuple["trimesh.Trimesh", Dict[str, Any]]:
    """
    Create channel meshes from port specifications.
    
    Parameters
    ----------
    domain : DomainSpec
        Domain specification
    ports : dict
        Port configuration with "inlets" and "outlets"
    policy : ChannelPolicy, optional
        Policy controlling channel creation
        
    Returns
    -------
    mesh : trimesh.Trimesh
        Combined channel mesh
    meta : dict
        Metadata about created channels
    """
    import trimesh
    
    if policy is None:
        policy = ChannelPolicy()
    
    meshes = []
    meta = {
        "channel_count": 0,
        "channel_type": policy.channel_type,
        "warnings": [],
    }
    
    # Get domain center for channel endpoints
    if hasattr(domain, 'center'):
        center = np.array([domain.center.x, domain.center.y, domain.center.z])
    else:
        center = np.array([0, 0, 0])
    
    # Create channels from inlets
    for inlet in ports.get("inlets", []):
        pos = np.array(inlet.get("position", (0, 0, 0)))
        radius = inlet.get("radius", 0.001)
        
        # Calculate end point (toward center)
        direction = center - pos
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        else:
            direction = np.array([0, 0, -1])
        
        # Channel length based on domain size
        channel_length = np.linalg.norm(center - pos) * 0.8
        end = pos + direction * channel_length
        
        if policy.channel_type == "straight":
            channel = create_straight_channel(
                tuple(pos), tuple(end), radius, policy.segments_per_curve
            )
        elif policy.channel_type == "tapered":
            end_radius = radius * policy.taper_factor
            channel = create_tapered_channel(
                tuple(pos), tuple(end), radius, end_radius, policy.segments_per_curve
            )
        elif policy.channel_type == "fang_hook":
            channel, hook_meta = create_fang_hook(
                tuple(pos), tuple(end), radius, policy.hook_depth,
                policy.hook_angle_deg, policy.segments_per_curve,
                policy=policy,
            )
            if hook_meta.get("constraint_warning"):
                meta["warnings"].append(hook_meta["constraint_warning"])
        else:
            channel = create_straight_channel(
                tuple(pos), tuple(end), radius, policy.segments_per_curve
            )
        
        meshes.append(channel)
        meta["channel_count"] += 1
    
    # Combine meshes
    if meshes:
        combined = trimesh.util.concatenate(meshes)
    else:
        combined = trimesh.Trimesh()
    
    return combined, meta


__all__ = [
    "create_straight_channel",
    "create_tapered_channel",
    "create_fang_hook",
    "create_channels_from_ports",
    "ChannelPolicy",
]
