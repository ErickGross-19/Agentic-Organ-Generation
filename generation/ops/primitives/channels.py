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

from aog_policies import ChannelPolicy, OperationReport

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
    hook_depth: Optional[float] = None,
    hook_angle_deg: Optional[float] = None,
    segments_per_curve: Optional[int] = None,
    effective_radius: Optional[float] = None,
    face_center: Optional[Tuple[float, float, float]] = None,
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
    hook_depth : float, optional
        Depth of the hook curve in meters (defaults to policy.hook_depth)
    hook_angle_deg : float, optional
        Angle of the hook in degrees (defaults to policy.hook_angle_deg or 90)
    segments_per_curve : int, optional
        Number of segments for the curved portion (defaults to policy.segments_per_curve or 16)
    effective_radius : float, optional
        Maximum effective radius for constraint enforcement
    face_center : tuple, optional
        Center of the face for radial outward direction calculation.
        If provided, hook bends radially outward from face center.
        If None, uses arbitrary perpendicular direction.
    policy : ChannelPolicy, optional
        Policy for additional constraints and default values
        
    Returns
    -------
    mesh : trimesh.Trimesh
        Fang hook mesh
    meta : dict
        Metadata including any constraint modifications
    """
    import trimesh
    
    # Use policy defaults if not specified
    if policy is None:
        policy = ChannelPolicy()
    
    if hook_depth is None:
        hook_depth = policy.hook_depth
    if hook_angle_deg is None:
        hook_angle_deg = policy.hook_angle_deg
    if segments_per_curve is None:
        segments_per_curve = policy.segments_per_curve
    
    start = np.array(start)
    end = np.array(end)
    
    meta = {
        "requested_hook_depth": hook_depth,
        "hook_depth_used": hook_depth,
        "constraint_modified": False,
        "bend_mode": policy.bend_mode,
        "face_center_used": face_center is not None,
    }
    
    # B4 FIX: Check effective radius constraint with distance from start to face center
    if effective_radius is not None and policy.enforce_effective_radius:
        # Calculate distance from start to face center in the face plane
        start_to_center_dist = 0.0
        if face_center is not None:
            face_center_arr = np.array(face_center)
            # Project onto face plane (perpendicular to direction)
            start_to_center = start - face_center_arr
            # Remove component along direction to get in-plane distance
            start_to_center_in_plane = start_to_center - np.dot(start_to_center, direction_norm) * direction_norm
            start_to_center_dist = np.linalg.norm(start_to_center_in_plane)
            meta["start_to_center_dist"] = start_to_center_dist
        
        # B4 FIX: max_hook_depth = (effective_radius - radius) - dist(start_xy, face_center_xy)
        max_hook_depth = (effective_radius - radius) - start_to_center_dist
        max_hook_depth = max(0.0, max_hook_depth)  # Ensure non-negative
        
        if hook_depth > max_hook_depth:
            meta["hook_depth_used"] = max_hook_depth
            meta["constraint_modified"] = True
            meta["constraint_warning"] = (
                f"Hook depth reduced from {hook_depth:.6f} to {max_hook_depth:.6f} "
                f"due to effective radius constraint (start_to_center_dist={start_to_center_dist:.6f})"
            )
            hook_depth = max_hook_depth
            logger.warning(meta["constraint_warning"])
    
    # Calculate the curve
    direction = end - start
    length = np.linalg.norm(direction)
    direction_norm = direction / length if length > 1e-9 else np.array([0, 0, -1])
    
    # Find perpendicular for hook direction
    # Use face-center radial outward if face_center is provided and bend_mode is "radial_out"
    if face_center is not None and policy.bend_mode == "radial_out":
        face_center = np.array(face_center)
        # Radial outward direction from face center to start point
        radial_out = start - face_center
        # Project onto plane perpendicular to direction
        radial_out = radial_out - np.dot(radial_out, direction_norm) * direction_norm
        radial_norm = np.linalg.norm(radial_out)
        if radial_norm > 1e-9:
            hook_perp = radial_out / radial_norm
            meta["radial_direction_used"] = True
        else:
            # Start is at face center, fall back to arbitrary perpendicular
            if abs(direction_norm[2]) < 0.9:
                hook_perp = np.cross(direction_norm, np.array([0, 0, 1]))
            else:
                hook_perp = np.cross(direction_norm, np.array([1, 0, 0]))
            hook_perp = hook_perp / np.linalg.norm(hook_perp)
            meta["radial_direction_used"] = False
            meta["radial_fallback_reason"] = "start_at_face_center"
    else:
        # Arbitrary perpendicular (legacy behavior)
        if abs(direction_norm[2]) < 0.9:
            hook_perp = np.cross(direction_norm, np.array([0, 0, 1]))
        else:
            hook_perp = np.cross(direction_norm, np.array([1, 0, 0]))
        hook_perp = hook_perp / np.linalg.norm(hook_perp)
        meta["radial_direction_used"] = False
    
    # Generate centerline points for the hook
    hook_angle_rad = np.radians(hook_angle_deg)
    centerline_points = []
    
    # Use policy fractions instead of hardcoded values
    straight_frac = policy.straight_fraction
    curve_frac = policy.curve_fraction
    
    # Straight section at start
    straight_length = length * straight_frac
    for i in range(segments_per_curve // 3):
        t = i / (segments_per_curve // 3)
        pos = start + direction_norm * straight_length * t
        centerline_points.append(pos)
    
    # Curved section
    curve_start = start + direction_norm * straight_length
    curve_length = length * curve_frac
    
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


def create_channel_from_policy(
    start: Tuple[float, float, float],
    direction: Tuple[float, float, float],
    radius: float,
    policy: ChannelPolicy,
    domain_depth: Optional[float] = None,
    face_center: Optional[Tuple[float, float, float]] = None,
    effective_radius: Optional[float] = None,
) -> Tuple["trimesh.Trimesh", OperationReport]:
    """
    Create a channel using the full ChannelPolicy configuration.
    
    This is the canonical entry point for policy-driven channel creation.
    It properly implements all policy fields including length_mode,
    start_offset, and stop_before_boundary.
    
    Parameters
    ----------
    start : tuple
        Start position (x, y, z) in meters (typically on domain surface)
    direction : tuple
        Direction vector pointing into the domain (will be normalized)
    radius : float
        Channel radius in meters
    policy : ChannelPolicy
        Policy controlling channel creation
    domain_depth : float, optional
        Depth of the domain in the direction of the channel.
        Required for length_mode="to_depth" or "to_center_fraction"
    face_center : tuple, optional
        Center of the face for radial outward direction calculation.
        Used for fang_hook profile with bend_mode="radial_out"
    effective_radius : float, optional
        Maximum effective radius for constraint enforcement
        
    Returns
    -------
    mesh : trimesh.Trimesh
        Channel mesh
    report : OperationReport
        Report with requested/effective policy and metadata
    """
    import trimesh
    
    start = np.array(start)
    direction = np.array(direction)
    direction_norm = np.linalg.norm(direction)
    if direction_norm > 1e-9:
        direction = direction / direction_norm
    else:
        direction = np.array([0, 0, -1])
    
    warnings = []
    meta = {
        "profile": policy.profile,
        "length_mode": policy.length_mode,
        "requested_length": policy.length,
        "start_offset_applied": policy.start_offset,
        "stop_before_boundary_applied": policy.stop_before_boundary,
    }
    
    # Apply start_offset
    actual_start = start + direction * policy.start_offset
    
    # Calculate channel length based on length_mode
    if policy.length_mode == "explicit":
        if policy.length is None:
            raise ValueError("length_mode='explicit' requires length to be set")
        channel_length = policy.length
    elif policy.length_mode == "to_center_fraction":
        if domain_depth is None:
            raise ValueError("length_mode='to_center_fraction' requires domain_depth")
        channel_length = domain_depth * policy.length_fraction
    elif policy.length_mode == "to_depth":
        if domain_depth is None:
            raise ValueError("length_mode='to_depth' requires domain_depth")
        channel_length = domain_depth - policy.stop_before_boundary
    else:
        raise ValueError(f"Unknown length_mode: {policy.length_mode}")
    
    # Apply stop_before_boundary
    channel_length = max(0.001, channel_length - policy.stop_before_boundary)
    
    meta["effective_length"] = channel_length
    
    # Calculate end point
    end = actual_start + direction * channel_length
    
    # Create channel based on profile
    if policy.profile == "cylinder":
        mesh = create_straight_channel(
            tuple(actual_start), tuple(end), radius, policy.radial_sections
        )
    elif policy.profile == "taper":
        end_radius = policy.radius_end if policy.radius_end is not None else radius * policy.taper_factor
        mesh = create_tapered_channel(
            tuple(actual_start), tuple(end), radius, end_radius,
            policy.radial_sections, policy.path_samples
        )
    elif policy.profile == "fang_hook":
        mesh, hook_meta = create_fang_hook(
            tuple(actual_start), tuple(end), radius,
            hook_depth=policy.hook_depth,
            hook_angle_deg=policy.hook_angle_deg,
            segments_per_curve=policy.segments_per_curve,
            effective_radius=effective_radius,
            face_center=face_center,
            policy=policy,
        )
        meta.update(hook_meta)
        if hook_meta.get("constraint_warning"):
            warnings.append(hook_meta["constraint_warning"])
    elif policy.profile == "path_channel":
        # Path channel uses the path_sweep module for more complex paths
        # For now, fall back to straight channel
        warnings.append("path_channel profile not fully implemented, using cylinder")
        mesh = create_straight_channel(
            tuple(actual_start), tuple(end), radius, policy.radial_sections
        )
    else:
        warnings.append(f"Unknown profile '{policy.profile}', using cylinder")
        mesh = create_straight_channel(
            tuple(actual_start), tuple(end), radius, policy.radial_sections
        )
    
    report = OperationReport(
        operation="create_channel_from_policy",
        success=True,
        requested_policy=policy.to_dict(),
        effective_policy=policy.to_dict(),
        warnings=warnings,
        metadata=meta,
    )
    
    return mesh, report


__all__ = [
    "create_straight_channel",
    "create_tapered_channel",
    "create_fang_hook",
    "create_channels_from_ports",
    "create_channel_from_policy",
    "ChannelPolicy",
]
