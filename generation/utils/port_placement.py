"""
Domain-aware port placement utilities.

This module provides functions for placing inlet/outlet ports on domain
surfaces with support for ridge constraints and effective radius calculations.

The main entry point is `place_ports_on_domain()` which accepts a domain object
and PortPlacementPolicy, returning positions, directions, and metadata.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List, Union, TYPE_CHECKING
import numpy as np
from math import cos, sin, pi, sqrt
import logging

from aog_policies import PortPlacementPolicy, OperationReport
from .faces import validate_face, face_normal, face_center, face_frame

if TYPE_CHECKING:
    from ..core.domain import BoxDomain, CylinderDomain, EllipsoidDomain

logger = logging.getLogger(__name__)


@dataclass
class PlacementResult:
    """Result of port placement calculation."""
    positions: List[Tuple[float, float, float]]
    effective_radius: float
    clamp_count: int
    projection_distances: List[float]
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "positions": self.positions,
            "effective_radius": self.effective_radius,
            "clamp_count": self.clamp_count,
            "projection_distances": self.projection_distances,
            "warnings": self.warnings,
        }


def compute_effective_radius(
    domain_radius: float,
    ridge_width: float = 0.0001,
    ridge_clearance: float = 0.0001,
    port_margin: float = 0.0005,
) -> float:
    """
    Compute the effective radius for port placement.
    
    For cylinder disk faces with ridges:
    - ridge_inner_radius = R - ridge_width - ridge_clearance
    - effective_radius = ridge_inner_radius - port_margin
    
    Parameters
    ----------
    domain_radius : float
        Radius of the domain (e.g., cylinder radius) in meters
    ridge_width : float
        Width of the ridge in meters (default: 0.1mm)
    ridge_clearance : float
        Clearance from ridge in meters (default: 0.1mm)
    port_margin : float
        Additional margin for ports in meters (default: 0.5mm)
        
    Returns
    -------
    float
        Effective radius for port placement in meters
    """
    ridge_inner_radius = domain_radius - ridge_width - ridge_clearance
    effective_radius = ridge_inner_radius - port_margin
    return max(0.0, effective_radius)


def place_ports_circle(
    num_ports: int,
    domain_radius: float,
    port_radius: float,
    z_position: float,
    policy: Optional[PortPlacementPolicy] = None,
) -> Tuple[PlacementResult, OperationReport]:
    """
    Place ports in a circular pattern on a domain face.
    
    Parameters
    ----------
    num_ports : int
        Number of ports to place
    domain_radius : float
        Radius of the domain in meters
    port_radius : float
        Radius of each port in meters
    z_position : float
        Z coordinate for port placement in meters
    policy : PortPlacementPolicy, optional
        Policy controlling placement parameters
        
    Returns
    -------
    result : PlacementResult
        Placement result with positions and metadata
    report : OperationReport
        Report with requested/effective policy
    """
    if policy is None:
        policy = PortPlacementPolicy()
    
    warnings = []
    clamp_count = 0
    
    # Compute effective radius
    effective_radius = compute_effective_radius(
        domain_radius,
        policy.ridge_width,
        policy.ridge_clearance,
        policy.port_margin,
    )
    
    # Maximum placement radius (center of ports must be inside effective radius)
    max_placement_radius = effective_radius - port_radius
    
    if max_placement_radius <= 0:
        warnings.append(
            f"Cannot place ports: max_placement_radius <= 0 "
            f"(effective_radius={effective_radius:.6f}, port_radius={port_radius:.6f})"
        )
        max_placement_radius = 0.001  # Fallback to small radius
        clamp_count = num_ports
    
    # Compute positions
    positions = []
    projection_distances = []
    
    if num_ports == 0:
        pass
    elif num_ports == 1:
        positions.append((0.0, 0.0, z_position))
        projection_distances.append(0.0)
    elif num_ports == 2:
        offset = max_placement_radius * policy.placement_fraction
        positions.append((offset, 0.0, z_position))
        positions.append((-offset, 0.0, z_position))
        projection_distances.extend([offset, offset])
    elif num_ports == 3:
        offset = max_placement_radius * policy.placement_fraction
        for i in range(3):
            angle = policy.angular_offset + 2 * pi * i / 3 - pi / 2
            x = offset * cos(angle)
            y = offset * sin(angle)
            positions.append((x, y, z_position))
            projection_distances.append(offset)
    elif num_ports == 4:
        offset = max_placement_radius / sqrt(2.0)
        for dx, dy in [(1, 1), (-1, 1), (-1, -1), (1, -1)]:
            x = offset * dx
            y = offset * dy
            positions.append((x, y, z_position))
            projection_distances.append(offset * sqrt(2))
    else:
        # General case: distribute evenly on circle
        offset = max_placement_radius * policy.placement_fraction
        for i in range(num_ports):
            angle = policy.angular_offset + 2 * pi * i / num_ports
            x = offset * cos(angle)
            y = offset * sin(angle)
            positions.append((x, y, z_position))
            projection_distances.append(offset)
    
    # Check and clamp positions that exceed effective radius
    clamped_positions = []
    for i, (x, y, z) in enumerate(positions):
        dist = sqrt(x * x + y * y)
        if dist + port_radius > effective_radius:
            # Clamp to effective radius
            if dist > 0:
                scale = (effective_radius - port_radius) / dist
                x = x * scale
                y = y * scale
                clamp_count += 1
        clamped_positions.append((x, y, z))
    
    if clamp_count > 0:
        warnings.append(f"Clamped {clamp_count} port positions to effective radius")
    
    result = PlacementResult(
        positions=clamped_positions,
        effective_radius=effective_radius,
        clamp_count=clamp_count,
        projection_distances=projection_distances,
        warnings=warnings,
    )
    
    report = OperationReport(
        operation="place_ports_circle",
        success=True,
        requested_policy=policy.to_dict(),
        effective_policy=policy.to_dict(),
        warnings=warnings,
        metadata={
            "num_ports": num_ports,
            "effective_radius": effective_radius,
            "max_placement_radius": max_placement_radius,
            "clamp_count": clamp_count,
        },
    )
    
    return result, report


def place_ports_grid(
    num_ports: int,
    domain_radius: float,
    port_radius: float,
    z_position: float,
    policy: Optional[PortPlacementPolicy] = None,
) -> Tuple[PlacementResult, OperationReport]:
    """
    Place ports in a grid pattern on a domain face.
    
    Parameters
    ----------
    num_ports : int
        Number of ports to place
    domain_radius : float
        Radius of the domain in meters
    port_radius : float
        Radius of each port in meters
    z_position : float
        Z coordinate for port placement in meters
    policy : PortPlacementPolicy, optional
        Policy controlling placement parameters
        
    Returns
    -------
    result : PlacementResult
        Placement result with positions and metadata
    report : OperationReport
        Report with requested/effective policy
    """
    if policy is None:
        policy = PortPlacementPolicy()
    
    warnings = []
    clamp_count = 0
    
    # Compute effective radius
    effective_radius = compute_effective_radius(
        domain_radius,
        policy.ridge_width,
        policy.ridge_clearance,
        policy.port_margin,
    )
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_ports)))
    spacing = 2 * effective_radius / (grid_size + 1)
    
    positions = []
    projection_distances = []
    
    count = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if count >= num_ports:
                break
            
            x = -effective_radius + spacing * (i + 1)
            y = -effective_radius + spacing * (j + 1)
            
            # Check if inside effective radius
            dist = sqrt(x * x + y * y)
            if dist + port_radius <= effective_radius:
                positions.append((x, y, z_position))
                projection_distances.append(dist)
                count += 1
            else:
                clamp_count += 1
        
        if count >= num_ports:
            break
    
    # If we couldn't place all ports, fall back to circle
    if len(positions) < num_ports:
        warnings.append(
            f"Grid placement could only fit {len(positions)} of {num_ports} ports, "
            f"falling back to circle pattern"
        )
        return place_ports_circle(num_ports, domain_radius, port_radius, z_position, policy)
    
    result = PlacementResult(
        positions=positions,
        effective_radius=effective_radius,
        clamp_count=clamp_count,
        projection_distances=projection_distances,
        warnings=warnings,
    )
    
    report = OperationReport(
        operation="place_ports_grid",
        success=True,
        requested_policy=policy.to_dict(),
        effective_policy=policy.to_dict(),
        warnings=warnings,
        metadata={
            "num_ports": num_ports,
            "effective_radius": effective_radius,
            "grid_size": grid_size,
            "spacing": spacing,
            "clamp_count": clamp_count,
        },
    )
    
    return result, report


def place_ports_center_rings(
    num_ports: int,
    domain_radius: float,
    port_radius: float,
    z_position: float,
    policy: Optional[PortPlacementPolicy] = None,
    spacing_factor: float = 0.5,
) -> Tuple[PlacementResult, OperationReport]:
    """
    Place ports in center + concentric rings pattern.
    
    Places 1 port at center, then fills outward in concentric rings.
    
    Parameters
    ----------
    num_ports : int
        Number of ports to place
    domain_radius : float
        Radius of the domain in meters
    port_radius : float
        Radius of each port in meters
    z_position : float
        Z coordinate for port placement in meters
    policy : PortPlacementPolicy, optional
        Policy controlling placement parameters
    spacing_factor : float
        Extra gap between ports as fraction of diameter
        
    Returns
    -------
    result : PlacementResult
        Placement result with positions and metadata
    report : OperationReport
        Report with requested/effective policy
    """
    if policy is None:
        policy = PortPlacementPolicy()
    
    warnings = []
    clamp_count = 0
    
    # Compute effective radius
    effective_radius = compute_effective_radius(
        domain_radius,
        policy.ridge_width,
        policy.ridge_clearance,
        policy.port_margin,
    )
    
    max_placement_radius = effective_radius - port_radius
    
    if max_placement_radius <= 0:
        warnings.append(f"Cannot place ports: max_placement_radius <= 0")
        max_placement_radius = 0.001
        clamp_count = num_ports
    
    positions = [(0.0, 0.0, z_position)]  # Center port
    projection_distances = [0.0]
    
    if num_ports == 1:
        pass
    else:
        pitch = 2.0 * port_radius * (1.0 + spacing_factor)
        ring_k = 1
        
        while len(positions) < num_ports:
            ring_radius = ring_k * pitch
            
            if ring_radius > max_placement_radius:
                warnings.append(
                    f"Could only fit {len(positions)} of {num_ports} ports "
                    f"within effective radius"
                )
                clamp_count = num_ports - len(positions)
                break
            
            circumference = 2.0 * pi * ring_radius
            n_on_ring = max(6, int(round(circumference / pitch)))
            
            for i in range(n_on_ring):
                if len(positions) >= num_ports:
                    break
                angle = policy.angular_offset + 2.0 * pi * i / n_on_ring
                x = ring_radius * cos(angle)
                y = ring_radius * sin(angle)
                positions.append((x, y, z_position))
                projection_distances.append(ring_radius)
            
            ring_k += 1
    
    positions = positions[:num_ports]
    projection_distances = projection_distances[:num_ports]
    
    result = PlacementResult(
        positions=positions,
        effective_radius=effective_radius,
        clamp_count=clamp_count,
        projection_distances=projection_distances,
        warnings=warnings,
    )
    
    report = OperationReport(
        operation="place_ports_center_rings",
        success=True,
        requested_policy=policy.to_dict(),
        effective_policy=policy.to_dict(),
        warnings=warnings,
        metadata={
            "num_ports": num_ports,
            "effective_radius": effective_radius,
            "max_placement_radius": max_placement_radius,
            "clamp_count": clamp_count,
        },
    )
    
    return result, report


def place_ports(
    num_ports: int,
    domain_radius: float,
    port_radius: float,
    z_position: float,
    policy: Optional[PortPlacementPolicy] = None,
) -> Tuple[PlacementResult, OperationReport]:
    """
    Place ports using the pattern specified in the policy.
    
    This is the main entry point for port placement.
    
    Parameters
    ----------
    num_ports : int
        Number of ports to place
    domain_radius : float
        Radius of the domain in meters
    port_radius : float
        Radius of each port in meters
    z_position : float
        Z coordinate for port placement in meters
    policy : PortPlacementPolicy, optional
        Policy controlling placement parameters
        
    Returns
    -------
    result : PlacementResult
        Placement result with positions and metadata
    report : OperationReport
        Report with requested/effective policy
    """
    if policy is None:
        policy = PortPlacementPolicy()
    
    if policy.pattern == "circle":
        return place_ports_circle(num_ports, domain_radius, port_radius, z_position, policy)
    elif policy.pattern == "grid":
        return place_ports_grid(num_ports, domain_radius, port_radius, z_position, policy)
    elif policy.pattern == "center_rings":
        return place_ports_center_rings(num_ports, domain_radius, port_radius, z_position, policy)
    else:
        # Default to circle
        return place_ports_circle(num_ports, domain_radius, port_radius, z_position, policy)


@dataclass
class DomainAwarePlacementResult:
    """Result of domain-aware port placement calculation."""
    positions: List[Tuple[float, float, float]]
    directions: List[Tuple[float, float, float]]
    face_center: Tuple[float, float, float]
    effective_radius: float
    clamp_count: int
    projection_stats: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "positions": self.positions,
            "directions": self.directions,
            "face_center": self.face_center,
            "effective_radius": self.effective_radius,
            "clamp_count": self.clamp_count,
            "projection_stats": self.projection_stats,
            "warnings": self.warnings,
        }


def _get_domain_face_radius(
    domain: Union["BoxDomain", "CylinderDomain", "EllipsoidDomain"],
    face: str,
) -> float:
    """
    Get the effective radius for port placement on a domain face.
    
    For cylinders: top/bottom faces use cylinder radius
    For boxes: uses half the smaller dimension of the face
    For ellipsoids: uses the semi-axis perpendicular to the face normal
    """
    from ..core.domain import BoxDomain, CylinderDomain, EllipsoidDomain
    
    canonical = validate_face(face)
    
    if isinstance(domain, CylinderDomain):
        if canonical in ("top", "bottom", "+z", "-z"):
            return domain.radius
        else:
            # Side faces - use half height as "radius"
            return domain.height / 2
    
    elif isinstance(domain, BoxDomain):
        dx = domain.x_max - domain.x_min
        dy = domain.y_max - domain.y_min
        dz = domain.z_max - domain.z_min
        
        if canonical in ("top", "bottom", "+z", "-z"):
            return min(dx, dy) / 2
        elif canonical in ("+x", "-x"):
            return min(dy, dz) / 2
        elif canonical in ("+y", "-y"):
            return min(dx, dz) / 2
    
    elif isinstance(domain, EllipsoidDomain):
        if canonical in ("top", "bottom", "+z", "-z"):
            return min(domain.semi_axis_a, domain.semi_axis_b)
        elif canonical in ("+x", "-x"):
            return min(domain.semi_axis_b, domain.semi_axis_c)
        elif canonical in ("+y", "-y"):
            return min(domain.semi_axis_a, domain.semi_axis_c)
    
    # Fallback
    return 0.001


def place_ports_on_domain(
    domain: Union["BoxDomain", "CylinderDomain", "EllipsoidDomain"],
    num_ports: int,
    port_radius: float,
    policy: Optional[PortPlacementPolicy] = None,
) -> Tuple[DomainAwarePlacementResult, OperationReport]:
    """
    Place ports on a domain face using the policy configuration.
    
    This is the main entry point for domain-aware port placement.
    It uses the face frame to build a reference plane and places ports
    in the local coordinate system, then transforms to world coordinates.
    
    Parameters
    ----------
    domain : BoxDomain, CylinderDomain, or EllipsoidDomain
        Domain to place ports on
    num_ports : int
        Number of ports to place
    port_radius : float
        Radius of each port in meters
    policy : PortPlacementPolicy, optional
        Policy controlling placement parameters
        
    Returns
    -------
    result : DomainAwarePlacementResult
        Placement result with positions, directions, and metadata
    report : OperationReport
        Report with requested/effective policy
    """
    if policy is None:
        policy = PortPlacementPolicy()
    
    warnings = []
    
    # Get face frame
    canonical_face = validate_face(policy.face)
    origin, normal, u_vec, v_vec, center = face_frame(canonical_face, domain)
    
    # Get domain radius for this face
    domain_radius = _get_domain_face_radius(domain, canonical_face)
    
    # Compute effective radius with ridge constraints
    if policy.ridge_constraint_enabled:
        effective_radius = compute_effective_radius(
            domain_radius,
            policy.ridge_width,
            policy.ridge_clearance,
            policy.port_margin,
        )
    else:
        effective_radius = domain_radius - policy.port_margin
    
    # Place ports in 2D local coordinates (on the face plane)
    # Use z_position=0 since we'll transform to world coordinates
    local_result, local_report = place_ports(
        num_ports=num_ports,
        domain_radius=domain_radius,
        port_radius=port_radius,
        z_position=0.0,
        policy=policy,
    )
    
    # Transform local positions to world coordinates
    u = np.array(u_vec)
    v = np.array(v_vec)
    n = np.array(normal)
    origin_pt = np.array(origin)
    
    world_positions = []
    for local_x, local_y, _ in local_result.positions:
        # Transform: world = origin + local_x * u + local_y * v
        world_pt = origin_pt + local_x * u + local_y * v
        world_positions.append(tuple(world_pt.tolist()))
    
    # All ports have the same direction (inward normal)
    inward_normal = tuple((-n).tolist())
    directions = [inward_normal] * num_ports
    
    # Compute projection stats
    projection_stats = {
        "min_distance_from_center": min(local_result.projection_distances) if local_result.projection_distances else 0.0,
        "max_distance_from_center": max(local_result.projection_distances) if local_result.projection_distances else 0.0,
        "mean_distance_from_center": np.mean(local_result.projection_distances) if local_result.projection_distances else 0.0,
        "domain_radius": domain_radius,
        "effective_radius": effective_radius,
    }
    
    result = DomainAwarePlacementResult(
        positions=world_positions,
        directions=directions,
        face_center=center,
        effective_radius=effective_radius,
        clamp_count=local_result.clamp_count,
        projection_stats=projection_stats,
        warnings=local_result.warnings + warnings,
    )
    
    report = OperationReport(
        operation="place_ports_on_domain",
        success=True,
        requested_policy=policy.to_dict(),
        effective_policy=policy.to_dict(),
        warnings=result.warnings,
        metadata={
            "num_ports": num_ports,
            "face": canonical_face,
            "face_center": center,
            "effective_radius": effective_radius,
            "domain_radius": domain_radius,
            "clamp_count": result.clamp_count,
            "projection_stats": projection_stats,
        },
    )
    
    return result, report


__all__ = [
    "compute_effective_radius",
    "place_ports",
    "place_ports_circle",
    "place_ports_grid",
    "place_ports_center_rings",
    "place_ports_on_domain",
    "PlacementResult",
    "DomainAwarePlacementResult",
    "PortPlacementPolicy",
]
