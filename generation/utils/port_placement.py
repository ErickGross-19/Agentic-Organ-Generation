"""
Domain-aware port placement utilities.

This module provides functions for placing inlet/outlet ports on domain
surfaces with support for ridge constraints and effective radius calculations.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List, TYPE_CHECKING
import numpy as np
from math import cos, sin, pi, sqrt
import logging

from aog_policies import PortPlacementPolicy, OperationReport

if TYPE_CHECKING:
    from ..specs.design_spec import DomainSpec

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


__all__ = [
    "compute_effective_radius",
    "place_ports",
    "place_ports_circle",
    "place_ports_grid",
    "place_ports_center_rings",
    "PlacementResult",
    "PortPlacementPolicy",
]
