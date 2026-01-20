"""
Preflight checks for the DesignSpec pipeline.

These checks run before heavy computation stages to fail early with
actionable error messages. They validate:
- Surface opening port placement (port must be on/near domain boundary)
- Union pitch sensibility (relative to smallest diameter and domain scale)
- Port radius compatibility (with wall thickness and domain size)
- Network bbox coverage (warn if network is tiny relative to domain)

PATCH 8: Added preflight validation for surface openings and union pitch.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
import logging
import numpy as np

if TYPE_CHECKING:
    import trimesh
    from aog_policies.generation import MeshMergePolicy
    from aog_policies.validity import ValidationPolicy

logger = logging.getLogger(__name__)


@dataclass
class PreflightResult:
    """Result of a preflight check."""
    check_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "error"  # "error", "warning", "info"


@dataclass
class PreflightReport:
    """Aggregated preflight check results."""
    results: List[PreflightResult] = field(default_factory=list)
    
    @property
    def passed(self) -> bool:
        """True if all error-severity checks passed."""
        return all(r.passed or r.severity != "error" for r in self.results)
    
    @property
    def has_warnings(self) -> bool:
        """True if any warning-severity checks failed."""
        return any(not r.passed and r.severity == "warning" for r in self.results)
    
    def add(self, result: PreflightResult) -> None:
        """Add a check result."""
        self.results.append(result)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "has_warnings": self.has_warnings,
            "results": [
                {
                    "check_name": r.check_name,
                    "passed": r.passed,
                    "message": r.message,
                    "details": r.details,
                    "severity": r.severity,
                }
                for r in self.results
            ],
        }


def check_surface_opening_port_placement(
    port_position: np.ndarray,
    port_direction: np.ndarray,
    port_radius: float,
    domain_bounds: Tuple[np.ndarray, np.ndarray],
    port_id: str = "unknown",
    tolerance: float = 0.001,  # 1mm default tolerance
) -> PreflightResult:
    """
    Check if a surface opening port is placed on or near the domain boundary.
    
    For a port to be a valid surface opening, it must be positioned such that
    the opening can reach the domain boundary. This check verifies:
    1. Port position is within tolerance of a domain face, OR
    2. Port direction points toward a nearby domain face
    
    Parameters
    ----------
    port_position : np.ndarray
        Port center position [x, y, z]
    port_direction : np.ndarray
        Port direction vector (normalized)
    port_radius : float
        Port radius in meters
    domain_bounds : tuple
        (min_corner, max_corner) of domain bounding box
    port_id : str
        Port identifier for error messages
    tolerance : float
        Distance tolerance for "near boundary" check
        
    Returns
    -------
    PreflightResult
        Check result with pass/fail and actionable message
    """
    min_corner, max_corner = domain_bounds
    pos = np.asarray(port_position)
    direction = np.asarray(port_direction)
    direction = direction / np.linalg.norm(direction)
    
    # Calculate distance to each domain face
    distances_to_faces = {
        "x_min": pos[0] - min_corner[0],
        "x_max": max_corner[0] - pos[0],
        "y_min": pos[1] - min_corner[1],
        "y_max": max_corner[1] - pos[1],
        "z_min": pos[2] - min_corner[2],
        "z_max": max_corner[2] - pos[2],
    }
    
    # Find closest face
    closest_face = min(distances_to_faces, key=distances_to_faces.get)
    min_distance = distances_to_faces[closest_face]
    
    # Check if port is on or near boundary
    on_boundary = min_distance <= tolerance
    
    # Check if port direction points toward boundary
    # Face normals (outward from domain)
    face_normals = {
        "x_min": np.array([-1, 0, 0]),
        "x_max": np.array([1, 0, 0]),
        "y_min": np.array([0, -1, 0]),
        "y_max": np.array([0, 1, 0]),
        "z_min": np.array([0, 0, -1]),
        "z_max": np.array([0, 0, 1]),
    }
    
    # Port direction should point outward (same direction as face normal)
    # for the opening to connect to exterior
    direction_alignment = np.dot(direction, face_normals[closest_face])
    points_toward_boundary = direction_alignment > 0.5  # Within ~60 degrees
    
    # Calculate how far the opening would need to extend to reach boundary
    extension_needed = min_distance if not on_boundary else 0.0
    
    details = {
        "port_id": port_id,
        "port_position": pos.tolist(),
        "port_direction": direction.tolist(),
        "port_radius": port_radius,
        "closest_face": closest_face,
        "distance_to_boundary": min_distance,
        "on_boundary": on_boundary,
        "direction_alignment": direction_alignment,
        "points_toward_boundary": points_toward_boundary,
        "extension_needed": extension_needed,
    }
    
    if on_boundary:
        return PreflightResult(
            check_name="surface_opening_port_placement",
            passed=True,
            message=f"Port '{port_id}' is on domain boundary (face: {closest_face})",
            details=details,
            severity="info",
        )
    elif points_toward_boundary and min_distance < port_radius * 2:
        return PreflightResult(
            check_name="surface_opening_port_placement",
            passed=True,
            message=(
                f"Port '{port_id}' is near boundary ({min_distance*1000:.2f}mm away) "
                f"and points toward it. Opening creation should reach boundary."
            ),
            details=details,
            severity="info",
        )
    else:
        return PreflightResult(
            check_name="surface_opening_port_placement",
            passed=False,
            message=(
                f"Port '{port_id}' marked as surface opening but is {min_distance*1000:.2f}mm "
                f"from nearest boundary (face: {closest_face}). "
                f"Direction alignment: {direction_alignment:.2f}. "
                f"Suggestion: Move port closer to boundary or adjust port direction to point "
                f"toward {closest_face} face."
            ),
            details=details,
            severity="error",
        )


def check_union_pitch_sensibility(
    voxel_pitch: float,
    min_channel_diameter: Optional[float],
    domain_size: np.ndarray,
    min_voxels_per_diameter: int = 4,
) -> PreflightResult:
    """
    Check if union voxel pitch is sensible for the geometry.
    
    Validates that:
    1. Pitch provides enough voxels across smallest channel diameter
    2. Pitch is not too fine relative to domain size (memory concerns)
    
    Parameters
    ----------
    voxel_pitch : float
        Voxel pitch in meters
    min_channel_diameter : float or None
        Smallest expected channel diameter in meters
    domain_size : np.ndarray
        Domain dimensions [x, y, z] in meters
    min_voxels_per_diameter : int
        Minimum voxels required across smallest diameter
        
    Returns
    -------
    PreflightResult
        Check result with pass/fail and actionable message
    """
    details = {
        "voxel_pitch": voxel_pitch,
        "min_channel_diameter": min_channel_diameter,
        "domain_size": domain_size.tolist() if hasattr(domain_size, 'tolist') else list(domain_size),
        "min_voxels_per_diameter": min_voxels_per_diameter,
    }
    
    # Check if pitch is too coarse for smallest channel
    if min_channel_diameter is not None and min_channel_diameter > 0:
        voxels_per_diameter = min_channel_diameter / voxel_pitch
        details["voxels_per_diameter"] = voxels_per_diameter
        
        if voxels_per_diameter < min_voxels_per_diameter:
            recommended_pitch = min_channel_diameter / min_voxels_per_diameter
            details["recommended_pitch"] = recommended_pitch
            
            return PreflightResult(
                check_name="union_pitch_sensibility",
                passed=False,
                message=(
                    f"Union voxel pitch ({voxel_pitch*1e6:.1f}um) is too coarse for "
                    f"smallest channel diameter ({min_channel_diameter*1e6:.1f}um). "
                    f"Only {voxels_per_diameter:.1f} voxels across diameter "
                    f"(need at least {min_voxels_per_diameter}). "
                    f"Suggestion: Set voxel_pitch to {recommended_pitch*1e6:.1f}um or less."
                ),
                details=details,
                severity="warning",
            )
    
    # Check if pitch is too fine (memory concerns)
    domain_size_arr = np.asarray(domain_size)
    estimated_voxels = np.prod(domain_size_arr / voxel_pitch)
    details["estimated_voxels"] = int(estimated_voxels)
    
    max_reasonable_voxels = 500_000_000  # 500M voxels
    if estimated_voxels > max_reasonable_voxels:
        return PreflightResult(
            check_name="union_pitch_sensibility",
            passed=False,
            message=(
                f"Union voxel pitch ({voxel_pitch*1e6:.1f}um) would create "
                f"~{estimated_voxels/1e9:.1f}B voxels for domain. "
                f"This may cause memory issues. "
                f"Suggestion: Increase voxel_pitch or reduce domain size."
            ),
            details=details,
            severity="warning",
        )
    
    return PreflightResult(
        check_name="union_pitch_sensibility",
        passed=True,
        message=(
            f"Union voxel pitch ({voxel_pitch*1e6:.1f}um) is sensible for domain. "
            f"Estimated voxels: {estimated_voxels/1e6:.1f}M"
        ),
        details=details,
        severity="info",
    )


def check_port_radius_compatibility(
    port_radius: float,
    wall_thickness: Optional[float],
    domain_size: np.ndarray,
    port_id: str = "unknown",
) -> PreflightResult:
    """
    Check if port radius is compatible with wall thickness and domain size.
    
    Parameters
    ----------
    port_radius : float
        Port radius in meters
    wall_thickness : float or None
        Wall/shell thickness in meters
    domain_size : np.ndarray
        Domain dimensions [x, y, z] in meters
    port_id : str
        Port identifier for error messages
        
    Returns
    -------
    PreflightResult
        Check result with pass/fail and actionable message
    """
    domain_size_arr = np.asarray(domain_size)
    min_domain_dim = np.min(domain_size_arr)
    
    details = {
        "port_id": port_id,
        "port_radius": port_radius,
        "wall_thickness": wall_thickness,
        "domain_size": domain_size_arr.tolist(),
        "min_domain_dim": min_domain_dim,
    }
    
    # Check if port is too large relative to domain
    if port_radius * 2 > min_domain_dim * 0.5:
        return PreflightResult(
            check_name="port_radius_compatibility",
            passed=False,
            message=(
                f"Port '{port_id}' radius ({port_radius*1000:.2f}mm) is too large "
                f"relative to domain (min dimension: {min_domain_dim*1000:.2f}mm). "
                f"Port diameter is {port_radius*2/min_domain_dim*100:.0f}% of domain. "
                f"Suggestion: Reduce port radius or increase domain size."
            ),
            details=details,
            severity="warning",
        )
    
    # Check if port is compatible with wall thickness
    if wall_thickness is not None and wall_thickness > 0:
        if port_radius < wall_thickness:
            return PreflightResult(
                check_name="port_radius_compatibility",
                passed=False,
                message=(
                    f"Port '{port_id}' radius ({port_radius*1000:.2f}mm) is smaller than "
                    f"wall thickness ({wall_thickness*1000:.2f}mm). "
                    f"This may cause the port to be blocked. "
                    f"Suggestion: Increase port radius or reduce wall thickness."
                ),
                details=details,
                severity="warning",
            )
    
    return PreflightResult(
        check_name="port_radius_compatibility",
        passed=True,
        message=f"Port '{port_id}' radius is compatible with domain and wall thickness.",
        details=details,
        severity="info",
    )


def check_network_bbox_coverage(
    network_bbox: Tuple[np.ndarray, np.ndarray],
    domain_bbox: Tuple[np.ndarray, np.ndarray],
    min_coverage_ratio: float = 0.1,
) -> PreflightResult:
    """
    Check if network bounding box has reasonable coverage of domain.
    
    Warns if the generated network is tiny relative to the domain,
    which may indicate a configuration issue.
    
    Parameters
    ----------
    network_bbox : tuple
        (min_corner, max_corner) of network bounding box
    domain_bbox : tuple
        (min_corner, max_corner) of domain bounding box
    min_coverage_ratio : float
        Minimum acceptable ratio of network volume to domain volume
        
    Returns
    -------
    PreflightResult
        Check result with pass/fail and actionable message
    """
    net_min, net_max = network_bbox
    dom_min, dom_max = domain_bbox
    
    net_size = np.asarray(net_max) - np.asarray(net_min)
    dom_size = np.asarray(dom_max) - np.asarray(dom_min)
    
    net_volume = np.prod(net_size)
    dom_volume = np.prod(dom_size)
    
    coverage_ratio = net_volume / dom_volume if dom_volume > 0 else 0.0
    
    # Also check linear coverage in each dimension
    linear_coverage = net_size / dom_size
    
    details = {
        "network_bbox": {
            "min": np.asarray(net_min).tolist(),
            "max": np.asarray(net_max).tolist(),
        },
        "domain_bbox": {
            "min": np.asarray(dom_min).tolist(),
            "max": np.asarray(dom_max).tolist(),
        },
        "network_size": net_size.tolist(),
        "domain_size": dom_size.tolist(),
        "volume_coverage_ratio": coverage_ratio,
        "linear_coverage": linear_coverage.tolist(),
    }
    
    if coverage_ratio < min_coverage_ratio:
        return PreflightResult(
            check_name="network_bbox_coverage",
            passed=False,
            message=(
                f"Network bounding box is tiny relative to domain. "
                f"Volume coverage: {coverage_ratio*100:.1f}% (min: {min_coverage_ratio*100:.0f}%). "
                f"Linear coverage: x={linear_coverage[0]*100:.0f}%, "
                f"y={linear_coverage[1]*100:.0f}%, z={linear_coverage[2]*100:.0f}%. "
                f"This may indicate the network is not filling the domain as intended. "
                f"Suggestion: Check growth parameters, port positions, or domain size."
            ),
            details=details,
            severity="warning",
        )
    
    return PreflightResult(
        check_name="network_bbox_coverage",
        passed=True,
        message=(
            f"Network has reasonable domain coverage. "
            f"Volume coverage: {coverage_ratio*100:.1f}%"
        ),
        details=details,
        severity="info",
    )


def run_preflight_checks(
    ports: List[Dict[str, Any]],
    domain_bounds: Tuple[np.ndarray, np.ndarray],
    voxel_pitch: float,
    min_channel_diameter: Optional[float] = None,
    wall_thickness: Optional[float] = None,
    network_bbox: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    validation_policy: Optional["ValidationPolicy"] = None,
) -> PreflightReport:
    """
    Run all preflight checks.
    
    Parameters
    ----------
    ports : list
        List of port dictionaries with position, direction, radius, is_surface_opening
    domain_bounds : tuple
        (min_corner, max_corner) of domain bounding box
    voxel_pitch : float
        Voxel pitch for union operation
    min_channel_diameter : float, optional
        Smallest expected channel diameter
    wall_thickness : float, optional
        Wall/shell thickness
    network_bbox : tuple, optional
        Network bounding box for coverage check
    validation_policy : ValidationPolicy, optional
        Validation policy with surface opening settings
        
    Returns
    -------
    PreflightReport
        Aggregated check results
    """
    report = PreflightReport()
    
    min_corner, max_corner = domain_bounds
    domain_size = np.asarray(max_corner) - np.asarray(min_corner)
    
    # Get tolerance from policy if available
    tolerance = 0.001  # 1mm default
    if validation_policy is not None:
        tolerance = getattr(validation_policy, 'surface_opening_tolerance', 0.001)
    
    # Check surface opening ports
    for port in ports:
        is_surface_opening = port.get("is_surface_opening", False)
        if is_surface_opening:
            position = np.asarray(port.get("position", [0, 0, 0]))
            direction = np.asarray(port.get("direction", [0, 0, 1]))
            radius = port.get("radius", 0.001)
            port_id = port.get("id", port.get("name", "unknown"))
            
            result = check_surface_opening_port_placement(
                port_position=position,
                port_direction=direction,
                port_radius=radius,
                domain_bounds=domain_bounds,
                port_id=port_id,
                tolerance=tolerance,
            )
            report.add(result)
    
    # Check union pitch sensibility
    result = check_union_pitch_sensibility(
        voxel_pitch=voxel_pitch,
        min_channel_diameter=min_channel_diameter,
        domain_size=domain_size,
    )
    report.add(result)
    
    # Check port radius compatibility
    for port in ports:
        position = np.asarray(port.get("position", [0, 0, 0]))
        radius = port.get("radius", 0.001)
        port_id = port.get("id", port.get("name", "unknown"))
        
        result = check_port_radius_compatibility(
            port_radius=radius,
            wall_thickness=wall_thickness,
            domain_size=domain_size,
            port_id=port_id,
        )
        report.add(result)
    
    # Check network bbox coverage if provided
    if network_bbox is not None:
        result = check_network_bbox_coverage(
            network_bbox=network_bbox,
            domain_bbox=domain_bounds,
        )
        report.add(result)
    
    return report


__all__ = [
    "PreflightResult",
    "PreflightReport",
    "check_surface_opening_port_placement",
    "check_union_pitch_sensibility",
    "check_port_radius_compatibility",
    "check_network_bbox_coverage",
    "run_preflight_checks",
]
