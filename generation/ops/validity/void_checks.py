"""
Validity checks for void meshes and embedded domains.

This module provides additional validity checks for void meshes,
including component analysis, domain containment, and shell integrity.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import logging

from ...core.domain import DomainSpec
from ...core.types import Point3D

if TYPE_CHECKING:
    import trimesh

logger = logging.getLogger(__name__)


class CheckStatus(str, Enum):
    """Status of a validity check."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class CheckResult:
    """Result of a single validity check."""
    name: str
    status: CheckStatus
    message: str = ""
    value: Optional[float] = None
    threshold: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "details": self.details,
        }


@dataclass
class VoidValidityReport:
    """Report from void validity checks."""
    overall_status: CheckStatus
    checks: List[CheckResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status.value,
            "checks": [c.to_dict() for c in self.checks],
            "warnings": self.warnings,
            "errors": self.errors,
            "metadata": self.metadata,
        }
    
    @property
    def passed(self) -> bool:
        return self.overall_status == CheckStatus.PASSED
    
    @property
    def has_warnings(self) -> bool:
        return self.overall_status == CheckStatus.WARNING or len(self.warnings) > 0


@dataclass
class DiameterReport:
    """Report on diameter shrinkage and drift."""
    original_min_diameter: float
    effective_min_diameter: float
    shrink_ratio: float
    drift_distance: float
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_min_diameter": self.original_min_diameter,
            "effective_min_diameter": self.effective_min_diameter,
            "shrink_ratio": self.shrink_ratio,
            "drift_distance": self.drift_distance,
            "warnings": self.warnings,
        }


def check_void_components(
    void_mesh: "trimesh.Trimesh",
    max_components: int = 1,
    min_component_volume_ratio: float = 0.01,
) -> CheckResult:
    """
    Check that void mesh has expected number of connected components.
    
    Parameters
    ----------
    void_mesh : trimesh.Trimesh
        Void mesh to check
    max_components : int
        Maximum allowed number of components
    min_component_volume_ratio : float
        Minimum volume ratio for a component to be counted
        
    Returns
    -------
    CheckResult
        Result of the check
    """
    try:
        components = void_mesh.split(only_watertight=False)
        
        if len(components) == 0:
            return CheckResult(
                name="void_components",
                status=CheckStatus.FAILED,
                message="Void mesh has no components",
                value=0,
                threshold=max_components,
            )
        
        # Filter by volume
        total_volume = abs(void_mesh.volume)
        significant_components = []
        
        for comp in components:
            comp_volume = abs(comp.volume)
            if total_volume > 0 and comp_volume / total_volume >= min_component_volume_ratio:
                significant_components.append(comp)
        
        n_components = len(significant_components)
        
        if n_components > max_components:
            return CheckResult(
                name="void_components",
                status=CheckStatus.WARNING,
                message=f"Void mesh has {n_components} significant components (max: {max_components})",
                value=n_components,
                threshold=max_components,
                details={
                    "total_components": len(components),
                    "significant_components": n_components,
                    "component_volumes": [abs(c.volume) for c in significant_components],
                },
            )
        
        return CheckResult(
            name="void_components",
            status=CheckStatus.PASSED,
            message=f"Void mesh has {n_components} component(s)",
            value=n_components,
            threshold=max_components,
            details={
                "total_components": len(components),
                "significant_components": n_components,
            },
        )
        
    except Exception as e:
        return CheckResult(
            name="void_components",
            status=CheckStatus.FAILED,
            message=f"Failed to check components: {e}",
        )


def check_void_inside_domain(
    void_mesh: "trimesh.Trimesh",
    domain: DomainSpec,
    tolerance: float = 1e-6,
    sample_count: int = 1000,
) -> CheckResult:
    """
    Check that void mesh is contained within the domain.
    
    Parameters
    ----------
    void_mesh : trimesh.Trimesh
        Void mesh to check
    domain : DomainSpec
        Domain that should contain the void
    tolerance : float
        Tolerance for boundary checks (meters)
    sample_count : int
        Number of sample points to check
        
    Returns
    -------
    CheckResult
        Result of the check
    """
    try:
        # Sample points on void mesh surface
        points, _ = void_mesh.sample(sample_count, return_index=True)
        
        # Check each point
        outside_count = 0
        max_outside_distance = 0.0
        
        for pt in points:
            point = Point3D.from_array(pt)
            dist = domain.distance_to_boundary(point)
            
            # Negative distance means outside
            if dist < -tolerance:
                outside_count += 1
                max_outside_distance = max(max_outside_distance, abs(dist))
        
        outside_ratio = outside_count / sample_count
        
        if outside_ratio > 0.05:  # More than 5% outside
            return CheckResult(
                name="void_inside_domain",
                status=CheckStatus.FAILED,
                message=f"{outside_ratio:.1%} of void surface is outside domain",
                value=outside_ratio,
                threshold=0.05,
                details={
                    "outside_count": outside_count,
                    "sample_count": sample_count,
                    "max_outside_distance": max_outside_distance,
                },
            )
        elif outside_ratio > 0.01:  # More than 1% outside
            return CheckResult(
                name="void_inside_domain",
                status=CheckStatus.WARNING,
                message=f"{outside_ratio:.1%} of void surface is outside domain",
                value=outside_ratio,
                threshold=0.05,
                details={
                    "outside_count": outside_count,
                    "sample_count": sample_count,
                    "max_outside_distance": max_outside_distance,
                },
            )
        
        return CheckResult(
            name="void_inside_domain",
            status=CheckStatus.PASSED,
            message="Void mesh is contained within domain",
            value=outside_ratio,
            threshold=0.05,
            details={
                "outside_count": outside_count,
                "sample_count": sample_count,
            },
        )
        
    except Exception as e:
        return CheckResult(
            name="void_inside_domain",
            status=CheckStatus.FAILED,
            message=f"Failed to check domain containment: {e}",
        )


def check_shell_watertight(
    mesh: "trimesh.Trimesh",
) -> CheckResult:
    """
    Check that mesh shell is watertight (manifold and closed).
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to check
        
    Returns
    -------
    CheckResult
        Result of the check
    """
    try:
        is_watertight = mesh.is_watertight
        
        if is_watertight:
            return CheckResult(
                name="shell_watertight",
                status=CheckStatus.PASSED,
                message="Mesh is watertight",
                value=1.0,
                threshold=1.0,
            )
        
        # Get more details about why it's not watertight
        details = {}
        
        # Check for holes
        try:
            edges = mesh.edges_unique
            edge_face_count = mesh.edges_unique_length
            boundary_edges = np.sum(edge_face_count == 1)
            details["boundary_edges"] = int(boundary_edges)
        except Exception:
            pass
        
        # Check for non-manifold edges
        try:
            non_manifold = np.sum(mesh.edges_unique_length > 2)
            details["non_manifold_edges"] = int(non_manifold)
        except Exception:
            pass
        
        return CheckResult(
            name="shell_watertight",
            status=CheckStatus.WARNING,
            message="Mesh is not watertight",
            value=0.0,
            threshold=1.0,
            details=details,
        )
        
    except Exception as e:
        return CheckResult(
            name="shell_watertight",
            status=CheckStatus.FAILED,
            message=f"Failed to check watertightness: {e}",
        )


def check_minimum_diameter(
    void_mesh: "trimesh.Trimesh",
    min_diameter: float,
    sample_count: int = 500,
) -> CheckResult:
    """
    Check that void channels meet minimum diameter requirement.
    
    Uses ray casting to estimate local channel diameters.
    
    Parameters
    ----------
    void_mesh : trimesh.Trimesh
        Void mesh to check
    min_diameter : float
        Minimum required diameter (meters)
    sample_count : int
        Number of sample points to check
        
    Returns
    -------
    CheckResult
        Result of the check
    """
    try:
        # Sample points inside the void
        # Use centroid and random interior points
        points = []
        
        # Add centroid
        points.append(void_mesh.centroid)
        
        # Sample surface points and move inward
        surface_pts, face_indices = void_mesh.sample(sample_count, return_index=True)
        normals = void_mesh.face_normals[face_indices]
        
        # Move points inward along normals
        for pt, normal in zip(surface_pts, normals):
            interior_pt = pt - normal * min_diameter * 0.5
            points.append(interior_pt)
        
        points = np.array(points)
        
        # For each point, cast rays in multiple directions to estimate local diameter
        min_found_diameter = float('inf')
        below_threshold_count = 0
        
        directions = _get_sample_directions(6)  # 6 directions for quick check
        
        for pt in points[:100]:  # Check subset for performance
            local_diameters = []
            
            for direction in directions:
                # Cast ray in both directions
                try:
                    # Forward ray
                    locations_fwd, _, _ = void_mesh.ray.intersects_location(
                        ray_origins=[pt],
                        ray_directions=[direction],
                    )
                    
                    # Backward ray
                    locations_bwd, _, _ = void_mesh.ray.intersects_location(
                        ray_origins=[pt],
                        ray_directions=[-direction],
                    )
                    
                    if len(locations_fwd) > 0 and len(locations_bwd) > 0:
                        dist_fwd = np.linalg.norm(locations_fwd[0] - pt)
                        dist_bwd = np.linalg.norm(locations_bwd[0] - pt)
                        diameter = dist_fwd + dist_bwd
                        local_diameters.append(diameter)
                        
                except Exception:
                    pass
            
            if local_diameters:
                local_min = min(local_diameters)
                min_found_diameter = min(min_found_diameter, local_min)
                
                if local_min < min_diameter:
                    below_threshold_count += 1
        
        if min_found_diameter == float('inf'):
            return CheckResult(
                name="minimum_diameter",
                status=CheckStatus.WARNING,
                message="Could not measure channel diameters",
            )
        
        if min_found_diameter < min_diameter * 0.9:  # 10% tolerance
            return CheckResult(
                name="minimum_diameter",
                status=CheckStatus.FAILED,
                message=f"Minimum diameter {min_found_diameter*1000:.2f}mm < required {min_diameter*1000:.2f}mm",
                value=min_found_diameter,
                threshold=min_diameter,
                details={
                    "below_threshold_count": below_threshold_count,
                },
            )
        elif min_found_diameter < min_diameter:
            return CheckResult(
                name="minimum_diameter",
                status=CheckStatus.WARNING,
                message=f"Minimum diameter {min_found_diameter*1000:.2f}mm slightly below {min_diameter*1000:.2f}mm",
                value=min_found_diameter,
                threshold=min_diameter,
            )
        
        return CheckResult(
            name="minimum_diameter",
            status=CheckStatus.PASSED,
            message=f"Minimum diameter {min_found_diameter*1000:.2f}mm >= {min_diameter*1000:.2f}mm",
            value=min_found_diameter,
            threshold=min_diameter,
        )
        
    except Exception as e:
        return CheckResult(
            name="minimum_diameter",
            status=CheckStatus.FAILED,
            message=f"Failed to check minimum diameter: {e}",
        )


def _get_sample_directions(n: int) -> np.ndarray:
    """Get n evenly distributed directions on unit sphere."""
    directions = []
    
    # Use golden spiral for even distribution
    golden_ratio = (1 + np.sqrt(5)) / 2
    
    for i in range(n):
        theta = 2 * np.pi * i / golden_ratio
        phi = np.arccos(1 - 2 * (i + 0.5) / n)
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        directions.append([x, y, z])
    
    return np.array(directions)


def report_diameter_shrink_and_drift(
    original_min_diameter: float,
    effective_min_diameter: float,
    original_voxel_pitch: float,
    effective_voxel_pitch: float,
    shrink_warning_threshold: float = 0.1,
    drift_warning_threshold: float = 0.05,
) -> DiameterReport:
    """
    Report on diameter shrinkage and drift due to voxelization.
    
    Parameters
    ----------
    original_min_diameter : float
        Originally specified minimum diameter (meters)
    effective_min_diameter : float
        Effective minimum diameter after voxelization (meters)
    original_voxel_pitch : float
        Originally specified voxel pitch (meters)
    effective_voxel_pitch : float
        Effective voxel pitch used (meters)
    shrink_warning_threshold : float
        Threshold for shrink ratio warning (0.1 = 10%)
    drift_warning_threshold : float
        Threshold for drift distance warning as ratio of diameter
        
    Returns
    -------
    DiameterReport
        Report with shrinkage and drift information
    """
    warnings = []
    
    # Calculate shrink ratio
    if original_min_diameter > 0:
        shrink_ratio = 1.0 - (effective_min_diameter / original_min_diameter)
    else:
        shrink_ratio = 0.0
    
    # Calculate drift (approximated by voxel pitch change)
    drift_distance = abs(effective_voxel_pitch - original_voxel_pitch)
    
    # Generate warnings
    if shrink_ratio > shrink_warning_threshold:
        warnings.append(
            f"Diameter shrunk by {shrink_ratio:.1%} "
            f"({original_min_diameter*1000:.2f}mm -> {effective_min_diameter*1000:.2f}mm)"
        )
    
    if original_min_diameter > 0 and drift_distance / original_min_diameter > drift_warning_threshold:
        warnings.append(
            f"Voxel pitch changed from {original_voxel_pitch*1000:.3f}mm to {effective_voxel_pitch*1000:.3f}mm, "
            f"potential drift of {drift_distance*1000:.3f}mm"
        )
    
    if effective_voxel_pitch > original_voxel_pitch:
        warnings.append(
            f"Voxel pitch was upscaled from {original_voxel_pitch*1000:.3f}mm to {effective_voxel_pitch*1000:.3f}mm"
        )
    
    return DiameterReport(
        original_min_diameter=original_min_diameter,
        effective_min_diameter=effective_min_diameter,
        shrink_ratio=shrink_ratio,
        drift_distance=drift_distance,
        warnings=warnings,
    )


def run_void_validity_checks(
    void_mesh: "trimesh.Trimesh",
    domain: Optional[DomainSpec] = None,
    min_diameter: Optional[float] = None,
    max_components: int = 1,
) -> VoidValidityReport:
    """
    Run all void validity checks.
    
    Parameters
    ----------
    void_mesh : trimesh.Trimesh
        Void mesh to validate
    domain : DomainSpec, optional
        Domain for containment check
    min_diameter : float, optional
        Minimum diameter requirement (meters)
    max_components : int
        Maximum allowed components
        
    Returns
    -------
    VoidValidityReport
        Comprehensive validity report
    """
    checks = []
    warnings = []
    errors = []
    
    # Component check
    comp_result = check_void_components(void_mesh, max_components)
    checks.append(comp_result)
    
    if comp_result.status == CheckStatus.WARNING:
        warnings.append(comp_result.message)
    elif comp_result.status == CheckStatus.FAILED:
        errors.append(comp_result.message)
    
    # Domain containment check
    if domain is not None:
        domain_result = check_void_inside_domain(void_mesh, domain)
        checks.append(domain_result)
        
        if domain_result.status == CheckStatus.WARNING:
            warnings.append(domain_result.message)
        elif domain_result.status == CheckStatus.FAILED:
            errors.append(domain_result.message)
    
    # Watertight check
    watertight_result = check_shell_watertight(void_mesh)
    checks.append(watertight_result)
    
    if watertight_result.status == CheckStatus.WARNING:
        warnings.append(watertight_result.message)
    elif watertight_result.status == CheckStatus.FAILED:
        errors.append(watertight_result.message)
    
    # Minimum diameter check
    if min_diameter is not None:
        diameter_result = check_minimum_diameter(void_mesh, min_diameter)
        checks.append(diameter_result)
        
        if diameter_result.status == CheckStatus.WARNING:
            warnings.append(diameter_result.message)
        elif diameter_result.status == CheckStatus.FAILED:
            errors.append(diameter_result.message)
    
    # Determine overall status
    if errors:
        overall_status = CheckStatus.FAILED
    elif warnings:
        overall_status = CheckStatus.WARNING
    else:
        overall_status = CheckStatus.PASSED
    
    return VoidValidityReport(
        overall_status=overall_status,
        checks=checks,
        warnings=warnings,
        errors=errors,
        metadata={
            "vertex_count": len(void_mesh.vertices),
            "face_count": len(void_mesh.faces),
            "volume": abs(void_mesh.volume),
            "is_watertight": void_mesh.is_watertight,
        },
    )


__all__ = [
    "check_void_components",
    "check_void_inside_domain",
    "check_shell_watertight",
    "check_minimum_diameter",
    "report_diameter_shrink_and_drift",
    "run_void_validity_checks",
    "CheckStatus",
    "CheckResult",
    "VoidValidityReport",
    "DiameterReport",
]
