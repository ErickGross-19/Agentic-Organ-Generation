"""
Canonical Validity Runner

G1 FIX: Single orchestrator for all validity checks.

This module provides a single entry point for running all validation checks
with policy-driven control. The runner takes artifacts and policies, runs
the checks registry, and aggregates results into a single JSON report.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, TYPE_CHECKING, Literal
import json
import logging
from pathlib import Path

from aog_policies.validity import ValidationPolicy, OpenPortPolicy, RepairPolicy
from aog_policies.resolution import ResolutionPolicy

if TYPE_CHECKING:
    import trimesh
    from generation.core.domain import Domain
    from generation.core.network import VascularNetwork

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """Result of a single validation check."""
    name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
            "warnings": self.warnings,
            "errors": self.errors,
        }


@dataclass
class ValidityReport:
    """
    Comprehensive validity report from the canonical runner.
    
    G1 FIX: Single JSON-serializable report with requested vs effective policies.
    """
    success: bool
    checks: List[CheckResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    requested_policies: Dict[str, Any] = field(default_factory=dict)
    effective_policies: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: Literal["ok", "warnings", "fail"] = "ok"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "checks": [c.to_dict() for c in self.checks],
            "warnings": self.warnings,
            "errors": self.errors,
            "requested_policies": self.requested_policies,
            "effective_policies": self.effective_policies,
            "metadata": self.metadata,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            f.write(self.to_json())


def run_validity_checks(
    domain: Optional["Domain"] = None,
    domain_mesh: Optional["trimesh.Trimesh"] = None,
    domain_with_void: Optional["trimesh.Trimesh"] = None,
    void_mesh: Optional["trimesh.Trimesh"] = None,
    network: Optional["VascularNetwork"] = None,
    ports: Optional[List[Dict[str, Any]]] = None,
    validation_policy: Optional[ValidationPolicy] = None,
    open_port_policy: Optional[OpenPortPolicy] = None,
    resolution_policy: Optional[ResolutionPolicy] = None,
    repair_policy: Optional[RepairPolicy] = None,
) -> ValidityReport:
    """
    Run all validity checks with policy-driven control.
    
    G1 FIX: Single entry point for the full validation suite.
    
    Parameters
    ----------
    domain : Domain, optional
        The domain object (for signed distance checks).
    domain_mesh : trimesh.Trimesh, optional
        The domain mesh (before void embedding).
    domain_with_void : trimesh.Trimesh, optional
        The domain mesh with void carved out.
    void_mesh : trimesh.Trimesh, optional
        The void mesh.
    network : VascularNetwork, optional
        The vascular network (for graph/flow checks).
    ports : list of dict, optional
        Port specifications for open-port validation.
    validation_policy : ValidationPolicy, optional
        Policy controlling which checks to run.
    open_port_policy : OpenPortPolicy, optional
        Policy for open-port validation.
    resolution_policy : ResolutionPolicy, optional
        Policy for resolution/pitch selection.
    repair_policy : RepairPolicy, optional
        Policy for mesh repair operations.
    
    Returns
    -------
    ValidityReport
        Comprehensive validation report with all check results.
    """
    if validation_policy is None:
        validation_policy = ValidationPolicy()
    if open_port_policy is None:
        open_port_policy = OpenPortPolicy()
    if resolution_policy is None:
        resolution_policy = ResolutionPolicy()
    if repair_policy is None:
        repair_policy = RepairPolicy()
    
    checks: List[CheckResult] = []
    all_warnings: List[str] = []
    all_errors: List[str] = []
    effective_policies: Dict[str, Any] = {}
    
    effective_policies["validation"] = validation_policy.to_dict()
    effective_policies["open_port"] = open_port_policy.to_dict()
    effective_policies["resolution"] = resolution_policy.to_dict()
    effective_policies["repair"] = repair_policy.to_dict()
    
    if domain_with_void is not None:
        if validation_policy.check_watertight:
            result = _check_watertight(domain_with_void, "domain_with_void")
            checks.append(result)
            if not result.passed:
                all_errors.append(result.message)
            all_warnings.extend(result.warnings)
        
        if validation_policy.check_components:
            result = _check_components(
                domain_with_void, 
                "domain_with_void",
                validation_policy.max_components,
            )
            checks.append(result)
            if not result.passed:
                all_errors.append(result.message)
            all_warnings.extend(result.warnings)
    
    if void_mesh is not None:
        if validation_policy.check_watertight:
            result = _check_watertight(void_mesh, "void")
            checks.append(result)
            if not result.passed:
                all_warnings.append(result.message)
            all_warnings.extend(result.warnings)
        
        if validation_policy.check_components:
            result = _check_components(void_mesh, "void", validation_policy.max_components)
            checks.append(result)
            all_warnings.extend(result.warnings)
        
        if validation_policy.check_min_diameter:
            result = _check_min_diameter(
                void_mesh, 
                "void",
                validation_policy.min_diameter_threshold,
            )
            checks.append(result)
            all_warnings.extend(result.warnings)
    
    if domain is not None and void_mesh is not None:
        if validation_policy.check_void_inside_domain:
            result = _check_void_inside_domain(
                void_mesh, 
                domain,
                ports=ports,
                validation_policy=validation_policy,
            )
            checks.append(result)
            if not result.passed:
                all_errors.append(result.message)
            all_warnings.extend(result.warnings)
    
    if (
        validation_policy.check_open_ports
        and open_port_policy.enabled
        and ports
        and domain_with_void is not None
        and domain_mesh is not None
    ):
        from .checks.open_ports import check_open_ports
        
        open_port_result = check_open_ports(
            ports=ports,
            domain_with_void_mesh=domain_with_void,
            original_domain_mesh=domain_mesh,
            policy=open_port_policy,
        )
        
        for port_result in open_port_result.port_results:
            checks.append(CheckResult(
                name=f"open_port_{port_result.port_id}",
                passed=port_result.is_open,
                message=f"Port {port_result.port_id} is {'open' if port_result.is_open else 'closed'}",
                details=port_result.to_dict(),
                warnings=port_result.warnings,
                errors=port_result.errors,
            ))
        
        all_warnings.extend(open_port_result.warnings)
        all_errors.extend(open_port_result.errors)
        
        effective_policies["open_port_result"] = {
            "ports_checked": open_port_result.ports_checked,
            "ports_open": open_port_result.ports_open,
            "ports_closed": open_port_result.ports_closed,
        }
    
    all_passed = all(c.passed for c in checks) if checks else True
    has_errors = len(all_errors) > 0
    has_warnings = len(all_warnings) > 0
    
    if has_errors:
        status: Literal["ok", "warnings", "fail"] = "fail"
        passed = False
    elif has_warnings:
        status = "warnings"
        passed = all_passed
    else:
        status = "ok"
        passed = True
    
    metadata = {
        "total_checks": len(checks),
        "passed_checks": sum(1 for c in checks if c.passed),
        "failed_checks": sum(1 for c in checks if not c.passed),
    }
    
    if domain_with_void is not None:
        metadata["domain_with_void_vertices"] = len(domain_with_void.vertices)
        metadata["domain_with_void_faces"] = len(domain_with_void.faces)
    
    if void_mesh is not None:
        metadata["void_vertices"] = len(void_mesh.vertices)
        metadata["void_faces"] = len(void_mesh.faces)
    
    return ValidityReport(
        success=passed,
        status=status,
        checks=checks,
        warnings=all_warnings,
        errors=all_errors,
        requested_policies={
            "validation": validation_policy.to_dict(),
            "open_port": open_port_policy.to_dict(),
            "resolution": resolution_policy.to_dict(),
            "repair": repair_policy.to_dict(),
        },
        effective_policies=effective_policies,
        metadata=metadata,
    )


def _check_watertight(mesh: "trimesh.Trimesh", mesh_name: str) -> CheckResult:
    """Check if mesh is watertight."""
    is_watertight = mesh.is_watertight
    
    return CheckResult(
        name=f"{mesh_name}_watertight",
        passed=is_watertight,
        message=f"{mesh_name} is {'watertight' if is_watertight else 'not watertight'}",
        details={
            "is_watertight": is_watertight,
            "euler_number": mesh.euler_number if hasattr(mesh, "euler_number") else None,
        },
    )


def _check_components(
    mesh: "trimesh.Trimesh", 
    mesh_name: str, 
    max_components: int,
) -> CheckResult:
    """Check number of connected components."""
    parts = mesh.split(only_watertight=False)
    component_count = len(parts)
    
    passed = component_count <= max_components
    
    return CheckResult(
        name=f"{mesh_name}_components",
        passed=passed,
        message=f"{mesh_name} has {component_count} component(s)" + 
                ("" if passed else f" (max: {max_components})"),
        details={
            "component_count": component_count,
            "max_components": max_components,
        },
    )


def _check_min_diameter(
    mesh: "trimesh.Trimesh", 
    mesh_name: str,
    threshold: float,
) -> CheckResult:
    """Check minimum feature diameter."""
    import numpy as np
    
    extents = mesh.extents
    min_extent = float(np.min(extents))
    
    passed = min_extent >= threshold
    
    return CheckResult(
        name=f"{mesh_name}_min_diameter",
        passed=passed,
        message=f"{mesh_name} minimum extent: {min_extent*1000:.3f}mm" +
                ("" if passed else f" (threshold: {threshold*1000:.3f}mm)"),
        details={
            "min_extent": min_extent,
            "threshold": threshold,
        },
    )


def _check_void_inside_domain(
    void_mesh: "trimesh.Trimesh",
    domain: "Domain",
    ports: Optional[List[Dict[str, Any]]] = None,
    validation_policy: Optional[ValidationPolicy] = None,
) -> CheckResult:
    """
    Check that void mesh is inside domain.
    
    Supports surface opening semantics: when validation_policy.allow_boundary_intersections_at_ports
    is True, ports marked with is_surface_opening=True are allowed to intersect the domain boundary.
    
    Parameters
    ----------
    void_mesh : trimesh.Trimesh
        The void mesh to check
    domain : Domain
        The domain that should contain the void
    ports : list of dict, optional
        Port specifications. Ports with is_surface_opening=True are treated as surface openings.
    validation_policy : ValidationPolicy, optional
        Policy controlling validation behavior including surface opening support.
    """
    import numpy as np
    from generation.core.types import Point3D
    from generation.ops.validity.void_checks import SurfaceOpeningPort
    
    if validation_policy is None:
        validation_policy = ValidationPolicy()
    
    # Extract surface opening ports if enabled
    surface_opening_ports = []
    if validation_policy.allow_boundary_intersections_at_ports and ports:
        for port in ports:
            # Check if port is marked as surface opening
            if port.get("is_surface_opening", False):
                position = np.array(port.get("position", [0, 0, 0]))
                direction = np.array(port.get("direction", [0, 0, -1]))
                # Normalize direction
                dir_norm = np.linalg.norm(direction)
                if dir_norm > 0:
                    direction = direction / dir_norm
                radius = port.get("radius", 0.001)
                name = port.get("name", "unknown")
                
                surface_opening_ports.append(SurfaceOpeningPort(
                    name=name,
                    position=position,
                    direction=direction,
                    radius=radius,
                ))
    
    sample_points = void_mesh.vertices[::max(1, len(void_mesh.vertices) // 100)]
    
    inside_count = 0
    outside_count = 0
    outside_in_opening_count = 0
    outside_not_in_opening_count = 0
    
    for point in sample_points:
        # Convert numpy array to Point3D for domain methods
        point_3d = Point3D.from_array(point)
        sd = domain.signed_distance(point_3d)
        if sd <= 0:
            inside_count += 1
        else:
            outside_count += 1
            
            # Check if point is in any surface opening neighborhood
            in_opening = False
            if surface_opening_ports:
                for port in surface_opening_ports:
                    if port.is_point_in_neighborhood(
                        point, 
                        validation_policy.surface_opening_tolerance
                    ):
                        in_opening = True
                        break
            
            if in_opening:
                outside_in_opening_count += 1
            else:
                outside_not_in_opening_count += 1
    
    total_samples = len(sample_points)
    fraction_inside = inside_count / total_samples if total_samples > 0 else 0
    outside_ratio = outside_count / total_samples if total_samples > 0 else 0
    outside_not_in_opening_ratio = outside_not_in_opening_count / total_samples if total_samples > 0 else 0
    outside_in_opening_ratio = outside_in_opening_count / total_samples if total_samples > 0 else 0
    
    # When surface openings are enabled, only count points outside that are
    # NOT in allowed opening regions as violations
    if surface_opening_ports:
        effective_outside_ratio = outside_not_in_opening_ratio
        passed = effective_outside_ratio <= 0.05  # Allow up to 5% outside (excluding openings)
        
        warnings = []
        if effective_outside_ratio > 0.01 and effective_outside_ratio <= 0.05:
            warnings.append(
                f"Some void vertices ({outside_not_in_opening_count}/{total_samples}) are outside domain "
                f"(excluding {outside_in_opening_count} in allowed openings)"
            )
        
        message = (
            f"Void is {'inside' if passed else 'not fully inside'} domain "
            f"({fraction_inside*100:.1f}% inside, {outside_in_opening_ratio*100:.1f}% in allowed openings)"
        )
    else:
        passed = fraction_inside >= 0.95
        
        warnings = []
        if fraction_inside < 1.0 and fraction_inside >= 0.95:
            warnings.append(
                f"Some void vertices ({outside_count}/{total_samples}) are outside domain"
            )
        
        message = f"Void is {'inside' if passed else 'not fully inside'} domain ({fraction_inside*100:.1f}% inside)"
    
    return CheckResult(
        name="void_inside_domain",
        passed=passed,
        message=message,
        details={
            "sample_count": total_samples,
            "inside_count": inside_count,
            "outside_count": outside_count,
            "outside_in_opening_count": outside_in_opening_count,
            "outside_not_in_opening_count": outside_not_in_opening_count,
            "fraction_inside": fraction_inside,
            "surface_openings_enabled": len(surface_opening_ports) > 0,
            "surface_opening_port_count": len(surface_opening_ports),
        },
        warnings=warnings,
    )


__all__ = [
    "CheckResult",
    "ValidityReport",
    "run_validity_checks",
]
