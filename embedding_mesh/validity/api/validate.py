"""
Public API for mesh and network validation.

This module provides the main entry points for validating vascular
network meshes with policy-controlled checks.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, TYPE_CHECKING, Literal
import logging

# Import ValidationPolicy from centralized aog_policies package
from aog_policies import ValidationPolicy

if TYPE_CHECKING:
    import trimesh
    from generation.core.network import VascularNetwork

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""
    check_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationReport:
    """Complete validation report."""
    passed: bool
    status: Literal["ok", "warnings", "fail"]
    checks: List[ValidationResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    requested_policy: Dict[str, Any] = field(default_factory=dict)
    effective_policy: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "status": self.status,
            "checks": [c.to_dict() for c in self.checks],
            "warnings": self.warnings,
            "errors": self.errors,
            "requested_policy": self.requested_policy,
            "effective_policy": self.effective_policy,
            "metadata": self.metadata,
        }


def validate_mesh(
    mesh: "trimesh.Trimesh",
    policy: Optional[ValidationPolicy] = None,
) -> ValidationReport:
    """
    Validate a mesh against the specified policy.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to validate
    policy : ValidationPolicy, optional
        Policy controlling which checks to run
        
    Returns
    -------
    ValidationReport
        Report with check results and status
    """
    if policy is None:
        policy = ValidationPolicy()
    
    checks = []
    warnings = []
    errors = []
    
    # Watertight check
    if policy.check_watertight:
        result = _check_watertight(mesh)
        checks.append(result)
        if not result.passed:
            errors.append(result.message)
    
    # Components check
    if policy.check_components:
        result = _check_components(mesh, policy.max_components)
        checks.append(result)
        if not result.passed:
            if result.details.get("component_count", 0) > policy.max_components * 2:
                errors.append(result.message)
            else:
                warnings.append(result.message)
    
    # Min diameter check
    if policy.check_min_diameter:
        result = _check_min_diameter(mesh, policy.min_diameter_threshold)
        checks.append(result)
        if not result.passed:
            warnings.append(result.message)
    
    # Bounds check
    if policy.check_bounds:
        result = _check_bounds(mesh)
        checks.append(result)
        if not result.passed:
            warnings.append(result.message)
    
    # Determine overall status
    all_passed = all(c.passed for c in checks)
    has_errors = len(errors) > 0
    has_warnings = len(warnings) > 0
    
    if has_errors:
        status = "fail"
        passed = False
    elif has_warnings:
        status = "warnings"
        passed = True
    else:
        status = "ok"
        passed = True
    
    return ValidationReport(
        passed=passed,
        status=status,
        checks=checks,
        warnings=warnings,
        errors=errors,
        requested_policy=policy.to_dict(),
        effective_policy=policy.to_dict(),
        metadata={
            "vertex_count": len(mesh.vertices),
            "face_count": len(mesh.faces),
        },
    )


def _check_watertight(mesh: "trimesh.Trimesh") -> ValidationResult:
    """Check if mesh is watertight."""
    is_watertight = mesh.is_watertight
    
    return ValidationResult(
        check_name="watertight",
        passed=is_watertight,
        message="Mesh is watertight" if is_watertight else "Mesh is not watertight",
        details={
            "is_watertight": is_watertight,
            "euler_number": mesh.euler_number if hasattr(mesh, 'euler_number') else None,
        },
    )


def _check_components(mesh: "trimesh.Trimesh", max_components: int) -> ValidationResult:
    """Check number of connected components."""
    parts = mesh.split(only_watertight=False)
    component_count = len(parts)
    
    passed = component_count <= max_components
    
    return ValidationResult(
        check_name="components",
        passed=passed,
        message=f"Mesh has {component_count} component(s)" + 
                ("" if passed else f" (max: {max_components})"),
        details={
            "component_count": component_count,
            "max_components": max_components,
        },
    )


def _check_min_diameter(mesh: "trimesh.Trimesh", threshold: float) -> ValidationResult:
    """Check minimum feature diameter."""
    import numpy as np
    
    # Estimate minimum diameter from bounding box
    extents = mesh.extents
    min_extent = float(np.min(extents))
    
    passed = min_extent >= threshold
    
    return ValidationResult(
        check_name="min_diameter",
        passed=passed,
        message=f"Minimum extent: {min_extent*1000:.3f}mm" +
                ("" if passed else f" (threshold: {threshold*1000:.3f}mm)"),
        details={
            "min_extent": min_extent,
            "threshold": threshold,
        },
    )


def _check_bounds(mesh: "trimesh.Trimesh") -> ValidationResult:
    """Check mesh bounds are reasonable."""
    import numpy as np
    
    extents = mesh.extents
    max_extent = float(np.max(extents))
    
    # Check for unreasonably large meshes (> 1 meter)
    passed = max_extent < 1.0
    
    return ValidationResult(
        check_name="bounds",
        passed=passed,
        message=f"Maximum extent: {max_extent*1000:.3f}mm" +
                ("" if passed else " (exceeds 1m, may be in wrong units)"),
        details={
            "extents": list(extents),
            "max_extent": max_extent,
        },
    )


def validate_network(
    network: "VascularNetwork",
    policy: Optional[ValidationPolicy] = None,
) -> ValidationReport:
    """
    Validate a vascular network.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to validate
    policy : ValidationPolicy, optional
        Policy controlling which checks to run
        
    Returns
    -------
    ValidationReport
        Report with check results and status
    """
    if policy is None:
        policy = ValidationPolicy()
    
    checks = []
    warnings = []
    errors = []
    
    # Check network has nodes
    if len(network.nodes) == 0:
        errors.append("Network has no nodes")
        checks.append(ValidationResult(
            check_name="has_nodes",
            passed=False,
            message="Network has no nodes",
        ))
    else:
        checks.append(ValidationResult(
            check_name="has_nodes",
            passed=True,
            message=f"Network has {len(network.nodes)} nodes",
            details={"node_count": len(network.nodes)},
        ))
    
    # Check network has segments
    if len(network.segments) == 0:
        errors.append("Network has no segments")
        checks.append(ValidationResult(
            check_name="has_segments",
            passed=False,
            message="Network has no segments",
        ))
    else:
        checks.append(ValidationResult(
            check_name="has_segments",
            passed=True,
            message=f"Network has {len(network.segments)} segments",
            details={"segment_count": len(network.segments)},
        ))
    
    # Check connectivity
    connectivity_result = _check_network_connectivity(network)
    checks.append(connectivity_result)
    if not connectivity_result.passed:
        warnings.append(connectivity_result.message)
    
    # Determine overall status
    has_errors = len(errors) > 0
    has_warnings = len(warnings) > 0
    
    if has_errors:
        status = "fail"
        passed = False
    elif has_warnings:
        status = "warnings"
        passed = True
    else:
        status = "ok"
        passed = True
    
    return ValidationReport(
        passed=passed,
        status=status,
        checks=checks,
        warnings=warnings,
        errors=errors,
        requested_policy=policy.to_dict(),
        effective_policy=policy.to_dict(),
        metadata={
            "node_count": len(network.nodes),
            "segment_count": len(network.segments),
        },
    )


def _check_network_connectivity(network: "VascularNetwork") -> ValidationResult:
    """Check network connectivity."""
    if not network.nodes:
        return ValidationResult(
            check_name="connectivity",
            passed=False,
            message="Network has no nodes",
        )
    
    # Build adjacency list
    adjacency = {nid: set() for nid in network.nodes}
    for segment in network.segments.values():
        if segment.start_node_id in adjacency and segment.end_node_id in adjacency:
            adjacency[segment.start_node_id].add(segment.end_node_id)
            adjacency[segment.end_node_id].add(segment.start_node_id)
    
    # Count connected components using BFS
    visited = set()
    num_components = 0
    
    for start_node in network.nodes:
        if start_node in visited:
            continue
        
        num_components += 1
        queue = [start_node]
        
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            
            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    queue.append(neighbor)
    
    is_connected = num_components == 1
    
    return ValidationResult(
        check_name="connectivity",
        passed=is_connected,
        message=f"Network has {num_components} connected component(s)",
        details={
            "num_components": num_components,
            "is_connected": is_connected,
        },
    )


def validate_artifacts(
    solid: "trimesh.Trimesh",
    void: "trimesh.Trimesh",
    shell: "trimesh.Trimesh",
    policy: Optional[ValidationPolicy] = None,
) -> ValidationReport:
    """
    Validate all artifacts from a generation run.
    
    Parameters
    ----------
    solid : trimesh.Trimesh
        Domain with void carved out
    void : trimesh.Trimesh
        Void mesh
    shell : trimesh.Trimesh
        Shell mesh
    policy : ValidationPolicy, optional
        Policy controlling which checks to run
        
    Returns
    -------
    ValidationReport
        Combined report for all artifacts
    """
    if policy is None:
        policy = ValidationPolicy()
    
    checks = []
    warnings = []
    errors = []
    
    # Validate solid
    solid_report = validate_mesh(solid, policy)
    for check in solid_report.checks:
        check.check_name = f"solid_{check.check_name}"
        checks.append(check)
    warnings.extend([f"solid: {w}" for w in solid_report.warnings])
    errors.extend([f"solid: {e}" for e in solid_report.errors])
    
    # Validate void
    void_report = validate_mesh(void, policy)
    for check in void_report.checks:
        check.check_name = f"void_{check.check_name}"
        checks.append(check)
    warnings.extend([f"void: {w}" for w in void_report.warnings])
    errors.extend([f"void: {e}" for e in void_report.errors])
    
    # Determine overall status
    has_errors = len(errors) > 0
    has_warnings = len(warnings) > 0
    
    if has_errors:
        status = "fail"
        passed = False
    elif has_warnings:
        status = "warnings"
        passed = True
    else:
        status = "ok"
        passed = True
    
    return ValidationReport(
        passed=passed,
        status=status,
        checks=checks,
        warnings=warnings,
        errors=errors,
        requested_policy=policy.to_dict(),
        effective_policy=policy.to_dict(),
        metadata={
            "solid_vertex_count": len(solid.vertices),
            "void_vertex_count": len(void.vertices),
            "shell_vertex_count": len(shell.vertices),
        },
    )


__all__ = [
    "validate_mesh",
    "validate_network",
    "validate_artifacts",
    "ValidationPolicy",
    "ValidationResult",
    "ValidationReport",
]
