"""
Validation Orchestrators

High-level functions that orchestrate pre-embedding and post-embedding validation.
These are the main entry points for running comprehensive validation checks.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, TYPE_CHECKING
from pathlib import Path
import json
import trimesh

if TYPE_CHECKING:
    from generation.core import VascularNetwork

from .pre_embedding.mesh_checks import run_all_mesh_checks, MeshCheckReport
from .pre_embedding.graph_checks import run_all_graph_checks, GraphCheckReport
from .pre_embedding.flow_checks import run_all_flow_checks, FlowCheckReport
from .post_embedding.connectivity_checks import run_all_connectivity_checks, ConnectivityCheckReport
from .post_embedding.printability_checks import (
    run_all_printability_checks, 
    PrintabilityCheckReport,
    ManufacturingConfig,
)
from .post_embedding.domain_checks import run_all_domain_checks, DomainCheckReport


@dataclass
class ValidationConfig:
    """
    Configuration for validation checks.
    
    All length units are in METERS (internal system units) unless otherwise noted.
    
    Attributes
    ----------
    voxel_pitch_m : float
        Voxel pitch for mesh analysis in meters.
        Default 0.0001 = 0.1mm, suitable for fine vascular structures.
    murray_gamma : float
        Murray's law exponent
    murray_tolerance : float
        Maximum acceptable Murray's law deviation
    max_branch_order : int
        Maximum expected branch order
    min_clearance : float
        Minimum required clearance between segments (meters)
    max_collisions : int
        Maximum acceptable number of collisions
    max_flow_balance_error : float
        Maximum acceptable flow balance error at junctions
    max_reynolds : float
        Maximum acceptable Reynolds number
    min_port_components : int
        Minimum required number of port components
    max_trapped_fraction : float
        Maximum acceptable fraction of trapped fluid
    expected_outlets : int
        Expected number of outlet openings
    manufacturing : ManufacturingConfig
        Manufacturing constraints for printability checks
    """
    voxel_pitch_m: float = 0.0001  # 0.1mm in meters
    murray_gamma: float = 3.0
    murray_tolerance: float = 0.15
    max_branch_order: int = 20
    min_clearance: float = 0.0
    max_collisions: int = 0
    max_flow_balance_error: float = 0.05
    max_reynolds: float = 2300.0
    min_port_components: int = 1
    max_trapped_fraction: float = 0.05
    expected_outlets: int = 2
    manufacturing: ManufacturingConfig = field(default_factory=ManufacturingConfig)


@dataclass
class ValidationReport:
    """
    Comprehensive validation report.
    
    Attributes
    ----------
    passed : bool
        Whether all critical checks passed
    status : str
        Overall status: "ok", "warnings", "fail"
    stage : str
        Validation stage: "pre_embedding" or "post_embedding"
    reports : Dict[str, Any]
        Individual check reports by category
    summary : Dict[str, Any]
        Summary statistics
    config : ValidationConfig
        Configuration used for validation
    """
    passed: bool
    status: str
    stage: str
    reports: Dict[str, Any]
    summary: Dict[str, Any]
    config: ValidationConfig
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "passed": self.passed,
            "status": self.status,
            "stage": self.stage,
            "summary": self.summary,
            "reports": {
                name: {
                    "passed": report.passed,
                    "status": report.status,
                    "summary": report.summary,
                    "checks": [
                        {
                            "check_name": c.check_name,
                            "passed": c.passed,
                            "message": c.message,
                            "details": c.details,
                            "warnings": c.warnings,
                        }
                        for c in report.checks
                    ]
                }
                for name, report in self.reports.items()
            },
        }
    
    def save_json(self, path: Union[str, Path]) -> None:
        """Save report to JSON file."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def run_pre_embedding_validation(
    mesh_path: Optional[Union[str, Path]] = None,
    mesh: Optional[trimesh.Trimesh] = None,
    network: Optional["VascularNetwork"] = None,
    config: Optional[ValidationConfig] = None,
) -> ValidationReport:
    """
    Run all pre-embedding validation checks.
    
    Pre-embedding checks validate the structure BEFORE it is embedded
    into a domain. They focus on intrinsic properties of the mesh and network.
    
    Parameters
    ----------
    mesh_path : str or Path, optional
        Path to mesh file (STL, OBJ, etc.)
    mesh : trimesh.Trimesh, optional
        Pre-loaded mesh object
    network : VascularNetwork, optional
        Vascular network for graph and flow checks
    config : ValidationConfig, optional
        Validation configuration (uses defaults if not provided)
        
    Returns
    -------
    ValidationReport
        Comprehensive validation report
        
    Examples
    --------
    >>> from validity import run_pre_embedding_validation
    >>> 
    >>> # Validate a mesh file
    >>> report = run_pre_embedding_validation(mesh_path="structure.stl")
    >>> print(f"Status: {report.status}")
    >>> 
    >>> # Validate with network for flow checks
    >>> report = run_pre_embedding_validation(
    ...     mesh_path="structure.stl",
    ...     network=vascular_network,
    ... )
    """
    if config is None:
        config = ValidationConfig()
    
    # Load mesh if path provided
    if mesh is None and mesh_path is not None:
        mesh = trimesh.load(mesh_path)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Expected Trimesh, got {type(mesh)}")
    
    reports = {}
    
    # Run mesh checks if mesh is available
    if mesh is not None:
        reports["mesh"] = run_all_mesh_checks(mesh)
    
    # Run graph and flow checks if network is available
    if network is not None:
        reports["graph"] = run_all_graph_checks(
            network,
            murray_gamma=config.murray_gamma,
            murray_tolerance=config.murray_tolerance,
            max_branch_order=config.max_branch_order,
            min_clearance=config.min_clearance,
            max_collisions=config.max_collisions,
        )
        reports["flow"] = run_all_flow_checks(
            network,
            max_flow_balance_error=config.max_flow_balance_error,
            max_reynolds=config.max_reynolds,
        )
    
    # Aggregate results
    all_passed = all(r.passed for r in reports.values())
    has_warnings = any(
        any(len(c.warnings) > 0 for c in r.checks)
        for r in reports.values()
    )
    
    if all_passed and not has_warnings:
        status = "ok"
    elif all_passed:
        status = "warnings"
    else:
        status = "fail"
    
    summary = {
        "total_categories": len(reports),
        "passed_categories": sum(1 for r in reports.values() if r.passed),
        "failed_categories": sum(1 for r in reports.values() if not r.passed),
        "total_checks": sum(len(r.checks) for r in reports.values()),
        "passed_checks": sum(
            sum(1 for c in r.checks if c.passed)
            for r in reports.values()
        ),
        "failed_checks": sum(
            sum(1 for c in r.checks if not c.passed)
            for r in reports.values()
        ),
        "total_warnings": sum(
            sum(len(c.warnings) for c in r.checks)
            for r in reports.values()
        ),
    }
    
    return ValidationReport(
        passed=all_passed,
        status=status,
        stage="pre_embedding",
        reports=reports,
        summary=summary,
        config=config,
    )


def run_post_embedding_validation(
    mesh_path: Optional[Union[str, Path]] = None,
    mesh: Optional[trimesh.Trimesh] = None,
    config: Optional[ValidationConfig] = None,
    manufacturing_config: Optional[Union[ManufacturingConfig, Dict[str, Any]]] = None,
) -> ValidationReport:
    """
    Run all post-embedding validation checks.
    
    Post-embedding checks validate the structure AFTER it is embedded
    into a domain. They focus on manufacturability, connectivity, and
    physical constraints.
    
    Parameters
    ----------
    mesh_path : str or Path, optional
        Path to embedded mesh file (domain with void)
    mesh : trimesh.Trimesh, optional
        Pre-loaded embedded mesh object
    config : ValidationConfig, optional
        Validation configuration (uses defaults if not provided)
    manufacturing_config : ManufacturingConfig or dict, optional
        Manufacturing constraints. Can be a ManufacturingConfig object
        or a dict with keys like 'min_channel_diameter', 'min_wall_thickness',
        'plate_size', etc.
        
    Returns
    -------
    ValidationReport
        Comprehensive validation report
        
    Examples
    --------
    >>> from validity import run_post_embedding_validation
    >>> 
    >>> # Validate an embedded mesh
    >>> report = run_post_embedding_validation(
    ...     mesh_path="domain_with_void.stl",
    ...     manufacturing_config={
    ...         "min_channel_diameter": 0.5,
    ...         "min_wall_thickness": 0.3,
    ...         "plate_size": (200, 200, 200),
    ...     }
    ... )
    >>> print(f"Status: {report.status}")
    """
    if config is None:
        config = ValidationConfig()
    
    # Handle manufacturing config
    if manufacturing_config is not None:
        if isinstance(manufacturing_config, dict):
            config.manufacturing = ManufacturingConfig(**manufacturing_config)
        else:
            config.manufacturing = manufacturing_config
    
    # Load mesh if path provided
    if mesh is None and mesh_path is not None:
        mesh = trimesh.load(mesh_path)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Expected Trimesh, got {type(mesh)}")
    
    if mesh is None:
        raise ValueError("Either mesh_path or mesh must be provided")
    
    reports = {}
    
    # Run connectivity checks
    reports["connectivity"] = run_all_connectivity_checks(
        mesh,
        pitch=config.voxel_pitch_m,
        min_port_components=config.min_port_components,
        max_trapped_fraction=config.max_trapped_fraction,
    )
    
    # Run printability checks
    reports["printability"] = run_all_printability_checks(
        mesh,
        config=config.manufacturing,
    )
    
    # Run domain checks
    reports["domain"] = run_all_domain_checks(
        mesh,
        expected_outlets=config.expected_outlets,
        pitch=config.voxel_pitch_m,
    )
    
    # Aggregate results
    all_passed = all(r.passed for r in reports.values())
    has_warnings = any(
        any(len(c.warnings) > 0 for c in r.checks)
        for r in reports.values()
    )
    
    if all_passed and not has_warnings:
        status = "ok"
    elif all_passed:
        status = "warnings"
    else:
        status = "fail"
    
    summary = {
        "total_categories": len(reports),
        "passed_categories": sum(1 for r in reports.values() if r.passed),
        "failed_categories": sum(1 for r in reports.values() if not r.passed),
        "total_checks": sum(len(r.checks) for r in reports.values()),
        "passed_checks": sum(
            sum(1 for c in r.checks if c.passed)
            for r in reports.values()
        ),
        "failed_checks": sum(
            sum(1 for c in r.checks if not c.passed)
            for r in reports.values()
        ),
        "total_warnings": sum(
            sum(len(c.warnings) for c in r.checks)
            for r in reports.values()
        ),
    }
    
    return ValidationReport(
        passed=all_passed,
        status=status,
        stage="post_embedding",
        reports=reports,
        summary=summary,
        config=config,
    )
