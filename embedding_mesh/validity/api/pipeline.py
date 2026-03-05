"""
Validation pipeline combining validation and repair.

This module provides the validate-repair-validate pipeline for
ensuring mesh quality.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple, TYPE_CHECKING
import logging

from .validate import validate_mesh, ValidationPolicy, ValidationReport
from .repair import repair_mesh, RepairPolicy, RepairReport

if TYPE_CHECKING:
    import trimesh
    from generation.core.network import VascularNetwork

logger = logging.getLogger(__name__)


@dataclass
class PipelineReport:
    """Report from validation pipeline."""
    success: bool
    initial_validation: Optional[ValidationReport] = None
    repair: Optional[RepairReport] = None
    final_validation: Optional[ValidationReport] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "initial_validation": self.initial_validation.to_dict() if self.initial_validation else None,
            "repair": self.repair.to_dict() if self.repair else None,
            "final_validation": self.final_validation.to_dict() if self.final_validation else None,
            "warnings": self.warnings,
            "errors": self.errors,
            "metadata": self.metadata,
        }


def validate_repair_validate(
    mesh: "trimesh.Trimesh",
    validation_policy: Optional[ValidationPolicy] = None,
    repair_policy: Optional[RepairPolicy] = None,
    skip_repair_if_valid: bool = True,
) -> tuple:
    """
    Run validate-repair-validate pipeline.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to process
    validation_policy : ValidationPolicy, optional
        Policy for validation checks
    repair_policy : RepairPolicy, optional
        Policy for repair operations
    skip_repair_if_valid : bool
        If True, skip repair if initial validation passes
        
    Returns
    -------
    result_mesh : trimesh.Trimesh
        Processed mesh (repaired if needed)
    report : PipelineReport
        Combined report from all stages
    """
    if validation_policy is None:
        validation_policy = ValidationPolicy()
    if repair_policy is None:
        repair_policy = RepairPolicy()
    
    warnings = []
    errors = []
    metadata = {}
    
    # Initial validation
    logger.info("Running initial validation...")
    initial_report = validate_mesh(mesh, validation_policy)
    metadata["initial_passed"] = initial_report.passed
    
    # Check if repair is needed
    if initial_report.passed and skip_repair_if_valid:
        logger.info("Initial validation passed, skipping repair")
        
        return mesh, PipelineReport(
            success=True,
            initial_validation=initial_report,
            repair=None,
            final_validation=initial_report,
            warnings=warnings,
            errors=errors,
            metadata=metadata,
        )
    
    # Run repair
    logger.info("Running repair...")
    repaired_mesh, repair_report = repair_mesh(mesh, repair_policy)
    warnings.extend(repair_report.warnings)
    metadata["repair_operations"] = repair_report.operations_applied
    
    # Final validation
    logger.info("Running final validation...")
    final_report = validate_mesh(repaired_mesh, validation_policy)
    metadata["final_passed"] = final_report.passed
    
    # Determine success
    success = final_report.passed
    if not success:
        errors.append("Final validation failed after repair")
    
    return repaired_mesh, PipelineReport(
        success=success,
        initial_validation=initial_report,
        repair=repair_report,
        final_validation=final_report,
        warnings=warnings,
        errors=errors,
        metadata=metadata,
    )


def run_full_pipeline(
    mesh: "trimesh.Trimesh",
    validation_policy: Optional[ValidationPolicy] = None,
    repair_policy: Optional[RepairPolicy] = None,
    max_repair_iterations: int = 3,
) -> tuple:
    """
    Run full validation pipeline with multiple repair iterations.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to process
    validation_policy : ValidationPolicy, optional
        Policy for validation checks
    repair_policy : RepairPolicy, optional
        Policy for repair operations
    max_repair_iterations : int
        Maximum number of repair iterations
        
    Returns
    -------
    result_mesh : trimesh.Trimesh
        Processed mesh
    report : PipelineReport
        Combined report
    """
    if validation_policy is None:
        validation_policy = ValidationPolicy()
    if repair_policy is None:
        repair_policy = RepairPolicy()
    
    current_mesh = mesh
    all_warnings = []
    all_errors = []
    iteration_reports = []
    
    for iteration in range(max_repair_iterations):
        logger.info(f"Pipeline iteration {iteration + 1}/{max_repair_iterations}")
        
        result_mesh, report = validate_repair_validate(
            current_mesh,
            validation_policy,
            repair_policy,
            skip_repair_if_valid=True,
        )
        
        iteration_reports.append(report.to_dict())
        all_warnings.extend(report.warnings)
        
        if report.success:
            logger.info(f"Pipeline succeeded on iteration {iteration + 1}")
            
            return result_mesh, PipelineReport(
                success=True,
                initial_validation=report.initial_validation,
                repair=report.repair,
                final_validation=report.final_validation,
                warnings=all_warnings,
                errors=all_errors,
                metadata={
                    "iterations": iteration + 1,
                    "iteration_reports": iteration_reports,
                },
            )
        
        current_mesh = result_mesh
    
    # Failed after all iterations
    all_errors.append(f"Pipeline failed after {max_repair_iterations} iterations")
    
    return current_mesh, PipelineReport(
        success=False,
        initial_validation=None,
        repair=None,
        final_validation=None,
        warnings=all_warnings,
        errors=all_errors,
        metadata={
            "iterations": max_repair_iterations,
            "iteration_reports": iteration_reports,
        },
    )


def validate_and_repair_artifacts(
    network: Optional["VascularNetwork"] = None,
    void_mesh: Optional["trimesh.Trimesh"] = None,
    domain_mesh: Optional["trimesh.Trimesh"] = None,
    embedded_mesh: Optional["trimesh.Trimesh"] = None,
    validation_policy: Optional[ValidationPolicy] = None,
    repair_policy: Optional[RepairPolicy] = None,
    check_pre_embedding: bool = True,
    check_post_embedding: bool = True,
    check_open_ports: bool = False,
) -> Tuple["trimesh.Trimesh", "ArtifactValidationReport"]:
    """
    Full validation and repair pipeline for generation artifacts.
    
    This is the canonical entrypoint for validating all artifacts from a
    generation run, supporting both pre-embedding and post-embedding checks.
    
    Parameters
    ----------
    network : VascularNetwork, optional
        Network to validate (pre-embedding)
    void_mesh : trimesh.Trimesh, optional
        Void mesh to validate
    domain_mesh : trimesh.Trimesh, optional
        Domain mesh for containment checks
    embedded_mesh : trimesh.Trimesh, optional
        Final embedded mesh (domain with void carved out)
    validation_policy : ValidationPolicy, optional
        Policy controlling validation checks
    repair_policy : RepairPolicy, optional
        Policy controlling repair operations
    check_pre_embedding : bool
        Whether to run pre-embedding checks (collision/topology on networks)
    check_post_embedding : bool
        Whether to run post-embedding checks (watertight, components, etc.)
    check_open_ports : bool
        Whether to check for open ports (not yet implemented)
        
    Returns
    -------
    result_mesh : trimesh.Trimesh
        Repaired embedded mesh (or void mesh if no embedded mesh provided)
    report : ArtifactValidationReport
        Comprehensive validation report
    """
    import trimesh
    from .validate import validate_mesh, validate_network, ValidationResult
    
    if validation_policy is None:
        validation_policy = ValidationPolicy()
    if repair_policy is None:
        repair_policy = RepairPolicy()
    
    warnings = []
    errors = []
    pre_embedding_checks = []
    post_embedding_checks = []
    metadata = {}
    
    # Handle open ports check (not implemented)
    if check_open_ports:
        warnings.append("check_open_ports=True: ignored (not implemented)")
        metadata["open_ports_check"] = "ignored (not implemented)"
    
    # Pre-embedding checks on network
    if check_pre_embedding and network is not None:
        logger.info("Running pre-embedding network checks...")
        
        # Collision/topology sanity checks
        net_report = validate_network(network, validation_policy)
        
        for check in net_report.checks:
            check.check_name = f"pre_embedding_{check.check_name}"
            pre_embedding_checks.append(check)
        
        warnings.extend([f"pre-embedding: {w}" for w in net_report.warnings])
        if not net_report.passed:
            errors.extend([f"pre-embedding: {e}" for e in net_report.errors])
        
        metadata["pre_embedding_passed"] = net_report.passed
    
    # Post-embedding checks
    result_mesh = embedded_mesh if embedded_mesh is not None else void_mesh
    
    if check_post_embedding and result_mesh is not None:
        logger.info("Running post-embedding checks...")
        
        # Watertight check
        watertight_result = ValidationResult(
            check_name="post_embedding_watertight",
            passed=result_mesh.is_watertight,
            message="Mesh is watertight" if result_mesh.is_watertight else "Mesh is not watertight",
            details={"is_watertight": result_mesh.is_watertight},
        )
        post_embedding_checks.append(watertight_result)
        if not watertight_result.passed:
            errors.append("post-embedding: Mesh is not watertight")
        
        # Void component count check
        if void_mesh is not None:
            void_parts = void_mesh.split(only_watertight=False)
            void_component_count = len(void_parts)
            
            max_void_components = validation_policy.max_components
            void_components_ok = void_component_count <= max_void_components
            
            void_comp_result = ValidationResult(
                check_name="post_embedding_void_components",
                passed=void_components_ok,
                message=f"Void has {void_component_count} component(s)" +
                        ("" if void_components_ok else f" (max: {max_void_components})"),
                details={
                    "void_component_count": void_component_count,
                    "max_components": max_void_components,
                },
            )
            post_embedding_checks.append(void_comp_result)
            
            if not void_components_ok:
                warnings.append(f"post-embedding: Void has {void_component_count} disconnected components")
        
        # Void inside domain check
        if void_mesh is not None and domain_mesh is not None:
            void_bounds = void_mesh.bounds
            domain_bounds = domain_mesh.bounds
            
            void_inside = (
                all(void_bounds[0] >= domain_bounds[0] - 1e-6) and
                all(void_bounds[1] <= domain_bounds[1] + 1e-6)
            )
            
            void_inside_result = ValidationResult(
                check_name="post_embedding_void_inside_domain",
                passed=void_inside,
                message="Void is inside domain bounds" if void_inside else "Void extends outside domain bounds",
                details={
                    "void_bounds": void_bounds.tolist() if hasattr(void_bounds, 'tolist') else list(void_bounds),
                    "domain_bounds": domain_bounds.tolist() if hasattr(domain_bounds, 'tolist') else list(domain_bounds),
                },
            )
            post_embedding_checks.append(void_inside_result)
            
            if not void_inside:
                warnings.append("post-embedding: Void extends outside domain bounds")
        
        # Printability proxies (min feature size)
        import numpy as np
        min_extent = float(np.min(result_mesh.extents))
        min_printable = 0.0002  # 0.2mm typical minimum feature size
        
        printability_ok = min_extent >= min_printable
        printability_result = ValidationResult(
            check_name="post_embedding_printability",
            passed=printability_ok,
            message=f"Minimum feature size: {min_extent*1000:.3f}mm" +
                    ("" if printability_ok else f" (below {min_printable*1000:.1f}mm threshold)"),
            details={
                "min_extent": min_extent,
                "min_printable_threshold": min_printable,
            },
        )
        post_embedding_checks.append(printability_result)
        
        if not printability_ok:
            warnings.append(f"post-embedding: Minimum feature size {min_extent*1000:.3f}mm may be too small for printing")
        
        # Drift/shrink warnings (check if mesh volume changed significantly)
        if void_mesh is not None and embedded_mesh is not None:
            void_volume = abs(void_mesh.volume)
            
            # Estimate expected void volume in embedded mesh
            # This is a heuristic - significant shrinkage indicates pitch was increased
            embedded_volume = abs(embedded_mesh.volume)
            domain_volume = abs(domain_mesh.volume) if domain_mesh is not None else embedded_volume * 2
            
            expected_void_fraction = void_volume / domain_volume if domain_volume > 0 else 0
            
            # If void volume is very small compared to original, warn about shrinkage
            if void_volume > 0:
                # Check if embedded mesh is significantly smaller than expected
                # This could indicate pitch was auto-adjusted
                metadata["void_volume"] = void_volume
                metadata["domain_volume"] = domain_volume
                metadata["void_fraction"] = expected_void_fraction
                
                if expected_void_fraction < 0.001:
                    warnings.append("post-embedding: Void volume is very small relative to domain - pitch may have been increased")
        
        metadata["post_embedding_passed"] = len([c for c in post_embedding_checks if not c.passed]) == 0
    
    # Run repair if needed
    repair_report = None
    if result_mesh is not None and not result_mesh.is_watertight:
        logger.info("Running repair on non-watertight mesh...")
        result_mesh, repair_report = repair_mesh(result_mesh, repair_policy)
        
        if repair_report.warnings:
            warnings.extend([f"repair: {w}" for w in repair_report.warnings])
    
    # Build comprehensive report
    all_checks = pre_embedding_checks + post_embedding_checks
    overall_passed = len(errors) == 0
    
    report = ArtifactValidationReport(
        success=overall_passed,
        pre_embedding_checks=pre_embedding_checks,
        post_embedding_checks=post_embedding_checks,
        repair=repair_report,
        warnings=warnings,
        errors=errors,
        metadata=metadata,
    )
    
    if result_mesh is None:
        result_mesh = trimesh.Trimesh()
    
    return result_mesh, report


@dataclass
class ArtifactValidationReport:
    """Report from artifact validation pipeline."""
    success: bool
    pre_embedding_checks: List["ValidationResult"] = field(default_factory=list)
    post_embedding_checks: List["ValidationResult"] = field(default_factory=list)
    repair: Optional[RepairReport] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        from .validate import ValidationResult
        return {
            "success": self.success,
            "pre_embedding_checks": [c.to_dict() for c in self.pre_embedding_checks],
            "post_embedding_checks": [c.to_dict() for c in self.post_embedding_checks],
            "repair": self.repair.to_dict() if self.repair else None,
            "warnings": self.warnings,
            "errors": self.errors,
            "metadata": self.metadata,
        }


__all__ = [
    "validate_repair_validate",
    "run_full_pipeline",
    "validate_and_repair_artifacts",
    "PipelineReport",
    "ArtifactValidationReport",
]
