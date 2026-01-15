"""
Validation pipeline combining validation and repair.

This module provides the validate-repair-validate pipeline for
ensuring mesh quality.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, TYPE_CHECKING
import logging

from .validate import validate_mesh, ValidationPolicy, ValidationReport
from .repair import repair_mesh, RepairPolicy, RepairReport

if TYPE_CHECKING:
    import trimesh

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


__all__ = [
    "validate_repair_validate",
    "run_full_pipeline",
    "PipelineReport",
]
