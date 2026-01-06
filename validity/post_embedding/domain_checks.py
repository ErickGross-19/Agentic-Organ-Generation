"""
Post-Embedding Domain Checks

Validates domain-specific constraints after embedding structure.
Checks include outlet openness and domain coverage.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy import ndimage
import trimesh


@dataclass
class DomainCheckResult:
    """Result of a domain check."""
    passed: bool
    check_name: str
    message: str
    details: Dict[str, Any]
    warnings: List[str]


def check_outlets_open(
    mesh: trimesh.Trimesh,
    expected_outlets: int = 2,
    pitch: float = 0.1,
) -> DomainCheckResult:
    """
    Check that outlets are open and not covered by domain material.
    
    Outlets should be accessible from the exterior of the domain
    for fluid to enter and exit.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The embedded mesh (domain with void)
    expected_outlets : int
        Expected number of outlet openings
    pitch : float
        Voxel pitch for analysis
        
    Returns
    -------
    DomainCheckResult
        Result with pass/fail status and details
    """
    try:
        vox = mesh.voxelized(pitch)
        vox_filled = vox.fill()
        fluid_mask = vox_filled.matrix.astype(bool)
    except Exception as e:
        return DomainCheckResult(
            passed=False,
            check_name="outlets_open",
            message=f"Voxelization failed: {e}",
            details={"error": str(e)},
            warnings=[str(e)],
        )
    
    if not fluid_mask.any():
        return DomainCheckResult(
            passed=False,
            check_name="outlets_open",
            message="No fluid volume found",
            details={"error": "Empty fluid mask"},
            warnings=["No fluid volume detected"],
        )
    
    # Count openings on each face of the bounding box
    face_openings = {
        "x_min": 0,
        "x_max": 0,
        "y_min": 0,
        "y_max": 0,
        "z_min": 0,
        "z_max": 0,
    }
    
    # Label connected components on each face
    structure_2d = ndimage.generate_binary_structure(rank=2, connectivity=1)
    
    # X faces
    if fluid_mask[0, :, :].any():
        _, n = ndimage.label(fluid_mask[0, :, :], structure=structure_2d)
        face_openings["x_min"] = n
    if fluid_mask[-1, :, :].any():
        _, n = ndimage.label(fluid_mask[-1, :, :], structure=structure_2d)
        face_openings["x_max"] = n
    
    # Y faces
    if fluid_mask[:, 0, :].any():
        _, n = ndimage.label(fluid_mask[:, 0, :], structure=structure_2d)
        face_openings["y_min"] = n
    if fluid_mask[:, -1, :].any():
        _, n = ndimage.label(fluid_mask[:, -1, :], structure=structure_2d)
        face_openings["y_max"] = n
    
    # Z faces
    if fluid_mask[:, :, 0].any():
        _, n = ndimage.label(fluid_mask[:, :, 0], structure=structure_2d)
        face_openings["z_min"] = n
    if fluid_mask[:, :, -1].any():
        _, n = ndimage.label(fluid_mask[:, :, -1], structure=structure_2d)
        face_openings["z_max"] = n
    
    total_openings = sum(face_openings.values())
    passed = total_openings >= expected_outlets
    
    details = {
        "face_openings": face_openings,
        "total_openings": total_openings,
        "expected_outlets": expected_outlets,
        "grid_shape": list(fluid_mask.shape),
        "pitch": pitch,
    }
    
    warnings = []
    if total_openings < expected_outlets:
        warnings.append(f"Found {total_openings} openings, expected at least {expected_outlets}")
    if total_openings == 0:
        warnings.append("No openings found - structure is completely enclosed")
    
    return DomainCheckResult(
        passed=passed,
        check_name="outlets_open",
        message=f"{total_openings} outlet opening(s) found (expected: {expected_outlets})",
        details=details,
        warnings=warnings,
    )


def check_domain_coverage(
    mesh: trimesh.Trimesh,
    domain_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    min_coverage_fraction: float = 0.01,
    max_coverage_fraction: float = 0.5,
) -> DomainCheckResult:
    """
    Check that the vascular structure appropriately covers the domain.
    
    Too little coverage means poor perfusion; too much coverage
    means insufficient solid material for structural integrity.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The embedded mesh (domain with void)
    domain_bounds : tuple of arrays, optional
        (min_corner, max_corner) of the domain. If None, uses mesh bounds.
    min_coverage_fraction : float
        Minimum acceptable void fraction
    max_coverage_fraction : float
        Maximum acceptable void fraction
        
    Returns
    -------
    DomainCheckResult
        Result with pass/fail status and details
    """
    # Get mesh bounds
    mesh_min, mesh_max = mesh.bounds
    mesh_size = mesh_max - mesh_min
    
    if domain_bounds is not None:
        domain_min, domain_max = domain_bounds
        domain_size = domain_max - domain_min
    else:
        domain_min, domain_max = mesh_min, mesh_max
        domain_size = mesh_size
    
    # Calculate volumes
    mesh_volume = mesh.volume if mesh.is_watertight else None
    domain_volume = float(np.prod(domain_size))
    
    if mesh_volume is not None and domain_volume > 0:
        # For a domain-with-void mesh, the mesh volume is the solid part
        # The void fraction is 1 - (solid_volume / domain_volume)
        solid_fraction = abs(mesh_volume) / domain_volume
        void_fraction = 1.0 - solid_fraction
    else:
        # Estimate using voxelization
        try:
            pitch = min(mesh_size) / 50.0
            vox = mesh.voxelized(pitch)
            vox_filled = vox.fill()
            solid_mask = vox_filled.matrix.astype(bool)
            solid_fraction = solid_mask.sum() / solid_mask.size
            void_fraction = 1.0 - solid_fraction
        except Exception:
            solid_fraction = None
            void_fraction = None
    
    if void_fraction is not None:
        passed = min_coverage_fraction <= void_fraction <= max_coverage_fraction
    else:
        passed = True  # Can't determine, assume OK
    
    details = {
        "mesh_bounds": {
            "min": mesh_min.tolist(),
            "max": mesh_max.tolist(),
            "size": mesh_size.tolist(),
        },
        "domain_volume": domain_volume,
        "mesh_volume": float(mesh_volume) if mesh_volume is not None else None,
        "solid_fraction": float(solid_fraction) if solid_fraction is not None else None,
        "void_fraction": float(void_fraction) if void_fraction is not None else None,
        "min_coverage_fraction": min_coverage_fraction,
        "max_coverage_fraction": max_coverage_fraction,
        "is_watertight": mesh.is_watertight,
    }
    
    warnings = []
    if void_fraction is not None:
        if void_fraction < min_coverage_fraction:
            warnings.append(f"Void fraction {void_fraction:.1%} below minimum {min_coverage_fraction:.1%}")
        if void_fraction > max_coverage_fraction:
            warnings.append(f"Void fraction {void_fraction:.1%} above maximum {max_coverage_fraction:.1%}")
    else:
        warnings.append("Could not determine void fraction")
    
    return DomainCheckResult(
        passed=passed,
        check_name="domain_coverage",
        message=f"Void fraction: {void_fraction:.1%}" if void_fraction is not None else "Could not determine coverage",
        details=details,
        warnings=warnings,
    )


@dataclass
class DomainCheckReport:
    """Aggregated report of all domain checks."""
    passed: bool
    status: str  # "ok", "warnings", "fail"
    checks: List[DomainCheckResult]
    summary: Dict[str, Any]


def run_all_domain_checks(
    mesh: trimesh.Trimesh,
    expected_outlets: int = 2,
    pitch: float = 0.1,
    min_coverage_fraction: float = 0.01,
    max_coverage_fraction: float = 0.5,
) -> DomainCheckReport:
    """
    Run all post-embedding domain checks.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The embedded mesh (domain with void)
    expected_outlets : int
        Expected number of outlet openings
    pitch : float
        Voxel pitch for analysis
    min_coverage_fraction : float
        Minimum acceptable void fraction
    max_coverage_fraction : float
        Maximum acceptable void fraction
        
    Returns
    -------
    DomainCheckReport
        Aggregated report with all check results
    """
    checks = [
        check_outlets_open(mesh, expected_outlets, pitch),
        check_domain_coverage(mesh, None, min_coverage_fraction, max_coverage_fraction),
    ]
    
    all_passed = all(c.passed for c in checks)
    has_warnings = any(len(c.warnings) > 0 for c in checks)
    
    if all_passed and not has_warnings:
        status = "ok"
    elif all_passed:
        status = "warnings"
    else:
        status = "fail"
    
    summary = {
        "total_checks": len(checks),
        "passed_checks": sum(1 for c in checks if c.passed),
        "failed_checks": sum(1 for c in checks if not c.passed),
        "total_warnings": sum(len(c.warnings) for c in checks),
    }
    
    return DomainCheckReport(
        passed=all_passed,
        status=status,
        checks=checks,
        summary=summary,
    )
