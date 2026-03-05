"""
Pre-Embedding Mesh Checks

Validates mesh quality before embedding into a domain.
Checks include watertightness, manifoldness, surface quality, and degenerate faces.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
import trimesh


@dataclass
class MeshCheckResult:
    """Result of a mesh check."""
    passed: bool
    check_name: str
    message: str
    details: Dict[str, Any]
    warnings: List[str]


def check_watertightness(mesh: trimesh.Trimesh) -> MeshCheckResult:
    """
    Check if mesh is watertight (closed, no holes).
    
    A watertight mesh is required for proper voxelization and embedding.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to check
        
    Returns
    -------
    MeshCheckResult
        Result with pass/fail status and details
    """
    is_watertight = mesh.is_watertight
    
    details = {
        "is_watertight": is_watertight,
        "euler_number": int(mesh.euler_number),
        "num_vertices": int(mesh.vertices.shape[0]),
        "num_faces": int(mesh.faces.shape[0]),
    }
    
    warnings = []
    if not is_watertight:
        warnings.append("Mesh is not watertight - may cause issues during voxelization")
        if mesh.euler_number != 2:
            warnings.append(f"Euler number is {mesh.euler_number}, expected 2 for closed surface")
    
    return MeshCheckResult(
        passed=is_watertight,
        check_name="watertightness",
        message="Mesh is watertight" if is_watertight else "Mesh is NOT watertight",
        details=details,
        warnings=warnings,
    )


def check_manifoldness(mesh: trimesh.Trimesh) -> MeshCheckResult:
    """
    Check if mesh is manifold (each edge shared by exactly 2 faces).
    
    Non-manifold edges can cause issues with mesh processing and boolean operations.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to check
        
    Returns
    -------
    MeshCheckResult
        Result with pass/fail status and details
    """
    try:
        if mesh.edges_unique is not None and mesh.edges_unique_inverse is not None:
            counts = np.bincount(mesh.edges_unique_inverse)
            non_manifold_edges = int(np.sum(counts > 2))
        else:
            non_manifold_edges = 0
    except AttributeError:
        non_manifold_edges = 0
    
    is_manifold = non_manifold_edges == 0
    
    details = {
        "is_manifold": is_manifold,
        "non_manifold_edges": non_manifold_edges,
        "total_edges": len(mesh.edges_unique) if mesh.edges_unique is not None else 0,
    }
    
    warnings = []
    if not is_manifold:
        warnings.append(f"Found {non_manifold_edges} non-manifold edges")
    
    return MeshCheckResult(
        passed=is_manifold,
        check_name="manifoldness",
        message="Mesh is manifold" if is_manifold else f"Mesh has {non_manifold_edges} non-manifold edges",
        details=details,
        warnings=warnings,
    )


def check_surface_quality(
    mesh: trimesh.Trimesh,
    max_aspect_ratio: float = 20.0,
    max_aspect_ratio_fraction: float = 0.1,
) -> MeshCheckResult:
    """
    Check surface quality metrics (face areas, edge lengths, aspect ratios).
    
    Poor surface quality can cause numerical issues in CFD simulations.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to check
    max_aspect_ratio : float
        Maximum acceptable aspect ratio for any face
    max_aspect_ratio_fraction : float
        Maximum fraction of faces allowed to exceed aspect ratio of 10
        
    Returns
    -------
    MeshCheckResult
        Result with pass/fail status and details
    """
    v = mesh.vertices
    f = mesh.faces
    areas = mesh.area_faces
    
    # Edge lengths
    edges = mesh.edges_unique
    lengths = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
    
    # Aspect ratios
    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]
    
    e01 = np.linalg.norm(v1 - v0, axis=1)
    e12 = np.linalg.norm(v2 - v1, axis=1)
    e20 = np.linalg.norm(v0 - v2, axis=1)
    
    longest_edge = np.maximum(e01, np.maximum(e12, e20))
    h = 2.0 * areas / (longest_edge + 1e-16)
    aspect_ratio = longest_edge / (h + 1e-16)
    
    max_ar = float(aspect_ratio.max())
    frac_over_10 = float(np.mean(aspect_ratio > 10.0))
    
    passed = max_ar <= max_aspect_ratio and frac_over_10 <= max_aspect_ratio_fraction
    
    details = {
        "min_face_area": float(areas.min()),
        "max_face_area": float(areas.max()),
        "mean_face_area": float(areas.mean()),
        "min_edge_length": float(lengths.min()),
        "max_edge_length": float(lengths.max()),
        "mean_edge_length": float(lengths.mean()),
        "max_aspect_ratio": max_ar,
        "frac_aspect_ratio_over_10": frac_over_10,
    }
    
    warnings = []
    if max_ar > max_aspect_ratio:
        warnings.append(f"Max aspect ratio {max_ar:.1f} exceeds threshold {max_aspect_ratio}")
    if frac_over_10 > max_aspect_ratio_fraction:
        warnings.append(f"{frac_over_10:.1%} of faces have aspect ratio > 10")
    
    return MeshCheckResult(
        passed=passed,
        check_name="surface_quality",
        message="Surface quality is acceptable" if passed else "Surface quality issues detected",
        details=details,
        warnings=warnings,
    )


def check_degenerate_faces(
    mesh: trimesh.Trimesh,
    area_eps: float = 1e-18,
    max_degenerate_fraction: float = 0.001,
) -> MeshCheckResult:
    """
    Check for degenerate (zero-area) faces.
    
    Degenerate faces can cause numerical issues and should be removed.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to check
    area_eps : float
        Threshold below which a face is considered degenerate
    max_degenerate_fraction : float
        Maximum acceptable fraction of degenerate faces
        
    Returns
    -------
    MeshCheckResult
        Result with pass/fail status and details
    """
    areas = mesh.area_faces
    degenerate_count = int(np.sum(areas <= area_eps))
    total_faces = len(areas)
    degenerate_fraction = degenerate_count / total_faces if total_faces > 0 else 0.0
    
    passed = degenerate_fraction <= max_degenerate_fraction
    
    details = {
        "degenerate_faces": degenerate_count,
        "total_faces": total_faces,
        "degenerate_fraction": degenerate_fraction,
        "area_threshold": area_eps,
    }
    
    warnings = []
    if degenerate_count > 0:
        warnings.append(f"Found {degenerate_count} degenerate faces ({degenerate_fraction:.4%})")
    
    return MeshCheckResult(
        passed=passed,
        check_name="degenerate_faces",
        message=f"No degenerate faces" if degenerate_count == 0 else f"{degenerate_count} degenerate faces found",
        details=details,
        warnings=warnings,
    )


@dataclass
class MeshCheckReport:
    """Aggregated report of all mesh checks."""
    passed: bool
    status: str  # "ok", "warnings", "fail"
    checks: List[MeshCheckResult]
    summary: Dict[str, Any]


def run_all_mesh_checks(
    mesh: trimesh.Trimesh,
    max_aspect_ratio: float = 20.0,
    max_aspect_ratio_fraction: float = 0.1,
    area_eps: float = 1e-18,
    max_degenerate_fraction: float = 0.001,
) -> MeshCheckReport:
    """
    Run all pre-embedding mesh checks.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to check
    max_aspect_ratio : float
        Maximum acceptable aspect ratio for surface quality check
    max_aspect_ratio_fraction : float
        Maximum fraction of faces with aspect ratio > 10
    area_eps : float
        Threshold for degenerate face detection
    max_degenerate_fraction : float
        Maximum acceptable fraction of degenerate faces
        
    Returns
    -------
    MeshCheckReport
        Aggregated report with all check results
    """
    checks = [
        check_watertightness(mesh),
        check_manifoldness(mesh),
        check_surface_quality(mesh, max_aspect_ratio, max_aspect_ratio_fraction),
        check_degenerate_faces(mesh, area_eps, max_degenerate_fraction),
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
    
    return MeshCheckReport(
        passed=all_passed,
        status=status,
        checks=checks,
        summary=summary,
    )
