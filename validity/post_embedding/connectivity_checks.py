"""
Post-Embedding Connectivity Checks

Validates fluid connectivity after embedding structure into a domain.
Checks include port accessibility, trapped fluid detection, and channel continuity.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy import ndimage
import trimesh


@dataclass
class ConnectivityCheckResult:
    """Result of a connectivity check."""
    passed: bool
    check_name: str
    message: str
    details: Dict[str, Any]
    warnings: List[str]


def _mesh_to_fluid_mask(
    mesh: trimesh.Trimesh,
    pitch: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Voxelize a mesh into a 3D boolean fluid mask.
    
    Returns
    -------
    fluid_mask : (nx, ny, nz) bool
        Boolean array of voxel occupancy
    bbox_min : (3,) float
        Lower corner of mesh bounding box
    spacing : (3,) float
        Per-axis voxel spacing
    """
    try:
        vox = mesh.voxelized(pitch)
        vox_filled = vox.fill()
        fluid_mask = vox_filled.matrix.astype(bool)
    except Exception as e:
        raise RuntimeError(f"Voxelization failed: {e}")
    
    if not fluid_mask.any():
        raise RuntimeError("Voxelization produced an empty fluid mask.")
    
    bbox_min, bbox_max = mesh.bounds
    bbox_min = np.asarray(bbox_min, dtype=float)
    bbox_max = np.asarray(bbox_max, dtype=float)
    
    nx, ny, nz = fluid_mask.shape
    dims = np.array([nx, ny, nz], dtype=float)
    spacing = (bbox_max - bbox_min) / dims
    
    return fluid_mask, bbox_min, spacing


def check_port_accessibility(
    mesh: trimesh.Trimesh,
    pitch: float = 0.1,
    min_port_components: int = 1,
) -> ConnectivityCheckResult:
    """
    Check if fluid ports are accessible from the exterior.
    
    Ports are fluid components that touch the bounding box faces.
    At least one port is required for fluid to enter/exit the structure.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The embedded mesh (domain with void)
    pitch : float
        Voxel pitch for analysis
    min_port_components : int
        Minimum required number of port components
        
    Returns
    -------
    ConnectivityCheckResult
        Result with pass/fail status and details
    """
    try:
        fluid_mask, bbox_min, spacing = _mesh_to_fluid_mask(mesh, pitch)
    except RuntimeError as e:
        return ConnectivityCheckResult(
            passed=False,
            check_name="port_accessibility",
            message=f"Voxelization failed: {e}",
            details={"error": str(e)},
            warnings=[str(e)],
        )
    
    # Label connected components
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    labels, num_labels = ndimage.label(fluid_mask, structure=structure)
    
    # Find components that touch bounding box faces (ports)
    port_labels_set = set()
    for face_slice in [
        labels[0, :, :],
        labels[-1, :, :],
        labels[:, 0, :],
        labels[:, -1, :],
        labels[:, :, 0],
        labels[:, :, -1],
    ]:
        port_labels_set.update(np.unique(face_slice))
    port_labels_set.discard(0)
    port_labels = sorted(list(port_labels_set))
    
    num_ports = len(port_labels)
    passed = num_ports >= min_port_components
    
    details = {
        "num_fluid_components": int(num_labels),
        "num_port_components": num_ports,
        "port_labels": port_labels,
        "min_required_ports": min_port_components,
        "grid_shape": list(fluid_mask.shape),
        "pitch": pitch,
    }
    
    warnings = []
    if num_ports < min_port_components:
        warnings.append(f"Only {num_ports} port(s) found, need at least {min_port_components}")
    if num_ports == 0:
        warnings.append("No ports found - fluid cannot enter/exit the structure")
    
    return ConnectivityCheckResult(
        passed=passed,
        check_name="port_accessibility",
        message=f"{num_ports} port(s) accessible from exterior",
        details=details,
        warnings=warnings,
    )


def check_trapped_fluid(
    mesh: trimesh.Trimesh,
    pitch: float = 0.1,
    max_trapped_fraction: float = 0.05,
) -> ConnectivityCheckResult:
    """
    Check for trapped fluid components (not connected to any port).
    
    Trapped fluid cannot be perfused and represents wasted volume.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The embedded mesh (domain with void)
    pitch : float
        Voxel pitch for analysis
    max_trapped_fraction : float
        Maximum acceptable fraction of trapped fluid
        
    Returns
    -------
    ConnectivityCheckResult
        Result with pass/fail status and details
    """
    try:
        fluid_mask, bbox_min, spacing = _mesh_to_fluid_mask(mesh, pitch)
    except RuntimeError as e:
        return ConnectivityCheckResult(
            passed=False,
            check_name="trapped_fluid",
            message=f"Voxelization failed: {e}",
            details={"error": str(e)},
            warnings=[str(e)],
        )
    
    num_fluid_voxels = int(fluid_mask.sum())
    
    # Label connected components
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    labels, num_labels = ndimage.label(fluid_mask, structure=structure)
    
    # Get component sizes
    component_sizes = ndimage.sum(fluid_mask, labels, index=range(1, num_labels + 1))
    component_sizes = [int(s) for s in component_sizes]
    
    # Find port components
    port_labels_set = set()
    for face_slice in [
        labels[0, :, :],
        labels[-1, :, :],
        labels[:, 0, :],
        labels[:, -1, :],
        labels[:, :, 0],
        labels[:, :, -1],
    ]:
        port_labels_set.update(np.unique(face_slice))
    port_labels_set.discard(0)
    port_labels = sorted(list(port_labels_set))
    
    # Calculate reachable and trapped volumes
    if port_labels:
        reachable_voxels = sum(component_sizes[l - 1] for l in port_labels)
    else:
        reachable_voxels = 0
    
    trapped_voxels = num_fluid_voxels - reachable_voxels
    trapped_fraction = trapped_voxels / num_fluid_voxels if num_fluid_voxels > 0 else 0.0
    reachable_fraction = reachable_voxels / num_fluid_voxels if num_fluid_voxels > 0 else 0.0
    
    # Find trapped components
    all_labels = set(range(1, num_labels + 1))
    trapped_labels = sorted(list(all_labels - set(port_labels)))
    trapped_sizes = [component_sizes[l - 1] for l in trapped_labels]
    
    passed = trapped_fraction <= max_trapped_fraction
    
    details = {
        "num_fluid_voxels": num_fluid_voxels,
        "num_fluid_components": int(num_labels),
        "reachable_voxels": reachable_voxels,
        "reachable_fraction": float(reachable_fraction),
        "trapped_voxels": trapped_voxels,
        "trapped_fraction": float(trapped_fraction),
        "num_trapped_components": len(trapped_labels),
        "trapped_component_sizes": trapped_sizes,
        "max_trapped_fraction": max_trapped_fraction,
    }
    
    warnings = []
    if trapped_fraction > max_trapped_fraction:
        warnings.append(f"Trapped fluid fraction {trapped_fraction:.1%} exceeds limit {max_trapped_fraction:.1%}")
    if len(trapped_labels) > 0:
        warnings.append(f"{len(trapped_labels)} trapped fluid component(s) detected")
    
    return ConnectivityCheckResult(
        passed=passed,
        check_name="trapped_fluid",
        message=f"Reachable: {reachable_fraction:.1%}, Trapped: {trapped_fraction:.1%}",
        details=details,
        warnings=warnings,
    )


def check_channel_continuity(
    mesh: trimesh.Trimesh,
    pitch: float = 0.1,
    require_single_component: bool = False,
) -> ConnectivityCheckResult:
    """
    Check that fluid channels are continuous (connected).
    
    For proper perfusion, all channels should ideally be connected.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The embedded mesh (domain with void)
    pitch : float
        Voxel pitch for analysis
    require_single_component : bool
        If True, require all fluid to be in a single connected component
        
    Returns
    -------
    ConnectivityCheckResult
        Result with pass/fail status and details
    """
    try:
        fluid_mask, bbox_min, spacing = _mesh_to_fluid_mask(mesh, pitch)
    except RuntimeError as e:
        return ConnectivityCheckResult(
            passed=False,
            check_name="channel_continuity",
            message=f"Voxelization failed: {e}",
            details={"error": str(e)},
            warnings=[str(e)],
        )
    
    # Label connected components
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    labels, num_labels = ndimage.label(fluid_mask, structure=structure)
    
    # Get component sizes
    component_sizes = ndimage.sum(fluid_mask, labels, index=range(1, num_labels + 1))
    component_sizes = [int(s) for s in component_sizes]
    
    # Find largest component
    if component_sizes:
        largest_component_size = max(component_sizes)
        largest_component_fraction = largest_component_size / sum(component_sizes)
    else:
        largest_component_size = 0
        largest_component_fraction = 0.0
    
    if require_single_component:
        passed = num_labels == 1
    else:
        # Pass if largest component contains most of the fluid
        passed = largest_component_fraction >= 0.9
    
    details = {
        "num_fluid_components": int(num_labels),
        "component_sizes": component_sizes,
        "largest_component_size": largest_component_size,
        "largest_component_fraction": float(largest_component_fraction),
        "require_single_component": require_single_component,
    }
    
    warnings = []
    if num_labels > 1:
        warnings.append(f"Fluid is split into {num_labels} disconnected components")
    if largest_component_fraction < 0.9 and num_labels > 1:
        warnings.append(f"Largest component only contains {largest_component_fraction:.1%} of fluid")
    
    return ConnectivityCheckResult(
        passed=passed,
        check_name="channel_continuity",
        message=f"{num_labels} component(s), largest: {largest_component_fraction:.1%}",
        details=details,
        warnings=warnings,
    )


@dataclass
class ConnectivityCheckReport:
    """Aggregated report of all connectivity checks."""
    passed: bool
    status: str  # "ok", "warnings", "fail"
    checks: List[ConnectivityCheckResult]
    summary: Dict[str, Any]


def run_all_connectivity_checks(
    mesh: trimesh.Trimesh,
    pitch: float = 0.1,
    min_port_components: int = 1,
    max_trapped_fraction: float = 0.05,
    require_single_component: bool = False,
) -> ConnectivityCheckReport:
    """
    Run all post-embedding connectivity checks.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The embedded mesh (domain with void)
    pitch : float
        Voxel pitch for analysis
    min_port_components : int
        Minimum required number of port components
    max_trapped_fraction : float
        Maximum acceptable fraction of trapped fluid
    require_single_component : bool
        If True, require all fluid to be in a single connected component
        
    Returns
    -------
    ConnectivityCheckReport
        Aggregated report with all check results
    """
    checks = [
        check_port_accessibility(mesh, pitch, min_port_components),
        check_trapped_fluid(mesh, pitch, max_trapped_fraction),
        check_channel_continuity(mesh, pitch, require_single_component),
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
    
    return ConnectivityCheckReport(
        passed=all_passed,
        status=status,
        checks=checks,
        summary=summary,
    )
