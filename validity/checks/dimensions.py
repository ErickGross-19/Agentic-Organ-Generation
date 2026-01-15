"""
Dimension checks for mesh validation.

This module provides dimension-related checks including minimum diameter
and thickness proxies.
"""

from typing import Dict, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    import trimesh


def check_dimensions(
    mesh: "trimesh.Trimesh",
    min_diameter: float = 0.0005,
    max_extent: float = 1.0,
) -> Dict[str, Any]:
    """
    Check mesh dimensions for manufacturability.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to check
    min_diameter : float
        Minimum feature diameter in meters (default: 0.5mm)
    max_extent : float
        Maximum allowed extent in meters (default: 1m)
        
    Returns
    -------
    dict
        Check result with keys:
        - passed: bool
        - message: str
        - details: dict with extents, min_extent, etc.
    """
    extents = mesh.extents
    min_extent_actual = float(np.min(extents))
    max_extent_actual = float(np.max(extents))
    
    issues = []
    
    # Check minimum extent
    if min_extent_actual < min_diameter:
        issues.append(
            f"Minimum extent {min_extent_actual*1000:.3f}mm "
            f"below threshold {min_diameter*1000:.3f}mm"
        )
    
    # Check maximum extent (sanity check for units)
    if max_extent_actual > max_extent:
        issues.append(
            f"Maximum extent {max_extent_actual*1000:.3f}mm "
            f"exceeds {max_extent*1000:.3f}mm (may be wrong units)"
        )
    
    passed = len(issues) == 0
    
    details = {
        "extents": list(extents),
        "min_extent": min_extent_actual,
        "max_extent": max_extent_actual,
        "min_diameter_threshold": min_diameter,
        "max_extent_threshold": max_extent,
    }
    
    if passed:
        message = f"Dimensions valid: {min_extent_actual*1000:.3f}mm - {max_extent_actual*1000:.3f}mm"
    else:
        message = "Dimension issues: " + ", ".join(issues)
    
    return {
        "passed": passed,
        "message": message,
        "details": details,
    }


def check_min_channel_diameter(
    mesh: "trimesh.Trimesh",
    threshold: float = 0.0005,
) -> Dict[str, Any]:
    """
    Estimate minimum channel diameter from mesh geometry.
    
    This is a proxy check that estimates minimum feature size
    from the mesh bounding box and volume.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to check
    threshold : float
        Minimum diameter threshold in meters (default: 0.5mm)
        
    Returns
    -------
    dict
        Check result with keys:
        - passed: bool
        - message: str
        - details: dict with estimated_min_diameter, etc.
    """
    # Estimate minimum diameter from volume and surface area
    try:
        volume = abs(mesh.volume)
        surface_area = mesh.area
        
        if surface_area > 0:
            # Hydraulic diameter approximation
            estimated_diameter = 4 * volume / surface_area
        else:
            estimated_diameter = float(np.min(mesh.extents))
    except Exception:
        estimated_diameter = float(np.min(mesh.extents))
    
    passed = estimated_diameter >= threshold
    
    details = {
        "estimated_min_diameter": estimated_diameter,
        "threshold": threshold,
    }
    
    if passed:
        message = f"Estimated min diameter: {estimated_diameter*1000:.3f}mm"
    else:
        message = (
            f"Estimated min diameter {estimated_diameter*1000:.3f}mm "
            f"below threshold {threshold*1000:.3f}mm"
        )
    
    return {
        "passed": passed,
        "message": message,
        "details": details,
    }


__all__ = ["check_dimensions", "check_min_channel_diameter"]
