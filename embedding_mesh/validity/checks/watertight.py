"""
Watertight check for mesh validation.

This module provides the watertight check function for mesh validation.
"""

from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import trimesh


def check_watertight(mesh: "trimesh.Trimesh") -> Dict[str, Any]:
    """
    Check if a mesh is watertight.
    
    A watertight mesh has no holes and forms a closed surface.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to check
        
    Returns
    -------
    dict
        Check result with keys:
        - passed: bool
        - message: str
        - details: dict with is_watertight, euler_number, etc.
    """
    is_watertight = mesh.is_watertight
    
    details = {
        "is_watertight": is_watertight,
    }
    
    # Additional diagnostics
    try:
        details["euler_number"] = mesh.euler_number
    except Exception:
        details["euler_number"] = None
    
    try:
        details["is_volume"] = mesh.is_volume
    except Exception:
        details["is_volume"] = None
    
    return {
        "passed": is_watertight,
        "message": "Mesh is watertight" if is_watertight else "Mesh is not watertight",
        "details": details,
    }


__all__ = ["check_watertight"]
