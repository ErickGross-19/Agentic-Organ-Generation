"""
Component check for mesh validation.

This module provides the component count check for mesh validation.
"""

from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import trimesh


def check_components(
    mesh: "trimesh.Trimesh",
    max_components: int = 1,
) -> Dict[str, Any]:
    """
    Check the number of connected components in a mesh.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to check
    max_components : int
        Maximum allowed number of components (default: 1)
        
    Returns
    -------
    dict
        Check result with keys:
        - passed: bool
        - message: str
        - details: dict with component_count, component_sizes, etc.
    """
    parts = mesh.split(only_watertight=False)
    component_count = len(parts)
    
    passed = component_count <= max_components
    
    # Get component sizes
    component_sizes = sorted([len(p.faces) for p in parts], reverse=True)
    
    details = {
        "component_count": component_count,
        "max_components": max_components,
        "component_sizes": component_sizes[:10],  # Top 10 only
    }
    
    message = f"Mesh has {component_count} component(s)"
    if not passed:
        message += f" (max: {max_components})"
    
    return {
        "passed": passed,
        "message": message,
        "details": details,
    }


__all__ = ["check_components"]
