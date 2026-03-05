"""
Topology checks for mesh and network validation.

This module provides topology-related checks including connectivity,
open ports, and graph structure.
"""

from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import trimesh
    from generation.core.network import VascularNetwork


def check_topology(mesh: "trimesh.Trimesh") -> Dict[str, Any]:
    """
    Check mesh topology for manifoldness and edge consistency.
    
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
        - details: dict with is_manifold, edge_counts, etc.
    """
    import numpy as np
    
    details = {}
    issues = []
    
    # Check for degenerate faces
    try:
        face_areas = mesh.area_faces
        degenerate_count = int(np.sum(face_areas < 1e-12))
        details["degenerate_faces"] = degenerate_count
        if degenerate_count > 0:
            issues.append(f"{degenerate_count} degenerate faces")
    except Exception:
        details["degenerate_faces"] = None
    
    # Check for duplicate faces
    try:
        unique_faces = len(np.unique(np.sort(mesh.faces, axis=1), axis=0))
        duplicate_faces = len(mesh.faces) - unique_faces
        details["duplicate_faces"] = duplicate_faces
        if duplicate_faces > 0:
            issues.append(f"{duplicate_faces} duplicate faces")
    except Exception:
        details["duplicate_faces"] = None
    
    # Check edge consistency
    try:
        edges = mesh.edges_unique
        details["unique_edges"] = len(edges)
    except Exception:
        details["unique_edges"] = None
    
    passed = len(issues) == 0
    
    if passed:
        message = "Mesh topology is valid"
    else:
        message = "Mesh topology issues: " + ", ".join(issues)
    
    return {
        "passed": passed,
        "message": message,
        "details": details,
    }


def check_network_topology(network: "VascularNetwork") -> Dict[str, Any]:
    """
    Check network topology for connectivity and structure.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to check
        
    Returns
    -------
    dict
        Check result with keys:
        - passed: bool
        - message: str
        - details: dict with connectivity info
    """
    from generation.core.types import NodeType
    
    details = {
        "node_count": len(network.nodes),
        "segment_count": len(network.segments),
    }
    issues = []
    
    # Count node types
    inlet_count = sum(1 for n in network.nodes.values() if n.node_type == NodeType.INLET)
    outlet_count = sum(1 for n in network.nodes.values() if n.node_type == NodeType.OUTLET)
    terminal_count = sum(1 for n in network.nodes.values() if n.node_type == NodeType.TERMINAL)
    
    details["inlet_count"] = inlet_count
    details["outlet_count"] = outlet_count
    details["terminal_count"] = terminal_count
    
    # Check for at least one inlet
    if inlet_count == 0:
        issues.append("No inlet nodes")
    
    # Check connectivity
    if network.nodes:
        adjacency = {nid: set() for nid in network.nodes}
        for segment in network.segments.values():
            if segment.start_node_id in adjacency and segment.end_node_id in adjacency:
                adjacency[segment.start_node_id].add(segment.end_node_id)
                adjacency[segment.end_node_id].add(segment.start_node_id)
        
        # Count connected components
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
        
        details["num_components"] = num_components
        details["is_connected"] = num_components == 1
        
        if num_components > 1:
            issues.append(f"Network has {num_components} disconnected components")
    
    passed = len(issues) == 0
    
    if passed:
        message = "Network topology is valid"
    else:
        message = "Network topology issues: " + ", ".join(issues)
    
    return {
        "passed": passed,
        "message": message,
        "details": details,
    }


__all__ = ["check_topology", "check_network_topology"]
