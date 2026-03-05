"""
Query and topology analysis functions.
"""

from typing import List, Dict, Optional
import numpy as np
from ..core.network import VascularNetwork


def get_leaf_nodes(
    network: VascularNetwork,
    vessel_type: Optional[str] = None,
) -> List[int]:
    """
    Get all leaf (terminal) nodes in the network.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to query
    vessel_type : str, optional
        Filter by vessel type ("arterial", "venous")
    
    Returns
    -------
    node_ids : List[int]
        List of terminal node IDs
    """
    leaf_nodes = []
    
    for node_id, node in network.nodes.items():
        if node.node_type == "terminal":
            if vessel_type is None or node.vessel_type == vessel_type:
                leaf_nodes.append(node_id)
    
    return leaf_nodes


def get_paths_from_inlet(
    network: VascularNetwork,
    inlet_id: int,
) -> List[List[int]]:
    """
    Get all paths from an inlet to leaf nodes.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to query
    inlet_id : int
        Inlet node ID
    
    Returns
    -------
    paths : List[List[int]]
        List of paths, where each path is a list of node IDs
    """
    inlet_node = network.get_node(inlet_id)
    if inlet_node is None or inlet_node.node_type != "inlet":
        return []
    
    children = {}
    for segment in network.segments.values():
        if segment.start_node_id not in children:
            children[segment.start_node_id] = []
        children[segment.start_node_id].append(segment.end_node_id)
    
    paths = []
    
    def dfs(node_id: int, path: List[int]):
        path.append(node_id)
        
        if node_id not in children or not children[node_id]:
            paths.append(path.copy())
        else:
            for child_id in children[node_id]:
                dfs(child_id, path)
        
        path.pop()
    
    dfs(inlet_id, [])
    
    return paths


def get_branch_order(
    network: VascularNetwork,
    node_id: int,
) -> int:
    """
    Get branch order of a node.
    
    Branch order is the number of bifurcations from the inlet.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to query
    node_id : int
        Node ID
    
    Returns
    -------
    order : int
        Branch order (0 for inlet, increases with each bifurcation)
    """
    node = network.get_node(node_id)
    if node is None:
        return -1
    
    return node.attributes.get("branch_order", 0)


def measure_segment_lengths(
    network: VascularNetwork,
    vessel_type: Optional[str] = None,
) -> Dict[str, float]:
    """
    Measure segment length statistics.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to query
    vessel_type : str, optional
        Filter by vessel type
    
    Returns
    -------
    stats : dict
        Dictionary with keys: mean, std, min, max, total, count
    """
    lengths = []
    
    for segment in network.segments.values():
        if vessel_type is None or segment.vessel_type == vessel_type:
            lengths.append(segment.geometry.length())
    
    if not lengths:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "total": 0.0,
            "count": 0,
        }
    
    lengths_arr = np.array(lengths)
    
    return {
        "mean": float(np.mean(lengths_arr)),
        "std": float(np.std(lengths_arr)),
        "min": float(np.min(lengths_arr)),
        "max": float(np.max(lengths_arr)),
        "total": float(np.sum(lengths_arr)),
        "count": len(lengths),
    }
