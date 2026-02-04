"""
Centralized radius accessors for vascular network analysis.

This module provides a single authoritative source for segment radii,
always preferring geometry-based values (seg.geometry.radius_start/end)
over attribute-based values (seg.attributes["radius"]).

Note: The library uses METERS internally for all geometry.
"""

from typing import Optional, Dict, List, Tuple, Set
import numpy as np

from ..core.network import VascularNetwork, VesselSegment, Node


def segment_radius_at_node(segment: VesselSegment, node_id: int) -> float:
    """
    Get the radius of a segment at a specific node.
    
    Always uses geometry-based radius (radius_start/end) as the
    authoritative source.
    
    Parameters
    ----------
    segment : VesselSegment
        The vessel segment
    node_id : int
        ID of the node (must be start_node_id or end_node_id)
        
    Returns
    -------
    float
        Radius at the specified node (meters)
        
    Raises
    ------
    ValueError
        If node_id is not connected to the segment
    """
    if node_id == segment.start_node_id:
        return segment.geometry.radius_start
    elif node_id == segment.end_node_id:
        return segment.geometry.radius_end
    else:
        raise ValueError(
            f"Node {node_id} is not connected to segment {segment.id}"
        )


def segment_mean_radius(segment: VesselSegment) -> float:
    """
    Get the mean radius of a segment.
    
    Always uses geometry-based radius as the authoritative source.
    
    Parameters
    ----------
    segment : VesselSegment
        The vessel segment
        
    Returns
    -------
    float
        Mean radius of the segment (meters)
    """
    return segment.geometry.mean_radius()


def get_junction_radii(
    network: VascularNetwork,
    node_id: int,
) -> Dict[int, float]:
    """
    Get radii of all segments connected to a junction node.
    
    Returns the radius at the junction point for each connected segment.
    
    Parameters
    ----------
    network : VascularNetwork
        The vascular network
    node_id : int
        ID of the junction node
        
    Returns
    -------
    dict
        Mapping from segment_id to radius at the junction
    """
    radii = {}
    connected_seg_ids = network.get_connected_segment_ids(node_id)
    
    for seg_id in connected_seg_ids:
        segment = network.segments.get(seg_id)
        if segment is not None:
            radii[seg_id] = segment_radius_at_node(segment, node_id)
    
    return radii


def identify_parent_segment_at_junction(
    network: VascularNetwork,
    node_id: int,
    root_node_id: Optional[int] = None,
    use_flow_direction: bool = True,
) -> Tuple[Optional[int], List[int]]:
    """
    Identify the parent segment at a junction using topology or flow direction.
    
    This function determines parent/child relationships at a junction node
    using one of these methods (in order of preference):
    1. Flow direction (if pressures are available and use_flow_direction=True)
    2. Branch order from root (if root_node_id is provided)
    3. Radius-based heuristic (fallback - largest radius is parent)
    
    Parameters
    ----------
    network : VascularNetwork
        The vascular network
    node_id : int
        ID of the junction node
    root_node_id : int, optional
        ID of the root (inlet) node for topology-based detection
    use_flow_direction : bool
        If True, try to use pressure/flow data for parent detection
        
    Returns
    -------
    parent_seg_id : int or None
        ID of the parent segment (None if node is root or detection failed)
    child_seg_ids : list of int
        IDs of child segments
    """
    connected_seg_ids = list(network.get_connected_segment_ids(node_id))
    
    if len(connected_seg_ids) == 0:
        return None, []
    
    if len(connected_seg_ids) == 1:
        return None, connected_seg_ids
    
    # Method 1: Flow direction (pressure-based)
    if use_flow_direction:
        parent_id = _identify_parent_by_flow(network, node_id, connected_seg_ids)
        if parent_id is not None:
            child_ids = [sid for sid in connected_seg_ids if sid != parent_id]
            return parent_id, child_ids
    
    # Method 2: Branch order from root (topology-based)
    if root_node_id is not None:
        parent_id = _identify_parent_by_topology(
            network, node_id, connected_seg_ids, root_node_id
        )
        if parent_id is not None:
            child_ids = [sid for sid in connected_seg_ids if sid != parent_id]
            return parent_id, child_ids
    
    # Method 3: Radius-based heuristic (fallback)
    parent_id = _identify_parent_by_radius(network, node_id, connected_seg_ids)
    child_ids = [sid for sid in connected_seg_ids if sid != parent_id]
    return parent_id, child_ids


def _identify_parent_by_flow(
    network: VascularNetwork,
    node_id: int,
    connected_seg_ids: List[int],
) -> Optional[int]:
    """
    Identify parent segment using flow/pressure direction.
    
    The parent is the segment with flow entering the junction.
    For arterial trees: higher pressure upstream (parent has higher pressure)
    For venous trees: lower pressure upstream (parent has lower pressure)
    """
    inflow_segments = []
    outflow_segments = []
    
    for seg_id in connected_seg_ids:
        segment = network.segments.get(seg_id)
        if segment is None:
            continue
        
        # Check if pressure data is available
        p_start = segment.attributes.get("pressure_start")
        p_end = segment.attributes.get("pressure_end")
        
        if p_start is None or p_end is None:
            continue
        
        # Determine flow direction based on pressure gradient
        if segment.start_node_id == node_id:
            # Segment starts at this node
            # If p_start > p_end, flow is outward (this is a child)
            # If p_start < p_end, flow is inward (this is parent)
            if p_start < p_end:
                inflow_segments.append(seg_id)
            else:
                outflow_segments.append(seg_id)
        else:
            # Segment ends at this node
            # If p_start > p_end, flow is inward (this is parent)
            # If p_start < p_end, flow is outward (this is a child)
            if p_start > p_end:
                inflow_segments.append(seg_id)
            else:
                outflow_segments.append(seg_id)
    
    # Parent is the inflow segment (should be exactly one for a tree)
    if len(inflow_segments) == 1:
        return inflow_segments[0]
    
    return None


def _identify_parent_by_topology(
    network: VascularNetwork,
    node_id: int,
    connected_seg_ids: List[int],
    root_node_id: int,
) -> Optional[int]:
    """
    Identify parent segment using graph traversal from root.
    
    The parent is the segment on the path from root to this node.
    """
    # BFS from root to find path
    visited: Set[int] = {root_node_id}
    parent_map: Dict[int, Optional[int]] = {root_node_id: None}  # node_id -> parent_seg_id
    queue = [root_node_id]
    
    while queue:
        current_node = queue.pop(0)
        
        if current_node == node_id:
            # Found the target node, return its parent segment
            return parent_map.get(node_id)
        
        for seg_id in network.get_connected_segment_ids(current_node):
            segment = network.segments.get(seg_id)
            if segment is None:
                continue
            
            other_node = (
                segment.end_node_id
                if segment.start_node_id == current_node
                else segment.start_node_id
            )
            
            if other_node not in visited:
                visited.add(other_node)
                parent_map[other_node] = seg_id
                queue.append(other_node)
    
    return None


def _identify_parent_by_radius(
    network: VascularNetwork,
    node_id: int,
    connected_seg_ids: List[int],
) -> Optional[int]:
    """
    Identify parent segment using radius heuristic (fallback).
    
    The parent is assumed to be the segment with the largest radius
    at the junction. This is a heuristic that works for well-formed
    trees following Murray's law, but may fail for loops or convergence.
    """
    max_radius = -1.0
    parent_id = None
    
    for seg_id in connected_seg_ids:
        segment = network.segments.get(seg_id)
        if segment is None:
            continue
        
        radius = segment_radius_at_node(segment, node_id)
        if radius > max_radius:
            max_radius = radius
            parent_id = seg_id
    
    return parent_id


def compute_murray_deviation_at_junction(
    network: VascularNetwork,
    node_id: int,
    gamma: float = 3.0,
    root_node_id: Optional[int] = None,
) -> Optional[float]:
    """
    Compute Murray's law deviation at a junction.
    
    Murray's law states: r_parent^gamma = sum(r_child^gamma)
    
    This function uses topology-based parent detection when possible,
    falling back to radius-based heuristic only when necessary.
    
    Parameters
    ----------
    network : VascularNetwork
        The vascular network
    node_id : int
        ID of the junction node
    gamma : float
        Murray's law exponent (default 3.0)
    root_node_id : int, optional
        ID of the root node for topology-based parent detection
        
    Returns
    -------
    deviation : float or None
        Relative deviation from Murray's law: |r_p^gamma - sum(r_c^gamma)| / r_p^gamma
        Returns None if junction is invalid (< 2 segments, or in a cycle)
    """
    node = network.nodes.get(node_id)
    if node is None or node.node_type != "junction":
        return None
    
    parent_seg_id, child_seg_ids = identify_parent_segment_at_junction(
        network, node_id, root_node_id=root_node_id
    )
    
    if parent_seg_id is None or len(child_seg_ids) == 0:
        return None
    
    parent_segment = network.segments.get(parent_seg_id)
    if parent_segment is None:
        return None
    
    parent_radius = segment_radius_at_node(parent_segment, node_id)
    
    child_radii = []
    for child_id in child_seg_ids:
        child_segment = network.segments.get(child_id)
        if child_segment is not None:
            child_radii.append(segment_radius_at_node(child_segment, node_id))
    
    if len(child_radii) == 0:
        return None
    
    # Murray's law: r_parent^gamma = sum(r_child^gamma)
    parent_term = parent_radius ** gamma
    child_sum = sum(r ** gamma for r in child_radii)
    
    if parent_term <= 0:
        return None
    
    deviation = abs(parent_term - child_sum) / parent_term
    return deviation


def is_junction_in_cycle(
    network: VascularNetwork,
    node_id: int,
) -> bool:
    """
    Check if a junction node is part of a cycle (loop/anastomosis).
    
    Junctions in cycles should be excluded from Murray's law scoring
    because parent/child relationships are ambiguous.
    
    Parameters
    ----------
    network : VascularNetwork
        The vascular network
    node_id : int
        ID of the junction node
        
    Returns
    -------
    bool
        True if the node is part of a cycle
    """
    # Simple cycle detection: if removing this node doesn't disconnect
    # any of its neighbors from each other, it's in a cycle
    connected_seg_ids = list(network.get_connected_segment_ids(node_id))
    
    if len(connected_seg_ids) < 2:
        return False
    
    # Get all neighbor nodes
    neighbor_nodes = set()
    for seg_id in connected_seg_ids:
        segment = network.segments.get(seg_id)
        if segment is not None:
            other_node = (
                segment.end_node_id
                if segment.start_node_id == node_id
                else segment.start_node_id
            )
            neighbor_nodes.add(other_node)
    
    if len(neighbor_nodes) < 2:
        return False
    
    # Check if any two neighbors can reach each other without going through node_id
    neighbor_list = list(neighbor_nodes)
    start_neighbor = neighbor_list[0]
    
    # BFS from start_neighbor, avoiding node_id
    visited: Set[int] = {start_neighbor, node_id}
    queue = [start_neighbor]
    
    while queue:
        current = queue.pop(0)
        
        for seg_id in network.get_connected_segment_ids(current):
            segment = network.segments.get(seg_id)
            if segment is None:
                continue
            
            other = (
                segment.end_node_id
                if segment.start_node_id == current
                else segment.start_node_id
            )
            
            if other == node_id:
                continue
            
            if other in neighbor_nodes and other != start_neighbor:
                # Found another neighbor without going through node_id
                return True
            
            if other not in visited:
                visited.add(other)
                queue.append(other)
    
    return False


def find_root_node_id(
    network: VascularNetwork,
    vessel_type: Optional[str] = None,
) -> Optional[int]:
    """
    Find the root (inlet) node ID for a tree in the network.
    
    Parameters
    ----------
    network : VascularNetwork
        The vascular network
    vessel_type : str, optional
        Filter by vessel type ("arterial", "venous")
        
    Returns
    -------
    int or None
        ID of the root node, or None if not found
    """
    for node in network.nodes.values():
        if node.node_type == "inlet":
            if vessel_type is None or node.vessel_type == vessel_type:
                return node.id
    
    # Fallback: look for outlet nodes (for venous trees)
    for node in network.nodes.values():
        if node.node_type == "outlet":
            if vessel_type is None or node.vessel_type == vessel_type:
                return node.id
    
    return None
