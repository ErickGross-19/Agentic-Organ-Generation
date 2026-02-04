"""
Tree manipulation operations for vascular networks.

Provides prune and add_branch operations for iterative refinement of
vascular network structures by LLM agents.
"""

from typing import Optional, List, Set, Tuple
import numpy as np
from ..core.types import Point3D, Direction3D, TubeGeometry
from ..core.network import VascularNetwork, Node, VesselSegment
from ..core.result import OperationResult, OperationStatus, Delta
from .growth import grow_branch, grow_to_point


def _build_adjacency(network: VascularNetwork) -> dict:
    """
    Build adjacency map from segments.
    
    Returns dict mapping node_id -> list of (neighbor_node_id, segment_id).
    """
    adjacency = {}
    for node_id in network.nodes:
        adjacency[node_id] = []
    
    for seg_id, seg in network.segments.items():
        adjacency[seg.start_node_id].append((seg.end_node_id, seg_id))
        adjacency[seg.end_node_id].append((seg.start_node_id, seg_id))
    
    return adjacency


def _compute_distances_from_root(
    network: VascularNetwork,
    root_node_id: int,
) -> dict:
    """
    Compute BFS distances from root to all reachable nodes.
    
    Returns dict mapping node_id -> distance (hop count from root).
    """
    adjacency = _build_adjacency(network)
    distances = {root_node_id: 0}
    queue = [root_node_id]
    
    while queue:
        current = queue.pop(0)
        current_dist = distances[current]
        
        for neighbor_id, _ in adjacency.get(current, []):
            if neighbor_id not in distances:
                distances[neighbor_id] = current_dist + 1
                queue.append(neighbor_id)
    
    return distances


def _find_downstream_nodes(
    network: VascularNetwork,
    start_node_id: int,
    root_node_id: int,
) -> Tuple[Set[int], Set[int]]:
    """
    Find all nodes downstream from start_node_id relative to root_node_id.
    
    Uses BFS from start_node_id, only traversing to nodes with strictly
    greater distance from root (i.e., truly downstream nodes).
    
    Parameters
    ----------
    network : VascularNetwork
        The network to traverse
    start_node_id : int
        Node to start traversal from
    root_node_id : int
        Root node that defines "upstream" direction
        
    Returns
    -------
    downstream_nodes : Set[int]
        Set of node IDs downstream from start_node_id
    downstream_segments : Set[int]
        Set of segment IDs connecting downstream nodes
    """
    adjacency = _build_adjacency(network)
    
    # Compute distances from root to determine upstream/downstream
    distances = _compute_distances_from_root(network, root_node_id)
    
    # If start_node is not reachable from root, return empty
    if start_node_id not in distances:
        return set(), set()
    
    start_distance = distances[start_node_id]
    
    downstream_nodes = set()
    downstream_segments = set()
    visited = set()
    queue = [start_node_id]
    
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        downstream_nodes.add(current)
        
        current_dist = distances.get(current, float('inf'))
        
        for neighbor_id, seg_id in adjacency.get(current, []):
            if neighbor_id in visited:
                continue
            
            neighbor_dist = distances.get(neighbor_id, float('inf'))
            
            # Only traverse to nodes that are strictly downstream
            # (greater distance from root) or at same level but not toward root
            if neighbor_dist > current_dist:
                downstream_segments.add(seg_id)
                queue.append(neighbor_id)
            elif neighbor_dist == current_dist and neighbor_dist > start_distance:
                # Same level but both are downstream of start - include
                downstream_segments.add(seg_id)
                queue.append(neighbor_id)
    
    return downstream_nodes, downstream_segments


def prune(
    network: VascularNetwork,
    node_id: Optional[int] = None,
    segment_id: Optional[int] = None,
    root_node_id: Optional[int] = None,
    prune_downstream: bool = True,
) -> OperationResult:
    """
    Prune a node/segment and optionally its downstream subtree from the network.
    
    This operation removes a specified node or segment and all nodes/segments
    that are downstream from it (relative to a root node). This is useful for
    iterative refinement where an LLM agent wants to remove a problematic
    branch and regrow it.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to modify
    node_id : int, optional
        Node to prune (and its downstream subtree)
    segment_id : int, optional
        Segment to prune (removes segment and downstream nodes)
    root_node_id : int, optional
        Root node defining "upstream" direction. If not provided,
        attempts to find an inlet node.
    prune_downstream : bool
        If True, removes entire downstream subtree. If False, only
        removes the specified node/segment.
        
    Returns
    -------
    result : OperationResult
        Result with delta containing deleted node/segment IDs
        
    Examples
    --------
    >>> # Prune a terminal branch
    >>> result = prune(network, node_id=42, root_node_id=0)
    >>> if result.is_success():
    ...     print(f"Pruned {len(result.delta.deleted_node_ids)} nodes")
    
    >>> # Prune a segment and its downstream tree
    >>> result = prune(network, segment_id=15, root_node_id=0)
    """
    if node_id is None and segment_id is None:
        return OperationResult.failure(
            message="Must specify either node_id or segment_id to prune",
            errors=["Missing prune target"],
        )
    
    if node_id is not None and segment_id is not None:
        return OperationResult.failure(
            message="Specify only one of node_id or segment_id",
            errors=["Ambiguous prune target"],
        )
    
    # Find root node if not provided
    if root_node_id is None:
        for nid, node in network.nodes.items():
            if node.node_type == "inlet":
                root_node_id = nid
                break
        if root_node_id is None:
            return OperationResult.failure(
                message="No root_node_id provided and no inlet found",
                errors=["Missing root node"],
            )
    
    # Validate root exists
    if root_node_id not in network.nodes:
        return OperationResult.failure(
            message=f"Root node {root_node_id} not found in network",
            errors=["Root node not found"],
        )
    
    deleted_node_ids = []
    deleted_segment_ids = []
    
    if segment_id is not None:
        # Prune by segment
        segment = network.get_segment(segment_id)
        if segment is None:
            return OperationResult.failure(
                message=f"Segment {segment_id} not found",
                errors=["Segment not found"],
            )
        
        # Determine which end is downstream
        # The downstream end is the one farther from root
        adjacency = _build_adjacency(network)
        
        # Simple heuristic: the end that's not connected to root path is downstream
        start_node = network.get_node(segment.start_node_id)
        end_node = network.get_node(segment.end_node_id)
        
        # Find downstream node (the one that's not closer to root)
        if segment.start_node_id == root_node_id:
            downstream_start = segment.end_node_id
        elif segment.end_node_id == root_node_id:
            downstream_start = segment.start_node_id
        else:
            # Neither is root, use branch_order to determine
            start_order = start_node.attributes.get("branch_order", 0)
            end_order = end_node.attributes.get("branch_order", 0)
            if end_order > start_order:
                downstream_start = segment.end_node_id
            else:
                downstream_start = segment.start_node_id
        
        deleted_segment_ids.append(segment_id)
        
        if prune_downstream:
            # Find all downstream nodes/segments
            downstream_nodes, downstream_segments = _find_downstream_nodes(
                network, downstream_start, root_node_id
            )
            deleted_node_ids.extend(downstream_nodes)
            deleted_segment_ids.extend(downstream_segments)
        
    else:
        # Prune by node
        if node_id not in network.nodes:
            return OperationResult.failure(
                message=f"Node {node_id} not found",
                errors=["Node not found"],
            )
        
        if node_id == root_node_id:
            return OperationResult.failure(
                message="Cannot prune the root node",
                errors=["Cannot prune root"],
            )
        
        if prune_downstream:
            # Find all downstream nodes/segments
            downstream_nodes, downstream_segments = _find_downstream_nodes(
                network, node_id, root_node_id
            )
            deleted_node_ids.extend(downstream_nodes)
            deleted_segment_ids.extend(downstream_segments)
        else:
            deleted_node_ids.append(node_id)
            # Find and delete segments connected to this node
            for seg_id, seg in list(network.segments.items()):
                if seg.start_node_id == node_id or seg.end_node_id == node_id:
                    deleted_segment_ids.append(seg_id)
    
    # Remove duplicates
    deleted_node_ids = list(set(deleted_node_ids))
    deleted_segment_ids = list(set(deleted_segment_ids))
    
    # Perform deletions (segments first to maintain consistency)
    for seg_id in deleted_segment_ids:
        if seg_id in network.segments:
            network.remove_segment(seg_id)
    
    for nid in deleted_node_ids:
        if nid in network.nodes:
            network.remove_node(nid)
    
    # Update node types for remaining nodes that lost children
    adjacency = _build_adjacency(network)
    for nid in network.nodes:
        node = network.nodes[nid]
        if node.node_type == "junction":
            # Check if still has multiple connections
            if len(adjacency.get(nid, [])) <= 1:
                node.node_type = "terminal"
    
    delta = Delta(
        deleted_node_ids=deleted_node_ids,
        deleted_segment_ids=deleted_segment_ids,
    )
    
    return OperationResult.success(
        message=f"Pruned {len(deleted_node_ids)} nodes and {len(deleted_segment_ids)} segments",
        delta=delta,
        rng_state=network.id_gen.get_state(),
    )


def add_branch(
    network: VascularNetwork,
    from_node_id: int,
    target_point: Optional[Tuple[float, float, float]] = None,
    direction: Optional[Tuple[float, float, float]] = None,
    length: Optional[float] = None,
    radius: Optional[float] = None,
    num_segments: int = 1,
    taper_factor: float = 0.95,
    check_collisions: bool = True,
    seed: Optional[int] = None,
) -> OperationResult:
    """
    Add a new branch to the network starting from an existing node.
    
    This is a convenience wrapper around grow_branch/grow_to_point that
    supports multi-segment branches with tapering. Useful for LLM agents
    to add new vascular structures during iterative refinement.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to modify
    from_node_id : int
        Node to start the branch from
    target_point : tuple, optional
        Target point (x, y, z) to grow toward. If provided, grows
        directly to this point (single segment).
    direction : tuple, optional
        Growth direction (dx, dy, dz). Used if target_point not provided.
    length : float, optional
        Total length of branch. Required if using direction.
    radius : float, optional
        Starting radius. If None, uses parent node's radius.
    num_segments : int
        Number of segments to create (for multi-segment branches)
    taper_factor : float
        Radius multiplier per segment (default 0.95 = 5% taper)
    check_collisions : bool
        Whether to check for collisions
    seed : int, optional
        Random seed for deterministic behavior
        
    Returns
    -------
    result : OperationResult
        Result with new_ids containing lists of created node/segment IDs
        
    Examples
    --------
    >>> # Add branch to specific point
    >>> result = add_branch(
    ...     network,
    ...     from_node_id=5,
    ...     target_point=(0.03, 0.02, 0.01),
    ...     radius=0.002,
    ... )
    
    >>> # Add multi-segment branch in direction
    >>> result = add_branch(
    ...     network,
    ...     from_node_id=5,
    ...     direction=(1, 0, 0),
    ...     length=0.01,
    ...     num_segments=3,
    ...     taper_factor=0.9,
    ... )
    """
    # Validate inputs
    parent_node = network.get_node(from_node_id)
    if parent_node is None:
        return OperationResult.failure(
            message=f"Node {from_node_id} not found",
            errors=["Parent node not found"],
        )
    
    if target_point is None and direction is None:
        return OperationResult.failure(
            message="Must specify either target_point or direction",
            errors=["Missing growth target"],
        )
    
    if direction is not None and length is None:
        return OperationResult.failure(
            message="Must specify length when using direction",
            errors=["Missing length"],
        )
    
    # Get starting radius
    if radius is None:
        radius = parent_node.attributes.get("radius")
        if radius is None:
            return OperationResult.failure(
                message="No radius specified and parent has no radius",
                errors=["Missing radius"],
            )
    
    created_node_ids = []
    created_segment_ids = []
    warnings = []
    
    if target_point is not None:
        # Single segment to target point
        result = grow_to_point(
            network,
            from_node_id=from_node_id,
            target_point=target_point,
            target_radius=radius,
            check_collisions=check_collisions,
            seed=seed,
        )
        
        if not result.is_success():
            return result
        
        created_node_ids.append(result.new_ids["node"])
        created_segment_ids.append(result.new_ids["segment"])
        warnings.extend(result.warnings)
        
    else:
        # Multi-segment branch in direction
        direction_arr = np.array(direction)
        direction_arr = direction_arr / np.linalg.norm(direction_arr)
        
        segment_length = length / num_segments
        current_node_id = from_node_id
        current_radius = radius
        
        for i in range(num_segments):
            result = grow_branch(
                network,
                from_node_id=current_node_id,
                length=segment_length,
                direction=Direction3D.from_array(direction_arr),
                target_radius=current_radius,
                check_collisions=check_collisions,
                seed=seed,
            )
            
            if not result.is_success():
                # Rollback created nodes/segments
                for seg_id in reversed(created_segment_ids):
                    network.remove_segment(seg_id)
                for node_id in reversed(created_node_ids):
                    network.remove_node(node_id)
                
                return OperationResult.failure(
                    message=f"Failed at segment {i+1}/{num_segments}: {result.message}",
                    errors=result.errors,
                )
            
            created_node_ids.append(result.new_ids["node"])
            created_segment_ids.append(result.new_ids["segment"])
            warnings.extend(result.warnings)
            
            # Update for next iteration
            current_node_id = result.new_ids["node"]
            current_radius *= taper_factor
    
    delta = Delta(
        created_node_ids=created_node_ids,
        created_segment_ids=created_segment_ids,
    )
    
    status = OperationStatus.SUCCESS if not warnings else OperationStatus.PARTIAL_SUCCESS
    
    return OperationResult(
        status=status,
        message=f"Added branch with {len(created_node_ids)} nodes and {len(created_segment_ids)} segments",
        new_ids={
            "nodes": created_node_ids,
            "segments": created_segment_ids,
        },
        warnings=warnings,
        delta=delta,
        rng_state=network.id_gen.get_state(),
    )
