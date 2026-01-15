"""
Network cleanup operations for vascular networks.

This module provides functions for cleaning up vascular networks including
node snapping, duplicate merging, and short segment pruning.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Set, TYPE_CHECKING
import numpy as np
import logging

from ...policies import NetworkCleanupPolicy, OperationReport

if TYPE_CHECKING:
    from ...core.network import VascularNetwork

logger = logging.getLogger(__name__)


def snap_nodes(
    network: "VascularNetwork",
    tolerance: float = 0.0001,
) -> Tuple["VascularNetwork", Dict[str, Any]]:
    """
    Snap nearby nodes together.
    
    Nodes within the tolerance distance are merged into a single node.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to clean
    tolerance : float
        Distance tolerance in meters (default: 0.1mm)
        
    Returns
    -------
    network : VascularNetwork
        Modified network (in-place)
    meta : dict
        Metadata about snapped nodes
    """
    from ...core.types import Point3D
    
    meta = {
        "nodes_snapped": 0,
        "snap_groups": [],
    }
    
    # Build position array for efficient distance computation
    node_ids = list(network.nodes.keys())
    positions = np.array([
        [network.nodes[nid].position.x, 
         network.nodes[nid].position.y, 
         network.nodes[nid].position.z]
        for nid in node_ids
    ])
    
    # Find nodes to merge using union-find
    parent = {nid: nid for nid in node_ids}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Find pairs within tolerance
    for i, nid1 in enumerate(node_ids):
        for j, nid2 in enumerate(node_ids[i+1:], i+1):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < tolerance:
                union(nid1, nid2)
    
    # Group nodes by their root
    groups = {}
    for nid in node_ids:
        root = find(nid)
        if root not in groups:
            groups[root] = []
        groups[root].append(nid)
    
    # Merge groups with multiple nodes
    for root, group in groups.items():
        if len(group) > 1:
            meta["snap_groups"].append(group)
            meta["nodes_snapped"] += len(group) - 1
            
            # Calculate centroid
            group_positions = np.array([
                [network.nodes[nid].position.x,
                 network.nodes[nid].position.y,
                 network.nodes[nid].position.z]
                for nid in group
            ])
            centroid = group_positions.mean(axis=0)
            
            # Keep the root node, update its position
            network.nodes[root].position = Point3D(*centroid)
            
            # Update segments to point to root
            for nid in group:
                if nid != root:
                    for seg in network.segments.values():
                        if seg.start_node_id == nid:
                            seg.start_node_id = root
                        if seg.end_node_id == nid:
                            seg.end_node_id = root
                    
                    # Remove merged node
                    del network.nodes[nid]
    
    return network, meta


def merge_duplicate_nodes(
    network: "VascularNetwork",
    tolerance: float = 0.0001,
) -> Tuple["VascularNetwork", Dict[str, Any]]:
    """
    Merge duplicate nodes at the same position.
    
    This is similar to snap_nodes but specifically targets exact duplicates.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to clean
    tolerance : float
        Distance tolerance in meters (default: 0.1mm)
        
    Returns
    -------
    network : VascularNetwork
        Modified network (in-place)
    meta : dict
        Metadata about merged nodes
    """
    # This is essentially the same as snap_nodes with a tighter tolerance
    return snap_nodes(network, tolerance)


def prune_short_segments(
    network: "VascularNetwork",
    min_length: float = 0.0001,
) -> Tuple["VascularNetwork", Dict[str, Any]]:
    """
    Remove segments shorter than the minimum length.
    
    Short segments are removed and their endpoints merged.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to clean
    min_length : float
        Minimum segment length in meters (default: 0.1mm)
        
    Returns
    -------
    network : VascularNetwork
        Modified network (in-place)
    meta : dict
        Metadata about pruned segments
    """
    from ...core.types import Point3D
    
    meta = {
        "segments_pruned": 0,
        "pruned_segment_ids": [],
    }
    
    segments_to_remove = []
    
    for seg_id, segment in network.segments.items():
        start_node = network.nodes.get(segment.start_node_id)
        end_node = network.nodes.get(segment.end_node_id)
        
        if start_node is None or end_node is None:
            continue
        
        # Calculate segment length
        start_pos = np.array([start_node.position.x, start_node.position.y, start_node.position.z])
        end_pos = np.array([end_node.position.x, end_node.position.y, end_node.position.z])
        length = np.linalg.norm(end_pos - start_pos)
        
        if length < min_length:
            segments_to_remove.append(seg_id)
    
    # Remove short segments and merge their endpoints
    for seg_id in segments_to_remove:
        segment = network.segments[seg_id]
        start_id = segment.start_node_id
        end_id = segment.end_node_id
        
        # Keep the start node, redirect segments from end to start
        for other_seg in network.segments.values():
            if other_seg.start_node_id == end_id:
                other_seg.start_node_id = start_id
            if other_seg.end_node_id == end_id:
                other_seg.end_node_id = start_id
        
        # Remove the segment
        del network.segments[seg_id]
        meta["segments_pruned"] += 1
        meta["pruned_segment_ids"].append(seg_id)
        
        # Remove the end node if it's no longer connected
        end_still_connected = any(
            seg.start_node_id == end_id or seg.end_node_id == end_id
            for seg in network.segments.values()
        )
        if not end_still_connected and end_id in network.nodes:
            del network.nodes[end_id]
    
    return network, meta


def cleanup_network(
    network: "VascularNetwork",
    policy: Optional[NetworkCleanupPolicy] = None,
) -> Tuple["VascularNetwork", OperationReport]:
    """
    Apply all cleanup operations to a network based on policy.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to clean
    policy : NetworkCleanupPolicy, optional
        Policy controlling cleanup operations
        
    Returns
    -------
    network : VascularNetwork
        Cleaned network (in-place)
    report : OperationReport
        Report with cleanup statistics
    """
    if policy is None:
        policy = NetworkCleanupPolicy()
    
    warnings = []
    metadata = {
        "operations_applied": [],
    }
    
    # Apply snap nodes
    if policy.enable_snap:
        network, snap_meta = snap_nodes(network, policy.snap_tol)
        metadata["snap"] = snap_meta
        metadata["operations_applied"].append("snap_nodes")
        
        if snap_meta["nodes_snapped"] > 0:
            logger.info(f"Snapped {snap_meta['nodes_snapped']} nodes")
    
    # Apply merge duplicates
    if policy.enable_merge:
        network, merge_meta = merge_duplicate_nodes(network, policy.merge_tol)
        metadata["merge"] = merge_meta
        metadata["operations_applied"].append("merge_duplicate_nodes")
    
    # Apply prune short segments
    if policy.enable_prune:
        network, prune_meta = prune_short_segments(network, policy.min_segment_length)
        metadata["prune"] = prune_meta
        metadata["operations_applied"].append("prune_short_segments")
        
        if prune_meta["segments_pruned"] > 0:
            logger.info(f"Pruned {prune_meta['segments_pruned']} short segments")
    
    # Final statistics
    metadata["final_node_count"] = len(network.nodes)
    metadata["final_segment_count"] = len(network.segments)
    
    report = OperationReport(
        operation="cleanup_network",
        success=True,
        requested_policy=policy.to_dict(),
        effective_policy=policy.to_dict(),
        warnings=warnings,
        metadata=metadata,
    )
    
    return network, report


__all__ = [
    "snap_nodes",
    "merge_duplicate_nodes",
    "prune_short_segments",
    "cleanup_network",
    "NetworkCleanupPolicy",
]
