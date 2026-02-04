"""
Network merging operations for vascular networks.

This module provides functions for merging multiple vascular networks
and resolving overlaps.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, TYPE_CHECKING
import numpy as np
import logging

from ...policies import OperationReport

if TYPE_CHECKING:
    from ...core.network import VascularNetwork

logger = logging.getLogger(__name__)


@dataclass
class NetworkMergePolicy:
    """
    Policy for network merging operations.
    
    JSON Schema:
    {
        "resolve_overlaps": bool,
        "overlap_tolerance": float (meters),
        "preserve_vessel_types": bool,
        "merge_radii_strategy": "max" | "min" | "average"
    }
    """
    resolve_overlaps: bool = True
    overlap_tolerance: float = 0.0002  # 0.2mm
    preserve_vessel_types: bool = True
    merge_radii_strategy: str = "max"  # "max", "min", "average"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "resolve_overlaps": self.resolve_overlaps,
            "overlap_tolerance": self.overlap_tolerance,
            "preserve_vessel_types": self.preserve_vessel_types,
            "merge_radii_strategy": self.merge_radii_strategy,
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "NetworkMergePolicy":
        return NetworkMergePolicy(**{k: v for k, v in d.items() if k in NetworkMergePolicy.__dataclass_fields__})


def merge_networks(
    networks: List["VascularNetwork"],
    policy: Optional[NetworkMergePolicy] = None,
) -> Tuple["VascularNetwork", OperationReport]:
    """
    Merge multiple vascular networks into one.
    
    Parameters
    ----------
    networks : List[VascularNetwork]
        Networks to merge
    policy : NetworkMergePolicy, optional
        Policy controlling merge behavior
        
    Returns
    -------
    merged : VascularNetwork
        Merged network
    report : OperationReport
        Report with merge statistics
    """
    from ...core.network import VascularNetwork, Node, VesselSegment
    from ...core.types import Point3D, NodeType
    
    if policy is None:
        policy = NetworkMergePolicy()
    
    if not networks:
        raise ValueError("No networks to merge")
    
    if len(networks) == 1:
        report = OperationReport(
            operation="merge_networks",
            success=True,
            requested_policy=policy.to_dict(),
            effective_policy=policy.to_dict(),
            warnings=[],
            metadata={"networks_merged": 1, "overlaps_resolved": 0},
        )
        return networks[0], report
    
    warnings = []
    metadata = {
        "networks_merged": len(networks),
        "overlaps_resolved": 0,
        "nodes_before_merge": sum(len(n.nodes) for n in networks),
        "segments_before_merge": sum(len(n.segments) for n in networks),
    }
    
    # Create merged network using first network's domain
    merged = VascularNetwork(domain=networks[0].domain)
    
    # Track node ID mapping for each network
    node_id_offset = 0
    segment_id_offset = 0
    
    for net_idx, network in enumerate(networks):
        node_mapping = {}  # old_id -> new_id
        
        # Copy nodes with new IDs
        for old_id, node in network.nodes.items():
            new_id = old_id + node_id_offset
            node_mapping[old_id] = new_id
            
            # Use node's radius or derive from policy tolerance
            node_radius = node.radius if hasattr(node, 'radius') else policy.overlap_tolerance
            
            new_node = Node(
                id=new_id,
                position=Point3D(node.position.x, node.position.y, node.position.z),
                radius=node_radius,
                node_type=node.node_type,
                vessel_type=node.vessel_type if hasattr(node, 'vessel_type') else "arterial",
            )
            merged.nodes[new_id] = new_node
        
        # Copy segments with new IDs
        for old_id, segment in network.segments.items():
            new_id = old_id + segment_id_offset
            
            # Use segment's radius or derive from policy tolerance
            seg_start_radius = segment.start_radius if hasattr(segment, 'start_radius') else policy.overlap_tolerance
            seg_end_radius = segment.end_radius if hasattr(segment, 'end_radius') else policy.overlap_tolerance
            
            new_segment = VesselSegment(
                id=new_id,
                start_node_id=node_mapping[segment.start_node_id],
                end_node_id=node_mapping[segment.end_node_id],
                start_radius=seg_start_radius,
                end_radius=seg_end_radius,
                vessel_type=segment.vessel_type if hasattr(segment, 'vessel_type') else "arterial",
            )
            merged.segments[new_id] = new_segment
        
        # Update offsets for next network
        if network.nodes:
            node_id_offset = max(merged.nodes.keys()) + 1
        if network.segments:
            segment_id_offset = max(merged.segments.keys()) + 1
    
    # Resolve overlaps if enabled
    if policy.resolve_overlaps:
        merged, overlap_count = _resolve_overlaps(merged, policy)
        metadata["overlaps_resolved"] = overlap_count
        
        if overlap_count > 0:
            logger.info(f"Resolved {overlap_count} overlapping nodes")
    
    metadata["nodes_after_merge"] = len(merged.nodes)
    metadata["segments_after_merge"] = len(merged.segments)
    
    report = OperationReport(
        operation="merge_networks",
        success=True,
        requested_policy=policy.to_dict(),
        effective_policy=policy.to_dict(),
        warnings=warnings,
        metadata=metadata,
    )
    
    return merged, report


def _resolve_overlaps(
    network: "VascularNetwork",
    policy: NetworkMergePolicy,
) -> Tuple["VascularNetwork", int]:
    """
    Resolve overlapping nodes in a network.
    
    Returns the modified network and count of resolved overlaps.
    """
    from ...core.types import Point3D
    
    tolerance = policy.overlap_tolerance
    overlap_count = 0
    
    # Build position array
    node_ids = list(network.nodes.keys())
    positions = np.array([
        [network.nodes[nid].position.x,
         network.nodes[nid].position.y,
         network.nodes[nid].position.z]
        for nid in node_ids
    ])
    
    # Find overlapping pairs using union-find
    parent = {nid: nid for nid in node_ids}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False
    
    # Find pairs within tolerance
    for i, nid1 in enumerate(node_ids):
        for j, nid2 in enumerate(node_ids[i+1:], i+1):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < tolerance:
                if union(nid1, nid2):
                    overlap_count += 1
    
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
            # Calculate merged position (centroid)
            group_positions = np.array([
                [network.nodes[nid].position.x,
                 network.nodes[nid].position.y,
                 network.nodes[nid].position.z]
                for nid in group
            ])
            centroid = group_positions.mean(axis=0)
            
            # Calculate merged radius based on strategy
            radii = [network.nodes[nid].radius for nid in group if hasattr(network.nodes[nid], 'radius')]
            if radii:
                if policy.merge_radii_strategy == "max":
                    merged_radius = max(radii)
                elif policy.merge_radii_strategy == "min":
                    merged_radius = min(radii)
                else:  # average
                    merged_radius = sum(radii) / len(radii)
            else:
                # Use policy tolerance as fallback radius
                merged_radius = policy.overlap_tolerance
            
            # Update root node
            network.nodes[root].position = Point3D(*centroid)
            network.nodes[root].radius = merged_radius
            
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
    
    # Remove self-referencing segments
    segments_to_remove = [
        seg_id for seg_id, seg in network.segments.items()
        if seg.start_node_id == seg.end_node_id
    ]
    for seg_id in segments_to_remove:
        del network.segments[seg_id]
    
    return network, overlap_count


__all__ = [
    "merge_networks",
    "NetworkMergePolicy",
]
