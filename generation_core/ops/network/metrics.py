"""
Network metrics computation for vascular networks.

This module provides functions for computing statistics and metrics
about vascular networks.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, TYPE_CHECKING
import numpy as np
import logging

if TYPE_CHECKING:
    from ...core.network import VascularNetwork

logger = logging.getLogger(__name__)


@dataclass
class NetworkMetrics:
    """
    Computed metrics for a vascular network.
    
    All length/radius values are in meters.
    """
    node_count: int = 0
    segment_count: int = 0
    terminal_count: int = 0
    inlet_count: int = 0
    outlet_count: int = 0
    junction_count: int = 0
    
    total_length: float = 0.0
    mean_segment_length: float = 0.0
    min_segment_length: float = 0.0
    max_segment_length: float = 0.0
    
    mean_radius: float = 0.0
    min_radius: float = 0.0
    max_radius: float = 0.0
    
    bounding_box: Dict[str, float] = field(default_factory=dict)
    
    connectivity: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_count": self.node_count,
            "segment_count": self.segment_count,
            "terminal_count": self.terminal_count,
            "inlet_count": self.inlet_count,
            "outlet_count": self.outlet_count,
            "junction_count": self.junction_count,
            "total_length": self.total_length,
            "mean_segment_length": self.mean_segment_length,
            "min_segment_length": self.min_segment_length,
            "max_segment_length": self.max_segment_length,
            "mean_radius": self.mean_radius,
            "min_radius": self.min_radius,
            "max_radius": self.max_radius,
            "bounding_box": self.bounding_box,
            "connectivity": self.connectivity,
        }


def compute_network_metrics(
    network: "VascularNetwork",
) -> NetworkMetrics:
    """
    Compute comprehensive metrics for a vascular network.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to analyze
        
    Returns
    -------
    NetworkMetrics
        Computed metrics
    """
    from ...core.types import NodeType
    
    metrics = NetworkMetrics()
    
    # Node counts
    metrics.node_count = len(network.nodes)
    metrics.segment_count = len(network.segments)
    
    for node in network.nodes.values():
        if node.node_type == NodeType.TERMINAL:
            metrics.terminal_count += 1
        elif node.node_type == NodeType.INLET:
            metrics.inlet_count += 1
        elif node.node_type == NodeType.OUTLET:
            metrics.outlet_count += 1
        elif node.node_type == NodeType.JUNCTION:
            metrics.junction_count += 1
    
    # Segment lengths
    segment_lengths = []
    for segment in network.segments.values():
        start_node = network.nodes.get(segment.start_node_id)
        end_node = network.nodes.get(segment.end_node_id)
        
        if start_node and end_node:
            start_pos = np.array([start_node.position.x, start_node.position.y, start_node.position.z])
            end_pos = np.array([end_node.position.x, end_node.position.y, end_node.position.z])
            length = np.linalg.norm(end_pos - start_pos)
            segment_lengths.append(length)
    
    if segment_lengths:
        metrics.total_length = sum(segment_lengths)
        metrics.mean_segment_length = np.mean(segment_lengths)
        metrics.min_segment_length = min(segment_lengths)
        metrics.max_segment_length = max(segment_lengths)
    
    # Radii
    radii = []
    for segment in network.segments.values():
        if hasattr(segment, 'start_radius') and segment.start_radius:
            radii.append(segment.start_radius)
        if hasattr(segment, 'end_radius') and segment.end_radius:
            radii.append(segment.end_radius)
    
    if radii:
        metrics.mean_radius = np.mean(radii)
        metrics.min_radius = min(radii)
        metrics.max_radius = max(radii)
    
    # Bounding box
    if network.nodes:
        positions = np.array([
            [node.position.x, node.position.y, node.position.z]
            for node in network.nodes.values()
        ])
        
        metrics.bounding_box = {
            "min_x": float(positions[:, 0].min()),
            "max_x": float(positions[:, 0].max()),
            "min_y": float(positions[:, 1].min()),
            "max_y": float(positions[:, 1].max()),
            "min_z": float(positions[:, 2].min()),
            "max_z": float(positions[:, 2].max()),
        }
    
    # Connectivity analysis
    metrics.connectivity = _compute_connectivity(network)
    
    return metrics


def _compute_connectivity(network: "VascularNetwork") -> Dict[str, Any]:
    """Compute connectivity metrics for the network."""
    connectivity = {
        "is_connected": True,
        "num_components": 1,
        "has_cycles": False,
    }
    
    if not network.nodes:
        connectivity["is_connected"] = False
        connectivity["num_components"] = 0
        return connectivity
    
    # Build adjacency list
    adjacency = {nid: set() for nid in network.nodes}
    for segment in network.segments.values():
        adjacency[segment.start_node_id].add(segment.end_node_id)
        adjacency[segment.end_node_id].add(segment.start_node_id)
    
    # Count connected components using BFS
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
    
    connectivity["num_components"] = num_components
    connectivity["is_connected"] = num_components == 1
    
    # Check for cycles using DFS
    visited = set()
    
    def has_cycle(node, parent):
        visited.add(node)
        for neighbor in adjacency.get(node, []):
            if neighbor not in visited:
                if has_cycle(neighbor, node):
                    return True
            elif neighbor != parent:
                return True
        return False
    
    for start_node in network.nodes:
        if start_node not in visited:
            if has_cycle(start_node, None):
                connectivity["has_cycles"] = True
                break
    
    return connectivity


__all__ = [
    "compute_network_metrics",
    "NetworkMetrics",
]
