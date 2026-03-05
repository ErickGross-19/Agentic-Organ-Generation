"""
Drift report for tracking specification drift.

This module provides functions for computing and reporting drift
between requested specifications and actual results.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import numpy as np
import logging

if TYPE_CHECKING:
    import trimesh
    from generation.core.network import VascularNetwork

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """
    Report on drift between specification and results.
    
    Tracks differences between requested parameters and actual
    achieved values after generation and embedding.
    """
    # Terminal count drift
    requested_terminals: Optional[int] = None
    achieved_terminals: Optional[int] = None
    terminal_drift_percent: Optional[float] = None
    
    # Radius drift
    requested_min_radius: Optional[float] = None
    achieved_min_radius: Optional[float] = None
    radius_drift_percent: Optional[float] = None
    
    # Volume drift
    requested_volume: Optional[float] = None
    achieved_volume: Optional[float] = None
    volume_drift_percent: Optional[float] = None
    
    # Connectivity
    connectivity_preserved: bool = True
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return asdict(self)


def compute_drift_report(
    network: Optional["VascularNetwork"] = None,
    void_mesh: Optional["trimesh.Trimesh"] = None,
    requested_spec: Optional[Dict[str, Any]] = None,
) -> DriftReport:
    """
    Compute drift metrics between specification and results.
    
    Parameters
    ----------
    network : VascularNetwork, optional
        Generated network
    void_mesh : trimesh.Trimesh, optional
        Generated void mesh
    requested_spec : dict, optional
        Original specification with requested values
        
    Returns
    -------
    DriftReport
        Report with drift metrics
    """
    report = DriftReport()
    
    if requested_spec is None:
        requested_spec = {}
    
    # Terminal count drift
    if network is not None:
        from generation.core.types import NodeType
        
        achieved_terminals = sum(
            1 for n in network.nodes.values() 
            if n.node_type == NodeType.TERMINAL
        )
        report.achieved_terminals = achieved_terminals
        
        requested_terminals = requested_spec.get("target_terminals")
        if requested_terminals is not None:
            report.requested_terminals = requested_terminals
            if requested_terminals > 0:
                drift = abs(achieved_terminals - requested_terminals) / requested_terminals
                report.terminal_drift_percent = drift * 100
                
                if drift > 0.1:  # More than 10% drift
                    report.warnings.append(
                        f"Terminal count drift: {drift*100:.1f}% "
                        f"(requested: {requested_terminals}, achieved: {achieved_terminals})"
                    )
    
    # Radius drift
    if network is not None:
        radii = []
        for segment in network.segments.values():
            if hasattr(segment, 'start_radius') and segment.start_radius:
                radii.append(segment.start_radius)
            if hasattr(segment, 'end_radius') and segment.end_radius:
                radii.append(segment.end_radius)
        
        if radii:
            achieved_min_radius = min(radii)
            report.achieved_min_radius = achieved_min_radius
            
            requested_min_radius = requested_spec.get("min_radius")
            if requested_min_radius is not None:
                report.requested_min_radius = requested_min_radius
                if requested_min_radius > 0:
                    drift = abs(achieved_min_radius - requested_min_radius) / requested_min_radius
                    report.radius_drift_percent = drift * 100
                    
                    if achieved_min_radius < requested_min_radius:
                        report.warnings.append(
                            f"Min radius below requested: {achieved_min_radius*1000:.3f}mm "
                            f"< {requested_min_radius*1000:.3f}mm"
                        )
    
    # Volume drift
    if void_mesh is not None:
        try:
            achieved_volume = abs(float(void_mesh.volume))
            report.achieved_volume = achieved_volume
            
            requested_volume = requested_spec.get("target_volume")
            if requested_volume is not None:
                report.requested_volume = requested_volume
                if requested_volume > 0:
                    drift = abs(achieved_volume - requested_volume) / requested_volume
                    report.volume_drift_percent = drift * 100
                    
                    if drift > 0.2:  # More than 20% drift
                        report.warnings.append(
                            f"Volume drift: {drift*100:.1f}% "
                            f"(requested: {requested_volume:.9e}, achieved: {achieved_volume:.9e})"
                        )
        except Exception as e:
            logger.warning(f"Could not compute volume drift: {e}")
    
    # Connectivity check
    if network is not None:
        adjacency = {nid: set() for nid in network.nodes}
        for segment in network.segments.values():
            if segment.start_node_id in adjacency and segment.end_node_id in adjacency:
                adjacency[segment.start_node_id].add(segment.end_node_id)
                adjacency[segment.end_node_id].add(segment.start_node_id)
        
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
        
        report.connectivity_preserved = num_components == 1
        
        if num_components > 1:
            report.warnings.append(
                f"Network has {num_components} disconnected components"
            )
    
    return report


__all__ = [
    "DriftReport",
    "compute_drift_report",
]
