"""
Core network data structures.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from .types import Point3D, TubeGeometry
from .ids import IDGenerator
from .domain import DomainSpec
from .result import Delta, OperationResult, OperationStatus


@dataclass
class Node:
    """
    Node in a vascular network.
    
    Represents junctions, inlets, outlets, or terminal points.
    """
    
    id: int
    position: Point3D
    node_type: str  # "junction", "inlet", "outlet", "terminal"
    vessel_type: str  # "arterial", "venous"
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "position": self.position.to_dict(),
            "node_type": self.node_type,
            "vessel_type": self.vessel_type,
            "attributes": self.attributes,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        """Create from dictionary."""
        return cls(
            id=d["id"],
            position=Point3D.from_dict(d["position"]),
            node_type=d["node_type"],
            vessel_type=d["vessel_type"],
            attributes=d.get("attributes", {}),
        )


@dataclass
class VesselSegment:
    """
    Vessel segment connecting two nodes.
    
    Represents a tubular blood vessel with geometry and properties.
    """
    
    id: int
    start_node_id: int
    end_node_id: int
    geometry: TubeGeometry
    vessel_type: str  # "arterial", "venous"
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def length(self) -> float:
        """Get segment length from geometry."""
        return self.geometry.length()
    
    @property
    def direction(self):
        """Get segment direction from geometry."""
        return self.geometry.direction()
    
    @property
    def mean_radius(self) -> float:
        """Get mean radius from geometry."""
        return self.geometry.mean_radius()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "start_node_id": self.start_node_id,
            "end_node_id": self.end_node_id,
            "geometry": self.geometry.to_dict(),
            "vessel_type": self.vessel_type,
            "attributes": self.attributes,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "VesselSegment":
        """Create from dictionary."""
        return cls(
            id=d["id"],
            start_node_id=d["start_node_id"],
            end_node_id=d["end_node_id"],
            geometry=TubeGeometry.from_dict(d["geometry"]),
            vessel_type=d["vessel_type"],
            attributes=d.get("attributes", {}),
        )


class VascularNetwork:
    """
    Complete vascular network with nodes, segments, and domain.
    
    This is the main data structure for LLM-driven vascular design.
    """
    
    def __init__(
        self,
        domain: DomainSpec,
        metadata: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize vascular network.
        
        Parameters
        ----------
        domain : DomainSpec
            Geometric domain for the network
        metadata : dict, optional
            Network metadata (name, organ type, units, etc.)
        seed : int, optional
            Random seed for deterministic operations
        """
        self.nodes: Dict[int, Node] = {}
        self.segments: Dict[int, VesselSegment] = {}
        self.domain = domain
        self.metadata = metadata or {}
        self.id_gen = IDGenerator(seed=seed)
        
        self._spatial_index = None
        self._adjacency_map: Dict[int, List[int]] = {}  # node_id -> list of segment_ids
        
        self._undo_stack: List[Delta] = []
        self._redo_stack: List[Delta] = []
    
    def add_node(self, node: Node) -> None:
        """Add a node to the network."""
        self.nodes[node.id] = node
        self._invalidate_spatial_index()
    
    def add_segment(self, segment: VesselSegment) -> None:
        """Add a segment to the network."""
        if segment.start_node_id not in self.nodes:
            raise ValueError(f"Start node {segment.start_node_id} not in network")
        if segment.end_node_id not in self.nodes:
            raise ValueError(f"End node {segment.end_node_id} not in network")
        self.segments[segment.id] = segment
        
        # Update adjacency map
        if segment.start_node_id not in self._adjacency_map:
            self._adjacency_map[segment.start_node_id] = []
        self._adjacency_map[segment.start_node_id].append(segment.id)
        
        if segment.end_node_id not in self._adjacency_map:
            self._adjacency_map[segment.end_node_id] = []
        self._adjacency_map[segment.end_node_id].append(segment.id)
        
        self._invalidate_spatial_index()
    
    def remove_node(self, node_id: int) -> None:
        """Remove a node from the network."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self._invalidate_spatial_index()
    
    def remove_segment(self, segment_id: int) -> None:
        """Remove a segment from the network."""
        if segment_id in self.segments:
            segment = self.segments[segment_id]
            
            # Update adjacency map
            if segment.start_node_id in self._adjacency_map:
                try:
                    self._adjacency_map[segment.start_node_id].remove(segment_id)
                except ValueError:
                    pass  # Segment not in list (shouldn't happen)
            
            if segment.end_node_id in self._adjacency_map:
                try:
                    self._adjacency_map[segment.end_node_id].remove(segment_id)
                except ValueError:
                    pass  # Segment not in list (shouldn't happen)
            
            del self.segments[segment_id]
            self._invalidate_spatial_index()
    
    def get_node(self, node_id: int) -> Optional[Node]:
        """Get node by ID."""
        return self.nodes.get(node_id)
    
    def get_segment(self, segment_id: int) -> Optional[VesselSegment]:
        """Get segment by ID."""
        return self.segments.get(segment_id)
    
    def get_connected_segment_ids(self, node_id: int) -> List[int]:
        """
        Get IDs of all segments connected to a node.
        
        Uses cached adjacency map for O(1) lookup instead of O(E) scan.
        
        Parameters
        ----------
        node_id : int
            ID of the node
            
        Returns
        -------
        List[int]
            List of segment IDs connected to this node
        """
        return list(self._adjacency_map.get(node_id, []))
    
    def get_spatial_index(self) -> "SpatialIndex":
        """Get or create spatial index."""
        if self._spatial_index is None:
            from ..spatial.grid_index import SpatialIndex
            self._spatial_index = SpatialIndex(self)
        return self._spatial_index
    
    def _invalidate_spatial_index(self) -> None:
        """Invalidate spatial index after modifications."""
        self._spatial_index = None
    
    def _rebuild_adjacency_map(self) -> None:
        """Rebuild adjacency map from current segments."""
        self._adjacency_map.clear()
        for seg_id, seg in self.segments.items():
            if seg.start_node_id not in self._adjacency_map:
                self._adjacency_map[seg.start_node_id] = []
            self._adjacency_map[seg.start_node_id].append(seg_id)
            
            if seg.end_node_id not in self._adjacency_map:
                self._adjacency_map[seg.end_node_id] = []
            self._adjacency_map[seg.end_node_id].append(seg_id)
    
    def sync_segment_geometry_from_nodes(self) -> None:
        """
        Sync segment geometry.start/end from node positions.
        
        P0-NEW-5: After operations that move nodes or rewire segments (NLP refinement,
        topology swaps, etc.), the segment.geometry.start/end may become inconsistent
        with the actual node positions. This method updates all segment geometries
        to match their connected node positions.
        
        Call this after any operation that:
        - Moves node positions
        - Rewires segment connections
        - Performs NLP refinement
        - Swaps topology
        
        Note: This preserves centerline_points if present, only updating the
        start/end positions. For segments with centerline_points, you may also
        need to update the intermediate points if the overall path has changed.
        """
        from .types import TubeGeometry
        
        for seg in self.segments.values():
            start_node = self.nodes.get(seg.start_node_id)
            end_node = self.nodes.get(seg.end_node_id)
            
            if start_node is None or end_node is None:
                continue
            
            # Update geometry start/end to match node positions
            seg.geometry = TubeGeometry(
                start=start_node.position,
                end=end_node.position,
                radius_start=seg.geometry.radius_start,
                radius_end=seg.geometry.radius_end,
                centerline_points=seg.geometry.centerline_points,
            )
        
        # Invalidate spatial index since geometry has changed
        self._invalidate_spatial_index()
    
    def snapshot(self) -> dict:
        """Create a snapshot of current network state."""
        return {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "segments": {sid: seg.to_dict() for sid, seg in self.segments.items()},
            "domain": self.domain.to_dict(),
            "metadata": self.metadata.copy(),
            "id_gen_state": self.id_gen.get_state(),
        }
    
    def restore(self, snapshot: dict) -> None:
        """Restore network from a snapshot."""
        self.nodes = {int(nid): Node.from_dict(n) for nid, n in snapshot["nodes"].items()}
        self.segments = {int(sid): VesselSegment.from_dict(s) for sid, s in snapshot["segments"].items()}
        self.metadata = snapshot["metadata"].copy()
        self.id_gen.set_state(snapshot["id_gen_state"])
        self._rebuild_adjacency_map()
        self._invalidate_spatial_index()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "schema_version": "1.0",
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "segments": {sid: seg.to_dict() for sid, seg in self.segments.items()},
            "domain": self.domain.to_dict(),
            "metadata": self.metadata,
            "id_gen_state": self.id_gen.get_state(),
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "VascularNetwork":
        """Create from dictionary."""
        from .domain import domain_from_dict
        
        domain = domain_from_dict(d["domain"])
        network = cls(domain=domain, metadata=d.get("metadata", {}))
        
        for nid, node_dict in d["nodes"].items():
            network.nodes[int(nid)] = Node.from_dict(node_dict)
        
        for sid, seg_dict in d["segments"].items():
            network.segments[int(sid)] = VesselSegment.from_dict(seg_dict)
        
        if "id_gen_state" in d:
            network.id_gen.set_state(d["id_gen_state"])
        
        # Rebuild adjacency map after loading segments
        network._rebuild_adjacency_map()
        
        return network
    
    def apply_delta(self, delta: Delta) -> OperationResult:
        """
        Apply a delta to the network.
        
        This is the core operation for undo/redo functionality.
        
        Parameters
        ----------
        delta : Delta
            The delta to apply
            
        Returns
        -------
        OperationResult
            Result of applying the delta
        """
        try:
            for node_id in delta.created_node_ids:
                if node_id in self.nodes:
                    return OperationResult.failure(
                        f"Cannot create node {node_id}: already exists"
                    )
            
            for seg_id in delta.created_segment_ids:
                if seg_id in self.segments:
                    return OperationResult.failure(
                        f"Cannot create segment {seg_id}: already exists"
                    )
            
            for node_id in delta.deleted_node_ids:
                if node_id not in self.nodes:
                    return OperationResult.failure(
                        f"Cannot delete node {node_id}: not found"
                    )
                self.remove_node(node_id)
            
            for seg_id in delta.deleted_segment_ids:
                if seg_id not in self.segments:
                    return OperationResult.failure(
                        f"Cannot delete segment {seg_id}: not found"
                    )
                self.remove_segment(seg_id)
            
            for node_id, node_dict in delta.modified_nodes.items():
                if node_id not in self.nodes:
                    return OperationResult.failure(
                        f"Cannot modify node {node_id}: not found"
                    )
                self.nodes[node_id] = Node.from_dict(node_dict)
            
            for seg_id, seg_dict in delta.modified_segments.items():
                if seg_id not in self.segments:
                    return OperationResult.failure(
                        f"Cannot modify segment {seg_id}: not found"
                    )
                self.segments[seg_id] = VesselSegment.from_dict(seg_dict)
            
            self._invalidate_spatial_index()
            
            return OperationResult.success("Delta applied successfully")
            
        except Exception as e:
            return OperationResult.failure(f"Error applying delta: {e}")
    
    def push_undo(self, delta: Delta) -> None:
        """
        Push a delta onto the undo stack.
        
        This should be called after every successful mutating operation.
        """
        self._undo_stack.append(delta)
        self._redo_stack.clear()
    
    def undo(self) -> OperationResult:
        """
        Undo the last operation.
        
        Returns
        -------
        OperationResult
            Result of the undo operation
        """
        if not self._undo_stack:
            return OperationResult.failure("Nothing to undo")
        
        delta = self._undo_stack.pop()
        
        inverse_delta = Delta(
            created_node_ids=delta.deleted_node_ids,
            created_segment_ids=delta.deleted_segment_ids,
            deleted_node_ids=delta.created_node_ids,
            deleted_segment_ids=delta.created_segment_ids,
        )
        
        result = self.apply_delta(inverse_delta)
        
        if result.is_success():
            self._redo_stack.append(delta)
            result.message = "Undo successful"
        
        return result
    
    def redo(self) -> OperationResult:
        """
        Redo the last undone operation.
        
        Returns
        -------
        OperationResult
            Result of the redo operation
        """
        if not self._redo_stack:
            return OperationResult.failure("Nothing to redo")
        
        delta = self._redo_stack.pop()
        result = self.apply_delta(delta)
        
        if result.is_success():
            self._undo_stack.append(delta)
            result.message = "Redo successful"
        
        return result
    
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self._undo_stack) > 0
    
    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self._redo_stack) > 0
