"""
Data structures for vascular trees with spatial indexing.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set
import numpy as np
from collections import defaultdict


@dataclass
class Node:
    """Represents a node in the vascular tree."""
    
    id: int
    position: np.ndarray  # (3,) array
    radius: float
    parent_id: Optional[int] = None
    children_ids: List[int] = field(default_factory=list)
    order: int = 0  # Branch order (distance from root in bifurcations)
    flow: float = 0.0  # Optional flow value
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "id": int(self.id),
            "position": self.position.tolist(),
            "radius": float(self.radius),
            "parent_id": int(self.parent_id) if self.parent_id is not None else None,
            "children_ids": [int(cid) for cid in self.children_ids],
            "order": int(self.order),
            "flow": float(self.flow),
        }


@dataclass
class Segment:
    """Represents a segment (edge) between two nodes."""
    
    id: int
    parent_node_id: int
    child_node_id: int
    length: float
    direction: np.ndarray  # (3,) unit vector
    radius_start: float
    radius_end: float
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "id": int(self.id),
            "parent_node_id": int(self.parent_node_id),
            "child_node_id": int(self.child_node_id),
            "length": float(self.length),
            "direction": self.direction.tolist(),
            "radius_start": float(self.radius_start),
            "radius_end": float(self.radius_end),
        }


class SpatialIndex:
    """
    Simple uniform grid-based spatial index for fast neighbor queries.
    
    Uses a 3D grid where each cell stores segment IDs that intersect it.
    """
    
    def __init__(self, cell_size: float = 0.005):
        """
        Initialize spatial index.
        
        Parameters
        ----------
        cell_size : float
            Size of grid cells (meters). Should be ~2-3x typical segment length.
        """
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int, int], Set[int]] = defaultdict(set)
        self.segments: Dict[int, Segment] = {}
        self.nodes: Dict[int, Node] = {}
    
    def _get_cell_coords(self, point: np.ndarray) -> Tuple[int, int, int]:
        """Convert world coordinates to grid cell coordinates."""
        return tuple(int(np.floor(p / self.cell_size)) for p in point)
    
    def _get_cells_for_segment(self, segment: Segment) -> Set[Tuple[int, int, int]]:
        """Get all grid cells that a segment intersects."""
        parent_node = self.nodes[segment.parent_node_id]
        child_node = self.nodes[segment.child_node_id]
        
        p1 = parent_node.position
        p2 = child_node.position
        
        cell1 = self._get_cell_coords(p1)
        cell2 = self._get_cell_coords(p2)
        
        cells = set()
        
        for i in range(min(cell1[0], cell2[0]), max(cell1[0], cell2[0]) + 1):
            for j in range(min(cell1[1], cell2[1]), max(cell1[1], cell2[1]) + 1):
                for k in range(min(cell1[2], cell2[2]), max(cell1[2], cell2[2]) + 1):
                    cells.add((i, j, k))
        
        return cells
    
    def add_node(self, node: Node):
        """Add a node to the index."""
        self.nodes[node.id] = node
    
    def add_segment(self, segment: Segment):
        """Add a segment to the spatial index."""
        self.segments[segment.id] = segment
        
        cells = self._get_cells_for_segment(segment)
        for cell in cells:
            self.grid[cell].add(segment.id)
    
    def query_nearby_segments(
        self,
        point: np.ndarray,
        radius: float,
    ) -> List[Segment]:
        """
        Query segments near a point.
        
        Parameters
        ----------
        point : (3,) array
            Query point
        radius : float
            Search radius
        
        Returns
        -------
        segments : List[Segment]
            Segments within radius of point
        """
        cell_radius = int(np.ceil(radius / self.cell_size))
        center_cell = self._get_cell_coords(point)
        
        candidate_segment_ids = set()
        for di in range(-cell_radius, cell_radius + 1):
            for dj in range(-cell_radius, cell_radius + 1):
                for dk in range(-cell_radius, cell_radius + 1):
                    cell = (
                        center_cell[0] + di,
                        center_cell[1] + dj,
                        center_cell[2] + dk,
                    )
                    candidate_segment_ids.update(self.grid.get(cell, set()))
        
        nearby = []
        for seg_id in candidate_segment_ids:
            segment = self.segments[seg_id]
            dist = self._point_to_segment_distance(point, segment)
            if dist <= radius:
                nearby.append(segment)
        
        return nearby
    
    def _point_to_segment_distance(self, point: np.ndarray, segment: Segment) -> float:
        """Compute minimum distance from point to segment centerline."""
        p1 = self.nodes[segment.parent_node_id].position
        p2 = self.nodes[segment.child_node_id].position
        
        v = p2 - p1
        length_sq = np.dot(v, v)
        
        if length_sq < 1e-10:
            return float(np.linalg.norm(point - p1))
        
        t = np.dot(point - p1, v) / length_sq
        t = np.clip(t, 0.0, 1.0)
        
        closest = p1 + t * v
        
        return float(np.linalg.norm(point - closest))


@dataclass
class ActiveTip:
    """Represents an active growth tip."""
    
    node_id: int
    direction: np.ndarray  # (3,) unit vector
    attempts: int = 0  # Number of failed growth attempts


class VascularTree:
    """Represents a complete vascular tree with spatial indexing."""
    
    def __init__(
        self,
        tree_type: str,
        root_position: np.ndarray,
        root_radius: float,
        initial_direction: np.ndarray,
        root_id: int = 0,
    ):
        """
        Initialize a vascular tree.
        
        Parameters
        ----------
        tree_type : str
            "arterial" or "venous"
        root_position : (3,) array
            Position of root node
        root_radius : float
            Radius of root vessel
        initial_direction : (3,) array
            Initial growth direction
        root_id : int
            ID for root node (default: 0)
        """
        self.tree_type = tree_type
        self.nodes: List[Node] = []
        self.nodes_by_id: Dict[int, Node] = {}
        self.segments: List[Segment] = []
        self.spatial_index = SpatialIndex(cell_size=0.005)
        self.active_tips: List[ActiveTip] = []
        
        root = Node(
            id=root_id,
            position=root_position.copy(),
            radius=root_radius,
            parent_id=None,
            order=0,
        )
        self.nodes.append(root)
        self.nodes_by_id[root.id] = root
        self.spatial_index.add_node(root)
        
        self.active_tips.append(ActiveTip(
            node_id=root_id,
            direction=initial_direction / np.linalg.norm(initial_direction),
        ))
    
    def add_node(self, node: Node):
        """Add a node to the tree."""
        self.nodes.append(node)
        self.nodes_by_id[node.id] = node
        self.spatial_index.add_node(node)
    
    def add_segment(self, segment: Segment):
        """Add a segment to the tree."""
        self.segments.append(segment)
        self.spatial_index.add_segment(segment)
    
    def get_node(self, node_id: int) -> Node:
        """Get node by ID."""
        return self.nodes_by_id[node_id]
    
    def to_dict(self):
        """Convert tree to dictionary for serialization."""
        return {
            "tree_type": self.tree_type,
            "nodes": [node.to_dict() for node in self.nodes],
            "segments": [seg.to_dict() for seg in self.segments],
        }
