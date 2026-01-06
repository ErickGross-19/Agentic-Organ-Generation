"""
Uniform grid-based spatial index for fast neighbor queries.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
from ..core.types import Point3D
from ..core.network import VascularNetwork, VesselSegment


class SpatialIndex:
    """
    Uniform 3D grid spatial index for vascular networks.
    
    Provides efficient collision detection and neighbor queries.
    """
    
    def __init__(self, network: VascularNetwork, cell_size: float = 0.005):
        """
        Initialize spatial index.
        
        Parameters
        ----------
        network : VascularNetwork
            Network to index
        cell_size : float
            Size of grid cells (meters). Should be ~2-3x typical segment length.
        """
        self.network = network
        self.cell_size = cell_size
        self.grid: Dict[Tuple[int, int, int], Set[int]] = defaultdict(set)
        
        self._build_index()
    
    def _get_cell_coords(self, point: Point3D) -> Tuple[int, int, int]:
        """Convert world coordinates to grid cell coordinates."""
        return (
            int(np.floor(point.x / self.cell_size)),
            int(np.floor(point.y / self.cell_size)),
            int(np.floor(point.z / self.cell_size)),
        )
    
    def _get_cells_for_segment(self, segment: VesselSegment) -> Set[Tuple[int, int, int]]:
        """Get all grid cells that a segment intersects."""
        start_node = self.network.get_node(segment.start_node_id)
        end_node = self.network.get_node(segment.end_node_id)
        
        if start_node is None or end_node is None:
            return set()
        
        start_pos = start_node.position
        end_pos = end_node.position
        
        cell1 = self._get_cell_coords(start_pos)
        cell2 = self._get_cell_coords(end_pos)
        
        cells = set()
        
        for i in range(min(cell1[0], cell2[0]), max(cell1[0], cell2[0]) + 1):
            for j in range(min(cell1[1], cell2[1]), max(cell1[1], cell2[1]) + 1):
                for k in range(min(cell1[2], cell2[2]), max(cell1[2], cell2[2]) + 1):
                    cells.add((i, j, k))
        
        return cells
    
    def _build_index(self) -> None:
        """Build spatial index from network segments."""
        self.grid.clear()
        
        for segment_id, segment in self.network.segments.items():
            cells = self._get_cells_for_segment(segment)
            for cell in cells:
                self.grid[cell].add(segment_id)
    
    def query_nearby_segments(
        self,
        point: Point3D,
        radius: float,
    ) -> List[VesselSegment]:
        """
        Query segments near a point.
        
        Parameters
        ----------
        point : Point3D
            Query point
        radius : float
            Search radius
        
        Returns
        -------
        segments : List[VesselSegment]
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
            segment = self.network.get_segment(seg_id)
            if segment is None:
                continue
            
            dist = self._point_to_segment_distance(point, segment)
            if dist <= radius:
                nearby.append(segment)
        
        return nearby
    
    def _point_to_segment_distance(self, point: Point3D, segment: VesselSegment) -> float:
        """Compute minimum distance from point to segment centerline."""
        start_node = self.network.get_node(segment.start_node_id)
        end_node = self.network.get_node(segment.end_node_id)
        
        if start_node is None or end_node is None:
            return float('inf')
        
        p1 = start_node.position.to_array()
        p2 = end_node.position.to_array()
        p = point.to_array()
        
        v = p2 - p1
        length_sq = np.dot(v, v)
        
        if length_sq < 1e-10:
            return float(np.linalg.norm(p - p1))
        
        t = np.dot(p - p1, v) / length_sq
        t = np.clip(t, 0.0, 1.0)
        
        closest = p1 + t * v
        
        return float(np.linalg.norm(p - closest))
    
    def get_collisions(
        self,
        min_clearance: float,
        exclude_connected: bool = True,
    ) -> List[Tuple[int, int, float]]:
        """
        Find all segment pairs that are too close.
        
        Parameters
        ----------
        min_clearance : float
            Minimum required clearance between segments
        exclude_connected : bool
            If True, exclude segments that share a node
        
        Returns
        -------
        collisions : List[Tuple[int, int, float]]
            List of (segment_id1, segment_id2, distance) tuples
        """
        collisions = []
        segment_ids = list(self.network.segments.keys())
        
        for i, seg_id1 in enumerate(segment_ids):
            seg1 = self.network.get_segment(seg_id1)
            if seg1 is None:
                continue
            
            start_node1 = self.network.get_node(seg1.start_node_id)
            if start_node1 is None:
                continue
            
            nearby = self.query_nearby_segments(
                start_node1.position,
                min_clearance * 3.0,
            )
            
            for seg2 in nearby:
                if seg2.id <= seg_id1:
                    continue
                
                if exclude_connected:
                    if (seg1.start_node_id == seg2.start_node_id or
                        seg1.start_node_id == seg2.end_node_id or
                        seg1.end_node_id == seg2.start_node_id or
                        seg1.end_node_id == seg2.end_node_id):
                        continue
                
                dist = self._segment_to_segment_distance(seg1, seg2)
                
                r1 = seg1.geometry.mean_radius()
                r2 = seg2.geometry.mean_radius()
                
                required_clearance = r1 + r2 + min_clearance
                
                if dist < required_clearance:
                    collisions.append((seg_id1, seg2.id, dist))
        
        return collisions
    
    def _segment_to_segment_distance(self, seg1: VesselSegment, seg2: VesselSegment) -> float:
        """Compute minimum distance between two segment centerlines."""
        start1 = self.network.get_node(seg1.start_node_id)
        end1 = self.network.get_node(seg1.end_node_id)
        start2 = self.network.get_node(seg2.start_node_id)
        end2 = self.network.get_node(seg2.end_node_id)
        
        if None in (start1, end1, start2, end2):
            return float('inf')
        
        p1_start = start1.position.to_array()
        p1_end = end1.position.to_array()
        p2_start = start2.position.to_array()
        p2_end = end2.position.to_array()
        
        points1 = [p1_start, (p1_start + p1_end) / 2, p1_end]
        points2 = [p2_start, (p2_start + p2_end) / 2, p2_end]
        
        min_dist = float('inf')
        for p1 in points1:
            for p2 in points2:
                dist = np.linalg.norm(p1 - p2)
                min_dist = min(min_dist, dist)
        
        return float(min_dist)
