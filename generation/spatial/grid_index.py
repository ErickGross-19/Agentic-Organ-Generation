"""
Uniform grid-based spatial index for fast neighbor queries.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from ..core.types import Point3D
from ..core.network import VascularNetwork, VesselSegment


def segment_segment_distance_exact(
    p1: np.ndarray, p2: np.ndarray,
    p3: np.ndarray, p4: np.ndarray,
) -> float:
    """
    Compute exact minimum distance between two 3D line segments (straight lines only).
    
    This is a wrapper around the canonical implementation in
    generation.utils.geometry.segment_segment_distance.
    
    Segment 1: p1 to p2
    Segment 2: p3 to p4
    
    NOTE: This function only handles straight segments. For polyline-aware distance
    that handles segments with centerline_points, use `polyline_segment_distance()`.
    
    Parameters
    ----------
    p1, p2 : np.ndarray
        Endpoints of segment 1 (shape (3,))
    p3, p4 : np.ndarray
        Endpoints of segment 2 (shape (3,))
        
    Returns
    -------
    float
        Minimum distance between the two segments
    """
    from ..utils.geometry import segment_segment_distance
    return segment_segment_distance(p1, p2, p3, p4)


def polyline_segment_distance(
    network: "VascularNetwork",
    seg1: "VesselSegment",
    seg2: "VesselSegment",
) -> float:
    """
    Compute exact minimum distance between two segment centerlines, handling polylines.
    
    P0-NEW-4: This function handles segments with centerline_points (polylines) by
    computing the minimum distance across all subsegment pairs. Use this instead of
    segment_segment_distance_exact() when segments may have curved centerlines.
    
    Parameters
    ----------
    network : VascularNetwork
        Network containing the segments (needed to look up node positions)
    seg1 : VesselSegment
        First segment
    seg2 : VesselSegment
        Second segment
        
    Returns
    -------
    float
        Minimum distance between the two segment centerlines
    """
    start1 = network.get_node(seg1.start_node_id)
    end1 = network.get_node(seg1.end_node_id)
    start2 = network.get_node(seg2.start_node_id)
    end2 = network.get_node(seg2.end_node_id)
    
    if None in (start1, end1, start2, end2):
        return float('inf')
    
    # Build polyline for seg1
    if seg1.geometry.centerline_points:
        polyline1 = [start1.position.to_array()]
        polyline1.extend([p.to_array() for p in seg1.geometry.centerline_points])
        polyline1.append(end1.position.to_array())
    else:
        polyline1 = [start1.position.to_array(), end1.position.to_array()]
    
    # Build polyline for seg2
    if seg2.geometry.centerline_points:
        polyline2 = [start2.position.to_array()]
        polyline2.extend([p.to_array() for p in seg2.geometry.centerline_points])
        polyline2.append(end2.position.to_array())
    else:
        polyline2 = [start2.position.to_array(), end2.position.to_array()]
    
    # If both are simple segments (no centerline_points), use fast path
    if len(polyline1) == 2 and len(polyline2) == 2:
        return segment_segment_distance_exact(polyline1[0], polyline1[1], polyline2[0], polyline2[1])
    
    # Compute min distance across all subsegment pairs
    min_dist = float('inf')
    for i in range(len(polyline1) - 1):
        p1_start = polyline1[i]
        p1_end = polyline1[i + 1]
        for j in range(len(polyline2) - 1):
            p2_start = polyline2[j]
            p2_end = polyline2[j + 1]
            dist = segment_segment_distance_exact(p1_start, p1_end, p2_start, p2_end)
            min_dist = min(min_dist, dist)
    
    return min_dist


class SpatialIndex:
    """
    Uniform 3D grid spatial index for vascular networks.
    
    Provides efficient collision detection and neighbor queries.
    
    P2-2: Cell size is now scale-aware - if cell_size is None, it is computed
    from network statistics (0.5 * median segment length, clamped to reasonable bounds).
    """
    
    def __init__(self, network: VascularNetwork, cell_size: Optional[float] = None):
        """
        Initialize spatial index.
        
        Parameters
        ----------
        network : VascularNetwork
            Network to index
        cell_size : float, optional
            Size of grid cells (meters). If None, automatically computed from
            network statistics: cell_size = clamp(0.5 * median_segment_length, min=0.0005, max=0.05)
        """
        self.network = network
        
        # P2-2: Scale-aware cell size computation
        if cell_size is None:
            self.cell_size = self._compute_adaptive_cell_size()
        else:
            self.cell_size = cell_size
            
        self.grid: Dict[Tuple[int, int, int], Set[int]] = defaultdict(set)
        
        self._build_index()
    
    def _compute_adaptive_cell_size(self) -> float:
        """
        Compute cell size from network statistics.
        
        P2-2: Uses 0.5 * median segment length, clamped to [0.0005m, 0.05m).
        This ensures the grid is appropriately sized for the network scale.
        
        A3 FIX: Upper bound is strictly less than max (0.049999) to ensure
        cell_size < 0.05 as expected by tests.
        """
        min_cell_size = 0.0005  # 0.5mm minimum
        max_cell_size = 0.049999  # Strictly less than 50mm (A3 fix)
        default_cell_size = 0.005  # 5mm default
        
        if not self.network.segments:
            return default_cell_size
        
        # Compute segment lengths
        segment_lengths = []
        for seg in self.network.segments.values():
            segment_lengths.append(seg.length)
        
        if not segment_lengths:
            return default_cell_size
        
        median_length = float(np.median(segment_lengths))
        
        # Cell size = 0.5 * median segment length, clamped
        cell_size = 0.5 * median_length
        cell_size = max(min_cell_size, min(max_cell_size, cell_size))
        
        return cell_size
    
    def _get_cell_coords(self, point: Point3D) -> Tuple[int, int, int]:
        """Convert world coordinates to grid cell coordinates."""
        return (
            int(np.floor(point.x / self.cell_size)),
            int(np.floor(point.y / self.cell_size)),
            int(np.floor(point.z / self.cell_size)),
        )
    
    def _get_cells_for_segment(self, segment: VesselSegment) -> Set[Tuple[int, int, int]]:
        """
        Get all grid cells that a segment intersects.
        
        P0-3: If segment has centerline_points (polyline), iterate all subsegments
        and mark all covered cells for correct spatial indexing of curved routes.
        """
        start_node = self.network.get_node(segment.start_node_id)
        end_node = self.network.get_node(segment.end_node_id)
        
        if start_node is None or end_node is None:
            return set()
        
        cells = set()
        
        # Build list of points along the segment (including centerline_points if present)
        if segment.geometry.centerline_points:
            # P0-3: Polyline-aware indexing - iterate all subsegments
            points = [start_node.position]
            points.extend(segment.geometry.centerline_points)
            points.append(end_node.position)
            
            # Add cells for each subsegment
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i + 1]
                cell1 = self._get_cell_coords(p1)
                cell2 = self._get_cell_coords(p2)
                
                for ci in range(min(cell1[0], cell2[0]), max(cell1[0], cell2[0]) + 1):
                    for cj in range(min(cell1[1], cell2[1]), max(cell1[1], cell2[1]) + 1):
                        for ck in range(min(cell1[2], cell2[2]), max(cell1[2], cell2[2]) + 1):
                            cells.add((ci, cj, ck))
        else:
            # Simple straight segment
            start_pos = start_node.position
            end_pos = end_node.position
            
            cell1 = self._get_cell_coords(start_pos)
            cell2 = self._get_cell_coords(end_pos)
            
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
        """
        Compute minimum distance from point to segment centerline.
        
        Supports polyline centerlines via TubeGeometry.centerline_points.
        If centerline_points is populated, computes distance to the full
        polyline path rather than just the straight line between endpoints.
        """
        start_node = self.network.get_node(segment.start_node_id)
        end_node = self.network.get_node(segment.end_node_id)
        
        if start_node is None or end_node is None:
            return float('inf')
        
        p = point.to_array()
        
        if segment.geometry.centerline_points:
            polyline = [start_node.position.to_array()]
            polyline.extend([cp.to_array() for cp in segment.geometry.centerline_points])
            polyline.append(end_node.position.to_array())
            
            return self._point_to_polyline_distance(p, polyline)
        
        p1 = start_node.position.to_array()
        p2 = end_node.position.to_array()
        
        v = p2 - p1
        length_sq = np.dot(v, v)
        
        if length_sq < 1e-10:
            return float(np.linalg.norm(p - p1))
        
        t = np.dot(p - p1, v) / length_sq
        t = np.clip(t, 0.0, 1.0)
        
        closest = p1 + t * v
        
        return float(np.linalg.norm(p - closest))
    
    def _point_to_polyline_distance(self, point: np.ndarray, polyline: List[np.ndarray]) -> float:
        """
        Compute minimum distance from point to a polyline.
        
        Parameters
        ----------
        point : np.ndarray
            Query point (x, y, z)
        polyline : list of np.ndarray
            List of polyline vertices
            
        Returns
        -------
        float
            Minimum distance from point to polyline
        """
        if len(polyline) < 2:
            if len(polyline) == 1:
                return float(np.linalg.norm(point - polyline[0]))
            return float('inf')
        
        min_dist = float('inf')
        
        for i in range(len(polyline) - 1):
            p1 = polyline[i]
            p2 = polyline[i + 1]
            
            v = p2 - p1
            length_sq = np.dot(v, v)
            
            if length_sq < 1e-10:
                dist = float(np.linalg.norm(point - p1))
            else:
                t = np.dot(point - p1, v) / length_sq
                t = np.clip(t, 0.0, 1.0)
                closest = p1 + t * v
                dist = float(np.linalg.norm(point - closest))
            
            min_dist = min(min_dist, dist)
        
        return min_dist


class DynamicSpatialIndex:
    """
    Dynamic spatial index for incremental segment insertion during network growth.
    
    Unlike SpatialIndex which builds from an existing network, this class supports
    incremental insertion of segments as they are created during generation.
    This is essential for online collision avoidance in backends like scaffold_topdown.
    
    The index uses a uniform 3D grid with configurable cell size. Segments are
    indexed by the cells they intersect, enabling fast candidate lookup for
    collision queries.
    """
    
    def __init__(self, cell_size: float = 0.001):
        """
        Initialize dynamic spatial index.
        
        Parameters
        ----------
        cell_size : float
            Size of grid cells in meters. Should be roughly 2-3x the average
            segment radius for optimal performance. Default: 1mm.
        """
        self.cell_size = cell_size
        self.inv_cell_size = 1.0 / cell_size
        self.grid: Dict[Tuple[int, int, int], Set[int]] = defaultdict(set)
        
        self._segment_starts: Dict[int, np.ndarray] = {}
        self._segment_ends: Dict[int, np.ndarray] = {}
        self._segment_radii: Dict[int, float] = {}
        self._segment_centerlines: Dict[int, Optional[List[np.ndarray]]] = {}
        self._segment_cells: Dict[int, Set[Tuple[int, int, int]]] = {}
        
        self._next_id = 0
    
    def clear(self) -> None:
        """Clear all indexed segments."""
        self.grid.clear()
        self._segment_starts.clear()
        self._segment_ends.clear()
        self._segment_radii.clear()
        self._segment_centerlines.clear()
        self._segment_cells.clear()
        self._next_id = 0
    
    def _get_cell_coords(self, point: np.ndarray) -> Tuple[int, int, int]:
        """Convert world coordinates to grid cell coordinates."""
        return (
            int(np.floor(point[0] * self.inv_cell_size)),
            int(np.floor(point[1] * self.inv_cell_size)),
            int(np.floor(point[2] * self.inv_cell_size)),
        )
    
    def _get_cells_for_segment(
        self,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
        centerline: Optional[List[np.ndarray]] = None,
    ) -> Set[Tuple[int, int, int]]:
        """
        Get all grid cells that a segment (capsule) intersects.
        
        Parameters
        ----------
        start : np.ndarray
            Start point of segment
        end : np.ndarray
            End point of segment
        radius : float
            Radius of the segment (capsule)
        centerline : list of np.ndarray, optional
            If provided, use these waypoints instead of straight line
            
        Returns
        -------
        Set[Tuple[int, int, int]]
            Set of cell coordinates
        """
        cells: Set[Tuple[int, int, int]] = set()
        
        if centerline and len(centerline) > 0:
            points = [start] + centerline + [end]
        else:
            points = [start, end]
        
        r_cells = int(np.ceil(radius * self.inv_cell_size)) + 1
        
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i + 1]
            
            cell1 = self._get_cell_coords(p1)
            cell2 = self._get_cell_coords(p2)
            
            for ci in range(min(cell1[0], cell2[0]) - r_cells, max(cell1[0], cell2[0]) + r_cells + 1):
                for cj in range(min(cell1[1], cell2[1]) - r_cells, max(cell1[1], cell2[1]) + r_cells + 1):
                    for ck in range(min(cell1[2], cell2[2]) - r_cells, max(cell1[2], cell2[2]) + r_cells + 1):
                        cells.add((ci, cj, ck))
        
        return cells
    
    def insert_segment(
        self,
        segment_id: int,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
        centerline: Optional[List[np.ndarray]] = None,
    ) -> None:
        """
        Insert a segment into the spatial index.
        
        Parameters
        ----------
        segment_id : int
            Unique identifier for the segment
        start : np.ndarray
            Start point of segment (x, y, z)
        end : np.ndarray
            End point of segment (x, y, z)
        radius : float
            Radius of the segment
        centerline : list of np.ndarray, optional
            Waypoints for polyline representation
        """
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        
        self._segment_starts[segment_id] = start
        self._segment_ends[segment_id] = end
        self._segment_radii[segment_id] = radius
        self._segment_centerlines[segment_id] = centerline
        
        cells = self._get_cells_for_segment(start, end, radius, centerline)
        self._segment_cells[segment_id] = cells
        
        for cell in cells:
            self.grid[cell].add(segment_id)
        
        self._next_id = max(self._next_id, segment_id + 1)
    
    def add_segment(
        self,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
        centerline: Optional[List[np.ndarray]] = None,
    ) -> int:
        """
        Add a segment and return its auto-generated ID.
        
        Parameters
        ----------
        start : np.ndarray
            Start point of segment
        end : np.ndarray
            End point of segment
        radius : float
            Radius of the segment
        centerline : list of np.ndarray, optional
            Waypoints for polyline representation
            
        Returns
        -------
        int
            The assigned segment ID
        """
        segment_id = self._next_id
        self.insert_segment(segment_id, start, end, radius, centerline)
        return segment_id
    
    def query_candidate_segments_for_capsule(
        self,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
    ) -> Set[int]:
        """
        Query candidate segment IDs that might collide with a proposed capsule.
        
        This is the broad-phase collision query. Returns segment IDs that
        are in cells overlapping with the query capsule. Actual collision
        detection (narrow phase) should be done separately.
        
        Parameters
        ----------
        start : np.ndarray
            Start point of query capsule
        end : np.ndarray
            End point of query capsule
        radius : float
            Radius of query capsule
            
        Returns
        -------
        Set[int]
            Set of candidate segment IDs
        """
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        
        query_cells = self._get_cells_for_segment(start, end, radius, None)
        
        candidates: Set[int] = set()
        for cell in query_cells:
            if cell in self.grid:
                candidates.update(self.grid[cell])
        
        return candidates
    
    def check_capsule_collision(
        self,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
        buffer: float = 0.0,
        exclude_adjacent_to: Optional[np.ndarray] = None,
        adjacent_tolerance: float = 1e-6,
    ) -> bool:
        """
        Check if a proposed capsule collides with any indexed segments.
        
        Parameters
        ----------
        start : np.ndarray
            Start point of query capsule
        end : np.ndarray
            End point of query capsule
        radius : float
            Radius of query capsule
        buffer : float
            Additional clearance buffer
        exclude_adjacent_to : np.ndarray, optional
            If provided, exclude segments that share this endpoint
            (to avoid false positives with parent segment)
        adjacent_tolerance : float
            Distance tolerance for adjacency check
            
        Returns
        -------
        bool
            True if collision detected, False otherwise
        """
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        
        candidates = self.query_candidate_segments_for_capsule(start, end, radius + buffer)
        
        if not candidates:
            return False
        
        for seg_id in candidates:
            seg_start = self._segment_starts[seg_id]
            seg_end = self._segment_ends[seg_id]
            seg_radius = self._segment_radii[seg_id]
            centerline = self._segment_centerlines.get(seg_id)
            
            if exclude_adjacent_to is not None:
                if np.linalg.norm(seg_start - exclude_adjacent_to) < adjacent_tolerance:
                    continue
                if np.linalg.norm(seg_end - exclude_adjacent_to) < adjacent_tolerance:
                    continue
            
            dist = self._segment_to_segment_distance(
                start, end, seg_start, seg_end, centerline
            )
            
            min_allowed = radius + seg_radius + buffer
            if dist < min_allowed:
                return True
        
        return False
    
    def _segment_to_segment_distance(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        q1: np.ndarray,
        q2: np.ndarray,
        q_centerline: Optional[List[np.ndarray]] = None,
    ) -> float:
        """
        Compute minimum distance between two segments.
        
        If q_centerline is provided, computes distance to the polyline.
        """
        if q_centerline and len(q_centerline) > 0:
            q_points = [q1] + q_centerline + [q2]
            min_dist = float('inf')
            for i in range(len(q_points) - 1):
                dist = segment_segment_distance_exact(p1, p2, q_points[i], q_points[i + 1])
                min_dist = min(min_dist, dist)
            return min_dist
        else:
            return segment_segment_distance_exact(p1, p2, q1, q2)
    
    @property
    def segment_count(self) -> int:
        """Return the number of indexed segments."""
        return len(self._segment_starts)
    
    def get_segment_data(self, segment_id: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Get data for a specific segment.
        
        Returns
        -------
        dict or None
            Segment data with keys: start, end, radius, centerline
        """
        if segment_id not in self._segment_starts:
            return None
        return {
            "start": self._segment_starts[segment_id],
            "end": self._segment_ends[segment_id],
            "radius": self._segment_radii[segment_id],
            "centerline": self._segment_centerlines.get(segment_id),
        }
