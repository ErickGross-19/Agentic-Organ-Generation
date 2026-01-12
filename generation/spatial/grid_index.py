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
    
    Segment 1: p1 to p2
    Segment 2: p3 to p4
    
    Uses the analytic formula for closest points on two line segments.
    This handles all cases including parallel segments and endpoint proximity.
    
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
    d1 = p2 - p1  # Direction of segment 1
    d2 = p4 - p3  # Direction of segment 2
    r = p1 - p3
    
    a = np.dot(d1, d1)  # |d1|^2
    e = np.dot(d2, d2)  # |d2|^2
    f = np.dot(d2, r)
    
    EPSILON = 1e-10
    
    # Check if both segments are degenerate (points)
    if a < EPSILON and e < EPSILON:
        return float(np.linalg.norm(r))
    
    # Check if segment 1 is degenerate (point)
    if a < EPSILON:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = np.dot(d1, r)
        
        # Check if segment 2 is degenerate (point)
        if e < EPSILON:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            # General non-degenerate case
            b = np.dot(d1, d2)
            denom = a * e - b * b  # Always >= 0
            
            # If segments are not parallel, compute closest point on line 1 to line 2
            if denom > EPSILON:
                s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
            else:
                # Segments are parallel, pick arbitrary s
                s = 0.0
            
            # Compute point on line 2 closest to S1(s)
            t = (b * s + f) / e
            
            # If t is outside [0,1], clamp and recompute s
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)
    
    # Compute closest points
    closest1 = p1 + s * d1
    closest2 = p3 + t * d2
    
    return float(np.linalg.norm(closest1 - closest2))


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
        
        P2-2: Uses 0.5 * median segment length, clamped to [0.0005m, 0.05m].
        This ensures the grid is appropriately sized for the network scale.
        """
        min_cell_size = 0.0005  # 0.5mm minimum
        max_cell_size = 0.05    # 50mm maximum
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
            Minimum required clearance between segment surfaces
        exclude_connected : bool
            If True, exclude segments that share a node
        
        Returns
        -------
        collisions : List[Tuple[int, int, float]]
            List of (segment_id1, segment_id2, clearance) tuples where
            clearance = centerline_distance - r1 - r2. Negative clearance
            indicates overlapping vessels.
        """
        collisions = []
        checked_pairs = set()
        segment_ids = list(self.network.segments.keys())
        
        # P2-NEW-1: Compute r_max once before iterating (was O(N²) when inside loop)
        if not self.network.segments:
            return collisions
        r_max = max(seg.geometry.mean_radius() for seg in self.network.segments.values())
        
        for seg_id1 in segment_ids:
            seg1 = self.network.get_segment(seg_id1)
            if seg1 is None:
                continue
            
            # P0-2: Query at start, mid, and end points to catch mid-segment collisions
            start_node1 = self.network.get_node(seg1.start_node_id)
            end_node1 = self.network.get_node(seg1.end_node_id)
            if start_node1 is None or end_node1 is None:
                continue
            
            r1 = seg1.geometry.mean_radius()
            
            # Compute midpoint
            mid_pos = Point3D(
                (start_node1.position.x + end_node1.position.x) / 2,
                (start_node1.position.y + end_node1.position.y) / 2,
                (start_node1.position.z + end_node1.position.z) / 2,
            )
            
            # Search radius uses pre-computed r_max
            search_radius = r1 + r_max + min_clearance + 0.001  # margin
            
            # Query at start, mid, and end points
            query_points = [start_node1.position, mid_pos, end_node1.position]
            candidate_segments = set()
            for qp in query_points:
                nearby = self.query_nearby_segments(qp, search_radius)
                for seg in nearby:
                    candidate_segments.add(seg.id)
            
            for seg2_id in candidate_segments:
                if seg2_id == seg_id1:
                    continue
                
                # Avoid checking same pair twice
                pair_key = (min(seg_id1, seg2_id), max(seg_id1, seg2_id))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)
                
                seg2 = self.network.get_segment(seg2_id)
                if seg2 is None:
                    continue
                
                if exclude_connected:
                    if (seg1.start_node_id == seg2.start_node_id or
                        seg1.start_node_id == seg2.end_node_id or
                        seg1.end_node_id == seg2.start_node_id or
                        seg1.end_node_id == seg2.end_node_id):
                        continue
                
                centerline_dist = self._segment_to_segment_distance(seg1, seg2)
                
                r2 = seg2.geometry.mean_radius()
                
                # P0-1: Return clearance (surface-to-surface distance), not centerline distance
                clearance = centerline_dist - r1 - r2
                
                if clearance < min_clearance:
                    collisions.append((seg_id1, seg2_id, clearance))
        
        return collisions
    
    def _segment_to_segment_distance(self, seg1: VesselSegment, seg2: VesselSegment) -> float:
        """
        Compute exact minimum distance between two segment centerlines.
        
        P0-3: If either segment is a polyline (has centerline_points), compute
        min distance across all subsegment pairs for correct collision detection
        of curved routes.
        
        Uses analytic formula for true minimum distance between 3D line segments,
        handling all cases including parallel and skew segments.
        """
        start1 = self.network.get_node(seg1.start_node_id)
        end1 = self.network.get_node(seg1.end_node_id)
        start2 = self.network.get_node(seg2.start_node_id)
        end2 = self.network.get_node(seg2.end_node_id)
        
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
        
        # P0-3: Compute min distance across all subsegment pairs
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
