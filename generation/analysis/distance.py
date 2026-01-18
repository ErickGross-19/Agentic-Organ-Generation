"""
Centralized distance calculations for vascular network analysis.

This module provides segment-based distance calculations that account for:
- Polyline centerlines (TubeGeometry.centerline_points)
- Vessel radius (distance to surface, not just centerline)
- Fast nearest-segment queries using SpatialIndex

Note: The library uses METERS internally for all geometry.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from ..core.types import Point3D
from ..core.network import VascularNetwork, VesselSegment


def point_to_polyline_distance(
    point: np.ndarray,
    polyline_points: List[np.ndarray],
) -> Tuple[float, float, int]:
    """
    Compute minimum distance from a point to a polyline.
    
    Parameters
    ----------
    point : np.ndarray
        Query point (x, y, z)
    polyline_points : list of np.ndarray
        List of polyline vertices
        
    Returns
    -------
    distance : float
        Minimum distance from point to polyline
    t_param : float
        Parameter along polyline (0 to 1) of closest point
    segment_idx : int
        Index of the polyline segment containing closest point
    """
    if len(polyline_points) < 2:
        if len(polyline_points) == 1:
            return float(np.linalg.norm(point - polyline_points[0])), 0.0, 0
        return float('inf'), 0.0, 0
    
    min_dist = float('inf')
    best_t = 0.0
    best_seg_idx = 0
    
    total_length = 0.0
    segment_lengths = []
    for i in range(len(polyline_points) - 1):
        seg_len = np.linalg.norm(polyline_points[i + 1] - polyline_points[i])
        segment_lengths.append(seg_len)
        total_length += seg_len
    
    cumulative_length = 0.0
    
    for i in range(len(polyline_points) - 1):
        p1 = polyline_points[i]
        p2 = polyline_points[i + 1]
        
        dist, local_t = _point_to_segment_distance_with_param(point, p1, p2)
        
        if dist < min_dist:
            min_dist = dist
            best_seg_idx = i
            
            if total_length > 0:
                best_t = (cumulative_length + local_t * segment_lengths[i]) / total_length
            else:
                best_t = 0.0
        
        cumulative_length += segment_lengths[i]
    
    return min_dist, best_t, best_seg_idx


def _point_to_segment_distance_with_param(
    point: np.ndarray,
    seg_start: np.ndarray,
    seg_end: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute distance from point to line segment with parameter.
    
    Returns
    -------
    distance : float
        Distance to segment
    t : float
        Parameter (0 to 1) along segment of closest point
    """
    v = seg_end - seg_start
    length_sq = np.dot(v, v)
    
    if length_sq < 1e-20:
        return float(np.linalg.norm(point - seg_start)), 0.0
    
    t = np.dot(point - seg_start, v) / length_sq
    t = np.clip(t, 0.0, 1.0)
    
    closest = seg_start + t * v
    distance = float(np.linalg.norm(point - closest))
    
    return distance, t


def point_to_vessel_surface_distance(
    point: np.ndarray,
    segment: VesselSegment,
    network: VascularNetwork,
) -> Tuple[float, float]:
    """
    Compute distance from point to vessel surface (accounting for radius).
    
    Distance to surface = distance to centerline - local radius
    
    Parameters
    ----------
    point : np.ndarray
        Query point (x, y, z)
    segment : VesselSegment
        Vessel segment
    network : VascularNetwork
        Network containing the segment (for node positions)
        
    Returns
    -------
    surface_distance : float
        Distance to vessel surface (negative if inside vessel)
    centerline_distance : float
        Distance to centerline
    """
    start_node = network.nodes[segment.start_node_id]
    end_node = network.nodes[segment.end_node_id]
    
    if segment.geometry.centerline_points:
        polyline = [start_node.position.to_array()]
        polyline.extend([p.to_array() for p in segment.geometry.centerline_points])
        polyline.append(end_node.position.to_array())
        
        centerline_dist, t_param, _ = point_to_polyline_distance(point, polyline)
    else:
        p1 = start_node.position.to_array()
        p2 = end_node.position.to_array()
        centerline_dist, t_param = _point_to_segment_distance_with_param(point, p1, p2)
    
    local_radius = (
        segment.geometry.radius_start * (1 - t_param) +
        segment.geometry.radius_end * t_param
    )
    
    surface_distance = centerline_dist - local_radius
    
    return surface_distance, centerline_dist


def compute_nearest_segment_distance(
    point: Point3D,
    network: VascularNetwork,
    vessel_type: Optional[str] = None,
    use_surface_distance: bool = True,
    search_radius_hint: Optional[float] = None,
    use_brute_force: bool = False,
) -> Tuple[float, int, float]:
    """
    Find the nearest segment to a point and compute distance.
    
    Uses SpatialIndex for efficient queries with adaptive radius expansion.
    Falls back to brute-force only when use_brute_force=True (for debugging).
    
    Parameters
    ----------
    point : Point3D
        Query point
    network : VascularNetwork
        Network to search
    vessel_type : str, optional
        Filter by vessel type ("arterial", "venous", etc.)
    use_surface_distance : bool
        If True, compute distance to vessel surface (accounting for radius)
        If False, compute distance to centerline only
    search_radius_hint : float, optional
        Initial search radius for SpatialIndex query. If None, uses adaptive
        strategy starting with typical vessel radius or cell size.
    use_brute_force : bool
        If True, skip SpatialIndex and check all segments (for debugging/validation)
        
    Returns
    -------
    distance : float
        Distance to nearest segment (surface or centerline)
    segment_id : int
        ID of nearest segment
    centerline_distance : float
        Distance to centerline (same as distance if use_surface_distance=False)
    """
    point_arr = point.to_array()
    
    # Use SpatialIndex with adaptive radius unless brute_force is requested
    if not use_brute_force and len(network.segments) > 0:
        return _compute_nearest_segment_with_spatial_index(
            point, point_arr, network, vessel_type, use_surface_distance, search_radius_hint
        )
    
    # Brute-force fallback (for debugging or empty networks)
    return _compute_nearest_segment_brute_force(
        point_arr, network, vessel_type, use_surface_distance
    )


def _compute_nearest_segment_with_spatial_index(
    point: Point3D,
    point_arr: np.ndarray,
    network: VascularNetwork,
    vessel_type: Optional[str],
    use_surface_distance: bool,
    search_radius_hint: Optional[float],
) -> Tuple[float, int, float]:
    """
    Find nearest segment using SpatialIndex with adaptive radius expansion.
    
    Strategy:
    1. Start with search_radius_hint or estimate from typical vessel radius
    2. Query SpatialIndex for nearby segments
    3. If no segments found, expand radius geometrically (2x) up to domain diagonal
    4. Compute exact distance only for candidates from spatial query
    """
    spatial_index = network.get_spatial_index()
    
    # Estimate domain size for max search radius
    all_positions = [n.position.to_array() for n in network.nodes.values()]
    if not all_positions:
        return float('inf'), -1, float('inf')
    
    positions_arr = np.array(all_positions)
    domain_min = np.min(positions_arr, axis=0)
    domain_max = np.max(positions_arr, axis=0)
    domain_diag = float(np.linalg.norm(domain_max - domain_min))
    
    # Estimate typical vessel radius for initial search from network data
    typical_radius = None
    if network.segments:
        radii = []
        for seg in list(network.segments.values())[:100]:  # Sample first 100
            radii.append(seg.geometry.mean_radius())
        typical_radius = float(np.median(radii)) if radii else None
    
    # If no radius data available, derive from domain extent
    if typical_radius is None:
        typical_radius = domain_diag / 1000.0 if domain_diag > 0 else 1e-6
    
    # Initial search radius
    if search_radius_hint is not None:
        initial_radius = search_radius_hint
    else:
        # Start with 10x typical radius or spatial index cell size
        initial_radius = max(typical_radius * 10, spatial_index.cell_size * 2)
    
    # Adaptive radius expansion
    search_radius = initial_radius
    max_radius = domain_diag * 1.5  # Allow slightly beyond domain
    expansion_factor = 2.0
    
    min_distance = float('inf')
    nearest_seg_id = -1
    nearest_centerline_dist = float('inf')
    
    while search_radius <= max_radius:
        # Query spatial index
        candidate_segments = spatial_index.query_nearby_segments(point, search_radius)
        
        # Filter by vessel type if specified
        if vessel_type is not None:
            candidate_segments = [s for s in candidate_segments if s.vessel_type == vessel_type]
        
        if candidate_segments:
            # Compute exact distances for candidates
            for segment in candidate_segments:
                if use_surface_distance:
                    surface_dist, centerline_dist = point_to_vessel_surface_distance(
                        point_arr, segment, network
                    )
                    dist = surface_dist
                else:
                    start_node = network.nodes[segment.start_node_id]
                    end_node = network.nodes[segment.end_node_id]
                    
                    if segment.geometry.centerline_points:
                        polyline = [start_node.position.to_array()]
                        polyline.extend([p.to_array() for p in segment.geometry.centerline_points])
                        polyline.append(end_node.position.to_array())
                        dist, _, _ = point_to_polyline_distance(point_arr, polyline)
                    else:
                        p1 = start_node.position.to_array()
                        p2 = end_node.position.to_array()
                        dist, _ = _point_to_segment_distance_with_param(point_arr, p1, p2)
                    
                    centerline_dist = dist
                
                if dist < min_distance:
                    min_distance = dist
                    nearest_seg_id = segment.id
                    nearest_centerline_dist = centerline_dist if use_surface_distance else dist
            
            # If we found segments and the best distance is within our search radius,
            # we can be confident we found the true nearest (no need to expand further)
            if min_distance < search_radius:
                break
        
        # Expand search radius
        search_radius *= expansion_factor
    
    return min_distance, nearest_seg_id, nearest_centerline_dist


def _compute_nearest_segment_brute_force(
    point_arr: np.ndarray,
    network: VascularNetwork,
    vessel_type: Optional[str],
    use_surface_distance: bool,
) -> Tuple[float, int, float]:
    """
    Brute-force nearest segment search (O(N) per point).
    
    Used as debug fallback to validate SpatialIndex results.
    """
    min_distance = float('inf')
    nearest_seg_id = -1
    nearest_centerline_dist = float('inf')
    
    segments_to_check = network.segments.values()
    if vessel_type is not None:
        segments_to_check = [s for s in segments_to_check if s.vessel_type == vessel_type]
    
    for segment in segments_to_check:
        if use_surface_distance:
            surface_dist, centerline_dist = point_to_vessel_surface_distance(
                point_arr, segment, network
            )
            dist = surface_dist
        else:
            start_node = network.nodes[segment.start_node_id]
            end_node = network.nodes[segment.end_node_id]
            
            if segment.geometry.centerline_points:
                polyline = [start_node.position.to_array()]
                polyline.extend([p.to_array() for p in segment.geometry.centerline_points])
                polyline.append(end_node.position.to_array())
                dist, _, _ = point_to_polyline_distance(point_arr, polyline)
            else:
                p1 = start_node.position.to_array()
                p2 = end_node.position.to_array()
                dist, _ = _point_to_segment_distance_with_param(point_arr, p1, p2)
            
            centerline_dist = dist
        
        if dist < min_distance:
            min_distance = dist
            nearest_seg_id = segment.id
            nearest_centerline_dist = centerline_dist if use_surface_distance else dist
    
    return min_distance, nearest_seg_id, nearest_centerline_dist


def compute_tissue_coverage_distances(
    tissue_points: np.ndarray,
    network: VascularNetwork,
    vessel_type: Optional[str] = None,
    use_surface_distance: bool = True,
    search_radius_hint: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute distances from tissue points to nearest vessel segments.
    
    This is the segment-based replacement for node-based coverage metrics.
    Uses SpatialIndex for efficient O(N log N) queries instead of O(N²) brute-force.
    
    Parameters
    ----------
    tissue_points : np.ndarray
        Array of tissue points (N, 3)
    network : VascularNetwork
        Vascular network
    vessel_type : str, optional
        Filter by vessel type
    use_surface_distance : bool
        If True, compute distance to vessel surface
    search_radius_hint : float, optional
        Initial search radius for SpatialIndex queries. If None, uses adaptive
        strategy. For perfusion analysis, consider using perfusion_threshold.
        
    Returns
    -------
    result : dict
        Dictionary containing:
        - distances: array of distances for each tissue point
        - nearest_segments: array of nearest segment IDs
        - centerline_distances: array of centerline distances
        - mean_distance: mean distance
        - max_distance: maximum distance
        - min_distance: minimum distance
    """
    n_points = len(tissue_points)
    distances = np.full(n_points, float('inf'))
    nearest_segments = np.full(n_points, -1, dtype=int)
    centerline_distances = np.full(n_points, float('inf'))
    
    for i, tp in enumerate(tissue_points):
        point = Point3D.from_array(tp)
        dist, seg_id, cl_dist = compute_nearest_segment_distance(
            point, network, vessel_type, use_surface_distance,
            search_radius_hint=search_radius_hint,
        )
        distances[i] = dist
        nearest_segments[i] = seg_id
        centerline_distances[i] = cl_dist
    
    return {
        "distances": distances,
        "nearest_segments": nearest_segments,
        "centerline_distances": centerline_distances,
        "mean_distance": float(np.mean(distances)) if n_points > 0 else 0.0,
        "max_distance": float(np.max(distances)) if n_points > 0 else 0.0,
        "min_distance": float(np.min(distances)) if n_points > 0 else 0.0,
    }


def compute_perfusion_distances(
    tissue_points: np.ndarray,
    network: VascularNetwork,
    use_surface_distance: bool = True,
    search_radius_hint: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute distances to both arterial and venous vessels for perfusion analysis.
    
    Uses SpatialIndex for efficient O(N log N) queries instead of O(N²) brute-force.
    
    Parameters
    ----------
    tissue_points : np.ndarray
        Array of tissue points (N, 3)
    network : VascularNetwork
        Vascular network with arterial and venous vessels
    use_surface_distance : bool
        If True, compute distance to vessel surface
    search_radius_hint : float, optional
        Initial search radius for SpatialIndex queries. If None, uses adaptive
        strategy. Consider using perfusion_threshold from EvalConfig.
        
    Returns
    -------
    result : dict
        Dictionary containing:
        - arterial_distances: distances to nearest arterial segment
        - venous_distances: distances to nearest venous segment
        - nearest_arterial_segments: IDs of nearest arterial segments
        - nearest_venous_segments: IDs of nearest venous segments
        - arterial_stats: statistics for arterial distances
        - venous_stats: statistics for venous distances
    """
    arterial_result = compute_tissue_coverage_distances(
        tissue_points, network, vessel_type="arterial",
        use_surface_distance=use_surface_distance,
        search_radius_hint=search_radius_hint,
    )
    
    venous_result = compute_tissue_coverage_distances(
        tissue_points, network, vessel_type="venous",
        use_surface_distance=use_surface_distance,
        search_radius_hint=search_radius_hint,
    )
    
    return {
        "arterial_distances": arterial_result["distances"],
        "venous_distances": venous_result["distances"],
        "nearest_arterial_segments": arterial_result["nearest_segments"],
        "nearest_venous_segments": venous_result["nearest_segments"],
        "arterial_stats": {
            "mean": arterial_result["mean_distance"],
            "max": arterial_result["max_distance"],
            "min": arterial_result["min_distance"],
        },
        "venous_stats": {
            "mean": venous_result["mean_distance"],
            "max": venous_result["max_distance"],
            "min": venous_result["min_distance"],
        },
    }


def find_underperfused_regions(
    tissue_points: np.ndarray,
    perfusion_scores: np.ndarray,
    threshold: float = 0.5,
    n_clusters: int = 5,
) -> List[Dict[str, Any]]:
    """
    Identify clusters of under-perfused tissue points.
    
    Uses simple spatial binning to identify multiple under-perfused regions
    instead of just computing a single centroid.
    
    Parameters
    ----------
    tissue_points : np.ndarray
        Array of tissue points (N, 3)
    perfusion_scores : np.ndarray
        Perfusion scores for each point (0-1, higher is better)
    threshold : float
        Points with score below this are considered under-perfused
    n_clusters : int
        Maximum number of clusters to identify
        
    Returns
    -------
    regions : list of dict
        List of under-perfused regions, each containing:
        - centroid: center of the region
        - point_count: number of points in region
        - mean_score: mean perfusion score in region
        - point_indices: indices of points in this region
    """
    under_perfused_mask = perfusion_scores < threshold
    under_perfused_indices = np.where(under_perfused_mask)[0]
    
    if len(under_perfused_indices) == 0:
        return []
    
    under_perfused_points = tissue_points[under_perfused_indices]
    under_perfused_scores = perfusion_scores[under_perfused_indices]
    
    if len(under_perfused_points) <= n_clusters:
        regions = []
        for i, idx in enumerate(under_perfused_indices):
            regions.append({
                "centroid": tissue_points[idx].tolist(),
                "point_count": 1,
                "mean_score": float(perfusion_scores[idx]),
                "point_indices": [int(idx)],
            })
        return regions
    
    min_coords = np.min(under_perfused_points, axis=0)
    max_coords = np.max(under_perfused_points, axis=0)
    
    n_bins_per_dim = max(2, int(np.ceil(n_clusters ** (1/3))))
    
    bin_size = (max_coords - min_coords) / n_bins_per_dim
    bin_size = np.maximum(bin_size, 1e-10)
    
    bins: Dict[Tuple[int, int, int], List[int]] = {}
    
    for i, point in enumerate(under_perfused_points):
        bin_idx = tuple(
            min(n_bins_per_dim - 1, int((point[d] - min_coords[d]) / bin_size[d]))
            for d in range(3)
        )
        if bin_idx not in bins:
            bins[bin_idx] = []
        bins[bin_idx].append(i)
    
    bin_scores = []
    for bin_idx, point_indices in bins.items():
        mean_score = np.mean(under_perfused_scores[point_indices])
        bin_scores.append((bin_idx, mean_score, len(point_indices)))
    
    bin_scores.sort(key=lambda x: x[1])
    
    regions = []
    for bin_idx, mean_score, count in bin_scores[:n_clusters]:
        local_indices = bins[bin_idx]
        global_indices = under_perfused_indices[local_indices]
        
        centroid = np.mean(under_perfused_points[local_indices], axis=0)
        
        regions.append({
            "centroid": centroid.tolist(),
            "point_count": count,
            "mean_score": float(mean_score),
            "point_indices": [int(idx) for idx in global_indices],
        })
    
    return regions
