"""
Coverage analysis for tissue perfusion.

This module provides segment-based coverage analysis that computes distances
to vessel surfaces (centerline - radius) rather than just to nodes.

Note: The library uses METERS internally for all geometry.
"""

from typing import Dict, Optional
import numpy as np
from ..core.types import Point3D
from ..core.network import VascularNetwork
from .distance import (
    compute_tissue_coverage_distances,
    find_underperfused_regions,
)


def compute_coverage(
    network: VascularNetwork,
    tissue_points: np.ndarray,
    diffusion_distance: float = 0.005,
    vessel_type: Optional[str] = None,
    use_segment_distance: bool = True,
) -> Dict:
    """
    Compute tissue coverage by vascular network using segment-based distances.
    
    By default, uses distance to vessel surface (centerline - radius) rather
    than distance to nodes for more accurate coverage assessment.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to analyze
    tissue_points : np.ndarray
        Array of tissue points (N, 3) to check for coverage
    diffusion_distance : float
        Maximum distance for oxygen/nutrient diffusion (meters)
    vessel_type : str, optional
        Filter by vessel type
    use_segment_distance : bool
        If True (default), use segment-based distance to vessel surface
        If False, use legacy node-based distance
    
    Returns
    -------
    report : dict
        Coverage report with:
        - fraction_covered: fraction of tissue points within diffusion distance
        - covered_points: indices of covered points
        - uncovered_points: indices of uncovered points
        - nearest_segments: for each point, nearest segment ID (if segment-based)
        - nearest_nodes: for each point, nearest node ID (if node-based)
        - coverage_distances: distances to nearest vessel for each point
        - uncovered_regions: list of under-covered region centroids
    """
    n_points = len(tissue_points)
    
    if n_points == 0:
        return {
            "fraction_covered": 0.0,
            "total_points": 0,
            "covered_count": 0,
            "uncovered_count": 0,
            "covered_points": [],
            "uncovered_points": [],
            "nearest_segments": [],
            "nearest_nodes": [],
            "coverage_distances": [],
            "uncovered_regions": [],
            "mean_coverage_distance": 0.0,
            "max_coverage_distance": 0.0,
        }
    
    if use_segment_distance:
        coverage_result = compute_tissue_coverage_distances(
            tissue_points, network, vessel_type=vessel_type, use_surface_distance=True
        )
        coverage_distances = coverage_result["distances"]
        nearest_segments = coverage_result["nearest_segments"]
        nearest_nodes = np.full(n_points, -1, dtype=int)
    else:
        coverage_distances = np.full(n_points, float('inf'))
        nearest_nodes = np.full(n_points, -1, dtype=int)
        nearest_segments = np.full(n_points, -1, dtype=int)
        
        nodes_to_check = [
            node for node in network.nodes.values()
            if vessel_type is None or node.vessel_type == vessel_type
        ]
        
        for i, tp_arr in enumerate(tissue_points):
            tp = Point3D.from_array(tp_arr)
            
            min_dist = float('inf')
            nearest_node_id = -1
            
            for node in nodes_to_check:
                dist = node.position.distance_to(tp)
                if dist < min_dist:
                    min_dist = dist
                    nearest_node_id = node.id
            
            coverage_distances[i] = min_dist
            nearest_nodes[i] = nearest_node_id
    
    covered = coverage_distances <= diffusion_distance
    covered_indices = np.where(covered)[0].tolist()
    uncovered_indices = np.where(~covered)[0].tolist()
    
    fraction_covered = float(np.sum(covered) / n_points)
    
    uncovered_regions = []
    if uncovered_indices:
        uncovered_distances = coverage_distances[uncovered_indices]
        
        normalized_scores = 1.0 - np.clip(uncovered_distances / (diffusion_distance * 2), 0, 1)
        
        regions = find_underperfused_regions(
            tissue_points, 
            np.where(covered, 1.0, normalized_scores),
            threshold=0.5,
            n_clusters=5,
        )
        
        for region in regions:
            if use_segment_distance:
                region["nearest_segment"] = int(nearest_segments[region["point_indices"][0]])
            else:
                region["nearest_node"] = int(nearest_nodes[region["point_indices"][0]])
        
        uncovered_regions = regions
    
    return {
        "fraction_covered": fraction_covered,
        "total_points": n_points,
        "covered_count": len(covered_indices),
        "uncovered_count": len(uncovered_indices),
        "covered_points": covered_indices,
        "uncovered_points": uncovered_indices,
        "nearest_segments": nearest_segments.tolist() if use_segment_distance else [],
        "nearest_nodes": nearest_nodes.tolist(),
        "coverage_distances": coverage_distances.tolist(),
        "uncovered_regions": uncovered_regions,
        "mean_coverage_distance": float(np.mean(coverage_distances)),
        "max_coverage_distance": float(np.max(coverage_distances)),
        "use_segment_distance": use_segment_distance,
    }
