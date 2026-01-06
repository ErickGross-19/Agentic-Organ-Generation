"""
Coverage analysis for tissue perfusion.
"""

from typing import Dict, List, Optional
import numpy as np
from ..core.types import Point3D
from ..core.network import VascularNetwork


def compute_coverage(
    network: VascularNetwork,
    tissue_points: np.ndarray,
    diffusion_distance: float = 0.005,
    vessel_type: Optional[str] = None,
) -> Dict:
    """
    Compute tissue coverage by vascular network.
    
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
    
    Returns
    -------
    report : dict
        Coverage report with:
        - fraction_covered: fraction of tissue points within diffusion distance
        - covered_points: indices of covered points
        - uncovered_points: indices of uncovered points
        - nearest_nodes: for uncovered points, nearest node IDs
        - coverage_distances: distances to nearest vessel for each point
    """
    n_points = len(tissue_points)
    covered = np.zeros(n_points, dtype=bool)
    coverage_distances = np.full(n_points, float('inf'))
    nearest_nodes = np.full(n_points, -1, dtype=int)
    
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
        
        if min_dist <= diffusion_distance:
            covered[i] = True
    
    covered_indices = np.where(covered)[0].tolist()
    uncovered_indices = np.where(~covered)[0].tolist()
    
    fraction_covered = np.sum(covered) / n_points if n_points > 0 else 0.0
    
    uncovered_regions = []
    if uncovered_indices:
        uncovered_points_arr = tissue_points[uncovered_indices]
        
        if len(uncovered_points_arr) > 0:
            centroid = np.mean(uncovered_points_arr, axis=0)
            uncovered_regions.append({
                "centroid": centroid.tolist(),
                "point_count": len(uncovered_indices),
                "nearest_node": int(nearest_nodes[uncovered_indices[0]]),
            })
    
    return {
        "fraction_covered": fraction_covered,
        "total_points": n_points,
        "covered_count": len(covered_indices),
        "uncovered_count": len(uncovered_indices),
        "covered_points": covered_indices,
        "uncovered_points": uncovered_indices,
        "nearest_nodes": nearest_nodes.tolist(),
        "coverage_distances": coverage_distances.tolist(),
        "uncovered_regions": uncovered_regions,
        "mean_coverage_distance": float(np.mean(coverage_distances)),
        "max_coverage_distance": float(np.max(coverage_distances)),
    }
