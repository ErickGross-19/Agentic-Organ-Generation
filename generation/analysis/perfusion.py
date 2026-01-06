"""
Perfusion analysis for dual-tree vascular networks.

Analyzes tissue perfusion based on proximity to both arterial and venous vessels.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from ..core.types import Point3D
from ..core.network import VascularNetwork


def compute_perfusion_metrics(
    network: VascularNetwork,
    tissue_points: np.ndarray,
    weights: Tuple[float, float] = (1.0, 1.0),
    distance_cap: Optional[float] = None,
) -> Dict:
    """
    Compute tissue perfusion metrics for dual-tree networks.
    
    Analyzes perfusion based on proximity to both arterial (supply) and
    venous (drainage) vessels. Good perfusion requires both nearby arterial
    supply and venous drainage.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to analyze (should contain both arterial and venous vessels)
    tissue_points : np.ndarray
        Array of tissue points (N, 3) to analyze
    weights : tuple of float
        Weights for (arterial, venous) distances in perfusion score
        Default (1.0, 1.0) weights them equally
    distance_cap : float, optional
        Maximum distance to consider for perfusion (meters)
        Points beyond this distance are considered unperfused
    
    Returns
    -------
    metrics : dict
        Perfusion metrics including:
        - perfusion_scores: array of perfusion scores for each point
        - arterial_distances: distances to nearest arterial vessel
        - venous_distances: distances to nearest venous vessel
        - well_perfused_fraction: fraction with good perfusion
        - under_perfused_regions: list of under-perfused region centroids
        - perfusion_stats: mean, min, max perfusion scores
    
    Notes
    -----
    Perfusion score is computed as:
        score = 1 / (w_a * d_a + w_v * d_v + epsilon)
    where d_a and d_v are distances to nearest arterial and venous vessels.
    Higher scores indicate better perfusion.
    """
    n_points = len(tissue_points)
    
    arterial_nodes = [n for n in network.nodes.values() if n.vessel_type == "arterial"]
    venous_nodes = [n for n in network.nodes.values() if n.vessel_type == "venous"]
    
    if not arterial_nodes and not venous_nodes:
        all_nodes = list(network.nodes.values())
        arterial_nodes = all_nodes
        venous_nodes = all_nodes
    
    arterial_distances = np.full(n_points, float('inf'))
    venous_distances = np.full(n_points, float('inf'))
    nearest_arterial = np.full(n_points, -1, dtype=int)
    nearest_venous = np.full(n_points, -1, dtype=int)
    
    for i, tp_arr in enumerate(tissue_points):
        tp = Point3D.from_array(tp_arr)
        
        if arterial_nodes:
            for node in arterial_nodes:
                dist = node.position.distance_to(tp)
                if dist < arterial_distances[i]:
                    arterial_distances[i] = dist
                    nearest_arterial[i] = node.id
        
        if venous_nodes:
            for node in venous_nodes:
                dist = node.position.distance_to(tp)
                if dist < venous_distances[i]:
                    venous_distances[i] = dist
                    nearest_venous[i] = node.id
    
    if distance_cap is not None:
        arterial_distances = np.minimum(arterial_distances, distance_cap)
        venous_distances = np.minimum(venous_distances, distance_cap)
    
    w_a, w_v = weights
    epsilon = 1e-6
    
    perfusion_scores = 1.0 / (w_a * arterial_distances + w_v * venous_distances + epsilon)
    
    if perfusion_scores.max() > 0:
        perfusion_scores = perfusion_scores / perfusion_scores.max()
    
    well_perfused_threshold = 0.5
    well_perfused_mask = perfusion_scores >= well_perfused_threshold
    well_perfused_fraction = np.sum(well_perfused_mask) / n_points if n_points > 0 else 0.0
    
    under_perfused_mask = ~well_perfused_mask
    under_perfused_indices = np.where(under_perfused_mask)[0]
    
    under_perfused_regions = []
    if len(under_perfused_indices) > 0:
        under_perfused_points = tissue_points[under_perfused_indices]
        centroid = np.mean(under_perfused_points, axis=0)
        
        under_perfused_regions.append({
            "centroid": centroid.tolist(),
            "point_count": len(under_perfused_indices),
            "mean_perfusion_score": float(np.mean(perfusion_scores[under_perfused_indices])),
            "nearest_arterial": int(nearest_arterial[under_perfused_indices[0]]),
            "nearest_venous": int(nearest_venous[under_perfused_indices[0]]),
        })
    
    perfusion_stats = {
        "mean": float(np.mean(perfusion_scores)),
        "median": float(np.median(perfusion_scores)),
        "min": float(np.min(perfusion_scores)),
        "max": float(np.max(perfusion_scores)),
        "std": float(np.std(perfusion_scores)),
    }
    
    distance_stats = {
        "arterial": {
            "mean": float(np.mean(arterial_distances)),
            "median": float(np.median(arterial_distances)),
            "max": float(np.max(arterial_distances)),
        },
        "venous": {
            "mean": float(np.mean(venous_distances)),
            "median": float(np.median(venous_distances)),
            "max": float(np.max(venous_distances)),
        },
    }
    
    return {
        "perfusion_scores": perfusion_scores.tolist(),
        "arterial_distances": arterial_distances.tolist(),
        "venous_distances": venous_distances.tolist(),
        "nearest_arterial": nearest_arterial.tolist(),
        "nearest_venous": nearest_venous.tolist(),
        "well_perfused_fraction": well_perfused_fraction,
        "well_perfused_threshold": well_perfused_threshold,
        "under_perfused_regions": under_perfused_regions,
        "perfusion_stats": perfusion_stats,
        "distance_stats": distance_stats,
        "total_points": n_points,
        "arterial_node_count": len(arterial_nodes),
        "venous_node_count": len(venous_nodes),
    }


def suggest_anastomosis_locations(
    network: VascularNetwork,
    perfusion_metrics: Dict,
    rules: Optional = None,
    k: int = 5,
) -> List[Dict]:
    """
    Suggest anastomosis locations based on perfusion analysis.
    
    Identifies arterial-venous node pairs that would improve tissue perfusion
    if connected via anastomosis.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to analyze
    perfusion_metrics : dict
        Output from compute_perfusion_metrics
    rules : InteractionRuleSpec, optional
        Interaction rules for anastomosis validation
    k : int
        Number of top candidates to return
    
    Returns
    -------
    candidates : list of dict
        List of anastomosis candidates, each containing:
        - arterial_node: arterial node ID
        - venous_node: venous node ID
        - distance: distance between nodes
        - score: candidate quality score (higher is better)
        - reason: explanation for suggestion
    """
    from ..rules.constraints import InteractionRuleSpec
    
    if rules is None:
        rules = InteractionRuleSpec()
    
    under_perfused_regions = perfusion_metrics.get("under_perfused_regions", [])
    
    if not under_perfused_regions:
        return []
    
    candidates = []
    
    for region in under_perfused_regions:
        centroid = np.array(region["centroid"])
        centroid_point = Point3D.from_array(centroid)
        
        arterial_nodes = [n for n in network.nodes.values() if n.vessel_type == "arterial"]
        venous_nodes = [n for n in network.nodes.values() if n.vessel_type == "venous"]
        
        arterial_distances = [(n, n.position.distance_to(centroid_point)) for n in arterial_nodes]
        venous_distances = [(n, n.position.distance_to(centroid_point)) for n in venous_nodes]
        
        arterial_distances.sort(key=lambda x: x[1])
        venous_distances.sort(key=lambda x: x[1])
        
        for a_node, a_dist in arterial_distances[:3]:
            for v_node, v_dist in venous_distances[:3]:
                node_distance = a_node.position.distance_to(v_node.position)
                
                if not rules.is_anastomosis_allowed(a_node.vessel_type, v_node.vessel_type):
                    continue
                
                a_radius = a_node.attributes.get("radius", 0.0005)
                v_radius = v_node.attributes.get("radius", 0.0005)
                
                is_terminal = (a_node.node_type == "terminal") and (v_node.node_type == "terminal")
                is_small = (a_radius < 0.001) and (v_radius < 0.001)
                
                proximity_score = 1.0 / (a_dist + v_dist + 0.001)
                connection_score = 1.0 / (node_distance + 0.001)
                terminal_bonus = 2.0 if is_terminal else 1.0
                small_bonus = 1.5 if is_small else 1.0
                
                score = proximity_score * connection_score * terminal_bonus * small_bonus
                
                candidates.append({
                    "arterial_node": a_node.id,
                    "venous_node": v_node.id,
                    "distance": node_distance,
                    "arterial_radius": a_radius,
                    "venous_radius": v_radius,
                    "score": score,
                    "reason": f"Would improve perfusion near under-perfused region (centroid: {centroid.tolist()})",
                    "region_centroid": centroid.tolist(),
                })
    
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    seen = set()
    unique_candidates = []
    for c in candidates:
        pair = tuple(sorted([c["arterial_node"], c["venous_node"]]))
        if pair not in seen:
            seen.add(pair)
            unique_candidates.append(c)
    
    return unique_candidates[:k]
