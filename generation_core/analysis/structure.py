"""Structural analysis functions for vascular networks."""

from typing import Dict
import numpy as np
from ..core.network import VascularNetwork


def compute_branch_stats(network: VascularNetwork) -> Dict:
    """
    Compute comprehensive branch statistics.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to analyze
        
    Returns
    -------
    stats : dict
        Dictionary with degree_histogram, branching_angles, path_lengths, etc.
    """
    degree_histogram = {}
    for node in network.nodes.values():
        connected_seg_ids = network.get_connected_segment_ids(node.id)
        degree = len(connected_seg_ids)
        degree_histogram[degree] = degree_histogram.get(degree, 0) + 1
    
    branching_angles = []
    for node in network.nodes.values():
        connected_seg_ids = network.get_connected_segment_ids(node.id)
        if len(connected_seg_ids) >= 2:
            seg_ids = list(connected_seg_ids)
            for i in range(len(seg_ids)):
                for j in range(i + 1, len(seg_ids)):
                    seg1 = network.segments.get(seg_ids[i])
                    seg2 = network.segments.get(seg_ids[j])
                    if seg1 and seg2:
                        dir1 = seg1.direction.to_array()
                        dir2 = seg2.direction.to_array()
                        cos_angle = np.clip(np.dot(dir1, dir2), -1.0, 1.0)
                        angle = np.degrees(np.arccos(abs(cos_angle)))
                        branching_angles.append(angle)
    
    stats = {
        'degree_histogram': degree_histogram,
        'num_bifurcations': degree_histogram.get(3, 0),
        'branching_angle_distribution': branching_angles,
        'mean_branching_angle': float(np.mean(branching_angles)) if branching_angles else 0.0,
    }
    
    return stats
