"""
Compute node junction metrics for visualization and analysis.
"""

import numpy as np
import networkx as nx
from typing import Dict, Optional


def compute_node_junction_metrics(G: nx.Graph) -> Dict[int, dict]:
    """
    Compute junction metrics for each node in the centerline graph.
    
    For each node, computes:
    - degree: number of incident edges
    - incident_radii: list of radii at the node end of each incident segment
    - effective_radius_murray3: Murray's law aggregation (sum(r^3))^(1/3)
    - mean_radius: average of incident radii
    - max_radius: maximum of incident radii
    - node_type: inferred type (junction, terminal, inlet, outlet)
    
    Parameters
    ----------
    G : networkx.Graph
        Centerline graph with node attribute 'radius' and optionally 'coord'
        
    Returns
    -------
    metrics : dict
        Dictionary keyed by node ID with junction metrics
    """
    metrics = {}
    
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        
        incident_edges = list(G.edges(node_id))
        degree = len(incident_edges)
        
        incident_radii = []
        for u, v in incident_edges:
            edge_data = G[u][v]
            
            if 'radius' in edge_data:
                r = float(edge_data['radius'])
            else:
                if u == node_id:
                    r = float(G.nodes[u].get('radius', 0.0))
                else:
                    r = float(G.nodes[v].get('radius', 0.0))
            
            if r > 0:
                incident_radii.append(r)
        
        if not incident_radii:
            node_radius = node_data.get('radius', 0.0)
            if node_radius > 0:
                incident_radii = [float(node_radius)]
        
        if incident_radii:
            effective_radius_murray3 = float(np.power(np.sum(np.power(incident_radii, 3)), 1.0/3.0))
            mean_radius = float(np.mean(incident_radii))
            max_radius = float(np.max(incident_radii))
        else:
            effective_radius_murray3 = 0.0
            mean_radius = 0.0
            max_radius = 0.0
        
        if degree == 0:
            node_type = 'isolated'
        elif degree == 1:
            node_type = 'terminal'
        elif degree == 2:
            node_type = 'pass-through'
        else:
            node_type = 'junction'
        
        if 'node_type' in node_data:
            node_type = node_data['node_type']
        
        metrics[node_id] = {
            'degree': degree,
            'incident_radii': incident_radii,
            'effective_radius_murray3': effective_radius_murray3,
            'mean_radius': mean_radius,
            'max_radius': max_radius,
            'node_type': node_type,
        }
    
    return metrics


def compute_node_display_sizes(
    G: nx.Graph,
    size_by: str = 'junction',
    base_px: float = 3.0,
    radius_scale: float = 2000.0,
    degree_scale: float = 1.0,
    degree_alpha: float = 0.3,
    min_px: Optional[float] = None,
    max_px: Optional[float] = None,
    inlet_outlet_boost: float = 1.3,
    k_px2: float = 0.1,
    radius_power: float = 1.5,
) -> Dict[int, float]:
    """
    Compute display sizes (in matplotlib scatter points^2) for each node.
    
    Parameters
    ----------
    G : networkx.Graph
        Centerline graph
    size_by : str
        Sizing method: 'radius_tiny' (radius-proportional, almost invisible), 
        'junction' (Murray + degree), 'max_radius', 'mean_radius', 'degree'
    base_px : float
        Base size in points^2
    radius_scale : float
        Scaling factor for radius (radii are in meters, need to scale to reasonable pixel sizes)
    degree_scale : float
        Scaling factor for degree contribution
    degree_alpha : float
        Exponent for degree scaling (degree^alpha)
    min_px : float
        Minimum node size
    max_px : float
        Maximum node size
    inlet_outlet_boost : float
        Multiplier for inlet/outlet nodes to make them stand out
    k_px2 : float
        Scaling factor for 'radius_tiny' mode (default: 0.1 for almost invisible)
    radius_power : float
        Exponent for radius scaling in 'radius_tiny' mode (default: 1.5)
        
    Returns
    -------
    sizes : dict
        Dictionary keyed by node ID with display size in points^2
    """
    metrics = compute_node_junction_metrics(G)
    sizes = {}
    
    if min_px is None:
        min_px = 0.02 if size_by == 'radius_tiny' else 2.0
    if max_px is None:
        max_px = 0.6 if size_by == 'radius_tiny' else 50.0
    
    if size_by == 'radius_tiny':
        eff_radii = []
        for m in metrics.values():
            eff = m['effective_radius_murray3']
            if eff > 0:
                eff_radii.append(eff)
        
        if eff_radii:
            r_ref = float(np.median(eff_radii))
        else:
            r_ref = 1.0
        
        for node_id, m in metrics.items():
            eff = m['effective_radius_murray3']
            if eff > 0 and r_ref > 0:
                size = k_px2 * ((eff / r_ref) ** radius_power)
            else:
                size = min_px
            
            size = np.clip(size, min_px, max_px)
            sizes[node_id] = float(size)
    else:
        for node_id, m in metrics.items():
            if size_by == 'junction':
                eff = m['effective_radius_murray3']
                deg = m['degree']
                size = base_px + radius_scale * eff + degree_scale * (deg ** degree_alpha)
            elif size_by == 'max_radius':
                size = base_px + radius_scale * m['max_radius']
            elif size_by == 'mean_radius':
                size = base_px + radius_scale * m['mean_radius']
            elif size_by == 'degree':
                deg = m['degree']
                size = base_px + degree_scale * (deg ** degree_alpha)
            else:
                size = base_px
            
            node_type = m['node_type']
            if node_type in ['inlet', 'outlet']:
                size *= inlet_outlet_boost
            elif node_type == 'terminal':
                size = base_px + radius_scale * m['mean_radius'] * 0.5
            
            size = np.clip(size, min_px, max_px)
            
            sizes[node_id] = float(size)
    
    return sizes
