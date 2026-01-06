"""
Adapter for converting between VascularNetwork and NetworkX graphs.

This enables integration with the existing vascular_network package
which uses NetworkX for flow analysis.
"""

import networkx as nx
import numpy as np
from typing import Dict, Tuple, Optional
from ..core.network import VascularNetwork, Node, VesselSegment
from ..core.types import Point3D, TubeGeometry
from ..core.result import OperationResult, OperationStatus


def to_networkx_graph(network: VascularNetwork) -> Tuple[nx.Graph, Dict[int, int]]:
    """
    Convert VascularNetwork to NetworkX graph.
    
    The resulting graph has node attributes:
    - 'coord': [x, y, z] position as list
    - 'radius': float radius
    - 'node_type': str type
    - 'vessel_type': str vessel type
    
    And edge attributes:
    - 'length': float segment length
    - 'radius': float mean radius
    - 'segment_id': int original segment ID
    
    Parameters
    ----------
    network : VascularNetwork
        The vascular network to convert
        
    Returns
    -------
    G : nx.Graph
        NetworkX graph representation
    node_id_map : dict
        Mapping from VascularNetwork node IDs to NetworkX node IDs
    """
    G = nx.Graph()
    node_id_map = {}
    
    for node_id, node in network.nodes.items():
        nx_id = len(node_id_map)
        node_id_map[node_id] = nx_id
        
        G.add_node(
            nx_id,
            coord=[node.position.x, node.position.y, node.position.z],
            radius=node.attributes.get('radius', 0.001),  # Default 1mm if not set
            node_type=node.node_type,
            vessel_type=node.vessel_type,
            original_id=node_id,
        )
    
    for seg_id, segment in network.segments.items():
        start_nx = node_id_map[segment.start_node_id]
        end_nx = node_id_map[segment.end_node_id]
        
        start_pos = np.array([
            segment.geometry.start.x,
            segment.geometry.start.y,
            segment.geometry.start.z,
        ])
        end_pos = np.array([
            segment.geometry.end.x,
            segment.geometry.end.y,
            segment.geometry.end.z,
        ])
        length = float(np.linalg.norm(end_pos - start_pos))
        
        mean_radius = (segment.geometry.radius_start + segment.geometry.radius_end) / 2.0
        
        G.add_edge(
            start_nx,
            end_nx,
            length=length,
            radius=mean_radius,
            segment_id=seg_id,
            vessel_type=segment.vessel_type,
        )
    
    return G, node_id_map


def from_networkx_graph(
    G: nx.Graph,
    domain,
    node_id_map: Optional[Dict[int, int]] = None,
) -> VascularNetwork:
    """
    Convert NetworkX graph to VascularNetwork.
    
    Parameters
    ----------
    G : nx.Graph
        NetworkX graph with node attributes 'coord' and 'radius'
    domain : DomainSpec
        Domain specification for the network
    node_id_map : dict, optional
        Mapping from NetworkX node IDs to VascularNetwork node IDs
        
    Returns
    -------
    VascularNetwork
        Reconstructed vascular network
    """
    from ..core.domain import EllipsoidDomain
    
    network = VascularNetwork(domain=domain)
    
    if node_id_map is None:
        nx_to_vascular = {}
    else:
        nx_to_vascular = {v: k for k, v in node_id_map.items()}
    
    for nx_id in G.nodes():
        node_data = G.nodes[nx_id]
        
        if nx_id in nx_to_vascular:
            vascular_id = nx_to_vascular[nx_id]
        else:
            vascular_id = network.id_gen.next_node_id()
            nx_to_vascular[nx_id] = vascular_id
        
        coord = node_data.get('coord', [0, 0, 0])
        radius = node_data.get('radius', 0.001)
        node_type = node_data.get('node_type', 'junction')
        vessel_type = node_data.get('vessel_type', 'arterial')
        
        node = Node(
            id=vascular_id,
            position=Point3D(x=coord[0], y=coord[1], z=coord[2]),
            node_type=node_type,
            vessel_type=vessel_type,
            attributes={'radius': radius},
        )
        network.add_node(node)
    
    for u, v in G.edges():
        edge_data = G.edges[u, v]
        
        u_vascular = nx_to_vascular[u]
        v_vascular = nx_to_vascular[v]
        
        u_node = network.nodes[u_vascular]
        v_node = network.nodes[v_vascular]
        
        radius = edge_data.get('radius', 0.001)
        
        segment = VesselSegment(
            id=network.id_gen.next_segment_id(),
            start_node_id=u_vascular,
            end_node_id=v_vascular,
            geometry=TubeGeometry(
                start=u_node.position,
                end=v_node.position,
                radius_start=radius,
                radius_end=radius,
            ),
            vessel_type=edge_data.get('vessel_type', 'arterial'),
        )
        network.add_segment(segment)
    
    return network
