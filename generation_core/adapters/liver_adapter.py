"""
Adapter for converting between liver-specific VascularTree and core VascularNetwork.

This enables interoperability between the liver generator's tree structure
and the generic generation/validation pipeline.
"""

from typing import Optional, Dict, Any
import numpy as np

from ..core.network import VascularNetwork, Node as CoreNode, VesselSegment
from ..core.types import Point3D, TubeGeometry
from ..core.domain import DomainSpec
from ..organ_generators.liver.tree import VascularTree, Node as LiverNode, Segment as LiverSegment


def liver_tree_to_network(
    tree: VascularTree,
    domain: DomainSpec,
    metadata: Optional[Dict[str, Any]] = None,
) -> VascularNetwork:
    """
    Convert a liver VascularTree to a core VascularNetwork.
    
    This adapter maps the liver-specific tree structure (with parent_id,
    children_ids, order) to the generic network structure (with segments
    connecting nodes).
    
    Parameters
    ----------
    tree : VascularTree
        Liver-specific vascular tree
    domain : DomainSpec
        Domain specification for the network
    metadata : dict, optional
        Additional metadata to include
        
    Returns
    -------
    VascularNetwork
        Core network representation
        
    Examples
    --------
    >>> from generation.organ_generators.liver.tree import VascularTree
    >>> from generation.adapters.liver_adapter import liver_tree_to_network
    >>> 
    >>> # Create liver tree
    >>> tree = VascularTree(
    ...     tree_type="arterial",
    ...     root_position=np.array([0, 0, 0.03]),
    ...     root_radius=0.002,
    ...     initial_direction=np.array([0, 0, -1]),
    ... )
    >>> 
    >>> # Convert to core network
    >>> network = liver_tree_to_network(tree, domain)
    """
    network = VascularNetwork(
        domain=domain,
        metadata=metadata or {},
    )
    
    # Add metadata about source
    network.metadata["source"] = "liver_generator"
    network.metadata["tree_type"] = tree.tree_type
    
    # Map liver node IDs to core node IDs
    liver_to_core_node_id: Dict[int, int] = {}
    
    # Convert nodes
    for liver_node in tree.nodes:
        core_node_id = network.id_gen.next_id()
        liver_to_core_node_id[liver_node.id] = core_node_id
        
        # Determine node type based on tree structure
        if liver_node.parent_id is None:
            node_type = "inlet" if tree.tree_type == "arterial" else "outlet"
        elif len(liver_node.children_ids) == 0:
            node_type = "terminal"
        elif len(liver_node.children_ids) >= 2:
            node_type = "junction"
        else:
            node_type = "junction"  # Single child is still a junction
        
        core_node = CoreNode(
            id=core_node_id,
            position=Point3D(
                x=float(liver_node.position[0]),
                y=float(liver_node.position[1]),
                z=float(liver_node.position[2]),
            ),
            node_type=node_type,
            vessel_type=tree.tree_type,
            attributes={
                "radius": liver_node.radius,
                "branch_order": liver_node.order,
                "flow": liver_node.flow,
                "liver_node_id": liver_node.id,
                "parent_id": liver_node.parent_id,
                "children_ids": liver_node.children_ids.copy(),
            },
        )
        network.add_node(core_node)
    
    # Convert segments
    for liver_segment in tree.segments:
        core_segment_id = network.id_gen.next_id()
        
        start_node_id = liver_to_core_node_id[liver_segment.parent_node_id]
        end_node_id = liver_to_core_node_id[liver_segment.child_node_id]
        
        start_node = network.get_node(start_node_id)
        end_node = network.get_node(end_node_id)
        
        core_segment = VesselSegment(
            id=core_segment_id,
            start_node_id=start_node_id,
            end_node_id=end_node_id,
            geometry=TubeGeometry(
                start=start_node.position,
                end=end_node.position,
                radius_start=liver_segment.radius_start,
                radius_end=liver_segment.radius_end,
            ),
            vessel_type=tree.tree_type,
            attributes={
                "liver_segment_id": liver_segment.id,
                "length": liver_segment.length,
                "direction": liver_segment.direction.tolist(),
                "radius": (liver_segment.radius_start + liver_segment.radius_end) / 2,
            },
        )
        network.add_segment(core_segment)
    
    return network


def network_to_liver_tree(
    network: VascularNetwork,
    vessel_type: str = "arterial",
    root_node_id: Optional[int] = None,
) -> VascularTree:
    """
    Convert a core VascularNetwork to a liver VascularTree.
    
    This adapter extracts a single tree from the network by following
    parent-child relationships from a root node.
    
    Parameters
    ----------
    network : VascularNetwork
        Core network to convert
    vessel_type : str
        Vessel type to extract ("arterial" or "venous")
    root_node_id : int, optional
        ID of root node. If not provided, finds inlet/outlet based on vessel_type.
        
    Returns
    -------
    VascularTree
        Liver-specific tree representation
        
    Notes
    -----
    This conversion may lose some information if the network has cycles
    or multiple roots, as VascularTree assumes a strict tree structure.
    """
    # Find root node
    if root_node_id is None:
        target_type = "inlet" if vessel_type == "arterial" else "outlet"
        for node in network.nodes.values():
            if node.node_type == target_type and node.vessel_type == vessel_type:
                root_node_id = node.id
                break
        
        if root_node_id is None:
            raise ValueError(f"No {target_type} node found for vessel_type={vessel_type}")
    
    root_node = network.get_node(root_node_id)
    if root_node is None:
        raise ValueError(f"Root node {root_node_id} not found")
    
    # Get initial direction from first segment
    connected_segs = network.get_connected_segment_ids(root_node_id)
    if connected_segs:
        first_seg = network.get_segment(connected_segs[0])
        if first_seg.start_node_id == root_node_id:
            other_node = network.get_node(first_seg.end_node_id)
        else:
            other_node = network.get_node(first_seg.start_node_id)
        
        direction = np.array([
            other_node.position.x - root_node.position.x,
            other_node.position.y - root_node.position.y,
            other_node.position.z - root_node.position.z,
        ])
        direction = direction / (np.linalg.norm(direction) + 1e-9)
    else:
        direction = np.array([0.0, 0.0, -1.0])
    
    # Create liver tree
    tree = VascularTree(
        tree_type=vessel_type,
        root_position=np.array([root_node.position.x, root_node.position.y, root_node.position.z]),
        root_radius=root_node.attributes.get("radius", 0.002),
        initial_direction=direction,
        root_id=0,
    )
    
    # Clear the auto-created root node (we'll add our own)
    # Also reinitialize spatial_index to avoid stale data
    tree.nodes.clear()
    tree.nodes_by_id.clear()
    tree.active_tips.clear()
    tree.spatial_index.nodes.clear()
    tree.spatial_index.segments.clear()
    tree.spatial_index.grid.clear()
    
    # Build tree structure using BFS from root
    core_to_liver_node_id: Dict[int, int] = {}
    visited_nodes = set()
    visited_segments = set()
    
    # Queue: (core_node_id, parent_liver_id, branch_order)
    queue = [(root_node_id, None, 0)]
    next_liver_id = 0
    next_segment_id = 0
    
    while queue:
        core_node_id, parent_liver_id, branch_order = queue.pop(0)
        
        if core_node_id in visited_nodes:
            continue
        visited_nodes.add(core_node_id)
        
        core_node = network.get_node(core_node_id)
        
        # Create liver node
        liver_node_id = next_liver_id
        next_liver_id += 1
        core_to_liver_node_id[core_node_id] = liver_node_id
        
        liver_node = LiverNode(
            id=liver_node_id,
            position=np.array([core_node.position.x, core_node.position.y, core_node.position.z]),
            radius=core_node.attributes.get("radius", 0.001),
            parent_id=parent_liver_id,
            children_ids=[],
            order=branch_order,
            flow=core_node.attributes.get("flow", 0.0),
        )
        tree.add_node(liver_node)
        
        # Update parent's children list
        if parent_liver_id is not None:
            parent_liver_node = tree.get_node(parent_liver_id)
            parent_liver_node.children_ids.append(liver_node_id)
        
        # Find connected segments and add children
        connected_segs = network.get_connected_segment_ids(core_node_id)
        child_count = 0
        
        for seg_id in connected_segs:
            if seg_id in visited_segments:
                continue
            
            segment = network.get_segment(seg_id)
            if segment.vessel_type != vessel_type:
                continue
            
            visited_segments.add(seg_id)
            
            # Determine child node
            if segment.start_node_id == core_node_id:
                child_core_id = segment.end_node_id
            else:
                child_core_id = segment.start_node_id
            
            if child_core_id in visited_nodes:
                continue
            
            child_count += 1
            
            # Add child to queue with incremented branch order if bifurcation
            new_order = branch_order + (1 if child_count > 1 else 0)
            queue.append((child_core_id, liver_node_id, new_order))
    
    # Create segments based on parent-child relationships
    for liver_node in tree.nodes:
        if liver_node.parent_id is not None:
            parent_node = tree.get_node(liver_node.parent_id)
            
            direction = liver_node.position - parent_node.position
            length = float(np.linalg.norm(direction))
            if length > 0:
                direction = direction / length
            else:
                direction = np.array([0.0, 0.0, 1.0])
            
            segment = LiverSegment(
                id=next_segment_id,
                parent_node_id=liver_node.parent_id,
                child_node_id=liver_node.id,
                length=length,
                direction=direction,
                radius_start=parent_node.radius,
                radius_end=liver_node.radius,
            )
            tree.add_segment(segment)
            next_segment_id += 1
    
    return tree


def merge_liver_trees_to_network(
    arterial_tree: VascularTree,
    venous_tree: VascularTree,
    domain: DomainSpec,
    metadata: Optional[Dict[str, Any]] = None,
) -> VascularNetwork:
    """
    Merge arterial and venous liver trees into a single VascularNetwork.
    
    This is useful for creating a complete dual-tree network from
    separately generated arterial and venous trees.
    
    Parameters
    ----------
    arterial_tree : VascularTree
        Arterial vascular tree
    venous_tree : VascularTree
        Venous vascular tree
    domain : DomainSpec
        Domain specification for the network
    metadata : dict, optional
        Additional metadata
        
    Returns
    -------
    VascularNetwork
        Combined network with both trees
    """
    # Convert arterial tree first
    network = liver_tree_to_network(arterial_tree, domain, metadata)
    
    # Add venous tree nodes and segments
    liver_to_core_node_id: Dict[int, int] = {}
    
    for liver_node in venous_tree.nodes:
        core_node_id = network.id_gen.next_id()
        liver_to_core_node_id[liver_node.id] = core_node_id
        
        if liver_node.parent_id is None:
            node_type = "outlet"
        elif len(liver_node.children_ids) == 0:
            node_type = "terminal"
        else:
            node_type = "junction"
        
        core_node = CoreNode(
            id=core_node_id,
            position=Point3D(
                x=float(liver_node.position[0]),
                y=float(liver_node.position[1]),
                z=float(liver_node.position[2]),
            ),
            node_type=node_type,
            vessel_type="venous",
            attributes={
                "radius": liver_node.radius,
                "branch_order": liver_node.order,
                "flow": liver_node.flow,
                "liver_node_id": liver_node.id,
            },
        )
        network.add_node(core_node)
    
    for liver_segment in venous_tree.segments:
        core_segment_id = network.id_gen.next_id()
        
        start_node_id = liver_to_core_node_id[liver_segment.parent_node_id]
        end_node_id = liver_to_core_node_id[liver_segment.child_node_id]
        
        start_node = network.get_node(start_node_id)
        end_node = network.get_node(end_node_id)
        
        core_segment = VesselSegment(
            id=core_segment_id,
            start_node_id=start_node_id,
            end_node_id=end_node_id,
            geometry=TubeGeometry(
                start=start_node.position,
                end=end_node.position,
                radius_start=liver_segment.radius_start,
                radius_end=liver_segment.radius_end,
            ),
            vessel_type="venous",
            attributes={
                "liver_segment_id": liver_segment.id,
                "length": liver_segment.length,
                "radius": (liver_segment.radius_start + liver_segment.radius_end) / 2,
            },
        )
        network.add_segment(core_segment)
    
    network.metadata["has_dual_trees"] = True
    
    return network
