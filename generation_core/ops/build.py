"""
Construction operations for building vascular networks.
"""

from typing import Optional, Tuple
from ..core.types import Point3D, Direction3D
from ..core.network import VascularNetwork, Node
from ..core.domain import DomainSpec
from ..core.result import OperationResult, OperationStatus, Delta


def create_network(
    domain: DomainSpec,
    metadata: Optional[dict] = None,
    seed: Optional[int] = None,
) -> VascularNetwork:
    """
    Create a new empty vascular network.
    
    Parameters
    ----------
    domain : DomainSpec
        Geometric domain for the network
    metadata : dict, optional
        Network metadata (name, organ type, units, etc.)
    seed : int, optional
        Random seed for deterministic operations
    
    Returns
    -------
    network : VascularNetwork
        New empty network
    
    Example
    -------
    >>> from vascular_lib import create_network, EllipsoidDomain
    >>> domain = EllipsoidDomain(0.12, 0.10, 0.08)
    >>> network = create_network(domain, seed=42)
    """
    if metadata is None:
        metadata = {
            "name": "Vascular Network",
            "units": "meters",
        }
    
    return VascularNetwork(domain=domain, metadata=metadata, seed=seed)


def add_inlet(
    network: VascularNetwork,
    position: Tuple[float, float, float] | Point3D,
    direction: Tuple[float, float, float] | Direction3D,
    radius: float,
    vessel_type: str = "arterial",
) -> OperationResult:
    """
    Add an inlet node to the network.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to modify
    position : tuple or Point3D
        Position of inlet
    direction : tuple or Direction3D
        Initial flow direction
    radius : float
        Inlet radius
    vessel_type : str
        Vessel type ("arterial" or "venous")
    
    Returns
    -------
    result : OperationResult
        Result with new_ids['node'] containing the inlet node ID
    
    Example
    -------
    >>> result = add_inlet(network, position=(0, 0, 0), direction=(1, 0, 0), radius=0.005)
    >>> inlet_id = result.new_ids['node']
    """
    if isinstance(position, tuple):
        position = Point3D.from_tuple(position)
    if isinstance(direction, tuple):
        direction = Direction3D.from_tuple(direction)
    
    if not network.domain.contains(position):
        return OperationResult.failure(
            message=f"Inlet position {position.to_tuple()} is outside domain",
            errors=["Position outside domain"],
        )
    
    node_id = network.id_gen.next_id()
    node = Node(
        id=node_id,
        position=position,
        node_type="inlet",
        vessel_type=vessel_type,
        attributes={
            "radius": radius,
            "direction": direction.to_dict(),
        },
    )
    
    network.add_node(node)
    
    delta = Delta(created_node_ids=[node_id])
    
    return OperationResult.success(
        message=f"Added {vessel_type} inlet at {position.to_tuple()}",
        new_ids={"node": node_id},
        delta=delta,
        rng_state=network.id_gen.get_state(),
    )


def add_outlet(
    network: VascularNetwork,
    position: Tuple[float, float, float] | Point3D,
    direction: Tuple[float, float, float] | Direction3D,
    radius: float,
    vessel_type: str = "venous",
) -> OperationResult:
    """
    Add an outlet node to the network.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to modify
    position : tuple or Point3D
        Position of outlet
    direction : tuple or Direction3D
        Initial flow direction
    radius : float
        Outlet radius
    vessel_type : str
        Vessel type ("arterial" or "venous")
    
    Returns
    -------
    result : OperationResult
        Result with new_ids['node'] containing the outlet node ID
    
    Example
    -------
    >>> result = add_outlet(network, position=(0.1, 0, 0), direction=(-1, 0, 0), radius=0.006)
    >>> outlet_id = result.new_ids['node']
    """
    if isinstance(position, tuple):
        position = Point3D.from_tuple(position)
    if isinstance(direction, tuple):
        direction = Direction3D.from_tuple(direction)
    
    if not network.domain.contains(position):
        return OperationResult.failure(
            message=f"Outlet position {position.to_tuple()} is outside domain",
            errors=["Position outside domain"],
        )
    
    node_id = network.id_gen.next_id()
    node = Node(
        id=node_id,
        position=position,
        node_type="outlet",
        vessel_type=vessel_type,
        attributes={
            "radius": radius,
            "direction": direction.to_dict(),
        },
    )
    
    network.add_node(node)
    
    delta = Delta(created_node_ids=[node_id])
    
    return OperationResult.success(
        message=f"Added {vessel_type} outlet at {position.to_tuple()}",
        new_ids={"node": node_id},
        delta=delta,
        rng_state=network.id_gen.get_state(),
    )
