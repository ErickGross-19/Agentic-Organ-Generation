"""
Space colonization backend wrapper.

Thin wrapper around existing ops/space_colonization.py to provide
a unified backend interface.

Note: The library uses METERS internally for all geometry.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .base import GenerationBackend, BackendConfig, GenerationState, GenerationAction
from ..core.network import VascularNetwork, Node
from ..core.domain import DomainSpec
from ..core.types import Point3D, Direction3D
from ..ops.space_colonization import space_colonization_step, SpaceColonizationConfig as SCConfig


@dataclass
class SpaceColonizationConfig(BackendConfig):
    """Configuration for space colonization backend."""
    
    attraction_distance: float = 0.010  # meters
    kill_distance: float = 0.002  # meters
    step_size: float = 0.002  # meters
    num_attractors: int = 1000
    max_iterations: int = 500
    branch_angle_deg: float = 30.0


class SpaceColonizationBackend(GenerationBackend):
    """
    Space colonization backend for vascular network generation.
    
    Wraps the existing space colonization operations to provide
    a unified backend interface.
    """
    
    @property
    def supports_dual_tree(self) -> bool:
        return True
    
    @property
    def supports_closed_loop(self) -> bool:
        return False
    
    def generate(
        self,
        domain: DomainSpec,
        num_outlets: int,
        inlet_position: np.ndarray,
        inlet_radius: float,
        vessel_type: str = "arterial",
        config: Optional[SpaceColonizationConfig] = None,
        rng_seed: Optional[int] = None,
    ) -> VascularNetwork:
        """
        Generate a vascular network using space colonization.
        
        Parameters
        ----------
        domain : DomainSpec
            Geometric domain for the network
        num_outlets : int
            Target number of terminal outlets (used to scale attractors)
        inlet_position : np.ndarray
            Position of the inlet node (x, y, z) in meters
        inlet_radius : float
            Radius of the inlet vessel in meters
        vessel_type : str
            Type of vessels ("arterial" or "venous")
        config : SpaceColonizationConfig, optional
            Space colonization configuration
        rng_seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        VascularNetwork
            Generated vascular network
        """
        if config is None:
            config = SpaceColonizationConfig()
        
        rng = np.random.default_rng(rng_seed if rng_seed is not None else config.seed)
        
        network = VascularNetwork(domain=domain, seed=rng_seed)
        
        inlet_point = Point3D.from_array(inlet_position)
        inlet_direction = self._compute_initial_direction(inlet_point, domain)
        
        inlet_node = Node(
            id=network.id_gen.next_id(),
            position=inlet_point,
            node_type="inlet",
            vessel_type=vessel_type,
            attributes={
                "radius": inlet_radius,
                "direction": inlet_direction.to_dict(),
                "branch_order": 0,
            },
        )
        network.add_node(inlet_node)
        
        num_attractors = max(config.num_attractors, num_outlets * 2)
        attractors = domain.sample_points(num_attractors, seed=int(rng.integers(0, 2**31)))
        attractor_list = [Point3D.from_array(a) for a in attractors]
        
        sc_config = SCConfig(
            attraction_distance=config.attraction_distance,
            kill_distance=config.kill_distance,
            step_size=config.step_size,
            branch_angle_deg=config.branch_angle_deg,
        )
        
        active_nodes = [inlet_node.id]
        
        for iteration in range(config.max_iterations):
            if not attractor_list or not active_nodes:
                break
            
            result = space_colonization_step(
                network=network,
                attractors=attractor_list,
                active_node_ids=active_nodes,
                config=sc_config,
                base_radius=inlet_radius,
                vessel_type=vessel_type,
            )
            
            if result.is_success():
                new_node_ids = result.new_ids.get("nodes", [])
                consumed_attractors = result.metadata.get("consumed_attractors", [])
                
                if new_node_ids:
                    active_nodes = new_node_ids
                
                attractor_list = [
                    a for i, a in enumerate(attractor_list)
                    if i not in consumed_attractors
                ]
            else:
                break
        
        self._mark_terminals(network)
        
        return network
    
    def step(
        self,
        state: GenerationState,
        action: GenerationAction,
    ) -> GenerationState:
        """
        Perform a single space colonization step.
        
        Supports actions:
        - "grow": Perform one iteration of space colonization
        """
        if action.action_type == "grow":
            config = state.metadata.get("config", SpaceColonizationConfig())
            attractors = state.metadata.get("attractors", [])
            active_nodes = state.metadata.get("active_nodes", [])
            inlet_radius = state.metadata.get("inlet_radius", 0.002)
            vessel_type = state.metadata.get("vessel_type", "arterial")
            
            if attractors and active_nodes:
                sc_config = SCConfig(
                    attraction_distance=config.attraction_distance,
                    kill_distance=config.kill_distance,
                    step_size=config.step_size,
                    branch_angle_deg=config.branch_angle_deg,
                )
                
                result = space_colonization_step(
                    network=state.network,
                    attractors=attractors,
                    active_node_ids=active_nodes,
                    config=sc_config,
                    base_radius=inlet_radius,
                    vessel_type=vessel_type,
                )
                
                if result.is_success():
                    new_node_ids = result.new_ids.get("nodes", [])
                    consumed_attractors = result.metadata.get("consumed_attractors", [])
                    
                    if new_node_ids:
                        state.metadata["active_nodes"] = new_node_ids
                    
                    state.metadata["attractors"] = [
                        a for i, a in enumerate(attractors)
                        if i not in consumed_attractors
                    ]
            
            state.iteration += 1
        
        return state
    
    def generate_dual_tree(
        self,
        domain: DomainSpec,
        arterial_outlets: int,
        venous_outlets: int,
        arterial_inlet: np.ndarray,
        venous_outlet: np.ndarray,
        arterial_radius: float,
        venous_radius: float,
        config: Optional[SpaceColonizationConfig] = None,
        rng_seed: Optional[int] = None,
        create_anastomoses: bool = False,
        num_anastomoses: int = 0,
    ) -> VascularNetwork:
        """
        Generate a dual arterial-venous network using space colonization.
        """
        if config is None:
            config = SpaceColonizationConfig()
        
        rng = np.random.default_rng(rng_seed if rng_seed is not None else config.seed)
        
        network = VascularNetwork(domain=domain, seed=rng_seed)
        
        arterial_inlet_point = Point3D.from_array(arterial_inlet)
        arterial_direction = self._compute_initial_direction(arterial_inlet_point, domain)
        
        arterial_inlet_node = Node(
            id=network.id_gen.next_id(),
            position=arterial_inlet_point,
            node_type="inlet",
            vessel_type="arterial",
            attributes={
                "radius": arterial_radius,
                "direction": arterial_direction.to_dict(),
                "branch_order": 0,
            },
        )
        network.add_node(arterial_inlet_node)
        
        num_attractors = max(config.num_attractors, (arterial_outlets + venous_outlets) * 2)
        all_attractors = domain.sample_points(num_attractors, seed=int(rng.integers(0, 2**31)))
        
        arterial_attractors = [Point3D.from_array(a) for a in all_attractors[:len(all_attractors)//2]]
        
        sc_config = SCConfig(
            attraction_distance=config.attraction_distance,
            kill_distance=config.kill_distance,
            step_size=config.step_size,
            branch_angle_deg=config.branch_angle_deg,
        )
        
        active_nodes = [arterial_inlet_node.id]
        for iteration in range(config.max_iterations // 2):
            if not arterial_attractors or not active_nodes:
                break
            
            result = space_colonization_step(
                network=network,
                attractors=arterial_attractors,
                active_node_ids=active_nodes,
                config=sc_config,
                base_radius=arterial_radius,
                vessel_type="arterial",
            )
            
            if result.is_success():
                new_node_ids = result.new_ids.get("nodes", [])
                consumed = result.metadata.get("consumed_attractors", [])
                
                if new_node_ids:
                    active_nodes = new_node_ids
                
                arterial_attractors = [
                    a for i, a in enumerate(arterial_attractors)
                    if i not in consumed
                ]
            else:
                break
        
        venous_outlet_point = Point3D.from_array(venous_outlet)
        venous_direction = self._compute_initial_direction(venous_outlet_point, domain)
        
        venous_outlet_node = Node(
            id=network.id_gen.next_id(),
            position=venous_outlet_point,
            node_type="outlet",
            vessel_type="venous",
            attributes={
                "radius": venous_radius,
                "direction": venous_direction.to_dict(),
                "branch_order": 0,
            },
        )
        network.add_node(venous_outlet_node)
        
        venous_attractors = [Point3D.from_array(a) for a in all_attractors[len(all_attractors)//2:]]
        
        active_nodes = [venous_outlet_node.id]
        for iteration in range(config.max_iterations // 2):
            if not venous_attractors or not active_nodes:
                break
            
            result = space_colonization_step(
                network=network,
                attractors=venous_attractors,
                active_node_ids=active_nodes,
                config=sc_config,
                base_radius=venous_radius,
                vessel_type="venous",
            )
            
            if result.is_success():
                new_node_ids = result.new_ids.get("nodes", [])
                consumed = result.metadata.get("consumed_attractors", [])
                
                if new_node_ids:
                    active_nodes = new_node_ids
                
                venous_attractors = [
                    a for i, a in enumerate(venous_attractors)
                    if i not in consumed
                ]
            else:
                break
        
        self._mark_terminals(network)
        
        if create_anastomoses and num_anastomoses > 0:
            self._create_anastomoses(network, num_anastomoses)
        
        return network
    
    def _compute_initial_direction(self, inlet_point: Point3D, domain: DomainSpec) -> Direction3D:
        """Compute initial growth direction pointing toward domain center."""
        bounds = domain.get_bounds()
        center = np.array([
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2,
        ])
        
        inlet_arr = inlet_point.to_array()
        direction = center - inlet_arr
        length = np.linalg.norm(direction)
        
        if length < 1e-10:
            return Direction3D(0, 0, 1)
        
        direction = direction / length
        return Direction3D.from_array(direction)
    
    def _mark_terminals(self, network: VascularNetwork) -> None:
        """Mark leaf nodes as terminals."""
        for node in network.nodes.values():
            if node.node_type in ["inlet", "outlet"]:
                continue
            
            connected = network.get_connected_segment_ids(node.id)
            if len(connected) == 1:
                node.node_type = "terminal"
            elif len(connected) > 1:
                node.node_type = "junction"
    
    def _create_anastomoses(self, network: VascularNetwork, num_anastomoses: int) -> None:
        """Create anastomoses between arterial and venous terminals."""
        from ..ops.anastomosis import create_anastomosis
        
        arterial_terminals = [
            n for n in network.nodes.values()
            if n.vessel_type == "arterial" and n.node_type == "terminal"
        ]
        venous_terminals = [
            n for n in network.nodes.values()
            if n.vessel_type == "venous" and n.node_type == "terminal"
        ]
        
        if not arterial_terminals or not venous_terminals:
            return
        
        pairs = []
        for a_node in arterial_terminals:
            for v_node in venous_terminals:
                dist = a_node.position.distance_to(v_node.position)
                pairs.append((a_node.id, v_node.id, dist))
        
        pairs.sort(key=lambda x: x[2])
        
        created = 0
        used_arterial = set()
        used_venous = set()
        
        for a_id, v_id, dist in pairs:
            if created >= num_anastomoses:
                break
            if a_id in used_arterial or v_id in used_venous:
                continue
            
            result = create_anastomosis(network, a_id, v_id, max_length=0.015)
            if result.is_success():
                created += 1
                used_arterial.add(a_id)
                used_venous.add(v_id)
