"""
Accelerated CCO (Constrained Constructive Optimization) hybrid backend.

Based on Sexton et al.'s "rapid model-guided design" pipeline for organ-scale
synthetic vascular generation. Implements four major accelerators:
1. Partial binding optimization
2. Partial implicit volumes (fast domain queries)
3. Collision avoidance triage
4. Closed-loop (A-V) vasculature generation

Note: The library uses METERS internally for all geometry.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Set
import numpy as np

from .base import GenerationBackend, BackendConfig, GenerationState, GenerationAction
from ..core.network import VascularNetwork, Node, VesselSegment
from ..core.domain import DomainSpec
from ..core.types import Point3D, TubeGeometry


@dataclass
class CCOConfig(BackendConfig):
    """Configuration for CCO hybrid backend."""
    
    murray_exponent: float = 3.0
    cost_length_weight: float = 1.0
    cost_radius_weight: float = 1.0
    boundary_penalty_weight: float = 10.0
    optimization_grid_resolution: int = 10
    candidate_edges_k: int = 50
    use_partial_binding: bool = True
    use_collision_triage: bool = True
    
    # Collision avoidance parameters
    collision_clearance: float = 0.0001  # 0.1mm minimum clearance between vessels
    collision_check_enabled: bool = True  # Enable actual collision prevention during insertion
    
    # Dual-tree outlet validation parameters
    min_terminal_separation_same_type: Optional[float] = None  # If None, uses min_terminal_separation
    min_terminal_separation_cross_type: Optional[float] = None  # A-V separation (if None, no cross-type check)
    encourage_av_proximity: bool = False  # If True, prefer outlets near opposite vessel type terminals


@dataclass
class SegmentRecord:
    """Array-friendly segment record for CCO optimization cache."""
    
    segment_id: int
    start_node_id: int
    end_node_id: int
    parent_segment_id: Optional[int]
    child_segment_ids: List[int]
    length: float
    radius: float
    resistance: float
    subtree_terminal_count: int
    subtree_equivalent_resistance: float
    path_to_root_resistance: float


class ArrayTreeView:
    """
    Array-based view of vascular tree for fast CCO operations.
    
    This is an optimization cache that maintains contiguous arrays of segment
    properties for efficient traversal and update during CCO insertion.
    The authoritative geometry lives in VascularNetwork.
    """
    
    def __init__(self, network: VascularNetwork, root_node_id: int):
        """
        Initialize ArrayTreeView from a VascularNetwork.
        
        Parameters
        ----------
        network : VascularNetwork
            Source network
        root_node_id : int
            ID of the root (inlet) node
        """
        self.network = network
        self.root_node_id = root_node_id
        self.segments: Dict[int, SegmentRecord] = {}
        self._segment_id_to_idx: Dict[int, int] = {}
        self._rebuild_from_network()
    
    def _rebuild_from_network(self) -> None:
        """Rebuild cache from network."""
        self.segments.clear()
        self._segment_id_to_idx.clear()
        
        parent_map = self._compute_parent_map()
        
        for idx, (seg_id, seg) in enumerate(self.network.segments.items()):
            parent_seg_id = parent_map.get(seg_id)
            child_ids = self._get_child_segment_ids(seg_id, parent_map)
            
            radius = seg.geometry.mean_radius()
            length = seg.length
            resistance = self._compute_resistance(length, radius)
            
            record = SegmentRecord(
                segment_id=seg_id,
                start_node_id=seg.start_node_id,
                end_node_id=seg.end_node_id,
                parent_segment_id=parent_seg_id,
                child_segment_ids=child_ids,
                length=length,
                radius=radius,
                resistance=resistance,
                subtree_terminal_count=0,
                subtree_equivalent_resistance=0.0,
                path_to_root_resistance=0.0,
            )
            self.segments[seg_id] = record
            self._segment_id_to_idx[seg_id] = idx
        
        self._compute_subtree_aggregates()
        self._compute_path_to_root_aggregates()
    
    def _compute_parent_map(self) -> Dict[int, Optional[int]]:
        """
        Compute parent segment for each segment based on tree structure.
        
        P0-8: Uses proper BFS/DFS traversal from root_node_id to build directed
        tree structure. Tracks incoming segment at each node to correctly identify
        parent-child relationships even at junctions with multiple branches.
        """
        parent_map: Dict[int, Optional[int]] = {}
        
        visited_nodes: Set[int] = {self.root_node_id}
        # Queue contains (node_id, incoming_seg_id) tuples
        # incoming_seg_id is None for root, otherwise the segment we arrived from
        queue: List[Tuple[int, Optional[int]]] = [(self.root_node_id, None)]
        
        while queue:
            node_id, incoming_seg_id = queue.pop(0)
            connected_segs = self.network.get_connected_segment_ids(node_id)
            
            for seg_id in connected_segs:
                # Skip the segment we arrived from (it's already processed)
                if seg_id == incoming_seg_id:
                    continue
                    
                seg = self.network.segments[seg_id]
                other_node = seg.end_node_id if seg.start_node_id == node_id else seg.start_node_id
                
                if other_node not in visited_nodes:
                    # P0-8: Parent is the incoming segment at this node
                    parent_map[seg_id] = incoming_seg_id
                    visited_nodes.add(other_node)
                    # Continue BFS with this segment as the incoming segment for other_node
                    queue.append((other_node, seg_id))
        
        return parent_map
    
    def _find_parent_segment(self, parent_node_id: int, current_seg_id: int) -> Optional[int]:
        """
        Find the parent segment of a given segment.
        
        Note: This method is kept for backward compatibility but the main
        parent mapping logic is now in _compute_parent_map using proper BFS.
        """
        connected = self.network.get_connected_segment_ids(parent_node_id)
        for seg_id in connected:
            if seg_id != current_seg_id:
                return seg_id
        return None
    
    def _get_child_segment_ids(self, seg_id: int, parent_map: Dict[int, Optional[int]]) -> List[int]:
        """Get child segment IDs for a segment."""
        children = []
        for other_id, parent_id in parent_map.items():
            if parent_id == seg_id:
                children.append(other_id)
        return children
    
    def _compute_resistance(self, length: float, radius: float, mu: float = 1e-3) -> float:
        """Compute Poiseuille resistance: R = 8*mu*L/(pi*r^4)."""
        if radius < 1e-10:
            return float('inf')
        return 8.0 * mu * length / (np.pi * radius**4)
    
    def _compute_subtree_aggregates(self) -> None:
        """Compute subtree terminal counts and equivalent resistances."""
        terminal_node_ids = {
            n.id for n in self.network.nodes.values()
            if n.node_type == "terminal"
        }
        
        def compute_subtree(seg_id: int) -> Tuple[int, float]:
            record = self.segments[seg_id]
            
            if not record.child_segment_ids:
                end_node = self.network.nodes[record.end_node_id]
                is_terminal = end_node.id in terminal_node_ids
                record.subtree_terminal_count = 1 if is_terminal else 0
                record.subtree_equivalent_resistance = record.resistance
                return record.subtree_terminal_count, record.subtree_equivalent_resistance
            
            total_terminals = 0
            child_resistances = []
            
            for child_id in record.child_segment_ids:
                child_terminals, child_eq_res = compute_subtree(child_id)
                total_terminals += child_terminals
                if child_eq_res > 0:
                    child_resistances.append(child_eq_res)
            
            if child_resistances:
                parallel_inv = sum(1.0 / r for r in child_resistances if r > 0)
                parallel_res = 1.0 / parallel_inv if parallel_inv > 0 else float('inf')
                eq_resistance = record.resistance + parallel_res
            else:
                eq_resistance = record.resistance
            
            record.subtree_terminal_count = total_terminals
            record.subtree_equivalent_resistance = eq_resistance
            return total_terminals, eq_resistance
        
        root_segments = [
            seg_id for seg_id, rec in self.segments.items()
            if rec.parent_segment_id is None
        ]
        for seg_id in root_segments:
            compute_subtree(seg_id)
    
    def _compute_path_to_root_aggregates(self) -> None:
        """Compute path-to-root resistance for each segment."""
        for seg_id, record in self.segments.items():
            path_resistance = 0.0
            current_id = record.parent_segment_id
            
            while current_id is not None:
                parent_record = self.segments.get(current_id)
                if parent_record:
                    path_resistance += parent_record.resistance
                    current_id = parent_record.parent_segment_id
                else:
                    break
            
            record.path_to_root_resistance = path_resistance
    
    def get_segment_endpoints(self, seg_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get segment endpoints as numpy arrays."""
        seg = self.network.segments[seg_id]
        start_node = self.network.nodes[seg.start_node_id]
        end_node = self.network.nodes[seg.end_node_id]
        return start_node.position.to_array(), end_node.position.to_array()
    
    def update_after_insertion(
        self,
        split_seg_id: int,
        new_seg_ids: List[int],
        bifurcation_node_id: int,
    ) -> None:
        """
        Update cache after inserting a new outlet.
        
        Uses partial binding to only update affected path to root.
        """
        self._rebuild_from_network()


class CCOHybridBackend(GenerationBackend):
    """
    Accelerated CCO hybrid backend for vascular network generation.
    
    Implements Sexton-style accelerators for organ-scale generation:
    - Partial binding optimization for fast bifurcation optimization
    - Partial implicit volumes for fast domain queries
    - Collision avoidance triage (cheap filter then expensive test)
    - Closed-loop (A-V) vasculature generation
    """
    
    @property
    def supports_dual_tree(self) -> bool:
        return True
    
    @property
    def supports_closed_loop(self) -> bool:
        return True
    
    def generate(
        self,
        domain: DomainSpec,
        num_outlets: int,
        inlet_position: np.ndarray,
        inlet_radius: float,
        vessel_type: str = "arterial",
        config: Optional[CCOConfig] = None,
        rng_seed: Optional[int] = None,
    ) -> VascularNetwork:
        """
        Generate a vascular network using accelerated CCO.
        
        Parameters
        ----------
        domain : DomainSpec
            Geometric domain for the network
        num_outlets : int
            Target number of terminal outlets
        inlet_position : np.ndarray
            Position of the inlet node (x, y, z) in meters
        inlet_radius : float
            Radius of the inlet vessel in meters
        vessel_type : str
            Type of vessels ("arterial" or "venous")
        config : CCOConfig, optional
            CCO configuration
        rng_seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        VascularNetwork
            Generated vascular network
        """
        if config is None:
            config = CCOConfig()
        
        rng = np.random.default_rng(rng_seed if rng_seed is not None else config.seed)
        
        network = VascularNetwork(domain=domain, seed=rng_seed)
        
        inlet_point = Point3D.from_array(inlet_position)
        inlet_node = Node(
            id=network.id_gen.next_id(),
            position=inlet_point,
            node_type="inlet",
            vessel_type=vessel_type,
            attributes={"radius": inlet_radius, "branch_order": 0},
        )
        network.add_node(inlet_node)
        
        first_outlet = self._sample_outlet_point(domain, rng, config)
        self._add_initial_segment(network, inlet_node, first_outlet, inlet_radius, vessel_type)
        
        tree_view = ArrayTreeView(network, inlet_node.id)
        
        for i in range(num_outlets - 1):
            outlet_point = self._sample_outlet_point(domain, rng, config)
            
            if not self._is_valid_outlet(outlet_point, network, config):
                continue
            
            self._insert_outlet(
                network, tree_view, outlet_point, inlet_radius, vessel_type, config, rng
            )
        
        self._rescale_radii(network, inlet_node.id, inlet_radius, config)
        
        return network
    
    def step(
        self,
        state: GenerationState,
        action: GenerationAction,
    ) -> GenerationState:
        """
        Perform a single CCO generation step.
        
        Supports actions:
        - "add_outlet": Sample and add a new outlet
        - "force_outlet": Add outlet at specific location
        """
        if action.action_type == "add_outlet":
            config = state.metadata.get("config", CCOConfig())
            rng = state.metadata.get("rng", np.random.default_rng())
            
            outlet_point = self._sample_outlet_point(state.network.domain, rng, config)
            
            if self._is_valid_outlet(outlet_point, state.network, config):
                tree_view = state.metadata.get("tree_view")
                if tree_view is None:
                    root_id = state.metadata.get("root_node_id")
                    tree_view = ArrayTreeView(state.network, root_id)
                
                inlet_radius = state.metadata.get("inlet_radius", 0.002)
                vessel_type = state.metadata.get("vessel_type", "arterial")
                
                self._insert_outlet(
                    state.network, tree_view, outlet_point,
                    inlet_radius, vessel_type, config, rng
                )
                
                state.remaining_outlets -= 1
            
            state.iteration += 1
            
        elif action.action_type == "force_outlet":
            target = action.parameters.get("target_point")
            if target is not None:
                target_point = Point3D.from_array(np.array(target))
                config = state.metadata.get("config", CCOConfig())
                rng = state.metadata.get("rng", np.random.default_rng())
                
                tree_view = state.metadata.get("tree_view")
                if tree_view is None:
                    root_id = state.metadata.get("root_node_id")
                    tree_view = ArrayTreeView(state.network, root_id)
                
                inlet_radius = state.metadata.get("inlet_radius", 0.002)
                vessel_type = state.metadata.get("vessel_type", "arterial")
                
                self._insert_outlet(
                    state.network, tree_view, target_point,
                    inlet_radius, vessel_type, config, rng
                )
                
                state.remaining_outlets -= 1
            
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
        config: Optional[CCOConfig] = None,
        rng_seed: Optional[int] = None,
        create_anastomoses: bool = False,
        num_anastomoses: int = 0,
    ) -> VascularNetwork:
        """
        Generate a dual arterial-venous network with optional anastomoses.
        """
        if config is None:
            config = CCOConfig()
        
        rng = np.random.default_rng(rng_seed if rng_seed is not None else config.seed)
        
        network = VascularNetwork(domain=domain, seed=rng_seed)
        
        arterial_inlet_point = Point3D.from_array(arterial_inlet)
        arterial_inlet_node = Node(
            id=network.id_gen.next_id(),
            position=arterial_inlet_point,
            node_type="inlet",
            vessel_type="arterial",
            attributes={"radius": arterial_radius, "branch_order": 0},
        )
        network.add_node(arterial_inlet_node)
        
        first_arterial = self._sample_outlet_point(domain, rng, config)
        self._add_initial_segment(network, arterial_inlet_node, first_arterial, arterial_radius, "arterial")
        
        arterial_tree_view = ArrayTreeView(network, arterial_inlet_node.id)
        
        for i in range(arterial_outlets - 1):
            outlet_point = self._sample_outlet_point(domain, rng, config)
            if self._is_valid_outlet(outlet_point, network, config, vessel_type="arterial"):
                self._insert_outlet(
                    network, arterial_tree_view, outlet_point,
                    arterial_radius, "arterial", config, rng
                )
        
        venous_outlet_point = Point3D.from_array(venous_outlet)
        venous_outlet_node = Node(
            id=network.id_gen.next_id(),
            position=venous_outlet_point,
            node_type="outlet",
            vessel_type="venous",
            attributes={"radius": venous_radius, "branch_order": 0},
        )
        network.add_node(venous_outlet_node)
        
        first_venous = self._sample_outlet_point(domain, rng, config)
        self._add_initial_segment(network, venous_outlet_node, first_venous, venous_radius, "venous")
        
        venous_tree_view = ArrayTreeView(network, venous_outlet_node.id)
        
        for i in range(venous_outlets - 1):
            outlet_point = self._sample_outlet_point(domain, rng, config)
            if self._is_valid_outlet(outlet_point, network, config, vessel_type="venous"):
                self._insert_outlet(
                    network, venous_tree_view, outlet_point,
                    venous_radius, "venous", config, rng
                )
        
        self._rescale_radii(network, arterial_inlet_node.id, arterial_radius, config)
        self._rescale_radii(network, venous_outlet_node.id, venous_radius, config)
        
        if create_anastomoses and num_anastomoses > 0:
            self._create_anastomoses(network, num_anastomoses, config)
        
        return network
    
    def _sample_outlet_point(
        self,
        domain: DomainSpec,
        rng: np.random.Generator,
        config: CCOConfig,
    ) -> Point3D:
        """Sample a random outlet point inside the domain."""
        points = domain.sample_points(1, seed=int(rng.integers(0, 2**31)))
        return Point3D.from_array(points[0])
    
    def _is_valid_outlet(
        self,
        point: Point3D,
        network: VascularNetwork,
        config: CCOConfig,
        vessel_type: Optional[str] = None,
    ) -> bool:
        """
        Check if outlet point satisfies constraints.
        
        Supports vessel-type-aware validation for dual-tree generation:
        - Enforces min separation against same vessel_type terminals
        - Optionally enforces different (smaller) separation across A-V
        - Can optionally encourage A-V proximity for better perfusion
        
        Parameters
        ----------
        point : Point3D
            Proposed outlet point
        network : VascularNetwork
            Current network
        config : CCOConfig
            Configuration with separation thresholds
        vessel_type : str, optional
            Vessel type of the proposed outlet ("arterial" or "venous").
            If None, uses legacy behavior (check against all terminals).
        """
        if not network.domain.contains(point):
            return False
        
        # Determine separation thresholds
        same_type_sep = config.min_terminal_separation_same_type
        if same_type_sep is None:
            same_type_sep = config.min_terminal_separation
        
        cross_type_sep = config.min_terminal_separation_cross_type
        # If cross_type_sep is None, we don't enforce cross-type separation
        
        for node in network.nodes.values():
            if node.node_type == "terminal":
                dist = point.distance_to(node.position)
                
                if vessel_type is None:
                    # Legacy behavior: check against all terminals
                    if dist < config.min_terminal_separation:
                        return False
                else:
                    # Vessel-type-aware validation
                    if node.vessel_type == vessel_type:
                        # Same vessel type: enforce same_type_sep
                        if dist < same_type_sep:
                            return False
                    else:
                        # Different vessel type (A-V): enforce cross_type_sep if set
                        if cross_type_sep is not None and dist < cross_type_sep:
                            return False
        
        return True
    
    def _add_initial_segment(
        self,
        network: VascularNetwork,
        inlet_node: Node,
        outlet_point: Point3D,
        radius: float,
        vessel_type: str,
    ) -> None:
        """Add the first segment from inlet to first outlet."""
        outlet_node = Node(
            id=network.id_gen.next_id(),
            position=outlet_point,
            node_type="terminal",
            vessel_type=vessel_type,
            attributes={"radius": radius * 0.8, "branch_order": 1},
        )
        network.add_node(outlet_node)
        
        geometry = TubeGeometry(
            start=inlet_node.position,
            end=outlet_point,
            radius_start=radius,
            radius_end=radius * 0.8,
        )
        
        segment = VesselSegment(
            id=network.id_gen.next_id(),
            start_node_id=inlet_node.id,
            end_node_id=outlet_node.id,
            geometry=geometry,
            vessel_type=vessel_type,
        )
        network.add_segment(segment)
    
    def _insert_outlet(
        self,
        network: VascularNetwork,
        tree_view: ArrayTreeView,
        outlet_point: Point3D,
        inlet_radius: float,
        vessel_type: str,
        config: CCOConfig,
        rng: np.random.Generator,
    ) -> bool:
        """
        Insert a new outlet into the tree by splitting an existing edge.
        
        Uses partial binding optimization for efficient cost evaluation.
        """
        candidate_edges = self._select_candidate_edges(
            network, tree_view, outlet_point, config
        )
        
        if not candidate_edges:
            return False
        
        best_edge = None
        best_cost = float('inf')
        best_bifurcation = None
        
        for seg_id in candidate_edges:
            bifurcation, cost = self._optimize_bifurcation_point(
                network, tree_view, seg_id, outlet_point, config
            )
            
            if cost < best_cost:
                best_cost = cost
                best_edge = seg_id
                best_bifurcation = bifurcation
        
        if best_edge is None or best_bifurcation is None:
            return False
        
        self._perform_insertion(
            network, tree_view, best_edge, best_bifurcation,
            outlet_point, vessel_type, config
        )
        
        return True
    
    def _select_candidate_edges(
        self,
        network: VascularNetwork,
        tree_view: ArrayTreeView,
        outlet_point: Point3D,
        config: CCOConfig,
    ) -> List[int]:
        """
        Select candidate edges for insertion using collision triage.
        
        Uses cheap spatial filter before expensive evaluation.
        """
        if config.use_collision_triage:
            spatial_index = network.get_spatial_index()
            search_radius = 0.05
            nearby_segments = spatial_index.query_nearby_segments(outlet_point, search_radius)
            
            if nearby_segments:
                candidates = [seg.id for seg in nearby_segments]
            else:
                candidates = list(network.segments.keys())
        else:
            candidates = list(network.segments.keys())
        
        scored_candidates = []
        for seg_id in candidates:
            seg = network.segments[seg_id]
            start_node = network.nodes[seg.start_node_id]
            end_node = network.nodes[seg.end_node_id]
            
            midpoint = Point3D(
                (start_node.position.x + end_node.position.x) / 2,
                (start_node.position.y + end_node.position.y) / 2,
                (start_node.position.z + end_node.position.z) / 2,
            )
            dist = outlet_point.distance_to(midpoint)
            scored_candidates.append((seg_id, dist))
        
        scored_candidates.sort(key=lambda x: x[1])
        return [seg_id for seg_id, _ in scored_candidates[:config.candidate_edges_k]]
    
    def _optimize_bifurcation_point(
        self,
        network: VascularNetwork,
        tree_view: ArrayTreeView,
        seg_id: int,
        outlet_point: Point3D,
        config: CCOConfig,
    ) -> Tuple[Point3D, float]:
        """
        Optimize bifurcation point for inserting outlet into edge.
        
        Constrains bifurcation to the plane of triangle (A, B, T) where
        A and B are edge endpoints and T is the outlet point.
        
        Uses partial binding cache for efficient cost evaluation.
        """
        seg = network.segments[seg_id]
        A = network.nodes[seg.start_node_id].position.to_array()
        B = network.nodes[seg.end_node_id].position.to_array()
        T = outlet_point.to_array()
        
        AB = B - A
        AT = T - A
        
        normal = np.cross(AB, AT)
        normal_len = np.linalg.norm(normal)
        
        if normal_len < 1e-10:
            t_opt = 0.5
            X_opt = A + t_opt * AB
            return Point3D.from_array(X_opt), self._compute_insertion_cost(
                network, tree_view, seg_id, Point3D.from_array(X_opt), outlet_point, config
            )
        
        best_X = None
        best_cost = float('inf')
        
        n_grid = config.optimization_grid_resolution
        for i in range(n_grid + 1):
            t = i / n_grid
            X_on_AB = A + t * AB
            
            for j in range(n_grid + 1):
                s = j / n_grid
                X = X_on_AB + s * (T - X_on_AB)
                
                X_point = Point3D.from_array(X)
                
                if not network.domain.contains(X_point):
                    continue
                
                cost = self._compute_insertion_cost(
                    network, tree_view, seg_id, X_point, outlet_point, config
                )
                
                if cost < best_cost:
                    best_cost = cost
                    best_X = X_point
        
        if best_X is None:
            t_opt = 0.5
            best_X = Point3D.from_array(A + t_opt * AB)
            best_cost = self._compute_insertion_cost(
                network, tree_view, seg_id, best_X, outlet_point, config
            )
        
        return best_X, best_cost
    
    def _compute_insertion_cost(
        self,
        network: VascularNetwork,
        tree_view: ArrayTreeView,
        seg_id: int,
        bifurcation_point: Point3D,
        outlet_point: Point3D,
        config: CCOConfig,
    ) -> float:
        """
        Compute cost of inserting outlet at bifurcation point.
        
        Cost = sum over affected segments of (radius^alpha * length)
        Plus penalty for domain boundary proximity.
        
        Also performs collision checks if config.collision_check_enabled is True.
        Returns inf if the proposed insertion would cause collisions.
        """
        seg = network.segments[seg_id]
        A = network.nodes[seg.start_node_id].position
        B = network.nodes[seg.end_node_id].position
        X = bifurcation_point
        T = outlet_point
        
        len_AX = A.distance_to(X)
        len_XB = X.distance_to(B)
        len_XT = X.distance_to(T)
        
        if len_AX < config.min_segment_length or len_XB < config.min_segment_length:
            return float('inf')
        if len_XT < config.min_segment_length:
            return float('inf')
        
        r_parent = seg.geometry.mean_radius()
        
        # Demand-based Murray splitting: use subtree terminal counts as demand proxy
        # r_i = r_parent * (f_i)^(1/gamma) where f_i = demand_i / sum(demand)
        gamma = config.murray_exponent
        
        # Get subtree terminal count for the segment being split
        seg_record = tree_view.segments.get(seg_id)
        existing_demand = seg_record.subtree_terminal_count if seg_record else 1
        new_demand = 1  # New outlet adds 1 terminal
        total_demand = existing_demand + new_demand
        
        if total_demand > 0:
            f_existing = existing_demand / total_demand
            f_new = new_demand / total_demand
            r_child1 = r_parent * (f_existing ** (1.0 / gamma))  # Existing subtree
            r_child2 = r_parent * (f_new ** (1.0 / gamma))  # New outlet
        else:
            # Fallback to equal split
            r_child1 = r_parent * 0.8
            r_child2 = r_parent * 0.8
        
        # Collision check: verify proposed new segments don't collide with existing segments
        if config.collision_check_enabled:
            if self._check_insertion_collision(
                network, seg_id, X, T, r_child1, r_child2, config.collision_clearance
            ):
                return float('inf')
        
        alpha = config.murray_exponent
        
        cost = (
            config.cost_radius_weight * (r_child1**alpha * len_XB + r_child2**alpha * len_XT) +
            config.cost_length_weight * (len_AX + len_XB + len_XT)
        )
        
        boundary_dist = network.domain.distance_to_boundary(X)
        if boundary_dist < 0.002:
            cost += config.boundary_penalty_weight * (0.002 - boundary_dist)
        
        return cost
    
    def _check_insertion_collision(
        self,
        network: VascularNetwork,
        split_seg_id: int,
        bifurcation_point: Point3D,
        outlet_point: Point3D,
        r_child1: float,
        r_child2: float,
        clearance: float,
    ) -> bool:
        """
        Check if proposed insertion would cause collisions with existing segments.
        
        P0-7: Tests all three new segments (A→X, X→B, X→T) for collisions,
        not assuming segment orientation.
        
        Uses 2-stage method:
        1. Cheap spatial query to find nearby segments
        2. Exact segment-segment distance check for candidates
        
        Parameters
        ----------
        network : VascularNetwork
            Current network
        split_seg_id : int
            ID of segment being split (excluded from collision check)
        bifurcation_point : Point3D
            Proposed bifurcation point X
        outlet_point : Point3D
            Target outlet point T
        r_child1 : float
            Radius of child segment X→B
        r_child2 : float
            Radius of new outlet segment X→T
        clearance : float
            Minimum required clearance between vessels
            
        Returns
        -------
        bool
            True if collision detected, False if safe
        """
        from ..spatial.grid_index import segment_segment_distance_exact
        
        X = bifurcation_point.to_array()
        T = outlet_point.to_array()
        
        # P0-7: Define endpoints from geometry, not assuming orientation
        split_seg = network.segments[split_seg_id]
        A = network.nodes[split_seg.start_node_id].position.to_array()
        B = network.nodes[split_seg.end_node_id].position.to_array()
        
        # Estimate radius for A→X segment (interpolated from parent)
        t = np.linalg.norm(X - A) / max(np.linalg.norm(B - A), 1e-10)
        r_at_X = split_seg.geometry.radius_start + t * (split_seg.geometry.radius_end - split_seg.geometry.radius_start)
        r_AX = (split_seg.geometry.radius_start + r_at_X) / 2  # Average radius for A→X
        
        # Stage 1: Cheap spatial query to find nearby segments
        spatial_index = network.get_spatial_index()
        
        # Search radius should cover all three new segments plus clearance
        search_radius = max(
            np.linalg.norm(X - A),
            np.linalg.norm(X - B),
            np.linalg.norm(X - T),
        ) + clearance + max(r_child1, r_child2, r_AX) * 2
        
        nearby_segments = spatial_index.query_nearby_segments(bifurcation_point, search_radius)
        
        # Stage 2: Exact collision check for candidates
        for seg in nearby_segments:
            # Skip the segment being split and its connected segments
            if seg.id == split_seg_id:
                continue
            if (seg.start_node_id == split_seg.start_node_id or
                seg.start_node_id == split_seg.end_node_id or
                seg.end_node_id == split_seg.start_node_id or
                seg.end_node_id == split_seg.end_node_id):
                continue
            
            seg_radius = seg.geometry.mean_radius()
            
            # P0-NEW-4 / P2-NEW-2: Build polyline for existing segment if it has centerline_points
            if seg.geometry.centerline_points:
                seg_start_node = network.nodes[seg.start_node_id]
                seg_end_node = network.nodes[seg.end_node_id]
                seg_polyline = [seg_start_node.position.to_array()]
                seg_polyline.extend([p.to_array() for p in seg.geometry.centerline_points])
                seg_polyline.append(seg_end_node.position.to_array())
                
                # Check collision with proposed A→X segment against polyline
                dist_AX = self._segment_to_polyline_distance(A, X, seg_polyline)
                required_clearance_AX = r_AX + seg_radius + clearance
                if dist_AX < required_clearance_AX:
                    return True
                
                # Check collision with proposed X→B segment against polyline
                dist_XB = self._segment_to_polyline_distance(X, B, seg_polyline)
                required_clearance_XB = r_child1 + seg_radius + clearance
                if dist_XB < required_clearance_XB:
                    return True
                
                # Check collision with proposed X→T segment against polyline
                dist_XT = self._segment_to_polyline_distance(X, T, seg_polyline)
                required_clearance_XT = r_child2 + seg_radius + clearance
                if dist_XT < required_clearance_XT:
                    return True
            else:
                # Simple straight segment - use fast path
                seg_start = network.nodes[seg.start_node_id].position.to_array()
                seg_end = network.nodes[seg.end_node_id].position.to_array()
                
                # P0-7: Check collision with proposed A→X segment
                dist_AX = segment_segment_distance_exact(A, X, seg_start, seg_end)
                required_clearance_AX = r_AX + seg_radius + clearance
                if dist_AX < required_clearance_AX:
                    return True
                
                # Check collision with proposed X→B segment
                dist_XB = segment_segment_distance_exact(X, B, seg_start, seg_end)
                required_clearance_XB = r_child1 + seg_radius + clearance
                if dist_XB < required_clearance_XB:
                    return True
                
                # Check collision with proposed X→T segment
                dist_XT = segment_segment_distance_exact(X, T, seg_start, seg_end)
                required_clearance_XT = r_child2 + seg_radius + clearance
                if dist_XT < required_clearance_XT:
                    return True
        
        return False
    
    def _segment_to_polyline_distance(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        polyline: List[np.ndarray],
    ) -> float:
        """
        Compute minimum distance from a straight segment to a polyline.
        
        P0-NEW-4 / P2-NEW-2: Helper for collision checking against polyline segments.
        
        Parameters
        ----------
        p1, p2 : np.ndarray
            Endpoints of the straight segment
        polyline : list of np.ndarray
            List of polyline vertices
            
        Returns
        -------
        float
            Minimum distance between the segment and polyline
        """
        from ..spatial.grid_index import segment_segment_distance_exact
        
        min_dist = float('inf')
        for i in range(len(polyline) - 1):
            dist = segment_segment_distance_exact(p1, p2, polyline[i], polyline[i + 1])
            min_dist = min(min_dist, dist)
        return min_dist
    
    def _perform_insertion(
        self,
        network: VascularNetwork,
        tree_view: ArrayTreeView,
        seg_id: int,
        bifurcation_point: Point3D,
        outlet_point: Point3D,
        vessel_type: str,
        config: CCOConfig,
    ) -> None:
        """
        Perform the actual insertion by splitting edge and adding new outlet.
        
        P1-1: Uses local radius-at-split computed via linear interpolation
        instead of mean radius for correct Murray behavior.
        """
        seg = network.segments[seg_id]
        start_node = network.nodes[seg.start_node_id]
        end_node = network.nodes[seg.end_node_id]
        
        # P1-1: Compute local radius at split point using linear interpolation
        # t = |A→X| / |A→B|, rX = lerp(r_start, r_end, t)
        A = start_node.position.to_array()
        B = end_node.position.to_array()
        X = bifurcation_point.to_array()
        
        seg_length = np.linalg.norm(B - A)
        if seg_length > 1e-10:
            t = np.linalg.norm(X - A) / seg_length
        else:
            t = 0.5
        
        # Linear interpolation for radius at split point
        r_at_split = seg.geometry.radius_start + t * (seg.geometry.radius_end - seg.geometry.radius_start)
        
        bifurcation_node = Node(
            id=network.id_gen.next_id(),
            position=bifurcation_point,
            node_type="junction",
            vessel_type=vessel_type,
            attributes={
                "radius": r_at_split,
                "branch_order": start_node.attributes.get("branch_order", 0) + 1,
            },
        )
        network.add_node(bifurcation_node)
        
        # P1-1: Use local radius at split for Murray calculations
        r_parent = r_at_split
        
        # Demand-based Murray splitting: use subtree terminal counts as demand proxy
        gamma = config.murray_exponent
        seg_record = tree_view.segments.get(seg_id)
        existing_demand = seg_record.subtree_terminal_count if seg_record else 1
        new_demand = 1  # New outlet adds 1 terminal
        total_demand = existing_demand + new_demand
        
        if total_demand > 0:
            f_existing = existing_demand / total_demand
            f_new = new_demand / total_demand
            r_child_existing = r_parent * (f_existing ** (1.0 / gamma))  # Existing subtree
            r_child_new = r_parent * (f_new ** (1.0 / gamma))  # New outlet
        else:
            # Fallback to equal split
            r_child_existing = r_parent * 0.8
            r_child_new = r_parent * 0.8
        
        outlet_node = Node(
            id=network.id_gen.next_id(),
            position=outlet_point,
            node_type="terminal",
            vessel_type=vessel_type,
            attributes={
                "radius": r_child_new,
                "branch_order": bifurcation_node.attributes.get("branch_order", 0) + 1,
            },
        )
        network.add_node(outlet_node)
        
        seg1_geometry = TubeGeometry(
            start=start_node.position,
            end=bifurcation_point,
            radius_start=seg.geometry.radius_start,
            radius_end=r_parent,
        )
        seg1 = VesselSegment(
            id=network.id_gen.next_id(),
            start_node_id=start_node.id,
            end_node_id=bifurcation_node.id,
            geometry=seg1_geometry,
            vessel_type=vessel_type,
        )
        
        seg2_geometry = TubeGeometry(
            start=bifurcation_point,
            end=end_node.position,
            radius_start=r_child_existing,
            radius_end=seg.geometry.radius_end,
        )
        seg2 = VesselSegment(
            id=network.id_gen.next_id(),
            start_node_id=bifurcation_node.id,
            end_node_id=end_node.id,
            geometry=seg2_geometry,
            vessel_type=vessel_type,
        )
        
        seg3_geometry = TubeGeometry(
            start=bifurcation_point,
            end=outlet_point,
            radius_start=r_child_new,
            radius_end=r_child_new * 0.9,
        )
        seg3 = VesselSegment(
            id=network.id_gen.next_id(),
            start_node_id=bifurcation_node.id,
            end_node_id=outlet_node.id,
            geometry=seg3_geometry,
            vessel_type=vessel_type,
        )
        
        network.remove_segment(seg_id)
        network.add_segment(seg1)
        network.add_segment(seg2)
        network.add_segment(seg3)
        
        tree_view.update_after_insertion(seg_id, [seg1.id, seg2.id, seg3.id], bifurcation_node.id)
    
    def _rescale_radii(
        self,
        network: VascularNetwork,
        root_node_id: int,
        root_radius: float,
        config: CCOConfig,
    ) -> None:
        """
        Rescale radii from root using Murray's law with demand-based splitting.
        
        P1-2: Uses terminal counts as demand proxy for Murray splitting instead
        of equal splitting across children. This produces more physiologically
        realistic radii where branches with more terminals get larger radii.
        
        For each branching node:
        - demand(child) = terminal count in child subtree
        - f_i = demand_i / sum(demands)
        - r_i = r_parent * (f_i)^(1/gamma)
        
        This satisfies Murray's law: r_parent^gamma = sum(r_i^gamma)
        """
        gamma = config.murray_exponent
        
        def count_terminals(node_id: int, visited: Set[int]) -> int:
            if node_id in visited:
                return 0
            visited.add(node_id)
            
            node = network.nodes[node_id]
            if node.node_type == "terminal":
                return 1
            
            total = 0
            for seg_id in network.get_connected_segment_ids(node_id):
                seg = network.segments[seg_id]
                other_id = seg.end_node_id if seg.start_node_id == node_id else seg.start_node_id
                total += count_terminals(other_id, visited)
            
            return total
        
        total_terminals = count_terminals(root_node_id, set())
        if total_terminals == 0:
            return
        
        def rescale_subtree(node_id: int, parent_radius: float, visited: Set[int]) -> None:
            if node_id in visited:
                return
            visited.add(node_id)
            
            connected_segs = network.get_connected_segment_ids(node_id)
            child_segs = []
            
            for seg_id in connected_segs:
                seg = network.segments[seg_id]
                other_id = seg.end_node_id if seg.start_node_id == node_id else seg.start_node_id
                if other_id not in visited:
                    child_segs.append((seg_id, other_id))
            
            if not child_segs:
                return
            
            n_children = len(child_segs)
            if n_children == 1:
                # Single child: slight taper
                child_radius = parent_radius * 0.9
                seg_id, child_node_id = child_segs[0]
                seg = network.segments[seg_id]
                
                if seg.start_node_id == node_id:
                    seg.geometry.radius_start = parent_radius
                    seg.geometry.radius_end = child_radius
                else:
                    seg.geometry.radius_start = child_radius
                    seg.geometry.radius_end = parent_radius
                
                network.nodes[child_node_id].attributes["radius"] = child_radius
                rescale_subtree(child_node_id, child_radius, visited)
            else:
                # P1-2: Demand-based Murray splitting for multiple children
                # Compute demand (terminal count) for each child subtree
                child_demands = []
                for seg_id, child_node_id in child_segs:
                    # Count terminals in this child's subtree
                    demand = count_terminals(child_node_id, visited.copy())
                    child_demands.append((seg_id, child_node_id, max(demand, 1)))
                
                total_demand = sum(d for _, _, d in child_demands)
                
                for seg_id, child_node_id, demand in child_demands:
                    # f_i = demand_i / total_demand
                    f_i = demand / total_demand
                    # r_i = r_parent * (f_i)^(1/gamma)
                    child_radius = parent_radius * (f_i ** (1.0 / gamma))
                    
                    seg = network.segments[seg_id]
                    if seg.start_node_id == node_id:
                        seg.geometry.radius_start = parent_radius
                        seg.geometry.radius_end = child_radius
                    else:
                        seg.geometry.radius_start = child_radius
                        seg.geometry.radius_end = parent_radius
                    
                    network.nodes[child_node_id].attributes["radius"] = child_radius
                    rescale_subtree(child_node_id, child_radius, visited)
        
        network.nodes[root_node_id].attributes["radius"] = root_radius
        rescale_subtree(root_node_id, root_radius, set())
    
    def _create_anastomoses(
        self,
        network: VascularNetwork,
        num_anastomoses: int,
        config: CCOConfig,
    ) -> None:
        """
        Create anastomoses between arterial and venous terminals.
        """
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
