"""
K-ary tree generation backend using recursive bifurcation.

.. deprecated::
    This backend is **deprecated** and will be removed in a future release.
    Use ``scaffold_topdown`` instead, which provides better collision avoidance
    and more flexible tree generation.

STATUS: DEPRECATED
REPLACEMENT: scaffold_topdown
EXPECTED REMOVAL: Future release

This backend generates vascular networks by recursively bifurcating
from inlet nodes to achieve a target terminal count.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List, Union
import numpy as np
import logging

from .base import GenerationBackend, BackendConfig, GenerationState
from ..core.network import VascularNetwork, Node, VesselSegment
from ..core.domain import DomainSpec
from ..core.types import Point3D, TubeGeometry

logger = logging.getLogger(__name__)


@dataclass
class KaryTreeConfig(BackendConfig):
    """
    Configuration for k-ary tree generation.
    
    Parameters
    ----------
    k : int
        Branching factor (default: 2 for binary tree)
    target_terminals : int
        Target number of terminal nodes
    terminal_tolerance : float
        Acceptable deviation from target (fraction, default: 0.1 = 10%)
    branch_length : float
        Initial branch length in meters. If None and tree_extent_fraction is set,
        branch_length will be computed from domain size.
    branch_length_decay : float
        Factor to multiply branch length at each level
    taper_factor : float
        Radius taper factor at each bifurcation (Murray's law: 0.794)
    angle_deg : float
        Bifurcation angle in degrees
    angle_variation_deg : float
        Random variation in bifurcation angle
    min_radius : float
        Minimum vessel radius in meters
    tree_extent_fraction : float
        Fraction of domain characteristic size to use for total tree extent.
        If set, branch_length is computed as:
        branch_length = (domain_size * tree_extent_fraction) / sum(decay^i for i in 0..depth-1)
        Default: 0.4 (tree fills ~40% of domain)
    use_domain_scaling : bool
        If True and branch_length is None, automatically compute branch_length
        from domain size using tree_extent_fraction. Default: True.
    primary_axis : tuple or None
        Primary growth direction as (x, y, z) unit vector. If None, defaults to
        the inward direction from the inlet (typically -Z for top-face inlets).
        This is the "growth_inward_direction" from port semantics.
    max_deviation_deg : float
        Maximum deviation angle from primary_axis in degrees. Child branches
        will be constrained to stay within this cone. Default: 90.0 (no constraint).
    upward_forbidden : bool
        If True, forbid any growth direction with positive Z component (dot(dir, +Z) > 0).
        This prevents tree from growing back toward the inlet face. Default: False.
    azimuth_jitter_deg : float
        Random jitter in azimuth angle (rotation around primary axis) in degrees.
        Default: 180.0 (full rotation allowed).
    elevation_jitter_deg : float
        Random jitter in elevation angle (angle from primary axis) in degrees.
        Default: same as angle_deg.
    wall_margin : float
        Minimum distance from domain boundary for internal nodes in meters.
        Default: 0.0 (no margin).
    multi_inlet_mode : str
        Mode for handling multiple inlets: "forest" (separate trees per inlet),
        "merge_to_trunk" (connect inlets to common trunk), or "forest_with_merge"
        (grow separate trees then merge where they collide). Default: "merge_to_trunk".
    collision_merge_distance : float
        Distance threshold for merging trees in forest_with_merge mode.
        Trees closer than this distance will be connected. Default: 0.0003 (0.3mm).
    trunk_depth_fraction : float
        Fraction of domain height for trunk junction point (merge_to_trunk mode).
        Default: 0.2 (20% into domain from inlet face).
    trunk_merge_radius : float or None
        Radius at trunk junction point. If None, auto-computed from inlet radii.
    max_inlets : int
        Maximum number of inlets supported. Default: 10.
    """
    k: int = 2
    target_terminals: int = 128
    terminal_tolerance: float = 0.1
    branch_length: Optional[float] = None
    branch_length_decay: float = 0.8
    taper_factor: float = 0.794  # Murray's law: 2^(-1/3)
    angle_deg: float = 30.0
    angle_variation_deg: float = 5.0
    min_radius: float = 0.0001  # 0.1mm
    tree_extent_fraction: float = 0.4
    use_domain_scaling: bool = True
    primary_axis: Optional[Tuple[float, float, float]] = None
    max_deviation_deg: float = 90.0
    upward_forbidden: bool = False
    azimuth_jitter_deg: float = 180.0
    elevation_jitter_deg: Optional[float] = None
    wall_margin: float = 0.0
    multi_inlet_mode: str = "merge_to_trunk"
    collision_merge_distance: float = 0.0003  # 0.3mm
    trunk_depth_fraction: float = 0.2
    trunk_merge_radius: Optional[float] = None
    max_inlets: int = 10


class KaryTreeBackend(GenerationBackend):
    """
    Generation backend using recursive k-ary bifurcation.
    
    This backend creates tree structures by recursively splitting
    branches until a target terminal count is reached.
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
        config: Optional[BackendConfig] = None,
        rng_seed: Optional[int] = None,
    ) -> VascularNetwork:
        """
        Generate a k-ary tree vascular network.
        
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
        config : KaryTreeConfig, optional
            Backend configuration
        rng_seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        VascularNetwork
            Generated vascular network
        """
        if config is None:
            config = KaryTreeConfig(target_terminals=num_outlets)
        elif not isinstance(config, KaryTreeConfig):
            # Convert BackendConfig to KaryTreeConfig
            config = KaryTreeConfig(
                target_terminals=num_outlets,
                seed=config.seed,
                min_segment_length=config.min_segment_length,
                max_segment_length=config.max_segment_length,
                min_radius=config.min_radius,
                check_collisions=config.check_collisions,
                collision_clearance=config.collision_clearance,
            )
        
        # D FIX: Use local RNG instead of global np.random.seed
        effective_seed = rng_seed if rng_seed is not None else config.seed
        rng = np.random.default_rng(effective_seed)
        
        # Create network
        network = VascularNetwork(domain=domain)
        
        # D FIX: Add inlet node using network.id_gen and network.add_node
        inlet_pos = Point3D(*inlet_position)
        inlet_id = network.id_gen.next_id()
        inlet_node = Node(
            id=inlet_id,
            position=inlet_pos,
            node_type="inlet",
            vessel_type=vessel_type,
            attributes={"radius": inlet_radius},
        )
        network.add_node(inlet_node)
        
        # Calculate depth needed for target terminals
        # For k-ary tree: terminals = k^depth
        depth = self._calculate_depth(config.k, config.target_terminals)
        
        # Compute branch length from domain size if not explicitly set
        branch_length = config.branch_length
        if branch_length is None and config.use_domain_scaling:
            domain_size = self._get_domain_characteristic_size(domain)
            # Total tree extent = sum of branch lengths at each level
            # = branch_length * sum(decay^i for i in 0..depth-1)
            # = branch_length * (1 - decay^depth) / (1 - decay)
            decay = config.branch_length_decay
            if decay < 1.0 and depth > 0:
                geometric_sum = (1 - decay**depth) / (1 - decay)
            else:
                geometric_sum = depth if depth > 0 else 1
            
            target_extent = domain_size * config.tree_extent_fraction
            branch_length = target_extent / geometric_sum
            
            logger.info(
                f"Domain-aware scaling: domain_size={domain_size*1000:.2f}mm, "
                f"tree_extent_fraction={config.tree_extent_fraction}, "
                f"computed branch_length={branch_length*1000:.3f}mm"
            )
        elif branch_length is None:
            # Fallback to default 1mm
            branch_length = 0.001
            logger.warning(
                "branch_length is None and use_domain_scaling is False, "
                "using default 1mm branch length"
            )
        
        logger.info(
            f"Generating k={config.k} tree with depth={depth} "
            f"for target_terminals={config.target_terminals}, "
            f"branch_length={branch_length*1000:.3f}mm"
        )
        
        # Determine primary growth direction
        if config.primary_axis is not None:
            primary_direction = np.array(config.primary_axis)
            norm = np.linalg.norm(primary_direction)
            if norm > 0:
                primary_direction = primary_direction / norm
            else:
                primary_direction = np.array([0.0, 0.0, -1.0])
        else:
            primary_direction = np.array([0.0, 0.0, -1.0])
        
        logger.info(
            f"Primary growth direction: {primary_direction}, "
            f"upward_forbidden={config.upward_forbidden}, "
            f"max_deviation_deg={config.max_deviation_deg}"
        )
        
        # D FIX: Generate tree recursively with local RNG
        self._generate_subtree(
            network=network,
            parent_node=inlet_node,
            parent_direction=primary_direction,
            primary_axis=primary_direction,
            current_depth=0,
            max_depth=depth,
            current_radius=inlet_radius,
            current_length=branch_length,
            config=config,
            vessel_type=vessel_type,
            domain=domain,
            rng=rng,
        )
        
        # Count terminals and check tolerance
        terminal_count = sum(1 for n in network.nodes.values() if n.node_type == "terminal")
        target = config.target_terminals
        tolerance = config.terminal_tolerance
        
        if abs(terminal_count - target) / max(target, 1) > tolerance:
            logger.warning(
                f"Terminal count {terminal_count} outside tolerance "
                f"({tolerance*100:.0f}%) of target {target}"
            )
        
        return network
    
    def generate_multi_inlet(
        self,
        domain: DomainSpec,
        num_outlets: int,
        inlets: List[Dict[str, Any]],
        vessel_type: str = "arterial",
        config: Optional[BackendConfig] = None,
        rng_seed: Optional[int] = None,
    ) -> VascularNetwork:
        """
        Generate a k-ary tree vascular network with multiple inlets.
        
        Parameters
        ----------
        domain : DomainSpec
            Geometric domain for the network
        num_outlets : int
            Target number of terminal outlets (total across all inlets)
        inlets : list of dict
            List of inlet specifications, each with:
            - position: [x, y, z] in meters
            - radius: float in meters
            - direction: [x, y, z] optional, growth direction (inward)
            - is_surface_opening: bool optional
        vessel_type : str
            Type of vessels ("arterial" or "venous")
        config : KaryTreeConfig, optional
            Backend configuration
        rng_seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        VascularNetwork
            Generated vascular network with multiple inlet trees
        """
        if config is None:
            config = KaryTreeConfig(target_terminals=num_outlets)
        elif not isinstance(config, KaryTreeConfig):
            config = KaryTreeConfig(
                target_terminals=num_outlets,
                seed=config.seed,
                min_segment_length=config.min_segment_length,
                max_segment_length=config.max_segment_length,
                min_radius=config.min_radius,
                check_collisions=config.check_collisions,
                collision_clearance=config.collision_clearance,
            )
        
        if len(inlets) > config.max_inlets:
            logger.warning(
                f"Number of inlets ({len(inlets)}) exceeds max_inlets ({config.max_inlets}), "
                f"truncating to {config.max_inlets}"
            )
            inlets = inlets[:config.max_inlets]
        
        if len(inlets) == 0:
            raise ValueError("At least one inlet is required")
        
        if len(inlets) == 1:
            inlet = inlets[0]
            inlet_position = np.array(inlet.get("position", [0, 0, 0]))
            inlet_radius = inlet.get("radius", 0.001)
            direction = inlet.get("direction", inlet.get("growth_inward_direction"))
            if direction is not None:
                config_dict = {k: getattr(config, k) for k in config.__dataclass_fields__}
                config_dict["primary_axis"] = tuple(direction)
                config = KaryTreeConfig(**config_dict)
            return self.generate(
                domain=domain,
                num_outlets=num_outlets,
                inlet_position=inlet_position,
                inlet_radius=inlet_radius,
                vessel_type=vessel_type,
                config=config,
                rng_seed=rng_seed,
            )
        
        effective_seed = rng_seed if rng_seed is not None else config.seed
        rng = np.random.default_rng(effective_seed)
        
        network = VascularNetwork(domain=domain)
        
        if config.multi_inlet_mode == "forest":
            return self._generate_forest_mode(
                network=network,
                domain=domain,
                num_outlets=num_outlets,
                inlets=inlets,
                vessel_type=vessel_type,
                config=config,
                rng=rng,
            )
        elif config.multi_inlet_mode == "forest_with_merge":
            return self._generate_forest_with_merge_mode(
                network=network,
                domain=domain,
                num_outlets=num_outlets,
                inlets=inlets,
                vessel_type=vessel_type,
                config=config,
                rng=rng,
            )
        else:
            return self._generate_merge_to_trunk_mode(
                network=network,
                domain=domain,
                num_outlets=num_outlets,
                inlets=inlets,
                vessel_type=vessel_type,
                config=config,
                rng=rng,
            )
    
    def _generate_forest_mode(
        self,
        network: VascularNetwork,
        domain: DomainSpec,
        num_outlets: int,
        inlets: List[Dict[str, Any]],
        vessel_type: str,
        config: KaryTreeConfig,
        rng: np.random.Generator,
    ) -> VascularNetwork:
        """Generate separate trees per inlet (forest mode)."""
        n_inlets = len(inlets)
        outlets_per_inlet = num_outlets // n_inlets
        remainder = num_outlets % n_inlets
        
        for i, inlet in enumerate(inlets):
            inlet_position = np.array(inlet.get("position", [0, 0, 0]))
            inlet_radius = inlet.get("radius", 0.001)
            direction = inlet.get("direction", inlet.get("growth_inward_direction"))
            
            if direction is not None:
                primary_axis = np.array(direction)
                norm = np.linalg.norm(primary_axis)
                if norm > 0:
                    primary_axis = primary_axis / norm
                else:
                    primary_axis = np.array([0.0, 0.0, -1.0])
            else:
                primary_axis = np.array([0.0, 0.0, -1.0])
            
            this_outlets = outlets_per_inlet + (1 if i < remainder else 0)
            depth = self._calculate_depth(config.k, this_outlets)
            
            branch_length = config.branch_length
            if branch_length is None and config.use_domain_scaling:
                domain_size = self._get_domain_characteristic_size(domain)
                decay = config.branch_length_decay
                if decay < 1.0 and depth > 0:
                    geometric_sum = (1 - decay**depth) / (1 - decay)
                else:
                    geometric_sum = depth if depth > 0 else 1
                target_extent = domain_size * config.tree_extent_fraction
                branch_length = target_extent / geometric_sum
            elif branch_length is None:
                branch_length = 0.001
            
            inlet_pos = Point3D(*inlet_position)
            inlet_id = network.id_gen.next_id()
            inlet_node = Node(
                id=inlet_id,
                position=inlet_pos,
                node_type="inlet",
                vessel_type=vessel_type,
                attributes={"radius": inlet_radius, "inlet_index": i},
            )
            network.add_node(inlet_node)
            
            self._generate_subtree(
                network=network,
                parent_node=inlet_node,
                parent_direction=primary_axis,
                primary_axis=primary_axis,
                current_depth=0,
                max_depth=depth,
                current_radius=inlet_radius,
                current_length=branch_length,
                config=config,
                vessel_type=vessel_type,
                domain=domain,
                rng=rng,
            )
        
        logger.info(
            f"Forest mode: generated {n_inlets} separate trees with "
            f"{sum(1 for n in network.nodes.values() if n.node_type == 'terminal')} total terminals"
        )
        
        return network
    
    def _generate_forest_with_merge_mode(
        self,
        network: VascularNetwork,
        domain: DomainSpec,
        num_outlets: int,
        inlets: List[Dict[str, Any]],
        vessel_type: str,
        config: KaryTreeConfig,
        rng: np.random.Generator,
    ) -> VascularNetwork:
        """
        Generate separate trees per inlet, then merge where they collide.
        
        This mode first generates independent trees from each inlet (like forest mode),
        then detects where segments from different trees come close to each other
        and creates junction nodes to merge them into a connected network.
        """
        n_inlets = len(inlets)
        outlets_per_inlet = num_outlets // n_inlets
        remainder = num_outlets % n_inlets
        
        # Track which nodes belong to which inlet tree
        inlet_node_sets: List[set] = [set() for _ in range(n_inlets)]
        inlet_root_ids: List[int] = []
        
        # Generate separate trees for each inlet (same as forest mode)
        for i, inlet in enumerate(inlets):
            inlet_position = np.array(inlet.get("position", [0, 0, 0]))
            inlet_radius = inlet.get("radius", 0.001)
            direction = inlet.get("direction", inlet.get("growth_inward_direction"))
            
            if direction is not None:
                primary_axis = np.array(direction)
                norm = np.linalg.norm(primary_axis)
                if norm > 0:
                    primary_axis = primary_axis / norm
                else:
                    primary_axis = np.array([0.0, 0.0, -1.0])
            elif config.primary_axis is not None:
                primary_axis = np.array(config.primary_axis)
            else:
                primary_axis = np.array([0.0, 0.0, -1.0])
            
            this_inlet_outlets = outlets_per_inlet + (1 if i < remainder else 0)
            depth = self._calculate_depth(this_inlet_outlets, config.k)
            
            branch_length = config.branch_length
            if branch_length is None and config.use_domain_scaling:
                domain_size = self._get_domain_characteristic_size(domain)
                decay = config.branch_length_decay
                if decay < 1.0 and depth > 0:
                    geometric_sum = (1 - decay**depth) / (1 - decay)
                else:
                    geometric_sum = depth if depth > 0 else 1
                target_extent = domain_size * config.tree_extent_fraction
                branch_length = target_extent / geometric_sum
            elif branch_length is None:
                branch_length = 0.001
            
            # Track nodes before adding this tree
            nodes_before = set(network.nodes.keys())
            
            inlet_pos = Point3D(*inlet_position)
            inlet_id = network.id_gen.next_id()
            inlet_node = Node(
                id=inlet_id,
                position=inlet_pos,
                node_type="inlet",
                vessel_type=vessel_type,
                attributes={"radius": inlet_radius, "inlet_index": i},
            )
            network.add_node(inlet_node)
            inlet_root_ids.append(inlet_id)
            
            self._generate_subtree(
                network=network,
                parent_node=inlet_node,
                parent_direction=primary_axis,
                primary_axis=primary_axis,
                current_depth=0,
                max_depth=depth,
                current_radius=inlet_radius,
                current_length=branch_length,
                config=config,
                vessel_type=vessel_type,
                domain=domain,
                rng=rng,
            )
            
            # Track nodes belonging to this inlet tree
            nodes_after = set(network.nodes.keys())
            inlet_node_sets[i] = nodes_after - nodes_before
        
        # Detect collisions between trees and merge them
        merge_distance = config.collision_merge_distance
        merges_performed = self._merge_colliding_trees(
            network=network,
            inlet_node_sets=inlet_node_sets,
            merge_distance=merge_distance,
            vessel_type=vessel_type,
        )
        
        logger.info(
            f"Forest with merge mode: generated {n_inlets} trees, performed {merges_performed} merges, "
            f"{sum(1 for n in network.nodes.values() if n.node_type == 'terminal')} total terminals"
        )
        
        return network
    
    def _merge_colliding_trees(
        self,
        network: VascularNetwork,
        inlet_node_sets: List[set],
        merge_distance: float,
        vessel_type: str,
    ) -> int:
        """
        Find and merge segments from different trees that are close to each other.
        
        Returns the number of merges performed.
        """
        merges_performed = 0
        n_trees = len(inlet_node_sets)
        
        # Build a list of (node_id, tree_index, position) for efficient lookup
        node_tree_map: Dict[int, int] = {}
        for tree_idx, node_set in enumerate(inlet_node_sets):
            for node_id in node_set:
                node_tree_map[node_id] = tree_idx
        
        # Find pairs of nodes from different trees that are close
        merge_candidates: List[Tuple[int, int, float]] = []
        
        for tree_i in range(n_trees):
            for tree_j in range(tree_i + 1, n_trees):
                for node_id_i in inlet_node_sets[tree_i]:
                    node_i = network.nodes.get(node_id_i)
                    if node_i is None:
                        continue
                    pos_i = np.array([node_i.position.x, node_i.position.y, node_i.position.z])
                    
                    for node_id_j in inlet_node_sets[tree_j]:
                        node_j = network.nodes.get(node_id_j)
                        if node_j is None:
                            continue
                        pos_j = np.array([node_j.position.x, node_j.position.y, node_j.position.z])
                        
                        dist = float(np.linalg.norm(pos_i - pos_j))
                        if dist <= merge_distance:
                            merge_candidates.append((node_id_i, node_id_j, dist))
        
        # Sort by distance (closest first) and merge
        merge_candidates.sort(key=lambda x: x[2])
        
        # Track which trees have been merged (using union-find logic)
        merged_trees: Dict[int, int] = {i: i for i in range(n_trees)}
        
        def find_root(tree_idx: int) -> int:
            while merged_trees[tree_idx] != tree_idx:
                tree_idx = merged_trees[tree_idx]
            return tree_idx
        
        for node_id_i, node_id_j, dist in merge_candidates:
            tree_i = node_tree_map.get(node_id_i)
            tree_j = node_tree_map.get(node_id_j)
            
            if tree_i is None or tree_j is None:
                continue
            
            root_i = find_root(tree_i)
            root_j = find_root(tree_j)
            
            # Skip if already in the same merged tree
            if root_i == root_j:
                continue
            
            # Create a connecting segment between the two nodes
            node_i = network.nodes.get(node_id_i)
            node_j = network.nodes.get(node_id_j)
            
            if node_i is None or node_j is None:
                continue
            
            # Get radii from node attributes
            radius_i = node_i.attributes.get("radius", 0.0001)
            radius_j = node_j.attributes.get("radius", 0.0001)
            
            # Create connecting segment
            segment_id = network.id_gen.next_id()
            geometry = TubeGeometry(
                start=node_i.position,
                end=node_j.position,
                radius_start=radius_i,
                radius_end=radius_j,
            )
            segment = VesselSegment(
                id=segment_id,
                start_node_id=node_id_i,
                end_node_id=node_id_j,
                geometry=geometry,
                vessel_type=vessel_type,
            )
            network.add_segment(segment)
            
            # Update node types if they were terminals
            if node_i.node_type == "terminal":
                node_i.node_type = "junction"
            if node_j.node_type == "terminal":
                node_j.node_type = "junction"
            
            # Merge the trees
            merged_trees[root_j] = root_i
            merges_performed += 1
            
            logger.debug(
                f"Merged trees {tree_i} and {tree_j} at distance {dist:.6f}m "
                f"(nodes {node_id_i} and {node_id_j})"
            )
        
        return merges_performed
    
    def _generate_merge_to_trunk_mode(
        self,
        network: VascularNetwork,
        domain: DomainSpec,
        num_outlets: int,
        inlets: List[Dict[str, Any]],
        vessel_type: str,
        config: KaryTreeConfig,
        rng: np.random.Generator,
    ) -> VascularNetwork:
        """Generate a single tree with inlets merging to a common trunk."""
        n_inlets = len(inlets)
        
        inlet_positions = []
        inlet_radii = []
        inlet_directions = []
        
        for inlet in inlets:
            pos = np.array(inlet.get("position", [0, 0, 0]))
            inlet_positions.append(pos)
            inlet_radii.append(inlet.get("radius", 0.001))
            direction = inlet.get("direction", inlet.get("growth_inward_direction"))
            if direction is not None:
                d = np.array(direction)
                norm = np.linalg.norm(d)
                if norm > 0:
                    d = d / norm
                else:
                    d = np.array([0.0, 0.0, -1.0])
            else:
                d = np.array([0.0, 0.0, -1.0])
            inlet_directions.append(d)
        
        centroid = np.mean(inlet_positions, axis=0)
        avg_direction = np.mean(inlet_directions, axis=0)
        avg_direction = avg_direction / np.linalg.norm(avg_direction)
        
        domain_size = self._get_domain_characteristic_size(domain)
        trunk_depth = domain_size * config.trunk_depth_fraction
        trunk_position = centroid + avg_direction * trunk_depth
        
        if config.trunk_merge_radius is not None:
            trunk_radius = config.trunk_merge_radius
        else:
            total_area = sum(r**2 for r in inlet_radii)
            trunk_radius = np.sqrt(total_area) * config.taper_factor
        
        trunk_id = network.id_gen.next_id()
        trunk_node = Node(
            id=trunk_id,
            position=Point3D(*trunk_position),
            node_type="junction",
            vessel_type=vessel_type,
            attributes={"radius": trunk_radius, "is_trunk": True},
        )
        network.add_node(trunk_node)
        
        for i, (pos, radius, direction) in enumerate(zip(inlet_positions, inlet_radii, inlet_directions)):
            inlet_pos = Point3D(*pos)
            inlet_id = network.id_gen.next_id()
            inlet_node = Node(
                id=inlet_id,
                position=inlet_pos,
                node_type="inlet",
                vessel_type=vessel_type,
                attributes={"radius": radius, "inlet_index": i},
            )
            network.add_node(inlet_node)
            
            segment_id = network.id_gen.next_id()
            geometry = TubeGeometry(
                start=inlet_pos,
                end=trunk_node.position,
                radius_start=radius,
                radius_end=trunk_radius,
            )
            segment = VesselSegment(
                id=segment_id,
                start_node_id=inlet_id,
                end_node_id=trunk_id,
                geometry=geometry,
                vessel_type=vessel_type,
            )
            network.add_segment(segment)
        
        depth = self._calculate_depth(config.k, num_outlets)
        
        branch_length = config.branch_length
        if branch_length is None and config.use_domain_scaling:
            decay = config.branch_length_decay
            if decay < 1.0 and depth > 0:
                geometric_sum = (1 - decay**depth) / (1 - decay)
            else:
                geometric_sum = depth if depth > 0 else 1
            target_extent = domain_size * config.tree_extent_fraction
            branch_length = target_extent / geometric_sum
        elif branch_length is None:
            branch_length = 0.001
        
        self._generate_subtree(
            network=network,
            parent_node=trunk_node,
            parent_direction=avg_direction,
            primary_axis=avg_direction,
            current_depth=0,
            max_depth=depth,
            current_radius=trunk_radius,
            current_length=branch_length,
            config=config,
            vessel_type=vessel_type,
            domain=domain,
            rng=rng,
        )
        
        terminal_count = sum(1 for n in network.nodes.values() if n.node_type == "terminal")
        logger.info(
            f"Merge-to-trunk mode: {n_inlets} inlets merged to trunk at depth "
            f"{trunk_depth*1000:.2f}mm, generated {terminal_count} terminals"
        )
        
        return network
    
    def _calculate_depth(self, k: int, target_terminals: int) -> int:
        """Calculate tree depth needed for target terminal count."""
        if target_terminals <= 1:
            return 0
        
        # k^depth >= target_terminals
        # depth >= log_k(target_terminals)
        depth = int(np.ceil(np.log(target_terminals) / np.log(k)))
        return max(1, depth)
    
    def _get_domain_characteristic_size(self, domain: DomainSpec) -> float:
        """
        Get a characteristic size for the domain to use for scaling.
        
        For cylinders: uses the smaller of radius and height/2
        For boxes: uses the smallest dimension / 2
        For ellipsoids: uses the smallest semi-axis
        
        Returns size in meters.
        """
        if hasattr(domain, 'radius') and hasattr(domain, 'height'):
            # Cylinder domain
            return min(domain.radius, domain.height / 2)
        elif hasattr(domain, 'x_min') and hasattr(domain, 'x_max'):
            # Box domain
            dx = domain.x_max - domain.x_min
            dy = domain.y_max - domain.y_min
            dz = domain.z_max - domain.z_min
            return min(dx, dy, dz) / 2
        elif hasattr(domain, 'semi_axes'):
            # Ellipsoid domain
            return min(domain.semi_axes)
        elif hasattr(domain, 'bbox'):
            # Generic domain with bounding box
            bbox = domain.bbox
            if hasattr(bbox, 'min') and hasattr(bbox, 'max'):
                dims = [bbox.max[i] - bbox.min[i] for i in range(3)]
                return min(dims) / 2
        
        # Fallback: assume 5mm characteristic size
        logger.warning(
            f"Could not determine characteristic size for domain type {type(domain).__name__}, "
            "using default 5mm"
        )
        return 0.005
    
    def _generate_subtree(
        self,
        network: VascularNetwork,
        parent_node: Node,
        parent_direction: np.ndarray,
        primary_axis: np.ndarray,
        current_depth: int,
        max_depth: int,
        current_radius: float,
        current_length: float,
        config: KaryTreeConfig,
        vessel_type: str,
        domain: DomainSpec,
        rng: np.random.Generator,
    ) -> None:
        """
        Recursively generate subtree from parent node.
        
        Parameters
        ----------
        network : VascularNetwork
            Network to add nodes/segments to
        parent_node : Node
            Parent node to branch from
        parent_direction : np.ndarray
            Direction of the parent segment (for bifurcation angle calculation)
        primary_axis : np.ndarray
            Primary growth direction (for constraint checking)
        current_depth : int
            Current depth in the tree
        max_depth : int
            Maximum depth to generate
        current_radius : float
            Current vessel radius
        current_length : float
            Current branch length
        config : KaryTreeConfig
            Configuration parameters
        vessel_type : str
            Type of vessels
        domain : DomainSpec
            Domain for containment checks
        rng : np.random.Generator
            Random number generator
        """
        if current_depth >= max_depth:
            parent_node.node_type = "terminal"
            return
        
        if current_radius < config.min_radius:
            parent_node.node_type = "terminal"
            return
        
        k = config.k
        child_radius = current_radius * config.taper_factor
        child_length = current_length * config.branch_length_decay
        
        base_angle = config.angle_deg
        angle_var = config.angle_variation_deg
        
        elevation_jitter = config.elevation_jitter_deg if config.elevation_jitter_deg is not None else angle_var
        azimuth_jitter = config.azimuth_jitter_deg
        max_deviation_rad = np.radians(config.max_deviation_deg)
        
        for i in range(k):
            angle_offset = (i - (k - 1) / 2) * base_angle * 2 / max(k - 1, 1)
            angle_offset += rng.uniform(-elevation_jitter, elevation_jitter)
            
            azimuth = rng.uniform(-azimuth_jitter, azimuth_jitter)
            
            child_direction = self._rotate_direction(
                parent_direction, 
                np.radians(angle_offset),
                np.radians(azimuth)
            )
            
            if config.max_deviation_deg < 90.0:
                dot_with_primary = np.dot(child_direction, primary_axis)
                angle_from_primary = np.arccos(np.clip(dot_with_primary, -1.0, 1.0))
                
                if angle_from_primary > max_deviation_rad:
                    blend_factor = max_deviation_rad / angle_from_primary
                    child_direction = child_direction * (1 - blend_factor) + primary_axis * blend_factor
                    child_direction = child_direction / np.linalg.norm(child_direction)
            
            if config.upward_forbidden:
                if child_direction[2] > 0:
                    child_direction[2] = -abs(child_direction[2]) * 0.1
                    child_direction = child_direction / np.linalg.norm(child_direction)
            
            child_pos = np.array([
                parent_node.position.x,
                parent_node.position.y,
                parent_node.position.z,
            ]) + child_direction * child_length
            
            if config.wall_margin > 0 and hasattr(domain, 'signed_distance'):
                point_3d = Point3D(*child_pos)
                sd = domain.signed_distance(point_3d)
                if sd > -config.wall_margin:
                    move_dist = config.wall_margin - (-sd) + 0.0001
                    if hasattr(domain, 'center'):
                        center = np.array([domain.center.x, domain.center.y, domain.center.z])
                    else:
                        center = np.array([0, 0, 0])
                    to_center = center - child_pos
                    if np.linalg.norm(to_center) > 0:
                        to_center = to_center / np.linalg.norm(to_center)
                        child_pos = child_pos + to_center * move_dist
            
            if hasattr(domain, 'contains'):
                if not domain.contains(Point3D(*child_pos)):
                    child_pos = self._project_inside(child_pos, domain)
            
            child_id = network.id_gen.next_id()
            child_node = Node(
                id=child_id,
                position=Point3D(*child_pos),
                node_type="junction",
                vessel_type=vessel_type,
                attributes={"radius": child_radius},
            )
            network.add_node(child_node)
            
            segment_id = network.id_gen.next_id()
            geometry = TubeGeometry(
                start=parent_node.position,
                end=child_node.position,
                radius_start=current_radius,
                radius_end=child_radius,
            )
            segment = VesselSegment(
                id=segment_id,
                start_node_id=parent_node.id,
                end_node_id=child_id,
                geometry=geometry,
                vessel_type=vessel_type,
            )
            network.add_segment(segment)
            
            self._generate_subtree(
                network=network,
                parent_node=child_node,
                parent_direction=child_direction,
                primary_axis=primary_axis,
                current_depth=current_depth + 1,
                max_depth=max_depth,
                current_radius=child_radius,
                current_length=child_length,
                config=config,
                vessel_type=vessel_type,
                domain=domain,
                rng=rng,
            )
    
    def _rotate_direction(
        self,
        direction: np.ndarray,
        polar_angle: float,
        azimuth_angle: float,
    ) -> np.ndarray:
        """Rotate a direction vector by polar and azimuth angles."""
        # Normalize input
        direction = direction / np.linalg.norm(direction)
        
        # Find perpendicular vectors
        if abs(direction[2]) < 0.9:
            perp1 = np.cross(direction, np.array([0, 0, 1]))
        else:
            perp1 = np.cross(direction, np.array([1, 0, 0]))
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(direction, perp1)
        
        # Rotate by polar angle around perp1
        cos_p = np.cos(polar_angle)
        sin_p = np.sin(polar_angle)
        rotated = direction * cos_p + np.cross(perp1, direction) * sin_p
        
        # Rotate by azimuth around original direction
        cos_a = np.cos(azimuth_angle)
        sin_a = np.sin(azimuth_angle)
        final = rotated * cos_a + np.cross(direction, rotated) * sin_a
        
        return final / np.linalg.norm(final)
    
    def _project_inside(self, position: np.ndarray, domain: DomainSpec) -> np.ndarray:
        """Project a position back inside the domain."""
        if hasattr(domain, 'project_inside'):
            result = domain.project_inside(Point3D(*position))
            return np.array([result.x, result.y, result.z])
        
        # Fallback: move toward center
        if hasattr(domain, 'center'):
            center = np.array([domain.center.x, domain.center.y, domain.center.z])
        else:
            center = np.array([0, 0, 0])
        
        direction = center - position
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
            return position + direction * 0.001  # Move 1mm toward center
        
        return position


__all__ = [
    "KaryTreeBackend",
    "KaryTreeConfig",
]
