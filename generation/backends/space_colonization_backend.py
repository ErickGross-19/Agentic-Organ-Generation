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
from ..ops.space_colonization import space_colonization_step, SpaceColonizationParams as SCParams
from ..utils.tissue_sampling import sample_tissue_points
from aog_policies import TissueSamplingPolicy


@dataclass
class SpaceColonizationConfig(BackendConfig):
    """Configuration for space colonization backend.
    
    Multi-inlet modes:
    - "blended": Soft, blended weighting of attractors across inlets (recommended for organic growth)
    - "partitioned_xy": Hard XY Voronoi partitioning (legacy behavior, creates cross patterns)
    - "forest": Separate trees per inlet with no merging
    - "forest_with_merge": Separate trees that merge where they collide
    """
    
    attraction_distance: float = 0.010  # meters (influence_radius)
    kill_distance: float = 0.002  # meters
    step_size: float = 0.002  # meters
    num_attractors: int = 1000
    max_iterations: int = 500
    branch_angle_deg: float = 30.0
    multi_inlet_mode: str = "blended"  # "blended", "partitioned_xy", "forest", or "forest_with_merge"
    collision_merge_distance: float = 0.0003  # 0.3mm in meters
    max_inlets: int = 10
    # Blended mode parameters
    multi_inlet_blend_sigma: float = 0.0  # If 0, auto-computed as domain_radius/2
    # Directional constraints for multi-inlet (less restrictive for blended mode)
    directional_bias: float = 0.5  # 0.0 = no bias, 1.0 = strict directional growth
    max_deviation_deg: float = 60.0  # Maximum angle from inlet direction
    
    # Radius and taper control (passed to SCParams)
    min_radius: float = 0.0001  # 0.1mm - minimum vessel radius
    taper_factor: float = 0.95  # Radius reduction per generation
    
    # Bifurcation control (passed to SCParams)
    encourage_bifurcation: bool = False  # Whether to encourage multiple children per node
    max_children_per_node: int = 2  # Maximum children to create (typically 2 for bifurcation)
    bifurcation_probability: float = 0.7  # Probability of bifurcating when conditions are met
    min_attractions_for_bifurcation: int = 3  # Minimum attraction points needed to consider bifurcation
    bifurcation_angle_threshold_deg: float = 40.0  # Minimum angle spread to trigger bifurcation
    
    # Step control
    max_steps: int = 100  # Maximum growth steps per space_colonization_step call


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
        tissue_sampling_policy: Optional[TissueSamplingPolicy] = None,
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
        tissue_sampling_policy : TissueSamplingPolicy, optional
            Policy controlling tissue/attractor point sampling strategy.
            If None, uses uniform sampling via domain.sample_points().
            
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
        
        # Use TissueSamplingPolicy if provided, otherwise fall back to domain.sample_points()
        if tissue_sampling_policy is not None:
            # Override n_points in policy with calculated num_attractors
            effective_policy = TissueSamplingPolicy(
                enabled=tissue_sampling_policy.enabled,
                strategy=tissue_sampling_policy.strategy,
                n_points=num_attractors,
                seed=int(rng.integers(0, 2**31)),
                min_distance_to_ports=tissue_sampling_policy.min_distance_to_ports,
                exclude_spheres=tissue_sampling_policy.exclude_spheres,
                depth_reference=tissue_sampling_policy.depth_reference,
                depth_min=tissue_sampling_policy.depth_min,
                depth_max=tissue_sampling_policy.depth_max,
                depth_distribution=tissue_sampling_policy.depth_distribution,
                depth_power=tissue_sampling_policy.depth_power,
                depth_lambda=tissue_sampling_policy.depth_lambda,
                depth_alpha=tissue_sampling_policy.depth_alpha,
                depth_beta=tissue_sampling_policy.depth_beta,
                radial_reference=tissue_sampling_policy.radial_reference,
                r_min=tissue_sampling_policy.r_min,
                r_max=tissue_sampling_policy.r_max,
                radial_distribution=tissue_sampling_policy.radial_distribution,
                radial_power=tissue_sampling_policy.radial_power,
                ring_r0=tissue_sampling_policy.ring_r0,
                ring_sigma=tissue_sampling_policy.ring_sigma,
                shell_thickness=tissue_sampling_policy.shell_thickness,
                shell_mode=tissue_sampling_policy.shell_mode,
                gaussian_mean=tissue_sampling_policy.gaussian_mean,
                gaussian_sigma=tissue_sampling_policy.gaussian_sigma,
                mixture_components=tissue_sampling_policy.mixture_components,
            )
            
            # Build ports dict for exclusion zones
            ports = {
                "inlets": [{"position": tuple(inlet_position)}],
                "outlets": [],
            }
            
            attractor_list, sampling_report = sample_tissue_points(
                domain=domain,
                ports=ports,
                policy=effective_policy,
                seed=int(rng.integers(0, 2**31)),
            )
        else:
            # Legacy path: use domain.sample_points() directly
            attractors = domain.sample_points(num_attractors, seed=int(rng.integers(0, 2**31)))
            attractor_list = [Point3D.from_array(a) for a in attractors]
        
        sc_config = SCParams(
            influence_radius=config.attraction_distance,
            kill_radius=config.kill_distance,
            step_size=config.step_size,
            min_radius=config.min_radius,
            taper_factor=config.taper_factor,
            max_steps=config.max_steps,
            encourage_bifurcation=config.encourage_bifurcation,
            max_children_per_node=config.max_children_per_node,
            bifurcation_probability=config.bifurcation_probability,
            min_attractions_for_bifurcation=config.min_attractions_for_bifurcation,
            bifurcation_angle_threshold_deg=config.bifurcation_angle_threshold_deg,
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
    
    def generate_multi_inlet(
        self,
        domain: DomainSpec,
        num_outlets: int,
        inlets: list,
        vessel_type: str = "arterial",
        config: Optional[SpaceColonizationConfig] = None,
        rng_seed: Optional[int] = None,
        tissue_sampling_policy: Optional[TissueSamplingPolicy] = None,
    ) -> VascularNetwork:
        """
        Generate a vascular network with multiple inlets using space colonization.
        
        Supports multiple modes:
        - "blended": Soft, blended weighting of attractors across inlets (organic growth)
        - "partitioned_xy": Hard XY Voronoi partitioning (legacy, creates cross patterns)
        - "forest": Separate trees per inlet with no merging
        - "forest_with_merge": Separate trees that merge where they collide
        
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
            - direction: [x, y, z] optional, growth direction
        vessel_type : str
            Type of vessels ("arterial" or "venous")
        config : SpaceColonizationConfig, optional
            Space colonization configuration
        rng_seed : int, optional
            Random seed for reproducibility
        tissue_sampling_policy : TissueSamplingPolicy, optional
            Policy controlling tissue/attractor point sampling strategy.
            
        Returns
        -------
        VascularNetwork
            Generated vascular network with multiple inlet trees
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if config is None:
            config = SpaceColonizationConfig()
        
        if len(inlets) > config.max_inlets:
            inlets = inlets[:config.max_inlets]
        
        if len(inlets) == 0:
            raise ValueError("At least one inlet is required")
        
        if len(inlets) == 1:
            inlet = inlets[0]
            inlet_position = np.array(inlet.get("position", [0, 0, 0]))
            inlet_radius = inlet.get("radius", 0.001)
            return self.generate(
                domain=domain,
                num_outlets=num_outlets,
                inlet_position=inlet_position,
                inlet_radius=inlet_radius,
                vessel_type=vessel_type,
                config=config,
                rng_seed=rng_seed,
                tissue_sampling_policy=tissue_sampling_policy,
            )
        
        rng = np.random.default_rng(rng_seed if rng_seed is not None else config.seed)
        network = VascularNetwork(domain=domain, seed=rng_seed)
        
        n_inlets = len(inlets)
        
        # Generate attractors for the entire domain
        num_attractors = max(config.num_attractors, num_outlets * 2)
        
        if tissue_sampling_policy is not None:
            effective_policy = TissueSamplingPolicy(
                enabled=tissue_sampling_policy.enabled,
                strategy=tissue_sampling_policy.strategy,
                n_points=num_attractors,
                seed=int(rng.integers(0, 2**31)),
                min_distance_to_ports=tissue_sampling_policy.min_distance_to_ports,
                exclude_spheres=tissue_sampling_policy.exclude_spheres,
                depth_reference=tissue_sampling_policy.depth_reference,
                depth_min=tissue_sampling_policy.depth_min,
                depth_max=tissue_sampling_policy.depth_max,
                depth_distribution=tissue_sampling_policy.depth_distribution,
                depth_power=tissue_sampling_policy.depth_power,
                depth_lambda=tissue_sampling_policy.depth_lambda,
                depth_alpha=tissue_sampling_policy.depth_alpha,
                depth_beta=tissue_sampling_policy.depth_beta,
                radial_reference=tissue_sampling_policy.radial_reference,
                r_min=tissue_sampling_policy.r_min,
                r_max=tissue_sampling_policy.r_max,
                radial_distribution=tissue_sampling_policy.radial_distribution,
                radial_power=tissue_sampling_policy.radial_power,
                ring_r0=tissue_sampling_policy.ring_r0,
                ring_sigma=tissue_sampling_policy.ring_sigma,
                shell_thickness=tissue_sampling_policy.shell_thickness,
                shell_mode=tissue_sampling_policy.shell_mode,
                gaussian_mean=tissue_sampling_policy.gaussian_mean,
                gaussian_sigma=tissue_sampling_policy.gaussian_sigma,
                mixture_components=tissue_sampling_policy.mixture_components,
            )
            
            ports = {
                "inlets": [{"position": tuple(inlet.get("position", [0, 0, 0]))} for inlet in inlets],
                "outlets": [],
            }
            
            attractor_list, _ = sample_tissue_points(
                domain=domain,
                ports=ports,
                policy=effective_policy,
                seed=int(rng.integers(0, 2**31)),
            )
        else:
            attractors = domain.sample_points(num_attractors, seed=int(rng.integers(0, 2**31)))
            attractor_list = [Point3D.from_array(a) for a in attractors]
        
        # Convert attractor_list to numpy array
        if attractor_list:
            all_tissue_points = np.array([[p.x, p.y, p.z] for p in attractor_list])
        else:
            all_tissue_points = np.zeros((0, 3))
        
        # Pre-compute all inlet positions
        inlet_positions = np.array([inlet.get("position", [0, 0, 0]) for inlet in inlets])
        
        # Compute blend_sigma for blended mode
        bounds = domain.get_bounds()
        domain_radius = max(
            (bounds[1] - bounds[0]) / 2,  # x extent / 2
            (bounds[3] - bounds[2]) / 2,  # y extent / 2
        )
        blend_sigma = config.multi_inlet_blend_sigma if config.multi_inlet_blend_sigma > 0 else domain_radius / 2
        
        # Use blended mode for organic growth
        if config.multi_inlet_mode == "blended":
            return self._generate_multi_inlet_blended(
                domain=domain,
                network=network,
                inlets=inlets,
                inlet_positions=inlet_positions,
                all_tissue_points=all_tissue_points,
                vessel_type=vessel_type,
                config=config,
                blend_sigma=blend_sigma,
                rng=rng,
                logger=logger,
            )
        
        # Legacy modes: partitioned_xy, forest, forest_with_merge
        return self._generate_multi_inlet_partitioned(
            domain=domain,
            network=network,
            inlets=inlets,
            inlet_positions=inlet_positions,
            all_tissue_points=all_tissue_points,
            vessel_type=vessel_type,
            config=config,
            rng=rng,
            logger=logger,
        )
    
    def _generate_multi_inlet_blended(
        self,
        domain: DomainSpec,
        network: VascularNetwork,
        inlets: list,
        inlet_positions: np.ndarray,
        all_tissue_points: np.ndarray,
        vessel_type: str,
        config: SpaceColonizationConfig,
        blend_sigma: float,
        rng: np.random.Generator,
        logger,
    ) -> VascularNetwork:
        """
        Generate multi-inlet network using overlapping attractor selection.
        
        This mode uses Gaussian-weighted sampling where each inlet samples its own
        subset of attractors, but subsets can OVERLAP (attractors are shared, not
        exclusively assigned). This removes Voronoi-like boundaries and produces
        organic, intertwined growth patterns.
        
        Key features:
        - Overlapping selection: each inlet samples weighted subset, subsets overlap
        - Gaussian weights: w_i = exp(-d_i^2 / (2*sigma^2)) for each attractor-inlet pair
        - Shared attractors: same attractor can influence multiple inlets simultaneously
        - Global kill: attractors removed when within kill_radius of ANY new node
        - Minimal directional constraints: allows organic branching in all directions
        """
        n_inlets = len(inlets)
        
        inlet_nodes = []
        inlet_directions = []
        active_tips_per_inlet: list = [set() for _ in range(n_inlets)]
        inlet_node_sets: list = [set() for _ in range(n_inlets)]
        
        for i, inlet in enumerate(inlets):
            inlet_position = np.array(inlet.get("position", [0, 0, 0]))
            inlet_radius = inlet.get("radius", 0.001)
            
            inlet_point = Point3D.from_array(inlet_position)
            inlet_direction = self._compute_initial_direction(inlet_point, domain)
            
            if "direction" in inlet:
                dir_arr = np.array(inlet["direction"])
                if np.linalg.norm(dir_arr) > 0:
                    dir_arr = dir_arr / np.linalg.norm(dir_arr)
                    inlet_direction = Direction3D.from_array(dir_arr)
            
            inlet_node = Node(
                id=network.id_gen.next_id(),
                position=inlet_point,
                node_type="inlet",
                vessel_type=vessel_type,
                attributes={
                    "radius": inlet_radius,
                    "direction": inlet_direction.to_dict(),
                    "branch_order": 0,
                    "inlet_index": i,
                },
            )
            network.add_node(inlet_node)
            
            inlet_nodes.append(inlet_node)
            inlet_directions.append(inlet_direction.to_array())
            active_tips_per_inlet[i].add(inlet_node.id)
            inlet_node_sets[i].add(inlet_node.id)
        
        def compute_attractor_weights(tissue_points: np.ndarray) -> np.ndarray:
            if len(tissue_points) == 0:
                return np.zeros((0, n_inlets))
            xy_points = tissue_points[:, :2]
            xy_inlets = inlet_positions[:, :2]
            diffs = xy_points[:, np.newaxis, :] - xy_inlets[np.newaxis, :, :]
            distances = np.linalg.norm(diffs, axis=2)
            weights = np.exp(-distances**2 / (2 * blend_sigma**2))
            return weights
        
        tissue_points = all_tissue_points.copy()
        
        for iteration in range(config.max_iterations):
            if len(tissue_points) == 0:
                logger.info(f"Blended mode: stopping at iteration {iteration}, no attractors left")
                break
            
            total_active = sum(len(tips) for tips in active_tips_per_inlet)
            if total_active == 0:
                logger.info(f"Blended mode: stopping at iteration {iteration}, no active tips")
                break
            
            weights = compute_attractor_weights(tissue_points)
            
            all_new_nodes = []
            
            for i in range(n_inlets):
                if not active_tips_per_inlet[i]:
                    continue
                
                inlet_weights = weights[:, i]
                
                if inlet_weights.sum() == 0:
                    continue
                
                sample_probs = inlet_weights / inlet_weights.sum()
                n_samples = min(len(tissue_points), max(100, len(tissue_points) // n_inlets))
                sampled_indices = rng.choice(
                    len(tissue_points),
                    size=n_samples,
                    replace=False,
                    p=sample_probs,
                )
                inlet_tissue_points = tissue_points[sampled_indices]
                
                sc_params = SCParams(
                    influence_radius=config.attraction_distance,
                    kill_radius=config.kill_distance,
                    step_size=config.step_size,
                    vessel_type=vessel_type,
                    preferred_direction=None,
                    directional_bias=config.directional_bias,
                    max_deviation_deg=config.max_deviation_deg,
                    min_radius=config.min_radius,
                    taper_factor=config.taper_factor,
                    max_steps=config.max_steps,
                    encourage_bifurcation=config.encourage_bifurcation,
                    max_children_per_node=config.max_children_per_node,
                    bifurcation_probability=config.bifurcation_probability,
                    min_attractions_for_bifurcation=config.min_attractions_for_bifurcation,
                    bifurcation_angle_threshold_deg=config.bifurcation_angle_threshold_deg,
                )
                
                seed_nodes = list(active_tips_per_inlet[i])
                
                result = space_colonization_step(
                    network=network,
                    tissue_points=inlet_tissue_points,
                    params=sc_params,
                    seed=int(rng.integers(0, 2**31)),
                    seed_nodes=seed_nodes,
                )
                
                if result.is_success():
                    new_node_ids = result.new_ids.get("nodes", [])
                    
                    if new_node_ids:
                        for node_id in new_node_ids:
                            active_tips_per_inlet[i].add(node_id)
                            inlet_node_sets[i].add(node_id)
                            all_new_nodes.append(node_id)
                        
                        tips_to_remove = set()
                        for tip_id in active_tips_per_inlet[i]:
                            if tip_id in new_node_ids:
                                continue
                            has_children = any(
                                seg.start_node_id == tip_id 
                                for seg in network.segments.values()
                            )
                            if has_children:
                                tips_to_remove.add(tip_id)
                        active_tips_per_inlet[i] -= tips_to_remove
            
            if all_new_nodes:
                for node_id in all_new_nodes:
                    node = network.nodes.get(node_id)
                    if node:
                        node_pos = np.array([node.position.x, node.position.y, node.position.z])
                        distances = np.linalg.norm(tissue_points - node_pos, axis=1)
                        tissue_points = tissue_points[distances > config.kill_distance]
            else:
                logger.info(f"Blended mode: no growth at iteration {iteration}")
                break
        
        self._mark_terminals(network)
        
        merges = self._merge_colliding_trees(
            network=network,
            inlet_node_sets=inlet_node_sets,
            merge_distance=config.collision_merge_distance,
            vessel_type=vessel_type,
        )
        logger.info(
            f"Blended mode: overlapping selection with {n_inlets} inlets, performed {merges} merges, "
            f"total nodes: {len(network.nodes)}"
        )
        
        return network
    
    def _generate_multi_inlet_partitioned(
        self,
        domain: DomainSpec,
        network: VascularNetwork,
        inlets: list,
        inlet_positions: np.ndarray,
        all_tissue_points: np.ndarray,
        vessel_type: str,
        config: SpaceColonizationConfig,
        rng: np.random.Generator,
        logger,
    ) -> VascularNetwork:
        """
        Generate multi-inlet network using partitioned/cylinder filtering (legacy mode).
        
        This is the original implementation that uses strict cylinder filtering
        and directional constraints. Kept for backward compatibility.
        """
        n_inlets = len(inlets)
        inlet_node_sets: list = [set() for _ in range(n_inlets)]
        
        for i, inlet in enumerate(inlets):
            inlet_position = np.array(inlet.get("position", [0, 0, 0]))
            inlet_radius = inlet.get("radius", 0.001)
            
            inlet_point = Point3D.from_array(inlet_position)
            inlet_direction = self._compute_initial_direction(inlet_point, domain)
            
            if "direction" in inlet:
                dir_arr = np.array(inlet["direction"])
                if np.linalg.norm(dir_arr) > 0:
                    dir_arr = dir_arr / np.linalg.norm(dir_arr)
                    inlet_direction = Direction3D.from_array(dir_arr)
            
            inlet_dir_arr = inlet_direction.to_array()
            
            sc_params = SCParams(
                influence_radius=config.attraction_distance,
                kill_radius=config.kill_distance,
                step_size=config.step_size,
                vessel_type=vessel_type,
                preferred_direction=tuple(inlet_dir_arr),
                directional_bias=1.0,
                max_deviation_deg=30.0,
                min_radius=config.min_radius,
                taper_factor=config.taper_factor,
                max_steps=config.max_steps,
                encourage_bifurcation=config.encourage_bifurcation,
                max_children_per_node=config.max_children_per_node,
                bifurcation_probability=config.bifurcation_probability,
                min_attractions_for_bifurcation=config.min_attractions_for_bifurcation,
                bifurcation_angle_threshold_deg=config.bifurcation_angle_threshold_deg,
            )
            
            tissue_points = self._filter_tissue_points_by_cylinder(
                all_tissue_points,
                inlet_position,
                inlet_dir_arr,
                cylinder_radius=1.0,
            )
            
            tissue_points = self._filter_tissue_points_by_direction(
                tissue_points,
                inlet_position,
                inlet_dir_arr,
                cone_angle_deg=30.0,
            )
            
            nodes_before = set(network.nodes.keys())
            
            inlet_node = Node(
                id=network.id_gen.next_id(),
                position=inlet_point,
                node_type="inlet",
                vessel_type=vessel_type,
                attributes={
                    "radius": inlet_radius,
                    "direction": inlet_direction.to_dict(),
                    "branch_order": 0,
                    "inlet_index": i,
                },
            )
            network.add_node(inlet_node)
            
            seed_nodes = [inlet_node.id]
            iterations_per_inlet = config.max_iterations // n_inlets
            
            for iteration in range(iterations_per_inlet):
                if len(tissue_points) == 0 or not seed_nodes:
                    break
                
                result = space_colonization_step(
                    network=network,
                    tissue_points=tissue_points,
                    params=sc_params,
                    seed=int(rng.integers(0, 2**31)),
                    seed_nodes=seed_nodes,
                )
                
                if result.is_success():
                    new_node_ids = result.new_ids.get("nodes", [])
                    
                    if new_node_ids:
                        seed_nodes = new_node_ids
                    
                    remaining_mask = result.metadata.get("remaining_tissue_mask")
                    if remaining_mask is not None:
                        tissue_points = tissue_points[remaining_mask]
                    else:
                        for node_id in new_node_ids:
                            node = network.nodes.get(node_id)
                            if node:
                                node_pos = np.array([node.position.x, node.position.y, node.position.z])
                                distances = np.linalg.norm(tissue_points - node_pos, axis=1)
                                tissue_points = tissue_points[distances > config.kill_distance]
                    
                    for node_id in new_node_ids:
                        node = network.nodes.get(node_id)
                        if node:
                            node_pos = np.array([node.position.x, node.position.y, node.position.z])
                            distances = np.linalg.norm(all_tissue_points - node_pos, axis=1)
                            all_tissue_points = all_tissue_points[distances > config.kill_distance]
                else:
                    break
            
            nodes_after = set(network.nodes.keys())
            inlet_node_sets[i] = nodes_after - nodes_before
        
        self._mark_terminals(network)
        
        if config.multi_inlet_mode == "forest_with_merge":
            merges = self._merge_colliding_trees(
                network=network,
                inlet_node_sets=inlet_node_sets,
                merge_distance=config.collision_merge_distance,
                vessel_type=vessel_type,
            )
            logger.info(
                f"Forest with merge mode: generated {n_inlets} trees, performed {merges} merges"
            )
        
        return network
    
    def _merge_colliding_trees(
        self,
        network: VascularNetwork,
        inlet_node_sets: list,
        merge_distance: float,
        vessel_type: str,
    ) -> int:
        """
        Find and merge segments from different trees that are close to each other.
        
        Returns the number of merges performed.
        """
        from ..core.types import TubeGeometry
        from ..core.network import VesselSegment
        
        merges_performed = 0
        n_trees = len(inlet_node_sets)
        
        # Build a map of node_id -> tree_index
        node_tree_map = {}
        for tree_idx, node_set in enumerate(inlet_node_sets):
            for node_id in node_set:
                node_tree_map[node_id] = tree_idx
        
        # Find pairs of nodes from different trees that are close
        merge_candidates = []
        
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
        
        # Track which trees have been merged (union-find)
        merged_trees = {i: i for i in range(n_trees)}
        
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
        
        return merges_performed
    
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
                sc_config = SCParams(
                    influence_radius=config.attraction_distance,
                    kill_radius=config.kill_distance,
                    step_size=config.step_size,
                    min_radius=config.min_radius,
                    taper_factor=config.taper_factor,
                    max_steps=config.max_steps,
                    encourage_bifurcation=config.encourage_bifurcation,
                    max_children_per_node=config.max_children_per_node,
                    bifurcation_probability=config.bifurcation_probability,
                    min_attractions_for_bifurcation=config.min_attractions_for_bifurcation,
                    bifurcation_angle_threshold_deg=config.bifurcation_angle_threshold_deg,
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
        tissue_sampling_policy: Optional[TissueSamplingPolicy] = None,
    ) -> VascularNetwork:
        """
        Generate a dual arterial-venous network using space colonization.
        
        Parameters
        ----------
        tissue_sampling_policy : TissueSamplingPolicy, optional
            Policy controlling tissue/attractor point sampling strategy.
            If None, uses uniform sampling via domain.sample_points().
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
        
        # Use TissueSamplingPolicy if provided, otherwise fall back to domain.sample_points()
        if tissue_sampling_policy is not None:
            effective_policy = TissueSamplingPolicy(
                enabled=tissue_sampling_policy.enabled,
                strategy=tissue_sampling_policy.strategy,
                n_points=num_attractors,
                seed=int(rng.integers(0, 2**31)),
                min_distance_to_ports=tissue_sampling_policy.min_distance_to_ports,
                exclude_spheres=tissue_sampling_policy.exclude_spheres,
                depth_reference=tissue_sampling_policy.depth_reference,
                depth_min=tissue_sampling_policy.depth_min,
                depth_max=tissue_sampling_policy.depth_max,
                depth_distribution=tissue_sampling_policy.depth_distribution,
                depth_power=tissue_sampling_policy.depth_power,
                depth_lambda=tissue_sampling_policy.depth_lambda,
                depth_alpha=tissue_sampling_policy.depth_alpha,
                depth_beta=tissue_sampling_policy.depth_beta,
                radial_reference=tissue_sampling_policy.radial_reference,
                r_min=tissue_sampling_policy.r_min,
                r_max=tissue_sampling_policy.r_max,
                radial_distribution=tissue_sampling_policy.radial_distribution,
                radial_power=tissue_sampling_policy.radial_power,
                ring_r0=tissue_sampling_policy.ring_r0,
                ring_sigma=tissue_sampling_policy.ring_sigma,
                shell_thickness=tissue_sampling_policy.shell_thickness,
                shell_mode=tissue_sampling_policy.shell_mode,
                gaussian_mean=tissue_sampling_policy.gaussian_mean,
                gaussian_sigma=tissue_sampling_policy.gaussian_sigma,
                mixture_components=tissue_sampling_policy.mixture_components,
            )
            
            ports = {
                "inlets": [{"position": tuple(arterial_inlet)}],
                "outlets": [{"position": tuple(venous_outlet)}],
            }
            
            all_attractor_points, _ = sample_tissue_points(
                domain=domain,
                ports=ports,
                policy=effective_policy,
                seed=int(rng.integers(0, 2**31)),
            )
            arterial_attractors = all_attractor_points[:len(all_attractor_points)//2]
        else:
            all_attractors = domain.sample_points(num_attractors, seed=int(rng.integers(0, 2**31)))
            arterial_attractors = [Point3D.from_array(a) for a in all_attractors[:len(all_attractors)//2]]
        
        sc_config = SCParams(
            influence_radius=config.attraction_distance,
            kill_radius=config.kill_distance,
            step_size=config.step_size,
            min_radius=config.min_radius,
            taper_factor=config.taper_factor,
            max_steps=config.max_steps,
            encourage_bifurcation=config.encourage_bifurcation,
            max_children_per_node=config.max_children_per_node,
            bifurcation_probability=config.bifurcation_probability,
            min_attractions_for_bifurcation=config.min_attractions_for_bifurcation,
            bifurcation_angle_threshold_deg=config.bifurcation_angle_threshold_deg,
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
        
        # Get venous attractors from the second half
        if tissue_sampling_policy is not None:
            venous_attractors = all_attractor_points[len(all_attractor_points)//2:]
        else:
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
    
    def _filter_tissue_points_by_cylinder(
        self,
        tissue_points: np.ndarray,
        inlet_position: np.ndarray,
        direction: np.ndarray,
        cylinder_radius: float = 1.0,
    ) -> np.ndarray:
        """
        Filter tissue points to only include those within a cylinder below the inlet.
        
        This creates a narrow column of tissue points directly below each inlet,
        preventing trees from growing horizontally towards other inlets.
        
        Parameters
        ----------
        tissue_points : np.ndarray
            Array of tissue points (N, 3)
        inlet_position : np.ndarray
            Position of the inlet
        direction : np.ndarray
            Growth direction (normalized, typically [0, 0, -1] for downward)
        cylinder_radius : float
            Radius of the cylinder in the XY plane (default 1.0mm)
            
        Returns
        -------
        np.ndarray
            Filtered tissue points within the cylinder
        """
        if len(tissue_points) == 0:
            return tissue_points
        
        # Compute XY distance from inlet to each tissue point
        # This creates a vertical cylinder around the inlet's XY position
        inlet_xy = inlet_position[:2]
        tissue_xy = tissue_points[:, :2]
        
        xy_distances = np.linalg.norm(tissue_xy - inlet_xy, axis=1)
        
        # Keep only points within the cylinder radius
        in_cylinder = xy_distances <= cylinder_radius
        
        return tissue_points[in_cylinder]
    
    def _filter_tissue_points_by_nearest_inlet(
        self,
        tissue_points: np.ndarray,
        inlet_position: np.ndarray,
        all_inlet_positions: np.ndarray,
        inlet_index: int,
    ) -> np.ndarray:
        """
        Filter tissue points to only include those closest to this inlet.
        
        This creates Voronoi-like spatial partitioning so each inlet's tree
        only grows towards points in its own region, preventing trees from
        growing towards each other.
        
        Parameters
        ----------
        tissue_points : np.ndarray
            Array of tissue points (N, 3)
        inlet_position : np.ndarray
            Position of the current inlet
        all_inlet_positions : np.ndarray
            Positions of all inlets (M, 3)
        inlet_index : int
            Index of the current inlet
            
        Returns
        -------
        np.ndarray
            Filtered tissue points closest to this inlet
        """
        if len(tissue_points) == 0 or len(all_inlet_positions) <= 1:
            return tissue_points
        
        # Compute distance from each tissue point to each inlet (in XY plane only)
        # Using XY distance ensures vertical partitioning - each inlet "owns" a column
        tissue_xy = tissue_points[:, :2]  # Only X, Y coordinates
        inlet_xy = all_inlet_positions[:, :2]  # Only X, Y coordinates
        
        # Compute distances from each tissue point to each inlet
        # Shape: (num_tissue_points, num_inlets)
        distances = np.zeros((len(tissue_points), len(all_inlet_positions)))
        for j, inlet_pos in enumerate(inlet_xy):
            distances[:, j] = np.linalg.norm(tissue_xy - inlet_pos, axis=1)
        
        # Find which inlet is closest for each tissue point
        closest_inlet = np.argmin(distances, axis=1)
        
        # Keep only points where this inlet is the closest
        mask = closest_inlet == inlet_index
        
        return tissue_points[mask]
    
    def _filter_tissue_points_by_direction(
        self,
        tissue_points: np.ndarray,
        origin: np.ndarray,
        direction: np.ndarray,
        cone_angle_deg: float = 90.0,
    ) -> np.ndarray:
        """
        Filter tissue points to only include those within a cone from the origin.
        
        This ensures each inlet's tree grows in its specified direction (typically
        downward) rather than towards other inlets.
        
        Parameters
        ----------
        tissue_points : np.ndarray
            Array of tissue points (N, 3)
        origin : np.ndarray
            Origin point (inlet position)
        direction : np.ndarray
            Growth direction (normalized)
        cone_angle_deg : float
            Half-angle of the cone in degrees (default 90 = hemisphere, only
            points with positive dot product with direction are included)
            
        Returns
        -------
        np.ndarray
            Filtered tissue points within the cone
        """
        if len(tissue_points) == 0:
            return tissue_points
        
        # Normalize direction
        direction = direction / np.linalg.norm(direction)
        
        # Vector from origin to each tissue point
        to_points = tissue_points - origin
        
        # Normalize vectors
        distances = np.linalg.norm(to_points, axis=1)
        # Avoid division by zero
        valid_mask = distances > 1e-10
        to_points_normalized = np.zeros_like(to_points)
        to_points_normalized[valid_mask] = to_points[valid_mask] / distances[valid_mask, np.newaxis]
        
        # Compute dot product with direction
        dot_products = np.dot(to_points_normalized, direction)
        
        # Convert cone angle to cosine threshold
        # cos(90°) = 0, so points with dot > 0 are within 90° cone (hemisphere)
        cos_threshold = np.cos(np.radians(cone_angle_deg))
        
        # Filter points within the cone
        in_cone = dot_products >= cos_threshold
        
        return tissue_points[in_cone]
    
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
