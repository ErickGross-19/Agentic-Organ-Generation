"""
Scaffold top-down branching backend with online collision avoidance.

This backend generates vascular networks using recursive top-down branching
with configurable splits, levels, tapering, and collision avoidance.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List, Set
import numpy as np
import logging

from .base import GenerationBackend, BackendConfig
from ..core.network import VascularNetwork, Node, VesselSegment
from ..core.domain import DomainSpec
from ..core.types import Point3D, TubeGeometry
from ..spatial.grid_index import DynamicSpatialIndex

logger = logging.getLogger(__name__)


@dataclass
class CollisionOnlineConfig:
    """Configuration for online collision avoidance during growth."""
    enabled: bool = True
    buffer_abs_m: float = 0.00002
    buffer_rel: float = 0.05
    cell_size_m: float = 0.0005
    rotation_attempts: int = 14
    reduction_factors: List[float] = field(default_factory=lambda: [0.6, 0.35])
    max_attempts_per_child: int = 18
    on_fail: str = "terminate_branch"
    # Merge on collision: connect to nearby existing branch instead of terminating
    merge_on_collision: bool = False
    merge_distance_m: float = 0.0002  # Max distance to merge target
    merge_prefer_same_inlet: bool = True  # Prefer merging to same inlet's branches
    # Ancestor-aware retry: retry with adjusted params before giving up
    fail_retry_rounds: int = 0  # Number of retry rounds (0 = no retry)
    fail_retry_mode: str = "both"  # "shrink_radius", "increase_step", "both", "none"
    fail_retry_shrink_factor: float = 0.85  # Radius shrink factor per retry
    fail_retry_step_boost: float = 1.2  # Step length boost factor per retry


@dataclass
class CollisionPostpassConfig:
    """Configuration for post-generation collision cleanup."""
    enabled: bool = True
    min_clearance_m: float = 0.00002
    strategy_order: List[str] = field(default_factory=lambda: ["shrink", "terminate"])
    shrink_factor: float = 0.9
    shrink_max_iterations: int = 6


@dataclass
class ScaffoldTopDownConfig(BackendConfig):
    """
    Configuration for scaffold top-down branching generation.
    
    Parameters
    ----------
    primary_axis : tuple
        Global growth direction as (x, y, z) unit vector. Default: (0, 0, -1) for downward.
    splits : int
        Number of child branches at each bifurcation. Default: 3.
    levels : int
        Maximum depth of branching. Default: 6.
    ratio : float
        Radius taper factor at each level (child_radius = parent_radius * ratio). Default: 0.78.
    step_length : float
        Initial step length in meters. Default: 0.002 (2mm).
    step_decay : float
        Factor to multiply step length at each level. Default: 0.92.
    spread : float
        Lateral spread distance in meters. Default: 0.0015 (1.5mm).
    spread_decay : float
        Factor to multiply spread at each level. Default: 0.90.
    cone_angle_deg : float
        Maximum cone angle for child branches in degrees. Default: 70.
    jitter_deg : float
        Random jitter in branch angles in degrees. Default: 12.
    curvature : float
        Curvature factor for curved branches (0=straight, 1=max curve). Default: 0.35.
    curve_samples : int
        Number of sample points for curved branches (polyline). Default: 7.
    wall_margin_m : float
        Minimum distance from domain boundary in meters. Default: 0.0001 (100um).
    min_radius : float
        Minimum vessel radius in meters. Default: 0.00005 (50um).
    collision_online : CollisionOnlineConfig
        Online collision avoidance configuration.
    collision_postpass : CollisionPostpassConfig
        Post-generation collision cleanup configuration.
    """
    primary_axis: Tuple[float, float, float] = (0.0, 0.0, -1.0)
    splits: int = 3
    levels: int = 6
    ratio: float = 0.78
    step_length: float = 0.002
    step_decay: float = 0.92
    spread: float = 0.0015
    spread_decay: float = 0.90
    cone_angle_deg: float = 70.0
    jitter_deg: float = 12.0
    curvature: float = 0.35
    curve_samples: int = 7
    wall_margin_m: float = 0.0001
    boundary_extra_m: float = 0.0  # Extra boundary clearance (auto-set to 2*voxel_pitch if known)
    # Bottom-zone taper: smoothly reduce spread near bottom boundary
    bottom_zone_height_m: float = 0.0003  # Height of bottom zone where spread tapers
    bottom_spread_scale_min: float = 0.2  # Minimum spread scale in bottom zone (20%)
    # Stop-before-boundary: enforce buffer before domain boundary
    stop_before_boundary_m: float = 0.0001  # Global buffer to stop before boundary
    stop_before_boundary_extra_m: float = 0.0001  # Additional safety buffer
    clamp_mode: str = "shorten_step"  # "terminate" | "shorten_step" | "project_inside"
    collision_online: CollisionOnlineConfig = field(default_factory=CollisionOnlineConfig)
    collision_postpass: CollisionPostpassConfig = field(default_factory=CollisionPostpassConfig)


class ScaffoldTopDownBackend(GenerationBackend):
    """
    Generation backend using top-down recursive branching with collision avoidance.
    
    This backend creates tree structures by recursively splitting branches
    with configurable branching factor, depth, and collision avoidance.
    The algorithm is inspired by scaffold_web_collision.py but operates
    entirely in meters and produces VascularNetwork output.
    """
    
    @property
    def supports_dual_tree(self) -> bool:
        return False
    
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
        Generate a scaffold top-down vascular network.
        
        Parameters
        ----------
        domain : DomainSpec
            Geometric domain for the network
        num_outlets : int
            Target number of terminal outlets (informational; actual count
            determined by splits^levels)
        inlet_position : np.ndarray
            Position of the inlet node (x, y, z) in meters
        inlet_radius : float
            Radius of the inlet vessel in meters
        vessel_type : str
            Type of vessels ("arterial" or "venous")
        config : ScaffoldTopDownConfig, optional
            Backend configuration
        rng_seed : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        VascularNetwork
            Generated vascular network
        """
        if config is None:
            config = ScaffoldTopDownConfig()
        elif not isinstance(config, ScaffoldTopDownConfig):
            config = self._convert_config(config)
        
        effective_seed = rng_seed if rng_seed is not None else config.seed
        rng = np.random.default_rng(effective_seed)
        
        network = VascularNetwork(domain=domain)
        
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
        
        primary_axis = np.array(config.primary_axis, dtype=np.float64)
        norm = np.linalg.norm(primary_axis)
        if norm > 0:
            primary_axis = primary_axis / norm
        else:
            primary_axis = np.array([0.0, 0.0, -1.0])
        
        cell_size = config.collision_online.cell_size_m if config.collision_online.enabled else 0.001
        spatial_index = DynamicSpatialIndex(cell_size=cell_size)
        
        self._stats = {
            "segments_proposed": 0,
            "segments_created": 0,
            "collisions_detected": 0,
            "rotations_successful": 0,
            "branches_terminated": 0,
            "branches_skipped": 0,
            "merges_attempted": 0,
            "merges_succeeded": 0,
            "merges_failed": 0,
            "retries_attempted": 0,
            "retries_succeeded": 0,
            "boundary_clearance_failures": 0,
            "min_observed_clearance": float("inf"),
            "bottom_zone_spread_scaled_count": 0,
            "stop_before_boundary_triggered_count": 0,
            "min_bottom_clearance_observed": float("inf"),
            "min_wall_clearance_observed": float("inf"),
        }
        
        initial_angle = 0.0
        
        self._branch(
            network=network,
            spatial_index=spatial_index,
            parent_node=inlet_node,
            parent_direction=primary_axis,
            primary_axis=primary_axis,
            current_radius=inlet_radius,
            current_step=config.step_length,
            current_spread=config.spread,
            remaining_levels=config.levels,
            angle=initial_angle,
            config=config,
            vessel_type=vessel_type,
            domain=domain,
            rng=rng,
        )
        
        terminal_count = sum(1 for n in network.nodes.values() if n.node_type == "terminal")
        
        logger.info(
            f"ScaffoldTopDown generation complete: "
            f"{len(network.segments)} segments, {terminal_count} terminals, "
            f"stats={self._stats}"
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
        Generate a scaffold top-down network with multiple inlets.
        
        Each inlet grows its own tree independently.
        """
        if config is None:
            config = ScaffoldTopDownConfig()
        elif not isinstance(config, ScaffoldTopDownConfig):
            config = self._convert_config(config)
        
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
            )
        
        effective_seed = rng_seed if rng_seed is not None else config.seed
        rng = np.random.default_rng(effective_seed)
        
        network = VascularNetwork(domain=domain)
        
        primary_axis = np.array(config.primary_axis, dtype=np.float64)
        norm = np.linalg.norm(primary_axis)
        if norm > 0:
            primary_axis = primary_axis / norm
        else:
            primary_axis = np.array([0.0, 0.0, -1.0])
        
        cell_size = config.collision_online.cell_size_m if config.collision_online.enabled else 0.001
        spatial_index = DynamicSpatialIndex(cell_size=cell_size)
        
        self._stats = {
            "segments_proposed": 0,
            "segments_created": 0,
            "collisions_detected": 0,
            "rotations_successful": 0,
            "branches_terminated": 0,
            "branches_skipped": 0,
            "merges_attempted": 0,
            "merges_succeeded": 0,
            "merges_failed": 0,
            "retries_attempted": 0,
            "retries_succeeded": 0,
            "boundary_clearance_failures": 0,
            "min_observed_clearance": float("inf"),
            "bottom_zone_spread_scaled_count": 0,
            "stop_before_boundary_triggered_count": 0,
            "min_bottom_clearance_observed": float("inf"),
            "min_wall_clearance_observed": float("inf"),
        }
        
        for i, inlet in enumerate(inlets):
            inlet_position = np.array(inlet.get("position", [0, 0, 0]))
            inlet_radius = inlet.get("radius", 0.001)
            
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
            
            initial_angle = 2 * np.pi * i / len(inlets)
            
            self._branch(
                network=network,
                spatial_index=spatial_index,
                parent_node=inlet_node,
                parent_direction=primary_axis,
                primary_axis=primary_axis,
                current_radius=inlet_radius,
                current_step=config.step_length,
                current_spread=config.spread,
                remaining_levels=config.levels,
                angle=initial_angle,
                config=config,
                vessel_type=vessel_type,
                domain=domain,
                rng=rng,
            )
        
        terminal_count = sum(1 for n in network.nodes.values() if n.node_type == "terminal")
        logger.info(
            f"ScaffoldTopDown multi-inlet generation complete: "
            f"{len(inlets)} inlets, {len(network.segments)} segments, "
            f"{terminal_count} terminals"
        )
        
        return network
    
    def _convert_config(self, config: BackendConfig) -> ScaffoldTopDownConfig:
        """Convert generic BackendConfig to ScaffoldTopDownConfig."""
        return ScaffoldTopDownConfig(
            seed=config.seed,
            min_segment_length=config.min_segment_length,
            max_segment_length=config.max_segment_length,
            min_radius=config.min_radius,
            check_collisions=config.check_collisions,
            collision_clearance=config.collision_clearance,
        )
    
    def _branch(
        self,
        network: VascularNetwork,
        spatial_index: DynamicSpatialIndex,
        parent_node: Node,
        parent_direction: np.ndarray,
        primary_axis: np.ndarray,
        current_radius: float,
        current_step: float,
        current_spread: float,
        remaining_levels: int,
        angle: float,
        config: ScaffoldTopDownConfig,
        vessel_type: str,
        domain: DomainSpec,
        rng: np.random.Generator,
        parent_seg_id: Optional[int] = None,
    ) -> None:
        """
        Recursively create branches with collision avoidance.
        
        This is the core branching algorithm that:
        1. Computes target endpoint with spread and curvature
        2. Checks for collisions and attempts rotation/reduction if needed
        3. Creates curved polyline segments
        4. Recursively branches children using junction-safe sibling generation
        
        Junction-safe sibling generation (for splits > 1):
        - All sibling candidates are generated and validated first
        - Sibling-sibling collision checks use junction-safe rules that ignore
          overlap within a small neighborhood around the shared parent junction
        - Only after all siblings are accepted are they committed to the spatial index
        - This prevents the first sibling from blocking subsequent siblings at the
          shared junction point
        """
        if current_radius < config.min_radius:
            return
        if remaining_levels < 0:
            return
        
        parent_pos = parent_node.position.to_array()
        
        child_radius = current_radius * config.ratio
        if child_radius < config.min_radius:
            parent_node.node_type = "terminal"
            return
        
        perp1, perp2 = self._get_perpendicular_axes(primary_axis)
        
        jitter_rad = np.deg2rad(config.jitter_deg)
        angle_jitter = rng.uniform(-jitter_rad, jitter_rad)
        effective_angle = angle + angle_jitter
        
        lateral_dir = np.cos(effective_angle) * perp1 + np.sin(effective_angle) * perp2
        
        if remaining_levels == config.levels:
            target_spread = 0.0
        else:
            target_spread = current_spread
        
        # Apply bottom-zone taper: smoothly reduce spread near bottom boundary
        d_bottom = self._get_distance_to_bottom(parent_pos, domain)
        if d_bottom < config.bottom_zone_height_m and config.bottom_zone_height_m > 0:
            t = d_bottom / config.bottom_zone_height_m
            s = t * t * (3 - 2 * t)  # smoothstep
            zone_scale = config.bottom_spread_scale_min + (1 - config.bottom_spread_scale_min) * s
            target_spread = target_spread * zone_scale
            self._stats["bottom_zone_spread_scaled_count"] += 1
        
        target_pos = (
            parent_pos
            + primary_axis * current_step
            + lateral_dir * target_spread
        )
        
        # Apply stop-before-boundary check with buffer
        target_pos, was_adjusted = self._apply_stop_before_boundary(
            target_pos=target_pos,
            parent_pos=parent_pos,
            radius=child_radius,
            domain=domain,
            config=config,
        )
        if was_adjusted is None:
            # clamp_mode == "terminate" and boundary violated
            self._stats["stop_before_boundary_triggered_count"] += 1
            parent_node.node_type = "terminal"
            return
        if was_adjusted:
            self._stats["stop_before_boundary_triggered_count"] += 1
        
        target_pos = self._clamp_to_domain(target_pos, domain, config.wall_margin_m)
        
        avg_radius = (current_radius + child_radius) / 2
        
        if config.collision_online.enabled:
            target_pos, collision_resolved = self._find_safe_position(
                parent_pos=parent_pos,
                target_pos=target_pos,
                avg_radius=avg_radius,
                primary_axis=primary_axis,
                perp1=perp1,
                perp2=perp2,
                current_spread=target_spread,
                spatial_index=spatial_index,
                config=config,
                domain=domain,
                rng=rng,
                parent_direction=parent_direction,
                parent_seg_id=parent_seg_id,
            )
            
            if not collision_resolved:
                if config.collision_online.on_fail == "terminate_branch":
                    self._stats["branches_terminated"] += 1
                    parent_node.node_type = "terminal"
                    return
                else:
                    self._stats["branches_skipped"] += 1
                    return
        
        seg_length = np.linalg.norm(target_pos - parent_pos)
        if seg_length < config.min_segment_length:
            parent_node.node_type = "terminal"
            return
        
        centerline_points, end_direction = self._compute_curved_path(
            start=parent_pos,
            end=target_pos,
            start_dir=parent_direction,
            primary_axis=primary_axis,
            curvature=config.curvature,
            n_samples=config.curve_samples,
        )
        
        end_node_id = network.id_gen.next_id()
        end_node_type = "terminal" if remaining_levels == 0 else "junction"
        end_node = Node(
            id=end_node_id,
            position=Point3D(*target_pos),
            node_type=end_node_type,
            vessel_type=vessel_type,
            attributes={"radius": child_radius},
        )
        network.add_node(end_node)
        
        segment_id = network.id_gen.next_id()
        geometry = TubeGeometry(
            start=parent_node.position,
            end=end_node.position,
            radius_start=current_radius,
            radius_end=child_radius,
            centerline_points=[Point3D(*p) for p in centerline_points] if centerline_points else None,
        )
        segment = VesselSegment(
            id=segment_id,
            start_node_id=parent_node.id,
            end_node_id=end_node.id,
            vessel_type=vessel_type,
            geometry=geometry,
        )
        network.add_segment(segment)
        
        self._stats["segments_created"] += 1
        
        if config.collision_online.enabled:
            spatial_index.insert_segment(
                segment_id=segment_id,
                start=parent_pos,
                end=target_pos,
                radius=avg_radius,
                centerline=centerline_points,
            )
        
        if remaining_levels > 0:
            child_step = current_step * config.step_decay
            child_spread = current_spread * config.spread_decay
            
            child_angles = self._compute_child_angles(
                parent_angle=effective_angle,
                splits=config.splits,
                cone_angle_deg=config.cone_angle_deg,
                rng=rng,
            )
            
            # Use junction-safe sibling generation for multiple children
            if config.splits > 1 and config.collision_online.enabled:
                self._generate_siblings_junction_safe(
                    network=network,
                    spatial_index=spatial_index,
                    parent_node=end_node,
                    parent_direction=end_direction,
                    primary_axis=primary_axis,
                    child_radius=child_radius,
                    child_step=child_step,
                    child_spread=child_spread,
                    remaining_levels=remaining_levels - 1,
                    child_angles=child_angles,
                    config=config,
                    vessel_type=vessel_type,
                    domain=domain,
                    rng=rng,
                    parent_seg_id=segment_id,
                    perp1=perp1,
                    perp2=perp2,
                )
            else:
                # Single child or collision checking disabled - use simple sequential approach
                for child_angle in child_angles:
                    self._branch(
                        network=network,
                        spatial_index=spatial_index,
                        parent_node=end_node,
                        parent_direction=end_direction,
                        primary_axis=primary_axis,
                        current_radius=child_radius,
                        current_step=child_step,
                        current_spread=child_spread,
                        remaining_levels=remaining_levels - 1,
                        angle=child_angle,
                        config=config,
                        vessel_type=vessel_type,
                        domain=domain,
                        rng=rng,
                        parent_seg_id=segment_id,
                    )
    
    def _generate_siblings_junction_safe(
        self,
        network: VascularNetwork,
        spatial_index: DynamicSpatialIndex,
        parent_node: Node,
        parent_direction: np.ndarray,
        primary_axis: np.ndarray,
        child_radius: float,
        child_step: float,
        child_spread: float,
        remaining_levels: int,
        child_angles: List[float],
        config: ScaffoldTopDownConfig,
        vessel_type: str,
        domain: DomainSpec,
        rng: np.random.Generator,
        parent_seg_id: int,
        perp1: np.ndarray,
        perp2: np.ndarray,
    ) -> None:
        """
        Generate multiple sibling branches using junction-safe collision checking.
        
        This method implements Option A from the junction-safe sibling generation:
        1. For each child, propose candidate endpoint and compute polyline
        2. Check collision against global index (excluding parent segment)
        3. Check collision against already-accepted siblings using junction-safe rules
        4. After all children processed, insert accepted segments into spatial index
        5. Recursively branch from accepted children
        
        Junction-safe rule: Allow sibling overlap within a sphere of radius
        junction_ignore around the parent junction point. Outside that region,
        collisions are enforced normally.
        """
        parent_pos = parent_node.position.to_array()
        
        # Compute grandchild radius (the end radius of sibling segments)
        grandchild_radius = child_radius * config.ratio
        
        # avg_radius is the average of start and end radius for the sibling segments
        # This matches the calculation in _branch(): avg_radius = (current_radius + child_radius) / 2
        avg_radius = (child_radius + grandchild_radius) / 2
        
        buffer = max(
            config.collision_online.buffer_abs_m,
            config.collision_online.buffer_rel * avg_radius,
        )
        
        # Compute junction ignore radius based on local scale
        # Per task spec: junction_ignore = max(2 * current_radius, 2 * child_radius, 3 * buffer_abs_m)
        # Here, child_radius is the "current_radius" for siblings, grandchild_radius is their "child_radius"
        junction_ignore = max(
            2 * child_radius,
            2 * grandchild_radius,
            3 * config.collision_online.buffer_abs_m,
        )
        
        # Build exclude_segment_ids set from parent_seg_id
        exclude_segment_ids = {parent_seg_id}
        
        # Collect accepted sibling candidates before committing to spatial index
        accepted_siblings: List[Dict[str, Any]] = []
        
        for child_angle in child_angles:
            # Compute target position for this child
            jitter_rad = np.deg2rad(config.jitter_deg)
            angle_jitter = rng.uniform(-jitter_rad, jitter_rad)
            effective_angle = child_angle + angle_jitter
            
            lateral_dir = np.cos(effective_angle) * perp1 + np.sin(effective_angle) * perp2
            
            target_pos = (
                parent_pos
                + primary_axis * child_step
                + lateral_dir * child_spread
            )
            target_pos = self._clamp_to_domain(target_pos, domain, config.wall_margin_m)
            
            # Try to find a safe position (checking against global index only)
            # Use retry logic if enabled
            retry_radius = avg_radius
            retry_step = child_step
            collision_resolved = False
            final_target_pos = target_pos
            
            for retry_round in range(config.collision_online.fail_retry_rounds + 1):
                if retry_round > 0:
                    self._stats["retries_attempted"] += 1
                    # Adjust parameters based on retry mode
                    mode = config.collision_online.fail_retry_mode
                    if mode in ("shrink_radius", "both"):
                        retry_radius *= config.collision_online.fail_retry_shrink_factor
                    if mode in ("increase_step", "both"):
                        retry_step *= config.collision_online.fail_retry_step_boost
                    
                    # Recompute target position with adjusted step
                    retry_target = (
                        parent_pos
                        + primary_axis * retry_step
                        + lateral_dir * child_spread
                    )
                    retry_target = self._clamp_to_domain(retry_target, domain, config.wall_margin_m)
                else:
                    retry_target = target_pos
                
                final_target_pos, collision_resolved = self._find_safe_position(
                    parent_pos=parent_pos,
                    target_pos=retry_target,
                    avg_radius=retry_radius,
                    primary_axis=primary_axis,
                    perp1=perp1,
                    perp2=perp2,
                    current_spread=child_spread,
                    spatial_index=spatial_index,
                    config=config,
                    domain=domain,
                    rng=rng,
                    parent_direction=parent_direction,
                    parent_seg_id=parent_seg_id,
                )
                
                if collision_resolved:
                    if retry_round > 0:
                        self._stats["retries_succeeded"] += 1
                    break
            
            target_pos = final_target_pos
            
            if not collision_resolved:
                # Try merge_on_collision if enabled
                if config.collision_online.merge_on_collision:
                    self._stats["merges_attempted"] += 1
                    merge_result = self._find_nearest_segment_for_merge(
                        target_pos=target_pos,
                        merge_distance=config.collision_online.merge_distance_m,
                        spatial_index=spatial_index,
                        exclude_segment_ids=exclude_segment_ids,
                        domain=domain,
                        config=config,
                        merge_radius=grandchild_radius,
                    )
                    
                    if merge_result is not None:
                        target_seg_id, merge_point, _ = merge_result
                        merge_success = self._attempt_merge_to_segment(
                            network=network,
                            spatial_index=spatial_index,
                            parent_node=parent_node,
                            parent_direction=parent_direction,
                            primary_axis=primary_axis,
                            child_radius=grandchild_radius,
                            target_seg_id=target_seg_id,
                            merge_point=merge_point,
                            config=config,
                            vessel_type=vessel_type,
                        )
                        if merge_success:
                            self._stats["merges_succeeded"] += 1
                            # Merge doesn't create a recursive branch, just continue to next sibling
                            continue
                        else:
                            self._stats["merges_failed"] += 1
                    else:
                        self._stats["merges_failed"] += 1
                
                self._stats["branches_terminated"] += 1
                continue
            
            # Compute the curved path for this candidate
            centerline_points, end_direction = self._compute_curved_path(
                start=parent_pos,
                end=target_pos,
                start_dir=parent_direction,
                primary_axis=primary_axis,
                curvature=config.curvature,
                n_samples=config.curve_samples,
            )
            
            # Build polyline for sibling-sibling collision checking
            polyline_points = [parent_pos]
            polyline_points.extend(centerline_points)
            polyline_points.append(target_pos)
            
            # Check collision against already-accepted siblings using junction-safe rules
            sibling_collision = False
            for accepted in accepted_siblings:
                if self._check_sibling_collision_junction_safe(
                    polyline1=polyline_points,
                    radius1=avg_radius,
                    polyline2=accepted["polyline"],
                    radius2=accepted["radius"],
                    junction_point=parent_pos,
                    junction_ignore=junction_ignore,
                    buffer=buffer,
                ):
                    sibling_collision = True
                    break
            
            if sibling_collision:
                self._stats["collisions_detected"] += 1
                self._stats["branches_terminated"] += 1
                continue
            
            # Check segment length
            seg_length = np.linalg.norm(target_pos - parent_pos)
            if seg_length < config.min_segment_length:
                continue
            
            # Accept this sibling candidate
            accepted_siblings.append({
                "target_pos": target_pos,
                "centerline_points": centerline_points,
                "end_direction": end_direction,
                "effective_angle": effective_angle,
                "polyline": polyline_points,
                "radius": avg_radius,
            })
        
        # If no children were accepted, mark parent as terminal
        if not accepted_siblings:
            parent_node.node_type = "terminal"
            return
        
        # Commit all accepted siblings to the network and spatial index
        for sibling in accepted_siblings:
            target_pos = sibling["target_pos"]
            centerline_points = sibling["centerline_points"]
            end_direction = sibling["end_direction"]
            effective_angle = sibling["effective_angle"]
            
            # Create end node
            end_node_id = network.id_gen.next_id()
            end_node_type = "terminal" if remaining_levels == 0 else "junction"
            end_node = Node(
                id=end_node_id,
                position=Point3D(*target_pos),
                node_type=end_node_type,
                vessel_type=vessel_type,
                attributes={"radius": grandchild_radius},  # End node radius is the segment's end radius
            )
            network.add_node(end_node)
            
            # Create segment
            segment_id = network.id_gen.next_id()
            geometry = TubeGeometry(
                start=parent_node.position,
                end=end_node.position,
                radius_start=child_radius,  # Start radius of this segment
                radius_end=grandchild_radius,  # End radius (tapered)
                centerline_points=[Point3D(*p) for p in centerline_points] if centerline_points else None,
            )
            segment = VesselSegment(
                id=segment_id,
                start_node_id=parent_node.id,
                end_node_id=end_node.id,
                vessel_type=vessel_type,
                geometry=geometry,
            )
            network.add_segment(segment)
            
            self._stats["segments_created"] += 1
            
            # Insert into spatial index
            spatial_index.insert_segment(
                segment_id=segment_id,
                start=parent_pos,
                end=target_pos,
                radius=avg_radius,
                centerline=centerline_points,
            )
            
            # Store segment_id for recursive branching
            sibling["segment_id"] = segment_id
            sibling["end_node"] = end_node
            sibling["end_direction"] = end_direction
        
        # Recursively branch from all accepted children
        for sibling in accepted_siblings:
            if remaining_levels > 0:
                next_child_angles = self._compute_child_angles(
                    parent_angle=sibling["effective_angle"],
                    splits=config.splits,
                    cone_angle_deg=config.cone_angle_deg,
                    rng=rng,
                )
                
                next_child_radius = grandchild_radius  # Use pre-computed value
                next_child_step = child_step * config.step_decay
                next_child_spread = child_spread * config.spread_decay
                
                if config.splits > 1:
                    self._generate_siblings_junction_safe(
                        network=network,
                        spatial_index=spatial_index,
                        parent_node=sibling["end_node"],
                        parent_direction=sibling["end_direction"],
                        primary_axis=primary_axis,
                        child_radius=next_child_radius,
                        child_step=next_child_step,
                        child_spread=next_child_spread,
                        remaining_levels=remaining_levels - 1,
                        child_angles=next_child_angles,
                        config=config,
                        vessel_type=vessel_type,
                        domain=domain,
                        rng=rng,
                        parent_seg_id=sibling["segment_id"],
                        perp1=perp1,
                        perp2=perp2,
                    )
                else:
                    for child_angle in next_child_angles:
                        self._branch(
                            network=network,
                            spatial_index=spatial_index,
                            parent_node=sibling["end_node"],
                            parent_direction=sibling["end_direction"],
                            primary_axis=primary_axis,
                            current_radius=next_child_radius,
                            current_step=next_child_step,
                            current_spread=next_child_spread,
                            remaining_levels=remaining_levels - 1,
                            angle=child_angle,
                            config=config,
                            vessel_type=vessel_type,
                            domain=domain,
                            rng=rng,
                            parent_seg_id=sibling["segment_id"],
                        )
    
    def _check_sibling_collision_junction_safe(
        self,
        polyline1: List[np.ndarray],
        radius1: float,
        polyline2: List[np.ndarray],
        radius2: float,
        junction_point: np.ndarray,
        junction_ignore: float,
        buffer: float,
    ) -> bool:
        """
        Check if two sibling polylines collide, ignoring overlap near the junction.
        
        This implements the junction-safe rule: allow sibling overlap within a sphere
        of radius junction_ignore around the junction point. Outside that region,
        collisions are enforced normally.
        
        Parameters
        ----------
        polyline1, polyline2 : List[np.ndarray]
            The two polylines to check
        radius1, radius2 : float
            Radii of the two polylines
        junction_point : np.ndarray
            The shared parent junction point
        junction_ignore : float
            Radius of the junction ignore sphere
        buffer : float
            Additional clearance buffer
            
        Returns
        -------
        bool
            True if collision detected outside junction zone, False otherwise
        """
        min_separation = radius1 + radius2 + buffer
        
        # Check each segment of polyline1 against each segment of polyline2
        for i in range(len(polyline1) - 1):
            seg1_start = polyline1[i]
            seg1_end = polyline1[i + 1]
            
            for j in range(len(polyline2) - 1):
                seg2_start = polyline2[j]
                seg2_end = polyline2[j + 1]
                
                # Compute closest points between the two segments
                dist, closest1, closest2 = self._segment_segment_closest_points(
                    seg1_start, seg1_end, seg2_start, seg2_end
                )
                
                if dist < min_separation:
                    # Check if both closest points are within junction ignore zone
                    dist_to_junction1 = np.linalg.norm(closest1 - junction_point)
                    dist_to_junction2 = np.linalg.norm(closest2 - junction_point)
                    
                    if dist_to_junction1 < junction_ignore and dist_to_junction2 < junction_ignore:
                        # Both points are near junction - ignore this collision
                        continue
                    else:
                        # Real collision outside junction zone
                        return True
        
        return False
    
    def _segment_segment_closest_points(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
        p4: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute the closest points between two line segments.
        
        Segment 1: p1 to p2
        Segment 2: p3 to p4
        
        Returns
        -------
        Tuple[float, np.ndarray, np.ndarray]
            (distance, closest_point_on_seg1, closest_point_on_seg2)
        """
        d1 = p2 - p1  # Direction of segment 1
        d2 = p4 - p3  # Direction of segment 2
        r = p1 - p3
        
        a = np.dot(d1, d1)  # Squared length of segment 1
        e = np.dot(d2, d2)  # Squared length of segment 2
        f = np.dot(d2, r)
        
        EPSILON = 1e-10
        
        # Check if either or both segments degenerate into points
        if a <= EPSILON and e <= EPSILON:
            # Both segments degenerate into points
            return np.linalg.norm(p1 - p3), p1.copy(), p3.copy()
        
        if a <= EPSILON:
            # First segment degenerates into a point
            s = 0.0
            t = np.clip(f / e, 0.0, 1.0)
        else:
            c = np.dot(d1, r)
            if e <= EPSILON:
                # Second segment degenerates into a point
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            else:
                # General nondegenerate case
                b = np.dot(d1, d2)
                denom = a * e - b * b
                
                if denom != 0.0:
                    s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
                else:
                    s = 0.0
                
                t = (b * s + f) / e
                
                if t < 0.0:
                    t = 0.0
                    s = np.clip(-c / a, 0.0, 1.0)
                elif t > 1.0:
                    t = 1.0
                    s = np.clip((b - c) / a, 0.0, 1.0)
        
        closest1 = p1 + d1 * s
        closest2 = p3 + d2 * t
        
        return np.linalg.norm(closest1 - closest2), closest1, closest2
    
    def _find_nearest_segment_for_merge(
        self,
        target_pos: np.ndarray,
        merge_distance: float,
        spatial_index: DynamicSpatialIndex,
        exclude_segment_ids: Optional[Set[int]] = None,
        domain: Optional[DomainSpec] = None,
        config: Optional[ScaffoldTopDownConfig] = None,
        merge_radius: float = 0.0,
    ) -> Optional[Tuple[int, np.ndarray, float]]:
        """
        Find the nearest segment within merge_distance of target_pos.
        
        Also validates that the merge point maintains tube-aware boundary clearance.
        
        Parameters
        ----------
        target_pos : np.ndarray
            The target position to merge near
        merge_distance : float
            Maximum distance to consider for merge
        spatial_index : DynamicSpatialIndex
            Spatial index containing existing segments
        exclude_segment_ids : Optional[Set[int]]
            Segment IDs to exclude from consideration
        domain : Optional[DomainSpec]
            Domain for boundary clearance checking
        config : Optional[ScaffoldTopDownConfig]
            Configuration with wall_margin_m and boundary_extra_m
        merge_radius : float
            Radius of the merging branch for boundary clearance calculation
            
        Returns
        -------
        Optional[Tuple[int, np.ndarray, float]]
            (segment_id, closest_point_on_segment, distance) or None if no suitable target
        """
        # Query candidates using a small capsule around target_pos
        candidates = spatial_index.query_candidate_segments_for_capsule(
            target_pos, target_pos, merge_distance
        )
        
        if not candidates:
            return None
        
        best_seg_id = None
        best_point = None
        best_dist = float('inf')
        
        for seg_id in candidates:
            if exclude_segment_ids and seg_id in exclude_segment_ids:
                continue
            
            seg_data = spatial_index.get_segment_data(seg_id)
            if seg_data is None:
                continue
            
            seg_start = seg_data["start"]
            seg_end = seg_data["end"]
            centerline = seg_data.get("centerline")
            
            # Build polyline for the segment
            if centerline and len(centerline) > 0:
                polyline = [seg_start] + centerline + [seg_end]
            else:
                polyline = [seg_start, seg_end]
            
            # Find closest point on the polyline to target_pos
            min_dist_to_seg = float('inf')
            closest_on_seg = None
            
            for i in range(len(polyline) - 1):
                p1 = polyline[i]
                p2 = polyline[i + 1]
                
                # Project target_pos onto segment p1-p2
                d = p2 - p1
                length_sq = np.dot(d, d)
                
                if length_sq < 1e-12:
                    closest = p1.copy()
                else:
                    t = np.clip(np.dot(target_pos - p1, d) / length_sq, 0.0, 1.0)
                    closest = p1 + t * d
                
                dist = np.linalg.norm(target_pos - closest)
                if dist < min_dist_to_seg:
                    min_dist_to_seg = dist
                    closest_on_seg = closest
            
            if min_dist_to_seg < best_dist and min_dist_to_seg < merge_distance:
                # Check boundary clearance for the merge point (radial + bottom for CylinderDomain)
                if domain is not None and config is not None and closest_on_seg is not None:
                    # Check radial clearance
                    dist_to_wall = self._get_radial_boundary_distance(closest_on_seg, domain)
                    required_clearance = (
                        config.wall_margin_m 
                        + merge_radius 
                        + config.boundary_extra_m
                        + config.stop_before_boundary_m
                        + config.stop_before_boundary_extra_m
                    )
                    if dist_to_wall < required_clearance:
                        self._stats["boundary_clearance_failures"] += 1
                        continue  # Reject this merge target
                    
                    # Check bottom clearance
                    dist_to_bottom = self._get_distance_to_bottom(closest_on_seg, domain)
                    if dist_to_bottom < required_clearance:
                        self._stats["boundary_clearance_failures"] += 1
                        continue  # Reject this merge target
                
                best_dist = min_dist_to_seg
                best_point = closest_on_seg
                best_seg_id = seg_id
        
        if best_seg_id is not None:
            return (best_seg_id, best_point, best_dist)
        return None
    
    def _attempt_merge_to_segment(
        self,
        network: VascularNetwork,
        spatial_index: DynamicSpatialIndex,
        parent_node: Node,
        parent_direction: np.ndarray,
        primary_axis: np.ndarray,
        child_radius: float,
        target_seg_id: int,
        merge_point: np.ndarray,
        config: ScaffoldTopDownConfig,
        vessel_type: str,
    ) -> bool:
        """
        Attempt to merge a branch into an existing segment.
        
        This creates a new node at merge_point, splits the target segment,
        and connects the parent to the new node.
        
        Parameters
        ----------
        network : VascularNetwork
            The network to modify
        spatial_index : DynamicSpatialIndex
            Spatial index to update
        parent_node : Node
            The parent node to connect from
        parent_direction : np.ndarray
            Direction of the parent branch
        primary_axis : np.ndarray
            Primary growth axis
        child_radius : float
            Radius of the new branch
        target_seg_id : int
            ID of the segment to merge into
        merge_point : np.ndarray
            Point on the target segment to merge at
        config : ScaffoldTopDownConfig
            Configuration
        vessel_type : str
            Vessel type
            
        Returns
        -------
        bool
            True if merge succeeded, False otherwise
        """
        # Get the target segment from the network
        target_seg = network.get_segment(target_seg_id)
        if target_seg is None:
            return False
        
        # Get segment endpoints
        start_node = network.get_node(target_seg.start_node_id)
        end_node = network.get_node(target_seg.end_node_id)
        if start_node is None or end_node is None:
            return False
        
        seg_start = start_node.position.to_array()
        seg_end = end_node.position.to_array()
        
        # Check if merge_point is too close to either endpoint (avoid degenerate splits)
        dist_to_start = np.linalg.norm(merge_point - seg_start)
        dist_to_end = np.linalg.norm(merge_point - seg_end)
        min_split_dist = config.min_segment_length * 0.5
        
        if dist_to_start < min_split_dist or dist_to_end < min_split_dist:
            # Too close to endpoint - connect directly to the nearest endpoint instead
            if dist_to_start < dist_to_end:
                merge_node = start_node
            else:
                merge_node = end_node
        else:
            # Create a new node at the merge point
            merge_node_id = network.id_gen.next_id()
            
            # Interpolate radius at merge point
            total_dist = np.linalg.norm(seg_end - seg_start)
            if total_dist > 1e-9:
                t = dist_to_start / total_dist
            else:
                t = 0.5
            merge_radius = (1 - t) * target_seg.geometry.radius_start + t * target_seg.geometry.radius_end
            
            merge_node = Node(
                id=merge_node_id,
                position=Point3D(*merge_point),
                node_type="junction",
                vessel_type=vessel_type,
                attributes={"radius": merge_radius, "merged": True},
            )
            network.add_node(merge_node)
            
            # Split the target segment: create two new segments
            # Segment 1: start_node -> merge_node
            seg1_id = network.id_gen.next_id()
            seg1_geometry = TubeGeometry(
                start=start_node.position,
                end=merge_node.position,
                radius_start=target_seg.geometry.radius_start,
                radius_end=merge_radius,
                centerline_points=None,  # Simplified - no centerline for split segments
            )
            seg1 = VesselSegment(
                id=seg1_id,
                start_node_id=start_node.id,
                end_node_id=merge_node.id,
                vessel_type=vessel_type,
                geometry=seg1_geometry,
            )
            network.add_segment(seg1)
            
            # Segment 2: merge_node -> end_node
            seg2_id = network.id_gen.next_id()
            seg2_geometry = TubeGeometry(
                start=merge_node.position,
                end=end_node.position,
                radius_start=merge_radius,
                radius_end=target_seg.geometry.radius_end,
                centerline_points=None,
            )
            seg2 = VesselSegment(
                id=seg2_id,
                start_node_id=merge_node.id,
                end_node_id=end_node.id,
                vessel_type=vessel_type,
                geometry=seg2_geometry,
            )
            network.add_segment(seg2)
            
            # Update spatial index: remove old segment, add new segments
            # Note: DynamicSpatialIndex doesn't have remove, so we just add new ones
            # The old segment data remains but won't cause issues
            avg_radius1 = (target_seg.geometry.radius_start + merge_radius) / 2
            spatial_index.insert_segment(
                segment_id=seg1_id,
                start=seg_start,
                end=merge_point,
                radius=avg_radius1,
                centerline=None,
            )
            
            avg_radius2 = (merge_radius + target_seg.geometry.radius_end) / 2
            spatial_index.insert_segment(
                segment_id=seg2_id,
                start=merge_point,
                end=seg_end,
                radius=avg_radius2,
                centerline=None,
            )
            
            # Remove the original segment from the network
            network.remove_segment(target_seg_id)
        
        # Now create the merge branch: parent_node -> merge_node
        parent_pos = parent_node.position.to_array()
        merge_pos = merge_node.position.to_array()
        
        # Compute curved path for the merge branch
        centerline_points, _ = self._compute_curved_path(
            start=parent_pos,
            end=merge_pos,
            start_dir=parent_direction,
            primary_axis=primary_axis,
            curvature=config.curvature,
            n_samples=config.curve_samples,
        )
        
        # Create the merge segment
        merge_seg_id = network.id_gen.next_id()
        parent_radius = parent_node.attributes.get("radius", child_radius)
        merge_geometry = TubeGeometry(
            start=parent_node.position,
            end=merge_node.position,
            radius_start=parent_radius,
            radius_end=child_radius,
            centerline_points=[Point3D(*p) for p in centerline_points] if centerline_points else None,
        )
        merge_seg = VesselSegment(
            id=merge_seg_id,
            start_node_id=parent_node.id,
            end_node_id=merge_node.id,
            vessel_type=vessel_type,
            geometry=merge_geometry,
            attributes={"merged": True},
        )
        network.add_segment(merge_seg)
        
        self._stats["segments_created"] += 1
        
        # Add to spatial index
        avg_radius = (parent_radius + child_radius) / 2
        spatial_index.insert_segment(
            segment_id=merge_seg_id,
            start=parent_pos,
            end=merge_pos,
            radius=avg_radius,
            centerline=centerline_points,
        )
        
        return True
    
    def _get_perpendicular_axes(
        self,
        primary_axis: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get two perpendicular axes to the primary axis."""
        if abs(primary_axis[0]) < 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])
        
        perp1 = np.cross(primary_axis, ref)
        perp1 = perp1 / np.linalg.norm(perp1)
        
        perp2 = np.cross(primary_axis, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)
        
        return perp1, perp2
    
    def _clamp_to_domain(
        self,
        pos: np.ndarray,
        domain: DomainSpec,
        margin: float,
    ) -> np.ndarray:
        """
        Clamp position to stay within domain boundary with margin.
        
        Uses the DomainSpec.project_inside() API which works correctly for
        all domain types (box, cylinder, ellipsoid, mesh, etc.).
        """
        point = Point3D(*pos)
        
        # Check if point is already safely inside with sufficient margin
        if domain.contains(point):
            dist = domain.distance_to_boundary(point)
            if dist >= margin:
                return pos
        
        # Use the domain's project_inside method with margin
        # This works correctly for all domain types (box, cylinder, ellipsoid, etc.)
        projected = domain.project_inside(point, margin=margin)
        return np.array([projected.x, projected.y, projected.z])
    
    def _find_safe_position(
        self,
        parent_pos: np.ndarray,
        target_pos: np.ndarray,
        avg_radius: float,
        primary_axis: np.ndarray,
        perp1: np.ndarray,
        perp2: np.ndarray,
        current_spread: float,
        spatial_index: DynamicSpatialIndex,
        config: ScaffoldTopDownConfig,
        domain: DomainSpec,
        rng: np.random.Generator,
        parent_direction: np.ndarray,
        parent_seg_id: Optional[int] = None,
    ) -> Tuple[np.ndarray, bool]:
        """
        Find a collision-free position for the branch endpoint.
        
        Validates the ENTIRE curved polyline for collisions, not just the
        straight-line segment. This prevents false negatives where the endpoint
        is collision-free but the actual curved path collides.
        
        Tries rotation attempts around the primary axis, then reduces spread
        if still colliding.
        
        Returns
        -------
        Tuple[np.ndarray, bool]
            (final_position, success)
        """
        self._stats["segments_proposed"] += 1
        
        buffer = max(
            config.collision_online.buffer_abs_m,
            config.collision_online.buffer_rel * avg_radius,
        )
        
        # Build exclude_segment_ids set from parent_seg_id
        exclude_segment_ids = {parent_seg_id} if parent_seg_id is not None else None
        
        # Check collision on the actual curved polyline, not just straight line
        if self._check_candidate_polyline_collision(
            parent_pos=parent_pos,
            target_pos=target_pos,
            parent_direction=parent_direction,
            primary_axis=primary_axis,
            avg_radius=avg_radius,
            buffer=buffer,
            spatial_index=spatial_index,
            config=config,
            exclude_segment_ids=exclude_segment_ids,
            domain=domain,
        ):
            # Collision or boundary violation detected, try alternatives
            self._stats["collisions_detected"] += 1
        else:
            # No collision on curved path and boundary clearance OK
            return target_pos, True
        
        displacement = target_pos - parent_pos
        axial_component = np.dot(displacement, primary_axis) * primary_axis
        lateral_component = displacement - axial_component
        
        rotation_attempts = config.collision_online.rotation_attempts
        for attempt in range(rotation_attempts):
            angle_offset = 2 * np.pi * attempt / rotation_attempts
            
            cos_a = np.cos(angle_offset)
            sin_a = np.sin(angle_offset)
            
            lateral_in_perp = np.array([
                np.dot(lateral_component, perp1),
                np.dot(lateral_component, perp2),
            ])
            rotated_lateral = np.array([
                cos_a * lateral_in_perp[0] - sin_a * lateral_in_perp[1],
                sin_a * lateral_in_perp[0] + cos_a * lateral_in_perp[1],
            ])
            new_lateral = rotated_lateral[0] * perp1 + rotated_lateral[1] * perp2
            
            new_target = parent_pos + axial_component + new_lateral
            new_target = self._clamp_to_domain(new_target, domain, config.wall_margin_m)
            
            # Check collision on the actual curved polyline
            if not self._check_candidate_polyline_collision(
                parent_pos=parent_pos,
                target_pos=new_target,
                parent_direction=parent_direction,
                primary_axis=primary_axis,
                avg_radius=avg_radius,
                buffer=buffer,
                spatial_index=spatial_index,
                config=config,
                exclude_segment_ids=exclude_segment_ids,
                domain=domain,
            ):
                self._stats["rotations_successful"] += 1
                return new_target, True
        
        for reduction_factor in config.collision_online.reduction_factors:
            reduced_lateral = lateral_component * reduction_factor
            
            for attempt in range(rotation_attempts):
                angle_offset = 2 * np.pi * attempt / rotation_attempts
                
                cos_a = np.cos(angle_offset)
                sin_a = np.sin(angle_offset)
                
                lateral_in_perp = np.array([
                    np.dot(reduced_lateral, perp1),
                    np.dot(reduced_lateral, perp2),
                ])
                rotated_lateral = np.array([
                    cos_a * lateral_in_perp[0] - sin_a * lateral_in_perp[1],
                    sin_a * lateral_in_perp[0] + cos_a * lateral_in_perp[1],
                ])
                new_lateral = rotated_lateral[0] * perp1 + rotated_lateral[1] * perp2
                
                new_target = parent_pos + axial_component + new_lateral
                new_target = self._clamp_to_domain(new_target, domain, config.wall_margin_m)
                
                # Check collision on the actual curved polyline
                if not self._check_candidate_polyline_collision(
                    parent_pos=parent_pos,
                    target_pos=new_target,
                    parent_direction=parent_direction,
                    primary_axis=primary_axis,
                    avg_radius=avg_radius,
                    buffer=buffer,
                    spatial_index=spatial_index,
                    config=config,
                    exclude_segment_ids=exclude_segment_ids,
                    domain=domain,
                ):
                    self._stats["rotations_successful"] += 1
                    return new_target, True
        
        return target_pos, False
    
    def _check_candidate_polyline_collision(
        self,
        parent_pos: np.ndarray,
        target_pos: np.ndarray,
        parent_direction: np.ndarray,
        primary_axis: np.ndarray,
        avg_radius: float,
        buffer: float,
        spatial_index: DynamicSpatialIndex,
        config: ScaffoldTopDownConfig,
        exclude_segment_ids: Optional[Set[int]] = None,
        domain: Optional[DomainSpec] = None,
    ) -> bool:
        """
        Check if a candidate curved polyline collides with existing segments
        or violates boundary clearance.
        
        Computes the curved path first, then validates the entire polyline
        for both segment collisions and tube-aware boundary clearance.
        
        Parameters
        ----------
        exclude_segment_ids : Optional[Set[int]]
            Segment IDs to exclude from collision checks (e.g., parent segment).
            This is the deterministic ID-based approach instead of coordinate-based
            adjacency exclusion.
        domain : Optional[DomainSpec]
            Domain for boundary clearance checking. If provided, validates that
            all polyline points maintain sufficient clearance from the boundary
            accounting for tube radius.
        
        Returns
        -------
        bool
            True if collision detected or boundary violated, False otherwise
        """
        # Compute the candidate curved path
        centerline, _ = self._compute_curved_path(
            start=parent_pos,
            end=target_pos,
            start_dir=parent_direction,
            primary_axis=primary_axis,
            curvature=config.curvature,
            n_samples=config.curve_samples,
        )
        
        # Build the full polyline: start -> centerline points -> end
        polyline_points = [parent_pos]
        polyline_points.extend(centerline)
        polyline_points.append(target_pos)
        
        # Check boundary clearance for NEW points only (centerline + target_pos)
        # Exclude parent_pos since it's already validated (may be at inlet/boundary)
        if domain is not None:
            new_points = centerline + [target_pos]
            if new_points:
                boundary_violation = self._check_polyline_boundary_clearance(
                    polyline_points=new_points,
                    radius=avg_radius,
                    domain=domain,
                    config=config,
                )
                if boundary_violation:
                    return True
        
        # Check collision on the entire polyline using segment ID-based exclusion
        return spatial_index.check_polyline_collision(
            points=polyline_points,
            radius=avg_radius,
            buffer=buffer,
            exclude_segment_ids=exclude_segment_ids,
        )
    
    def _check_polyline_boundary_clearance(
        self,
        polyline_points: List[np.ndarray],
        radius: float,
        domain: DomainSpec,
        config: ScaffoldTopDownConfig,
    ) -> bool:
        """
        Check if any point in the polyline violates tube-aware boundary clearance.
        
        For each point, requires:
            distance_to_boundary(p) >= wall_margin_m + radius + boundary_extra_m
        
        This ensures the tube surface (not just centerline) stays within the domain.
        
        For CylinderDomain, only RADIAL clearance is checked (not axial), since
        inlets are typically at the axial faces and branches naturally start close
        to those faces. The user's concern is preventing radial breaches (tubes
        touching the cylinder wall), not axial breaches.
        
        Parameters
        ----------
        polyline_points : List[np.ndarray]
            Points along the polyline centerline
        radius : float
            Tube radius at this segment (average of start/end radii)
        domain : DomainSpec
            Domain for boundary distance calculation
        config : ScaffoldTopDownConfig
            Configuration with wall_margin_m and boundary_extra_m
        
        Returns
        -------
        bool
            True if boundary clearance violated, False otherwise
        """
        required_clearance = config.wall_margin_m + radius + config.boundary_extra_m
        
        for point in polyline_points:
            point_3d = Point3D(*point)
            
            # For CylinderDomain, only check RADIAL clearance (not axial)
            # Inlets are at axial faces, so branches naturally start close to them
            dist_to_boundary = self._get_radial_boundary_distance(point, domain)
            
            # Track minimum observed clearance for reporting
            effective_clearance = dist_to_boundary - radius
            if effective_clearance < self._stats.get("min_observed_clearance", float("inf")):
                self._stats["min_observed_clearance"] = effective_clearance
            
            if dist_to_boundary < required_clearance:
                self._stats["boundary_clearance_failures"] += 1
                return True
        
        return False
    
    def _get_radial_boundary_distance(
        self,
        point: np.ndarray,
        domain: DomainSpec,
    ) -> float:
        """
        Get the radial distance to boundary for a point.
        
        For CylinderDomain, returns only the radial distance (ignoring axial).
        For other domain types, falls back to the standard distance_to_boundary.
        
        This is used for boundary clearance checking where we only care about
        radial breaches (tubes touching the cylinder wall), not axial breaches
        (which are expected near inlet faces).
        """
        from ..core.domain import CylinderDomain
        
        if isinstance(domain, CylinderDomain):
            # Compute radial distance only
            dx = point[0] - domain.center.x
            dy = point[1] - domain.center.y
            r_xy = np.sqrt(dx**2 + dy**2)
            return float(domain.radius - r_xy)
        else:
            # Fall back to standard distance_to_boundary for other domain types
            point_3d = Point3D(*point)
            return domain.distance_to_boundary(point_3d)
    
    def _get_distance_to_bottom(
        self,
        point: np.ndarray,
        domain: DomainSpec,
    ) -> float:
        """
        Get the distance from a point to the bottom face of the domain.
        
        For CylinderDomain, returns distance to the bottom z-face.
        For other domain types, returns a large value (no bottom-zone taper).
        """
        from ..core.domain import CylinderDomain
        
        if isinstance(domain, CylinderDomain):
            half_height = domain.height / 2
            z_min = domain.center.z - half_height
            return float(point[2] - z_min)
        else:
            return float("inf")
    
    def _get_distance_to_wall(
        self,
        point: np.ndarray,
        domain: DomainSpec,
    ) -> float:
        """
        Get the distance from a point to the radial wall of the domain.
        
        For CylinderDomain, returns distance to the cylinder wall.
        For other domain types, falls back to distance_to_boundary.
        """
        from ..core.domain import CylinderDomain
        
        if isinstance(domain, CylinderDomain):
            dx = point[0] - domain.center.x
            dy = point[1] - domain.center.y
            r_xy = np.sqrt(dx**2 + dy**2)
            return float(domain.radius - r_xy)
        else:
            point_3d = Point3D(*point)
            return domain.distance_to_boundary(point_3d)
    
    def _apply_stop_before_boundary(
        self,
        target_pos: np.ndarray,
        parent_pos: np.ndarray,
        radius: float,
        domain: DomainSpec,
        config: ScaffoldTopDownConfig,
    ) -> Tuple[np.ndarray, Optional[bool]]:
        """
        Apply stop-before-boundary check with buffer.
        
        Enforces: distance_to_boundary >= wall_margin_m + radius + boundary_extra_m 
                  + stop_before_boundary_m + stop_before_boundary_extra_m
        
        Returns
        -------
        Tuple[np.ndarray, Optional[bool]]
            (adjusted_target_pos, was_adjusted)
            - was_adjusted = None means clamp_mode == "terminate" and boundary violated
            - was_adjusted = True means position was adjusted
            - was_adjusted = False means no adjustment needed
        """
        required_clearance = (
            config.wall_margin_m 
            + radius 
            + config.boundary_extra_m 
            + config.stop_before_boundary_m 
            + config.stop_before_boundary_extra_m
        )
        
        from ..core.domain import CylinderDomain
        
        adjusted_pos = target_pos.copy()
        was_adjusted = False
        
        if isinstance(domain, CylinderDomain):
            # Check bottom face clearance
            half_height = domain.height / 2
            z_min = domain.center.z - half_height
            d_bottom = adjusted_pos[2] - z_min
            
            # Track min bottom clearance
            effective_bottom_clearance = d_bottom - radius
            if effective_bottom_clearance < self._stats.get("min_bottom_clearance_observed", float("inf")):
                self._stats["min_bottom_clearance_observed"] = effective_bottom_clearance
            
            if d_bottom < required_clearance:
                if config.clamp_mode == "terminate":
                    return target_pos, None
                elif config.clamp_mode == "shorten_step":
                    # Shorten the axial component so endpoint lands at required_clearance
                    adjusted_pos[2] = z_min + required_clearance
                    was_adjusted = True
                elif config.clamp_mode == "project_inside":
                    adjusted_pos[2] = z_min + required_clearance
                    was_adjusted = True
            
            # Check radial wall clearance
            dx = adjusted_pos[0] - domain.center.x
            dy = adjusted_pos[1] - domain.center.y
            r_xy = np.sqrt(dx**2 + dy**2)
            d_wall = domain.radius - r_xy
            
            # Track min wall clearance
            effective_wall_clearance = d_wall - radius
            if effective_wall_clearance < self._stats.get("min_wall_clearance_observed", float("inf")):
                self._stats["min_wall_clearance_observed"] = effective_wall_clearance
            
            if d_wall < required_clearance and r_xy > 1e-9:
                if config.clamp_mode == "terminate":
                    return target_pos, None
                elif config.clamp_mode in ("shorten_step", "project_inside"):
                    # Scale down radial position to meet clearance
                    max_r = domain.radius - required_clearance
                    if max_r > 0:
                        scale = max_r / r_xy
                        adjusted_pos[0] = domain.center.x + dx * scale
                        adjusted_pos[1] = domain.center.y + dy * scale
                        was_adjusted = True
                    else:
                        return target_pos, None
        else:
            # For non-cylinder domains, use general distance_to_boundary
            point_3d = Point3D(*adjusted_pos)
            d_boundary = domain.distance_to_boundary(point_3d)
            if d_boundary < required_clearance:
                if config.clamp_mode == "terminate":
                    return target_pos, None
                # For other clamp modes, we can't easily adjust, so just warn
                was_adjusted = False
        
        return adjusted_pos, was_adjusted
    
    def _compute_curved_path(
        self,
        start: np.ndarray,
        end: np.ndarray,
        start_dir: np.ndarray,
        primary_axis: np.ndarray,
        curvature: float,
        n_samples: int,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Compute curved path between start and end using cubic Bezier.
        
        Returns
        -------
        Tuple[List[np.ndarray], np.ndarray]
            (list of intermediate points, end direction)
        """
        if curvature < 0.01 or n_samples < 3:
            end_dir = end - start
            norm = np.linalg.norm(end_dir)
            if norm > 1e-9:
                end_dir = end_dir / norm
            else:
                end_dir = primary_axis
            return [], end_dir
        
        dist = np.linalg.norm(end - start)
        ctrl_dist = dist * 0.4 * curvature
        
        p0 = start
        p1 = start + start_dir * ctrl_dist
        p3 = end
        
        end_to_start = start - end
        end_to_start_norm = np.linalg.norm(end_to_start)
        if end_to_start_norm > 1e-9:
            end_to_start = end_to_start / end_to_start_norm
        
        blend = 0.5 + curvature * 0.3
        p2_dir = blend * primary_axis + (1 - blend) * (-end_to_start)
        p2_dir_norm = np.linalg.norm(p2_dir)
        if p2_dir_norm > 1e-9:
            p2_dir = p2_dir / p2_dir_norm
        p2 = end - p2_dir * ctrl_dist
        
        centerline = []
        for i in range(1, n_samples - 1):
            t = i / (n_samples - 1)
            mt = 1 - t
            pt = (
                mt**3 * p0
                + 3 * mt**2 * t * p1
                + 3 * mt * t**2 * p2
                + t**3 * p3
            )
            centerline.append(pt)
        
        if len(centerline) >= 2:
            end_dir = centerline[-1] - centerline[-2]
        else:
            end_dir = p3 - p2
        
        norm = np.linalg.norm(end_dir)
        if norm > 1e-9:
            end_dir = end_dir / norm
        else:
            end_dir = primary_axis
        
        return centerline, end_dir
    
    def _compute_child_angles(
        self,
        parent_angle: float,
        splits: int,
        cone_angle_deg: float,
        rng: np.random.Generator,
    ) -> List[float]:
        """Compute angles for child branches."""
        if splits == 1:
            return [parent_angle]
        
        base_spacing = 2 * np.pi / splits
        
        child_angles = []
        for i in range(splits):
            base_angle = i * base_spacing
            
            max_jitter = min(np.deg2rad(cone_angle_deg) * 0.3, base_spacing * 0.3)
            jitter = rng.uniform(-max_jitter, max_jitter)
            
            child_angles.append(base_angle + jitter)
        
        return child_angles
    
    def get_generation_stats(self) -> Dict[str, int]:
        """Return statistics from the last generation run."""
        return dict(self._stats) if hasattr(self, "_stats") else {}
