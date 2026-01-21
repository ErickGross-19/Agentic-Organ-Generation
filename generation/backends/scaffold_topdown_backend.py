"""
Scaffold top-down branching backend with online collision avoidance.

This backend generates vascular networks using recursive top-down branching
with configurable splits, levels, tapering, and collision avoidance.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
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
    ) -> None:
        """
        Recursively create branches with collision avoidance.
        
        This is the core branching algorithm that:
        1. Computes target endpoint with spread and curvature
        2. Checks for collisions and attempts rotation/reduction if needed
        3. Creates curved polyline segments
        4. Recursively branches children
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
        
        target_pos = (
            parent_pos
            + primary_axis * current_step
            + lateral_dir * target_spread
        )
        
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
                )
    
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
        ):
            # Collision detected, try alternatives
            self._stats["collisions_detected"] += 1
        else:
            # No collision on curved path
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
    ) -> bool:
        """
        Check if a candidate curved polyline collides with existing segments.
        
        Computes the curved path first, then validates the entire polyline.
        
        Returns
        -------
        bool
            True if collision detected, False otherwise
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
        
        # Check collision on the entire polyline
        return spatial_index.check_polyline_collision(
            points=polyline_points,
            radius=avg_radius,
            buffer=buffer,
            exclude_adjacent_to=parent_pos,
        )
    
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
