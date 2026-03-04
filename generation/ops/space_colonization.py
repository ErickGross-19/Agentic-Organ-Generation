"""
Space colonization algorithm for organic vascular growth.

This module implements a policy-driven space colonization algorithm that produces
tree-like vascular structures by:
- Preventing "inlet starburst" (root spawning many children immediately)
- Enabling proper branching when attractor field supports it
- Using trunk-first growth with apical dominance and angular-clustering-based splitting

All behavior is controlled via SpaceColonizationPolicy - no hidden constants.
Behavior is reproducible when seed is fixed.
Max split degree per node <= 3.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict, Tuple
import logging
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from ..core.types import Point3D, Direction3D
from ..core.network import VascularNetwork
from ..core.result import OperationResult, OperationStatus, Delta
from ..rules.constraints import BranchingConstraints
from .growth import grow_branch
from ._gpu_nn import (
    nearest_neighbor as _nn_query,
    range_search as _range_search,
    vectorized_direction_average as _vec_dir_avg,
    batch_collision_prefilter,
    batch_direction_average,
    PersistentGPUIndex,
)
from ._spatial_hash import SpatialHash
from ..spatial.grid_index import DynamicSpatialIndex

_logger = logging.getLogger(__name__)


@dataclass
class SpaceColonizationParams:
    """Parameters for space colonization algorithm."""
    
    influence_radius: float = 0.015  # 15mm - radius within which tissue points attract tips
    kill_radius: float = 0.003  # 3mm - radius within which tissue points are "perfused"
    step_size: float = 0.005  # 5mm - growth step size
    min_radius: float = 0.0003  # 0.3mm - minimum vessel radius
    taper_factor: float = 0.95  # Radius reduction per generation
    vessel_type: str = "arterial"
    max_steps: int = 100  # Maximum growth steps per call
    grow_from_terminals_only: bool = False  # If True, only grow from terminal nodes (not inlet/outlet)
    
    preferred_direction: Optional[tuple] = None  # (x, y, z) preferred growth direction
    directional_bias: float = 0.0  # 0-1: weight for preferred direction (0=pure attraction, 1=pure directional)
    max_deviation_deg: float = 180.0  # Maximum angle deviation from preferred direction (hard constraint)
    smoothing_weight: float = 0.2  # 0-1: weight for previous direction smoothing
    
    encourage_bifurcation: bool = False  # Whether to encourage multiple children per node
    min_attractions_for_bifurcation: int = 3  # Minimum attraction points needed to consider bifurcation
    max_children_per_node: int = 2  # Maximum children to create (typically 2 for bifurcation)
    bifurcation_angle_threshold_deg: float = 40.0  # Minimum angle spread to trigger bifurcation
    bifurcation_probability: float = 0.7  # Probability of bifurcating when conditions are met
    
    # Phase 1b: Quality constraints
    max_curvature_deg: Optional[float] = None  # Maximum curvature angle (None = no limit)
    min_clearance: Optional[float] = None  # Minimum clearance from other segments (None = no check)
    collision_mode: str = "break"  # "break" (stop), "deflect" (steer away), "merge" (connect to nearby)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "influence_radius": self.influence_radius,
            "kill_radius": self.kill_radius,
            "step_size": self.step_size,
            "min_radius": self.min_radius,
            "taper_factor": self.taper_factor,
            "vessel_type": self.vessel_type,
            "max_steps": self.max_steps,
            "preferred_direction": self.preferred_direction,
            "directional_bias": self.directional_bias,
            "max_deviation_deg": self.max_deviation_deg,
            "smoothing_weight": self.smoothing_weight,
            "encourage_bifurcation": self.encourage_bifurcation,
            "min_attractions_for_bifurcation": self.min_attractions_for_bifurcation,
            "max_children_per_node": self.max_children_per_node,
            "bifurcation_angle_threshold_deg": self.bifurcation_angle_threshold_deg,
            "bifurcation_probability": self.bifurcation_probability,
            "max_curvature_deg": self.max_curvature_deg,
            "min_clearance": self.min_clearance,
            "collision_mode": self.collision_mode,
            "grow_from_terminals_only": self.grow_from_terminals_only,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "SpaceColonizationParams":
        """Create from dictionary."""
        return cls(
            influence_radius=d.get("influence_radius", 0.015),
            kill_radius=d.get("kill_radius", 0.003),
            step_size=d.get("step_size", 0.005),
            min_radius=d.get("min_radius", 0.0003),
            taper_factor=d.get("taper_factor", 0.95),
            vessel_type=d.get("vessel_type", "arterial"),
            max_steps=d.get("max_steps", 100),
            preferred_direction=d.get("preferred_direction", None),
            directional_bias=d.get("directional_bias", 0.0),
            max_deviation_deg=d.get("max_deviation_deg", 180.0),
            smoothing_weight=d.get("smoothing_weight", 0.2),
            encourage_bifurcation=d.get("encourage_bifurcation", False),
            min_attractions_for_bifurcation=d.get("min_attractions_for_bifurcation", 3),
            max_children_per_node=d.get("max_children_per_node", 2),
            bifurcation_angle_threshold_deg=d.get("bifurcation_angle_threshold_deg", 40.0),
            bifurcation_probability=d.get("bifurcation_probability", 0.7),
            max_curvature_deg=d.get("max_curvature_deg"),
            min_clearance=d.get("min_clearance"),
            collision_mode=d.get("collision_mode", "break"),
            grow_from_terminals_only=d.get("grow_from_terminals_only", False),
        )


def space_colonization_step(
    network: VascularNetwork,
    tissue_points: np.ndarray,
    params: Optional[SpaceColonizationParams] = None,
    constraints: Optional[BranchingConstraints] = None,
    seed: Optional[int] = None,
    seed_nodes: Optional[List[str]] = None,
) -> OperationResult:
    """
    Perform space colonization growth step.
    
    This algorithm grows vascular networks towards tissue points that need
    perfusion, creating organic space-filling patterns.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to grow
    tissue_points : np.ndarray
        Array of tissue points (N, 3) that need perfusion
    params : SpaceColonizationParams, optional
        Algorithm parameters
    constraints : BranchingConstraints, optional
        Branching constraints
    seed : int, optional
        Random seed
    seed_nodes : List[str], optional
        List of node IDs to use as seed nodes for growth. If None, uses all
        inlet/outlet nodes of the specified vessel type (default behavior)
    
    Returns
    -------
    result : OperationResult
        Result with metadata about growth progress
    
    Algorithm
    ---------
    1. For each tissue point, find nearest terminal node within influence_radius
    2. For each terminal node, compute average direction to its attracted tissue points
    3. Grow each terminal node in its attraction direction
    4. Remove tissue points within kill_radius of any node (they're "perfused")
    5. Repeat until no tissue points remain or no growth possible
    """
    if params is None:
        params = SpaceColonizationParams()
    
    if constraints is None:
        # Create constraints with min_segment_length equal to step_size
        # This ensures segments are at least as long as the growth step
        # Callers should pass explicit constraints with policy-driven min_segment_length
        constraints = BranchingConstraints(
            min_segment_length=params.step_size,
            min_radius=params.min_radius,
            collision_min_clearance=params.min_clearance if params.min_clearance is not None else 0.001,
        )
    
    rng = np.random.default_rng(seed) if seed is not None else network.id_gen.rng
    
    if seed_nodes is not None:
        terminal_nodes = [
            network.nodes[node_id] for node_id in seed_nodes
            if node_id in network.nodes and network.nodes[node_id].vessel_type == params.vessel_type
        ]
        if params.grow_from_terminals_only:
            terminal_nodes = [
                node for node in terminal_nodes
                if node.node_type == "terminal"
            ]
    elif params.grow_from_terminals_only:
        # Only grow from terminal nodes (exclude inlet/outlet)
        terminal_nodes = [
            node for node in network.nodes.values()
            if node.node_type == "terminal" and
            node.vessel_type == params.vessel_type
        ]
    else:
        terminal_nodes = [
            node for node in network.nodes.values()
            if node.node_type in ("terminal", "inlet", "outlet") and
            node.vessel_type == params.vessel_type
        ]
    
    if not terminal_nodes:
        return OperationResult.failure(
            message=f"No terminal nodes of type {params.vessel_type} found",
            errors=["No terminal nodes"],
        )
    
    if isinstance(tissue_points, np.ndarray) and tissue_points.ndim == 2:
        tissue_points_array = tissue_points.astype(np.float64, copy=False)
    else:
        tissue_points_array = np.array([
            [p.x, p.y, p.z] if isinstance(p, Point3D) else [p[0], p[1], p[2]]
            for p in tissue_points
        ], dtype=np.float64)
    active_tissue_points = set(range(len(tissue_points_array)))
    initial_count = len(tissue_points_array)
    
    new_node_ids = []
    new_segment_ids = []
    warnings = []
    steps_taken = 0
    
    _logger.info(
        "SC init: %d terminal nodes, %d tissue points, influence_radius=%.4f, "
        "kill_radius=%.4f, step_size=%.6f, max_steps=%d",
        len(terminal_nodes), initial_count, params.influence_radius,
        params.kill_radius, params.step_size, params.max_steps,
    )
    
    cell_size = max(params.step_size * 3, 0.001)
    _sc_spatial_index = DynamicSpatialIndex(cell_size=cell_size)
    for seg_id, seg in network.segments.items():
        start = np.array([seg.geometry.start.x, seg.geometry.start.y, seg.geometry.start.z])
        end = np.array([seg.geometry.end.x, seg.geometry.end.y, seg.geometry.end.z])
        radius = seg.geometry.mean_radius()
        _sc_spatial_index.insert_segment(seg_id, start, end, radius)
    
    _sc_parent_of: dict = {}
    _sc_seg_node_map: dict = {}
    for seg in network.segments.values():
        _sc_parent_of[seg.end_node_id] = seg.start_node_id
        _sc_seg_node_map[seg.id] = (seg.start_node_id, seg.end_node_id)
    _sc_max_radius = max(
        (seg.geometry.mean_radius() for seg in network.segments.values()),
        default=0.001,
    )
    _sc_step_sz = params.step_size if params.step_size > 0 else 0.0001
    _sc_clearance = params.min_clearance if params.min_clearance is not None else constraints.collision_min_clearance
    _sc_excl_depth = max(int((2 * _sc_max_radius + _sc_clearance) / _sc_step_sz) + 5, 10)
    
    pbar = tqdm(total=params.max_steps, desc="Space colonization", unit="step")
    
    for step in range(params.max_steps):
        if not active_tissue_points:
            _logger.info("SC step %d: no active tissue points remain, stopping", step)
            pbar.close()
            break
        
        if seed_nodes is not None:
            terminal_nodes = [
                node for node in network.nodes.values()
                if (node.id in seed_nodes or node.id in new_node_ids) and
                node.node_type in ("terminal", "inlet", "outlet") and
                node.vessel_type == params.vessel_type
            ]
            if params.grow_from_terminals_only:
                terminal_nodes = [
                    node for node in terminal_nodes
                    if node.node_type == "terminal"
                ]
        elif params.grow_from_terminals_only:
            # Only grow from terminal nodes (exclude inlet/outlet)
            terminal_nodes = [
                node for node in network.nodes.values()
                if node.node_type == "terminal" and
                node.vessel_type == params.vessel_type
            ]
        else:
            terminal_nodes = [
                node for node in network.nodes.values()
                if node.node_type in ("terminal", "inlet", "outlet") and
                node.vessel_type == params.vessel_type
            ]
        
        attractions: Dict[int, List[int]] = {node.id: [] for node in terminal_nodes}
        
        if terminal_nodes:
            terminal_positions = np.array([
                [node.position.x, node.position.y, node.position.z]
                for node in terminal_nodes
            ])
            terminal_id_list = [node.id for node in terminal_nodes]
            terminal_id_arr = np.array(terminal_id_list)
            
            active_tp_indices = np.array(list(active_tissue_points), dtype=np.intp)
            if len(active_tp_indices) > 0:
                active_tp_positions = tissue_points_array[active_tp_indices]
                
                distances, nearest_indices = _nn_query(active_tp_positions, terminal_positions, k=1)
                
                within_range = distances < params.influence_radius
                valid_tp = active_tp_indices[within_range]
                valid_nearest = terminal_id_arr[nearest_indices[within_range]]
                for tp_idx, tid in zip(valid_tp, valid_nearest):
                    attractions[int(tid)].append(int(tp_idx))
                
                if step == 0:
                    n_attracted = int(within_range.sum())
                    _logger.info(
                        "SC step 0: %d/%d attractors within influence_radius (%.4f) of %d tips",
                        n_attracted, len(active_tp_indices), params.influence_radius, len(terminal_nodes),
                    )
                    if n_attracted == 0:
                        _logger.warning(
                            "SC step 0: NO attractors within influence_radius! "
                            "Nearest attractor is %.6f away (influence_radius=%.4f). "
                            "Consider increasing attraction_distance or num_attractors.",
                            float(distances.min()) if len(distances) > 0 else float('inf'),
                            params.influence_radius,
                        )
        
        grown_any = False
        _step_bif = {"attempted": 0, "angle_low": 0, "prob_skip": 0, "clearance_fail": 0, "grow_fail": 0, "success": 0}
        for node in terminal_nodes:
            if not attractions[node.id]:
                continue
            
            attracted_indices = attractions[node.id]
            attracted_positions = tissue_points_array[attracted_indices]
            num_attractions = len(attracted_indices)
            
            # Check if bifurcation conditions are met
            should_bifurcate = (
                params.encourage_bifurcation and
                num_attractions >= params.min_attractions_for_bifurcation
            )
            
            if should_bifurcate:
                _step_bif["attempted"] += 1
                node_pos = np.array([node.position.x, node.position.y, node.position.z])
                raw_dirs = attracted_positions - node_pos
                dir_norms = np.linalg.norm(raw_dirs, axis=1)
                valid_mask = dir_norms > 1e-10
                attraction_vectors = list(raw_dirs[valid_mask] / dir_norms[valid_mask, np.newaxis])
                
                if len(attraction_vectors) >= 2:
                    angle_spread = _compute_angle_spread(attraction_vectors)
                    
                    if angle_spread >= params.bifurcation_angle_threshold_deg:
                        if rng.random() < params.bifurcation_probability:
                            # Cluster attractions
                            clusters = _cluster_attractions_by_angle(
                                attraction_vectors,
                                max_clusters=min(params.max_children_per_node, len(attraction_vectors))
                            )
                            
                            parent_radius = node.attributes.get("radius", params.min_radius * 2)
                            
                            n_children = len(clusters)
                            if n_children > 1:
                                child_radii = [parent_radius * (1.0 / n_children) ** (1.0/3.0) * params.taper_factor 
                                             for _ in range(n_children)]
                            else:
                                child_radii = [parent_radius * params.taper_factor]
                            
                            for cluster_idx, cluster in enumerate(clusters):
                                if cluster_idx >= params.max_children_per_node:
                                    break
                                
                                # Compute average direction for this cluster
                                cluster_direction = np.mean([attraction_vectors[i] for i in cluster], axis=0)
                                cluster_direction = cluster_direction / np.linalg.norm(cluster_direction)
                                
                                # Apply directional blending and curvature constraints
                                cluster_direction = _apply_directional_blending(cluster_direction, node, params)
                                cluster_direction = _apply_curvature_constraint(cluster_direction, node, params)
                                
                                growth_direction = Direction3D.from_array(cluster_direction)
                                
                                new_radius = child_radii[cluster_idx]
                                # Policy-driven clamping: clamp to min_radius instead of skipping
                                new_radius = max(new_radius, params.min_radius)
                                
                                result = grow_branch(
                                    network,
                                    from_node_id=node.id,
                                    length=params.step_size,
                                    direction=growth_direction,
                                    target_radius=new_radius,
                                    constraints=constraints,
                                    check_collisions=True,
                                    seed=seed,
                                    spatial_index=_sc_spatial_index,
                                    collision_mode=params.collision_mode,
                                    _parent_of=_sc_parent_of,
                                    _seg_node_map=_sc_seg_node_map,
                                    _excl_depth=_sc_excl_depth,
                                )
                                
                                if result.is_success():
                                    _step_bif["success"] += 1
                                    _new_seg_id = result.new_ids["segment"]
                                    new_segment_ids.append(_new_seg_id)
                                    _seg = network.segments.get(_new_seg_id)
                                    if _seg is not None:
                                        _sc_spatial_index.insert_segment(
                                            _new_seg_id,
                                            np.array([_seg.geometry.start.x, _seg.geometry.start.y, _seg.geometry.start.z]),
                                            np.array([_seg.geometry.end.x, _seg.geometry.end.y, _seg.geometry.end.z]),
                                            _seg.geometry.mean_radius(),
                                        )
                                        _sc_parent_of[_seg.end_node_id] = _seg.start_node_id
                                        _sc_seg_node_map[_seg.id] = (_seg.start_node_id, _seg.end_node_id)
                                    if not result.new_ids.get("merged"):
                                        new_node_ids.append(result.new_ids["node"])
                                    grown_any = True
                                else:
                                    _step_bif["grow_fail"] += 1
                                    if result.errors:
                                        _logger.warning(
                                            "SC step %d: grow_branch failed (tip=%s): %s",
                                            step, node.id, "; ".join(result.errors[:3]),
                                        )
                                    warnings.extend(result.errors)
                            
                            continue
                        else:
                            _step_bif["prob_skip"] += 1
                    else:
                        _step_bif["angle_low"] += 1
            
            node_pos_arr = np.array([node.position.x, node.position.y, node.position.z])
            raw_directions = attracted_positions - node_pos_arr
            direction_norms = np.linalg.norm(raw_directions, axis=1)
            valid = direction_norms > 1e-10
            if not np.any(valid):
                continue
            avg_direction = (raw_directions[valid] / direction_norms[valid, np.newaxis]).sum(axis=0)
            
            if np.linalg.norm(avg_direction) < 1e-10:
                continue
            
            avg_direction = avg_direction / np.linalg.norm(avg_direction)
            
            avg_direction = _apply_directional_blending(avg_direction, node, params)
            avg_direction = _apply_curvature_constraint(avg_direction, node, params)
            
            growth_direction = Direction3D.from_array(avg_direction)
            
            new_pos = Point3D(
                node.position.x + growth_direction.dx * params.step_size,
                node.position.y + growth_direction.dy * params.step_size,
                node.position.z + growth_direction.dz * params.step_size,
            )
            
            if not _check_clearance(new_pos, network, node.id, params,
                                    _excl_depth=_sc_excl_depth,
                                    _parent_of=_sc_parent_of,
                                    _seg_node_map=_sc_seg_node_map):
                continue
            
            parent_radius = node.attributes.get("radius", params.min_radius * 2)
            new_radius = parent_radius * params.taper_factor
            
            # Policy-driven clamping: clamp to min_radius instead of skipping growth
            # This ensures growth continues even when taper would drop below min_radius
            new_radius = max(new_radius, params.min_radius)
            
            result = grow_branch(
                network,
                from_node_id=node.id,
                length=params.step_size,
                direction=growth_direction,
                target_radius=new_radius,
                constraints=constraints,
                check_collisions=True,
                seed=seed,
                spatial_index=_sc_spatial_index,
                collision_mode=params.collision_mode,
                _parent_of=_sc_parent_of,
                _seg_node_map=_sc_seg_node_map,
                _excl_depth=_sc_excl_depth,
            )
            
            if result.is_success():
                _new_seg_id = result.new_ids["segment"]
                new_segment_ids.append(_new_seg_id)
                _seg = network.segments.get(_new_seg_id)
                if _seg is not None:
                    _sc_spatial_index.insert_segment(
                        _new_seg_id,
                        np.array([_seg.geometry.start.x, _seg.geometry.start.y, _seg.geometry.start.z]),
                        np.array([_seg.geometry.end.x, _seg.geometry.end.y, _seg.geometry.end.z]),
                        _seg.geometry.mean_radius(),
                    )
                    _sc_parent_of[_seg.end_node_id] = _seg.start_node_id
                    _sc_seg_node_map[_seg.id] = (_seg.start_node_id, _seg.end_node_id)
                if not result.new_ids.get("merged"):
                    new_node_ids.append(result.new_ids["node"])
                grown_any = True
            else:
                if result.errors:
                    _logger.warning(
                        "SC step %d: grow_branch failed (tip=%s): %s",
                        step, node.id, "; ".join(result.errors[:3]),
                    )
                warnings.extend(result.errors)
        
        if step % 50 == 0 or not grown_any:
            _logger.info(
                "SC step %d bifurcation: attempted=%d angle_low=%d prob_skip=%d clearance_fail=%d grow_fail=%d success=%d",
                step, _step_bif["attempted"], _step_bif["angle_low"], _step_bif["prob_skip"],
                _step_bif["clearance_fail"], _step_bif["grow_fail"], _step_bif["success"],
            )
        
        if not grown_any:
            n_total_attracted = sum(len(v) for v in attractions.values())
            _logger.warning(
                "SC step %d: no growth occurred (attracted=%d, tips=%d). Stopping.",
                step, n_total_attracted, len(terminal_nodes),
            )
            if n_total_attracted == 0:
                min_dist = float('inf')
                try:
                    if len(distances) > 0:
                        min_dist = float(distances.min())
                except NameError:
                    pass
                _logger.warning(
                    "No attractors within influence_radius (%.4f). Min distance to nearest tip: %.6f",
                    params.influence_radius, min_dist,
                )
            pbar.close()
            break
        
        steps_taken += 1
        pbar.update(1)
        pbar.set_postfix({
            'nodes': len(new_node_ids),
            'coverage': f'{(initial_count - len(active_tissue_points)) / initial_count:.1%}' if initial_count > 0 else '0%'
        })
        
        if network.nodes and active_tissue_points:
            all_node_positions = np.array([
                [n.position.x, n.position.y, n.position.z]
                for n in network.nodes.values()
            ])
            
            active_tp_idx_arr = np.array(list(active_tissue_points), dtype=np.intp)
            active_tp_positions = tissue_points_array[active_tp_idx_arr]
            
            if len(all_node_positions) > 5000:
                sh = SpatialHash(params.kill_radius)
                sh.build(all_node_positions)
                kill_mask = sh.has_neighbor_mask(
                    active_tp_positions, all_node_positions,
                    params.kill_radius,
                )
            else:
                kill_mask = _range_search(active_tp_positions, all_node_positions, params.kill_radius)
            
            killed_indices = active_tp_idx_arr[kill_mask]
            active_tissue_points -= set(killed_indices.tolist())
    
    pbar.close()
    
    perfused_count = initial_count - len(active_tissue_points)
    coverage_fraction = perfused_count / initial_count if initial_count > 0 else 0.0
    
    delta = Delta(
        created_node_ids=new_node_ids,
        created_segment_ids=new_segment_ids,
    )
    
    if new_node_ids:
        status = OperationStatus.SUCCESS if not warnings else OperationStatus.PARTIAL_SUCCESS
        message = f"Grew {len(new_node_ids)} nodes in {steps_taken} steps, {coverage_fraction:.1%} coverage"
    else:
        status = OperationStatus.WARNING
        message = "No growth occurred"
    
    return OperationResult(
        status=status,
        message=message,
        new_ids={
            "nodes": new_node_ids,
            "segments": new_segment_ids,
        },
        warnings=warnings,
        delta=delta,
        rng_state=network.id_gen.get_state(),
        metadata={
            "steps_taken": steps_taken,
            "nodes_grown": len(new_node_ids),
            "initial_tissue_points": initial_count,
            "perfused_tissue_points": perfused_count,
            "coverage_fraction": coverage_fraction,
        },
    )


def _compute_angle_spread(vectors: List[np.ndarray]) -> float:
    """
    Compute maximum pairwise angle between unit vectors.
    
    Returns angle in degrees.
    """
    if len(vectors) < 2:
        return 0.0
    
    mat = np.array(vectors)
    cos_matrix = np.clip(mat @ mat.T, -1.0, 1.0)
    upper_indices = np.triu_indices_from(cos_matrix, k=1)
    if len(upper_indices[0]) == 0:
        return 0.0
    min_cos = cos_matrix[upper_indices].min()
    return float(np.degrees(np.arccos(min_cos)))


def _cluster_attractions_by_angle(
    attraction_vectors: List[np.ndarray],
    max_clusters: int = 2,
) -> List[List[int]]:
    """
    Cluster attraction vectors into groups using k-means with farthest-first initialization.
    
    Returns list of cluster indices (each cluster is a list of vector indices).
    """
    n = len(attraction_vectors)
    
    if n == 0:
        return []
    if n == 1:
        return [[0]]
    if max_clusters <= 1:
        return [[i for i in range(n)]]
    
    vec_array = np.array(attraction_vectors)
    norms = np.linalg.norm(vec_array, axis=1, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1.0)
    normalized_mat = vec_array / norms
    
    if n <= max_clusters:
        return [[i] for i in range(n)]
    
    K = min(max_clusters, n)
    
    similarity_matrix = normalized_mat @ normalized_mat.T
    
    centroid_indices = [0]
    centroids = [normalized_mat[0].copy()]
    
    for _ in range(K - 1):
        sims_to_centroids = similarity_matrix[:, centroid_indices]
        min_sims = sims_to_centroids.min(axis=1)
        dists = 1.0 - min_sims
        dists[centroid_indices] = -1.0
        farthest_idx = int(np.argmax(dists))
        centroids.append(normalized_mat[farthest_idx].copy())
        centroid_indices.append(farthest_idx)
    
    centroid_mat = np.array(centroids)
    for iteration in range(10):
        assignments = np.argmax(normalized_mat @ centroid_mat.T, axis=1)
        
        clusters = [[] for _ in range(K)]
        for idx, c in enumerate(assignments):
            clusters[c].append(idx)
        
        changed = False
        for c in range(K):
            if clusters[c]:
                new_centroid = normalized_mat[clusters[c]].mean(axis=0)
                centroid_norm = np.linalg.norm(new_centroid)
                
                if centroid_norm > 1e-10:
                    new_centroid = new_centroid / centroid_norm
                    
                    if np.linalg.norm(new_centroid - centroid_mat[c]) > 1e-6:
                        changed = True
                        centroid_mat[c] = new_centroid
        
        if not changed:
            break
    
    clusters = [c for c in clusters if c]
    
    return clusters if clusters else [[i for i in range(n)]]


def _apply_directional_blending(
    avg_direction: np.ndarray,
    node,
    params: SpaceColonizationParams,
) -> np.ndarray:
    """Apply directional constraint blending to a growth direction."""
    if params.preferred_direction is not None and params.directional_bias > 0:
        d_pref = np.array(params.preferred_direction)
        d_pref = d_pref / np.linalg.norm(d_pref)
        
        d_prev = None
        if "direction" in node.attributes and params.smoothing_weight > 0:
            prev_dir = Direction3D.from_dict(node.attributes["direction"])
            d_prev = prev_dir.to_array()
        
        v_attr = avg_direction
        beta = params.directional_bias
        w_prev = params.smoothing_weight if d_prev is not None else 0.0
        
        if d_prev is not None:
            blended = (1 - beta - w_prev) * v_attr + beta * d_pref + w_prev * d_prev
        else:
            blended = (1 - beta) * v_attr + beta * d_pref
        
        blended_norm = np.linalg.norm(blended)
        if blended_norm > 1e-10:
            blended = blended / blended_norm
        else:
            blended = d_pref
        
        if params.max_deviation_deg < 180.0:
            angle_to_pref = np.arccos(np.clip(np.dot(blended, d_pref), -1.0, 1.0))
            max_angle_rad = np.radians(params.max_deviation_deg)
            
            if angle_to_pref > max_angle_rad:
                axis = np.cross(blended, d_pref)
                axis_norm = np.linalg.norm(axis)
                
                if axis_norm > 1e-10:
                    axis = axis / axis_norm
                    rotation_angle = angle_to_pref - max_angle_rad
                    cos_rot = np.cos(rotation_angle)
                    sin_rot = np.sin(rotation_angle)
                    
                    blended = (blended * cos_rot +
                             np.cross(axis, blended) * sin_rot +
                             axis * np.dot(axis, blended) * (1 - cos_rot))
                    blended = blended / np.linalg.norm(blended)
                else:
                    blended = d_pref
        
        return blended
    
    return avg_direction


def _apply_curvature_constraint(
    growth_direction: np.ndarray,
    node,
    params: SpaceColonizationParams,
) -> np.ndarray:
    """
    Apply maximum curvature constraint to growth direction.
    
    If the node has a previous direction and max_curvature_deg is set,
    constrains the new direction to not exceed the maximum bend angle.
    """
    if params.max_curvature_deg is None:
        return growth_direction
    
    # Get previous direction
    if "direction" not in node.attributes:
        return growth_direction  # No previous direction, no constraint
    
    prev_dir = Direction3D.from_dict(node.attributes["direction"])
    d_prev = prev_dir.to_array()
    
    # Compute angle between previous and proposed direction
    cos_angle = np.clip(np.dot(d_prev, growth_direction), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(abs(cos_angle)))
    
    # If within limit, return as-is
    if angle_deg <= params.max_curvature_deg:
        return growth_direction
    
    # Project growth_direction onto cone around d_prev
    max_angle_rad = np.radians(params.max_curvature_deg)
    
    # Rotation axis: perpendicular to both vectors
    axis = np.cross(d_prev, growth_direction)
    axis_norm = np.linalg.norm(axis)
    
    if axis_norm < 1e-10:
        # Vectors are parallel or anti-parallel
        return d_prev if cos_angle > 0 else -d_prev
    
    axis = axis / axis_norm
    
    # Rotate d_prev by max_angle_rad around axis
    cos_rot = np.cos(max_angle_rad)
    sin_rot = np.sin(max_angle_rad)
    
    constrained = (d_prev * cos_rot +
                   np.cross(axis, d_prev) * sin_rot +
                   axis * np.dot(axis, d_prev) * (1 - cos_rot))
    
    return constrained / np.linalg.norm(constrained)


def _check_clearance(
    new_position: Point3D,
    network: VascularNetwork,
    from_node_id: int,
    params: SpaceColonizationParams,
    _children_by_node: Optional[dict] = None,
    _seg_by_node: Optional[dict] = None,
    _excl_depth: Optional[int] = None,
    _parent_of: Optional[dict] = None,
    _seg_node_map: Optional[dict] = None,
) -> bool:
    """
    Check if new position maintains minimum clearance from other segments.
    
    Uses SpatialIndex for efficient local neighborhood queries instead of
    scanning all segments (O(local) instead of O(segments)).
    
    When _parent_of and _seg_node_map are provided, uses lazy per-candidate
    ancestry check (O(candidates * excl_depth)) instead of the exhaustive
    exclusion-set walk (O(tree_depth * subtree_size)).
    
    Returns True if clearance is acceptable, False otherwise.
    """
    if params.min_clearance is None:
        return True

    if _excl_depth is None:
        max_radius = max(
            (seg.mean_radius for seg in network.segments.values()),
            default=0.001,
        )
        step_sz = params.step_size if params.step_size > 0 else 0.0001
        clearance = params.min_clearance if params.min_clearance is not None else 0.0
        _excl_depth = max(int((2 * max_radius + clearance) / step_sz) + 5, 10)

    use_lazy = _parent_of is not None and _seg_node_map is not None

    if use_lazy:
        tip_ancestors: set = set()
        cur = from_node_id
        for _ in range(_excl_depth + 1):
            tip_ancestors.add(cur)
            nxt = _parent_of.get(cur)
            if nxt is None:
                break
            cur = nxt

        search_radius = params.min_clearance * 3.0
        spatial_index = network.get_spatial_index()
        nearby_segments = spatial_index.query_nearby_segments(new_position, search_radius)

        for seg in nearby_segments:
            if seg.id in _seg_node_map:
                cand_start_nid, cand_end_nid = _seg_node_map[seg.id]
                skip = False
                for cand_nid in (cand_start_nid, cand_end_nid):
                    cur = cand_nid
                    for _ in range(_excl_depth + 1):
                        if cur in tip_ancestors:
                            skip = True
                            break
                        nxt = _parent_of.get(cur)
                        if nxt is None:
                            break
                        cur = nxt
                    if skip:
                        break
                if skip:
                    continue

            p1 = network.nodes[seg.start_node_id].position
            p2 = network.nodes[seg.end_node_id].position

            v = np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])
            w = np.array([new_position.x - p1.x, new_position.y - p1.y, new_position.z - p1.z])

            v_len_sq = np.dot(v, v)
            if v_len_sq < 1e-10:
                dist = np.linalg.norm(w)
            else:
                t = np.clip(np.dot(w, v) / v_len_sq, 0.0, 1.0)
                projection = p1.to_array() + t * v
                dist = np.linalg.norm(new_position.to_array() - projection)

            seg_radius = seg.mean_radius
            required_clearance = params.min_clearance + seg_radius

            if dist < required_clearance:
                return False

        return True

    if _children_by_node is None:
        _children_by_node = {}
        for seg in network.segments.values():
            _children_by_node.setdefault(seg.start_node_id, []).append(seg)
    if _seg_by_node is None:
        _seg_by_node = {}
        for seg in network.segments.values():
            _seg_by_node.setdefault(seg.start_node_id, []).append(seg)
            _seg_by_node.setdefault(seg.end_node_id, []).append(seg)

    ancestor_seg_ids: set = set()
    visited_ancestor: set = set()
    cur_nid = from_node_id
    while cur_nid is not None and cur_nid not in visited_ancestor:
        visited_ancestor.add(cur_nid)
        stack: list = [(cur_nid, 0)]
        while stack:
            nid, d = stack.pop()
            if d >= _excl_depth:
                continue
            for seg in _children_by_node.get(nid, []):
                ancestor_seg_ids.add(seg.id)
                stack.append((seg.end_node_id, d + 1))
        next_nid = None
        for seg in _seg_by_node.get(cur_nid, []):
            ancestor_seg_ids.add(seg.id)
            if seg.end_node_id == cur_nid and seg.start_node_id not in visited_ancestor:
                next_nid = seg.start_node_id
        cur_nid = next_nid

    search_radius = params.min_clearance * 3.0

    spatial_index = network.get_spatial_index()
    nearby_segments = spatial_index.query_nearby_segments(new_position, search_radius)
    
    for seg in nearby_segments:
        if seg.id in ancestor_seg_ids:
            continue
        
        p1 = network.nodes[seg.start_node_id].position
        p2 = network.nodes[seg.end_node_id].position
        
        v = np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])
        w = np.array([new_position.x - p1.x, new_position.y - p1.y, new_position.z - p1.z])
        
        v_len_sq = np.dot(v, v)
        if v_len_sq < 1e-10:
            dist = np.linalg.norm(w)
        else:
            t = np.clip(np.dot(w, v) / v_len_sq, 0.0, 1.0)
            projection = p1.to_array() + t * v
            dist = np.linalg.norm(new_position.to_array() - projection)
        
        seg_radius = seg.mean_radius
        required_clearance = params.min_clearance + seg_radius
        
        if dist < required_clearance:
            return False
    
    return True


@dataclass
class TipState:
    """State tracking for a tip node during space colonization."""
    node_id: int
    steps_since_split: int = 0
    total_steps: int = 0
    distance_from_root: float = 0.0
    is_root: bool = False


@dataclass
class SpaceColonizationMetrics:
    """Metrics for space colonization run."""
    root_degree: int = 0
    trunk_length: float = 0.0
    trunk_nodes: int = 0
    trunk_segments: int = 0
    split_event_count: int = 0
    bifurcation_count: int = 0
    trifurcation_count: int = 0
    degree_histogram: Dict[int, int] = field(default_factory=dict)
    branch_node_count: int = 0
    terminal_count: int = 0
    average_segment_length: float = 0.0
    
    def to_dict(self) -> Dict[str, any]:
        return {
            "root_degree": self.root_degree,
            "trunk_length": self.trunk_length,
            "trunk_nodes": self.trunk_nodes,
            "trunk_segments": self.trunk_segments,
            "split_event_count": self.split_event_count,
            "bifurcation_count": self.bifurcation_count,
            "trifurcation_count": self.trifurcation_count,
            "degree_histogram": self.degree_histogram,
            "branch_node_count": self.branch_node_count,
            "terminal_count": self.terminal_count,
            "average_segment_length": self.average_segment_length,
        }


def _greedy_angular_clustering(
    vectors: List[np.ndarray],
    angle_threshold_deg: float,
    max_clusters: int = 3,
) -> List[List[int]]:
    """
    Cluster direction vectors using greedy angular clustering.
    
    Algorithm:
    - Start first cluster with first vector
    - For each vector, assign to existing cluster if angle to cluster mean <= threshold
    - Otherwise start new cluster (up to max_clusters)
    
    Parameters
    ----------
    vectors : List[np.ndarray]
        List of unit direction vectors
    angle_threshold_deg : float
        Maximum angle (degrees) to assign to existing cluster
    max_clusters : int
        Maximum number of clusters to create
    
    Returns
    -------
    List[List[int]]
        List of clusters, each cluster is a list of vector indices
    """
    if not vectors:
        return []
    
    if len(vectors) == 1:
        return [[0]]
    
    threshold_rad = np.radians(angle_threshold_deg)
    cos_threshold = np.cos(threshold_rad)
    
    clusters: List[List[int]] = []
    cluster_means: List[np.ndarray] = []
    
    for idx, vec in enumerate(vectors):
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            continue
        unit_vec = vec / norm
        
        assigned = False
        best_cluster = -1
        best_similarity = -1.0
        
        for c_idx, c_mean in enumerate(cluster_means):
            similarity = np.dot(unit_vec, c_mean)
            if similarity >= cos_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_cluster = c_idx
                assigned = True
        
        if assigned and best_cluster >= 0:
            clusters[best_cluster].append(idx)
            cluster_vecs = [vectors[i] for i in clusters[best_cluster]]
            new_mean = np.mean(cluster_vecs, axis=0)
            new_mean_norm = np.linalg.norm(new_mean)
            if new_mean_norm > 1e-10:
                cluster_means[best_cluster] = new_mean / new_mean_norm
        elif len(clusters) < max_clusters:
            clusters.append([idx])
            cluster_means.append(unit_vec.copy())
        else:
            best_cluster = 0
            best_similarity = -1.0
            for c_idx, c_mean in enumerate(cluster_means):
                similarity = np.dot(unit_vec, c_mean)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = c_idx
            clusters[best_cluster].append(idx)
            cluster_vecs = [vectors[i] for i in clusters[best_cluster]]
            new_mean = np.mean(cluster_vecs, axis=0)
            new_mean_norm = np.linalg.norm(new_mean)
            if new_mean_norm > 1e-10:
                cluster_means[best_cluster] = new_mean / new_mean_norm
    
    return clusters


def _compute_cluster_support(
    cluster_indices: List[int],
    total_attractors: int,
) -> float:
    """Compute support (fraction of attractors) for a cluster."""
    if total_attractors == 0:
        return 0.0
    return len(cluster_indices) / total_attractors


def _merge_weakest_cluster(
    clusters: List[List[int]],
    vectors: List[np.ndarray],
) -> List[List[int]]:
    """
    Merge the weakest (smallest) cluster into the nearest cluster.
    
    Returns clusters with one fewer cluster.
    """
    if len(clusters) <= 2:
        return clusters
    
    min_size = float('inf')
    weakest_idx = 0
    for i, c in enumerate(clusters):
        if len(c) < min_size:
            min_size = len(c)
            weakest_idx = i
    
    weakest_mean = np.mean([vectors[i] for i in clusters[weakest_idx]], axis=0)
    weakest_mean_norm = np.linalg.norm(weakest_mean)
    if weakest_mean_norm > 1e-10:
        weakest_mean = weakest_mean / weakest_mean_norm
    
    best_target = -1
    best_similarity = -2.0
    for i, c in enumerate(clusters):
        if i == weakest_idx:
            continue
        c_mean = np.mean([vectors[j] for j in c], axis=0)
        c_mean_norm = np.linalg.norm(c_mean)
        if c_mean_norm > 1e-10:
            c_mean = c_mean / c_mean_norm
            similarity = np.dot(weakest_mean, c_mean)
            if similarity > best_similarity:
                best_similarity = similarity
                best_target = i
    
    if best_target >= 0:
        clusters[best_target].extend(clusters[weakest_idx])
        del clusters[weakest_idx]
    
    return clusters


def _apply_noise_to_direction(
    direction: np.ndarray,
    noise_angle_deg: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply small random noise to a direction vector.
    
    Parameters
    ----------
    direction : np.ndarray
        Unit direction vector
    noise_angle_deg : float
        Maximum noise angle in degrees
    rng : np.random.Generator
        Random number generator
    
    Returns
    -------
    np.ndarray
        Noisy unit direction vector
    """
    if noise_angle_deg <= 0:
        return direction
    
    noise_rad = np.radians(noise_angle_deg)
    theta = rng.uniform(0, 2 * np.pi)
    phi = rng.uniform(0, noise_rad)
    
    perp1 = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(direction, perp1)) > 0.9:
        perp1 = np.array([0.0, 1.0, 0.0])
    perp1 = perp1 - np.dot(perp1, direction) * direction
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direction, perp1)
    
    noisy = (direction * np.cos(phi) + 
             perp1 * np.sin(phi) * np.cos(theta) + 
             perp2 * np.sin(phi) * np.sin(theta))
    
    return noisy / np.linalg.norm(noisy)


def _select_active_tips_probabilistic(
    tip_states: List[TipState],
    tip_supports: Dict[int, int],
    alpha: float,
    active_fraction: float,
    min_active: int,
    rng: np.random.Generator,
) -> List[TipState]:
    """
    Select active tips using probabilistic sampling based on support.
    
    Probability proportional to (support^alpha + eps).
    """
    if not tip_states:
        return []
    
    eps = 1e-6
    weights = []
    for ts in tip_states:
        support = tip_supports.get(ts.node_id, 0)
        weight = (support ** alpha) + eps
        weights.append(weight)
    
    total_weight = sum(weights)
    probs = [w / total_weight for w in weights]
    
    target_count = max(min_active, int(np.ceil(active_fraction * len(tip_states))))
    target_count = min(target_count, len(tip_states))
    
    selected_indices = set()
    while len(selected_indices) < target_count:
        idx = rng.choice(len(tip_states), p=probs)
        selected_indices.add(idx)
    
    return [tip_states[i] for i in selected_indices]


def _select_active_tips_topk(
    tip_states: List[TipState],
    tip_supports: Dict[int, int],
    active_fraction: float,
    min_active: int,
) -> List[TipState]:
    """
    Select active tips using top-k selection based on support.
    """
    if not tip_states:
        return []
    
    sorted_tips = sorted(
        tip_states,
        key=lambda ts: tip_supports.get(ts.node_id, 0),
        reverse=True,
    )
    
    target_count = max(min_active, int(np.ceil(active_fraction * len(tip_states))))
    target_count = min(target_count, len(tip_states))
    
    return sorted_tips[:target_count]


def _compute_network_metrics(
    network: VascularNetwork,
    root_node_id: Optional[int],
) -> SpaceColonizationMetrics:
    """
    Compute structural metrics for the network.
    
    Returns metrics including root degree, degree histogram, branch count, etc.
    """
    metrics = SpaceColonizationMetrics()
    
    out_degrees: Dict[int, int] = {node_id: 0 for node_id in network.nodes}
    
    for seg in network.segments.values():
        out_degrees[seg.start_node_id] = out_degrees.get(seg.start_node_id, 0) + 1
    
    if root_node_id is not None and root_node_id in out_degrees:
        metrics.root_degree = out_degrees[root_node_id]
    
    for node_id, degree in out_degrees.items():
        metrics.degree_histogram[degree] = metrics.degree_histogram.get(degree, 0) + 1
        if degree >= 2:
            metrics.branch_node_count += 1
    
    metrics.terminal_count = sum(
        1 for n in network.nodes.values() if n.node_type == "terminal"
    )
    
    if network.segments:
        total_length = 0.0
        for seg in network.segments.values():
            p1 = network.nodes[seg.start_node_id].position
            p2 = network.nodes[seg.end_node_id].position
            length = np.sqrt(
                (p2.x - p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2
            )
            total_length += length
        metrics.average_segment_length = total_length / len(network.segments)
    
    return metrics


def space_colonization_step_v2(
    network: VascularNetwork,
    tissue_points: np.ndarray,
    params: Optional[SpaceColonizationParams] = None,
    constraints: Optional[BranchingConstraints] = None,
    seed: Optional[int] = None,
    seed_nodes: Optional[List[str]] = None,
    sc_policy: Optional["SpaceColonizationPolicy"] = None,
    disable_progress: bool = False,
) -> OperationResult:
    """
    Policy-driven space colonization with trunk-first growth, apical dominance,
    and angular-clustering-based splitting.
    
    This version implements:
    A) Trunk-first + root suppression: Prevents "inlet starburst"
    B) Apical dominance: Reduces parallel linear growth
    C) Angular clustering: Enables proper branching when attractor field supports it
    
    Parameters
    ----------
    network : VascularNetwork
        Network to grow
    tissue_points : np.ndarray
        Array of tissue points (N, 3) that need perfusion
    params : SpaceColonizationParams, optional
        Algorithm parameters (legacy, used for compatibility)
    constraints : BranchingConstraints, optional
        Branching constraints
    seed : int, optional
        Random seed
    seed_nodes : List[str], optional
        List of node IDs to use as seed nodes for growth
    sc_policy : SpaceColonizationPolicy, optional
        Policy controlling all behavior. If None, uses defaults.
    
    Returns
    -------
    result : OperationResult
        Result with metadata about growth progress including tree-shape metrics
    """
    from aog_policies.space_colonization import SpaceColonizationPolicy
    
    if params is None:
        params = SpaceColonizationParams()
    
    if sc_policy is None:
        sc_policy = SpaceColonizationPolicy()
    
    if constraints is None:
        constraints = BranchingConstraints(
            min_segment_length=max(params.step_size, sc_policy.min_branch_segment_length),
            min_radius=params.min_radius,
            collision_min_clearance=params.min_clearance if params.min_clearance is not None else 0.001,
        )
    
    rng = np.random.default_rng(seed) if seed is not None else network.id_gen.rng
    
    root_node_id: Optional[int] = None
    inlet_direction: Optional[np.ndarray] = None
    
    if seed_nodes is not None:
        initial_nodes = [
            network.nodes[node_id] for node_id in seed_nodes
            if node_id in network.nodes and network.nodes[node_id].vessel_type == params.vessel_type
        ]
    else:
        initial_nodes = [
            node for node in network.nodes.values()
            if node.node_type in ("inlet", "outlet") and
            node.vessel_type == params.vessel_type
        ]
    
    if not initial_nodes:
        return OperationResult.failure(
            message=f"No inlet/outlet nodes of type {params.vessel_type} found",
            errors=["No seed nodes"],
        )
    
    root_node = initial_nodes[0]
    root_node_id = root_node.id
    
    if "direction" in root_node.attributes:
        dir_data = root_node.attributes["direction"]
        if isinstance(dir_data, dict):
            inlet_direction = np.array([dir_data.get("dx", 0), dir_data.get("dy", 0), dir_data.get("dz", 1)])
        elif isinstance(dir_data, (list, tuple)):
            inlet_direction = np.array(dir_data)
        else:
            inlet_direction = np.array([0, 0, 1])
    else:
        inlet_direction = np.array([0, 0, 1])
    
    inlet_direction = inlet_direction / np.linalg.norm(inlet_direction)
    
    tip_states: Dict[int, TipState] = {}
    for node in initial_nodes:
        tip_states[node.id] = TipState(
            node_id=node.id,
            steps_since_split=sc_policy.split_cooldown_steps,
            total_steps=0,
            distance_from_root=0.0,
            is_root=(node.id == root_node_id),
        )
    
    tissue_points_list = [
        p if isinstance(p, Point3D) else Point3D.from_array(p)
        for p in tissue_points
    ]
    active_tissue_points = set(range(len(tissue_points_list)))
    initial_count = len(tissue_points_list)
    
    new_node_ids = []
    new_segment_ids = []
    warnings = []
    steps_taken = 0
    
    metrics = SpaceColonizationMetrics()
    trunk_phase_complete = False
    trunk_tip_id = root_node_id
    
    pbar = tqdm(total=params.max_steps, desc="Space colonization v2", unit="step", disable=disable_progress)
    
    for step in range(params.max_steps):
        if not active_tissue_points:
            pbar.close()
            break
        
        current_tips = [
            network.nodes[ts.node_id] for ts in tip_states.values()
            if ts.node_id in network.nodes and
            network.nodes[ts.node_id].node_type in ("terminal", "inlet", "outlet")
        ]
        
        if not current_tips:
            pbar.close()
            break
        
        tip_positions = np.array([
            [node.position.x, node.position.y, node.position.z]
            for node in current_tips
        ])
        tip_kdtree = cKDTree(tip_positions)
        tip_id_list = [node.id for node in current_tips]
        
        attractions: Dict[int, List[int]] = {node.id: [] for node in current_tips}
        tip_supports: Dict[int, int] = {}
        
        active_tp_indices = list(active_tissue_points)
        if active_tp_indices:
            active_tp_positions = np.array([
                [tissue_points_list[idx].x, tissue_points_list[idx].y, tissue_points_list[idx].z]
                for idx in active_tp_indices
            ])
            
            distances, nearest_indices = tip_kdtree.query(active_tp_positions, k=1)
            
            for i, tp_idx in enumerate(active_tp_indices):
                if distances[i] < params.influence_radius:
                    nearest_tip_id = tip_id_list[nearest_indices[i]]
                    attractions[nearest_tip_id].append(tp_idx)
        
        for tip_id in tip_id_list:
            tip_supports[tip_id] = len(attractions[tip_id])
        
        in_trunk_phase = (
            step < sc_policy.trunk_steps or
            (trunk_tip_id is not None and 
             tip_states.get(trunk_tip_id, TipState(0)).distance_from_root < sc_policy.branch_enable_after_distance)
        )
        
        if in_trunk_phase and not trunk_phase_complete:
            active_tip_states = [ts for ts in tip_states.values() if ts.node_id == trunk_tip_id]
            if not active_tip_states and tip_states:
                active_tip_states = [list(tip_states.values())[0]]
        else:
            if not trunk_phase_complete:
                trunk_phase_complete = True
                metrics.trunk_nodes = len(new_node_ids)
                metrics.trunk_segments = len(new_segment_ids)
                if trunk_tip_id in tip_states:
                    metrics.trunk_length = tip_states[trunk_tip_id].distance_from_root
            
            all_tip_states = list(tip_states.values())
            
            if sc_policy.dominance_mode == "probabilistic":
                active_tip_states = _select_active_tips_probabilistic(
                    all_tip_states,
                    tip_supports,
                    sc_policy.apical_dominance_alpha,
                    sc_policy.active_tip_fraction,
                    sc_policy.min_active_tips,
                    rng,
                )
            else:
                active_tip_states = _select_active_tips_topk(
                    all_tip_states,
                    tip_supports,
                    sc_policy.active_tip_fraction,
                    sc_policy.min_active_tips,
                )
        
        grown_any = False
        
        for tip_state in active_tip_states:
            node_id = tip_state.node_id
            if node_id not in network.nodes:
                continue
            
            node = network.nodes[node_id]
            attracted_indices = attractions.get(node_id, [])
            
            if not attracted_indices:
                continue
            
            attracted_points = [tissue_points_list[idx] for idx in attracted_indices]
            num_attractions = len(attracted_points)
            
            attraction_vectors = []
            for tp in attracted_points:
                direction = np.array([
                    tp.x - node.position.x,
                    tp.y - node.position.y,
                    tp.z - node.position.z,
                ])
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 1e-10:
                    attraction_vectors.append(direction / direction_norm)
            
            if not attraction_vectors:
                continue
            
            is_root_node = tip_state.is_root
            can_split = (
                sc_policy.enable_cluster_splitting and
                not in_trunk_phase and
                tip_state.steps_since_split >= sc_policy.split_cooldown_steps and
                num_attractions >= sc_policy.min_attractors_to_split
            )
            
            if is_root_node:
                existing_children = sum(
                    1 for seg in network.segments.values()
                    if seg.start_node_id == node_id
                )
                if existing_children >= sc_policy.max_root_children:
                    continue
                can_split = False
            
            existing_children_count = sum(
                1 for seg in network.segments.values()
                if seg.start_node_id == node_id
            )
            if existing_children_count >= sc_policy.max_children_per_node_total:
                continue
            
            remaining_slots = sc_policy.max_children_per_node_total - existing_children_count
            
            if can_split and len(attraction_vectors) >= 2:
                clusters = _greedy_angular_clustering(
                    attraction_vectors,
                    sc_policy.cluster_angle_threshold_deg,
                    max_clusters=min(sc_policy.max_children_per_split, remaining_slots),
                )
                
                if len(clusters) >= 2:
                    if len(clusters) == 3:
                        if rng.random() >= sc_policy.allow_trifurcation_prob:
                            clusters = _merge_weakest_cluster(clusters, attraction_vectors)
                    
                    if len(clusters) > 3:
                        cluster_supports = [
                            (i, len(c)) for i, c in enumerate(clusters)
                        ]
                        cluster_supports.sort(key=lambda x: x[1], reverse=True)
                        top_3_indices = [cs[0] for cs in cluster_supports[:3]]
                        clusters = [clusters[i] for i in sorted(top_3_indices)]
                    
                    parent_radius = node.attributes.get("radius", params.min_radius * 2)
                    n_children = len(clusters)
                    
                    if sc_policy.split_strength_mode == "proportional_to_cluster_support":
                        total_support = sum(len(c) for c in clusters)
                        child_radii = []
                        for c in clusters:
                            fraction = len(c) / total_support if total_support > 0 else 1.0 / n_children
                            child_radius = parent_radius * (fraction ** (1.0/3.0)) * params.taper_factor
                            child_radii.append(max(child_radius, params.min_radius))
                    else:
                        child_radius = parent_radius * (1.0 / n_children) ** (1.0/3.0) * params.taper_factor
                        child_radii = [max(child_radius, params.min_radius)] * n_children
                    
                    
                    children_created = 0
                    for cluster_idx, cluster in enumerate(clusters):
                        if children_created >= remaining_slots:
                            break
                        
                        cluster_vecs = [attraction_vectors[i] for i in cluster]
                        cluster_direction = np.mean(cluster_vecs, axis=0)
                        cluster_direction = cluster_direction / np.linalg.norm(cluster_direction)
                        
                        cluster_direction = _apply_noise_to_direction(
                            cluster_direction,
                            sc_policy.noise_angle_deg,
                            rng,
                        )
                        
                        cluster_direction = _apply_directional_blending(cluster_direction, node, params)
                        cluster_direction = _apply_curvature_constraint(cluster_direction, node, params)
                        
                        growth_direction = Direction3D.from_array(cluster_direction)
                        
                        new_pos = Point3D(
                            node.position.x + growth_direction.dx * params.step_size,
                            node.position.y + growth_direction.dy * params.step_size,
                            node.position.z + growth_direction.dz * params.step_size,
                        )
                        
                        if not _check_clearance(new_pos, network, node_id, params):
                            continue
                        
                        new_radius = child_radii[cluster_idx]
                        
                        result = grow_branch(
                            network,
                            from_node_id=node_id,
                            length=params.step_size,
                            direction=growth_direction,
                            target_radius=new_radius,
                            constraints=constraints,
                            check_collisions=True,
                            seed=seed,
                            collision_mode=params.collision_mode,
                        )
                        
                        if result.is_success():
                            new_segment_ids.append(result.new_ids["segment"])
                            grown_any = True
                            children_created += 1
                            if not result.new_ids.get("merged"):
                                new_node_id = result.new_ids["node"]
                                new_node_ids.append(new_node_id)
                                tip_states[new_node_id] = TipState(
                                    node_id=new_node_id,
                                    steps_since_split=0,
                                    total_steps=tip_state.total_steps + 1,
                                    distance_from_root=tip_state.distance_from_root + params.step_size,
                                    is_root=False,
                                )
                        else:
                            warnings.extend(result.errors)
                    
                    if children_created > 0:
                        metrics.split_event_count += 1
                        if children_created == 2:
                            metrics.bifurcation_count += 1
                        elif children_created == 3:
                            metrics.trifurcation_count += 1
                        
                        if node_id in tip_states:
                            del tip_states[node_id]
                    
                    continue
            
            if in_trunk_phase and sc_policy.trunk_direction_mode == "inlet_direction":
                avg_direction = inlet_direction.copy()
            else:
                avg_direction = np.mean(attraction_vectors, axis=0)
                avg_direction = avg_direction / np.linalg.norm(avg_direction)
            
            avg_direction = _apply_noise_to_direction(
                avg_direction,
                sc_policy.noise_angle_deg,
                rng,
            )
            
            avg_direction = _apply_directional_blending(avg_direction, node, params)
            avg_direction = _apply_curvature_constraint(avg_direction, node, params)
            
            growth_direction = Direction3D.from_array(avg_direction)
            
            new_pos = Point3D(
                node.position.x + growth_direction.dx * params.step_size,
                node.position.y + growth_direction.dy * params.step_size,
                node.position.z + growth_direction.dz * params.step_size,
            )
            
            if not _check_clearance(new_pos, network, node_id, params):
                continue
            
            parent_radius = node.attributes.get("radius", params.min_radius * 2)
            new_radius = parent_radius * params.taper_factor
            new_radius = max(new_radius, params.min_radius)
            
            result = grow_branch(
                network,
                from_node_id=node_id,
                length=params.step_size,
                direction=growth_direction,
                target_radius=new_radius,
                constraints=constraints,
                check_collisions=True,
                seed=seed,
                collision_mode=params.collision_mode,
            )
            
            if result.is_success():
                new_segment_ids.append(result.new_ids["segment"])
                grown_any = True
                if not result.new_ids.get("merged"):
                    new_node_id = result.new_ids["node"]
                    new_node_ids.append(new_node_id)
                    new_tip_state = TipState(
                        node_id=new_node_id,
                        steps_since_split=tip_state.steps_since_split + 1,
                        total_steps=tip_state.total_steps + 1,
                        distance_from_root=tip_state.distance_from_root + params.step_size,
                        is_root=False,
                    )
                    tip_states[new_node_id] = new_tip_state
                    if in_trunk_phase and node_id == trunk_tip_id:
                        trunk_tip_id = new_node_id
                if node_id in tip_states:
                    del tip_states[node_id]
            else:
                if result.errors:
                    _logger.warning(
                        "SC v2 step %d: grow_branch failed (tip=%s): %s",
                        step, node_id, "; ".join(result.errors[:3]),
                    )
                warnings.extend(result.errors)
        
        if not grown_any:
            _logger.warning(
                "SC v2 step %d: no growth occurred (tips=%d, active_tissue=%d). Stopping.",
                step, len(tip_states), len(active_tissue_points),
            )
            pbar.close()
            break
        
        steps_taken += 1
        pbar.update(1)
        pbar.set_postfix({
            'nodes': len(new_node_ids),
            'tips': len(tip_states),
            'coverage': f'{(initial_count - len(active_tissue_points)) / initial_count:.1%}' if initial_count > 0 else '0%'
        })
        
        if network.nodes and active_tissue_points:
            all_node_positions = np.array([
                [node.position.x, node.position.y, node.position.z]
                for node in network.nodes.values()
            ])
            node_kdtree = cKDTree(all_node_positions)
            
            active_tp_indices = list(active_tissue_points)
            active_tp_positions = np.array([
                [tissue_points_list[idx].x, tissue_points_list[idx].y, tissue_points_list[idx].z]
                for idx in active_tp_indices
            ])
            
            nearby_results = node_kdtree.query_ball_point(active_tp_positions, params.kill_radius)
            
            for i, tp_idx in enumerate(active_tp_indices):
                if nearby_results[i]:
                    active_tissue_points.discard(tp_idx)
    
    pbar.close()
    
    final_metrics = _compute_network_metrics(network, root_node_id)
    metrics.root_degree = final_metrics.root_degree
    metrics.degree_histogram = final_metrics.degree_histogram
    metrics.branch_node_count = final_metrics.branch_node_count
    metrics.terminal_count = final_metrics.terminal_count
    metrics.average_segment_length = final_metrics.average_segment_length
    
    perfused_count = initial_count - len(active_tissue_points)
    coverage_fraction = perfused_count / initial_count if initial_count > 0 else 0.0
    
    delta = Delta(
        created_node_ids=new_node_ids,
        created_segment_ids=new_segment_ids,
    )
    
    if new_node_ids:
        status = OperationStatus.SUCCESS if not warnings else OperationStatus.PARTIAL_SUCCESS
        message = f"Grew {len(new_node_ids)} nodes in {steps_taken} steps, {coverage_fraction:.1%} coverage"
    else:
        status = OperationStatus.WARNING
        message = "No growth occurred"
    
    return OperationResult(
        status=status,
        message=message,
        new_ids={
            "nodes": new_node_ids,
            "segments": new_segment_ids,
        },
        warnings=warnings,
        delta=delta,
        rng_state=network.id_gen.get_state(),
        metadata={
            "steps_taken": steps_taken,
            "nodes_grown": len(new_node_ids),
            "initial_tissue_points": initial_count,
            "perfused_tissue_points": perfused_count,
            "coverage_fraction": coverage_fraction,
            "tree_metrics": metrics.to_dict(),
        },
    )


@dataclass
class SpaceColonizationState:
    """
    Persistent state for space colonization across multiple single-step calls.
    
    This enables efficient multi-inlet interleaving without nested loops or
    repeated progress bar creation.
    """
    network: VascularNetwork
    params: SpaceColonizationParams
    constraints: BranchingConstraints
    
    tissue_points: np.ndarray
    active_tissue_indices: Set[int] = field(default_factory=set)
    initial_tissue_count: int = 0
    
    active_tip_ids: Set[int] = field(default_factory=set)
    tip_states: Dict[int, TipState] = field(default_factory=dict)
    
    tip_kdtree: Optional[cKDTree] = None
    all_nodes_kdtree: Optional[cKDTree] = None
    tip_kdtree_node_ids: List[int] = field(default_factory=list)
    all_nodes_kdtree_node_ids: List[int] = field(default_factory=list)
    
    node_positions_array: Optional[np.ndarray] = None
    node_ids_array: Optional[np.ndarray] = None
    n_tracked_nodes: int = 0
    _node_pos_capacity: int = 0
    
    _kill_spatial_hash: Optional[SpatialHash] = None
    _collision_spatial_index: Optional[DynamicSpatialIndex] = None
    _persistent_gpu_index: Optional[PersistentGPUIndex] = None
    
    global_step: int = 0
    steps_since_tip_kdtree_rebuild: int = 0
    steps_since_all_nodes_kdtree_rebuild: int = 0
    nodes_added_since_tip_kdtree_rebuild: int = 0
    nodes_added_since_all_nodes_kdtree_rebuild: int = 0
    
    consecutive_stall_steps: int = 0
    total_nodes_added: int = 0
    total_segments_added: int = 0
    
    rng: Optional[np.random.Generator] = None
    inlet_id: Optional[int] = None
    vessel_type: str = "arterial"
    
    kdtree_rebuild_tip_every: int = 5
    kdtree_rebuild_all_nodes_every: int = 15
    kdtree_rebuild_all_nodes_min_new_nodes: int = 5
    stall_steps_threshold: int = 10
    
    def needs_tip_kdtree_rebuild(self) -> bool:
        """Check if tip KD-tree needs rebuilding."""
        if self.tip_kdtree is None:
            return True
        if self.steps_since_tip_kdtree_rebuild >= self.kdtree_rebuild_tip_every:
            return True
        return False
    
    def needs_all_nodes_kdtree_rebuild(self) -> bool:
        """Check if all-nodes KD-tree needs rebuilding."""
        if self.all_nodes_kdtree is None:
            return True
        if self.steps_since_all_nodes_kdtree_rebuild >= self.kdtree_rebuild_all_nodes_every:
            return True
        if self.nodes_added_since_all_nodes_kdtree_rebuild >= self.kdtree_rebuild_all_nodes_min_new_nodes:
            return True
        return False
    
    def _ensure_position_capacity(self, needed: int) -> None:
        """Grow the pre-allocated position/id arrays if necessary."""
        if self.node_positions_array is None:
            cap = max(needed, 4096)
            self.node_positions_array = np.empty((cap, 3), dtype=np.float64)
            self.node_ids_array = np.empty(cap, dtype=np.intp)
            self._node_pos_capacity = cap
            return
        if needed > self._node_pos_capacity:
            new_cap = max(needed, self._node_pos_capacity * 2)
            new_pos = np.empty((new_cap, 3), dtype=np.float64)
            new_ids = np.empty(new_cap, dtype=np.intp)
            new_pos[:self.n_tracked_nodes] = self.node_positions_array[:self.n_tracked_nodes]
            new_ids[:self.n_tracked_nodes] = self.node_ids_array[:self.n_tracked_nodes]
            self.node_positions_array = new_pos
            self.node_ids_array = new_ids
            self._node_pos_capacity = new_cap

    def sync_node_positions(self) -> None:
        """Rebuild the contiguous position array from the network."""
        nodes = list(self.network.nodes.values())
        n = len(nodes)
        self._ensure_position_capacity(n)
        for i, node in enumerate(nodes):
            self.node_positions_array[i, 0] = node.position.x
            self.node_positions_array[i, 1] = node.position.y
            self.node_positions_array[i, 2] = node.position.z
            self.node_ids_array[i] = node.id
        self.n_tracked_nodes = n

    def append_node_position(self, node_id: int, position: Point3D) -> None:
        """Append a single new node to the contiguous array (avoids full resync)."""
        self._ensure_position_capacity(self.n_tracked_nodes + 1)
        idx = self.n_tracked_nodes
        self.node_positions_array[idx, 0] = position.x
        self.node_positions_array[idx, 1] = position.y
        self.node_positions_array[idx, 2] = position.z
        self.node_ids_array[idx] = node_id
        self.n_tracked_nodes += 1

    def get_all_node_positions(self) -> np.ndarray:
        """Return (N, 3) view of all tracked node positions."""
        return self.node_positions_array[:self.n_tracked_nodes]

    def rebuild_tip_kdtree(self) -> None:
        """Rebuild the KD-tree for active tip nodes."""
        if self.n_tracked_nodes > 0 and self.active_tip_ids:
            tip_id_set = self.active_tip_ids
            ids_arr = self.node_ids_array[:self.n_tracked_nodes]
            mask = np.array([int(nid) in tip_id_set for nid in ids_arr], dtype=bool)
            if np.any(mask):
                positions = self.node_positions_array[:self.n_tracked_nodes][mask]
                self.tip_kdtree = cKDTree(positions)
                self.tip_kdtree_node_ids = ids_arr[mask].tolist()
            else:
                self.tip_kdtree = None
                self.tip_kdtree_node_ids = []
        else:
            tip_nodes = [
                self.network.nodes[tid] for tid in self.active_tip_ids
                if tid in self.network.nodes
            ]
            if tip_nodes:
                positions = np.array([
                    [n.position.x, n.position.y, n.position.z] for n in tip_nodes
                ])
                self.tip_kdtree = cKDTree(positions)
                self.tip_kdtree_node_ids = [n.id for n in tip_nodes]
            else:
                self.tip_kdtree = None
                self.tip_kdtree_node_ids = []
        self.steps_since_tip_kdtree_rebuild = 0

    def rebuild_all_nodes_kdtree(self) -> None:
        """Rebuild the KD-tree for all nodes (used for kill radius)."""
        self.sync_node_positions()
        if self.n_tracked_nodes > 0:
            positions = self.get_all_node_positions()
            self.all_nodes_kdtree = cKDTree(positions)
            self.all_nodes_kdtree_node_ids = self.node_ids_array[:self.n_tracked_nodes].tolist()
            if self.n_tracked_nodes > 5000:
                self._kill_spatial_hash = SpatialHash(self.params.kill_radius)
                self._kill_spatial_hash.build(positions)
            else:
                self._kill_spatial_hash = None
        else:
            self.all_nodes_kdtree = None
            self.all_nodes_kdtree_node_ids = []
            self._kill_spatial_hash = None
        self.steps_since_all_nodes_kdtree_rebuild = 0
        self.nodes_added_since_all_nodes_kdtree_rebuild = 0
    
    def build_collision_spatial_index(self) -> None:
        """Build DynamicSpatialIndex from all existing network segments."""
        cell_size = max(self.params.step_size * 3, 0.001)
        self._collision_spatial_index = DynamicSpatialIndex(cell_size=cell_size)
        for seg_id, seg in self.network.segments.items():
            start = np.array([seg.geometry.start.x, seg.geometry.start.y, seg.geometry.start.z])
            end = np.array([seg.geometry.end.x, seg.geometry.end.y, seg.geometry.end.z])
            radius = seg.geometry.mean_radius()
            self._collision_spatial_index.insert_segment(seg_id, start, end, radius)
    
    def insert_segment_into_spatial_index(self, seg_id: int) -> None:
        """Insert a newly created segment into the collision spatial index."""
        if self._collision_spatial_index is None:
            return
        seg = self.network.segments.get(seg_id)
        if seg is None:
            return
        start = np.array([seg.geometry.start.x, seg.geometry.start.y, seg.geometry.start.z])
        end = np.array([seg.geometry.end.x, seg.geometry.end.y, seg.geometry.end.z])
        radius = seg.geometry.mean_radius()
        self._collision_spatial_index.insert_segment(seg_id, start, end, radius)
    
    def is_stalled(self) -> bool:
        """Check if growth has stalled."""
        return self.consecutive_stall_steps >= self.stall_steps_threshold
    
    def is_exhausted(self) -> bool:
        """Check if no more growth is possible."""
        return len(self.active_tissue_indices) == 0 or len(self.active_tip_ids) == 0


@dataclass
class SingleStepResult:
    """Result of a single space colonization step."""
    nodes_added: int = 0
    segments_added: int = 0
    attractors_killed: int = 0
    active_attractors: int = 0
    active_tips: int = 0
    coverage_fraction: float = 0.0
    stalled: bool = False
    exhausted: bool = False
    new_node_ids: List[int] = field(default_factory=list)
    new_segment_ids: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, any]:
        return {
            "nodes_added": self.nodes_added,
            "segments_added": self.segments_added,
            "attractors_killed": self.attractors_killed,
            "active_attractors": self.active_attractors,
            "active_tips": self.active_tips,
            "coverage_fraction": self.coverage_fraction,
            "stalled": self.stalled,
            "exhausted": self.exhausted,
        }


def create_space_colonization_state(
    network: VascularNetwork,
    tissue_points: np.ndarray,
    params: Optional[SpaceColonizationParams] = None,
    constraints: Optional[BranchingConstraints] = None,
    seed: Optional[int] = None,
    seed_node_ids: Optional[List[int]] = None,
    inlet_id: Optional[int] = None,
    vessel_type: str = "arterial",
    kdtree_rebuild_tip_every: int = 1,
    kdtree_rebuild_all_nodes_every: int = 10,
    kdtree_rebuild_all_nodes_min_new_nodes: int = 5,
    stall_steps_threshold: int = 10,
) -> SpaceColonizationState:
    """
    Create initial state for space colonization.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to grow
    tissue_points : np.ndarray
        Array of tissue points (N, 3) that need perfusion
    params : SpaceColonizationParams, optional
        Algorithm parameters
    constraints : BranchingConstraints, optional
        Branching constraints
    seed : int, optional
        Random seed
    seed_node_ids : List[int], optional
        List of node IDs to use as seed nodes for growth
    inlet_id : int, optional
        Identifier for this inlet (for multi-inlet tracking)
    vessel_type : str
        Type of vessels ("arterial" or "venous")
    kdtree_rebuild_tip_every : int
        Rebuild tip KD-tree every N steps
    kdtree_rebuild_all_nodes_every : int
        Rebuild all-nodes KD-tree every N steps
    kdtree_rebuild_all_nodes_min_new_nodes : int
        Rebuild all-nodes KD-tree if this many nodes added
    stall_steps_threshold : int
        Mark as stalled after this many steps with no growth
        
    Returns
    -------
    SpaceColonizationState
        Initial state for space colonization
    """
    if params is None:
        params = SpaceColonizationParams()
    
    if constraints is None:
        constraints = BranchingConstraints(
            min_segment_length=params.step_size,
            min_radius=params.min_radius,
            collision_min_clearance=params.min_clearance if params.min_clearance is not None else 0.001,
        )
    
    rng = np.random.default_rng(seed) if seed is not None else network.id_gen.rng
    
    if isinstance(tissue_points, list):
        tissue_points_arr = np.array([
            [p.x, p.y, p.z] if isinstance(p, Point3D) else p
            for p in tissue_points
        ])
    else:
        tissue_points_arr = tissue_points
    
    active_tissue_indices = set(range(len(tissue_points_arr)))
    
    if seed_node_ids is not None:
        active_tip_ids = set(
            nid for nid in seed_node_ids
            if nid in network.nodes and network.nodes[nid].vessel_type == vessel_type
        )
    else:
        active_tip_ids = set(
            node.id for node in network.nodes.values()
            if node.node_type in ("terminal", "inlet", "outlet") and
            node.vessel_type == vessel_type
        )
    
    tip_states = {}
    for tip_id in active_tip_ids:
        tip_states[tip_id] = TipState(
            node_id=tip_id,
            steps_since_split=0,
            total_steps=0,
            distance_from_root=0.0,
            is_root=(tip_id in seed_node_ids) if seed_node_ids else True,
        )
    
    state = SpaceColonizationState(
        network=network,
        params=params,
        constraints=constraints,
        tissue_points=tissue_points_arr,
        active_tissue_indices=active_tissue_indices,
        initial_tissue_count=len(tissue_points_arr),
        active_tip_ids=active_tip_ids,
        tip_states=tip_states,
        rng=rng,
        inlet_id=inlet_id,
        vessel_type=vessel_type,
        kdtree_rebuild_tip_every=kdtree_rebuild_tip_every,
        kdtree_rebuild_all_nodes_every=kdtree_rebuild_all_nodes_every,
        kdtree_rebuild_all_nodes_min_new_nodes=kdtree_rebuild_all_nodes_min_new_nodes,
        stall_steps_threshold=stall_steps_threshold,
    )
    
    state.rebuild_tip_kdtree()
    state.rebuild_all_nodes_kdtree()
    state.build_collision_spatial_index()
    state._persistent_gpu_index = PersistentGPUIndex(tissue_points_arr)
    
    return state


def space_colonization_one_step(
    state: SpaceColonizationState,
) -> SingleStepResult:
    """
    Perform exactly ONE iteration of space colonization growth.
    
    This function is designed for efficient multi-inlet interleaving.
    It does NOT create progress bars or print output.
    
    Parameters
    ----------
    state : SpaceColonizationState
        Persistent state object (modified in-place)
        
    Returns
    -------
    SingleStepResult
        Result of this single step
    """
    result = SingleStepResult()
    params = state.params
    network = state.network
    constraints = state.constraints
    rng = state.rng
    
    if state.is_exhausted():
        result.exhausted = True
        result.active_attractors = len(state.active_tissue_indices)
        result.active_tips = len(state.active_tip_ids)
        _logger.info(
            "SC one_step (inlet=%s) step %d: exhausted (tips=%d, attractors=%d)",
            state.inlet_id, state.global_step,
            len(state.active_tip_ids), len(state.active_tissue_indices),
        )
        return result
    
    if state.needs_tip_kdtree_rebuild():
        state.rebuild_tip_kdtree()
    
    if state.tip_kdtree is None or len(state.tip_kdtree_node_ids) == 0:
        result.exhausted = True
        result.active_attractors = len(state.active_tissue_indices)
        result.active_tips = 0
        _logger.warning(
            "SC one_step (inlet=%s) step %d: no tips in KDTree",
            state.inlet_id, state.global_step,
        )
        return result
    
    active_tp_indices = np.array(list(state.active_tissue_indices), dtype=np.intp)
    if len(active_tp_indices) == 0:
        result.exhausted = True
        result.active_tips = len(state.active_tip_ids)
        _logger.info(
            "SC one_step (inlet=%s) step %d: no active attractors remain",
            state.inlet_id, state.global_step,
        )
        return result
    
    active_tp_positions = state.tissue_points[active_tp_indices]
    
    tip_positions = np.array([
        [state.network.nodes[tid].position.x,
         state.network.nodes[tid].position.y,
         state.network.nodes[tid].position.z]
        for tid in state.tip_kdtree_node_ids
        if tid in state.network.nodes
    ])
    tip_id_arr = np.array(state.tip_kdtree_node_ids, dtype=np.intp)
    
    gpu_idx = state._persistent_gpu_index
    if gpu_idx is not None and gpu_idx.on_gpu:
        distances, nearest_indices = gpu_idx.nn_query(active_tp_indices, tip_positions)
    else:
        distances, nearest_indices = _nn_query(active_tp_positions, tip_positions, k=1)
    
    attractions: Dict[int, List[int]] = {tid: [] for tid in state.active_tip_ids}
    within_range = distances < params.influence_radius
    valid_tp = active_tp_indices[within_range]
    valid_nearest = tip_id_arr[nearest_indices[within_range]]
    for tp_idx, tid in zip(valid_tp, valid_nearest):
        tid_int = int(tid)
        if tid_int in attractions:
            attractions[tid_int].append(int(tp_idx))
    
    if state.global_step == 0:
        n_attracted = int(within_range.sum())
        _logger.info(
            "SC one_step (inlet=%s) step 0: %d/%d attractors within influence_radius (%.4f) of %d tips",
            state.inlet_id, n_attracted, len(active_tp_indices),
            params.influence_radius, len(state.tip_kdtree_node_ids),
        )
        if n_attracted == 0:
            _logger.warning(
                "SC one_step (inlet=%s) step 0: NO attractors within influence_radius! "
                "Nearest attractor is %.6f away (influence_radius=%.4f). "
                "Consider increasing attraction_distance or num_attractors.",
                state.inlet_id,
                float(distances.min()) if len(distances) > 0 else float('inf'),
                params.influence_radius,
            )
    
    grown_any = False
    new_node_ids = []
    new_segment_ids = []
    warnings = []
    
    bifurc_tip_ids = []
    linear_tip_ids = []
    linear_tip_positions_list = []
    linear_attracted_positions_list = []
    
    for tip_id in list(state.active_tip_ids):
        if tip_id not in network.nodes:
            state.active_tip_ids.discard(tip_id)
            continue
        
        attracted_indices = attractions.get(tip_id, [])
        if not attracted_indices:
            continue
        
        num_attractions = len(attracted_indices)
        should_bifurcate = (
            params.encourage_bifurcation and
            num_attractions >= params.min_attractions_for_bifurcation
        )
        
        if should_bifurcate:
            bifurc_tip_ids.append(tip_id)
        else:
            node = network.nodes[tip_id]
            linear_tip_ids.append(tip_id)
            linear_tip_positions_list.append(
                np.array([node.position.x, node.position.y, node.position.z])
            )
            linear_attracted_positions_list.append(
                state.tissue_points[attracted_indices]
            )
    
    linear_directions = {}
    if linear_tip_ids:
        tip_pos_batch = np.array(linear_tip_positions_list)
        batch_dirs = batch_direction_average(tip_pos_batch, linear_attracted_positions_list)
        for i, tid in enumerate(linear_tip_ids):
            d = batch_dirs[i]
            if np.linalg.norm(d) < 1e-10:
                continue
            node = network.nodes[tid]
            d = _apply_directional_blending(d, node, params)
            d = _apply_curvature_constraint(d, node, params)
            if np.linalg.norm(d) > 1e-10:
                linear_directions[tid] = d
    
    cand_starts = []
    cand_ends = []
    cand_radii_list = []
    cand_tip_ids_for_prefilter = []
    
    for tid, direction in linear_directions.items():
        node = network.nodes[tid]
        start = np.array([node.position.x, node.position.y, node.position.z])
        end = start + direction * params.step_size
        parent_radius = node.attributes.get("radius", params.min_radius * 2)
        radius = max(parent_radius * params.taper_factor, params.min_radius)
        cand_starts.append(start)
        cand_ends.append(end)
        cand_radii_list.append(radius)
        cand_tip_ids_for_prefilter.append(tid)
    
    skip_collision = set()
    if cand_starts and network.segments:
        seg_starts_list = []
        seg_ends_list = []
        seg_radii_list = []
        for seg in network.segments.values():
            s_node = network.nodes[seg.start_node_id]
            e_node = network.nodes[seg.end_node_id]
            seg_starts_list.append([s_node.position.x, s_node.position.y, s_node.position.z])
            seg_ends_list.append([e_node.position.x, e_node.position.y, e_node.position.z])
            seg_radii_list.append(seg.attributes.get("radius", params.min_radius))
        
        might_collide = batch_collision_prefilter(
            np.array(cand_starts),
            np.array(cand_ends),
            np.array(cand_radii_list),
            np.array(seg_starts_list),
            np.array(seg_ends_list),
            np.array(seg_radii_list),
            buffer=params.min_clearance if params.min_clearance is not None else 0.0,
        )
        for i, tid in enumerate(cand_tip_ids_for_prefilter):
            if not might_collide[i]:
                skip_collision.add(tid)
    
    for tip_id in bifurc_tip_ids:
        node = network.nodes[tip_id]
        attracted_indices = attractions.get(tip_id, [])
        attracted_positions = state.tissue_points[attracted_indices]
        
        node_pos = np.array([node.position.x, node.position.y, node.position.z])
        raw_dirs = attracted_positions - node_pos
        dir_norms = np.linalg.norm(raw_dirs, axis=1)
        valid_mask = dir_norms > 1e-10
        attraction_vectors = list(raw_dirs[valid_mask] / dir_norms[valid_mask, np.newaxis])
        
        if len(attraction_vectors) < 2:
            continue
        
        angle_spread = _compute_angle_spread(attraction_vectors)
        if angle_spread < params.bifurcation_angle_threshold_deg:
            continue
        if rng.random() >= params.bifurcation_probability:
            continue
        
        clusters = _cluster_attractions_by_angle(
            attraction_vectors,
            max_clusters=min(params.max_children_per_node, len(attraction_vectors))
        )
        
        parent_radius = node.attributes.get("radius", params.min_radius * 2)
        n_children = len(clusters)
        if n_children > 1:
            child_radii = [
                parent_radius * (1.0 / n_children) ** (1.0/3.0) * params.taper_factor
                for _ in range(n_children)
            ]
        else:
            child_radii = [parent_radius * params.taper_factor]
        
        children_created = 0
        for cluster_idx, cluster in enumerate(clusters):
            if cluster_idx >= params.max_children_per_node:
                break
            
            cluster_direction = np.mean([attraction_vectors[i] for i in cluster], axis=0)
            cluster_direction = cluster_direction / np.linalg.norm(cluster_direction)
            
            cluster_direction = _apply_directional_blending(cluster_direction, node, params)
            cluster_direction = _apply_curvature_constraint(cluster_direction, node, params)
            
            growth_direction = Direction3D.from_array(cluster_direction)
            
            new_pos = Point3D(
                node.position.x + growth_direction.dx * params.step_size,
                node.position.y + growth_direction.dy * params.step_size,
                node.position.z + growth_direction.dz * params.step_size,
            )
            
            if not _check_clearance(new_pos, network, tip_id, params):
                continue
            
            new_radius = max(child_radii[cluster_idx], params.min_radius)
            
            branch_result = grow_branch(
                network,
                from_node_id=tip_id,
                length=params.step_size,
                direction=growth_direction,
                target_radius=new_radius,
                constraints=constraints,
                check_collisions=True,
                seed=int(rng.integers(0, 2**31)) if rng else None,
                spatial_index=state._collision_spatial_index,
                collision_mode=params.collision_mode,
            )
            
            if branch_result.is_success():
                new_seg_id = branch_result.new_ids["segment"]
                new_segment_ids.append(new_seg_id)
                grown_any = True
                children_created += 1
                state.insert_segment_into_spatial_index(new_seg_id)
                if not branch_result.new_ids.get("merged"):
                    new_node_id = branch_result.new_ids["node"]
                    new_node_ids.append(new_node_id)
                    new_node_obj = network.nodes[new_node_id]
                    state.append_node_position(new_node_id, new_node_obj.position)
                    state.active_tip_ids.add(new_node_id)
                    state.tip_states[new_node_id] = TipState(
                        node_id=new_node_id,
                        steps_since_split=0,
                        total_steps=state.tip_states.get(tip_id, TipState(tip_id)).total_steps + 1,
                        distance_from_root=state.tip_states.get(tip_id, TipState(tip_id)).distance_from_root + params.step_size,
                        is_root=False,
                    )
            else:
                if branch_result.errors:
                    _logger.warning(
                        "SC one_step (inlet=%s) step %d: bifurcation grow_branch failed (tip=%s): %s",
                        state.inlet_id, state.global_step, tip_id,
                        "; ".join(branch_result.errors[:3]),
                    )
                warnings.extend(branch_result.errors)
        
        if children_created > 0:
            state.active_tip_ids.discard(tip_id)
            if tip_id in state.tip_states:
                del state.tip_states[tip_id]
    
    for tip_id in linear_tip_ids:
        if tip_id not in linear_directions:
            continue
        
        node = network.nodes[tip_id]
        avg_direction = linear_directions[tip_id]
        growth_direction = Direction3D.from_array(avg_direction)
        
        new_pos = Point3D(
            node.position.x + growth_direction.dx * params.step_size,
            node.position.y + growth_direction.dy * params.step_size,
            node.position.z + growth_direction.dz * params.step_size,
        )
        
        if not _check_clearance(new_pos, network, tip_id, params):
            continue
        
        parent_radius = node.attributes.get("radius", params.min_radius * 2)
        new_radius = max(parent_radius * params.taper_factor, params.min_radius)
        
        do_collision = tip_id not in skip_collision
        
        branch_result = grow_branch(
            network,
            from_node_id=tip_id,
            length=params.step_size,
            direction=growth_direction,
            target_radius=new_radius,
            constraints=constraints,
            check_collisions=do_collision,
            seed=int(rng.integers(0, 2**31)) if rng else None,
            spatial_index=state._collision_spatial_index if do_collision else None,
            collision_mode=params.collision_mode,
        )
        
        if branch_result.is_success():
            new_seg_id = branch_result.new_ids["segment"]
            new_segment_ids.append(new_seg_id)
            grown_any = True
            state.insert_segment_into_spatial_index(new_seg_id)
            state.active_tip_ids.discard(tip_id)
            if tip_id in state.tip_states:
                del state.tip_states[tip_id]
            if not branch_result.new_ids.get("merged"):
                new_node_id = branch_result.new_ids["node"]
                new_node_ids.append(new_node_id)
                new_node_obj = network.nodes[new_node_id]
                state.append_node_position(new_node_id, new_node_obj.position)
                state.active_tip_ids.add(new_node_id)
                state.tip_states[new_node_id] = TipState(
                    node_id=new_node_id,
                    steps_since_split=1,
                    total_steps=1,
                    distance_from_root=params.step_size,
                    is_root=False,
                )
        else:
            if branch_result.errors:
                _logger.warning(
                    "SC one_step (inlet=%s) step %d: grow_branch failed (tip=%s): %s",
                    state.inlet_id, state.global_step, tip_id,
                    "; ".join(branch_result.errors[:3]),
                )
            warnings.extend(branch_result.errors)
    
    if not grown_any and state.global_step > 0:
        n_total_attracted = sum(len(v) for v in attractions.values())
        _logger.warning(
            "SC one_step (inlet=%s) step %d: no growth (attracted=%d, tips=%d)",
            state.inlet_id, state.global_step,
            n_total_attracted, len(state.active_tip_ids),
        )
    
    if new_node_ids:
        state.nodes_added_since_tip_kdtree_rebuild += len(new_node_ids)
        state.nodes_added_since_all_nodes_kdtree_rebuild += len(new_node_ids)
        
        if state.needs_all_nodes_kdtree_rebuild():
            state.rebuild_all_nodes_kdtree()
        
        if state.active_tissue_indices:
            kill_tp_indices = np.array(list(state.active_tissue_indices), dtype=np.intp)
            all_pos = state.get_all_node_positions()
            
            if gpu_idx is not None and gpu_idx.on_gpu:
                kill_mask = gpu_idx.kill_within_radius(kill_tp_indices, all_pos, params.kill_radius)
            elif state._kill_spatial_hash is not None:
                kill_tp_positions = state.tissue_points[kill_tp_indices]
                kill_mask = state._kill_spatial_hash.has_neighbor_mask(
                    kill_tp_positions, all_pos, params.kill_radius,
                )
            elif state.all_nodes_kdtree is not None:
                kill_tp_positions = state.tissue_points[kill_tp_indices]
                kill_mask = _range_search(kill_tp_positions, all_pos, params.kill_radius)
            else:
                kill_mask = np.zeros(len(kill_tp_indices), dtype=bool)
            
            killed = kill_tp_indices[kill_mask]
            state.active_tissue_indices -= set(killed.tolist())
            result.attractors_killed = int(kill_mask.sum())
    
    state.global_step += 1
    state.steps_since_tip_kdtree_rebuild += 1
    state.steps_since_all_nodes_kdtree_rebuild += 1
    
    if grown_any:
        state.consecutive_stall_steps = 0
        state.total_nodes_added += len(new_node_ids)
        state.total_segments_added += len(new_segment_ids)
    else:
        state.consecutive_stall_steps += 1
    
    result.nodes_added = len(new_node_ids)
    result.segments_added = len(new_segment_ids)
    result.active_attractors = len(state.active_tissue_indices)
    result.active_tips = len(state.active_tip_ids)
    result.stalled = state.is_stalled()
    result.exhausted = state.is_exhausted()
    result.new_node_ids = new_node_ids
    result.new_segment_ids = new_segment_ids
    result.warnings = warnings
    
    if state.initial_tissue_count > 0:
        perfused = state.initial_tissue_count - len(state.active_tissue_indices)
        result.coverage_fraction = perfused / state.initial_tissue_count
    
    return result


def run_space_colonization_multi_step(
    state: SpaceColonizationState,
    max_steps: int,
    progress: bool = False,
    progress_desc: str = "Space colonization",
) -> OperationResult:
    """
    Run multiple space colonization steps using the single-step function.
    
    This is a convenience wrapper that provides backward compatibility
    with the old multi-step interface while using the new single-step
    implementation internally.
    
    Parameters
    ----------
    state : SpaceColonizationState
        Persistent state object
    max_steps : int
        Maximum number of steps to run
    progress : bool
        Whether to show progress bar
    progress_desc : str
        Description for progress bar
        
    Returns
    -------
    OperationResult
        Combined result of all steps
    """
    all_new_node_ids = []
    all_new_segment_ids = []
    all_warnings = []
    steps_taken = 0
    
    pbar = None
    if progress:
        pbar = tqdm(total=max_steps, desc=progress_desc, unit="step")
    
    try:
        for step in range(max_steps):
            result = space_colonization_one_step(state)
            
            all_new_node_ids.extend(result.new_node_ids)
            all_new_segment_ids.extend(result.new_segment_ids)
            all_warnings.extend(result.warnings)
            
            if result.nodes_added > 0:
                steps_taken += 1
            
            if pbar:
                pbar.update(1)
                pbar.set_postfix({
                    'nodes': len(all_new_node_ids),
                    'coverage': f'{result.coverage_fraction:.1%}'
                })
            
            if result.exhausted or result.stalled:
                break
    finally:
        if pbar:
            pbar.close()
    
    delta = Delta(
        created_node_ids=all_new_node_ids,
        created_segment_ids=all_new_segment_ids,
    )
    
    if all_new_node_ids:
        status = OperationStatus.SUCCESS if not all_warnings else OperationStatus.PARTIAL_SUCCESS
        message = f"Grew {len(all_new_node_ids)} nodes in {steps_taken} steps"
    else:
        status = OperationStatus.WARNING
        message = "No growth occurred"
    
    return OperationResult(
        status=status,
        message=message,
        new_ids={
            "nodes": all_new_node_ids,
            "segments": all_new_segment_ids,
        },
        warnings=all_warnings,
        delta=delta,
        rng_state=state.network.id_gen.get_state(),
        metadata={
            "steps_taken": steps_taken,
            "nodes_grown": len(all_new_node_ids),
            "initial_tissue_points": state.initial_tissue_count,
            "perfused_tissue_points": state.initial_tissue_count - len(state.active_tissue_indices),
            "coverage_fraction": (state.initial_tissue_count - len(state.active_tissue_indices)) / state.initial_tissue_count if state.initial_tissue_count > 0 else 0.0,
        },
    )
