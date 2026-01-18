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
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from ..core.types import Point3D, Direction3D
from ..core.network import VascularNetwork
from ..core.result import OperationResult, OperationStatus, Delta
from ..rules.constraints import BranchingConstraints


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
    
    # Handle both Point3D objects and array-like inputs
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
    
    pbar = tqdm(total=params.max_steps, desc="Space colonization", unit="step")
    
    for step in range(params.max_steps):
        if not active_tissue_points:
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
        
        # Build KDTree from terminal node positions for O(n log n) nearest neighbor queries
        if terminal_nodes:
            terminal_positions = np.array([
                [node.position.x, node.position.y, node.position.z]
                for node in terminal_nodes
            ])
            terminal_kdtree = cKDTree(terminal_positions)
            terminal_id_list = [node.id for node in terminal_nodes]
            
            # Get active tissue point positions
            active_tp_indices = list(active_tissue_points)
            if active_tp_indices:
                active_tp_positions = np.array([
                    [tissue_points_list[idx].x, tissue_points_list[idx].y, tissue_points_list[idx].z]
                    for idx in active_tp_indices
                ])
                
                # Query nearest terminal for each active tissue point
                distances, nearest_indices = terminal_kdtree.query(active_tp_positions, k=1)
                
                # Assign tissue points to their nearest terminal within influence_radius
                for i, tp_idx in enumerate(active_tp_indices):
                    if distances[i] < params.influence_radius:
                        nearest_terminal_id = terminal_id_list[nearest_indices[i]]
                        attractions[nearest_terminal_id].append(tp_idx)
        
        grown_any = False
        for node in terminal_nodes:
            if not attractions[node.id]:
                continue
            
            attracted_points = [tissue_points_list[idx] for idx in attractions[node.id]]
            num_attractions = len(attracted_points)
            
            # Check if bifurcation conditions are met
            should_bifurcate = (
                params.encourage_bifurcation and
                num_attractions >= params.min_attractions_for_bifurcation
            )
            
            if should_bifurcate:
                # Compute attraction vectors
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
                            
                            from .growth import grow_branch
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
                                
                                # Check clearance
                                new_pos = Point3D(
                                    node.position.x + growth_direction.dx * params.step_size,
                                    node.position.y + growth_direction.dy * params.step_size,
                                    node.position.z + growth_direction.dz * params.step_size,
                                )
                                
                                if not _check_clearance(new_pos, network, node.id, params):
                                    continue
                                
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
                                )
                                
                                if result.is_success():
                                    new_node_ids.append(result.new_ids["node"])
                                    new_segment_ids.append(result.new_ids["segment"])
                                    grown_any = True
                                else:
                                    warnings.extend(result.errors)
                            
                            continue
            
            avg_direction = np.zeros(3)
            
            for tp in attracted_points:
                direction = np.array([
                    tp.x - node.position.x,
                    tp.y - node.position.y,
                    tp.z - node.position.z,
                ])
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 1e-10:
                    avg_direction += direction / direction_norm
            
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
            
            if not _check_clearance(new_pos, network, node.id, params):
                continue
            
            parent_radius = node.attributes.get("radius", params.min_radius * 2)
            new_radius = parent_radius * params.taper_factor
            
            # Policy-driven clamping: clamp to min_radius instead of skipping growth
            # This ensures growth continues even when taper would drop below min_radius
            new_radius = max(new_radius, params.min_radius)
            
            from .growth import grow_branch
            result = grow_branch(
                network,
                from_node_id=node.id,
                length=params.step_size,
                direction=growth_direction,
                target_radius=new_radius,
                constraints=constraints,
                check_collisions=True,
                seed=seed,
            )
            
            if result.is_success():
                new_node_ids.append(result.new_ids["node"])
                new_segment_ids.append(result.new_ids["segment"])
                grown_any = True
            else:
                warnings.extend(result.errors)
        
        if not grown_any:
            pbar.close()
            break
        
        steps_taken += 1
        pbar.update(1)
        pbar.set_postfix({
            'nodes': len(new_node_ids),
            'coverage': f'{(initial_count - len(active_tissue_points)) / initial_count:.1%}' if initial_count > 0 else '0%'
        })
        
        # Use KDTree for efficient kill radius pruning - O(n log n) instead of O(n²)
        if network.nodes and active_tissue_points:
            all_node_positions = np.array([
                [node.position.x, node.position.y, node.position.z]
                for node in network.nodes.values()
            ])
            node_kdtree = cKDTree(all_node_positions)
            
            # Get positions of active tissue points
            active_tp_indices = list(active_tissue_points)
            active_tp_positions = np.array([
                [tissue_points_list[idx].x, tissue_points_list[idx].y, tissue_points_list[idx].z]
                for idx in active_tp_indices
            ])
            
            # Query all tissue points within kill_radius of any node
            # query_ball_point returns indices of nodes within radius for each query point
            nearby_results = node_kdtree.query_ball_point(active_tp_positions, params.kill_radius)
            
            # Remove tissue points that have at least one node within kill_radius
            for i, tp_idx in enumerate(active_tp_indices):
                if nearby_results[i]:  # Non-empty list means at least one node is within kill_radius
                    active_tissue_points.discard(tp_idx)
    
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
    
    max_angle = 0.0
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            cos_angle = np.clip(np.dot(vectors[i], vectors[j]), -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            max_angle = max(max_angle, angle)
    
    return max_angle


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
    
    normalized_vectors = []
    for vec in attraction_vectors:
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            normalized_vectors.append(vec / norm)
        else:
            normalized_vectors.append(vec)
    
    if n <= max_clusters:
        return [[i] for i in range(n)]
    
    K = min(max_clusters, n)
    
    # Farthest-first initialization for K centroids
    centroids = []
    centroid_indices = []
    
    centroids.append(normalized_vectors[0].copy())
    centroid_indices.append(0)
    
    for _ in range(K - 1):
        max_min_dist = -1.0
        farthest_idx = 0
        
        for i in range(n):
            if i in centroid_indices:
                continue
            
            # Find minimum distance (maximum similarity) to existing centroids
            min_sim = 1.0
            for centroid in centroids:
                sim = np.dot(normalized_vectors[i], centroid)
                if sim < min_sim:
                    min_sim = sim
            
            # Distance metric: 1 - similarity (higher is more separated)
            min_dist = 1.0 - min_sim
            
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                farthest_idx = i
        
        centroids.append(normalized_vectors[farthest_idx].copy())
        centroid_indices.append(farthest_idx)
    
    for iteration in range(10):
        clusters = [[] for _ in range(K)]
        
        # Assign each vector to nearest centroid (highest dot product)
        for idx, vec in enumerate(normalized_vectors):
            best_cluster = 0
            best_sim = np.dot(vec, centroids[0])
            
            for c in range(1, K):
                sim = np.dot(vec, centroids[c])
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = c
            
            clusters[best_cluster].append(idx)
        
        # Update centroids
        changed = False
        for c in range(K):
            if clusters[c]:
                new_centroid = np.mean([normalized_vectors[idx] for idx in clusters[c]], axis=0)
                centroid_norm = np.linalg.norm(new_centroid)
                
                if centroid_norm > 1e-10:
                    new_centroid = new_centroid / centroid_norm
                    
                    if np.linalg.norm(new_centroid - centroids[c]) > 1e-6:
                        changed = True
                        centroids[c] = new_centroid
        
        if not changed:
            break
    
    # Filter out empty clusters
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
) -> bool:
    """
    Check if new position maintains minimum clearance from other segments.
    
    Uses SpatialIndex for efficient local neighborhood queries instead of
    scanning all segments (O(local) instead of O(segments)).
    
    Returns True if clearance is acceptable, False otherwise.
    """
    if params.min_clearance is None:
        return True  # No clearance check
    
    # Use spatial index for efficient local neighborhood query
    # Search radius includes clearance plus typical segment radius
    search_radius = params.min_clearance * 3.0  # Conservative search radius
    
    spatial_index = network.get_spatial_index()
    nearby_segments = spatial_index.query_nearby_segments(new_position, search_radius)
    
    # Check distance only to nearby segments not connected to from_node
    for seg in nearby_segments:
        # Skip segments connected to from_node
        if seg.start_node_id == from_node_id or seg.end_node_id == from_node_id:
            continue
        
        # Compute distance from new_position to segment
        p1 = network.nodes[seg.start_node_id].position
        p2 = network.nodes[seg.end_node_id].position
        
        # Distance from point to line segment
        v = np.array([p2.x - p1.x, p2.y - p1.y, p2.z - p1.z])
        w = np.array([new_position.x - p1.x, new_position.y - p1.y, new_position.z - p1.z])
        
        v_len_sq = np.dot(v, v)
        if v_len_sq < 1e-10:
            # Degenerate segment
            dist = np.linalg.norm(w)
        else:
            t = np.clip(np.dot(w, v) / v_len_sq, 0.0, 1.0)
            projection = p1.to_array() + t * v
            dist = np.linalg.norm(new_position.to_array() - projection)
        
        # Check clearance (accounting for radii)
        seg_radius = seg.attributes.get("radius", 0.001)
        required_clearance = params.min_clearance + seg_radius
        
        if dist < required_clearance:
            return False  # Too close
    
    return True  # Clearance OK


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
                    
                    from .growth import grow_branch
                    
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
                        )
                        
                        if result.is_success():
                            new_node_id = result.new_ids["node"]
                            new_node_ids.append(new_node_id)
                            new_segment_ids.append(result.new_ids["segment"])
                            grown_any = True
                            children_created += 1
                            
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
            
            from .growth import grow_branch
            result = grow_branch(
                network,
                from_node_id=node_id,
                length=params.step_size,
                direction=growth_direction,
                target_radius=new_radius,
                constraints=constraints,
                check_collisions=True,
                seed=seed,
            )
            
            if result.is_success():
                new_node_id = result.new_ids["node"]
                new_node_ids.append(new_node_id)
                new_segment_ids.append(result.new_ids["segment"])
                grown_any = True
                
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
                warnings.extend(result.errors)
        
        if not grown_any:
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
