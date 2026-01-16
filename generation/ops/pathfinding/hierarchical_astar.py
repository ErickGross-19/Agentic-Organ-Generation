"""
Hierarchical A* pathfinding for scale-aware vascular network routing.

This module provides coarse-to-fine corridor pathfinding that enables
pathfinding at fine pitches (e.g., 2.5µm) in large domains (e.g., 50mm)
without exploding memory.

Algorithm:
1. Coarse voxel grid at pitch_coarse (e.g., 50-200µm)
2. Find coarse path (A*)
3. Build corridor volume around coarse path (radius = clearance + local_radius + buffer)
4. Voxelize only the corridor at pitch_fine (e.g., 2.5-10µm)
5. Run fine A* in corridor
6. Smooth and resample path

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import heapq
import logging
import time

import numpy as np

from ...core.domain import DomainSpec
from ...core.types import Point3D
from .astar_voxel import (
    PathfindingResult,
    AStarNode,
    find_path,
)
from aog_policies.pathfinding import PathfindingPolicy, HierarchicalPathfindingPolicy
from aog_policies.resolution import ResolutionPolicy

if TYPE_CHECKING:
    from ...core.network import VascularNetwork
    import trimesh

logger = logging.getLogger(__name__)


@dataclass
class HierarchicalPathfindingResult:
    """Result of hierarchical pathfinding operation."""
    success: bool
    path_pts: Optional[List[np.ndarray]] = None
    path_length: float = 0.0
    coarse_nodes_explored: int = 0
    fine_nodes_explored: int = 0
    time_elapsed: float = 0.0
    coarse_time: float = 0.0
    fine_time: float = 0.0
    warning_messages: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    skipped_waypoints: List[int] = field(default_factory=list)
    effective_fine_pitch: float = 0.0
    pitch_was_relaxed: bool = False
    corridor_voxels: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "path_pts": [p.tolist() for p in self.path_pts] if self.path_pts else None,
            "path_length": self.path_length,
            "coarse_nodes_explored": self.coarse_nodes_explored,
            "fine_nodes_explored": self.fine_nodes_explored,
            "time_elapsed": self.time_elapsed,
            "coarse_time": self.coarse_time,
            "fine_time": self.fine_time,
            "warnings": self.warning_messages,
            "errors": self.errors,
            "skipped_waypoints": self.skipped_waypoints,
            "effective_fine_pitch": self.effective_fine_pitch,
            "pitch_was_relaxed": self.pitch_was_relaxed,
            "corridor_voxels": self.corridor_voxels,
            "metadata": self.metadata,
        }
    
    def to_pathfinding_result(self) -> PathfindingResult:
        """Convert to standard PathfindingResult for compatibility."""
        return PathfindingResult(
            success=self.success,
            path_pts=self.path_pts,
            path_length=self.path_length,
            nodes_explored=self.coarse_nodes_explored + self.fine_nodes_explored,
            time_elapsed=self.time_elapsed,
            warnings=self.warning_messages,
            errors=self.errors,
            skipped_waypoints=self.skipped_waypoints,
            metadata={
                **self.metadata,
                "hierarchical": True,
                "effective_fine_pitch": self.effective_fine_pitch,
                "pitch_was_relaxed": self.pitch_was_relaxed,
                "corridor_voxels": self.corridor_voxels,
            },
        )


class CorridorVoxelMap:
    """
    Voxelized corridor around a coarse path for fine-grained pathfinding.
    
    Only voxelizes the region around the coarse path, enabling fine-pitch
    pathfinding without exploding memory. Uses lazy evaluation with caching
    to avoid O(N³) precomputation.
    """
    
    def __init__(
        self,
        domain: DomainSpec,
        coarse_path: List[np.ndarray],
        corridor_radius: float,
        pitch: float,
        clearance: float = 0.0,
        lazy_evaluation: bool = True,
    ):
        """
        Initialize corridor voxel map.
        
        Parameters
        ----------
        domain : DomainSpec
            Domain specification for containment checks.
        coarse_path : list of np.ndarray
            Coarse path points defining the corridor centerline.
        corridor_radius : float
            Radius of the corridor around the path.
        pitch : float
            Voxel pitch (resolution) in meters.
        clearance : float
            Additional clearance for obstacles.
        lazy_evaluation : bool
            If True, use lazy evaluation for corridor/domain checks
            instead of precomputing the full grid. Default True.
        """
        self.domain = domain
        self.coarse_path = coarse_path
        self.corridor_radius = corridor_radius
        self.pitch = pitch
        self.clearance = clearance
        self.lazy_evaluation = lazy_evaluation
        
        self._compute_corridor_bounds()
        
        self.shape = np.ceil(
            (self.max_bound - self.min_bound) / pitch
        ).astype(int)
        self.shape = np.maximum(self.shape, 1)
        
        self._grid = np.zeros(tuple(self.shape), dtype=bool)
        
        # Lazy evaluation: cache checked voxels
        if lazy_evaluation:
            self._corridor_cache: Dict[Tuple[int, int, int], bool] = {}
            self._in_corridor = None
            self._outside_domain = None
        else:
            self._corridor_cache = None
            self._in_corridor = np.zeros(tuple(self.shape), dtype=bool)
            self._outside_domain = np.zeros(tuple(self.shape), dtype=bool)
            self._compute_corridor_mask()
    
    def _compute_corridor_bounds(self) -> None:
        """Compute bounding box of the corridor."""
        path_array = np.array(self.coarse_path)
        self.min_bound = path_array.min(axis=0) - self.corridor_radius - 2 * self.pitch
        self.max_bound = path_array.max(axis=0) + self.corridor_radius + 2 * self.pitch
        
        domain_bounds = self.domain.get_bounds()
        domain_min = np.array([domain_bounds[0], domain_bounds[2], domain_bounds[4]])
        domain_max = np.array([domain_bounds[1], domain_bounds[3], domain_bounds[5]])
        
        self.min_bound = np.maximum(self.min_bound, domain_min)
        self.max_bound = np.minimum(self.max_bound, domain_max)
    
    def _compute_corridor_mask(self) -> None:
        """Compute mask for voxels inside the corridor and domain (legacy full-grid method)."""
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    pos = self.voxel_to_world((i, j, k))
                    
                    point = Point3D(x=pos[0], y=pos[1], z=pos[2])
                    if not self.domain.contains(point):
                        self._outside_domain[i, j, k] = True
                        continue
                    
                    dist_to_path = self._distance_to_path(pos)
                    if dist_to_path <= self.corridor_radius:
                        self._in_corridor[i, j, k] = True
    
    def _is_in_corridor_and_domain(self, voxel: Tuple[int, int, int]) -> bool:
        """Check if voxel is inside corridor and domain (lazy evaluation with caching)."""
        if self._in_corridor is not None:
            # Using precomputed grids
            return self._in_corridor[voxel] and not self._outside_domain[voxel]
        
        # Lazy evaluation with cache
        if voxel in self._corridor_cache:
            return self._corridor_cache[voxel]
        
        pos = self.voxel_to_world(voxel)
        point = Point3D(x=pos[0], y=pos[1], z=pos[2])
        
        # Check domain containment first
        if not self.domain.contains(point):
            self._corridor_cache[voxel] = False
            return False
        
        # Check corridor distance
        dist_to_path = self._distance_to_path(pos)
        is_in_corridor = dist_to_path <= self.corridor_radius
        self._corridor_cache[voxel] = is_in_corridor
        return is_in_corridor
    
    def _distance_to_path(self, point: np.ndarray) -> float:
        """Compute minimum distance from point to coarse path."""
        min_dist = float('inf')
        
        for i in range(len(self.coarse_path) - 1):
            seg_start = self.coarse_path[i]
            seg_end = self.coarse_path[i + 1]
            dist = self._point_to_segment_distance(point, seg_start, seg_end)
            min_dist = min(min_dist, dist)
        
        return min_dist
    
    def _point_to_segment_distance(
        self,
        point: np.ndarray,
        seg_start: np.ndarray,
        seg_end: np.ndarray,
    ) -> float:
        """Compute distance from point to line segment."""
        v = seg_end - seg_start
        w = point - seg_start
        
        c1 = np.dot(w, v)
        if c1 <= 0:
            return float(np.linalg.norm(point - seg_start))
        
        c2 = np.dot(v, v)
        if c2 <= c1:
            return float(np.linalg.norm(point - seg_end))
        
        t = c1 / c2
        closest = seg_start + t * v
        return float(np.linalg.norm(point - closest))
    
    def world_to_voxel(self, pos: np.ndarray) -> Tuple[int, int, int]:
        """Convert world position to voxel indices."""
        voxel = np.floor((pos - self.min_bound) / self.pitch).astype(int)
        voxel = np.clip(voxel, 0, self.shape - 1)
        return tuple(voxel)
    
    def voxel_to_world(self, voxel: Tuple[int, int, int]) -> np.ndarray:
        """Convert voxel indices to world position (center of voxel)."""
        return self.min_bound + (np.array(voxel) + 0.5) * self.pitch
    
    def is_valid_voxel(self, voxel: Tuple[int, int, int]) -> bool:
        """Check if voxel indices are within bounds."""
        return all(0 <= v < s for v, s in zip(voxel, self.shape))
    
    def is_free(self, voxel: Tuple[int, int, int]) -> bool:
        """Check if a voxel is free (not obstacle, inside corridor and domain)."""
        if not self.is_valid_voxel(voxel):
            return False
        return not self._grid[voxel] and self._is_in_corridor_and_domain(voxel)
    
    def add_capsule_obstacle(
        self,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
    ) -> None:
        """Add a capsule obstacle."""
        inflated_radius = radius + self.clearance
        
        min_pt = np.minimum(start, end) - inflated_radius
        max_pt = np.maximum(start, end) + inflated_radius
        
        min_voxel = self.world_to_voxel(min_pt)
        max_voxel = self.world_to_voxel(max_pt)
        
        for i in range(min_voxel[0], max_voxel[0] + 1):
            for j in range(min_voxel[1], max_voxel[1] + 1):
                for k in range(min_voxel[2], max_voxel[2] + 1):
                    if not self.is_valid_voxel((i, j, k)):
                        continue
                    
                    voxel_center = self.voxel_to_world((i, j, k))
                    dist = self._point_to_segment_distance(voxel_center, start, end)
                    
                    if dist <= inflated_radius:
                        self._grid[i, j, k] = True
    
    def add_network_obstacles(
        self,
        network: "VascularNetwork",
        radius_margin: float = 0.0,
    ) -> None:
        """Add network segments as obstacles."""
        for segment in network.segments.values():
            start = segment.geometry.start.to_array()
            end = segment.geometry.end.to_array()
            radius = segment.geometry.mean_radius()
            
            inflated_radius = radius + self.clearance + radius_margin
            self.add_capsule_obstacle(start, end, inflated_radius)
    
    def get_voxel_count(self) -> int:
        """Get total number of voxels in the corridor."""
        return int(np.sum(self._in_corridor))


def _estimate_corridor_voxels(
    coarse_path: List[np.ndarray],
    corridor_radius: float,
    pitch: float,
) -> int:
    """Estimate number of voxels in a corridor."""
    path_length = sum(
        np.linalg.norm(coarse_path[i + 1] - coarse_path[i])
        for i in range(len(coarse_path) - 1)
    )
    
    cylinder_volume = np.pi * corridor_radius**2 * path_length
    
    voxel_volume = pitch**3
    
    return int(cylinder_volume / voxel_volume * 1.5)


def _relax_pitch_for_corridor(
    coarse_path: List[np.ndarray],
    corridor_radius: float,
    base_pitch: float,
    max_voxels: int,
    pitch_step_factor: float,
    max_pitch: float = 1e-3,
) -> Tuple[float, bool, str]:
    """
    Relax pitch if corridor would exceed voxel budget.
    
    Returns
    -------
    tuple
        (effective_pitch, was_relaxed, warning_message)
    """
    pitch = base_pitch
    was_relaxed = False
    warning = ""
    
    while True:
        estimated_voxels = _estimate_corridor_voxels(coarse_path, corridor_radius, pitch)
        
        if estimated_voxels <= max_voxels:
            break
        
        if pitch >= max_pitch:
            warning = (
                f"Pitch relaxation hit max_pitch limit ({max_pitch:.2e} m). "
                f"Estimated corridor voxels {estimated_voxels:,} exceeds budget {max_voxels:,}."
            )
            break
        
        new_pitch = pitch * pitch_step_factor
        new_pitch = min(new_pitch, max_pitch)
        
        if not was_relaxed:
            warning = (
                f"Fine pitch relaxed from {base_pitch:.2e} m to {new_pitch:.2e} m "
                f"to fit corridor voxel budget {max_voxels:,}. "
                f"Min diameter resolution may be reduced."
            )
        
        pitch = new_pitch
        was_relaxed = True
    
    return pitch, was_relaxed, warning


def find_path_hierarchical(
    domain: DomainSpec,
    start: np.ndarray,
    goal: np.ndarray,
    local_radius: float = 0.0,
    obstacles: Optional[List[Dict[str, Any]]] = None,
    network: Optional["VascularNetwork"] = None,
    mesh_obstacles: Optional[List["trimesh.Trimesh"]] = None,
    policy: Optional[HierarchicalPathfindingPolicy] = None,
    resolution_policy: Optional[ResolutionPolicy] = None,
) -> HierarchicalPathfindingResult:
    """
    Find a path using hierarchical coarse-to-fine A*.
    
    C1 FIX: This is the mandatory pathfinding entry point for all A* requests.
    C2 FIX: Coarse-stage uses resolution resolver for voxel budgeting.
    
    Parameters
    ----------
    domain : DomainSpec
        Domain specification for bounds and containment.
    start : np.ndarray
        Start position (x, y, z) in meters.
    goal : np.ndarray
        Goal position (x, y, z) in meters.
    local_radius : float, optional
        Local vessel radius for obstacle inflation.
    obstacles : list of dict, optional
        List of capsule obstacles with 'start', 'end', 'radius' keys.
    network : VascularNetwork, optional
        Network to use as obstacles.
    mesh_obstacles : list of trimesh.Trimesh, optional
        Mesh obstacles to avoid.
    policy : HierarchicalPathfindingPolicy, optional
        Pathfinding configuration.
    resolution_policy : ResolutionPolicy, optional
        Resolution policy for pitch derivation and voxel budgeting.
    
    Returns
    -------
    HierarchicalPathfindingResult
        Result containing path points and metadata.
    """
    if policy is None:
        policy = HierarchicalPathfindingPolicy()
    
    start_time = time.time()
    result_warnings = []
    
    # C2 FIX: Use resolution resolver for coarse pitch if resolution_policy provided
    effective_coarse_pitch = policy.pitch_coarse
    coarse_pitch_relaxed = False
    
    if resolution_policy is not None and policy.use_resolution_policy:
        from ...utils.resolution_resolver import resolve_pitch
        
        # Get domain bounding box for pitch calculation
        domain_bounds = domain.get_bounds()
        bbox = (
            domain_bounds[0], domain_bounds[1],  # x_min, x_max
            domain_bounds[2], domain_bounds[3],  # y_min, y_max
            domain_bounds[4], domain_bounds[5],  # z_min, z_max
        )
        
        # Resolve coarse pitch with voxel budget
        resolution_result = resolve_pitch(
            op_name="pathfinding_coarse",
            requested_pitch=policy.pitch_coarse,
            bbox=bbox,
            resolution_policy=resolution_policy,
            max_voxels_override=policy.max_voxels_coarse,
        )
        
        effective_coarse_pitch = resolution_result["effective_pitch"]
        coarse_pitch_relaxed = resolution_result.get("was_relaxed", False)
        
        if resolution_result.get("warnings"):
            result_warnings.extend(resolution_result["warnings"])
            for warning in resolution_result["warnings"]:
                logger.warning(warning)
    
    coarse_policy = policy.to_coarse_policy()
    coarse_policy.voxel_pitch = effective_coarse_pitch  # Use resolved pitch
    
    coarse_result = find_path(
        domain=domain,
        start=start,
        goal=goal,
        obstacles=obstacles,
        network=network,
        mesh_obstacles=mesh_obstacles,
        policy=coarse_policy,
    )
    
    coarse_time = time.time() - start_time
    
    if not coarse_result.success:
        return HierarchicalPathfindingResult(
            success=False,
            coarse_nodes_explored=coarse_result.nodes_explored,
            coarse_time=coarse_time,
            time_elapsed=time.time() - start_time,
            errors=["Coarse pathfinding failed: " + "; ".join(coarse_result.errors)],
        )
    
    coarse_path = coarse_result.path_pts
    
    corridor_radius = policy.clearance + local_radius + policy.corridor_radius_buffer
    
    effective_fine_pitch = policy.pitch_fine
    pitch_was_relaxed = False
    
    if policy.auto_relax_fine_pitch:
        effective_fine_pitch, pitch_was_relaxed, relax_warning = _relax_pitch_for_corridor(
            coarse_path=coarse_path,
            corridor_radius=corridor_radius,
            base_pitch=policy.pitch_fine,
            max_voxels=policy.max_voxels_fine,
            pitch_step_factor=policy.pitch_step_factor,
        )
        
        if relax_warning:
            result_warnings.append(relax_warning)
            logger.warning(relax_warning)
    
    fine_start_time = time.time()
    
    corridor_map = CorridorVoxelMap(
        domain=domain,
        coarse_path=coarse_path,
        corridor_radius=corridor_radius,
        pitch=effective_fine_pitch,
        clearance=policy.clearance,
    )
    
    corridor_voxels = corridor_map.get_voxel_count()
    
    if obstacles:
        for obs in obstacles:
            corridor_map.add_capsule_obstacle(
                np.array(obs["start"]),
                np.array(obs["end"]),
                obs["radius"],
            )
    
    if network:
        corridor_map.add_network_obstacles(network, radius_margin=local_radius)
    
    start_voxel = corridor_map.world_to_voxel(start)
    goal_voxel = corridor_map.world_to_voxel(goal)
    
    if not corridor_map.is_free(start_voxel):
        return HierarchicalPathfindingResult(
            success=False,
            coarse_nodes_explored=coarse_result.nodes_explored,
            coarse_time=coarse_time,
            time_elapsed=time.time() - start_time,
            errors=["Start position is blocked in fine corridor"],
            warnings=result_warnings,
        )
    
    if not corridor_map.is_free(goal_voxel):
        return HierarchicalPathfindingResult(
            success=False,
            coarse_nodes_explored=coarse_result.nodes_explored,
            coarse_time=coarse_time,
            time_elapsed=time.time() - start_time,
            errors=["Goal position is blocked in fine corridor"],
            warnings=result_warnings,
        )
    
    fine_policy = policy.to_fine_policy()
    fine_policy.voxel_pitch = effective_fine_pitch
    
    path_voxels, fine_nodes_explored = _astar_search_corridor(
        corridor_map=corridor_map,
        start_voxel=start_voxel,
        goal_voxel=goal_voxel,
        policy=fine_policy,
        timeout=fine_policy.timeout_s,
        start_time=fine_start_time,
    )
    
    fine_time = time.time() - fine_start_time
    
    if path_voxels is None:
        return HierarchicalPathfindingResult(
            success=False,
            coarse_nodes_explored=coarse_result.nodes_explored,
            fine_nodes_explored=fine_nodes_explored,
            coarse_time=coarse_time,
            fine_time=fine_time,
            time_elapsed=time.time() - start_time,
            errors=["Fine pathfinding failed in corridor"],
            warnings=result_warnings,
            effective_fine_pitch=effective_fine_pitch,
            pitch_was_relaxed=pitch_was_relaxed,
            corridor_voxels=corridor_voxels,
        )
    
    path_pts = [corridor_map.voxel_to_world(v) for v in path_voxels]
    
    path_pts[0] = start.copy()
    path_pts[-1] = goal.copy()
    
    if fine_policy.smoothing_enabled and len(path_pts) > 2:
        path_pts = _smooth_path_corridor(
            path_pts,
            corridor_map,
            fine_policy.smoothing_iters,
            fine_policy.smoothing_strength,
        )
    
    path_length = sum(
        np.linalg.norm(path_pts[i + 1] - path_pts[i])
        for i in range(len(path_pts) - 1)
    )
    
    return HierarchicalPathfindingResult(
        success=True,
        path_pts=path_pts,
        path_length=path_length,
        coarse_nodes_explored=coarse_result.nodes_explored,
        fine_nodes_explored=fine_nodes_explored,
        coarse_time=coarse_time,
        fine_time=fine_time,
        time_elapsed=time.time() - start_time,
        warnings=result_warnings,
        effective_fine_pitch=effective_fine_pitch,
        pitch_was_relaxed=pitch_was_relaxed,
        corridor_voxels=corridor_voxels,
        metadata={
            "coarse_pitch": effective_coarse_pitch,  # C2 FIX: Use effective pitch
            "coarse_pitch_requested": policy.pitch_coarse,
            "coarse_pitch_relaxed": coarse_pitch_relaxed,
            "fine_pitch": effective_fine_pitch,
            "corridor_radius": corridor_radius,
            "coarse_path_length": coarse_result.path_length,
        },
    )


def _astar_search_corridor(
    corridor_map: CorridorVoxelMap,
    start_voxel: Tuple[int, int, int],
    goal_voxel: Tuple[int, int, int],
    policy: PathfindingPolicy,
    timeout: float,
    start_time: float,
) -> Tuple[Optional[List[Tuple[int, int, int]]], int]:
    """
    Run A* search within a corridor voxel map.
    
    Returns
    -------
    tuple
        (path_voxels, nodes_explored) or (None, nodes_explored) if no path found.
    """
    def heuristic(voxel: Tuple[int, int, int]) -> float:
        dx = abs(voxel[0] - goal_voxel[0])
        dy = abs(voxel[1] - goal_voxel[1])
        dz = abs(voxel[2] - goal_voxel[2])
        return policy.heuristic_weight * corridor_map.pitch * (dx + dy + dz)
    
    if policy.diagonal_movement:
        neighbors = [
            (dx, dy, dz)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            for dz in [-1, 0, 1]
            if (dx, dy, dz) != (0, 0, 0)
        ]
    else:
        neighbors = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
        ]
    
    start_node = AStarNode(
        voxel=start_voxel,
        g=0.0,
        h=heuristic(start_voxel),
    )
    
    open_set = [start_node]
    closed_set: set = set()
    g_scores: Dict[Tuple[int, int, int], float] = {start_voxel: 0.0}
    nodes_explored = 0
    
    while open_set:
        if time.time() - start_time > timeout:
            logger.warning("A* search timed out in corridor")
            return None, nodes_explored
        
        if nodes_explored >= policy.max_nodes:
            logger.warning("A* search exceeded max nodes in corridor")
            return None, nodes_explored
        
        current = heapq.heappop(open_set)
        
        if current.voxel in closed_set:
            continue
        
        closed_set.add(current.voxel)
        nodes_explored += 1
        
        if current.voxel == goal_voxel:
            path = []
            node = current
            while node is not None:
                path.append(node.voxel)
                node = node.parent
            return path[::-1], nodes_explored
        
        for delta in neighbors:
            neighbor_voxel = (
                current.voxel[0] + delta[0],
                current.voxel[1] + delta[1],
                current.voxel[2] + delta[2],
            )
            
            if neighbor_voxel in closed_set:
                continue
            
            if not corridor_map.is_free(neighbor_voxel):
                continue
            
            move_cost = corridor_map.pitch * np.sqrt(
                delta[0]**2 + delta[1]**2 + delta[2]**2
            )
            
            if current.direction is not None and delta != current.direction:
                move_cost += policy.turn_penalty * corridor_map.pitch
            
            tentative_g = current.g + move_cost
            
            if neighbor_voxel in g_scores and tentative_g >= g_scores[neighbor_voxel]:
                continue
            
            g_scores[neighbor_voxel] = tentative_g
            
            neighbor_node = AStarNode(
                voxel=neighbor_voxel,
                g=tentative_g,
                h=heuristic(neighbor_voxel),
                parent=current,
                direction=delta,
            )
            
            heapq.heappush(open_set, neighbor_node)
    
    return None, nodes_explored


def _smooth_path_corridor(
    path_pts: List[np.ndarray],
    corridor_map: CorridorVoxelMap,
    iterations: int,
    strength: float,
) -> List[np.ndarray]:
    """Smooth path while keeping it within the corridor."""
    smoothed = [p.copy() for p in path_pts]
    
    for _ in range(iterations):
        new_smoothed = [smoothed[0].copy()]
        
        for i in range(1, len(smoothed) - 1):
            avg = (smoothed[i - 1] + smoothed[i + 1]) / 2
            new_pos = smoothed[i] + strength * (avg - smoothed[i])
            
            voxel = corridor_map.world_to_voxel(new_pos)
            if corridor_map.is_free(voxel):
                new_smoothed.append(new_pos)
            else:
                new_smoothed.append(smoothed[i].copy())
        
        new_smoothed.append(smoothed[-1].copy())
        smoothed = new_smoothed
    
    return smoothed


__all__ = [
    "HierarchicalPathfindingPolicy",
    "HierarchicalPathfindingResult",
    "CorridorVoxelMap",
    "find_path_hierarchical",
]
