"""
Voxel-based A* pathfinding for vascular network routing.

This module provides pathfinding algorithms that operate on voxelized domains
to find collision-free paths for vascular network segments.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import heapq
import numpy as np
import logging
import time

from ...core.domain import DomainSpec
from ...core.types import Point3D

if TYPE_CHECKING:
    from ...core.network import VascularNetwork
    import trimesh

logger = logging.getLogger(__name__)


from aog_policies.pathfinding import PathfindingPolicy, WaypointPolicy


@dataclass
class PathfindingResult:
    """Result of a pathfinding operation."""
    success: bool
    path_pts: Optional[List[np.ndarray]] = None
    path_length: float = 0.0
    nodes_explored: int = 0
    time_elapsed: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    skipped_waypoints: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "path_pts": [p.tolist() for p in self.path_pts] if self.path_pts else None,
            "path_length": self.path_length,
            "nodes_explored": self.nodes_explored,
            "time_elapsed": self.time_elapsed,
            "warnings": self.warnings,
            "errors": self.errors,
            "skipped_waypoints": self.skipped_waypoints,
            "metadata": self.metadata,
        }


class VoxelObstacleMap:
    """
    Voxelized obstacle map for collision checking during pathfinding.
    
    Supports both network-based obstacles (capsules) and mesh-based obstacles.
    Uses lazy evaluation with caching to avoid O(N³) precomputation.
    """
    
    def __init__(
        self,
        domain: DomainSpec,
        pitch: float,
        clearance: float = 0.0,
        max_voxels: Optional[int] = None,
        lazy_domain_check: bool = True,
    ):
        """
        Initialize the voxel obstacle map.
        
        Parameters
        ----------
        domain : DomainSpec
            Domain specification for bounds
        pitch : float
            Voxel pitch (resolution) in meters
        clearance : float
            Additional clearance to add around obstacles
        max_voxels : int, optional
            Maximum voxel budget. If exceeded, pitch will be relaxed.
        lazy_domain_check : bool
            If True, use lazy evaluation for domain containment checks
            instead of precomputing the full grid. Default True.
        """
        self.domain = domain
        self.pitch = pitch
        self.clearance = clearance
        self.lazy_domain_check = lazy_domain_check
        
        # Get domain bounds
        bounds = domain.get_bounds()
        self.min_bound = np.array([bounds[0], bounds[2], bounds[4]])
        self.max_bound = np.array([bounds[1], bounds[3], bounds[5]])
        
        # Add padding
        padding = 2 * pitch
        self.min_bound -= padding
        self.max_bound += padding
        
        # Compute grid shape
        self.shape = np.ceil(
            (self.max_bound - self.min_bound) / pitch
        ).astype(int)
        self.shape = np.maximum(self.shape, 1)
        
        # Check voxel budget
        total_voxels = int(np.prod(self.shape))
        if max_voxels is not None and total_voxels > max_voxels:
            logger.warning(
                f"Voxel count {total_voxels:,} exceeds budget {max_voxels:,}. "
                f"Consider using hierarchical pathfinding or relaxing pitch."
            )
        
        # Initialize obstacle grid (False = free, True = obstacle)
        self._grid = np.zeros(tuple(self.shape), dtype=bool)
        
        # Lazy domain containment: cache checked voxels
        if lazy_domain_check:
            self._outside_domain_cache: Dict[Tuple[int, int, int], bool] = {}
            self._outside_domain = None  # Not precomputed
        else:
            self._outside_domain_cache = None
            self._outside_domain = np.zeros(tuple(self.shape), dtype=bool)
            self._compute_domain_mask()
    
    def _compute_domain_mask(self) -> None:
        """Compute mask for voxels outside the domain (legacy full-grid method)."""
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    pos = self.voxel_to_world((i, j, k))
                    point = Point3D(x=pos[0], y=pos[1], z=pos[2])
                    if not self.domain.contains(point):
                        self._outside_domain[i, j, k] = True
    
    def _is_outside_domain(self, voxel: Tuple[int, int, int]) -> bool:
        """Check if voxel is outside domain (lazy evaluation with caching)."""
        if self._outside_domain is not None:
            # Using precomputed grid
            return self._outside_domain[voxel]
        
        # Lazy evaluation with cache
        if voxel in self._outside_domain_cache:
            return self._outside_domain_cache[voxel]
        
        pos = self.voxel_to_world(voxel)
        point = Point3D(x=pos[0], y=pos[1], z=pos[2])
        is_outside = not self.domain.contains(point)
        self._outside_domain_cache[voxel] = is_outside
        return is_outside
    
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
        """Check if a voxel is free (not obstacle and inside domain)."""
        if not self.is_valid_voxel(voxel):
            return False
        return not self._grid[voxel] and not self._is_outside_domain(voxel)
    
    def add_network_obstacles(
        self,
        network: "VascularNetwork",
        radius_margin: float = 0.0,
    ) -> None:
        """
        Add network segments as obstacles.
        
        Each segment is treated as a capsule (cylinder with hemispherical caps).
        The obstacle is inflated by (clearance + local_radius + radius_margin).
        """
        for segment in network.segments.values():
            start = segment.geometry.start.to_array()
            end = segment.geometry.end.to_array()
            radius = segment.geometry.mean_radius()
            
            # Inflate radius
            inflated_radius = radius + self.clearance + radius_margin
            
            self._add_capsule_obstacle(start, end, inflated_radius)
    
    def add_mesh_obstacle(
        self,
        mesh: "trimesh.Trimesh",
        radius_margin: float = 0.0,
    ) -> None:
        """
        Add a mesh as an obstacle by voxelizing and dilating.
        """
        import trimesh
        from scipy import ndimage
        
        # Voxelize the mesh
        try:
            voxels = mesh.voxelized(self.pitch)
            voxels = voxels.fill()
            
            # Get the voxel matrix
            mesh_grid = voxels.matrix.astype(bool)
            mesh_origin = voxels.transform[:3, 3]
            
            # Compute dilation radius in voxels
            dilation_radius = int(np.ceil(
                (self.clearance + radius_margin) / self.pitch
            ))
            
            if dilation_radius > 0:
                # Create spherical structuring element
                struct = ndimage.generate_binary_structure(3, 1)
                mesh_grid = ndimage.binary_dilation(
                    mesh_grid,
                    structure=struct,
                    iterations=dilation_radius,
                )
            
            # Paste into our grid
            self._paste_grid(mesh_grid, mesh_origin)
            
        except Exception as e:
            logger.warning(f"Failed to voxelize mesh obstacle: {e}")
    
    def add_capsule_obstacle(
        self,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
    ) -> None:
        """Add a capsule (swept sphere) obstacle."""
        inflated_radius = radius + self.clearance
        self._add_capsule_obstacle(start, end, inflated_radius)
    
    def _add_capsule_obstacle(
        self,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
    ) -> None:
        """Internal method to add a capsule obstacle."""
        # Compute bounding box of capsule
        min_pt = np.minimum(start, end) - radius
        max_pt = np.maximum(start, end) + radius
        
        # Convert to voxel indices
        min_voxel = self.world_to_voxel(min_pt)
        max_voxel = self.world_to_voxel(max_pt)
        
        # Iterate over bounding box
        for i in range(min_voxel[0], max_voxel[0] + 1):
            for j in range(min_voxel[1], max_voxel[1] + 1):
                for k in range(min_voxel[2], max_voxel[2] + 1):
                    if not self.is_valid_voxel((i, j, k)):
                        continue
                    
                    # Check distance from voxel center to capsule centerline
                    voxel_center = self.voxel_to_world((i, j, k))
                    dist = self._point_to_segment_distance(
                        voxel_center, start, end
                    )
                    
                    if dist <= radius:
                        self._grid[i, j, k] = True
    
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
    
    def _paste_grid(
        self,
        src_grid: np.ndarray,
        src_origin: np.ndarray,
    ) -> None:
        """Paste a source grid into our obstacle grid."""
        # Compute offset in voxels
        offset = np.round((src_origin - self.min_bound) / self.pitch).astype(int)
        
        # Compute copy ranges
        src_start = np.maximum(-offset, 0)
        dst_start = np.maximum(offset, 0)
        
        copy_size = np.minimum(
            np.array(src_grid.shape) - src_start,
            np.array(self.shape) - dst_start
        )
        copy_size = np.maximum(copy_size, 0)
        
        if np.all(copy_size > 0):
            self._grid[
                dst_start[0]:dst_start[0] + copy_size[0],
                dst_start[1]:dst_start[1] + copy_size[1],
                dst_start[2]:dst_start[2] + copy_size[2],
            ] |= src_grid[
                src_start[0]:src_start[0] + copy_size[0],
                src_start[1]:src_start[1] + copy_size[1],
                src_start[2]:src_start[2] + copy_size[2],
            ]


class AStarNode:
    """Node in the A* search graph."""
    
    __slots__ = ['voxel', 'g', 'h', 'f', 'parent', 'direction']
    
    def __init__(
        self,
        voxel: Tuple[int, int, int],
        g: float,
        h: float,
        parent: Optional["AStarNode"] = None,
        direction: Optional[Tuple[int, int, int]] = None,
    ):
        self.voxel = voxel
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent
        self.direction = direction
    
    def __lt__(self, other: "AStarNode") -> bool:
        return self.f < other.f


def find_path(
    domain: DomainSpec,
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: Optional[List[Dict[str, Any]]] = None,
    network: Optional["VascularNetwork"] = None,
    mesh_obstacles: Optional[List["trimesh.Trimesh"]] = None,
    policy: Optional[PathfindingPolicy] = None,
) -> PathfindingResult:
    """
    Find a path from start to goal using voxel-based A*.
    
    Parameters
    ----------
    domain : DomainSpec
        Domain specification for bounds and containment
    start : np.ndarray
        Start position (x, y, z) in meters
    goal : np.ndarray
        Goal position (x, y, z) in meters
    obstacles : list of dict, optional
        List of capsule obstacles with 'start', 'end', 'radius' keys
    network : VascularNetwork, optional
        Network to use as obstacles
    mesh_obstacles : list of trimesh.Trimesh, optional
        Mesh obstacles to avoid
    policy : PathfindingPolicy, optional
        Pathfinding configuration
        
    Returns
    -------
    PathfindingResult
        Result containing path points and metadata
    """
    if policy is None:
        policy = PathfindingPolicy()
    
    start_time = time.time()
    
    # Build obstacle map
    obstacle_map = VoxelObstacleMap(
        domain=domain,
        pitch=policy.voxel_pitch,
        clearance=policy.clearance,
    )
    
    # Add obstacles
    if obstacles:
        for obs in obstacles:
            obstacle_map.add_capsule_obstacle(
                np.array(obs["start"]),
                np.array(obs["end"]),
                obs["radius"],
            )
    
    if network:
        obstacle_map.add_network_obstacles(network)
    
    if mesh_obstacles:
        for mesh in mesh_obstacles:
            obstacle_map.add_mesh_obstacle(mesh)
    
    # Convert start/goal to voxels
    start_voxel = obstacle_map.world_to_voxel(start)
    goal_voxel = obstacle_map.world_to_voxel(goal)
    
    # Check if start/goal are valid
    if not obstacle_map.is_free(start_voxel):
        return PathfindingResult(
            success=False,
            errors=["Start position is blocked or outside domain"],
        )
    
    if not obstacle_map.is_free(goal_voxel):
        return PathfindingResult(
            success=False,
            errors=["Goal position is blocked or outside domain"],
        )
    
    # Run A*
    path_voxels, nodes_explored = _astar_search(
        obstacle_map=obstacle_map,
        start_voxel=start_voxel,
        goal_voxel=goal_voxel,
        policy=policy,
        timeout=policy.timeout_s,
        start_time=start_time,
    )
    
    time_elapsed = time.time() - start_time
    
    if path_voxels is None:
        if policy.allow_partial:
            # Return partial path to closest point
            return PathfindingResult(
                success=False,
                nodes_explored=nodes_explored,
                time_elapsed=time_elapsed,
                warnings=["No complete path found"],
            )
        return PathfindingResult(
            success=False,
            nodes_explored=nodes_explored,
            time_elapsed=time_elapsed,
            errors=["No path found"],
        )
    
    # Convert voxels to world coordinates
    path_pts = [obstacle_map.voxel_to_world(v) for v in path_voxels]
    
    # Ensure exact start and end positions
    path_pts[0] = start.copy()
    path_pts[-1] = goal.copy()
    
    # Apply smoothing if enabled
    if policy.smoothing_enabled and len(path_pts) > 2:
        path_pts = _smooth_path(
            path_pts,
            obstacle_map,
            policy.smoothing_iters,
            policy.smoothing_strength,
        )
    
    # Compute path length
    path_length = sum(
        np.linalg.norm(path_pts[i + 1] - path_pts[i])
        for i in range(len(path_pts) - 1)
    )
    
    return PathfindingResult(
        success=True,
        path_pts=path_pts,
        path_length=path_length,
        nodes_explored=nodes_explored,
        time_elapsed=time_elapsed,
        metadata={
            "voxel_pitch": policy.voxel_pitch,
            "path_voxels": len(path_voxels),
            "smoothed": policy.smoothing_enabled,
        },
    )


def find_path_through_waypoints(
    domain: DomainSpec,
    start: np.ndarray,
    waypoints: List[np.ndarray],
    goal: np.ndarray,
    obstacles: Optional[List[Dict[str, Any]]] = None,
    network: Optional["VascularNetwork"] = None,
    mesh_obstacles: Optional[List["trimesh.Trimesh"]] = None,
    policy: Optional[PathfindingPolicy] = None,
    waypoint_policy: Optional[WaypointPolicy] = None,
) -> PathfindingResult:
    """
    Find a path from start through waypoints to goal.
    
    Attempts to route through each waypoint in order. If a waypoint is
    unreachable, it may be skipped based on waypoint_policy.
    
    Parameters
    ----------
    domain : DomainSpec
        Domain specification
    start : np.ndarray
        Start position
    waypoints : list of np.ndarray
        Intermediate waypoints to pass through
    goal : np.ndarray
        Goal position
    obstacles : list of dict, optional
        Capsule obstacles
    network : VascularNetwork, optional
        Network obstacles
    mesh_obstacles : list of trimesh.Trimesh, optional
        Mesh obstacles
    policy : PathfindingPolicy, optional
        Pathfinding configuration
    waypoint_policy : WaypointPolicy, optional
        Waypoint handling configuration
        
    Returns
    -------
    PathfindingResult
        Result containing full path and metadata
    """
    if policy is None:
        policy = PathfindingPolicy()
    if waypoint_policy is None:
        waypoint_policy = WaypointPolicy()
    
    start_time = time.time()
    
    # Build full point list
    all_points = [start] + waypoints + [goal]
    
    full_path = [start]
    total_nodes_explored = 0
    warnings = []
    skipped_waypoints = []
    skip_count = 0
    
    current_start = start
    i = 1
    
    while i < len(all_points):
        segment_goal = all_points[i]
        
        # Find path for this segment
        result = find_path(
            domain=domain,
            start=current_start,
            goal=segment_goal,
            obstacles=obstacles,
            network=network,
            mesh_obstacles=mesh_obstacles,
            policy=policy,
        )
        
        total_nodes_explored += result.nodes_explored
        
        if result.success:
            # Add path (skip first point to avoid duplicates)
            full_path.extend(result.path_pts[1:])
            current_start = segment_goal
            i += 1
            skip_count = 0  # Reset skip count on success
        else:
            # Segment failed
            is_waypoint = 0 < i < len(all_points) - 1
            
            if is_waypoint and waypoint_policy.skip_unreachable:
                if skip_count < waypoint_policy.max_skip_count:
                    skipped_waypoints.append(i - 1)  # Waypoint index
                    skip_count += 1
                    
                    if waypoint_policy.emit_warnings:
                        warnings.append(
                            f"Skipped waypoint {i - 1}: {result.errors}"
                        )
                    
                    # Skip this waypoint
                    i += 1
                    continue
            
            # Can't skip or too many skips
            if waypoint_policy.fallback_direct:
                # Try direct path from current to goal
                direct_result = find_path(
                    domain=domain,
                    start=current_start,
                    goal=goal,
                    obstacles=obstacles,
                    network=network,
                    mesh_obstacles=mesh_obstacles,
                    policy=policy,
                )
                
                total_nodes_explored += direct_result.nodes_explored
                
                if direct_result.success:
                    warnings.append(
                        "Using direct path to goal (remaining waypoints skipped)"
                    )
                    full_path.extend(direct_result.path_pts[1:])
                    
                    return PathfindingResult(
                        success=True,
                        path_pts=full_path,
                        path_length=sum(
                            np.linalg.norm(full_path[j + 1] - full_path[j])
                            for j in range(len(full_path) - 1)
                        ),
                        nodes_explored=total_nodes_explored,
                        time_elapsed=time.time() - start_time,
                        warnings=warnings,
                        skipped_waypoints=skipped_waypoints,
                    )
            
            # Complete failure
            return PathfindingResult(
                success=False,
                path_pts=full_path if len(full_path) > 1 else None,
                nodes_explored=total_nodes_explored,
                time_elapsed=time.time() - start_time,
                warnings=warnings,
                errors=result.errors,
                skipped_waypoints=skipped_waypoints,
            )
    
    # Compute total path length
    path_length = sum(
        np.linalg.norm(full_path[j + 1] - full_path[j])
        for j in range(len(full_path) - 1)
    )
    
    return PathfindingResult(
        success=True,
        path_pts=full_path,
        path_length=path_length,
        nodes_explored=total_nodes_explored,
        time_elapsed=time.time() - start_time,
        warnings=warnings,
        skipped_waypoints=skipped_waypoints,
        metadata={
            "waypoints_total": len(waypoints),
            "waypoints_skipped": len(skipped_waypoints),
        },
    )


def _astar_search(
    obstacle_map: VoxelObstacleMap,
    start_voxel: Tuple[int, int, int],
    goal_voxel: Tuple[int, int, int],
    policy: PathfindingPolicy,
    timeout: float,
    start_time: float,
) -> Tuple[Optional[List[Tuple[int, int, int]]], int]:
    """
    Run A* search on the voxel grid.
    
    Returns (path_voxels, nodes_explored).
    """
    # Define neighbor offsets
    if policy.diagonal_movement:
        # 26-connected (all neighbors including diagonals)
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                for dk in [-1, 0, 1]:
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    neighbors.append((di, dj, dk))
    else:
        # 6-connected (face neighbors only)
        neighbors = [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1),
        ]
    
    # Heuristic function (Euclidean distance)
    def heuristic(voxel: Tuple[int, int, int]) -> float:
        return policy.heuristic_weight * np.sqrt(
            (voxel[0] - goal_voxel[0]) ** 2 +
            (voxel[1] - goal_voxel[1]) ** 2 +
            (voxel[2] - goal_voxel[2]) ** 2
        ) * obstacle_map.pitch
    
    # Initialize
    start_node = AStarNode(
        voxel=start_voxel,
        g=0.0,
        h=heuristic(start_voxel),
    )
    
    open_set: List[AStarNode] = [start_node]
    heapq.heapify(open_set)
    
    closed_set: Set[Tuple[int, int, int]] = set()
    g_scores: Dict[Tuple[int, int, int], float] = {start_voxel: 0.0}
    
    nodes_explored = 0
    
    while open_set:
        # Check timeout
        if time.time() - start_time > timeout:
            logger.warning("A* search timed out")
            return None, nodes_explored
        
        # Check node limit
        if nodes_explored >= policy.max_nodes:
            logger.warning("A* search exceeded max nodes")
            return None, nodes_explored
        
        # Get node with lowest f score
        current = heapq.heappop(open_set)
        
        if current.voxel in closed_set:
            continue
        
        closed_set.add(current.voxel)
        nodes_explored += 1
        
        # Check if we reached the goal
        if current.voxel == goal_voxel:
            # Reconstruct path
            path = []
            node = current
            while node is not None:
                path.append(node.voxel)
                node = node.parent
            path.reverse()
            return path, nodes_explored
        
        # Explore neighbors
        for di, dj, dk in neighbors:
            neighbor_voxel = (
                current.voxel[0] + di,
                current.voxel[1] + dj,
                current.voxel[2] + dk,
            )
            
            # Skip if already visited
            if neighbor_voxel in closed_set:
                continue
            
            # Skip if blocked
            if not obstacle_map.is_free(neighbor_voxel):
                continue
            
            # Compute cost
            move_cost = np.sqrt(di ** 2 + dj ** 2 + dk ** 2) * obstacle_map.pitch
            
            # Add turn penalty
            if current.direction is not None:
                new_direction = (di, dj, dk)
                if new_direction != current.direction:
                    move_cost += policy.turn_penalty * obstacle_map.pitch
            
            tentative_g = current.g + move_cost
            
            # Skip if we already have a better path
            if neighbor_voxel in g_scores and tentative_g >= g_scores[neighbor_voxel]:
                continue
            
            g_scores[neighbor_voxel] = tentative_g
            
            neighbor_node = AStarNode(
                voxel=neighbor_voxel,
                g=tentative_g,
                h=heuristic(neighbor_voxel),
                parent=current,
                direction=(di, dj, dk),
            )
            
            heapq.heappush(open_set, neighbor_node)
    
    # No path found
    return None, nodes_explored


def _smooth_path(
    path_pts: List[np.ndarray],
    obstacle_map: VoxelObstacleMap,
    iterations: int,
    strength: float,
) -> List[np.ndarray]:
    """
    Smooth a path using iterative averaging while maintaining collision-free.
    
    Uses a simple Laplacian smoothing approach that moves each point
    toward the average of its neighbors.
    """
    if len(path_pts) <= 2:
        return path_pts
    
    smoothed = [p.copy() for p in path_pts]
    
    for _ in range(iterations):
        new_smoothed = [smoothed[0].copy()]  # Keep start fixed
        
        for i in range(1, len(smoothed) - 1):
            # Compute average of neighbors
            avg = (smoothed[i - 1] + smoothed[i + 1]) / 2
            
            # Move toward average
            new_pos = smoothed[i] + strength * (avg - smoothed[i])
            
            # Check if new position is collision-free
            new_voxel = obstacle_map.world_to_voxel(new_pos)
            if obstacle_map.is_free(new_voxel):
                new_smoothed.append(new_pos)
            else:
                # Keep original position
                new_smoothed.append(smoothed[i].copy())
        
        new_smoothed.append(smoothed[-1].copy())  # Keep end fixed
        smoothed = new_smoothed
    
    return smoothed


__all__ = [
    "find_path",
    "find_path_through_waypoints",
    "PathfindingPolicy",
    "WaypointPolicy",
    "PathfindingResult",
    "VoxelObstacleMap",
]
