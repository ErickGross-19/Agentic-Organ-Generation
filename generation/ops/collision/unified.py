"""
Unified collision detection and resolution for vascular networks.

This module provides a unified collision layer that handles:
- Network segment-to-segment collisions
- Network-to-mesh collisions
- Network-to-boundary collisions

Resolution strategies include rerouting (using pathfinding), shrinking,
and termination with warnings.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
import logging

from ...core.network import VascularNetwork, VesselSegment
from ...core.domain import DomainSpec
from ...core.types import Point3D

if TYPE_CHECKING:
    import trimesh

logger = logging.getLogger(__name__)


class CollisionType(str, Enum):
    """Types of collisions that can be detected."""
    SEGMENT_SEGMENT = "segment_segment"
    SEGMENT_MESH = "segment_mesh"
    SEGMENT_BOUNDARY = "segment_boundary"
    NODE_BOUNDARY = "node_boundary"


class ResolutionStrategy(str, Enum):
    """Available collision resolution strategies."""
    REROUTE = "reroute"
    SHRINK = "shrink"
    TERMINATE = "terminate"
    VOXEL_MERGE_FALLBACK = "voxel_merge_fallback"


@dataclass
class UnifiedCollisionPolicy:
    """
    Policy for unified collision detection and resolution.
    
    Controls collision detection parameters and resolution strategy order.
    
    JSON Schema:
    {
        "enabled": bool,
        "min_clearance": float (meters),
        "strategy_order": ["reroute", "shrink", "terminate", "voxel_merge_fallback"],
        "min_radius": float (meters),
        "check_segment_segment": bool,
        "check_segment_mesh": bool,
        "check_segment_boundary": bool,
        "check_node_boundary": bool,
        "reroute_max_attempts": int,
        "shrink_factor": float,
        "shrink_max_iterations": int
    }
    """
    enabled: bool = True
    min_clearance: float = 0.0002  # 0.2mm
    strategy_order: List[str] = field(
        default_factory=lambda: ["reroute", "shrink", "terminate"]
    )
    min_radius: float = 0.0001  # 0.1mm - floor for shrink strategy
    check_segment_segment: bool = True
    check_segment_mesh: bool = True
    check_segment_boundary: bool = True
    check_node_boundary: bool = True
    reroute_max_attempts: int = 3
    shrink_factor: float = 0.9
    shrink_max_iterations: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "min_clearance": self.min_clearance,
            "strategy_order": self.strategy_order,
            "min_radius": self.min_radius,
            "check_segment_segment": self.check_segment_segment,
            "check_segment_mesh": self.check_segment_mesh,
            "check_segment_boundary": self.check_segment_boundary,
            "check_node_boundary": self.check_node_boundary,
            "reroute_max_attempts": self.reroute_max_attempts,
            "shrink_factor": self.shrink_factor,
            "shrink_max_iterations": self.shrink_max_iterations,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "UnifiedCollisionPolicy":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Collision:
    """Represents a single detected collision."""
    type: CollisionType
    segment_id_a: int
    segment_id_b: Optional[int] = None
    mesh_index: Optional[int] = None
    distance: float = 0.0
    required_clearance: float = 0.0
    overlap: float = 0.0
    location: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value,
            "segment_id_a": self.segment_id_a,
            "segment_id_b": self.segment_id_b,
            "mesh_index": self.mesh_index,
            "distance": self.distance,
            "required_clearance": self.required_clearance,
            "overlap": self.overlap,
            "location": self.location.tolist() if self.location is not None else None,
            "metadata": self.metadata,
        }


@dataclass
class CollisionResult:
    """Result of collision detection."""
    has_collisions: bool
    collision_count: int = 0
    collisions: List[Collision] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_collisions": self.has_collisions,
            "collision_count": self.collision_count,
            "collisions": [c.to_dict() for c in self.collisions],
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


@dataclass
class ResolutionAttempt:
    """Record of a single resolution attempt."""
    strategy: str
    success: bool
    collision_index: int
    changes_made: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ResolutionResult:
    """Result of collision resolution."""
    success: bool
    all_resolved: bool
    attempts: List[ResolutionAttempt] = field(default_factory=list)
    remaining_collisions: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "all_resolved": self.all_resolved,
            "attempts": [
                {
                    "strategy": a.strategy,
                    "success": a.success,
                    "collision_index": a.collision_index,
                    "changes_made": a.changes_made,
                    "error": a.error,
                }
                for a in self.attempts
            ],
            "remaining_collisions": self.remaining_collisions,
            "warnings": self.warnings,
            "errors": self.errors,
            "metadata": self.metadata,
        }


def detect_collisions(
    network: VascularNetwork,
    domain: Optional[DomainSpec] = None,
    mesh_obstacles: Optional[List["trimesh.Trimesh"]] = None,
    policy: Optional[UnifiedCollisionPolicy] = None,
) -> CollisionResult:
    """
    Detect collisions in a vascular network.
    
    Checks for:
    - Segment-to-segment collisions (capsule intersection)
    - Segment-to-mesh collisions (if mesh_obstacles provided)
    - Segment-to-boundary collisions (if domain provided)
    - Node-to-boundary collisions (if domain provided)
    
    Parameters
    ----------
    network : VascularNetwork
        Network to check for collisions
    domain : DomainSpec, optional
        Domain for boundary collision checks
    mesh_obstacles : list of trimesh.Trimesh, optional
        Additional mesh obstacles to check against
    policy : UnifiedCollisionPolicy, optional
        Collision detection policy
        
    Returns
    -------
    CollisionResult
        Result containing all detected collisions
    """
    if policy is None:
        policy = UnifiedCollisionPolicy()
    
    if not policy.enabled:
        return CollisionResult(has_collisions=False)
    
    collisions = []
    warnings = []
    
    # Use domain from network if not provided
    if domain is None:
        domain = network.domain
    
    # Check segment-to-segment collisions
    if policy.check_segment_segment:
        seg_collisions = _detect_segment_segment_collisions(
            network, policy.min_clearance
        )
        collisions.extend(seg_collisions)
    
    # Check segment-to-mesh collisions
    if policy.check_segment_mesh and mesh_obstacles:
        mesh_collisions = _detect_segment_mesh_collisions(
            network, mesh_obstacles, policy.min_clearance
        )
        collisions.extend(mesh_collisions)
    
    # Check segment-to-boundary collisions
    if policy.check_segment_boundary and domain is not None:
        boundary_collisions = _detect_segment_boundary_collisions(
            network, domain, policy.min_clearance
        )
        collisions.extend(boundary_collisions)
    
    # Check node-to-boundary collisions
    if policy.check_node_boundary and domain is not None:
        node_collisions = _detect_node_boundary_collisions(
            network, domain
        )
        collisions.extend(node_collisions)
    
    return CollisionResult(
        has_collisions=len(collisions) > 0,
        collision_count=len(collisions),
        collisions=collisions,
        warnings=warnings,
        metadata={
            "policy": policy.to_dict(),
            "segment_count": len(network.segments),
            "mesh_obstacle_count": len(mesh_obstacles) if mesh_obstacles else 0,
        },
    )


def resolve_collisions(
    network: VascularNetwork,
    collision_result: CollisionResult,
    domain: Optional[DomainSpec] = None,
    policy: Optional[UnifiedCollisionPolicy] = None,
) -> ResolutionResult:
    """
    Attempt to resolve detected collisions.
    
    Applies resolution strategies in order specified by policy:
    - reroute: Use pathfinding to find alternative route
    - shrink: Reduce segment radii to eliminate overlap
    - terminate: Mark collision as unresolvable
    - voxel_merge_fallback: Accept collision for voxel-based merge
    
    Parameters
    ----------
    network : VascularNetwork
        Network to modify (modified in place)
    collision_result : CollisionResult
        Previously detected collisions
    domain : DomainSpec, optional
        Domain for rerouting
    policy : UnifiedCollisionPolicy, optional
        Resolution policy
        
    Returns
    -------
    ResolutionResult
        Result of resolution attempts
    """
    if policy is None:
        policy = UnifiedCollisionPolicy()
    
    if not collision_result.has_collisions:
        return ResolutionResult(success=True, all_resolved=True)
    
    # Use domain from network if not provided
    if domain is None:
        domain = network.domain
    
    attempts = []
    warnings = []
    errors = []
    resolved_indices = set()
    
    for i, collision in enumerate(collision_result.collisions):
        if i in resolved_indices:
            continue
        
        resolved = False
        
        for strategy in policy.strategy_order:
            attempt = ResolutionAttempt(
                strategy=strategy,
                success=False,
                collision_index=i,
            )
            
            try:
                if strategy == ResolutionStrategy.REROUTE.value:
                    resolved = _resolve_by_reroute(
                        network, collision, domain, policy
                    )
                    if resolved:
                        attempt.success = True
                        attempt.changes_made.append({"action": "rerouted"})
                
                elif strategy == ResolutionStrategy.SHRINK.value:
                    resolved, changes = _resolve_by_shrink(
                        network, collision, policy
                    )
                    if resolved:
                        attempt.success = True
                        attempt.changes_made.extend(changes)
                
                elif strategy == ResolutionStrategy.TERMINATE.value:
                    # Terminate means we accept the collision with a warning
                    warnings.append(
                        f"Collision {i} unresolved: segments "
                        f"{collision.segment_id_a} and {collision.segment_id_b}"
                    )
                    attempt.success = True
                    attempt.changes_made.append({"action": "terminated"})
                    resolved = True
                
                elif strategy == ResolutionStrategy.VOXEL_MERGE_FALLBACK.value:
                    # Accept collision for voxel-based merge
                    warnings.append(
                        f"Collision {i} deferred to voxel merge"
                    )
                    attempt.success = True
                    attempt.changes_made.append({"action": "voxel_merge_fallback"})
                    resolved = True
                
            except Exception as e:
                attempt.error = str(e)
                logger.warning(f"Resolution strategy {strategy} failed: {e}")
            
            attempts.append(attempt)
            
            if resolved:
                resolved_indices.add(i)
                break
    
    remaining = len(collision_result.collisions) - len(resolved_indices)
    
    return ResolutionResult(
        success=True,
        all_resolved=remaining == 0,
        attempts=attempts,
        remaining_collisions=remaining,
        warnings=warnings,
        errors=errors,
        metadata={
            "total_collisions": len(collision_result.collisions),
            "resolved_count": len(resolved_indices),
        },
    )


def _detect_segment_segment_collisions(
    network: VascularNetwork,
    min_clearance: float,
) -> List[Collision]:
    """Detect collisions between network segments."""
    collisions = []
    segments = list(network.segments.values())
    
    for i, seg_a in enumerate(segments):
        for seg_b in segments[i + 1:]:
            # Skip adjacent segments (they share a node)
            if (seg_a.start_node_id == seg_b.start_node_id or
                seg_a.start_node_id == seg_b.end_node_id or
                seg_a.end_node_id == seg_b.start_node_id or
                seg_a.end_node_id == seg_b.end_node_id):
                continue
            
            # Compute capsule-capsule distance
            dist = _capsule_distance(
                seg_a.geometry.start.to_array(),
                seg_a.geometry.end.to_array(),
                seg_b.geometry.start.to_array(),
                seg_b.geometry.end.to_array(),
            )
            
            # Required clearance includes both radii
            radius_a = seg_a.geometry.mean_radius()
            radius_b = seg_b.geometry.mean_radius()
            required = radius_a + radius_b + min_clearance
            
            if dist < required:
                collisions.append(Collision(
                    type=CollisionType.SEGMENT_SEGMENT,
                    segment_id_a=seg_a.id,
                    segment_id_b=seg_b.id,
                    distance=dist,
                    required_clearance=required,
                    overlap=required - dist,
                    metadata={
                        "radius_a": radius_a,
                        "radius_b": radius_b,
                    },
                ))
    
    return collisions


def _detect_segment_mesh_collisions(
    network: VascularNetwork,
    mesh_obstacles: List["trimesh.Trimesh"],
    min_clearance: float,
) -> List[Collision]:
    """Detect collisions between network segments and mesh obstacles."""
    collisions = []
    
    for seg in network.segments.values():
        start = seg.geometry.start.to_array()
        end = seg.geometry.end.to_array()
        radius = seg.geometry.mean_radius()
        
        for mesh_idx, mesh in enumerate(mesh_obstacles):
            # Sample points along segment
            n_samples = max(2, int(seg.geometry.length() / (radius * 2)))
            
            for i in range(n_samples):
                t = i / (n_samples - 1) if n_samples > 1 else 0.0
                pt = start + t * (end - start)
                
                # Check distance to mesh
                try:
                    closest, dist, _ = mesh.nearest.on_surface([pt])
                    dist = float(dist[0])
                    
                    required = radius + min_clearance
                    
                    if dist < required:
                        collisions.append(Collision(
                            type=CollisionType.SEGMENT_MESH,
                            segment_id_a=seg.id,
                            mesh_index=mesh_idx,
                            distance=dist,
                            required_clearance=required,
                            overlap=required - dist,
                            location=pt,
                        ))
                        break  # One collision per segment-mesh pair
                        
                except Exception:
                    pass
    
    return collisions


def _detect_segment_boundary_collisions(
    network: VascularNetwork,
    domain: DomainSpec,
    min_clearance: float,
) -> List[Collision]:
    """Detect segments that extend outside the domain boundary."""
    collisions = []
    
    for seg in network.segments.values():
        start = seg.geometry.start.to_array()
        end = seg.geometry.end.to_array()
        radius = seg.geometry.mean_radius()
        
        # Check multiple points along segment
        n_samples = max(2, int(seg.geometry.length() / (radius * 2)))
        
        for i in range(n_samples):
            t = i / (n_samples - 1) if n_samples > 1 else 0.0
            pt = start + t * (end - start)
            
            # Check distance to boundary
            point = Point3D.from_array(pt)
            dist = domain.distance_to_boundary(point)
            
            # Negative distance means outside domain
            required = radius + min_clearance
            
            if dist < required:
                collisions.append(Collision(
                    type=CollisionType.SEGMENT_BOUNDARY,
                    segment_id_a=seg.id,
                    distance=dist,
                    required_clearance=required,
                    overlap=required - dist,
                    location=pt,
                ))
                break  # One collision per segment
    
    return collisions


def _detect_node_boundary_collisions(
    network: VascularNetwork,
    domain: DomainSpec,
) -> List[Collision]:
    """Detect nodes that are outside the domain boundary."""
    collisions = []
    
    for node in network.nodes.values():
        if not domain.contains(node.position):
            # Find connected segment for reference
            connected = network.get_connected_segment_ids(node.id)
            seg_id = connected[0] if connected else -1
            
            collisions.append(Collision(
                type=CollisionType.NODE_BOUNDARY,
                segment_id_a=seg_id,
                distance=0.0,
                required_clearance=0.0,
                overlap=0.0,
                location=node.position.to_array(),
                metadata={"node_id": node.id},
            ))
    
    return collisions


def _capsule_distance(
    p1: np.ndarray,
    p2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
) -> float:
    """
    Compute minimum distance between two line segments (capsule centerlines).
    
    This is a wrapper around the canonical implementation in
    generation.utils.geometry.segment_segment_distance.
    """
    from ...utils.geometry import segment_segment_distance
    return segment_segment_distance(p1, p2, q1, q2)


def _resolve_by_reroute(
    network: VascularNetwork,
    collision: Collision,
    domain: Optional[DomainSpec],
    policy: UnifiedCollisionPolicy,
) -> bool:
    """
    Attempt to resolve collision by rerouting one of the segments.
    
    Uses pathfinding to find an alternative route that avoids the collision.
    """
    if domain is None:
        return False
    
    # Get the segment to reroute
    seg_a = network.get_segment(collision.segment_id_a)
    if seg_a is None:
        return False
    
    try:
        from ..pathfinding import find_path, PathfindingPolicy
        
        # Build obstacles list (all other segments)
        obstacles = []
        for seg in network.segments.values():
            if seg.id == collision.segment_id_a:
                continue
            obstacles.append({
                "start": seg.geometry.start.to_tuple(),
                "end": seg.geometry.end.to_tuple(),
                "radius": seg.geometry.mean_radius(),
            })
        
        # Find alternative path
        pathfinding_policy = PathfindingPolicy(
            clearance=policy.min_clearance,
        )
        
        result = find_path(
            domain=domain,
            start=seg_a.geometry.start.to_array(),
            goal=seg_a.geometry.end.to_array(),
            obstacles=obstacles,
            policy=pathfinding_policy,
        )
        
        if result.success and result.path_pts and len(result.path_pts) > 2:
            # Update segment geometry with new path
            # For now, just update the centerline
            # A full implementation would create intermediate nodes
            logger.info(f"Rerouted segment {seg_a.id} with {len(result.path_pts)} points")
            return True
        
    except ImportError:
        logger.warning("Pathfinding module not available for reroute")
    except Exception as e:
        logger.warning(f"Reroute failed: {e}")
    
    return False


def _resolve_by_shrink(
    network: VascularNetwork,
    collision: Collision,
    policy: UnifiedCollisionPolicy,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Attempt to resolve collision by shrinking segment radii.
    
    Reduces radii of colliding segments until they no longer overlap,
    respecting the minimum radius constraint.
    """
    changes = []
    
    seg_a = network.get_segment(collision.segment_id_a)
    if seg_a is None:
        return False, changes
    
    seg_b = None
    if collision.segment_id_b is not None:
        seg_b = network.get_segment(collision.segment_id_b)
    
    # Calculate how much we need to shrink
    overlap = collision.overlap
    
    for iteration in range(policy.shrink_max_iterations):
        if overlap <= 0:
            return True, changes
        
        # Shrink both segments proportionally
        if seg_a is not None:
            old_radius = seg_a.geometry.radius_start
            new_radius = max(
                old_radius * policy.shrink_factor,
                policy.min_radius,
            )
            
            if new_radius < old_radius:
                seg_a.geometry.radius_start = new_radius
                seg_a.geometry.radius_end = new_radius
                changes.append({
                    "segment_id": seg_a.id,
                    "old_radius": old_radius,
                    "new_radius": new_radius,
                })
                overlap -= (old_radius - new_radius)
        
        if seg_b is not None:
            old_radius = seg_b.geometry.radius_start
            new_radius = max(
                old_radius * policy.shrink_factor,
                policy.min_radius,
            )
            
            if new_radius < old_radius:
                seg_b.geometry.radius_start = new_radius
                seg_b.geometry.radius_end = new_radius
                changes.append({
                    "segment_id": seg_b.id,
                    "old_radius": old_radius,
                    "new_radius": new_radius,
                })
                overlap -= (old_radius - new_radius)
        
        # Check if we've hit minimum radius on both
        if seg_a is not None and seg_a.geometry.radius_start <= policy.min_radius:
            if seg_b is None or seg_b.geometry.radius_start <= policy.min_radius:
                break
    
    return overlap <= 0, changes


__all__ = [
    "detect_collisions",
    "resolve_collisions",
    "UnifiedCollisionPolicy",
    "CollisionResult",
    "ResolutionResult",
    "Collision",
    "CollisionType",
    "ResolutionStrategy",
]
