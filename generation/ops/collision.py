"""
Collision detection and avoidance operations.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from ..core.network import VascularNetwork, Node, VesselSegment
from ..core.types import Point3D, Direction3D, TubeGeometry
from ..core.result import OperationResult, OperationStatus, ErrorCode, Delta


@dataclass
class RepairReport:
    """Report from collision repair attempt."""
    segment_id: int
    strategy_used: str
    attempts: int
    success: bool
    final_direction: Optional[Direction3D] = None
    residual_clearance: Optional[float] = None
    error_message: Optional[str] = None


def get_collisions(
    network: VascularNetwork,
    min_clearance: float = 0.001,
    exclude_connected: bool = True,
) -> OperationResult:
    """
    Find all segment pairs that are too close.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to check
    min_clearance : float
        Minimum required clearance between segments (meters)
    exclude_connected : bool
        If True, exclude segments that share a node
    
    Returns
    -------
    result : OperationResult
        Result with metadata['collisions'] containing list of collision tuples
    """
    spatial_index = network.get_spatial_index()
    
    collisions = spatial_index.get_collisions(
        min_clearance=min_clearance,
        exclude_connected=exclude_connected,
    )
    
    if collisions:
        return OperationResult(
            status=OperationStatus.WARNING,
            message=f"Found {len(collisions)} collisions",
            warnings=[f"Segments {c[0]} and {c[1]} too close (distance: {c[2]:.4f}m)" for c in collisions[:5]],
            metadata={"collisions": collisions, "count": len(collisions)},
        )
    else:
        return OperationResult.success(
            message="No collisions found",
            metadata={"collisions": [], "count": 0},
        )


def avoid_collisions(
    network: VascularNetwork,
    min_clearance: float = 0.001,
    repair_strategy: str = "report",
) -> OperationResult:
    """
    Detect and optionally repair collisions.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to check/repair
    min_clearance : float
        Minimum required clearance
    repair_strategy : str
        Strategy for handling collisions:
        - "report": Just report collisions
        - "reroute": Attempt to reroute branches
        - "shrink": Reduce radius/length
        - "terminate": Mark colliding branches as terminated
    
    Returns
    -------
    result : OperationResult
        Result with collision information
    """
    if repair_strategy == "report":
        return get_collisions(network, min_clearance=min_clearance)
    
    # Get initial collisions
    collision_result = get_collisions(network, min_clearance=min_clearance)
    collisions = collision_result.metadata.get('collisions', [])
    
    if not collisions:
        return OperationResult.success("No collisions to repair")
    
    if repair_strategy == "reroute":
        return _repair_by_reroute(network, collisions, min_clearance)
    elif repair_strategy == "shrink":
        return _repair_by_shrink(network, collisions, min_clearance)
    elif repair_strategy == "terminate":
        return _repair_by_terminate(network, collisions)
    else:
        return OperationResult.failure(
            f"Unknown repair strategy: {repair_strategy}",
            error_codes=[ErrorCode.INVALID_PARAMETER.value],
        )


def _repair_by_reroute(
    network: VascularNetwork,
    collisions: List[Tuple],
    min_clearance: float,
    max_attempts: int = 10,
    cone_angle: float = 45.0,
    angular_step: float = 5.0,
) -> OperationResult:
    """
    Repair collisions by rerouting segments.
    
    Strategy: Sample directions in a cone around the desired direction,
    check clearance, pick best direction.
    """
    reports = []
    delta = Delta()
    
    # Get segments involved in collisions
    collision_segments = set()
    for seg_id_a, seg_id_b, distance in collisions:
        collision_segments.add(seg_id_a)
        collision_segments.add(seg_id_b)
    
    for seg_id in collision_segments:
        segment = network.segments.get(seg_id)
        if not segment:
            continue
        
        original_direction = _compute_direction(segment)
        best_direction = None
        best_clearance = 0.0
        
        for attempt in range(max_attempts):
            candidate_direction = _sample_cone_direction(
                original_direction,
                cone_angle,
                angular_step,
                attempt,
            )
            
            start_pos = np.array([
                segment.geometry.start.x,
                segment.geometry.start.y,
                segment.geometry.start.z,
            ])
            length = np.linalg.norm(np.array([
                segment.geometry.end.x - segment.geometry.start.x,
                segment.geometry.end.y - segment.geometry.start.y,
                segment.geometry.end.z - segment.geometry.start.z,
            ]))
            
            new_end = start_pos + candidate_direction.to_array() * length
            
            if not network.domain.contains(Point3D(x=new_end[0], y=new_end[1], z=new_end[2])):
                continue
            
            # Check clearance
            clearance = _check_segment_clearance(
                network, seg_id, start_pos, new_end, segment.geometry.radius_start
            )
            
            if clearance >= min_clearance:
                best_direction = candidate_direction
                best_clearance = clearance
                break
            elif clearance > best_clearance:
                best_direction = candidate_direction
                best_clearance = clearance
        
        if best_direction and best_clearance >= min_clearance:
            start_pos = np.array([
                segment.geometry.start.x,
                segment.geometry.start.y,
                segment.geometry.start.z,
            ])
            length = np.linalg.norm(np.array([
                segment.geometry.end.x - segment.geometry.start.x,
                segment.geometry.end.y - segment.geometry.start.y,
                segment.geometry.end.z - segment.geometry.start.z,
            ]))
            new_end = start_pos + best_direction.to_array() * length
            
            old_segment_dict = segment.to_dict()
            segment.geometry.end = Point3D(x=new_end[0], y=new_end[1], z=new_end[2])
            
            delta.modified_segments[seg_id] = old_segment_dict
            
            reports.append(RepairReport(
                segment_id=seg_id,
                strategy_used="reroute",
                attempts=max_attempts,
                success=True,
                final_direction=best_direction,
                residual_clearance=best_clearance,
            ))
        else:
            reports.append(RepairReport(
                segment_id=seg_id,
                strategy_used="reroute",
                attempts=max_attempts,
                success=False,
                residual_clearance=best_clearance,
                error_message=f"Could not find direction with clearance >= {min_clearance}",
            ))
    
    # Check final collisions
    final_collision_result = get_collisions(network, min_clearance=min_clearance)
    final_collisions = final_collision_result.metadata.get('collisions', [])
    
    success_count = sum(1 for r in reports if r.success)
    
    if not final_collisions:
        result = OperationResult.success(
            f"Reroute successful: {success_count}/{len(reports)} segments rerouted",
            delta=delta,
            metadata={
                'reports': [r.__dict__ for r in reports],
                'initial_collisions': len(collisions),
                'final_collisions': 0,
            },
        )
    else:
        result = OperationResult.partial_success(
            f"Reroute partial: {success_count}/{len(reports)} segments rerouted, {len(final_collisions)} collisions remain",
            delta=delta,
            metadata={
                'reports': [r.__dict__ for r in reports],
                'initial_collisions': len(collisions),
                'final_collisions': len(final_collisions),
            },
        )
        result.error_codes.append(ErrorCode.REROUTE_FAILED.value)
    
    return result


def _repair_by_shrink(
    network: VascularNetwork,
    collisions: List[Tuple],
    min_clearance: float,
    shrink_factor: float = 0.9,
    min_radius: float = 0.0001,
) -> OperationResult:
    """
    Repair collisions by shrinking radius and/or length.
    """
    reports = []
    delta = Delta()
    
    collision_segments = set()
    for seg_id_a, seg_id_b, distance in collisions:
        collision_segments.add(seg_id_a)
        collision_segments.add(seg_id_b)
    
    for seg_id in collision_segments:
        segment = network.segments.get(seg_id)
        if not segment:
            continue
        
        old_segment_dict = segment.to_dict()
        
        new_radius_start = segment.geometry.radius_start * shrink_factor
        new_radius_end = segment.geometry.radius_end * shrink_factor
        
        if new_radius_start >= min_radius and new_radius_end >= min_radius:
            segment.geometry.radius_start = new_radius_start
            segment.geometry.radius_end = new_radius_end
            
            delta.modified_segments[seg_id] = old_segment_dict
            
            reports.append(RepairReport(
                segment_id=seg_id,
                strategy_used="shrink",
                attempts=1,
                success=True,
                residual_clearance=None,
            ))
        else:
            reports.append(RepairReport(
                segment_id=seg_id,
                strategy_used="shrink",
                attempts=1,
                success=False,
                error_message=f"Radius would be below minimum ({min_radius})",
            ))
    
    final_collision_result = get_collisions(network, min_clearance=min_clearance)
    final_collisions = final_collision_result.metadata.get('collisions', [])
    
    success_count = sum(1 for r in reports if r.success)
    
    if not final_collisions:
        result = OperationResult.success(
            f"Shrink successful: {success_count}/{len(reports)} segments shrunk",
            delta=delta,
            metadata={
                'reports': [r.__dict__ for r in reports],
                'initial_collisions': len(collisions),
                'final_collisions': 0,
            },
        )
    else:
        result = OperationResult.partial_success(
            f"Shrink partial: {success_count}/{len(reports)} segments shrunk, {len(final_collisions)} collisions remain",
            delta=delta,
            metadata={
                'reports': [r.__dict__ for r in reports],
                'initial_collisions': len(collisions),
                'final_collisions': len(final_collisions),
            },
        )
        result.error_codes.append(ErrorCode.SHRINK_FAILED.value)
    
    return result


def _repair_by_terminate(
    network: VascularNetwork,
    collisions: List[Tuple],
) -> OperationResult:
    """
    Repair collisions by marking end nodes as terminal.
    """
    reports = []
    delta = Delta()
    
    collision_segments = set()
    for seg_id_a, seg_id_b, distance in collisions:
        collision_segments.add(seg_id_a)
        collision_segments.add(seg_id_b)
    
    for seg_id in collision_segments:
        segment = network.segments.get(seg_id)
        if not segment:
            continue
        
        end_node = network.nodes.get(segment.end_node_id)
        if not end_node:
            continue
        
        old_node_dict = end_node.to_dict()
        
        if end_node.node_type != "terminal":
            end_node.node_type = "terminal"
            delta.modified_nodes[end_node.id] = old_node_dict
            
            reports.append(RepairReport(
                segment_id=seg_id,
                strategy_used="terminate",
                attempts=1,
                success=True,
            ))
    
    result = OperationResult.success(
        f"Terminate successful: {len(reports)} branches terminated",
        delta=delta,
        metadata={
            'reports': [r.__dict__ for r in reports],
            'terminated_count': len(reports),
        },
    )
    result.error_codes.append(ErrorCode.TERMINATED_DUE_TO_COLLISIONS.value)
    
    return result


def _compute_direction(segment: VesselSegment) -> Direction3D:
    """Compute direction vector from segment."""
    start = np.array([
        segment.geometry.start.x,
        segment.geometry.start.y,
        segment.geometry.start.z,
    ])
    end = np.array([
        segment.geometry.end.x,
        segment.geometry.end.y,
        segment.geometry.end.z,
    ])
    direction = end - start
    length = np.linalg.norm(direction)
    if length > 0:
        direction = direction / length
    return Direction3D(x=direction[0], y=direction[1], z=direction[2])


def _sample_cone_direction(
    base_direction: Direction3D,
    cone_angle: float,
    angular_step: float,
    attempt: int,
) -> Direction3D:
    """
    Sample a direction within a cone around base_direction.
    
    Uses systematic sampling with increasing angle.
    """
    base = base_direction.to_array()
    
    theta = min(cone_angle, angular_step * (attempt + 1)) * np.pi / 180.0
    
    phi = (attempt * 137.5) * np.pi / 180.0  # Golden angle for good coverage
    
    if abs(base[2]) < 0.9:
        perp1 = np.cross(base, np.array([0, 0, 1]))
    else:
        perp1 = np.cross(base, np.array([1, 0, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(base, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)
    
    direction = (
        base * np.cos(theta) +
        perp1 * np.sin(theta) * np.cos(phi) +
        perp2 * np.sin(theta) * np.sin(phi)
    )
    direction = direction / np.linalg.norm(direction)
    
    return Direction3D(x=direction[0], y=direction[1], z=direction[2])


def _check_segment_clearance(
    network: VascularNetwork,
    seg_id: int,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
) -> float:
    """
    Check minimum clearance of a segment to all other segments.
    
    Returns minimum distance to other segments.
    """
    min_clearance = float('inf')
    
    for other_id, other_seg in network.segments.items():
        if other_id == seg_id:
            continue
        
        segment = network.segments[seg_id]
        if (other_seg.start_node_id == segment.start_node_id or
            other_seg.start_node_id == segment.end_node_id or
            other_seg.end_node_id == segment.start_node_id or
            other_seg.end_node_id == segment.end_node_id):
            continue
        
        other_start = np.array([
            other_seg.geometry.start_point.x,
            other_seg.geometry.start_point.y,
            other_seg.geometry.start_point.z,
        ])
        other_end = np.array([
            other_seg.geometry.end_point.x,
            other_seg.geometry.end_point.y,
            other_seg.geometry.end_point.z,
        ])
        other_radius = (other_seg.geometry.radius_start + other_seg.geometry.radius_end) / 2
        
        distance = _segment_to_segment_distance(start, end, other_start, other_end)
        clearance = distance - radius - other_radius
        
        min_clearance = min(min_clearance, clearance)
    
    return min_clearance


def _segment_to_segment_distance(
    p1: np.ndarray,
    p2: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
) -> float:
    """
    Compute minimum distance between two line segments.
    """
    d1 = p2 - p1
    d2 = q2 - q1
    r = p1 - q1
    
    a = np.dot(d1, d1)
    b = np.dot(d1, d2)
    c = np.dot(d2, d2)
    d = np.dot(d1, r)
    e = np.dot(d2, r)
    
    denom = a * c - b * b
    
    if abs(denom) < 1e-10:
        s = 0.0
        t = d / a if abs(a) > 1e-10 else 0.0
    else:
        s = (b * d - a * e) / denom
        t = (c * d - b * e) / denom
    
    s = np.clip(s, 0, 1)
    t = np.clip(t, 0, 1)
    
    closest_p = p1 + t * d1
    closest_q = q1 + s * d2
    
    return np.linalg.norm(closest_p - closest_q)
