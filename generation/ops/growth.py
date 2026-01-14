"""
Growth operations for extending vascular networks.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np
from ..core.types import Point3D, Direction3D, TubeGeometry
from ..core.network import VascularNetwork, Node, VesselSegment
from ..core.result import OperationResult, OperationStatus, Delta, ErrorCode
from ..rules.constraints import BranchingConstraints, RadiusRuleSpec, DegradationRuleSpec
from ..rules.radius import apply_radius_rule


@dataclass
class KaryTreeSpec:
    """
    Configuration for K-ary tree generation with leader + lateral branching.
    
    All tunable parameters are centralized here - no magic numbers in the algorithm.
    This spec produces tree-like structures with a dominant trunk and hierarchical
    branching, smooth direction memory, space-aware growth, and soft collision avoidance.
    
    Default values are tuned for a 10mm diameter x 2mm height cylinder domain.
    """
    
    K: int = 4
    num_levels: int = 4
    
    leader_angle_deg_start: float = 12.0
    leader_angle_deg_min: float = 5.0
    leader_angle_decay_levels: float = 3.0
    
    side_angle_deg_start: float = 35.0
    side_angle_deg_min: float = 18.0
    side_angle_decay_levels: float = 5.0
    
    leader_length_mult: float = 1.0
    side_length_mult_start: float = 0.55
    side_length_mult_end: float = 0.85
    
    side_radius_factor: float = 0.75
    
    azimuth_sigma_deg_start: float = 15.0
    azimuth_sigma_deg_end: float = 5.0
    
    use_golden_angle_rotation: bool = True
    golden_angle_deg: float = 137.50776405
    
    envelope_r_frac_start: float = 0.35
    envelope_r_frac_end: float = 0.95
    
    enable_soft_collision: bool = True
    collision_sample_stride: int = 1
    collision_clearance_factor: float = 2.0
    collision_attempts_per_child: int = 3
    collision_azimuth_jitter_deg: float = 10.0
    
    boundary_safety: float = 0.95
    z_margin_voxels: int = 2
    wall_margin_m: float = 0.0
    enforce_downward_bias: bool = True
    downward_bias_min_dz: float = -0.05
    
    enable_reaim_instead_of_skip: bool = True
    reaim_attempts: int = 3
    reaim_angle_shrink: float = 0.8
    reaim_length_shrink: float = 0.85
    
    min_segment_length_m: float = 5e-5
    
    def get_leader_angle_deg(self, level: int) -> float:
        """Get leader branch angle for a given level (decays toward min)."""
        decay_factor = min(level / max(self.leader_angle_decay_levels, 1.0), 1.0)
        return self.leader_angle_deg_start - decay_factor * (self.leader_angle_deg_start - self.leader_angle_deg_min)
    
    def get_side_angle_deg(self, level: int) -> float:
        """Get lateral branch angle for a given level (decays toward min)."""
        decay_factor = min(level / max(self.side_angle_decay_levels, 1.0), 1.0)
        return self.side_angle_deg_start - decay_factor * (self.side_angle_deg_start - self.side_angle_deg_min)
    
    def get_side_length_mult(self, level: int) -> float:
        """Get lateral branch length multiplier for a given level (grows toward end)."""
        if self.num_levels <= 1:
            return self.side_length_mult_end
        frac = level / (self.num_levels - 1)
        return self.side_length_mult_start + frac * (self.side_length_mult_end - self.side_length_mult_start)
    
    def get_azimuth_sigma_deg(self, level: int) -> float:
        """Get azimuth preference sigma for a given level (tightens toward end)."""
        if self.num_levels <= 1:
            return self.azimuth_sigma_deg_end
        frac = level / (self.num_levels - 1)
        return self.azimuth_sigma_deg_start - frac * (self.azimuth_sigma_deg_start - self.azimuth_sigma_deg_end)
    
    def get_envelope_r_frac(self, level: int) -> float:
        """Get allowable radial fraction for a given level (expands toward end)."""
        if self.num_levels <= 1:
            return self.envelope_r_frac_end
        frac = level / (self.num_levels - 1)
        return self.envelope_r_frac_start + frac * (self.envelope_r_frac_end - self.envelope_r_frac_start)


def grow_branch(
    network: VascularNetwork,
    from_node_id: int,
    length: float,
    direction: Optional[Tuple[float, float, float] | Direction3D] = None,
    target_radius: Optional[float] = None,
    constraints: Optional[BranchingConstraints] = None,
    check_collisions: bool = True,
    seed: Optional[int] = None,
) -> OperationResult:
    """
    Grow a branch from an existing node.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to modify
    from_node_id : int
        Node to grow from
    length : float
        Length of new segment
    direction : tuple or Direction3D, optional
        Growth direction (if None, uses node's stored direction)
    target_radius : float, optional
        Radius of new segment (if None, uses parent radius)
    constraints : BranchingConstraints, optional
        Branching constraints
    check_collisions : bool
        Whether to check for collisions
    seed : int, optional
        Random seed for deterministic behavior
    
    Returns
    -------
    result : OperationResult
        Result with new_ids containing 'node' and 'segment'
    """
    if constraints is None:
        constraints = BranchingConstraints()
    
    parent_node = network.get_node(from_node_id)
    if parent_node is None:
        return OperationResult.failure(
            message=f"Node {from_node_id} not found",
            errors=["Node not found"],
        )
    
    if direction is None:
        if "direction" in parent_node.attributes:
            direction = Direction3D.from_dict(parent_node.attributes["direction"])
        else:
            return OperationResult.failure(
                message=f"No direction specified and node has no stored direction",
                errors=["Missing direction"],
            )
    elif isinstance(direction, tuple):
        direction = Direction3D.from_tuple(direction)
    
    if target_radius is None:
        if "radius" in parent_node.attributes:
            target_radius = parent_node.attributes["radius"]
        else:
            return OperationResult.failure(
                message=f"No radius specified and node has no stored radius",
                errors=["Missing radius"],
            )
    
    if length < constraints.min_segment_length:
        return OperationResult.failure(
            message=f"Length {length} below minimum {constraints.min_segment_length}",
            errors=["Length too short"],
        )
    
    if length > constraints.max_segment_length:
        return OperationResult.failure(
            message=f"Length {length} above maximum {constraints.max_segment_length}",
            errors=["Length too long"],
        )
    
    if target_radius < constraints.min_radius:
        return OperationResult.failure(
            message=f"Radius {target_radius} below minimum {constraints.min_radius}",
            errors=["Radius too small"],
        )
    
    direction_arr = direction.to_array()
    new_position = Point3D(
        parent_node.position.x + length * direction_arr[0],
        parent_node.position.y + length * direction_arr[1],
        parent_node.position.z + length * direction_arr[2],
    )
    
    if not network.domain.contains(new_position):
        new_position = network.domain.project_inside(new_position)
        if not network.domain.contains(new_position):
            return OperationResult.failure(
                message=f"New position outside domain",
                errors=["Position outside domain"],
            )
    
    warnings = []
    if check_collisions:
        from .collision import check_segment_collision_swept, check_domain_boundary_clearance
        
        parent_pos = np.array([
            parent_node.position.x,
            parent_node.position.y,
            parent_node.position.z,
        ])
        new_pos = np.array([
            new_position.x,
            new_position.y,
            new_position.z,
        ])
        
        has_collision, collision_details = check_segment_collision_swept(
            network,
            new_seg_start=parent_pos,
            new_seg_end=new_pos,
            new_seg_radius=target_radius,
            exclude_node_ids=[from_node_id],
            min_clearance=0.0005,
        )
        
        for detail in collision_details:
            warnings.append(
                f"Near collision with segment {detail['segment_id']} "
                f"(clearance: {detail['clearance']:.4f}m, required: {detail['min_required']:.4f}m)"
            )
        
        if hasattr(network, 'domain') and network.domain is not None:
            fits, margin = check_domain_boundary_clearance(
                network.domain,
                parent_pos,
                new_pos,
                target_radius,
                wall_thickness=0.0003,
            )
            if not fits:
                warnings.append(f"Tube may extend outside domain (margin: {margin:.4f}m)")
    
    new_node_id = network.id_gen.next_id()
    new_node = Node(
        id=new_node_id,
        position=new_position,
        node_type="terminal",
        vessel_type=parent_node.vessel_type,
        attributes={
            "radius": target_radius,
            "direction": direction.to_dict(),
            "branch_order": parent_node.attributes.get("branch_order", 0) + 1,
        },
    )
    
    segment_id = network.id_gen.next_id()
    parent_radius = parent_node.attributes.get("radius", target_radius)
    geometry = TubeGeometry(
        start=parent_node.position,
        end=new_position,
        radius_start=parent_radius,
        radius_end=target_radius,
    )
    
    segment = VesselSegment(
        id=segment_id,
        start_node_id=from_node_id,
        end_node_id=new_node_id,
        geometry=geometry,
        vessel_type=parent_node.vessel_type,
    )
    
    network.add_node(new_node)
    network.add_segment(segment)
    
    if parent_node.node_type == "terminal":
        parent_node.node_type = "junction"
    
    delta = Delta(
        created_node_ids=[new_node_id],
        created_segment_ids=[segment_id],
    )
    
    status = OperationStatus.SUCCESS if not warnings else OperationStatus.PARTIAL_SUCCESS
    
    return OperationResult(
        status=status,
        message=f"Grew branch from node {from_node_id}",
        new_ids={"node": new_node_id, "segment": segment_id},
        warnings=warnings,
        delta=delta,
        rng_state=network.id_gen.get_state(),
    )


def grow_to_point(
    network: VascularNetwork,
    from_node_id: int,
    target_point: Tuple[float, float, float] | Point3D,
    target_radius: Optional[float] = None,
    constraints: Optional[BranchingConstraints] = None,
    check_collisions: bool = True,
    fail_on_collision: bool = True,
    use_polyline_routing: bool = False,
    seed: Optional[int] = None,
) -> OperationResult:
    """
    Grow a branch from an existing node to a target point.
    
    This is a convenience function that computes the direction and length
    from the parent node to the target point, then grows a branch. By default
    grows a straight segment, but can optionally use polyline routing to
    avoid collisions or stay within the domain.
    
    Note: The library uses METERS internally for all geometry.
    
    P2-3: fail_on_collision defaults to True for agent workflows. When True,
    the operation fails if a collision is detected (unless polyline routing
    succeeds). When False, collisions generate warnings but don't fail.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to modify
    from_node_id : int
        Node to grow from
    target_point : tuple or Point3D
        Target point to grow to (x, y, z) in meters
    target_radius : float, optional
        Radius of new segment in meters (if None, uses parent radius)
    constraints : BranchingConstraints, optional
        Branching constraints
    check_collisions : bool
        Whether to check for collisions
    fail_on_collision : bool
        If True (default), fail the operation when collision is detected
        and polyline routing is not enabled or fails. If False, collisions
        generate warnings but the operation succeeds.
    use_polyline_routing : bool
        If True, attempt polyline routing when straight-line growth
        would collide or exit domain. Creates a TubeGeometry with
        centerline_points for curved paths.
    seed : int, optional
        Random seed for deterministic behavior
    
    Returns
    -------
    result : OperationResult
        Result with new_ids containing 'node' and 'segment'
        If polyline routing was used, the segment geometry will have
        centerline_points populated.
    
    Examples
    --------
    >>> from generation.core.network import VascularNetwork
    >>> from generation.core.domain import BoxDomain
    >>> from generation.ops.growth import grow_to_point
    >>> 
    >>> # Create network with inlet node
    >>> domain = BoxDomain.from_center_and_size((0, 0, 0), 0.1, 0.1, 0.1)
    >>> network = VascularNetwork(domain=domain)
    >>> inlet_id = network.add_inlet((0, 0, 0), radius=0.005)
    >>> 
    >>> # Grow branch to specific point
    >>> result = grow_to_point(
    ...     network,
    ...     from_node_id=inlet_id,
    ...     target_point=(0.02, 0.01, 0.005),  # Target coordinates in meters
    ...     target_radius=0.003,
    ... )
    >>> 
    >>> if result.is_success():
    ...     new_node_id = result.new_ids['node']
    ...     print(f"Created node at target point: {new_node_id}")
    """
    if constraints is None:
        constraints = BranchingConstraints()
    
    parent_node = network.get_node(from_node_id)
    if parent_node is None:
        return OperationResult.failure(
            message=f"Node {from_node_id} not found",
            errors=["Node not found"],
        )
    
    if isinstance(target_point, tuple):
        target_point = Point3D(target_point[0], target_point[1], target_point[2])
    
    if not network.domain.contains(target_point):
        return OperationResult.failure(
            message=f"Target point {target_point} is outside domain",
            errors=["Target point outside domain"],
        )
    
    parent_pos = parent_node.position
    dx = target_point.x - parent_pos.x
    dy = target_point.y - parent_pos.y
    dz = target_point.z - parent_pos.z
    
    length = np.sqrt(dx**2 + dy**2 + dz**2)
    
    if length < 1e-6:
        return OperationResult.failure(
            message=f"Target point is too close to parent node (distance: {length:.6f})",
            errors=["Target point too close"],
        )
    
    direction = Direction3D(dx / length, dy / length, dz / length)
    
    if target_radius is None:
        if "radius" in parent_node.attributes:
            target_radius = parent_node.attributes["radius"]
        else:
            return OperationResult.failure(
                message=f"No radius specified and node has no stored radius",
                errors=["Missing radius"],
            )
    
    # Validate length against constraints
    if length < constraints.min_segment_length:
        return OperationResult.failure(
            message=f"Distance to target {length:.4f} below minimum segment length {constraints.min_segment_length}",
            errors=["Length too short"],
        )
    
    if length > constraints.max_segment_length:
        return OperationResult.failure(
            message=f"Distance to target {length:.4f} above maximum segment length {constraints.max_segment_length}",
            errors=["Length too long"],
        )
    
    if target_radius < constraints.min_radius:
        return OperationResult.failure(
            message=f"Radius {target_radius} below minimum {constraints.min_radius}",
            errors=["Radius too small"],
        )
    
    warnings = []
    centerline_points: List[Point3D] = []
    
    parent_pos = np.array([
        parent_node.position.x,
        parent_node.position.y,
        parent_node.position.z,
    ])
    target_pos = np.array([
        target_point.x,
        target_point.y,
        target_point.z,
    ])
    
    if check_collisions:
        from .collision import check_segment_collision_swept, check_domain_boundary_clearance
        
        has_collision, collision_details = check_segment_collision_swept(
            network,
            new_seg_start=parent_pos,
            new_seg_end=target_pos,
            new_seg_radius=target_radius,
            exclude_node_ids=[from_node_id],
            min_clearance=0.001,
        )
        
        domain_fits = True
        if hasattr(network, 'domain') and network.domain is not None:
            domain_fits, margin = check_domain_boundary_clearance(
                network.domain,
                parent_pos,
                target_pos,
                target_radius,
                wall_thickness=0.0003,
            )
        
        if (has_collision or not domain_fits) and use_polyline_routing:
            centerline_points = _compute_polyline_route(
                network, parent_pos, target_pos, target_radius, from_node_id
            )
            
            if centerline_points:
                warnings.append(
                    f"Used polyline routing with {len(centerline_points)} waypoints "
                    f"to avoid {'collision' if has_collision else 'domain boundary'}"
                )
            else:
                if fail_on_collision:
                    error_msgs = []
                    for detail in collision_details:
                        error_msgs.append(
                            f"Collision with segment {detail['segment_id']} "
                            f"(clearance: {detail['clearance']:.4f}m, required: {detail['min_required']:.4f}m)"
                        )
                    if not domain_fits:
                        error_msgs.append(f"Tube extends outside domain (margin: {margin:.4f}m)")
                    return OperationResult.failure(
                        message="Growth failed: collision detected and polyline routing failed",
                        errors=error_msgs,
                    )
                else:
                    for detail in collision_details:
                        warnings.append(
                            f"Near collision with segment {detail['segment_id']} "
                            f"(clearance: {detail['clearance']:.4f}m, required: {detail['min_required']:.4f}m)"
                        )
                    if not domain_fits:
                        warnings.append(f"Tube may extend outside domain (margin: {margin:.4f}m)")
        elif has_collision or not domain_fits:
            if fail_on_collision:
                error_msgs = []
                for detail in collision_details:
                    error_msgs.append(
                        f"Collision with segment {detail['segment_id']} "
                        f"(clearance: {detail['clearance']:.4f}m, required: {detail['min_required']:.4f}m)"
                    )
                if not domain_fits:
                    error_msgs.append(f"Tube extends outside domain (margin: {margin:.4f}m)")
                return OperationResult.failure(
                    message="Growth failed: collision detected",
                    errors=error_msgs,
                )
            else:
                for detail in collision_details:
                    warnings.append(
                        f"Near collision with segment {detail['segment_id']} "
                        f"(clearance: {detail['clearance']:.4f}m, required: {detail['min_required']:.4f}m)"
                    )
                if not domain_fits:
                    warnings.append(f"Tube may extend outside domain (margin: {margin:.4f}m)")
    
    new_node_id = network.id_gen.next_id()
    new_node = Node(
        id=new_node_id,
        position=target_point,
        node_type="terminal",
        vessel_type=parent_node.vessel_type,
        attributes={
            "radius": target_radius,
            "direction": direction.to_dict(),
            "branch_order": parent_node.attributes.get("branch_order", 0) + 1,
        },
    )
    
    segment_id = network.id_gen.next_id()
    parent_radius_val = parent_node.attributes.get("radius", target_radius)
    geometry = TubeGeometry(
        start=parent_node.position,
        end=target_point,
        radius_start=parent_radius_val,
        radius_end=target_radius,
        centerline_points=centerline_points if centerline_points else None,
    )
    
    segment = VesselSegment(
        id=segment_id,
        start_node_id=from_node_id,
        end_node_id=new_node_id,
        geometry=geometry,
        vessel_type=parent_node.vessel_type,
    )
    
    network.add_node(new_node)
    network.add_segment(segment)
    
    if parent_node.node_type == "terminal":
        parent_node.node_type = "junction"
    
    delta = Delta(
        created_node_ids=[new_node_id],
        created_segment_ids=[segment_id],
    )
    
    status = OperationStatus.SUCCESS if not warnings else OperationStatus.PARTIAL_SUCCESS
    
    return OperationResult(
        status=status,
        message=f"Grew branch from node {from_node_id} to point {target_point}",
        new_ids={"node": new_node_id, "segment": segment_id},
        warnings=warnings,
        delta=delta,
        rng_state=network.id_gen.get_state(),
    )


def bifurcate(
    network: VascularNetwork,
    at_node_id: int,
    child_lengths: Tuple[float, float],
    angle_deg: float = 45.0,
    radius_rule: Optional[RadiusRuleSpec] = None,
    degradation_rule: Optional[DegradationRuleSpec] = None,
    constraints: Optional[BranchingConstraints] = None,
    check_collisions: bool = True,
    seed: Optional[int] = None,
) -> OperationResult:
    """
    Create a bifurcation at an existing node.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to modify
    at_node_id : int
        Node to bifurcate from
    child_lengths : tuple of float
        Lengths of two child branches
    angle_deg : float
        Branching angle in degrees (each child deviates by this angle)
    radius_rule : RadiusRuleSpec, optional
        Rule for computing child radii (default: Murray's law)
    degradation_rule : DegradationRuleSpec, optional
        Rule for radius degradation across generations (default: none)
    constraints : BranchingConstraints, optional
        Branching constraints
    check_collisions : bool
        Whether to check for collisions
    seed : int, optional
        Random seed
    
    Returns
    -------
    result : OperationResult
        Result with new_ids containing 'nodes' and 'segments' lists
    """
    if constraints is None:
        constraints = BranchingConstraints()
    
    if radius_rule is None:
        radius_rule = RadiusRuleSpec.murray()
    
    parent_node = network.get_node(at_node_id)
    if parent_node is None:
        return OperationResult.failure(
            message=f"Node {at_node_id} not found",
            errors=["Node not found"],
        )
    
    if "direction" not in parent_node.attributes:
        return OperationResult.failure(
            message=f"Node has no stored direction",
            errors=["Missing direction"],
        )
    
    parent_direction = Direction3D.from_dict(parent_node.attributes["direction"])
    parent_radius = parent_node.attributes.get("radius", 0.005)
    parent_generation = parent_node.attributes.get("branch_order", 0)
    child_generation = parent_generation + 1
    
    if angle_deg > constraints.max_branch_angle_deg:
        return OperationResult.failure(
            message=f"Angle {angle_deg} exceeds maximum {constraints.max_branch_angle_deg}",
            errors=["Angle too large"],
        )
    
    if degradation_rule is not None:
        should_term, term_reason = degradation_rule.should_terminate(parent_radius, child_generation)
        if should_term:
            return OperationResult.failure(
                message=f"Bifurcation blocked by degradation rule: {term_reason}",
                error_codes=[
                    ErrorCode.BELOW_MIN_TERMINAL_RADIUS.value if "radius" in term_reason.lower()
                    else ErrorCode.MAX_GENERATION_EXCEEDED.value
                ],
            )
    
    rng = np.random.default_rng(seed) if seed is not None else network.id_gen.rng
    r1, r2 = apply_radius_rule(parent_radius, radius_rule, rng)
    
    if degradation_rule is not None:
        r1 = degradation_rule.apply_degradation(r1, child_generation)
        r2 = degradation_rule.apply_degradation(r2, child_generation)
    
    if r1 < constraints.min_radius or r2 < constraints.min_radius:
        return OperationResult.failure(
            message=f"Child radii below minimum after degradation",
            errors=["Radii too small"],
            error_codes=[ErrorCode.RADIUS_TOO_SMALL.value],
        )
    
    parent_dir_arr = parent_direction.to_array()
    
    if abs(parent_dir_arr[2]) < 0.9:
        perp = np.array([0, 0, 1])
    else:
        perp = np.array([1, 0, 0])
    
    perp = perp - np.dot(perp, parent_dir_arr) * parent_dir_arr
    perp = perp / np.linalg.norm(perp)
    
    angle_rad = np.radians(angle_deg)
    
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    child1_dir = cos_a * parent_dir_arr + sin_a * perp
    child1_dir = child1_dir / np.linalg.norm(child1_dir)
    
    child2_dir = cos_a * parent_dir_arr - sin_a * perp
    child2_dir = child2_dir / np.linalg.norm(child2_dir)
    
    result1 = grow_branch(
        network,
        from_node_id=at_node_id,
        length=child_lengths[0],
        direction=Direction3D.from_array(child1_dir),
        target_radius=r1,
        constraints=constraints,
        check_collisions=check_collisions,
        seed=seed,
    )
    
    if not result1.is_success():
        return result1
    
    result2 = grow_branch(
        network,
        from_node_id=at_node_id,
        length=child_lengths[1],
        direction=Direction3D.from_array(child2_dir),
        target_radius=r2,
        constraints=constraints,
        check_collisions=check_collisions,
        seed=seed,
    )
    
    if not result2.is_success():
        network.remove_node(result1.new_ids["node"])
        network.remove_segment(result1.new_ids["segment"])
        return result2
    
    all_warnings = result1.warnings + result2.warnings
    
    delta = Delta(
        created_node_ids=[result1.new_ids["node"], result2.new_ids["node"]],
        created_segment_ids=[result1.new_ids["segment"], result2.new_ids["segment"]],
    )
    
    status = OperationStatus.SUCCESS if not all_warnings else OperationStatus.PARTIAL_SUCCESS
    
    return OperationResult(
        status=status,
        message=f"Created bifurcation at node {at_node_id}",
        new_ids={
            "nodes": [result1.new_ids["node"], result2.new_ids["node"]],
            "segments": [result1.new_ids["segment"], result2.new_ids["segment"]],
        },
        warnings=all_warnings,
        delta=delta,
        rng_state=network.id_gen.get_state(),
    )


def _compute_polyline_route(
    network: VascularNetwork,
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    radius: float,
    exclude_node_id: int,
    max_waypoints: int = 5,
) -> List[Point3D]:
    """
    Compute a polyline route from start to end that avoids collisions.
    
    Uses a simple waypoint-based approach: tries to find intermediate points
    that avoid obstacles by detouring around them.
    
    Parameters
    ----------
    network : VascularNetwork
        Network containing existing segments to avoid
    start_pos : np.ndarray
        Starting position
    end_pos : np.ndarray
        Target position
    radius : float
        Radius of the new segment
    exclude_node_id : int
        Node ID to exclude from collision checks
    max_waypoints : int
        Maximum number of waypoints to try
        
    Returns
    -------
    waypoints : list of Point3D
        List of intermediate waypoints (not including start/end)
        Empty list if no valid route found
    """
    from .collision import check_segment_collision_swept
    
    direct_vec = end_pos - start_pos
    direct_length = np.linalg.norm(direct_vec)
    
    if direct_length < 1e-6:
        return []
    
    direct_dir = direct_vec / direct_length
    
    if abs(direct_dir[2]) < 0.9:
        perp1 = np.cross(direct_dir, np.array([0, 0, 1]))
    else:
        perp1 = np.cross(direct_dir, np.array([1, 0, 0]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(direct_dir, perp1)
    
    detour_distances = [0.005, 0.01, 0.02, 0.03]
    detour_directions = [perp1, -perp1, perp2, -perp2, perp1 + perp2, perp1 - perp2, -perp1 + perp2, -perp1 - perp2]
    
    for detour_dist in detour_distances:
        for detour_dir in detour_directions:
            detour_dir = detour_dir / np.linalg.norm(detour_dir)
            
            mid_point = (start_pos + end_pos) / 2 + detour_dir * detour_dist
            
            if hasattr(network, 'domain') and network.domain is not None:
                mid_point3d = Point3D.from_array(mid_point)
                if not network.domain.contains(mid_point3d):
                    continue
            
            has_collision1, _ = check_segment_collision_swept(
                network,
                new_seg_start=start_pos,
                new_seg_end=mid_point,
                new_seg_radius=radius,
                exclude_node_ids=[exclude_node_id],
                min_clearance=0.0005,
            )
            
            if has_collision1:
                continue
            
            has_collision2, _ = check_segment_collision_swept(
                network,
                new_seg_start=mid_point,
                new_seg_end=end_pos,
                new_seg_radius=radius,
                exclude_node_ids=[exclude_node_id],
                min_clearance=0.0005,
            )
            
            if not has_collision2:
                return [Point3D.from_array(mid_point)]
    
    for detour_dist in detour_distances:
        for detour_dir in detour_directions:
            detour_dir = detour_dir / np.linalg.norm(detour_dir)
            
            quarter_point = start_pos + direct_vec * 0.25 + detour_dir * detour_dist
            three_quarter_point = start_pos + direct_vec * 0.75 + detour_dir * detour_dist
            
            if hasattr(network, 'domain') and network.domain is not None:
                qp = Point3D.from_array(quarter_point)
                tqp = Point3D.from_array(three_quarter_point)
                if not network.domain.contains(qp) or not network.domain.contains(tqp):
                    continue
            
            segments_ok = True
            test_points = [start_pos, quarter_point, three_quarter_point, end_pos]
            
            for i in range(len(test_points) - 1):
                has_collision, _ = check_segment_collision_swept(
                    network,
                    new_seg_start=test_points[i],
                    new_seg_end=test_points[i + 1],
                    new_seg_radius=radius,
                    exclude_node_ids=[exclude_node_id],
                    min_clearance=0.0005,
                )
                if has_collision:
                    segments_ok = False
                    break
            
            if segments_ok:
                return [
                    Point3D.from_array(quarter_point),
                    Point3D.from_array(three_quarter_point),
                ]
    
    return []


def _compute_tube_safe_step_length(
    start: np.ndarray,
    direction: np.ndarray,
    tube_radius: float,
    domain: "DomainSpec",
    requested_length: float,
) -> float:
    """
    Compute the maximum step length that keeps the entire tube inside the domain.
    
    This performs ray-domain intersection with radius margins to ensure the tube
    surface (not just centerline) stays inside the domain boundaries.
    
    Parameters
    ----------
    start : np.ndarray
        Starting point [x, y, z]
    direction : np.ndarray
        Normalized direction vector [dx, dy, dz]
    tube_radius : float
        Radius of the tube (vessel)
    domain : DomainSpec
        Domain to check against
    requested_length : float
        Desired step length
        
    Returns
    -------
    float
        Maximum safe step length (may be less than requested_length)
    """
    from ..core.domain import CylinderDomain, BoxDomain
    
    if isinstance(domain, CylinderDomain):
        cx = domain.center.x
        cy = domain.center.y
        cz = domain.center.z
        R = domain.radius
        H = domain.height
        half_h = H / 2.0
        
        effective_radius = R - tube_radius
        if effective_radius <= 0:
            return 0.0
        
        z_min = cz - half_h + tube_radius
        z_max = cz + half_h - tube_radius
        
        dx, dy, dz = direction[0], direction[1], direction[2]
        px, py, pz = start[0], start[1], start[2]
        
        ox = px - cx
        oy = py - cy
        start_r_xy = np.sqrt(ox * ox + oy * oy)
        if start_r_xy > effective_radius:
            return 0.0
        
        if pz < z_min or pz > z_max:
            return 0.0
        
        max_t = requested_length
        
        a = dx * dx + dy * dy
        b = 2.0 * (ox * dx + oy * dy)
        c = ox * ox + oy * oy - effective_radius * effective_radius
        
        if a > 1e-12:
            discriminant = b * b - 4.0 * a * c
            if discriminant >= 0:
                sqrt_disc = np.sqrt(discriminant)
                t1 = (-b - sqrt_disc) / (2.0 * a)
                t2 = (-b + sqrt_disc) / (2.0 * a)
                
                if t2 > 0 and t2 < max_t:
                    max_t = max(t2, 0.0)
        
        if abs(dz) > 1e-12:
            if dz < 0:
                t_bottom = (z_min - pz) / dz
                if t_bottom > 0 and t_bottom < max_t:
                    max_t = t_bottom
            else:
                t_top = (z_max - pz) / dz
                if t_top > 0 and t_top < max_t:
                    max_t = t_top
        
        return max(max_t, 0.0)
    
    elif isinstance(domain, BoxDomain):
        effective_x_min = domain.x_min + tube_radius
        effective_x_max = domain.x_max - tube_radius
        effective_y_min = domain.y_min + tube_radius
        effective_y_max = domain.y_max - tube_radius
        effective_z_min = domain.z_min + tube_radius
        effective_z_max = domain.z_max - tube_radius
        
        if (effective_x_min >= effective_x_max or 
            effective_y_min >= effective_y_max or 
            effective_z_min >= effective_z_max):
            return 0.0
        
        px, py, pz = start[0], start[1], start[2]
        if (px < effective_x_min or px > effective_x_max or
            py < effective_y_min or py > effective_y_max or
            pz < effective_z_min or pz > effective_z_max):
            return 0.0
        
        max_t = requested_length
        dx, dy, dz = direction[0], direction[1], direction[2]
        
        if abs(dx) > 1e-12:
            if dx > 0:
                t = (effective_x_max - px) / dx
            else:
                t = (effective_x_min - px) / dx
            if t > 0 and t < max_t:
                max_t = t
        
        if abs(dy) > 1e-12:
            if dy > 0:
                t = (effective_y_max - py) / dy
            else:
                t = (effective_y_min - py) / dy
            if t > 0 and t < max_t:
                max_t = t
        
        if abs(dz) > 1e-12:
            if dz > 0:
                t = (effective_z_max - pz) / dz
            else:
                t = (effective_z_min - pz) / dz
            if t > 0 and t < max_t:
                max_t = t
        
        return max(max_t, 0.0)
    
    else:
        end = start + requested_length * direction
        end_pt = Point3D(end[0], end[1], end[2])
        if domain.contains(end_pt):
            return requested_length
        
        projected = domain.project_inside(end_pt)
        vec = np.array([projected.x, projected.y, projected.z]) - start
        return max(np.linalg.norm(vec) - tube_radius, 0.0)


def grow_kary_tree_to_depths(
    network: VascularNetwork,
    root_node_id: int,
    K: int,
    depths: List[float],
    taper_fn: Optional[callable] = None,
    angle_deg: float = 45.0,
    angle_decay: float = 0.9,
    azimuth_mode: str = "random_uniform",
    boundary_mode: str = "strict",
    child_length_scale_fn: Optional[callable] = None,
    constraints: Optional[BranchingConstraints] = None,
    seed: Optional[int] = None,
) -> OperationResult:
    """
    Grow a K-ary tree from a root node to specified bifurcation depths.
    
    This implements non-planar 3D branching where each bifurcation creates K children
    evenly distributed around 360 degrees with a random azimuth offset. The tree
    grows downward (negative Z) from the root to the specified depths.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to modify
    root_node_id : int
        Node to grow tree from (typically an inlet)
    K : int
        Number of children at each bifurcation (e.g., 2 for binary, 4 for quad)
    depths : list of float
        Bifurcation depths from the root (positive values, measured downward)
        Example: [0.25e-3, 0.50e-3, 0.75e-3] for 3 levels at 0.25mm, 0.5mm, 0.75mm
    taper_fn : callable, optional
        Function(level, num_levels, parent_radius) -> child_radius
        If None, uses Murray's law scaling
    angle_deg : float
        Base branching angle in degrees (default: 45)
    angle_decay : float
        Angle decay factor per level (default: 0.9)
    azimuth_mode : str
        How to distribute children around the parent direction:
        - "random_uniform": Random azimuth offset, children evenly spaced (default)
        - "fixed": No random offset, deterministic placement
    boundary_mode : str
        How to handle domain boundaries:
        - "strict": Clamp step length so tube stays inside domain (default)
        - "project_inside": Allow domain projection for more growth
    child_length_scale_fn : callable, optional
        Function(level, num_levels) -> scale factor for child lengths
        If None, uses uniform scaling
    constraints : BranchingConstraints, optional
        Branching constraints
    seed : int, optional
        Random seed for deterministic behavior
        
    Returns
    -------
    result : OperationResult
        Result with new_ids containing 'nodes' and 'segments' lists
        
    Examples
    --------
    >>> # Create a 4-ary tree with 7 bifurcation levels
    >>> depths = [0.25e-3, 0.50e-3, 0.75e-3, 1.0e-3, 1.25e-3, 1.5e-3, 1.75e-3]
    >>> result = grow_kary_tree_to_depths(
    ...     network, inlet_id, K=4, depths=depths,
    ...     taper_fn=lambda lvl, n, r: r * 0.7,  # 30% taper per level
    ...     seed=42,
    ... )
    """
    if constraints is None:
        constraints = BranchingConstraints()
    
    rng = np.random.default_rng(seed)
    
    root_node = network.get_node(root_node_id)
    if root_node is None:
        return OperationResult.failure(
            message=f"Root node {root_node_id} not found",
            errors=["Node not found"],
        )
    
    if "radius" not in root_node.attributes:
        return OperationResult.failure(
            message="Root node has no stored radius",
            errors=["Missing radius"],
        )
    
    root_radius = root_node.attributes["radius"]
    root_pos = np.array([root_node.position.x, root_node.position.y, root_node.position.z])
    
    if "direction" in root_node.attributes:
        root_dir = Direction3D.from_dict(root_node.attributes["direction"]).to_array()
    else:
        root_dir = np.array([0.0, 0.0, -1.0])
    
    num_levels = len(depths)
    all_node_ids = []
    all_segment_ids = []
    all_warnings = []
    
    tips = [(root_node_id, root_pos, root_dir, root_radius, 0)]
    
    for level in range(num_levels):
        target_depth = depths[level]
        new_tips = []
        
        current_angle = angle_deg * (angle_decay ** level)
        angle_rad = np.radians(current_angle)
        
        for tip_node_id, tip_pos, tip_dir, tip_radius, tip_level in tips:
            if tip_level != level:
                new_tips.append((tip_node_id, tip_pos, tip_dir, tip_radius, tip_level))
                continue
            
            if taper_fn is not None:
                child_radius = taper_fn(level, num_levels, tip_radius)
            else:
                child_radius = tip_radius * (0.5 ** (1.0 / 3.0))
            
            child_radius = max(child_radius, constraints.min_radius)
            
            if level < num_levels - 1:
                next_depth = depths[level + 1]
            else:
                next_depth = target_depth + (target_depth - depths[level - 1] if level > 0 else target_depth)
            
            base_length = next_depth - target_depth
            if child_length_scale_fn is not None:
                length_scale = child_length_scale_fn(level, num_levels)
            else:
                length_scale = 1.0
            child_length = base_length * length_scale
            child_length = max(child_length, constraints.min_segment_length)
            
            tip_dir_norm = tip_dir / (np.linalg.norm(tip_dir) + 1e-12)
            
            if abs(tip_dir_norm[2]) < 0.9:
                perp1 = np.cross(tip_dir_norm, np.array([0, 0, 1]))
            else:
                perp1 = np.cross(tip_dir_norm, np.array([1, 0, 0]))
            perp1 = perp1 / (np.linalg.norm(perp1) + 1e-12)
            perp2 = np.cross(tip_dir_norm, perp1)
            perp2 = perp2 / (np.linalg.norm(perp2) + 1e-12)
            
            if azimuth_mode == "random_uniform":
                azimuth_offset = rng.uniform(0, 2 * np.pi)
            else:
                azimuth_offset = 0.0
            
            for k in range(K):
                child_azimuth = azimuth_offset + (2 * np.pi * k / K)
                
                lateral = np.cos(child_azimuth) * perp1 + np.sin(child_azimuth) * perp2
                
                child_dir = np.cos(angle_rad) * tip_dir_norm + np.sin(angle_rad) * lateral
                child_dir = child_dir / (np.linalg.norm(child_dir) + 1e-12)
                
                if boundary_mode == "strict" and hasattr(network, 'domain') and network.domain is not None:
                    max_tube_radius = max(tip_radius, child_radius)
                    safe_length = _compute_tube_safe_step_length(
                        start=tip_pos,
                        direction=child_dir,
                        tube_radius=max_tube_radius,
                        domain=network.domain,
                        requested_length=child_length,
                    )
                    
                    if safe_length < constraints.min_segment_length:
                        all_warnings.append(
                            f"Level {level}, child {k}: tube-safe step too short "
                            f"({safe_length:.6f}m < {constraints.min_segment_length:.6f}m)"
                        )
                        continue
                    
                    child_length = safe_length
                
                child_end = tip_pos + child_length * child_dir
                
                child_end_pt = Point3D(child_end[0], child_end[1], child_end[2])
                child_dir_obj = Direction3D(child_dir[0], child_dir[1], child_dir[2])
                
                new_node_id = network.id_gen.next_id()
                new_node = Node(
                    id=new_node_id,
                    position=child_end_pt,
                    node_type="terminal",
                    vessel_type=root_node.vessel_type,
                    attributes={
                        "radius": child_radius,
                        "direction": child_dir_obj.to_dict(),
                        "branch_order": level + 1,
                        "tree_level": level,
                    },
                )
                
                segment_id = network.id_gen.next_id()
                geometry = TubeGeometry(
                    start=Point3D(tip_pos[0], tip_pos[1], tip_pos[2]),
                    end=child_end_pt,
                    radius_start=tip_radius,
                    radius_end=child_radius,
                )
                
                segment = VesselSegment(
                    id=segment_id,
                    start_node_id=tip_node_id,
                    end_node_id=new_node_id,
                    geometry=geometry,
                    vessel_type=root_node.vessel_type,
                )
                
                network.add_node(new_node)
                network.add_segment(segment)
                
                all_node_ids.append(new_node_id)
                all_segment_ids.append(segment_id)
                
                new_tips.append((new_node_id, child_end, child_dir, child_radius, level + 1))
            
            tip_node = network.get_node(tip_node_id)
            if tip_node is not None and tip_node.node_type == "terminal":
                tip_node.node_type = "junction"
        
        tips = new_tips
    
    delta = Delta(
        created_node_ids=all_node_ids,
        created_segment_ids=all_segment_ids,
    )
    
    status = OperationStatus.SUCCESS if not all_warnings else OperationStatus.PARTIAL_SUCCESS
    
    return OperationResult(
        status=status,
        message=f"Grew {K}-ary tree with {num_levels} levels from node {root_node_id}",
        new_ids={
            "nodes": all_node_ids,
            "segments": all_segment_ids,
        },
        warnings=all_warnings,
        delta=delta,
        rng_state=network.id_gen.get_state(),
    )


def _build_collision_samples(
    network: VascularNetwork,
    stride: int = 1,
) -> List[Tuple[np.ndarray, float]]:
    """
    Build a list of (center_point, radius) samples from existing segments for collision checking.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to sample from
    stride : int
        Sample every Nth segment (1 = all segments)
    
    Returns
    -------
    List of (center_point, radius) tuples
    """
    samples = []
    for i, seg in enumerate(network.segments.values()):
        if i % stride != 0:
            continue
        start_node = network.get_node(seg.start_node_id)
        end_node = network.get_node(seg.end_node_id)
        if start_node is None or end_node is None:
            continue
        
        start_pos = np.array([start_node.position.x, start_node.position.y, start_node.position.z])
        end_pos = np.array([end_node.position.x, end_node.position.y, end_node.position.z])
        center = (start_pos + end_pos) / 2.0
        
        if seg.geometry and hasattr(seg.geometry, 'radius_start'):
            radius = max(seg.geometry.radius_start, seg.geometry.radius_end)
        else:
            radius = start_node.attributes.get("radius", 0.0001)
        
        samples.append((center, radius))
    
    return samples


def _check_soft_collision(
    point: np.ndarray,
    radius: float,
    samples: List[Tuple[np.ndarray, float]],
    clearance_factor: float,
) -> bool:
    """
    Check if a point would cause a soft collision with existing samples.
    
    Returns True if collision detected (too close to existing geometry).
    """
    for sample_center, sample_radius in samples:
        dist = np.linalg.norm(point - sample_center)
        min_clearance = clearance_factor * max(radius, sample_radius)
        if dist < min_clearance:
            return True
    return False


def grow_kary_tree_v2(
    network: VascularNetwork,
    root_node_id: int,
    depths: List[float],
    spec: Optional[KaryTreeSpec] = None,
    taper_fn: Optional[callable] = None,
    constraints: Optional[BranchingConstraints] = None,
    seed: Optional[int] = None,
) -> OperationResult:
    """
    Grow a K-ary tree with leader + lateral branching pattern (upgraded algorithm).
    
    This implements a tree-like structure with:
    - Dominant trunk (leader branch) + hierarchical lateral branches
    - Smooth direction memory (azimuth preference)
    - Golden angle rotation to avoid periodic geometry
    - Level-dependent growth envelope
    - Soft collision avoidance
    - Re-aim strategy instead of skip-on-insufficient-room
    - Tube-safe boundary clamping
    
    Parameters
    ----------
    network : VascularNetwork
        Network to modify
    root_node_id : int
        Node to grow tree from (typically an inlet)
    depths : list of float
        Bifurcation depths from the root (positive values, measured downward)
    spec : KaryTreeSpec, optional
        Configuration for tree generation (uses defaults if None)
    taper_fn : callable, optional
        Function(level, num_levels, parent_radius, is_leader) -> child_radius
        If None, uses Murray's law scaling with side_radius_factor for laterals
    constraints : BranchingConstraints, optional
        Branching constraints
    seed : int, optional
        Random seed for deterministic behavior
        
    Returns
    -------
    result : OperationResult
        Result with new_ids containing 'nodes' and 'segments' lists
    """
    if spec is None:
        spec = KaryTreeSpec()
    
    if constraints is None:
        constraints = BranchingConstraints()
    
    rng = np.random.default_rng(seed)
    
    root_node = network.get_node(root_node_id)
    if root_node is None:
        return OperationResult.failure(
            message=f"Root node {root_node_id} not found",
            errors=["Node not found"],
        )
    
    if "radius" not in root_node.attributes:
        return OperationResult.failure(
            message="Root node has no stored radius",
            errors=["Missing radius"],
        )
    
    root_radius = root_node.attributes["radius"]
    root_pos = np.array([root_node.position.x, root_node.position.y, root_node.position.z])
    
    if "direction" in root_node.attributes:
        root_dir = Direction3D.from_dict(root_node.attributes["direction"]).to_array()
    else:
        root_dir = np.array([0.0, 0.0, -1.0])
    
    num_levels = len(depths)
    spec_with_levels = KaryTreeSpec(
        K=spec.K,
        num_levels=num_levels,
        leader_angle_deg_start=spec.leader_angle_deg_start,
        leader_angle_deg_min=spec.leader_angle_deg_min,
        leader_angle_decay_levels=spec.leader_angle_decay_levels,
        side_angle_deg_start=spec.side_angle_deg_start,
        side_angle_deg_min=spec.side_angle_deg_min,
        side_angle_decay_levels=spec.side_angle_decay_levels,
        leader_length_mult=spec.leader_length_mult,
        side_length_mult_start=spec.side_length_mult_start,
        side_length_mult_end=spec.side_length_mult_end,
        side_radius_factor=spec.side_radius_factor,
        azimuth_sigma_deg_start=spec.azimuth_sigma_deg_start,
        azimuth_sigma_deg_end=spec.azimuth_sigma_deg_end,
        use_golden_angle_rotation=spec.use_golden_angle_rotation,
        golden_angle_deg=spec.golden_angle_deg,
        envelope_r_frac_start=spec.envelope_r_frac_start,
        envelope_r_frac_end=spec.envelope_r_frac_end,
        enable_soft_collision=spec.enable_soft_collision,
        collision_sample_stride=spec.collision_sample_stride,
        collision_clearance_factor=spec.collision_clearance_factor,
        collision_attempts_per_child=spec.collision_attempts_per_child,
        collision_azimuth_jitter_deg=spec.collision_azimuth_jitter_deg,
        boundary_safety=spec.boundary_safety,
        z_margin_voxels=spec.z_margin_voxels,
        wall_margin_m=spec.wall_margin_m,
        enforce_downward_bias=spec.enforce_downward_bias,
        downward_bias_min_dz=spec.downward_bias_min_dz,
        enable_reaim_instead_of_skip=spec.enable_reaim_instead_of_skip,
        reaim_attempts=spec.reaim_attempts,
        reaim_angle_shrink=spec.reaim_angle_shrink,
        reaim_length_shrink=spec.reaim_length_shrink,
        min_segment_length_m=spec.min_segment_length_m,
    )
    spec = spec_with_levels
    
    all_node_ids = []
    all_segment_ids = []
    all_warnings = []
    
    initial_azimuth_pref = rng.uniform(0, 2 * np.pi)
    
    tips = [(root_node_id, root_pos, root_dir, root_radius, 0, initial_azimuth_pref)]
    
    from ..core.domain import CylinderDomain
    domain_radius = None
    domain_center_xy = None
    if hasattr(network, 'domain') and network.domain is not None:
        if isinstance(network.domain, CylinderDomain):
            domain_radius = network.domain.radius
            domain_center_xy = np.array([network.domain.center.x, network.domain.center.y])
    
    for level in range(num_levels):
        target_depth = depths[level]
        new_tips = []
        
        collision_samples = []
        if spec.enable_soft_collision:
            collision_samples = _build_collision_samples(network, spec.collision_sample_stride)
        
        leader_angle_deg = spec.get_leader_angle_deg(level)
        side_angle_deg = spec.get_side_angle_deg(level)
        side_length_mult = spec.get_side_length_mult(level)
        azimuth_sigma_deg = spec.get_azimuth_sigma_deg(level)
        envelope_r_frac = spec.get_envelope_r_frac(level)
        
        golden_rotation = 0.0
        if spec.use_golden_angle_rotation:
            golden_rotation = np.radians(spec.golden_angle_deg) * level
        
        for tip_node_id, tip_pos, tip_dir, tip_radius, tip_level, tip_azimuth_pref in tips:
            if tip_level != level:
                new_tips.append((tip_node_id, tip_pos, tip_dir, tip_radius, tip_level, tip_azimuth_pref))
                continue
            
            if level < num_levels - 1:
                next_depth = depths[level + 1]
            else:
                next_depth = target_depth + (target_depth - depths[level - 1] if level > 0 else target_depth)
            
            base_length = next_depth - target_depth
            base_length = max(base_length, spec.min_segment_length_m)
            
            tip_dir_norm = tip_dir / (np.linalg.norm(tip_dir) + 1e-12)
            
            if abs(tip_dir_norm[2]) < 0.9:
                perp1 = np.cross(tip_dir_norm, np.array([0, 0, 1]))
            else:
                perp1 = np.cross(tip_dir_norm, np.array([1, 0, 0]))
            perp1 = perp1 / (np.linalg.norm(perp1) + 1e-12)
            perp2 = np.cross(tip_dir_norm, perp1)
            perp2 = perp2 / (np.linalg.norm(perp2) + 1e-12)
            
            azimuth_sigma_rad = np.radians(azimuth_sigma_deg)
            sampled_azimuth = tip_azimuth_pref + rng.normal(0, azimuth_sigma_rad)
            sampled_azimuth += golden_rotation
            
            K = spec.K
            for k in range(K):
                is_leader = (k == 0)
                
                if is_leader:
                    angle_deg = leader_angle_deg
                    length_mult = spec.leader_length_mult
                    radius_factor = 1.0
                    child_azimuth = sampled_azimuth
                else:
                    angle_deg = side_angle_deg
                    length_mult = side_length_mult
                    radius_factor = spec.side_radius_factor
                    child_azimuth = sampled_azimuth + (2 * np.pi * (k - 1) / (K - 1)) if K > 1 else sampled_azimuth
                
                angle_rad = np.radians(angle_deg)
                child_length = base_length * length_mult
                
                if taper_fn is not None:
                    child_radius = taper_fn(level, num_levels, tip_radius, is_leader)
                else:
                    base_child_radius = tip_radius * (0.5 ** (1.0 / 3.0))
                    child_radius = base_child_radius * radius_factor
                
                child_radius = max(child_radius, constraints.min_radius)
                
                lateral = np.cos(child_azimuth) * perp1 + np.sin(child_azimuth) * perp2
                child_dir = np.cos(angle_rad) * tip_dir_norm + np.sin(angle_rad) * lateral
                child_dir = child_dir / (np.linalg.norm(child_dir) + 1e-12)
                
                if spec.enforce_downward_bias and child_dir[2] > spec.downward_bias_min_dz:
                    child_dir[2] = spec.downward_bias_min_dz
                    child_dir = child_dir / (np.linalg.norm(child_dir) + 1e-12)
                
                if domain_radius is not None and domain_center_xy is not None:
                    tip_xy = tip_pos[:2] - domain_center_xy
                    tip_r = np.linalg.norm(tip_xy)
                    allowed_r = envelope_r_frac * domain_radius
                    
                    if tip_r > allowed_r * 0.8:
                        outward_dir = tip_xy / (tip_r + 1e-12)
                        child_xy_component = child_dir[:2]
                        outward_component = np.dot(child_xy_component, outward_dir)
                        
                        if outward_component > 0:
                            child_dir[:2] -= 0.5 * outward_component * outward_dir
                            child_dir = child_dir / (np.linalg.norm(child_dir) + 1e-12)
                
                success = False
                final_child_end = None
                final_child_dir = None
                final_child_length = child_length
                
                for attempt in range(max(1, spec.reaim_attempts if spec.enable_reaim_instead_of_skip else 1)):
                    attempt_angle_rad = angle_rad * (spec.reaim_angle_shrink ** attempt)
                    attempt_length = child_length * (spec.reaim_length_shrink ** attempt)
                    
                    if attempt > 0:
                        jitter = rng.uniform(-np.radians(spec.collision_azimuth_jitter_deg), 
                                            np.radians(spec.collision_azimuth_jitter_deg))
                        attempt_azimuth = child_azimuth + jitter
                        lateral = np.cos(attempt_azimuth) * perp1 + np.sin(attempt_azimuth) * perp2
                        attempt_dir = np.cos(attempt_angle_rad) * tip_dir_norm + np.sin(attempt_angle_rad) * lateral
                        attempt_dir = attempt_dir / (np.linalg.norm(attempt_dir) + 1e-12)
                        
                        if spec.enforce_downward_bias and attempt_dir[2] > spec.downward_bias_min_dz:
                            attempt_dir[2] = spec.downward_bias_min_dz
                            attempt_dir = attempt_dir / (np.linalg.norm(attempt_dir) + 1e-12)
                    else:
                        attempt_dir = child_dir
                    
                    if hasattr(network, 'domain') and network.domain is not None:
                        max_tube_radius = max(tip_radius, child_radius)
                        safe_length = _compute_tube_safe_step_length(
                            start=tip_pos,
                            direction=attempt_dir,
                            tube_radius=max_tube_radius + spec.wall_margin_m,
                            domain=network.domain,
                            requested_length=attempt_length,
                        )
                        
                        safe_length = safe_length * spec.boundary_safety
                        
                        if safe_length < spec.min_segment_length_m:
                            continue
                        
                        attempt_length = safe_length
                    
                    attempt_end = tip_pos + attempt_length * attempt_dir
                    
                    if spec.enable_soft_collision and collision_samples:
                        collision_ok = True
                        for coll_attempt in range(spec.collision_attempts_per_child):
                            if coll_attempt > 0:
                                jitter = rng.uniform(-np.radians(spec.collision_azimuth_jitter_deg),
                                                    np.radians(spec.collision_azimuth_jitter_deg))
                                jittered_azimuth = child_azimuth + jitter
                                lateral = np.cos(jittered_azimuth) * perp1 + np.sin(jittered_azimuth) * perp2
                                jittered_dir = np.cos(attempt_angle_rad) * tip_dir_norm + np.sin(attempt_angle_rad) * lateral
                                jittered_dir = jittered_dir / (np.linalg.norm(jittered_dir) + 1e-12)
                                attempt_end = tip_pos + attempt_length * jittered_dir
                                attempt_dir = jittered_dir
                            
                            has_collision = _check_soft_collision(
                                attempt_end, child_radius, collision_samples, spec.collision_clearance_factor
                            )
                            
                            if not has_collision:
                                collision_ok = True
                                break
                            else:
                                collision_ok = False
                        
                        if not collision_ok:
                            continue
                    
                    success = True
                    final_child_end = attempt_end
                    final_child_dir = attempt_dir
                    final_child_length = attempt_length
                    break
                
                if not success:
                    all_warnings.append(
                        f"Level {level}, child {k}: could not place branch after {spec.reaim_attempts} attempts"
                    )
                    continue
                
                child_end_pt = Point3D(final_child_end[0], final_child_end[1], final_child_end[2])
                child_dir_obj = Direction3D(final_child_dir[0], final_child_dir[1], final_child_dir[2])
                
                new_node_id = network.id_gen.next_id()
                new_azimuth_pref = child_azimuth
                
                new_node = Node(
                    id=new_node_id,
                    position=child_end_pt,
                    node_type="terminal",
                    vessel_type=root_node.vessel_type,
                    attributes={
                        "radius": child_radius,
                        "direction": child_dir_obj.to_dict(),
                        "branch_order": level + 1,
                        "tree_level": level,
                        "is_leader": is_leader,
                        "azimuth_pref": new_azimuth_pref,
                    },
                )
                
                segment_id = network.id_gen.next_id()
                geometry = TubeGeometry(
                    start=Point3D(tip_pos[0], tip_pos[1], tip_pos[2]),
                    end=child_end_pt,
                    radius_start=tip_radius,
                    radius_end=child_radius,
                )
                
                segment = VesselSegment(
                    id=segment_id,
                    start_node_id=tip_node_id,
                    end_node_id=new_node_id,
                    geometry=geometry,
                    vessel_type=root_node.vessel_type,
                )
                
                network.add_node(new_node)
                network.add_segment(segment)
                
                all_node_ids.append(new_node_id)
                all_segment_ids.append(segment_id)
                
                new_tips.append((new_node_id, final_child_end, final_child_dir, child_radius, level + 1, new_azimuth_pref))
            
            tip_node = network.get_node(tip_node_id)
            if tip_node is not None and tip_node.node_type == "terminal":
                tip_node.node_type = "junction"
        
        tips = new_tips
    
    delta = Delta(
        created_node_ids=all_node_ids,
        created_segment_ids=all_segment_ids,
    )
    
    status = OperationStatus.SUCCESS if not all_warnings else OperationStatus.PARTIAL_SUCCESS
    
    return OperationResult(
        status=status,
        message=f"Grew {spec.K}-ary tree (v2) with {num_levels} levels from node {root_node_id}",
        new_ids={
            "nodes": all_node_ids,
            "segments": all_segment_ids,
        },
        warnings=all_warnings,
        delta=delta,
        rng_state=network.id_gen.get_state(),
    )
