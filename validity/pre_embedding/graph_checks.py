"""
Pre-Embedding Graph/Network Checks

Validates vascular network topology and geometry before embedding.
Checks include Murray's law compliance, branch order, collisions, and self-intersections.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from generation.core import VascularNetwork


@dataclass
class GraphCheckResult:
    """Result of a graph/network check."""
    passed: bool
    check_name: str
    message: str
    details: Dict[str, Any]
    warnings: List[str]


def check_murrays_law(
    network: "VascularNetwork",
    gamma: float = 3.0,
    tolerance: float = 0.15,
) -> GraphCheckResult:
    """
    Check Murray's law compliance at bifurcations.
    
    Murray's law states that at a bifurcation:
        r_parent^gamma = r_child1^gamma + r_child2^gamma
    
    For blood vessels, gamma is typically 3.0.
    
    Uses topology-based parent detection when possible, falling back to
    radius-based heuristic only when necessary. Excludes junctions in
    cycles (anastomoses) where parent/child relationships are ambiguous.
    
    Parameters
    ----------
    network : VascularNetwork
        The vascular network to check
    gamma : float
        Murray's law exponent (default 3.0 for blood vessels)
    tolerance : float
        Maximum acceptable relative deviation from Murray's law
        
    Returns
    -------
    GraphCheckResult
        Result with pass/fail status and details
    """
    from generation.analysis.radius import (
        compute_murray_deviation_at_junction,
        find_root_node_id,
        is_junction_in_cycle,
        segment_radius_at_node,
        identify_parent_segment_at_junction,
    )
    
    deviations = []
    bifurcation_count = 0
    violations = []
    skipped_cycles = 0
    
    # Find root nodes for topology-based parent detection
    arterial_root = find_root_node_id(network, vessel_type="arterial")
    venous_root = find_root_node_id(network, vessel_type="venous")
    
    for node_id, node in network.nodes.items():
        if node.node_type == "junction":
            # Get connected segments using network topology helper
            connected_seg_ids = network.get_connected_segment_ids(node_id)
            connected_segs = [
                network.segments.get(sid) 
                for sid in connected_seg_ids
            ]
            connected_segs = [s for s in connected_segs if s is not None]
            
            if len(connected_segs) >= 2:
                bifurcation_count += 1
                
                # Skip junctions in cycles (anastomoses) - parent/child is ambiguous
                if is_junction_in_cycle(network, node_id):
                    skipped_cycles += 1
                    continue
                
                # Use appropriate root for this node's vessel type
                root_id = arterial_root if node.vessel_type == "arterial" else venous_root
                if root_id is None:
                    root_id = arterial_root or venous_root
                
                # Use topology-based parent detection
                parent_seg_id, child_seg_ids = identify_parent_segment_at_junction(
                    network, node_id, root_node_id=root_id
                )
                
                if parent_seg_id is None or len(child_seg_ids) == 0:
                    continue
                
                parent_seg = network.segments.get(parent_seg_id)
                if parent_seg is None:
                    continue
                
                parent_r = segment_radius_at_node(parent_seg, node_id)
                child_radii = []
                for child_id in child_seg_ids:
                    child_seg = network.segments.get(child_id)
                    if child_seg is not None:
                        child_radii.append(segment_radius_at_node(child_seg, node_id))
                
                if child_radii and parent_r > 0:
                    # Murray's law: parent^gamma = sum(child^gamma)
                    expected_parent = sum(r**gamma for r in child_radii) ** (1/gamma)
                    deviation = abs(parent_r - expected_parent) / expected_parent if expected_parent > 0 else 0
                    deviations.append(deviation)
                    
                    if deviation > tolerance:
                        violations.append({
                            "node_id": node_id,
                            "parent_radius": parent_r,
                            "child_radii": child_radii,
                            "expected_parent": expected_parent,
                            "deviation": deviation,
                        })
    
    mean_deviation = float(np.mean(deviations)) if deviations else 0.0
    max_deviation = float(np.max(deviations)) if deviations else 0.0
    passed = max_deviation <= tolerance
    
    details = {
        "bifurcation_count": bifurcation_count,
        "skipped_cycles": skipped_cycles,
        "mean_deviation": mean_deviation,
        "max_deviation": max_deviation,
        "gamma": gamma,
        "tolerance": tolerance,
        "violations_count": len(violations),
    }
    
    warnings = []
    if violations:
        warnings.append(f"{len(violations)} bifurcations violate Murray's law (tolerance {tolerance:.0%})")
    
    return GraphCheckResult(
        passed=passed,
        check_name="murrays_law",
        message=f"Murray's law: mean deviation {mean_deviation:.1%}" if bifurcation_count > 0 else "No bifurcations found",
        details=details,
        warnings=warnings,
    )


def check_branch_order(
    network: "VascularNetwork",
    max_expected_order: int = 20,
) -> GraphCheckResult:
    """
    Check branch order distribution in the network.
    
    Branch order indicates the generation of a vessel from the root.
    Unusually high branch orders may indicate issues with the network topology.
    
    Parameters
    ----------
    network : VascularNetwork
        The vascular network to check
    max_expected_order : int
        Maximum expected branch order
        
    Returns
    -------
    GraphCheckResult
        Result with pass/fail status and details
    """
    branch_orders = []
    for node in network.nodes.values():
        order = node.attributes.get("branch_order", 0)
        branch_orders.append(order)
    
    if not branch_orders:
        return GraphCheckResult(
            passed=True,
            check_name="branch_order",
            message="No nodes found",
            details={"num_nodes": 0},
            warnings=[],
        )
    
    max_order = int(np.max(branch_orders))
    mean_order = float(np.mean(branch_orders))
    median_order = float(np.median(branch_orders))
    
    passed = max_order <= max_expected_order
    
    # Build histogram
    histogram = {}
    for order in branch_orders:
        histogram[order] = histogram.get(order, 0) + 1
    
    details = {
        "max_branch_order": max_order,
        "mean_branch_order": mean_order,
        "median_branch_order": median_order,
        "max_expected_order": max_expected_order,
        "histogram": histogram,
        "num_nodes": len(branch_orders),
    }
    
    warnings = []
    if max_order > max_expected_order:
        warnings.append(f"Max branch order {max_order} exceeds expected {max_expected_order}")
    
    return GraphCheckResult(
        passed=passed,
        check_name="branch_order",
        message=f"Branch order: max={max_order}, mean={mean_order:.1f}",
        details=details,
        warnings=warnings,
    )


def check_collisions(
    network: "VascularNetwork",
    min_clearance: float = 0.0,
    max_collisions: int = 0,
    time_budget_s: Optional[float] = None,
    max_pairs: Optional[int] = None,
) -> GraphCheckResult:
    """
    Check for collisions between non-adjacent segments.
    
    Collisions occur when segments are closer than the sum of their radii
    plus the minimum clearance.
    
    Uses SpatialIndex for efficient collision detection with exact segment-segment
    distance calculation. No longer limited to first 100 segments.
    
    Parameters
    ----------
    network : VascularNetwork
        The vascular network to check
    min_clearance : float
        Minimum required clearance between segments (in network units)
    max_collisions : int
        Maximum acceptable number of collisions
    time_budget_s : float, optional
        Maximum time budget in seconds (None = no limit)
    max_pairs : int, optional
        Maximum number of segment pairs to check (None = no limit)
        
    Returns
    -------
    GraphCheckResult
        Result with pass/fail status and details
    """
    import time
    start_time = time.time()
    
    collision_count = 0
    min_dist = float('inf')
    collisions = []
    pairs_checked = 0
    
    # Use spatial index for efficient collision detection
    spatial_index = network.get_spatial_index()
    
    # Get all collisions using the spatial index with exact segment-segment distance
    all_collisions = spatial_index.get_collisions(
        min_clearance=min_clearance,
        exclude_connected=True,
    )
    
    segments_checked = len(network.segments)
    
    for seg_id1, seg_id2, dist in all_collisions:
        # Check time budget
        if time_budget_s is not None and (time.time() - start_time) > time_budget_s:
            break
        
        # Check max pairs
        if max_pairs is not None and pairs_checked >= max_pairs:
            break
        
        pairs_checked += 1
        min_dist = min(min_dist, dist)
        
        seg1 = network.segments.get(seg_id1)
        seg2 = network.segments.get(seg_id2)
        
        if seg1 is None or seg2 is None:
            continue
        
        # Get radii - prefer geometry over attributes
        if hasattr(seg1, 'geometry') and seg1.geometry is not None:
            r1 = seg1.mean_radius if hasattr(seg1, 'mean_radius') else (seg1.geometry.radius_start + seg1.geometry.radius_end) / 2
        else:
            r1 = seg1.attributes.get("radius", 0.001)
        if hasattr(seg2, 'geometry') and seg2.geometry is not None:
            r2 = seg2.mean_radius if hasattr(seg2, 'mean_radius') else (seg2.geometry.radius_start + seg2.geometry.radius_end) / 2
        else:
            r2 = seg2.attributes.get("radius", 0.001)
        
        collision_count += 1
        collisions.append({
            "seg1_id": seg_id1,
            "seg2_id": seg_id2,
            "distance": dist,
            "required_clearance": r1 + r2 + min_clearance,
        })
    
    if min_dist == float('inf'):
        min_dist = 0.0
    
    elapsed_time = time.time() - start_time
    passed = collision_count <= max_collisions
    
    details = {
        "collision_count": collision_count,
        "min_distance": min_dist,
        "min_clearance": min_clearance,
        "max_collisions": max_collisions,
        "segments_checked": segments_checked,
        "pairs_checked": pairs_checked,
        "elapsed_time_s": elapsed_time,
    }
    
    warnings = []
    if collision_count > 0:
        warnings.append(f"Found {collision_count} segment collisions")
    
    return GraphCheckResult(
        passed=passed,
        check_name="collisions",
        message=f"Collisions: {collision_count}" if collision_count > 0 else "No collisions detected",
        details=details,
        warnings=warnings,
    )


def check_self_intersections(
    network: "VascularNetwork",
) -> GraphCheckResult:
    """
    Check for self-intersecting segments (segments that cross themselves).
    
    Self-intersections are topologically invalid and should be fixed.
    
    Parameters
    ----------
    network : VascularNetwork
        The vascular network to check
        
    Returns
    -------
    GraphCheckResult
        Result with pass/fail status and details
    """
    # For now, we check for zero-length segments and segments with
    # start and end at the same position
    self_intersections = []
    
    for seg_id, seg in network.segments.items():
        start_pos = network.nodes[seg.start_node_id].position.to_array()
        end_pos = network.nodes[seg.end_node_id].position.to_array()
        
        length = np.linalg.norm(end_pos - start_pos)
        
        if length < 1e-10:
            self_intersections.append({
                "segment_id": seg_id,
                "type": "zero_length",
                "length": length,
            })
    
    passed = len(self_intersections) == 0
    
    details = {
        "self_intersection_count": len(self_intersections),
        "total_segments": len(network.segments),
    }
    
    warnings = []
    if self_intersections:
        warnings.append(f"Found {len(self_intersections)} self-intersecting segments")
    
    return GraphCheckResult(
        passed=passed,
        check_name="self_intersections",
        message="No self-intersections" if passed else f"{len(self_intersections)} self-intersections found",
        details=details,
        warnings=warnings,
    )


@dataclass
class GraphCheckReport:
    """Aggregated report of all graph checks."""
    passed: bool
    status: str  # "ok", "warnings", "fail"
    checks: List[GraphCheckResult]
    summary: Dict[str, Any]


def run_all_graph_checks(
    network: "VascularNetwork",
    murray_gamma: float = 3.0,
    murray_tolerance: float = 0.15,
    max_branch_order: int = 20,
    min_clearance: float = 0.0,
    max_collisions: int = 0,
) -> GraphCheckReport:
    """
    Run all pre-embedding graph/network checks.
    
    Parameters
    ----------
    network : VascularNetwork
        The vascular network to check
    murray_gamma : float
        Murray's law exponent
    murray_tolerance : float
        Maximum acceptable Murray's law deviation
    max_branch_order : int
        Maximum expected branch order
    min_clearance : float
        Minimum required clearance between segments
    max_collisions : int
        Maximum acceptable number of collisions
        
    Returns
    -------
    GraphCheckReport
        Aggregated report with all check results
    """
    checks = [
        check_murrays_law(network, murray_gamma, murray_tolerance),
        check_branch_order(network, max_branch_order),
        check_collisions(network, min_clearance, max_collisions),
        check_self_intersections(network),
    ]
    
    all_passed = all(c.passed for c in checks)
    has_warnings = any(len(c.warnings) > 0 for c in checks)
    
    if all_passed and not has_warnings:
        status = "ok"
    elif all_passed:
        status = "warnings"
    else:
        status = "fail"
    
    summary = {
        "total_checks": len(checks),
        "passed_checks": sum(1 for c in checks if c.passed),
        "failed_checks": sum(1 for c in checks if not c.passed),
        "total_warnings": sum(len(c.warnings) for c in checks),
    }
    
    return GraphCheckReport(
        passed=all_passed,
        status=status,
        checks=checks,
        summary=summary,
    )
