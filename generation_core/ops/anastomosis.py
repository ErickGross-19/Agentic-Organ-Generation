"""
Anastomosis operations for connecting arterial and venous trees.
"""

from typing import Optional
import numpy as np
from ..core.network import VascularNetwork, Node, VesselSegment
from ..core.types import Point3D, TubeGeometry
from ..core.result import OperationResult, OperationStatus, ErrorCode, Delta
from ..rules.constraints import InteractionRuleSpec


def create_anastomosis(
    network: VascularNetwork,
    arterial_node_id: int,
    venous_node_id: int,
    rules: Optional[InteractionRuleSpec] = None,
    radius: Optional[float] = None,
    max_length: float = 0.010,  # 10mm default max capillary length
    dry_run: bool = False,
) -> OperationResult:
    """
    Create an anastomosis (capillary connection) between arterial and venous nodes.
    
    This operation creates a small-diameter segment connecting arterial and venous
    trees, representing capillary exchange vessels.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to modify
    arterial_node_id : int
        ID of arterial node
    venous_node_id : int
        ID of venous node
    rules : InteractionRuleSpec, optional
        Interaction rules (default: allow arterial-venous anastomosis)
    radius : float, optional
        Radius of anastomosis segment (default: mean of node radii, capped at 0.0003m)
    max_length : float
        Maximum allowed length for anastomosis (meters)
    dry_run : bool
        If True, validate but don't apply changes
    
    Returns
    -------
    result : OperationResult
        Result with new_ids containing 'segment' if successful
        
    Notes
    -----
    Anastomoses are marked with attributes['segment_kind'] = 'anastomosis' to
    distinguish them from regular vessel segments for flow analysis.
    """
    if rules is None:
        rules = InteractionRuleSpec()
    
    arterial_node = network.get_node(arterial_node_id)
    venous_node = network.get_node(venous_node_id)
    
    if arterial_node is None:
        return OperationResult.failure(
            message=f"Arterial node {arterial_node_id} not found",
            error_codes=[ErrorCode.NODE_NOT_FOUND.value],
        )
    
    if venous_node is None:
        return OperationResult.failure(
            message=f"Venous node {venous_node_id} not found",
            error_codes=[ErrorCode.NODE_NOT_FOUND.value],
        )
    
    if arterial_node.vessel_type not in ["arterial", "generic"]:
        return OperationResult.failure(
            message=f"Node {arterial_node_id} is not arterial (type: {arterial_node.vessel_type})",
            error_codes=[ErrorCode.INCOMPATIBLE_VESSEL_TYPES.value],
        )
    
    if venous_node.vessel_type not in ["venous", "generic"]:
        return OperationResult.failure(
            message=f"Node {venous_node_id} is not venous (type: {venous_node.vessel_type})",
            error_codes=[ErrorCode.INCOMPATIBLE_VESSEL_TYPES.value],
        )
    
    if not rules.is_anastomosis_allowed(arterial_node.vessel_type, venous_node.vessel_type):
        return OperationResult.failure(
            message=f"Anastomosis not allowed between {arterial_node.vessel_type} and {venous_node.vessel_type}",
            error_codes=[ErrorCode.ANASTOMOSIS_NOT_ALLOWED.value],
        )
    
    distance = arterial_node.position.distance_to(venous_node.position)
    
    if distance > max_length:
        return OperationResult.failure(
            message=f"Distance {distance:.4f}m exceeds max_length {max_length:.4f}m",
            error_codes=[ErrorCode.ANASTOMOSIS_TOO_LONG.value],
        )
    
    if radius is None:
        arterial_radius = arterial_node.attributes.get("radius", 0.0005)
        venous_radius = venous_node.attributes.get("radius", 0.0005)
        radius = min((arterial_radius + venous_radius) / 2.0, 0.0003)
    
    if radius < 0.00001 or radius > 0.001:  # 10 microns to 1mm
        return OperationResult.failure(
            message=f"Anastomosis radius {radius:.6f}m out of range [0.00001, 0.001]",
            error_codes=[ErrorCode.ANASTOMOSIS_RADIUS_OUT_OF_RANGE.value],
        )
    
    warnings = []
    
    spatial_index = network.get_spatial_index()
    midpoint = Point3D(
        x=(arterial_node.position.x + venous_node.position.x) / 2,
        y=(arterial_node.position.y + venous_node.position.y) / 2,
        z=(arterial_node.position.z + venous_node.position.z) / 2,
    )
    
    nearby = spatial_index.query_nearby_segments(midpoint, radius * 5.0)
    for seg in nearby:
        if seg.start_node_id in [arterial_node_id, venous_node_id]:
            continue
        if seg.end_node_id in [arterial_node_id, venous_node_id]:
            continue
        
        dist = spatial_index._point_to_segment_distance(midpoint, seg)
        min_clearance = radius + seg.geometry.mean_radius() + 0.0002
        
        if dist < min_clearance:
            warnings.append(
                f"Anastomosis near segment {seg.id} (clearance: {dist:.4f}m, "
                f"min: {min_clearance:.4f}m)"
            )
    
    if dry_run:
        return OperationResult.success(
            message=f"Anastomosis validated (would connect nodes {arterial_node_id} and {venous_node_id})",
            warnings=warnings,
            metadata={
                "distance": distance,
                "radius": radius,
                "dry_run": True,
            },
        )
    
    segment_id = network.id_gen.next_id()
    geometry = TubeGeometry(
        start=arterial_node.position,
        end=venous_node.position,
        radius_start=radius,
        radius_end=radius,
    )
    
    segment = VesselSegment(
        id=segment_id,
        start_node_id=arterial_node_id,
        end_node_id=venous_node_id,
        geometry=geometry,
        vessel_type="capillary",  # Mark as capillary
        attributes={
            "segment_kind": "anastomosis",  # Special marker for flow solver
            "resistance_factor": 100.0,  # High resistance for capillary bed
        },
    )
    
    network.add_segment(segment)
    
    if arterial_node.node_type == "terminal":
        arterial_node.node_type = "junction"
    if venous_node.node_type == "terminal":
        venous_node.node_type = "junction"
    
    delta = Delta(
        created_segment_ids=[segment_id],
    )
    
    status = OperationStatus.SUCCESS if not warnings else OperationStatus.PARTIAL_SUCCESS
    
    return OperationResult(
        status=status,
        message=f"Created anastomosis connecting nodes {arterial_node_id} and {venous_node_id}",
        new_ids={"segment": segment_id},
        warnings=warnings,
        delta=delta,
        metadata={
            "distance": distance,
            "radius": radius,
            "arterial_node": arterial_node_id,
            "venous_node": venous_node_id,
        },
    )


def check_tree_interactions(
    network: VascularNetwork,
    rules: Optional[InteractionRuleSpec] = None,
) -> OperationResult:
    """
    Check for cross-type clearance violations and anastomosis opportunities.
    
    Analyzes the network for:
    - Clearance violations between different vessel types
    - Potential anastomosis locations (close arterial-venous pairs)
    
    Parameters
    ----------
    network : VascularNetwork
        Network to analyze
    rules : InteractionRuleSpec, optional
        Interaction rules for clearance and anastomosis
    
    Returns
    -------
    result : OperationResult
        Result with metadata containing:
        - violations: list of clearance violations
        - anastomosis_candidates: list of potential anastomosis pairs
        - clearance_stats: statistics on cross-type distances
    """
    if rules is None:
        rules = InteractionRuleSpec()
    
    violations = []
    anastomosis_candidates = []
    
    arterial_nodes = [n for n in network.nodes.values() if n.vessel_type == "arterial"]
    venous_nodes = [n for n in network.nodes.values() if n.vessel_type == "venous"]
    
    if not arterial_nodes or not venous_nodes:
        return OperationResult.success(
            message="No multi-tree interaction to check (need both arterial and venous nodes)",
            metadata={
                "violations": [],
                "anastomosis_candidates": [],
                "arterial_count": len(arterial_nodes),
                "venous_count": len(venous_nodes),
            },
        )
    
    min_clearance = rules.get_min_distance("arterial", "venous")
    anastomosis_allowed = rules.is_anastomosis_allowed("arterial", "venous")
    
    distances = []
    for a_node in arterial_nodes:
        for v_node in venous_nodes:
            dist = a_node.position.distance_to(v_node.position)
            distances.append(dist)
            
            if dist < min_clearance:
                violations.append({
                    "arterial_node": a_node.id,
                    "venous_node": v_node.id,
                    "distance": dist,
                    "min_clearance": min_clearance,
                    "violation_amount": min_clearance - dist,
                })
            
            if anastomosis_allowed and 0.002 <= dist <= 0.010:  # 2-10mm range
                a_radius = a_node.attributes.get("radius", 0.0005)
                v_radius = v_node.attributes.get("radius", 0.0005)
                
                if (a_node.node_type == "terminal" or a_radius < 0.001) and \
                   (v_node.node_type == "terminal" or v_radius < 0.001):
                    anastomosis_candidates.append({
                        "arterial_node": a_node.id,
                        "venous_node": v_node.id,
                        "distance": dist,
                        "arterial_radius": a_radius,
                        "venous_radius": v_radius,
                        "score": 1.0 / dist,  # Closer is better
                    })
    
    anastomosis_candidates.sort(key=lambda x: x["score"], reverse=True)
    
    clearance_stats = {
        "mean_distance": float(np.mean(distances)) if distances else 0.0,
        "min_distance": float(np.min(distances)) if distances else 0.0,
        "max_distance": float(np.max(distances)) if distances else 0.0,
        "pairs_checked": len(distances),
    }
    
    warnings = []
    if violations:
        warnings.append(f"Found {len(violations)} clearance violations")
    
    status = OperationStatus.WARNING if violations else OperationStatus.SUCCESS
    error_codes = [ErrorCode.CLEARANCE_VIOLATION.value] if violations else []
    
    return OperationResult(
        status=status,
        message=f"Checked {len(distances)} arterial-venous pairs: "
                f"{len(violations)} violations, {len(anastomosis_candidates)} candidates",
        warnings=warnings,
        error_codes=error_codes,
        metadata={
            "violations": violations,
            "anastomosis_candidates": anastomosis_candidates,
            "clearance_stats": clearance_stats,
            "arterial_count": len(arterial_nodes),
            "venous_count": len(venous_nodes),
        },
    )
