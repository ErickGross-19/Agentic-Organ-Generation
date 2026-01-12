"""Network evaluation API for quality assessment."""

from typing import Optional, Dict, Any, Union
import numpy as np
from dataclasses import dataclass

from ..specs.eval_result import (
    EvalResult, CoverageMetrics, FlowMetrics, StructureMetrics,
    ValidityMetrics, EvalScores
)
from ..core.network import VascularNetwork
from ..analysis.solver import solve_flow, compute_component_flows


@dataclass
class EvalConfig:
    """Configuration for network evaluation."""
    coverage_weight: float = 0.4
    flow_weight: float = 0.4
    structure_weight: float = 0.2
    reynolds_turbulent_threshold: float = 2300.0
    murray_tolerance: float = 0.15
    use_segment_distance: bool = True
    enable_perfusion: bool = False
    perfusion_distance_cap: Optional[float] = None
    perfusion_weights: tuple = (1.0, 1.0)
    well_perfused_threshold: float = 0.5
    perfusion_threshold: float = 0.005
    enable_cfd: bool = False
    cfd_fidelity: str = "0D"
    cfd_output_dir: Optional[str] = None


def evaluate_network(
    network: VascularNetwork,
    tissue_points: Union[np.ndarray, int],
    config: Optional[EvalConfig] = None,
) -> EvalResult:
    """
    Evaluate vascular network quality with comprehensive metrics.
    
    Uses segment-based distance calculations by default for more accurate
    coverage assessment. Optionally includes perfusion analysis for
    dual-tree networks.
    """
    if config is None:
        config = EvalConfig()
    
    if isinstance(tissue_points, int):
        if network.domain is None:
            raise ValueError("Network must have a domain to generate tissue points")
        tissue_points = network.domain.sample_points(n_points=tissue_points)
    
    coverage = _compute_coverage_metrics(network, tissue_points, config)
    flow = _compute_flow_metrics(network, config)
    structure = _compute_structure_metrics(network, config)
    validity = _compute_validity_metrics(network)
    scores = _compute_scores(coverage, flow, structure, config)
    
    metadata = {
        "num_tissue_points": len(tissue_points),
        "config": {
            "coverage_weight": config.coverage_weight,
            "flow_weight": config.flow_weight,
            "structure_weight": config.structure_weight,
            "use_segment_distance": config.use_segment_distance,
        },
    }
    
    if config.enable_perfusion:
        perfusion_metrics = _compute_perfusion_metrics(network, tissue_points, config)
        metadata["perfusion"] = perfusion_metrics
    
    if config.enable_cfd:
        cfd_metrics = _compute_cfd_metrics(network, config)
        metadata["cfd"] = cfd_metrics
    
    return EvalResult(
        coverage=coverage, flow=flow, structure=structure,
        validity=validity, scores=scores,
        metadata=metadata,
    )


def _compute_perfusion_metrics(
    network: VascularNetwork,
    tissue_points: np.ndarray,
    config: EvalConfig,
) -> Dict[str, Any]:
    """
    Compute perfusion metrics using segment-based distances.
    
    Analyzes tissue perfusion based on proximity to both arterial (supply)
    and venous (drainage) vessels.
    """
    from ..analysis.perfusion import compute_perfusion_metrics_segment_based
    
    return compute_perfusion_metrics_segment_based(
        network=network,
        tissue_points=tissue_points,
        weights=config.perfusion_weights,
        distance_cap=config.perfusion_distance_cap,
        well_perfused_threshold=config.well_perfused_threshold,
    )


def _compute_cfd_metrics(
    network: VascularNetwork,
    config: EvalConfig,
) -> Dict[str, Any]:
    """
    Compute CFD metrics using the CFD pipeline.
    
    Runs hemodynamic simulation at the specified fidelity level
    and returns flow/pressure metrics.
    """
    from ..cfd.pipeline import run_cfd_pipeline, CFDConfig
    
    cfd_config = CFDConfig(
        fidelity=config.cfd_fidelity,
        output_dir=config.cfd_output_dir,
    )
    
    result = run_cfd_pipeline(network, cfd_config)
    
    return {
        "success": result.success,
        "fidelity": result.fidelity,
        "pressure_drop": result.metrics.pressure.pressure_drop_root_to_terminals,
        "flow_uniformity": result.metrics.flow.flow_uniformity,
        "perfusable_fraction": result.metrics.perfusion.perfusable_fraction,
        "wall_time_seconds": result.wall_time_seconds,
        "warnings": result.warnings,
        "errors": result.errors,
    }


def _compute_coverage_metrics(
    network: VascularNetwork,
    tissue_points: np.ndarray,
    config: Optional[EvalConfig] = None,
) -> CoverageMetrics:
    """
    Compute coverage and perfusion metrics using segment-based distances.
    
    Uses distance to vessel surface (centerline - radius) rather than
    distance to nodes for more accurate coverage assessment.
    """
    if config is None:
        config = EvalConfig()
    
    if len(tissue_points) == 0:
        return CoverageMetrics(0.0, 0, 0.0, 0.0, 0.0)
    
    perfusion_threshold = config.perfusion_threshold
    
    if config.use_segment_distance:
        from ..analysis.distance import compute_tissue_coverage_distances
        
        coverage_result = compute_tissue_coverage_distances(
            tissue_points, network, vessel_type=None, use_surface_distance=True
        )
        distances = coverage_result["distances"]
    else:
        distances = []
        for tp in tissue_points:
            min_dist = float('inf')
            for node in network.nodes.values():
                dist = np.linalg.norm(node.position.to_array() - tp)
                min_dist = min(min_dist, dist)
            distances.append(min_dist)
        distances = np.array(distances)
    
    perfused = distances < perfusion_threshold
    coverage_fraction = float(np.mean(perfused))
    unperfused_points = int(np.sum(~perfused))
    
    if len(distances) > 1 and np.mean(distances) > 0:
        cv = np.std(distances) / np.mean(distances)
        perfusion_uniformity = float(1.0 / (1.0 + cv))
    else:
        perfusion_uniformity = 1.0
    
    return CoverageMetrics(
        coverage_fraction=coverage_fraction,
        unperfused_points=unperfused_points,
        perfusion_uniformity=perfusion_uniformity,
        mean_distance_to_vessel=float(np.mean(distances)),
        max_distance_to_vessel=float(np.max(distances)),
    )


def _compute_flow_metrics(network: VascularNetwork, config: EvalConfig) -> FlowMetrics:
    """Compute hemodynamic flow metrics."""
    try:
        component_flows = compute_component_flows(network)
        
        total_flow_art = 0.0
        total_flow_ven = 0.0
        pressure_drop_art = 0.0
        pressure_drop_ven = 0.0
        
        for comp_id, flow_data in component_flows.items():
            if flow_data["vessel_type"] == "arterial":
                total_flow_art += flow_data["total_flow"]
                pressure_drop_art = max(pressure_drop_art, flow_data.get("pressure_drop", 0.0))
            elif flow_data["vessel_type"] == "venous":
                total_flow_ven += flow_data["total_flow"]
                pressure_drop_ven = max(pressure_drop_ven, flow_data.get("pressure_drop", 0.0))
        
        flow_balance_error = abs(total_flow_art - total_flow_ven) / total_flow_art if total_flow_art > 0 else 1.0
        
        pressures = []
        reynolds_numbers = []
        
        for seg in network.segments.values():
            if "pressure_start" in seg.attributes:
                pressures.append(seg.attributes["pressure_start"])
            if "pressure_end" in seg.attributes:
                pressures.append(seg.attributes["pressure_end"])
            if "reynolds" in seg.attributes:
                reynolds_numbers.append(seg.attributes["reynolds"])
        
        if pressures:
            min_pressure = float(np.min(pressures))
            mean_pressure = float(np.mean(pressures))
            max_pressure = float(np.max(pressures))
        else:
            min_pressure = mean_pressure = max_pressure = 0.0
        
        if reynolds_numbers:
            turbulent = np.array(reynolds_numbers) > config.reynolds_turbulent_threshold
            turbulent_fraction = float(np.mean(turbulent))
            max_reynolds = float(np.max(reynolds_numbers))
        else:
            turbulent_fraction = 0.0
            max_reynolds = 0.0
        
        return FlowMetrics(
            total_flow_arterial=total_flow_art, total_flow_venous=total_flow_ven,
            flow_balance_error=flow_balance_error, min_pressure=min_pressure,
            mean_pressure=mean_pressure, max_pressure=max_pressure,
            turbulent_fraction=turbulent_fraction, max_reynolds=max_reynolds,
            pressure_drop_arterial=pressure_drop_art, pressure_drop_venous=pressure_drop_ven,
        )
    except Exception as e:
        return FlowMetrics(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _compute_structure_metrics(network: VascularNetwork, config: EvalConfig) -> StructureMetrics:
    """Compute structural and topological metrics."""
    num_nodes = len(network.nodes)
    num_segments = len(network.segments)
    num_terminals = sum(1 for n in network.nodes.values() if n.node_type == "terminal")
    total_length = sum(seg.length for seg in network.segments.values())
    
    branch_orders = [n.attributes.get("branch_order", 0) for n in network.nodes.values()]
    if branch_orders:
        mean_branch_order = float(np.mean(branch_orders))
        median_branch_order = float(np.median(branch_orders))
        max_branch_order = int(np.max(branch_orders))
    else:
        mean_branch_order = median_branch_order = 0.0
        max_branch_order = 0
    
    degree_histogram = {}
    for node in network.nodes.values():
        connected_seg_ids = network.get_connected_segment_ids(node.id)
        degree = len(connected_seg_ids)
        degree_histogram[degree] = degree_histogram.get(degree, 0) + 1
    
    branching_angles = []
    for node in network.nodes.values():
        connected_seg_ids = network.get_connected_segment_ids(node.id)
        if node.node_type == "junction" and len(connected_seg_ids) >= 2:
            seg_ids = list(connected_seg_ids)
            for i in range(len(seg_ids)):
                for j in range(i + 1, len(seg_ids)):
                    seg1 = network.segments.get(seg_ids[i])
                    seg2 = network.segments.get(seg_ids[j])
                    if seg1 and seg2:
                        dir1 = seg1.direction.to_array()
                        dir2 = seg2.direction.to_array()
                        cos_angle = np.clip(np.dot(dir1, dir2), -1.0, 1.0)
                        angle = np.degrees(np.arccos(abs(cos_angle)))
                        branching_angles.append(angle)
    
    mean_branching_angle = float(np.mean(branching_angles)) if branching_angles else 0.0
    
    from ..analysis.radius import (
        compute_murray_deviation_at_junction,
        find_root_node_id,
        is_junction_in_cycle,
    )
    
    murray_deviations = []
    
    # Find root nodes for topology-based parent detection
    arterial_root = find_root_node_id(network, vessel_type="arterial")
    venous_root = find_root_node_id(network, vessel_type="venous")
    
    for node in network.nodes.values():
        if node.node_type == "junction":
            # Skip junctions in cycles (anastomoses) - parent/child is ambiguous
            if is_junction_in_cycle(network, node.id):
                continue
            
            # Use appropriate root for this node's vessel type
            root_id = arterial_root if node.vessel_type == "arterial" else venous_root
            if root_id is None:
                root_id = arterial_root or venous_root
            
            deviation = compute_murray_deviation_at_junction(
                network, node.id,
                gamma=3.0,
                root_node_id=root_id,
            )
            
            if deviation is not None:
                murray_deviations.append(deviation)
    
    murray_deviation = float(np.mean(murray_deviations)) if murray_deviations else 0.0
    
    # Use SpatialIndex for collision detection (consistent with validity checks)
    collision_count = 0
    min_clearance = float('inf')
    try:
        spatial_index = network.get_spatial_index()
        # Get all collisions with zero clearance (actual intersections)
        collisions = spatial_index.get_collisions(
            min_clearance=0.0,
            exclude_connected=True,
        )
        collision_count = len(collisions)
        
        # Compute min clearance from collision pairs
        for seg1_id, seg2_id, clearance in collisions:
            min_clearance = min(min_clearance, clearance)
        
        # If no collisions, estimate min clearance from nearby segment pairs
        if min_clearance == float('inf') and network.segments:
            # Sample a few segments and find their nearest non-connected neighbors
            from ..analysis.radius import segment_mean_radius
            from ..core.types import Point3D
            seg_list = list(network.segments.values())[:50]
            for seg in seg_list:
                # Compute midpoint manually (TubeGeometry doesn't have midpoint property)
                start_pos = network.nodes[seg.start_node_id].position
                end_pos = network.nodes[seg.end_node_id].position
                midpoint = Point3D(
                    (start_pos.x + end_pos.x) / 2,
                    (start_pos.y + end_pos.y) / 2,
                    (start_pos.z + end_pos.z) / 2,
                )
                nearby = spatial_index.query_nearby_segments(midpoint, 0.01)
                for other_seg in nearby:
                    if other_seg.id == seg.id:
                        continue
                    if (seg.start_node_id in (other_seg.start_node_id, other_seg.end_node_id) or
                        seg.end_node_id in (other_seg.start_node_id, other_seg.end_node_id)):
                        continue
                    # Compute centerline distance
                    from ..spatial.grid_index import segment_segment_distance_exact
                    p1 = network.nodes[seg.start_node_id].position.to_array()
                    p2 = network.nodes[seg.end_node_id].position.to_array()
                    p3 = network.nodes[other_seg.start_node_id].position.to_array()
                    p4 = network.nodes[other_seg.end_node_id].position.to_array()
                    dist = segment_segment_distance_exact(p1, p2, p3, p4)
                    r1 = segment_mean_radius(seg)
                    r2 = segment_mean_radius(other_seg)
                    clearance = dist - r1 - r2
                    min_clearance = min(min_clearance, max(0.0, clearance))
    except Exception:
        # Fallback to simple estimation if SpatialIndex fails
        pass
    
    if min_clearance == float('inf'):
        min_clearance = 0.0
    
    return StructureMetrics(
        total_length=total_length, num_nodes=num_nodes, num_segments=num_segments,
        num_terminals=num_terminals, mean_branch_order=mean_branch_order,
        median_branch_order=median_branch_order, max_branch_order=max_branch_order,
        degree_histogram=degree_histogram, mean_branching_angle=mean_branching_angle,
        murray_deviation=murray_deviation, collision_count=collision_count,
        min_clearance=min_clearance,
    )


def _compute_validity_metrics(network: VascularNetwork) -> ValidityMetrics:
    """
    Compute validity and quality checks.
    
    Performs actual checks for:
    - Self-intersections (collisions between non-adjacent segments)
    - Watertight geometry (all segments connected, no dangling nodes)
    - Parameter warnings (very small radii, very short segments)
    """
    from ..analysis.radius import segment_mean_radius
    
    parameter_warnings = []
    error_codes = []
    
    # Check for self-intersections using SpatialIndex collision detection
    has_self_intersections = False
    try:
        spatial_index = network.get_spatial_index()
        collisions = spatial_index.get_collisions(
            min_clearance=0.0,  # Just check for actual intersections
            exclude_connected=True,  # Exclude segments that share a node
        )
        if collisions:
            has_self_intersections = True
            error_codes.append(f"COLLISION_COUNT_{len(collisions)}")
    except Exception as e:
        # If collision check fails, report as unknown
        parameter_warnings.append(f"Could not check collisions: {str(e)}")
    
    # Check for watertight geometry (all nodes connected, no isolated components)
    is_watertight = True
    try:
        # Check for dangling nodes (nodes with only one connection that aren't terminals/inlets/outlets)
        for node in network.nodes.values():
            connected_segs = network.get_connected_segment_ids(node.id)
            if len(connected_segs) == 0:
                is_watertight = False
                error_codes.append(f"ISOLATED_NODE_{node.id}")
            elif len(connected_segs) == 1 and node.node_type not in ("terminal", "inlet", "outlet"):
                is_watertight = False
                error_codes.append(f"DANGLING_NODE_{node.id}")
        
        # Check for disconnected components (simple connectivity check)
        if network.nodes:
            visited = set()
            start_node = next(iter(network.nodes.keys()))
            to_visit = [start_node]
            while to_visit:
                node_id = to_visit.pop()
                if node_id in visited:
                    continue
                visited.add(node_id)
                for seg_id in network.get_connected_segment_ids(node_id):
                    seg = network.segments.get(seg_id)
                    if seg:
                        other_id = seg.end_node_id if seg.start_node_id == node_id else seg.start_node_id
                        if other_id not in visited:
                            to_visit.append(other_id)
            
            if len(visited) < len(network.nodes):
                is_watertight = False
                error_codes.append(f"DISCONNECTED_COMPONENTS_{len(network.nodes) - len(visited)}_UNREACHABLE")
    except Exception as e:
        is_watertight = None  # Unknown
        parameter_warnings.append(f"Could not check watertight: {str(e)}")
    
    # Check for parameter warnings (very small radii, very short segments)
    for seg in network.segments.values():
        r = segment_mean_radius(seg)
        if r < 0.0001:  # 0.1mm
            parameter_warnings.append(f"Segment {seg.id} has very small radius: {r:.6f}m")
            break
    
    for seg in network.segments.values():
        if seg.length < 0.0001:  # 0.1mm
            parameter_warnings.append(f"Segment {seg.id} is very short: {seg.length:.6f}m")
            break
    
    return ValidityMetrics(
        is_watertight=is_watertight if is_watertight is not None else False,
        has_self_intersections=has_self_intersections,
        parameter_warnings=parameter_warnings,
        error_codes=error_codes,
    )


def _compute_scores(
    coverage: CoverageMetrics,
    flow: FlowMetrics,
    structure: StructureMetrics,
    config: EvalConfig,
) -> EvalScores:
    """Compute normalized quality scores (0-1, higher is better)."""
    coverage_score = 0.7 * coverage.coverage_fraction + 0.3 * coverage.perfusion_uniformity
    flow_score = max(0.0, 1.0 - flow.flow_balance_error - 0.1 * flow.turbulent_fraction)
    structure_score = max(0.0, 1.0 - structure.murray_deviation - 0.1 * min(structure.collision_count / 10.0, 1.0))
    overall_score = (
        config.coverage_weight * coverage_score +
        config.flow_weight * flow_score +
        config.structure_weight * structure_score
    )
    
    return EvalScores(
        coverage_score=coverage_score,
        flow_score=flow_score,
        structure_score=structure_score,
        overall_score=overall_score,
    )
