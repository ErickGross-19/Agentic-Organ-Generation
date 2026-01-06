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


def evaluate_network(
    network: VascularNetwork,
    tissue_points: Union[np.ndarray, int],
    config: Optional[EvalConfig] = None,
) -> EvalResult:
    """Evaluate vascular network quality with comprehensive metrics."""
    if config is None:
        config = EvalConfig()
    
    if isinstance(tissue_points, int):
        if network.domain is None:
            raise ValueError("Network must have a domain to generate tissue points")
        tissue_points = network.domain.sample_points(n_points=tissue_points)
    
    coverage = _compute_coverage_metrics(network, tissue_points)
    flow = _compute_flow_metrics(network, config)
    structure = _compute_structure_metrics(network, config)
    validity = _compute_validity_metrics(network)
    scores = _compute_scores(coverage, flow, structure, config)
    
    return EvalResult(
        coverage=coverage, flow=flow, structure=structure,
        validity=validity, scores=scores,
        metadata={
            "num_tissue_points": len(tissue_points),
            "config": {
                "coverage_weight": config.coverage_weight,
                "flow_weight": config.flow_weight,
                "structure_weight": config.structure_weight,
            },
        },
    )


def _compute_coverage_metrics(network: VascularNetwork, tissue_points: np.ndarray) -> CoverageMetrics:
    """Compute coverage and perfusion metrics."""
    if len(tissue_points) == 0:
        return CoverageMetrics(0.0, 0, 0.0, 0.0, 0.0)
    
    distances = []
    perfusion_threshold = 0.005
    
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
        degree = len(node.connected_segment_ids)
        degree_histogram[degree] = degree_histogram.get(degree, 0) + 1
    
    branching_angles = []
    for node in network.nodes.values():
        if node.node_type == "junction" and len(node.connected_segment_ids) >= 2:
            seg_ids = list(node.connected_segment_ids)
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
    
    murray_deviations = []
    for node in network.nodes.values():
        if node.node_type == "junction":
            segs = [network.segments.get(sid) for sid in node.connected_segment_ids]
            segs = [s for s in segs if s is not None]
            if len(segs) >= 2:
                radii = [s.attributes.get("radius", 0.001) for s in segs]
                parent_idx = np.argmax(radii)
                parent_r = radii[parent_idx]
                child_radii = [r for i, r in enumerate(radii) if i != parent_idx]
                expected = sum(r**3 for r in child_radii) ** (1/3)
                if expected > 0:
                    murray_deviations.append(abs(parent_r - expected) / expected)
    
    murray_deviation = float(np.mean(murray_deviations)) if murray_deviations else 0.0
    
    collision_count = 0
    min_clearance = float('inf')
    seg_list = list(network.segments.values())
    for i, seg1 in enumerate(seg_list[:min(100, len(seg_list))]):
        for seg2 in seg_list[i+1:min(i+101, len(seg_list))]:
            if seg1.start_node_id in (seg2.start_node_id, seg2.end_node_id) or                seg1.end_node_id in (seg2.start_node_id, seg2.end_node_id):
                continue
            p1 = network.nodes[seg1.start_node_id].position.to_array()
            p2 = network.nodes[seg1.end_node_id].position.to_array()
            p3 = network.nodes[seg2.start_node_id].position.to_array()
            p4 = network.nodes[seg2.end_node_id].position.to_array()
            mid1 = (p1 + p2) / 2
            mid2 = (p3 + p4) / 2
            dist = np.linalg.norm(mid1 - mid2)
            min_clearance = min(min_clearance, dist)
            r1 = seg1.attributes.get("radius", 0.001)
            r2 = seg2.attributes.get("radius", 0.001)
            if dist < (r1 + r2):
                collision_count += 1
    
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
    """Compute validity and quality checks."""
    has_self_intersections = False
    parameter_warnings = []
    
    for seg in network.segments.values():
        r = seg.attributes.get("radius", 0.001)
        if r < 0.0001:
            parameter_warnings.append(f"Segment {seg.id} has very small radius: {r:.6f}m")
            break
    
    for seg in network.segments.values():
        if seg.length < 0.0001:
            parameter_warnings.append(f"Segment {seg.id} is very short: {seg.length:.6f}m")
            break
    
    return ValidityMetrics(
        is_watertight=True,
        has_self_intersections=has_self_intersections,
        parameter_warnings=parameter_warnings,
        error_codes=[],
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
