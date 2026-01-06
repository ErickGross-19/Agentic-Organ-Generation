"""
Flow analysis and hemodynamic plausibility checks.
"""

from typing import Dict, List, Optional
import numpy as np
from ..core.network import VascularNetwork


def estimate_flows(
    network: VascularNetwork,
    inlet_pressures: Dict[int, float],
    outlet_pressures: Dict[int, float],
    blood_viscosity: float = 0.004,  # Pa·s (typical blood viscosity)
) -> Dict:
    """
    import warnings
    warnings.warn(
        "estimate_flows() is deprecated. Use compute_component_flows() from solver.py instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    Estimate flows using simplified network flow model.
    
    Uses Poiseuille's law for resistance and solves linear system.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to analyze
    inlet_pressures : dict
        Mapping of inlet node IDs to pressures (Pa)
    outlet_pressures : dict
        Mapping of outlet node IDs to pressures (Pa)
    blood_viscosity : float
        Blood viscosity (Pa·s)
    
    Returns
    -------
    result : dict
        Flow analysis with:
        - segment_flows: dict mapping segment ID to flow rate (m³/s)
        - node_pressures: dict mapping node ID to pressure (Pa)
        - total_inlet_flow: total flow entering network
        - total_outlet_flow: total flow leaving network
        - flow_balance_error: relative error in flow conservation
    """
    
    node_ids = list(network.nodes.keys())
    segment_ids = list(network.segments.keys())
    
    if not node_ids or not segment_ids:
        return {
            "segment_flows": {},
            "node_pressures": {},
            "total_inlet_flow": 0.0,
            "total_outlet_flow": 0.0,
            "flow_balance_error": 0.0,
        }
    
    segment_resistances = {}
    
    for seg_id, segment in network.segments.items():
        length = segment.geometry.length()
        radius = segment.geometry.mean_radius()
        
        if radius < 1e-6 or length < 1e-6:
            resistance = 1e12  # Very high resistance for degenerate segments
        else:
            resistance = (8.0 * blood_viscosity * length) / (np.pi * radius**4)
        
        segment_resistances[seg_id] = resistance
    
    
    
    segment_flows = {}
    node_pressures = {}
    
    for inlet_id, pressure in inlet_pressures.items():
        node_pressures[inlet_id] = pressure
    
    for outlet_id, pressure in outlet_pressures.items():
        node_pressures[outlet_id] = pressure
    
    
    for seg_id, segment in network.segments.items():
        start_pressure = node_pressures.get(segment.start_node_id, 10000.0)
        end_pressure = node_pressures.get(segment.end_node_id, 5000.0)
        
        resistance = segment_resistances[seg_id]
        flow = (start_pressure - end_pressure) / resistance
        
        segment_flows[seg_id] = flow
    
    total_inlet_flow = sum(
        segment_flows.get(seg_id, 0.0)
        for seg_id, seg in network.segments.items()
        if seg.start_node_id in inlet_pressures
    )
    
    total_outlet_flow = sum(
        segment_flows.get(seg_id, 0.0)
        for seg_id, seg in network.segments.items()
        if seg.end_node_id in outlet_pressures
    )
    
    if total_inlet_flow > 0:
        flow_balance_error = abs(total_inlet_flow - total_outlet_flow) / total_inlet_flow
    else:
        flow_balance_error = 0.0
    
    return {
        "segment_flows": segment_flows,
        "node_pressures": node_pressures,
        "segment_resistances": segment_resistances,
        "total_inlet_flow": total_inlet_flow,
        "total_outlet_flow": total_outlet_flow,
        "flow_balance_error": flow_balance_error,
    }


def check_hemodynamic_plausibility(
    network: VascularNetwork,
    flow_result: Optional[Dict] = None,
) -> List[str]:
    """
    Check hemodynamic plausibility of network.
    
    Returns warnings for potential issues.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to check
    flow_result : dict, optional
        Result from estimate_flows()
    
    Returns
    -------
    warnings : List[str]
        List of warning messages
    """
    warnings = []
    
    for segment in network.segments.values():
        if segment.geometry.radius_end > segment.geometry.radius_start * 1.1:
            warnings.append(
                f"Segment {segment.id}: radius increases downstream "
                f"({segment.geometry.radius_start:.4f} -> {segment.geometry.radius_end:.4f}m)"
            )
    
    for node in network.nodes.values():
        if node.node_type != "junction":
            continue
        
        parent_segments = [
            seg for seg in network.segments.values()
            if seg.end_node_id == node.id
        ]
        
        child_segments = [
            seg for seg in network.segments.values()
            if seg.start_node_id == node.id
        ]
        
        if len(parent_segments) == 1 and len(child_segments) >= 2:
            parent_r = parent_segments[0].geometry.radius_end
            children_r_sum = sum(seg.geometry.radius_start**3 for seg in child_segments)
            
            expected_parent_r = children_r_sum ** (1/3)
            relative_error = abs(parent_r - expected_parent_r) / parent_r
            
            if relative_error > 0.3:  # 30% deviation
                warnings.append(
                    f"Node {node.id}: Murray's law violation "
                    f"(parent r={parent_r:.4f}m, expected {expected_parent_r:.4f}m)"
                )
    
    if flow_result is not None:
        if flow_result["flow_balance_error"] > 0.1:  # 10% error
            warnings.append(
                f"Flow balance error: {flow_result['flow_balance_error']:.1%} "
                f"(inlet: {flow_result['total_inlet_flow']:.2e} m³/s, "
                f"outlet: {flow_result['total_outlet_flow']:.2e} m³/s)"
            )
    
    blood_density = 1060  # kg/m³
    blood_viscosity = 0.004  # Pa·s
    
    if flow_result is not None:
        for seg_id, flow in flow_result["segment_flows"].items():
            segment = network.get_segment(seg_id)
            if segment is None:
                continue
            
            radius = segment.geometry.mean_radius()
            area = np.pi * radius**2
            
            if area > 1e-10:
                velocity = abs(flow) / area
                reynolds = (blood_density * velocity * 2 * radius) / blood_viscosity
                
                if reynolds > 2300:
                    warnings.append(
                        f"Segment {seg_id}: turbulent flow (Re={reynolds:.0f} > 2300)"
                    )
    
    return warnings
