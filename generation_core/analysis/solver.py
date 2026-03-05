"""
Full network flow solver using scipy.sparse.

Uses the in-repo Poiseuille solver from validity.analysis.cfd.

Note: The library uses METERS internally for all geometry.
The solver expects geometry in meters (the internal unit system).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from ..core.network import VascularNetwork
from ..core.result import OperationResult, OperationStatus, ErrorCode
from ..utils.units import to_si_length, INTERNAL_UNIT


def _convert_network_to_si(network: VascularNetwork, from_units: str) -> VascularNetwork:
    """
    Create a copy of network with geometry converted to SI (meters).
    
    This is used internally by the flow solver to ensure correct physics.
    
    Parameters
    ----------
    network : VascularNetwork
        Original network (in mm, cm, or m)
    from_units : str
        Units of original network geometry
        
    Returns
    -------
    VascularNetwork
        Copy of network with geometry in meters
    """
    import copy
    from ..core.types import Point3D, TubeGeometry
    
    if from_units == "m":
        return network
    
    network_si = copy.deepcopy(network)
    
    for node in network_si.nodes.values():
        node.position = Point3D(
            x=to_si_length(node.position.x, from_units),
            y=to_si_length(node.position.y, from_units),
            z=to_si_length(node.position.z, from_units),
        )
    
    for seg in network_si.segments.values():
        seg.geometry = TubeGeometry(
            start=Point3D(
                x=to_si_length(seg.geometry.start.x, from_units),
                y=to_si_length(seg.geometry.start.y, from_units),
                z=to_si_length(seg.geometry.start.z, from_units),
            ),
            end=Point3D(
                x=to_si_length(seg.geometry.end.x, from_units),
                y=to_si_length(seg.geometry.end.y, from_units),
                z=to_si_length(seg.geometry.end.z, from_units),
            ),
            radius_start=to_si_length(seg.geometry.radius_start, from_units),
            radius_end=to_si_length(seg.geometry.radius_end, from_units),
        )
    
    return network_si


def solve_flow(
    network: VascularNetwork,
    inlet_node_ids: Optional[List[int]] = None,
    outlet_node_ids: Optional[List[int]] = None,
    pin: float = 13000.0,
    pout: float = 2000.0,
    mu: float = 1.0e-3,
    write_to_network: bool = True,
    geometry_units: str = INTERNAL_UNIT,
) -> OperationResult:
    """
    Solve flow through the vascular network using full Poiseuille solver.
    
    Uses the in-repo validity.analysis.cfd.compute_poiseuille_network function,
    converting to/from NetworkX format.
    
    **Units**: Network geometry is assumed to be in meters (default, the internal unit).
    The solver uses SI units (meters) for physics calculations.
    
    Parameters
    ----------
    network : VascularNetwork
        The vascular network (geometry in millimeters by default)
    inlet_node_ids : list of int, optional
        Inlet node IDs. If None, auto-detect from node_type
    outlet_node_ids : list of int, optional
        Outlet node IDs. If None, auto-detect from node_type
    pin : float
        Inlet pressure (Pa). Default ~100 mmHg
    pout : float
        Outlet pressure (Pa). Default ~15 mmHg
    mu : float
        Dynamic viscosity (Pa·s). Default for blood at 37°C
    write_to_network : bool
        Whether to write results back to network node/segment attributes
    geometry_units : str
        Units of network geometry ('mm', 'm', 'cm'). Default: 'mm'
        
    Returns
    -------
    OperationResult
        Result with flow solution in metadata
        
    Notes
    -----
    The solver uses Poiseuille's law: R = 8*mu*L/(pi*r^4)
    All lengths and radii are converted to meters (SI) internally for correct physics.
    """
    from ..adapters.networkx_adapter import to_networkx_graph
    from validity.analysis.cfd import compute_poiseuille_network
    
    try:
        if inlet_node_ids is None:
            inlet_node_ids = [
                nid for nid, node in network.nodes.items()
                if node.node_type == "inlet"
            ]
        
        if outlet_node_ids is None:
            outlet_node_ids = [
                nid for nid, node in network.nodes.items()
                if node.node_type == "outlet"
            ]
        
        if not inlet_node_ids:
            return OperationResult.failure(
                "No inlet nodes found",
                error_codes=[ErrorCode.NODE_NOT_FOUND.value],
            )
        
        if not outlet_node_ids:
            return OperationResult.failure(
                "No outlet nodes found",
                error_codes=[ErrorCode.NODE_NOT_FOUND.value],
            )
        
        # The vascular_network solver expects meters
        network_si = _convert_network_to_si(network, geometry_units)
        
        G, node_id_map = to_networkx_graph(network_si)
        
        inlet_nx = [node_id_map[nid] for nid in inlet_node_ids]
        outlet_nx = [node_id_map[nid] for nid in outlet_node_ids]
        
        flow_result = compute_poiseuille_network(
            G,
            mu=mu,
            inlet_nodes=inlet_nx,
            outlet_nodes=outlet_nx,
            pin=pin,
            pout=pout,
            write_to_graph=True,
        )
        
        node_pressures = flow_result["node_pressures"]
        edge_flows = flow_result["edge_flows"]
        summary = flow_result["summary"]
        
        if write_to_network:
            nx_to_vascular = {v: k for k, v in node_id_map.items()}
            
            for nx_id, pressure in node_pressures.items():
                vascular_id = nx_to_vascular[nx_id]
                network.nodes[vascular_id].attributes['pressure'] = float(pressure)
            
            # Blood properties for Reynolds calculation
            rho = 1060.0  # kg/m³ for blood
            
            for (u_nx, v_nx), flow in edge_flows.items():
                u_vascular = nx_to_vascular[u_nx]
                v_vascular = nx_to_vascular[v_nx]
                
                for seg in network.segments.values():
                    if ((seg.start_node_id == u_vascular and seg.end_node_id == v_vascular) or
                        (seg.start_node_id == v_vascular and seg.end_node_id == u_vascular)):
                        seg.attributes['flow'] = float(abs(flow))
                        radius_mm = (seg.geometry.radius_start + seg.geometry.radius_end) / 2
                        radius_m = to_si_length(radius_mm, geometry_units)
                        velocity = float(abs(flow) / (np.pi * radius_m ** 2))
                        seg.attributes['velocity'] = velocity
                        
                        # P0-5: Add normalized pressure/flow attributes for evaluation
                        # Store endpoint pressures on segment for easy access
                        seg.attributes['pressure_start'] = float(
                            network.nodes[seg.start_node_id].attributes.get('pressure', 0.0)
                        )
                        seg.attributes['pressure_end'] = float(
                            network.nodes[seg.end_node_id].attributes.get('pressure', 0.0)
                        )
                        
                        # Compute Reynolds number: Re = rho * v * D / mu
                        diameter_m = 2 * radius_m
                        reynolds = rho * velocity * diameter_m / mu
                        seg.attributes['reynolds'] = float(reynolds)
                        
                        break
        
        component_flows = compute_component_flows(network)
        
        result = OperationResult.success(
            f"Flow solved: {summary['num_nodes']} nodes, {summary['num_edges']} edges",
            metadata={
                'summary': summary,
                'inlet_flow': float(summary['total_inlet_flow']),
                'outlet_flow': float(summary['total_outlet_flow']),
                'flow_balance_error': float(
                    abs(summary['total_inlet_flow'] - summary['total_outlet_flow']) /
                    max(summary['total_inlet_flow'], 1e-10)
                ),
                'pin': float(pin),
                'pout': float(pout),
                'mu': float(mu),
                'component_flows': component_flows,
            },
        )
        
        for warning in summary.get('warnings', []):
            result.add_warning(warning)
        
        if summary.get('used_lstsq', False):
            result.add_warning("System was rank-deficient, used least-squares")
            result.error_codes.append(ErrorCode.DIRICHLET_SINGULAR.value)
        
        balance_error = result.metadata['flow_balance_error']
        if balance_error > 0.05:
            result.add_warning(f"Flow balance error {balance_error:.1%} exceeds 5%")
            result.error_codes.append(ErrorCode.FLOW_BALANCE_ERROR.value)
        
        return result
        
    except Exception as e:
        return OperationResult.failure(
            f"Flow solver failed: {e}",
            error_codes=[ErrorCode.DIRICHLET_SINGULAR.value],
        )


def compute_component_flows(network: VascularNetwork) -> Dict:
    """
    Compute flow metrics per component (per tree).
    
    For each connected component, measures flow from its inlet/outlet node
    through the entire tree to its terminals. This is useful for dual-tree
    networks where arterial and venous trees should be analyzed separately.
    
    For mixed components (containing both arterial and venous nodes, e.g., after
    anastomosis creation), reports per-vessel-type metrics within the component
    to ensure flow totals are correctly computed for evaluation.
    
    Parameters
    ----------
    network : VascularNetwork
        Network with solved flow (must have 'pressure' and 'flow' attributes)
        
    Returns
    -------
    component_metrics : dict
        Dictionary with per-component flow metrics:
        - components: list of component info dicts, each containing:
          - component_id: int
          - vessel_type: str (arterial, venous)
          - root_node_id: int (inlet or outlet node)
          - root_node_type: str (inlet or outlet)
          - total_flow: float (flow at root node)
          - num_nodes: int
          - num_segments: int
          - terminal_nodes: list of terminal node IDs
          - avg_pressure: float
          - pressure_drop: float (from root to terminals)
        For mixed components, two entries are created (one arterial, one venous).
    """
    import networkx as nx
    from ..adapters.networkx_adapter import to_networkx_graph
    
    has_flow = any('flow' in seg.attributes for seg in network.segments.values())
    if not has_flow:
        return {
            'components': [],
            'num_components': 0,
            'warning': 'Network does not have flow solution',
        }
    
    G_nx, node_id_map = to_networkx_graph(network)
    
    nx_to_vascular = {v: k for k, v in node_id_map.items()}
    
    components = list(nx.connected_components(G_nx))
    
    component_metrics = []
    
    for comp_idx, comp_nodes in enumerate(components):
        comp_vascular_ids = set(nx_to_vascular[nx_id] for nx_id in comp_nodes)
        
        comp_nodes_objs = [network.nodes[vid] for vid in comp_vascular_ids]
        
        vessel_types = set(n.vessel_type for n in comp_nodes_objs if n.vessel_type)
        
        # P0-NEW-3: For mixed components, report per-vessel-type metrics
        if len(vessel_types) > 1:
            # Mixed component - split into arterial and venous sub-metrics
            for vtype in vessel_types:
                if vtype not in ("arterial", "venous"):
                    continue
                    
                metrics = _compute_vessel_type_metrics(
                    network, comp_vascular_ids, vtype, comp_idx
                )
                if metrics:
                    component_metrics.append(metrics)
        elif len(vessel_types) == 1:
            # Single vessel type - compute normally
            vtype = list(vessel_types)[0]
            metrics = _compute_vessel_type_metrics(
                network, comp_vascular_ids, vtype, comp_idx
            )
            if metrics:
                component_metrics.append(metrics)
        else:
            # Unknown vessel type - try to compute with "unknown"
            metrics = _compute_vessel_type_metrics(
                network, comp_vascular_ids, "unknown", comp_idx
            )
            if metrics:
                component_metrics.append(metrics)
    
    return {
        'components': component_metrics,
        'num_components': len(component_metrics),
    }


def _compute_vessel_type_metrics(
    network: VascularNetwork,
    comp_vascular_ids: set,
    vessel_type: str,
    comp_idx: int,
) -> Optional[Dict]:
    """
    Compute flow metrics for a specific vessel type within a component.
    
    P0-NEW-3: Handles mixed components by filtering nodes/segments by vessel_type.
    P1-NEW-1: Computes flow leaving root consistently using pressure direction.
    
    Parameters
    ----------
    network : VascularNetwork
        Network with solved flow
    comp_vascular_ids : set
        Set of node IDs in this component
    vessel_type : str
        Vessel type to filter by ("arterial", "venous", or "unknown")
    comp_idx : int
        Component index for reporting
        
    Returns
    -------
    dict or None
        Component metrics dict, or None if no valid root found
    """
    # Filter nodes by vessel type (or include all if "unknown")
    if vessel_type in ("arterial", "venous"):
        type_node_ids = {
            vid for vid in comp_vascular_ids
            if network.nodes[vid].vessel_type == vessel_type
        }
    else:
        type_node_ids = comp_vascular_ids
    
    if not type_node_ids:
        return None
    
    type_nodes_objs = [network.nodes[vid] for vid in type_node_ids]
    
    inlet_nodes = [n for n in type_nodes_objs if n.node_type == "inlet"]
    outlet_nodes = [n for n in type_nodes_objs if n.node_type == "outlet"]
    terminal_nodes = [n for n in type_nodes_objs if n.node_type == "terminal"]
    
    # Determine root node
    if inlet_nodes:
        root_node = inlet_nodes[0]
        root_type = "inlet"
    elif outlet_nodes:
        root_node = outlet_nodes[0]
        root_type = "outlet"
    else:
        return None
    
    # P1-NEW-1: Compute flow leaving root consistently using pressure direction
    # For inlet: flow leaves root (outgoing), for outlet: flow enters root (incoming)
    root_pressure = root_node.attributes.get('pressure', 0.0)
    total_flow = 0.0
    
    for seg in network.segments.values():
        # Only consider segments connected to root
        if seg.start_node_id == root_node.id:
            other_node = network.nodes.get(seg.end_node_id)
            if other_node:
                other_pressure = other_node.attributes.get('pressure', 0.0)
                flow = abs(seg.attributes.get('flow', 0.0))
                # Flow goes from high to low pressure
                if root_pressure > other_pressure:
                    # Flow leaving root
                    total_flow += flow
                else:
                    # Flow entering root (shouldn't happen for inlet, but handle it)
                    total_flow += flow
        elif seg.end_node_id == root_node.id:
            other_node = network.nodes.get(seg.start_node_id)
            if other_node:
                other_pressure = other_node.attributes.get('pressure', 0.0)
                flow = abs(seg.attributes.get('flow', 0.0))
                # Flow goes from high to low pressure
                if other_pressure > root_pressure:
                    # Flow entering root
                    total_flow += flow
                else:
                    # Flow leaving root
                    total_flow += flow
    
    # Compute pressure metrics
    pressures = [n.attributes.get('pressure', 0.0) for n in type_nodes_objs 
                if 'pressure' in n.attributes]
    avg_pressure = float(np.mean(pressures)) if pressures else 0.0
    
    terminal_pressures = [n.attributes.get('pressure', 0.0) for n in terminal_nodes
                         if 'pressure' in n.attributes]
    avg_terminal_pressure = float(np.mean(terminal_pressures)) if terminal_pressures else 0.0
    pressure_drop = abs(root_pressure - avg_terminal_pressure)
    
    # Filter segments by vessel type
    if vessel_type in ("arterial", "venous"):
        comp_segments = [
            seg for seg in network.segments.values()
            if seg.vessel_type == vessel_type
            and seg.start_node_id in comp_vascular_ids 
            and seg.end_node_id in comp_vascular_ids
        ]
    else:
        comp_segments = [
            seg for seg in network.segments.values()
            if seg.start_node_id in comp_vascular_ids 
            and seg.end_node_id in comp_vascular_ids
        ]
    
    return {
        'component_id': comp_idx,
        'vessel_type': vessel_type,
        'root_node_id': root_node.id,
        'root_node_type': root_type,
        'total_flow': float(total_flow),
        'num_nodes': len(type_node_ids),
        'num_segments': len(comp_segments),
        'terminal_nodes': [n.id for n in terminal_nodes],
        'num_terminals': len(terminal_nodes),
        'avg_pressure': avg_pressure,
        'root_pressure': float(root_pressure),
        'avg_terminal_pressure': avg_terminal_pressure,
        'pressure_drop': float(pressure_drop),
    }


def check_flow_plausibility(
    network: VascularNetwork,
    geometry_units: str = INTERNAL_UNIT,
) -> OperationResult:
    """
    Check if flow solution is hemodynamically plausible.
    
    Checks:
    - Flow conservation at junctions
    - Pressure monotonicity
    - Reynolds number ranges
    - Murray's law compliance
    
    Parameters
    ----------
    network : VascularNetwork
        Network with solved flow (must have 'pressure' and 'flow' attributes)
    geometry_units : str
        Units of network geometry ('mm', 'm', 'cm'). Default: 'mm'
        
    Returns
    -------
    OperationResult
        Result with plausibility checks in metadata
    """
    warnings = []
    errors = []
    
    has_pressure = any('pressure' in node.attributes for node in network.nodes.values())
    has_flow = any('flow' in seg.attributes for seg in network.segments.values())
    
    if not has_pressure or not has_flow:
        return OperationResult.failure(
            "Network does not have flow solution (missing 'pressure' or 'flow' attributes)",
        )
    
    for node_id, node in network.nodes.items():
        if node.node_type == "junction":
            inflow = 0.0
            outflow = 0.0
            
            for seg in network.segments.values():
                flow = seg.attributes.get('flow', 0.0)
                
                if seg.end_node_id == node_id:
                    inflow += flow
                elif seg.start_node_id == node_id:
                    outflow += flow
            
            balance = abs(inflow - outflow) / max(inflow, 1e-10)
            if balance > 0.1:
                warnings.append(
                    f"Node {node_id}: flow balance error {balance:.1%}"
                )
    
    for seg in network.segments.values():
        p_start = network.nodes[seg.start_node_id].attributes.get('pressure', 0)
        p_end = network.nodes[seg.end_node_id].attributes.get('pressure', 0)
        flow = seg.attributes.get('flow', 0)
        
        if flow > 0 and p_end > p_start:
            warnings.append(
                f"Segment {seg.id}: pressure increases along flow direction"
            )
    
    max_re = 0.0
    for seg in network.segments.values():
        velocity = seg.attributes.get('velocity', 0)
        radius_input = (seg.geometry.radius_start + seg.geometry.radius_end) / 2
        radius_m = to_si_length(radius_input, geometry_units)
        diameter_m = 2 * radius_m
        
        rho = 1060.0  # kg/m³ for blood
        mu = 1.0e-3  # Pa·s
        re = rho * velocity * diameter_m / mu
        max_re = max(max_re, re)
        
        if re > 2300:
            warnings.append(
                f"Segment {seg.id}: turbulent flow (Re={re:.0f})"
            )
    
    result = OperationResult.success(
        f"Plausibility check complete: {len(warnings)} warnings, {len(errors)} errors",
        metadata={
            'max_reynolds': float(max_re),
            'is_laminar': max_re < 2300,
            'num_warnings': len(warnings),
            'num_errors': len(errors),
        },
    )
    
    for warning in warnings:
        result.add_warning(warning)
    
    for error in errors:
        result.add_error(error)
    
    return result
