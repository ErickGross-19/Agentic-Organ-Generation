"""
Adapter for creating comprehensive reports integrating with existing package.
"""

from typing import Dict, Any, Optional
from ..core.network import VascularNetwork
from ..core.result import OperationResult, OperationStatus


def make_full_report(
    network: VascularNetwork,
    include_flow: bool = True,
    include_geometry: bool = True,
    include_coverage: bool = False,
    tissue_points: Optional[Any] = None,
    mu: float = 1.0e-3,
    pin: float = 13000.0,
    pout: float = 2000.0,
) -> Dict[str, Any]:
    """
    Create comprehensive LLM-friendly report combining all metrics.
    
    Integrates with existing vascular_network package for detailed analysis.
    
    Parameters
    ----------
    network : VascularNetwork
        The vascular network to analyze
    include_flow : bool
        Whether to include flow analysis
    include_geometry : bool
        Whether to include geometry metrics
    include_coverage : bool
        Whether to include coverage analysis
    tissue_points : array-like, optional
        Tissue points for coverage analysis
    mu : float
        Dynamic viscosity (Pa·s)
    pin : float
        Inlet pressure (Pa)
    pout : float
        Outlet pressure (Pa)
        
    Returns
    -------
    dict
        Comprehensive report with all metrics
    """
    from .networkx_adapter import to_networkx_graph
    
    report = {
        "schema_version": "1.0",
        "network_summary": {
            "num_nodes": len(network.nodes),
            "num_segments": len(network.segments),
            "domain": network.domain.to_dict(),
        },
        "metadata": network.metadata.copy(),
    }
    
    G, node_id_map = to_networkx_graph(network)
    
    if include_geometry:
        try:
            from validity.analysis.metrics import compute_centerline_geometry_metrics
            geom_metrics = compute_centerline_geometry_metrics(G)
            report["geometry_metrics"] = geom_metrics
        except Exception as e:
            report["geometry_metrics"] = {"error": str(e)}
    
    if include_flow:
        try:
            from validity.analysis.cfd import compute_poiseuille_network
            
            inlet_nodes = [
                node_id_map[nid] for nid, node in network.nodes.items()
                if node.node_type == "inlet"
            ]
            outlet_nodes = [
                node_id_map[nid] for nid, node in network.nodes.items()
                if node.node_type == "outlet"
            ]
            
            if inlet_nodes and outlet_nodes:
                flow_result = compute_poiseuille_network(
                    G,
                    mu=mu,
                    inlet_nodes=inlet_nodes,
                    outlet_nodes=outlet_nodes,
                    pin=pin,
                    pout=pout,
                    write_to_graph=True,
                )
                
                report["flow_analysis"] = flow_result["summary"]
                
                from validity.analysis.metrics import compute_flow_metrics
                flow_metrics = compute_flow_metrics(G, mu=mu, rho=1060.0, Q_attr="Q")
                report["flow_metrics"] = flow_metrics
            else:
                report["flow_analysis"] = {
                    "error": "No inlet or outlet nodes found",
                    "inlet_count": len(inlet_nodes),
                    "outlet_count": len(outlet_nodes),
                }
        except Exception as e:
            report["flow_analysis"] = {"error": str(e)}
    
    if include_coverage and tissue_points is not None:
        try:
            from ..analysis.coverage import compute_coverage
            coverage_result = compute_coverage(
                network,
                tissue_points,
                diffusion_distance=0.01,  # 10mm default
            )
            report["coverage_analysis"] = coverage_result.metadata
        except Exception as e:
            report["coverage_analysis"] = {"error": str(e)}
    
    report["quality_flags"] = _compute_quality_flags(report)
    
    report["summary"] = _generate_summary(report)
    
    return report


def _compute_quality_flags(report: Dict[str, Any]) -> Dict[str, Any]:
    """Compute quality flags from report data."""
    flags = {}
    
    if "flow_analysis" in report and "error" not in report["flow_analysis"]:
        flow = report["flow_analysis"]
        inlet_flow = flow.get("total_inlet_flow", 0)
        outlet_flow = flow.get("total_outlet_flow", 0)
        
        if inlet_flow > 0:
            balance_error = abs(inlet_flow - outlet_flow) / inlet_flow
            flags["flow_balanced"] = balance_error < 0.05
            flags["flow_balance_error"] = float(balance_error)
        else:
            flags["flow_balanced"] = False
            flags["flow_balance_error"] = None
    
    if "flow_metrics" in report and "error" not in report["flow_metrics"]:
        flow_metrics = report["flow_metrics"]
        max_re = flow_metrics.get("max_reynolds", 0)
        flags["laminar_flow"] = max_re < 2300
        flags["max_reynolds"] = float(max_re)
    
    return flags


def _generate_summary(report: Dict[str, Any]) -> str:
    """Generate natural language summary."""
    summary_parts = []
    
    net_summary = report.get("network_summary", {})
    summary_parts.append(
        f"Network has {net_summary.get('num_nodes', 0)} nodes "
        f"and {net_summary.get('num_segments', 0)} segments."
    )
    
    flags = report.get("quality_flags", {})
    if flags.get("flow_balanced"):
        summary_parts.append("Flow is well balanced.")
    elif "flow_balance_error" in flags:
        error = flags["flow_balance_error"]
        if error is not None:
            summary_parts.append(f"Flow balance error: {error:.1%}")
    
    if flags.get("laminar_flow"):
        summary_parts.append("Flow is laminar throughout.")
    elif "max_reynolds" in flags:
        max_re = flags["max_reynolds"]
        if max_re > 2300:
            summary_parts.append(f"Turbulent flow detected (Re={max_re:.0f}).")
    
    return " ".join(summary_parts)
