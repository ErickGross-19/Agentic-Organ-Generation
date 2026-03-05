import json
from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, Optional, Union
from datetime import datetime
import networkx as nx
import numpy as np

from ..models import ValidationReport
from ..analysis.metrics import compute_centerline_geometry_metrics, compute_flow_metrics


def centerline_graph_to_json(
    G: nx.Graph,
    node_pressures: Optional[Dict[Any, float]] = None,
    edge_flows: Optional[Dict[tuple, float]] = None,
) -> Dict[str, Any]:
    """
    Convert a centerline NetworkX graph (and optional Poiseuille results)
    into a JSON-serializable dict with 'nodes' and 'edges'.

    Each node includes:
      - 'id'
      - any existing node attributes (coord, radius, etc.)
      - optional 'pressure' from node_pressures

    Each edge includes:
      - 'u', 'v'
      - any existing edge attributes (length, etc.)
      - optional 'flow' from edge_flows
    """
    nodes_json: list[Dict[str, Any]] = []
    edges_json: list[Dict[str, Any]] = []

    node_pressures = node_pressures or {}
    edge_flows = edge_flows or {}

    for n, attrs in G.nodes(data=True):
        node_data: Dict[str, Any] = {
            "id": n if isinstance(n, (int, str)) else str(n)
        }

        for key, value in attrs.items():
            if isinstance(value, np.ndarray):
                node_data[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                node_data[key] = list(value)
            elif isinstance(value, (np.floating, np.integer)):
                node_data[key] = float(value)
            else:
                node_data[key] = value

        if n in node_pressures:
            node_data["pressure"] = float(node_pressures[n])

        nodes_json.append(node_data)

    for u, v, attrs in G.edges(data=True):
        edge_data: Dict[str, Any] = {
            "u": u if isinstance(u, (int, str)) else str(u),
            "v": v if isinstance(v, (int, str)) else str(v),
        }

        for key, value in attrs.items():
            if isinstance(value, np.ndarray):
                edge_data[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                edge_data[key] = list(value)
            elif isinstance(value, (np.floating, np.integer)):
                edge_data[key] = float(value)
            else:
                edge_data[key] = value

        if (u, v) in edge_flows:
            edge_data["flow"] = float(edge_flows[(u, v)])
        elif (v, u) in edge_flows:
            edge_data["flow"] = float(edge_flows[(v, u)])

        edges_json.append(edge_data)

    return {"nodes": nodes_json, "edges": edges_json}


def save_validation_report_json(
    report: ValidationReport,
    json_path: str | Path,
    centerline_graph: Optional[nx.Graph] = None,
    node_pressures: Optional[Dict[Any, float]] = None,
    edge_flows: Optional[Dict[tuple, float]] = None,
) -> None:
    """
    Save a ValidationReport to JSON, optionally including the full centerline graph.

    Parameters
    ----------
    report : ValidationReport
        The validation report to save.
    json_path : str or Path
        Path where the JSON file will be saved.
    centerline_graph : nx.Graph, optional
        If provided, the full centerline graph will be serialized and included.
    node_pressures : dict, optional
        Node pressures from Poiseuille analysis.
    edge_flows : dict, optional
        Edge flows from Poiseuille analysis.
    """
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(report)

    if centerline_graph is not None:
        data["centerline_graph"] = centerline_graph_to_json(
            centerline_graph,
            node_pressures=node_pressures,
            edge_flows=edge_flows,
        )

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[save_validation_report_json] Saved to {json_path}")


def make_llm_context_report(
    report: ValidationReport,
    G: Optional[nx.Graph] = None,
    voxel_pitch_used: Optional[float] = None,
    smooth_iters_used: Optional[int] = None,
    include_distributions: bool = True,
) -> Dict[str, Any]:
    """
    Create a comprehensive JSON report optimized for LLM context.
    
    This report provides all STL geometry properties and flow metrics in a format
    that helps LLMs understand the vascular network and optimize Python file generation.
    
    Parameters
    ----------
    report : ValidationReport
        The validation report containing mesh diagnostics, connectivity, and flow data.
    G : nx.Graph, optional
        Centerline graph with solved Poiseuille flow. If provided, detailed geometry
        and flow distributions will be included.
    voxel_pitch_used : float, optional
        The voxel pitch used in processing (for recommendations).
    smooth_iters_used : int, optional
        The smoothing iterations used (for recommendations).
    include_distributions : bool, default=True
        Whether to include percentile distributions for edge lengths, radii, and flows.
    
    Returns
    -------
    dict
        Comprehensive report with schema_version, summary_text, recommendations,
        quality_flags, geometry, flow, components, and recommended_parameters.
    """
    
    def to_python_type(val):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(val, (np.floating, np.float32, np.float64)):
            return float(val)
        elif isinstance(val, (np.integer, np.int32, np.int64)):
            return int(val)
        elif isinstance(val, np.ndarray):
            return val.tolist()
        elif isinstance(val, (list, tuple)):
            return [to_python_type(v) for v in val]
        elif isinstance(val, dict):
            return {k: to_python_type(v) for k, v in val.items()}
        return val
    
    def compute_percentiles(arr, percentiles=[5, 25, 50, 75, 95]):
        """Compute percentiles for an array, handling NaN values."""
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {f"p{p}": None for p in percentiles}
        return {f"p{p}": float(np.percentile(arr, p)) for p in percentiles}
    
    after_repair = report.after_repair
    conn = report.connectivity
    poi = report.poiseuille_summary or {}
    
    is_watertight = after_repair.watertight
    is_single_component = after_repair.num_components == 1
    volume = after_repair.volume if after_repair.volume is not None else 0.0
    bbox_extents = after_repair.bounding_box_extents
    
    if bbox_extents and len(bbox_extents) == 3:
        extents_arr = np.array(bbox_extents)
        slenderness = float(np.max(extents_arr) / (np.min(extents_arr) + 1e-10))
    else:
        slenderness = None
    
    reachable_fraction = conn.get("reachable_fraction", 0.0)
    num_components = conn.get("num_fluid_components", 0)
    
    geometry_section = {
        "watertight": is_watertight,
        "volume": to_python_type(volume),
        "surface_area": None,
        "extents": to_python_type(bbox_extents) if bbox_extents else None,
        "slenderness": to_python_type(slenderness) if slenderness else None,
        "components": {
            "count": to_python_type(num_components),
            "reachable_fraction": to_python_type(reachable_fraction),
        },
    }
    
    flow_section = {}
    components_section = []
    
    if G is not None and G.number_of_nodes() > 0:
        try:
            geom_metrics = compute_centerline_geometry_metrics(G)
            
            degrees = dict(G.degree())
            degree_hist = {}
            for d in set(degrees.values()):
                degree_hist[str(d)] = sum(1 for deg in degrees.values() if deg == d)
            
            centerline_data = {
                "n_nodes": to_python_type(geom_metrics["n_nodes"]),
                "n_edges": to_python_type(geom_metrics["n_edges"]),
                "n_leaves": to_python_type(geom_metrics["n_leaves"]),
                "n_branch_nodes": to_python_type(geom_metrics["n_branch_nodes"]),
                "total_length": to_python_type(geom_metrics["length_total"]),
                "length_mean": to_python_type(geom_metrics["length_mean"]),
                "length_median": to_python_type(geom_metrics["length_median"]),
                "radius_mean": to_python_type(geom_metrics["radius_mean_global"]),
                "radius_median": to_python_type(geom_metrics["radius_median_global"]),
                "degree_histogram": degree_hist,
            }
            
            if include_distributions:
                centerline_data["distributions"] = {
                    "edge_length_percentiles": to_python_type(compute_percentiles(geom_metrics["edge_lengths"])),
                    "radius_percentiles": to_python_type(compute_percentiles(geom_metrics["edge_radii"])),
                }
            
            geometry_section["centerline"] = centerline_data
            
            if any(G.nodes[n].get("pressure") is not None for n in G.nodes):
                flow_metrics = compute_flow_metrics(G, mu=1.0e-3)
                
                Q_arr = np.asarray(flow_metrics["Q"], dtype=float)
                Q_arr = np.abs(Q_arr)
                Re_arr = np.asarray(flow_metrics["Re"], dtype=float)
                
                laminar_fraction = float(np.sum(Re_arr < 2300) / len(Re_arr)) if len(Re_arr) > 0 else 0.0
                
                Q_in = poi.get("total_inlet_flow", poi.get("total_inflow", 0.0))
                Q_out = poi.get("total_outlet_flow", poi.get("total_outflow", 0.0))
                conservation_error = abs(Q_in - Q_out) / max(abs(Q_in), abs(Q_out), 1e-20)
                
                pressures = [G.nodes[n].get("pressure", np.nan) for n in G.nodes]
                pressures = np.array([p for p in pressures if np.isfinite(p)])
                
                flow_section = {
                    "inlet_count": poi.get("num_inlets", None),
                    "outlet_count": poi.get("num_outlets", None),
                    "total_inlet_flow": to_python_type(Q_in),
                    "total_outlet_flow": to_python_type(Q_out),
                    "conservation_error": to_python_type(conservation_error),
                    "pressure_min": to_python_type(np.min(pressures)) if pressures.size > 0 else None,
                    "pressure_max": to_python_type(np.max(pressures)) if pressures.size > 0 else None,
                    "pressure_drop": to_python_type(np.max(pressures) - np.min(pressures)) if pressures.size > 0 else None,
                    "reynolds": {
                        "min": to_python_type(np.min(Re_arr)) if len(Re_arr) > 0 else None,
                        "p50": to_python_type(np.median(Re_arr)) if len(Re_arr) > 0 else None,
                        "p95": to_python_type(np.percentile(Re_arr, 95)) if len(Re_arr) > 0 else None,
                        "max": to_python_type(np.max(Re_arr)) if len(Re_arr) > 0 else None,
                        "laminar_fraction": to_python_type(laminar_fraction),
                    },
                }
                
                if include_distributions:
                    flow_section["edge_flow_percentiles"] = to_python_type(compute_percentiles(Q_arr))
                    flow_section["hydraulic_resistance_percentiles"] = to_python_type(
                        compute_percentiles(flow_metrics["R_hyd"])
                    )
        except Exception as e:
            print(f"[make_llm_context_report] Warning: Could not compute detailed metrics: {e}")
    
    has_bifurcations = geometry_section.get("centerline", {}).get("n_branch_nodes", 0) > 0
    good_flow_balance = flow_section.get("conservation_error", 1.0) < 0.1
    laminar_dominant = flow_section.get("reynolds", {}).get("laminar_fraction", 0.0) > 0.8
    
    quality_flags = {
        "is_watertight": {
            "value": is_watertight,
            "reason": "Post-repair watertight check passed" if is_watertight else "Mesh not watertight after repair"
        },
        "single_component": {
            "value": is_single_component,
            "reason": f"Mesh has {after_repair.num_components} component(s)"
        },
        "has_bifurcations": {
            "value": has_bifurcations,
            "reason": f"Centerline has {geometry_section.get('centerline', {}).get('n_branch_nodes', 0)} branch nodes"
        },
        "good_flow_balance": {
            "value": good_flow_balance,
            "reason": f"Flow conservation error {flow_section.get('conservation_error', 1.0):.1%} (threshold 10%)"
        },
        "laminar_flow_dominant": {
            "value": laminar_dominant,
            "reason": f"Laminar fraction {flow_section.get('reynolds', {}).get('laminar_fraction', 0.0):.1%} (Re < 2300)"
        },
    }
    
    summary_parts = []
    if is_watertight:
        summary_parts.append(f"Watertight STL with {after_repair.num_components} component(s).")
    else:
        summary_parts.append("Non-watertight STL.")
    
    if geometry_section.get("centerline"):
        n_branches = geometry_section["centerline"].get("n_branch_nodes", 0)
        if n_branches > 0:
            summary_parts.append(f"Centerline shows branching structure ({n_branches} branch nodes).")
        else:
            summary_parts.append("Centerline is unbranched.")
    
    if flow_section:
        Re_p95 = flow_section.get("reynolds", {}).get("p95")
        if Re_p95 and Re_p95 < 2300:
            summary_parts.append(f"Flow is laminar-dominant (Re p95 ~ {Re_p95:.0f}).")
        elif Re_p95:
            summary_parts.append(f"Flow may be transitional (Re p95 ~ {Re_p95:.0f}).")
        
        if good_flow_balance:
            err_pct = flow_section.get("conservation_error", 0.0) * 100
            summary_parts.append(f"Inlet/outlet flow balanced within {err_pct:.1f}%.")
    
    summary_text = " ".join(summary_parts)
    
    recommendations = []
    
    if geometry_section.get("centerline"):
        min_radius = geometry_section["centerline"].get("radius_mean", 0.1)
        if voxel_pitch_used and min_radius:
            if voxel_pitch_used > min_radius / 3:
                suggested_pitch = min_radius / 4
                recommendations.append(
                    f"Consider finer voxel_pitch ≈ {suggested_pitch:.3f} to better preserve small features (min radius ~ {min_radius:.3f})"
                )
    
    if geometry_section.get("centerline", {}).get("n_nodes", 0) > 1000:
        recommendations.append("Prefer sparse linear solver for large graphs (n_nodes > 1000)")
    
    if smooth_iters_used and smooth_iters_used > 15 and has_bifurcations:
        recommendations.append("Consider reducing smooth_iters to ~10 to preserve branch topology")
    
    if flow_section.get("reynolds", {}).get("max", 0) > 2300:
        recommendations.append("Some edges have Re > 2300; consider turbulence effects if Re > 4000")
    
    recommended_params = {
        "voxel_pitch": to_python_type(voxel_pitch_used) if voxel_pitch_used else 0.1,
        "smooth_iters": smooth_iters_used if smooth_iters_used else 20,
        "max_voxels": 3e7,
    }
    
    if geometry_section.get("centerline", {}).get("n_nodes", 0) > 1000:
        recommended_params["solver"] = "sparse"
    
    llm_report = {
        "schema_version": "1.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "units": "SI",
        "source": {
            "input_file": report.input_file,
            "cleaned_stl": report.cleaned_stl,
            "voxel_pitch_used": to_python_type(voxel_pitch_used) if voxel_pitch_used else None,
            "smooth_iters_used": smooth_iters_used,
            "bbox_min": to_python_type(conn.get("bbox_min")) if conn.get("bbox_min") else None,
            "grid_shape": to_python_type(conn.get("grid_shape")) if conn.get("grid_shape") else None,
        },
        "summary_text": summary_text,
        "recommendations": recommendations,
        "quality_flags": quality_flags,
        "geometry": geometry_section,
        "flow": flow_section if flow_section else None,
        "components": components_section if components_section else None,
        "recommended_parameters": recommended_params,
    }
    
    return llm_report


def save_llm_context_report_json(
    report: ValidationReport,
    json_path: Union[str, Path],
    G: Optional[nx.Graph] = None,
    voxel_pitch_used: Optional[float] = None,
    smooth_iters_used: Optional[int] = None,
    include_distributions: bool = True,
) -> None:
    """
    Save a comprehensive LLM-optimized context report to JSON.
    
    Parameters
    ----------
    report : ValidationReport
        The validation report.
    json_path : str or Path
        Path where the JSON file will be saved.
    G : nx.Graph, optional
        Centerline graph with solved Poiseuille flow.
    voxel_pitch_used : float, optional
        The voxel pitch used in processing.
    smooth_iters_used : int, optional
        The smoothing iterations used.
    include_distributions : bool, default=True
        Whether to include percentile distributions.
    """
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    llm_report = make_llm_context_report(
        report=report,
        G=G,
        voxel_pitch_used=voxel_pitch_used,
        smooth_iters_used=smooth_iters_used,
        include_distributions=include_distributions,
    )
    
    with open(json_path, "w") as f:
        json.dump(llm_report, f, indent=2)
    
    print(f"[save_llm_context_report_json] Saved LLM context report to {json_path}")
