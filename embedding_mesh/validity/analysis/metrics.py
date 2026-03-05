"""
Geometric and flow metrics for centerline graphs.
"""

import numpy as np
import networkx as nx
from typing import Dict, Any


def compute_centerline_geometry_metrics(G: nx.Graph) -> Dict[str, Any]:
    """
    Extract basic geometric metrics from a centerline graph.

    Requires node attribute:
        - 'coord' : (3,) array-like in world units
        - optional 'radius' : local radius (same units)

    Returns
    -------
    metrics : dict with keys like:
        - n_nodes, n_edges
        - n_leaves, n_branch_nodes, n_degree2
        - edge_lengths : list[float]
        - edge_radii   : list[float] (mean radius per edge)
        - length_total : float
        - length_mean, length_median
        - radius_mean_global, radius_median_global
        - bounding_box : { 'min': [x,y,z], 'max': [x,y,z], 'extents': [dx,dy,dz] }
    """
    if G.number_of_nodes() == 0:
        raise ValueError("Centerline graph has no nodes.")

    # degree statistics
    degrees = dict(G.degree())
    leaves = [n for n, d in degrees.items() if d == 1]
    branch_nodes = [n for n, d in degrees.items() if d > 2]
    deg2_nodes = [n for n, d in degrees.items() if d == 2]

    # coords & radii at nodes
    coords = {}
    radii = {}
    for n, data in G.nodes(data=True):
        c = np.asarray(data.get("coord", [0.0, 0.0, 0.0]), dtype=float)
        coords[n] = c
        r = data.get("radius", None)
        if r is not None:
            radii[n] = float(r)

    # bounding box from node coordinates
    all_pts = np.vstack(list(coords.values()))
    bb_min = all_pts.min(axis=0)
    bb_max = all_pts.max(axis=0)
    bb_ext = bb_max - bb_min

    # per-edge lengths and radii
    edge_lengths = []
    edge_radii = []

    for u, v, data in G.edges(data=True):
        # length: prefer stored 'length', else Euclidean
        L = data.get("length", None)
        if L is None:
            cu = coords[u]
            cv = coords[v]
            L = float(np.linalg.norm(cu - cv))
        else:
            L = float(L)
        edge_lengths.append(L)

        # radius: mean of endpoints if available, else edge attribute
        ru = radii.get(u, None)
        rv = radii.get(v, None)
        re = data.get("radius", None)
        if re is not None:
            r_edge = float(re)
        elif (ru is not None) and (rv is not None):
            r_edge = 0.5 * (ru + rv)
        elif ru is not None:
            r_edge = float(ru)
        elif rv is not None:
            r_edge = float(rv)
        else:
            r_edge = np.nan
        edge_radii.append(r_edge)

    edge_lengths = np.asarray(edge_lengths, dtype=float)
    edge_radii = np.asarray(edge_radii, dtype=float)

    length_total = float(edge_lengths.sum())
    length_mean = float(np.mean(edge_lengths))
    length_median = float(np.median(edge_lengths))

    finite_r = edge_radii[np.isfinite(edge_radii)]
    if finite_r.size > 0:
        r_mean_global = float(np.mean(finite_r))
        r_median_global = float(np.median(finite_r))
    else:
        r_mean_global = np.nan
        r_median_global = np.nan

    metrics = {
        "n_nodes": int(G.number_of_nodes()),
        "n_edges": int(G.number_of_edges()),
        "n_leaves": int(len(leaves)),
        "n_branch_nodes": int(len(branch_nodes)),
        "n_degree2": int(len(deg2_nodes)),
        "leaves": leaves,
        "branch_nodes": branch_nodes,
        "edge_lengths": edge_lengths.tolist(),
        "edge_radii": edge_radii.tolist(),
        "length_total": length_total,
        "length_mean": length_mean,
        "length_median": length_median,
        "radius_mean_global": r_mean_global,
        "radius_median_global": r_median_global,
        "bounding_box": {
            "min": bb_min.tolist(),
            "max": bb_max.tolist(),
            "extents": bb_ext.tolist(),
        },
    }
    return metrics


def compute_flow_metrics(
    G: nx.Graph,
    mu: float = 1.0e-3,
    rho: float = 1000.0,
    Q_attr: str = "Q",
) -> Dict[str, Any]:
    """
    Compute flow-related metrics on a Poiseuille-solved centerline graph.

    Requires per-edge:
        - attribute Q_attr (default 'Q') : volumetric flow rate
    Requires per-node or per-edge:
        - 'radius' at nodes or edges
        - 'length' at edges or inferred from node coords

    Returns
    -------
    metrics : dict with arrays:
        - Q            : |Q| per edge
        - radius_mean  : radius per edge
        - length       : length per edge
        - u_mean       : mean velocity per edge
        - Re           : Reynolds number per edge
        - R_hyd        : hydraulic resistance per edge
      and summary scalars like:
        - Q_total_in, Q_total_out
        - mass_balance_rel_error
        - Re_max, Re_median, etc.
    """
    # prepare node radii & coords
    coords = {}
    radii = {}
    for n, data in G.nodes(data=True):
        c = np.asarray(data.get("coord", [0.0, 0.0, 0.0]), dtype=float)
        coords[n] = c
        r = data.get("radius", None)
        if r is not None:
            radii[n] = float(r)

    Q_arr = []
    R_arr = []
    L_arr = []
    u_arr = []
    Re_arr = []
    R_hyd_arr = []

    for u, v, data in G.edges(data=True):
        Q_val = data.get(Q_attr, None)
        if Q_val is None:
            Q_val = np.nan
        else:
            Q_val = float(Q_val)
        Q_arr.append(abs(Q_val))

        # radius
        ru = radii.get(u, None)
        rv = radii.get(v, None)
        re = data.get("radius", None)
        if re is not None:
            r_edge = float(re)
        elif (ru is not None) and (rv is not None):
            r_edge = 0.5 * (ru + rv)
        elif ru is not None:
            r_edge = float(ru)
        elif rv is not None:
            r_edge = float(rv)
        else:
            r_edge = np.nan
        R_arr.append(r_edge)

        # length
        L = data.get("length", None)
        if L is None:
            cu = coords[u]
            cv = coords[v]
            L = float(np.linalg.norm(cu - cv))
        else:
            L = float(L)
        L_arr.append(L)

        if np.isfinite(r_edge) and r_edge > 0 and np.isfinite(Q_val):
            A = np.pi * r_edge**2
            u_mean = abs(Q_val) / A
        else:
            u_mean = np.nan
        u_arr.append(u_mean)

        if np.isfinite(u_mean) and np.isfinite(r_edge) and r_edge > 0:
            Re_val = rho * u_mean * (2.0 * r_edge) / mu
        else:
            Re_val = np.nan
        Re_arr.append(Re_val)

        if np.isfinite(r_edge) and r_edge > 0 and np.isfinite(L) and L > 0:
            R_hyd = (8.0 * mu * L) / (np.pi * r_edge**4)
        else:
            R_hyd = np.nan
        R_hyd_arr.append(R_hyd)

    Q_arr = np.asarray(Q_arr, dtype=float)
    R_arr = np.asarray(R_arr, dtype=float)
    L_arr = np.asarray(L_arr, dtype=float)
    u_arr = np.asarray(u_arr, dtype=float)
    Re_arr = np.asarray(Re_arr, dtype=float)
    R_hyd_arr = np.asarray(R_hyd_arr, dtype=float)

    Q_total = float(np.nansum(Q_arr))

    metrics = {
        "Q": Q_arr.tolist(),
        "radius_mean": R_arr.tolist(),
        "length": L_arr.tolist(),
        "u_mean": u_arr.tolist(),
        "Re": Re_arr.tolist(),
        "R_hyd": R_hyd_arr.tolist(),
        "Q_total": Q_total,
        "Q_mean": float(np.nanmean(Q_arr)) if Q_arr.size else np.nan,
        "Q_median": float(np.nanmedian(Q_arr)) if Q_arr.size else np.nan,
        "Re_max": float(np.nanmax(Re_arr)) if Re_arr.size else np.nan,
        "Re_mean": float(np.nanmean(Re_arr)) if Re_arr.size else np.nan,
        "Re_median": float(np.nanmedian(Re_arr)) if Re_arr.size else np.nan,
    }
    return metrics


def auto_boundaries_for_component(Gc: nx.Graph, alpha_z: float = 0.7):
    """
    Pick inlet/outlet nodes for one connected component Gc.

    Inlet: node at minimum z with highest degree (root-like).
    Outlets: all leaf nodes with z > z_min + alpha_z * (z_max - z_min).

    Returns
    -------
    inlet : int or None
    outlets : list[int]
    info : dict (z_min, z_max, alpha_z)
    """
    nodes = list(Gc.nodes())
    if not nodes:
        return None, [], {"z_min": None, "z_max": None, "alpha_z": alpha_z}

    # coords & z
    z_vals = {}
    for n in nodes:
        coord = np.asarray(Gc.nodes[n].get("coord", (0, 0, 0)), dtype=float)
        z_vals[n] = coord[2] if coord.size >= 3 else 0.0

    z_array = np.array(list(z_vals.values()), dtype=float)
    z_min, z_max = float(z_array.min()), float(z_array.max())
    dz = z_max - z_min

    # inlet candidate(s): min z
    z_eps = 0.05 * dz if dz > 0 else 0.0
    candidate_inlets = [n for n in nodes if abs(z_vals[n] - z_min) <= z_eps]
    if not candidate_inlets:
        candidate_inlets = [min(z_vals, key=z_vals.get)]

    # choose inlet with highest degree
    inlet = max(candidate_inlets, key=lambda n: Gc.degree(n))

    # outlets: leaves near the top
    leaves = [n for n in nodes if Gc.degree(n) == 1 and n != inlet]
    z_thresh = z_min + alpha_z * dz
    outlets = [n for n in leaves if z_vals[n] >= z_thresh]

    # If none classified as outlets, fall back to single leaf with max z
    if not outlets and leaves:
        outlets = [max(leaves, key=lambda n: z_vals[n])]

    info = {"z_min": z_min, "z_max": z_max, "alpha_z": alpha_z}
    return inlet, outlets, info


def solve_poiseuille_by_component(
    G: nx.Graph,
    mu: float = 1.0e-3,
    delta_p: float = 1000.0,
    alpha_z: float = 0.7,
    write_to_graph: bool = True,
):
    """
    Run Poiseuille on each connected component of G separately.

    For each component:
      - auto-pick inlet/outlets (see auto_boundaries_for_component)
      - call compute_poiseuille_network on that subgraph
      - optionally write pressures/Q back into the original G
      - store a per-component summary

    Returns
    -------
    G : networkx.Graph
        Same graph (with 'pressure' on nodes and 'Q' on edges if write_to_graph=True).
    comp_summaries : list[dict]
        One summary dict per component.
    """
    from .cfd import compute_poiseuille_network
    
    comp_summaries = []
    for comp_id, nodes in enumerate(nx.connected_components(G), start=1):
        nodes = list(nodes)
        Gc = G.subgraph(nodes).copy()
        for n in nodes:
            G.nodes[n]["component_id"] = comp_id

        inlet, outlets, z_info = auto_boundaries_for_component(Gc, alpha_z=alpha_z)

        if inlet is None or not outlets:
            comp_summaries.append(
                {
                    "component_id": comp_id,
                    "num_nodes": Gc.number_of_nodes(),
                    "num_edges": Gc.number_of_edges(),
                    "inlet": inlet,
                    "outlets": outlets,
                    "flow_solved": False,
                    "reason": "No valid inlet/outlet found",
                    "z_min": z_info["z_min"],
                    "z_max": z_info["z_max"],
                }
            )
            continue

        res = compute_poiseuille_network(
            Gc,
            mu=mu,
            inlet_nodes=[inlet],
            outlet_nodes=outlets,
            pin=delta_p,
            pout=0.0,
            write_to_graph=False,
        )

        node_pressures = res["node_pressures"]
        edge_flows = res["edge_flows"]
        summary = res["summary"]

        if write_to_graph:
            for n, P in node_pressures.items():
                G.nodes[n]["pressure"] = P
            for (u, v), Q_uv in edge_flows.items():
                if G.has_edge(u, v):
                    G.edges[u, v]["Q"] = Q_uv

        comp_summary = {
            "component_id": comp_id,
            "num_nodes": Gc.number_of_nodes(),
            "num_edges": Gc.number_of_edges(),
            "inlet": inlet,
            "outlets": outlets,
            "flow_solved": True,
            "z_min": z_info["z_min"],
            "z_max": z_info["z_max"],
            "total_inlet_flow": summary["total_inlet_flow"],
            "total_outlet_flow": summary["total_outlet_flow"],
            "warnings": summary.get("warnings", []),
        }
        comp_summaries.append(comp_summary)

    return G, comp_summaries
