import numpy as np
import networkx as nx
from math import pi
from typing import Optional, Dict, Any


def compute_poiseuille_network(
    G: nx.Graph,
    mu: float = 1.0,
    inlet_nodes=None,
    outlet_nodes=None,
    pin: float = 1.0,
    pout: float = 0.0,
    write_to_graph: bool = True,
):
    """
    Solve a Poiseuille flow network on a centerline graph.

    Auto-BC mode:
      - If inlet_nodes and outlet_nodes are both None:
          * find all degree-1 nodes (leaves)
          * choose inlet = leaf with minimum z
          * choose outlet = leaf with maximum z
    """

    warnings_list: list[str] = []

    components = list(nx.connected_components(G))

    edge_conductance: dict[tuple[int, int], float] = {}

    def edge_G(u, v) -> float:
        data = G.edges[u, v]

        ru = G.nodes[u].get("radius", None)
        rv = G.nodes[v].get("radius", None)
        r_edge = data.get("radius", None)

        if r_edge is not None:
            r = float(r_edge)
        elif ru is None and rv is None:
            return 0.0
        elif ru is None:
            r = float(rv)
        elif rv is None:
            r = float(ru)
        else:
            r = 0.5 * (float(ru) + float(rv))

        if r <= 0.0:
            return 0.0

        length = data.get("length", None)
        if length is None:
            cu = np.asarray(G.nodes[u].get("coord", [0, 0, 0]), dtype=float)
            cv = np.asarray(G.nodes[v].get("coord", [0, 0, 0]), dtype=float)
            length = float(np.linalg.norm(cu - cv))
        length = max(float(length), 1e-8)

        return (pi * r**4) / (8.0 * mu * length)

    for u, v in G.edges:
        G_uv = edge_G(u, v)
        edge_conductance[(u, v)] = G_uv
        edge_conductance[(v, u)] = G_uv

    leaves = [n for n in G.nodes if G.degree(n) == 1]

    if inlet_nodes is None and outlet_nodes is None:
        if len(leaves) < 2:
            warnings_list.append(
                "Less than two leaf nodes; cannot auto-select inlet/outlet."
            )
            inlet_nodes = []
            outlet_nodes = []
        else:
            leaves_with_z = []
            for n in leaves:
                coord = np.asarray(G.nodes[n].get("coord", [0, 0, 0]), dtype=float)
                z = coord[2] if coord.size >= 3 else 0.0
                leaves_with_z.append((n, z))

            leaves_with_z.sort(key=lambda x: x[1])
            inlet_nodes = [leaves_with_z[0][0]]
            outlet_nodes = [leaves_with_z[-1][0]]
    else:
        inlet_nodes = list(inlet_nodes) if inlet_nodes is not None else []
        outlet_nodes = list(outlet_nodes) if outlet_nodes is not None else []

    inlet_nodes = [n for n in inlet_nodes if n in G]
    outlet_nodes = [n for n in outlet_nodes if n in G]

    if not inlet_nodes or not outlet_nodes:
        warnings_list.append(
            "Missing inlets or outlets after validation; system may be singular."
        )

    boundary_nodes = set(inlet_nodes) | set(outlet_nodes)
    interior_nodes = [n for n in G.nodes if n not in boundary_nodes]

    boundary_pressure = {n: float(pin) for n in inlet_nodes}
    for n in outlet_nodes:
        boundary_pressure.setdefault(n, float(pout))

    n_int = len(interior_nodes)
    used_lstsq = False

    if n_int > 0:
        idx = {n: i for i, n in enumerate(interior_nodes)}
        A = np.zeros((n_int, n_int), dtype=float)
        b = np.zeros(n_int, dtype=float)

        for n in interior_nodes:
            i = idx[n]
            for m in G.neighbors(n):
                G_nm = edge_conductance.get((n, m), 0.0)
                if G_nm == 0.0:
                    continue
                if m in interior_nodes:
                    j = idx[m]
                    A[i, i] += G_nm
                    A[i, j] -= G_nm
                else:
                    Pm = boundary_pressure.get(m, 0.0)
                    A[i, i] += G_nm
                    b[i] += G_nm * Pm

        rank_A = int(np.linalg.matrix_rank(A))
        if rank_A < A.shape[0]:
            warnings_list.append("Matrix is rank-deficient; using least-squares.")

        try:
            P_int = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            P_int, *_ = np.linalg.lstsq(A, b, rcond=None)
            used_lstsq = True

        node_pressures: dict[int, float] = {}
        for n in interior_nodes:
            node_pressures[n] = float(P_int[idx[n]])
        for n, P in boundary_pressure.items():
            node_pressures[n] = float(P)

        matrix_shape = A.shape
        matrix_rank = rank_A
    else:
        node_pressures = {
            n: float(boundary_pressure.get(n, pout)) for n in G.nodes
        }
        matrix_shape = (0, 0)
        matrix_rank = 0

    edge_flows: dict[tuple[int, int], float] = {}
    for u, v in G.edges:
        G_uv = edge_conductance.get((u, v), 0.0)
        if G_uv == 0.0:
            edge_flows[(u, v)] = 0.0
            continue
        Pu = node_pressures[u]
        Pv = node_pressures[v]
        edge_flows[(u, v)] = float(G_uv * (Pu - Pv))

    total_inlet_flow = 0.0
    total_outlet_flow = 0.0

    for n in inlet_nodes:
        flux = 0.0
        for m in G.neighbors(n):
            G_nm = edge_conductance.get((n, m), 0.0)
            if G_nm == 0.0:
                continue
            flux += G_nm * (node_pressures[n] - node_pressures[m])
        total_inlet_flow += flux

    for n in outlet_nodes:
        flux = 0.0
        for m in G.neighbors(n):
            G_nm = edge_conductance.get((n, m), 0.0)
            if G_nm == 0.0:
                continue
            flux += G_nm * (node_pressures[n] - node_pressures[m])
        total_outlet_flow += -flux

    summary = {
        "num_nodes": int(G.number_of_nodes()),
        "num_edges": int(G.number_of_edges()),
        "num_components": int(len(components)),
        "num_inlets": int(len(inlet_nodes)),
        "num_outlets": int(len(outlet_nodes)),
        "num_interior": int(len(interior_nodes)),
        "matrix_shape": tuple(matrix_shape),
        "matrix_rank": int(matrix_rank),
        "used_lstsq": bool(used_lstsq),
        "total_inlet_flow": float(total_inlet_flow),
        "total_outlet_flow": float(total_outlet_flow),
        "total_inflow": float(total_inlet_flow),
        "total_outflow": float(total_outlet_flow),
        "pin": float(pin),
        "pout": float(pout),
        "warnings": warnings_list,
        "inlet_node_ids": [str(n) for n in inlet_nodes],
        "outlet_node_ids": [str(n) for n in outlet_nodes],
    }

    if write_to_graph:
        for n, P in node_pressures.items():
            G.nodes[n]["pressure"] = P
        for (u, v), Q in edge_flows.items():
            if G.has_edge(u, v):
                G.edges[u, v]["Q"] = Q

    return {
        "node_pressures": node_pressures,
        "edge_flows": edge_flows,
        "summary": summary,
        "G": G,
    }


def ensure_poiseuille_solved(G: nx.Graph, delta_p: float = 1.0, mu: float = 1.0e-3):
    """
    If G has no 'pressure' yet, pick one inlet and one outlet (min-z, max-z
    leaf) and run Poiseuille; otherwise return G unchanged.
    """
    if any("pressure" in data for _, data in G.nodes(data=True)):
        return G

    res = compute_poiseuille_network(
        G,
        mu=mu,
        inlet_nodes=None,
        outlet_nodes=None,
        pin=delta_p,
        pout=0.0,
        write_to_graph=True,
    )
    return res["G"]
