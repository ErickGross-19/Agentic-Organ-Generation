"""
Shared visualization and analysis helpers for the experiment notebook.

Provides matplotlib 2-D projections, plotly interactive 3-D views,
stat summaries, and head-to-head comparison utilities.

Usage
-----
    from notebook_utils import plot_network_2d, plot_network_3d, print_stats
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from generation.core.network import VascularNetwork


def _extract_segments(
    network: VascularNetwork,
) -> List[Dict[str, Any]]:
    segments = []
    for seg in network.segments.values():
        start = network.nodes[seg.start_node_id].position
        end = network.nodes[seg.end_node_id].position
        centerline = seg.geometry.centerline_points or []
        points = [start] + centerline + [end]
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        zs = [p.z for p in points]
        segments.append({
            "xs": xs,
            "ys": ys,
            "zs": zs,
            "vessel_type": seg.vessel_type,
            "radius": seg.geometry.mean_radius(),
            "length": seg.length,
        })
    return segments


def plot_network_2d(
    network: VascularNetwork,
    projection: str = "xy",
    ax=None,
    title: Optional[str] = None,
    color_by: str = "vessel_type",
    linewidth_scale: float = 1.0,
    show_nodes: bool = True,
    figsize: Tuple[float, float] = (8, 8),
):
    """
    Plot a 2-D projection of the network with matplotlib.

    Parameters
    ----------
    network : VascularNetwork
    projection : str
        "xy" (top-down), "xz" (front), "yz" (side)
    ax : matplotlib Axes, optional
        If None a new figure is created.
    title : str, optional
    color_by : str
        "vessel_type" → red/blue, "radius" → colormap
    linewidth_scale : float
        Multiplier on radius-based linewidths.
    show_nodes : bool
        Overlay node markers (inlet=green, terminal=orange, junction=gray).
    figsize : tuple
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import matplotlib.colors as mcolors

    axis_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    if projection not in axis_map:
        raise ValueError(f"projection must be one of {list(axis_map)}")
    ai, bi = axis_map[projection]
    axis_labels = {0: "X (m)", 1: "Y (m)", 2: "Z (m)"}

    segs = _extract_segments(network)
    if not segs:
        print("Network has no segments to plot.")
        return ax

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    lines = []
    colors = []
    widths = []

    type_color = {"arterial": "tab:red", "venous": "tab:blue"}
    radii_all = [s["radius"] for s in segs]
    r_min, r_max = min(radii_all), max(radii_all)

    for s in segs:
        coords = list(zip([s["xs"], s["ys"], s["zs"]][ai],
                          [s["xs"], s["ys"], s["zs"]][bi]))
        lines.append(coords)

        if color_by == "vessel_type":
            colors.append(type_color.get(s["vessel_type"], "gray"))
        else:
            norm = (s["radius"] - r_min) / (r_max - r_min + 1e-15)
            colors.append(plt.cm.viridis(norm))

        lw = 0.5 + 3.0 * ((s["radius"] - r_min) / (r_max - r_min + 1e-15))
        widths.append(lw * linewidth_scale)

    lc = LineCollection(lines, colors=colors, linewidths=widths)
    ax.add_collection(lc)

    if show_nodes:
        node_styles = {
            "inlet": ("^", "limegreen", 60),
            "terminal": ("o", "orange", 20),
            "junction": (".", "silver", 8),
            "outlet": ("v", "dodgerblue", 60),
        }
        for node in network.nodes.values():
            marker, c, sz = node_styles.get(node.node_type, (".", "gray", 5))
            pos = node.position
            a_val = [pos.x, pos.y, pos.z][ai]
            b_val = [pos.x, pos.y, pos.z][bi]
            ax.scatter(a_val, b_val, marker=marker, c=c, s=sz, zorder=5, edgecolors="k", linewidths=0.3)

    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel(axis_labels[ai])
    ax.set_ylabel(axis_labels[bi])
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_network_3d(
    network: VascularNetwork,
    title: Optional[str] = None,
    color_by: str = "vessel_type",
    line_width: float = 3.0,
    opacity: float = 1.0,
    show_nodes: bool = True,
    width: int = 800,
    height: int = 700,
):
    """
    Interactive 3-D plot using plotly.

    Parameters
    ----------
    network : VascularNetwork
    title : str, optional
    color_by : str
        "vessel_type" or "radius"
    line_width : float
    opacity : float
    show_nodes : bool
    width, height : int
        Figure dimensions in pixels.

    Returns
    -------
    plotly Figure (call .show() to display in notebook)
    """
    import plotly.graph_objects as go

    segs = _extract_segments(network)
    if not segs:
        print("Network has no segments to plot.")
        return None

    type_color = {"arterial": "red", "venous": "blue"}
    radii_all = [s["radius"] for s in segs]
    r_min, r_max = min(radii_all), max(radii_all)

    traces = []
    for s in segs:
        if color_by == "vessel_type":
            col = type_color.get(s["vessel_type"], "gray")
        else:
            norm = (s["radius"] - r_min) / (r_max - r_min + 1e-15)
            r_int = int(255 * norm)
            col = f"rgb({255 - r_int},{r_int},{128})"

        traces.append(go.Scatter3d(
            x=s["xs"], y=s["ys"], z=s["zs"],
            mode="lines",
            line=dict(color=col, width=line_width),
            opacity=opacity,
            showlegend=False,
            hovertext=f"r={s['radius']*1e3:.3f}mm  L={s['length']*1e3:.2f}mm  {s['vessel_type']}",
            hoverinfo="text",
        ))

    if show_nodes:
        node_groups: Dict[str, Dict[str, list]] = {}
        for node in network.nodes.values():
            nt = node.node_type
            if nt not in node_groups:
                node_groups[nt] = {"x": [], "y": [], "z": [], "text": []}
            g = node_groups[nt]
            g["x"].append(node.position.x)
            g["y"].append(node.position.y)
            g["z"].append(node.position.z)
            g["text"].append(f"id={node.id} {nt} {node.vessel_type}")

        marker_map = {
            "inlet": ("diamond", "limegreen", 6),
            "terminal": ("circle", "orange", 3),
            "junction": ("circle", "lightgray", 2),
            "outlet": ("diamond", "dodgerblue", 6),
        }
        for nt, g in node_groups.items():
            sym, col, sz = marker_map.get(nt, ("circle", "gray", 2))
            traces.append(go.Scatter3d(
                x=g["x"], y=g["y"], z=g["z"],
                mode="markers",
                marker=dict(symbol=sym, color=col, size=sz, line=dict(width=0.5, color="black")),
                name=nt,
                text=g["text"],
                hoverinfo="text",
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title or "Vascular Network 3D",
        width=width,
        height=height,
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
    )
    return fig


def print_stats(stats: Dict[str, Any], label: str = "") -> None:
    """Pretty-print a stats dictionary as a table."""
    header = f"  Stats: {label}  " if label else "  Network Stats  "
    print(f"\n{'=' * 50}")
    print(f"{header:^50}")
    print(f"{'=' * 50}")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:40s}  {v:.6g}")
        else:
            print(f"  {k:40s}  {v}")
    print(f"{'=' * 50}\n")


def compare_networks(
    networks: Dict[str, Tuple[VascularNetwork, Dict[str, Any]]],
    projection: str = "xy",
    figsize_per: Tuple[float, float] = (6, 6),
):
    """
    Side-by-side 2-D comparison of multiple networks.

    Parameters
    ----------
    networks : dict
        {label: (VascularNetwork, stats_dict)}
    projection : str
    figsize_per : tuple
        Size per subplot.
    """
    import matplotlib.pyplot as plt

    n = len(networks)
    if n == 0:
        print("No networks to compare.")
        return

    fig, axes = plt.subplots(1, n, figsize=(figsize_per[0] * n, figsize_per[1]))
    if n == 1:
        axes = [axes]

    for ax, (label, (net, stats)) in zip(axes, networks.items()):
        plot_network_2d(net, projection=projection, ax=ax, title=label, show_nodes=True)
        info = (
            f"nodes={stats['nodes']}  segs={stats['segments']}  "
            f"terms={stats['terminals']}  t={stats['elapsed_seconds']:.2f}s"
        )
        ax.set_xlabel(info, fontsize=8)

    plt.tight_layout()
    plt.show()


def compare_stats_table(
    all_stats: Dict[str, Dict[str, Any]],
) -> "pandas.DataFrame":
    """
    Return a pandas DataFrame comparing stats across runs.

    Parameters
    ----------
    all_stats : dict
        {label: stats_dict}

    Returns
    -------
    DataFrame with one row per run.
    """
    import pandas as pd
    return pd.DataFrame(all_stats).T


def network_to_dataframe(network: VascularNetwork) -> "pandas.DataFrame":
    """
    Convert segment-level data to a DataFrame for analysis.

    Columns: segment_id, start_node_id, end_node_id, vessel_type,
             length_m, radius_start_m, radius_end_m, mean_radius_m
    """
    import pandas as pd
    rows = []
    for seg in network.segments.values():
        rows.append({
            "segment_id": seg.id,
            "start_node_id": seg.start_node_id,
            "end_node_id": seg.end_node_id,
            "vessel_type": seg.vessel_type,
            "length_m": seg.length,
            "radius_start_m": seg.geometry.radius_start,
            "radius_end_m": seg.geometry.radius_end,
            "mean_radius_m": seg.geometry.mean_radius(),
        })
    return pd.DataFrame(rows)


def save_network_json(network: VascularNetwork, path: str) -> str:
    """Save network to JSON. Returns the path written."""
    import json
    data = network.to_dict()
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path
