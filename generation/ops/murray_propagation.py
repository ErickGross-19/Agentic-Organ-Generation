"""
Post-hoc Murray's law radius propagation.

Propagates radii backward from terminal nodes toward the root using
Murray's law: r_parent^gamma = sum(r_child^gamma).

This replaces taper_factor-based radii from space colonization with
flow-based accumulation for biologically realistic vessel sizing.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Set, List
from collections import deque
import logging

from ..core.network import VascularNetwork
from ..analysis.radius import (
    find_root_node_id,
    identify_parent_segment_at_junction,
    segment_radius_at_node,
)

logger = logging.getLogger(__name__)


@dataclass
class MurrayPropagationResult:
    """Result of Murray's law radius propagation."""

    nodes_updated: int
    segments_updated: int
    mean_deviation_before: float
    mean_deviation_after: float
    terminal_radius_used: float


def propagate_murray_radii(
    network: VascularNetwork,
    terminal_radius: float = 0.0003,
    gamma: float = 3.0,
    vessel_type: Optional[str] = None,
) -> MurrayPropagationResult:
    """
    Post-hoc Murray's law radius assignment: propagate from terminals toward root.

    Algorithm:
    1. Find all terminal nodes
    2. Assign terminal_radius to terminal segments
    3. BFS from terminals toward root:
       At each junction: r_parent = (sum(r_child^gamma))^(1/gamma)
    4. Update segment geometry (radius_start, radius_end) accordingly
    """
    if not network.nodes or not network.segments:
        return MurrayPropagationResult(0, 0, 0.0, 0.0, terminal_radius)

    mean_dev_before = _compute_mean_murray_deviation(network, gamma, vessel_type)

    root_id = find_root_node_id(network, vessel_type=vessel_type)
    if root_id is None:
        for node in network.nodes.values():
            if node.node_type in ("inlet", "outlet"):
                if vessel_type is None or node.vessel_type == vessel_type:
                    root_id = node.id
                    break

    terminals: List[int] = []
    for node in network.nodes.values():
        if vessel_type is not None and node.vessel_type != vessel_type:
            continue
        if node.node_type == "terminal":
            terminals.append(node.id)

    if not terminals:
        return MurrayPropagationResult(0, 0, mean_dev_before, mean_dev_before, terminal_radius)

    child_radius_at_junction: Dict[int, Dict[int, float]] = {}
    children_expected: Dict[int, int] = {}
    segments_updated: Set[int] = set()
    nodes_updated: Set[int] = set()

    for node_id in terminals:
        seg_ids = list(network.get_connected_segment_ids(node_id))
        for seg_id in seg_ids:
            seg = network.segments.get(seg_id)
            if seg is None:
                continue
            if seg.start_node_id == node_id:
                seg.geometry.radius_start = terminal_radius
            else:
                seg.geometry.radius_end = terminal_radius
            segments_updated.add(seg_id)

    queue: deque = deque()

    for node_id in terminals:
        seg_ids = list(network.get_connected_segment_ids(node_id))
        for seg_id in seg_ids:
            seg = network.segments.get(seg_id)
            if seg is None:
                continue
            other_node = (
                seg.end_node_id if seg.start_node_id == node_id else seg.start_node_id
            )
            other = network.nodes.get(other_node)
            if other is None:
                continue

            if other.node_type == "junction":
                if other_node not in child_radius_at_junction:
                    child_radius_at_junction[other_node] = {}
                    n_children = len(list(network.get_connected_segment_ids(other_node))) - 1
                    children_expected[other_node] = max(n_children, 1)
                child_radius_at_junction[other_node][seg_id] = terminal_radius

                if len(child_radius_at_junction[other_node]) >= children_expected[other_node]:
                    queue.append(other_node)

    visited_junctions: Set[int] = set()

    while queue:
        junction_id = queue.popleft()
        if junction_id in visited_junctions:
            continue
        visited_junctions.add(junction_id)

        child_radii = child_radius_at_junction.get(junction_id, {})
        if not child_radii:
            continue

        child_sum = sum(r ** gamma for r in child_radii.values())
        parent_radius = child_sum ** (1.0 / gamma)

        parent_seg_id, child_seg_ids = identify_parent_segment_at_junction(
            network, junction_id, root_node_id=root_id
        )

        for child_seg_id, child_r in child_radii.items():
            child_seg = network.segments.get(child_seg_id)
            if child_seg is None:
                continue
            if child_seg.start_node_id == junction_id:
                child_seg.geometry.radius_start = child_r
            else:
                child_seg.geometry.radius_end = child_r
            segments_updated.add(child_seg_id)

        if parent_seg_id is not None:
            parent_seg = network.segments.get(parent_seg_id)
            if parent_seg is not None:
                if parent_seg.start_node_id == junction_id:
                    parent_seg.geometry.radius_start = parent_radius
                else:
                    parent_seg.geometry.radius_end = parent_radius
                segments_updated.add(parent_seg_id)

                other_node = (
                    parent_seg.end_node_id
                    if parent_seg.start_node_id == junction_id
                    else parent_seg.start_node_id
                )
                other = network.nodes.get(other_node)
                if other is not None and other.node_type == "junction":
                    if other_node not in child_radius_at_junction:
                        child_radius_at_junction[other_node] = {}
                        n_children = len(list(network.get_connected_segment_ids(other_node))) - 1
                        children_expected[other_node] = max(n_children, 1)
                    child_radius_at_junction[other_node][parent_seg_id] = parent_radius

                    if len(child_radius_at_junction[other_node]) >= children_expected[other_node]:
                        queue.append(other_node)

                elif other is not None and other.node_type in ("inlet", "outlet"):
                    if parent_seg.start_node_id == other_node:
                        parent_seg.geometry.radius_start = parent_radius
                    else:
                        parent_seg.geometry.radius_end = parent_radius

        nodes_updated.add(junction_id)

    mean_dev_after = _compute_mean_murray_deviation(network, gamma, vessel_type)

    logger.info(
        "Murray propagation: updated %d segments, %d junctions. "
        "Deviation %.4f -> %.4f",
        len(segments_updated),
        len(nodes_updated),
        mean_dev_before,
        mean_dev_after,
    )

    return MurrayPropagationResult(
        nodes_updated=len(nodes_updated),
        segments_updated=len(segments_updated),
        mean_deviation_before=mean_dev_before,
        mean_deviation_after=mean_dev_after,
        terminal_radius_used=terminal_radius,
    )


def _compute_mean_murray_deviation(
    network: VascularNetwork,
    gamma: float,
    vessel_type: Optional[str],
) -> float:
    from ..analysis.radius import compute_murray_deviation_at_junction

    root_id = find_root_node_id(network, vessel_type=vessel_type)
    deviations = []
    for node in network.nodes.values():
        if node.node_type != "junction":
            continue
        if vessel_type is not None and node.vessel_type != vessel_type:
            continue
        dev = compute_murray_deviation_at_junction(
            network, node.id, gamma=gamma, root_node_id=root_id
        )
        if dev is not None:
            deviations.append(dev)
    return float(sum(deviations) / len(deviations)) if deviations else 0.0
