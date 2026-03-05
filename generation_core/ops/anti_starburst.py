"""
Anti-starburst branching enforcement for ODC.

Four mechanisms to prevent linear starburst growth and enforce
hierarchical tree structure:
1. Generation-based tissue visibility
2. Maximum initial branches
3. Branching quota enforcement
4. Exploration vs exploitation mode

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .odc_state import ODCState
    from .odc_params import ODCParams
    from ..core.domain import DomainSpec


def compute_tissue_depths(
    tissue_points: np.ndarray,
    domain: "DomainSpec",
    depth_axis: int = 2,
) -> np.ndarray:
    """
    Compute normalized depth (0-1) for each tissue point.

    0 = closest to inlet (top), 1 = deepest (bottom).
    """
    if len(tissue_points) == 0:
        return np.array([], dtype=np.float64)

    bounds = domain.get_bounds()

    if depth_axis == 0:
        ax_min, ax_max = bounds[0], bounds[1]
        vals = tissue_points[:, 0]
    elif depth_axis == 1:
        ax_min, ax_max = bounds[2], bounds[3]
        vals = tissue_points[:, 1]
    else:
        ax_min, ax_max = bounds[4], bounds[5]
        vals = tissue_points[:, 2]

    span = ax_max - ax_min
    if span < 1e-12:
        return np.zeros(len(tissue_points), dtype=np.float64)

    depths = (vals - ax_min) / span
    depths = 1.0 - depths
    return np.clip(depths, 0.0, 1.0)


def get_visible_tissue_mask(
    state: "ODCState",
    node_generation: int,
    params: "ODCParams",
) -> np.ndarray:
    """
    Compute which tissue points are visible to a node at given generation.

    Returns boolean mask of visible tissue points.
    """
    if not params.progressive_tissue_reveal:
        return state.active_tissue_mask.copy()

    if node_generation < params.min_generations_before_tissue:
        return np.zeros(len(state.all_tissue_points), dtype=bool)

    max_visible_depth = params.reveal_depth_per_generation * node_generation

    tissue_depths = compute_tissue_depths(
        state.all_tissue_points,
        state.network.domain,
    )

    depth_mask = tissue_depths <= max_visible_depth
    visible_mask = state.active_tissue_mask & depth_mask

    return visible_mask


def enforce_max_initial_branches(
    state: "ODCState",
    inlet_node_id: int,
    params: "ODCParams",
) -> bool:
    """
    Check if inlet can create more branches.

    Returns True if branching is allowed, False if limit reached.
    """
    children = state.get_children(inlet_node_id)
    return len(children) < params.max_initial_branches


def select_growth_direction_with_branching_quota(
    state: "ODCState",
    tip_node_id: int,
    nearby_attractors: np.ndarray,
    params: "ODCParams",
) -> Tuple[np.ndarray, bool]:
    """
    Select growth direction while enforcing branching quota.

    Returns (direction, should_bifurcate).
    """
    path_length = state.compute_path_length_to_inlet(tip_node_id)
    expected_bifurcations = path_length * params.branching_quota_per_length
    actual_bifurcations = state.count_bifurcations_on_path(tip_node_id)

    should_bifurcate = actual_bifurcations < expected_bifurcations

    tip_node = state.network.get_node(tip_node_id)
    if tip_node is None or len(nearby_attractors) == 0:
        return np.array([0.0, 0.0, -1.0]), False

    tip_pos = tip_node.position.to_array()

    if should_bifurcate and len(nearby_attractors) >= 2:
        directions = nearby_attractors - tip_pos
        norms = np.linalg.norm(directions, axis=1)
        valid = norms > 1e-10
        if np.sum(valid) >= 2:
            valid_dirs = directions[valid] / norms[valid, np.newaxis]
            mid = np.mean(valid_dirs, axis=0)
            norm_mid = np.linalg.norm(mid)
            if norm_mid > 1e-10:
                mid = mid / norm_mid
            return mid, True

    directions = nearby_attractors - tip_pos
    norms = np.linalg.norm(directions, axis=1)
    valid = norms > 1e-10
    if not np.any(valid):
        return np.array([0.0, 0.0, -1.0]), False

    avg_dir = np.mean(directions[valid] / norms[valid, np.newaxis], axis=0)
    norm = np.linalg.norm(avg_dir)
    if norm > 1e-10:
        avg_dir = avg_dir / norm
    return avg_dir, False


def check_generation_requirements(
    state: "ODCState",
    params: "ODCParams",
) -> Dict[str, Any]:
    """
    Check if tree meets minimum generation depth requirements.
    """
    max_gen = 0
    tips_below: List[int] = []

    for tip_id in state.sc_state.active_tip_ids:
        tip_gen = state.get_node_generation(tip_id)
        max_gen = max(max_gen, tip_gen)
        if tip_gen < params.force_bifurcation_depth:
            tips_below.append(tip_id)

    return {
        "meets_requirements": max_gen >= params.force_bifurcation_depth,
        "current_max_generation": max_gen,
        "required_generation": params.force_bifurcation_depth,
        "tips_below_requirement": tips_below,
    }


def enforce_minimum_generations(
    state: "ODCState",
    params: "ODCParams",
) -> None:
    """
    For tips below generation requirement, mark them for forced bifurcation
    and exploration-only mode.
    """
    gen_status = check_generation_requirements(state, params)

    for tip_id in gen_status["tips_below_requirement"]:
        state.force_bifurcate_nodes.add(tip_id)
        state.exploration_mode_nodes.add(tip_id)


def compute_growth_direction_exploration(
    tip_pos: np.ndarray,
    nearby_attractors: np.ndarray,
    sibling_tips: List[np.ndarray],
    params: "ODCParams",
) -> np.ndarray:
    """
    Exploration mode: spread out, avoid siblings, don't converge on tissue.
    """
    repulsion = np.zeros(3)
    for sibling_pos in sibling_tips:
        diff = tip_pos - sibling_pos
        dist = np.linalg.norm(diff) + 1e-10
        repulsion += diff / (dist ** 2)

    if np.linalg.norm(repulsion) > 0:
        repulsion = repulsion / np.linalg.norm(repulsion)

    if len(nearby_attractors) > 0:
        centroid = np.mean(nearby_attractors, axis=0)
        attraction = centroid - tip_pos
        norm = np.linalg.norm(attraction)
        if norm > 1e-10:
            attraction = attraction / norm
        else:
            attraction = np.zeros(3)
    else:
        attraction = np.zeros(3)

    exploration_weight = 0.6
    direction = (1 - exploration_weight) * attraction + exploration_weight * repulsion

    if params.preferred_direction is not None:
        pref = np.array(params.preferred_direction)
        norm_pref = np.linalg.norm(pref)
        if norm_pref > 1e-10:
            pref = pref / norm_pref
        direction = (1 - params.directional_bias) * direction + params.directional_bias * pref

    norm = np.linalg.norm(direction)
    if norm > 1e-10:
        direction = direction / norm
    else:
        direction = np.array([0.0, 0.0, -1.0])

    return direction
