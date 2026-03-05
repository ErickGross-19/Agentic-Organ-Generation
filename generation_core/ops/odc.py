"""
Core Optimized Directed Colonization (ODC) algorithm.

Priority-weighted space colonization that grows vascular networks toward
hierarchical tissue targets, unlocking lower-priority levels only after
higher-priority levels achieve sufficient coverage.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, TYPE_CHECKING
import logging
import numpy as np

from .space_colonization import (
    SpaceColonizationParams,
    SpaceColonizationState,
    create_space_colonization_state,
    space_colonization_one_step,
)
from ..tissue.hierarchical import TissueLevel, HierarchicalTissueSpec
from ..core.network import VascularNetwork

if TYPE_CHECKING:
    from ..core.domain import DomainSpec

logger = logging.getLogger(__name__)


@dataclass
class ODCResult:
    """Result from ODC colonization run."""

    network: VascularNetwork
    growth_order: Dict[int, int]
    levels_reached: Dict[int, float]
    iterations_used: int
    metadata: Dict[str, Any] = field(default_factory=dict)


def run_odc_colonization(
    domain: "DomainSpec",
    tissue_spec: HierarchicalTissueSpec,
    ports: Dict[str, Any],
    params: Dict[str, Any],
    seed: Optional[int] = None,
) -> ODCResult:
    """
    Core ODC algorithm: priority-weighted space colonization.

    Algorithm:
    1. Start with tissue_spec.levels[0] (highest priority) as active attractors
    2. Run space colonization using SpaceColonizationParams built from params
    3. When coverage of current level exceeds level.coverage_threshold:
       unlock next level by adding its points to active attractors
    4. Track growth_order: which step each node was created
    5. Continue until all levels exhausted or max_steps reached
    """
    from ..ops import create_network, add_inlet
    from ..rules.constraints import BranchingConstraints

    errors = tissue_spec.validate()
    if errors:
        logger.warning("Tissue spec validation warnings: %s", errors)

    sc_params = _build_colonization_params(params)

    inlets = ports.get("inlets", [])
    if not inlets:
        raise ValueError("At least one inlet is required in ports")

    inlet = inlets[0]
    inlet_pos = inlet.get("position", [0, 0, 0])
    if isinstance(inlet_pos, list):
        inlet_pos = tuple(inlet_pos)
    inlet_radius = inlet.get("radius", 0.002)
    inlet_direction = inlet.get("direction", inlet.get("growth_inward_direction", [0, 0, -1]))
    if isinstance(inlet_direction, list):
        inlet_direction = tuple(inlet_direction)
    vessel_type = inlet.get("vessel_type", sc_params.vessel_type)

    network = create_network(domain)
    inlet_result = add_inlet(
        network,
        position=inlet_pos,
        direction=inlet_direction,
        radius=inlet_radius,
        vessel_type=vessel_type,
    )

    if not inlet_result.is_success():
        logger.error("Failed to add inlet: %s", inlet_result.errors)
        return ODCResult(
            network=network,
            growth_order={},
            levels_reached={},
            iterations_used=0,
            metadata={"error": "inlet_creation_failed"},
        )

    inlet_node_id = inlet_result.new_ids.get("node")
    seed_node_ids = [inlet_node_id] if inlet_node_id is not None else None

    growth_order: Dict[int, int] = {}
    levels_reached: Dict[int, float] = {}
    global_step = 0
    max_steps = params.get("max_steps", sc_params.max_steps)

    current_level_idx = 0
    active_points = _get_level_points(tissue_spec, current_level_idx)

    if len(active_points) == 0:
        return ODCResult(
            network=network,
            growth_order=growth_order,
            levels_reached=levels_reached,
            iterations_used=0,
            metadata={"warning": "no_tissue_points"},
        )

    constraints = BranchingConstraints(
        min_segment_length=sc_params.step_size,
        min_radius=sc_params.min_radius,
    )

    state = create_space_colonization_state(
        network=network,
        tissue_points=active_points,
        params=sc_params,
        constraints=constraints,
        seed=seed,
        seed_node_ids=seed_node_ids,
        vessel_type=vessel_type,
    )

    levels_unlocked = {0: True}
    stall_counter = 0
    max_stall = params.get("max_stall_steps", 30)

    logger.info(
        "ODC starting: %d levels, %d total points, max_steps=%d",
        tissue_spec.num_levels,
        tissue_spec.total_points,
        max_steps,
    )

    for step in range(max_steps):
        step_result = space_colonization_one_step(state)

        for nid in step_result.new_node_ids:
            growth_order[nid] = global_step

        global_step += 1

        if step_result.nodes_added == 0:
            stall_counter += 1
        else:
            stall_counter = 0

        level_unlocked_this_step = False
        if current_level_idx < tissue_spec.num_levels:
            level = tissue_spec.levels[current_level_idx]
            coverage = _check_level_coverage(network, level)
            levels_reached[level.priority] = coverage

            if coverage >= level.coverage_threshold or stall_counter >= max_stall:
                next_idx = current_level_idx + 1
                if next_idx < tissue_spec.num_levels:
                    new_points = tissue_spec.levels[next_idx].points
                    _inject_attractors(state, new_points)
                    levels_unlocked[next_idx] = True
                    current_level_idx = next_idx
                    stall_counter = 0
                    level_unlocked_this_step = True

                    logger.info(
                        "Level %d unlocked (coverage=%.3f, step=%d). "
                        "Added %d new attractors.",
                        next_idx,
                        coverage,
                        global_step,
                        len(new_points),
                    )
                elif stall_counter >= max_stall:
                    logger.info(
                        "All levels processed. Stopping at step %d.",
                        global_step,
                    )
                    break

        if not level_unlocked_this_step and (step_result.exhausted or step_result.stalled):
            if current_level_idx + 1 < tissue_spec.num_levels:
                next_idx = current_level_idx + 1
                new_points = tissue_spec.levels[next_idx].points
                _inject_attractors(state, new_points)
                levels_unlocked[next_idx] = True
                current_level_idx = next_idx
                stall_counter = 0
            else:
                break

    for i, level in enumerate(tissue_spec.levels):
        if level.priority not in levels_reached:
            coverage = _check_level_coverage(network, level)
            levels_reached[level.priority] = coverage

    metadata = {
        "levels_unlocked": levels_unlocked,
        "total_steps": global_step,
        "final_stall_counter": stall_counter,
        "params_used": params,
        "nodes_created": len(growth_order),
    }

    logger.info(
        "ODC complete: %d steps, %d nodes, levels_reached=%s",
        global_step,
        len(growth_order),
        {k: f"{v:.3f}" for k, v in levels_reached.items()},
    )

    return ODCResult(
        network=network,
        growth_order=growth_order,
        levels_reached=levels_reached,
        iterations_used=global_step,
        metadata=metadata,
    )


def _build_colonization_params(params: Dict[str, Any]) -> SpaceColonizationParams:
    """Convert ODC param dict to SpaceColonizationParams for the inner loop."""
    return SpaceColonizationParams(
        influence_radius=params.get("influence_radius", 0.015),
        kill_radius=params.get("kill_radius", 0.003),
        step_size=params.get("step_size", 0.005),
        min_radius=params.get("min_radius", 0.0003),
        taper_factor=params.get("taper_factor", 0.95),
        vessel_type=params.get("vessel_type", "arterial"),
        max_steps=params.get("max_steps", 500),
        smoothing_weight=params.get("smoothing_weight", 0.4),
        encourage_bifurcation=True,
        min_attractions_for_bifurcation=params.get("min_attractions_for_bifurcation", 3),
        max_children_per_node=params.get("max_children_per_node", 2),
        bifurcation_angle_threshold_deg=params.get("bifurcation_angle_threshold", 40.0),
        bifurcation_probability=params.get("bifurcation_probability", 0.7),
        max_curvature_deg=params.get("max_curvature_deg", 45.0),
        min_clearance=params.get("min_clearance"),
    )


def _get_level_points(
    tissue_spec: HierarchicalTissueSpec,
    level_idx: int,
) -> np.ndarray:
    """Get tissue points for a specific level index."""
    if level_idx < 0 or level_idx >= tissue_spec.num_levels:
        return np.empty((0, 3), dtype=np.float64)
    return tissue_spec.levels[level_idx].points


def _check_level_coverage(
    network: VascularNetwork,
    level: TissueLevel,
) -> float:
    """Quick coverage check for a single level (fraction of points reached)."""
    if level.num_points == 0:
        return 1.0

    node_positions = []
    for node in network.nodes.values():
        node_positions.append(node.position.to_array())

    if not node_positions:
        return 0.0

    node_positions_arr = np.array(node_positions)

    reached = 0
    for pt in level.points:
        dists = np.linalg.norm(node_positions_arr - pt, axis=1)
        if float(np.min(dists)) < level.coverage_threshold:
            reached += 1

    return reached / level.num_points


def _inject_attractors(
    state: SpaceColonizationState,
    new_points: np.ndarray,
) -> None:
    """Inject new attractor points into an active SpaceColonizationState."""
    if len(new_points) == 0:
        return

    old_count = len(state.tissue_points)
    state.tissue_points = np.concatenate(
        [state.tissue_points, new_points], axis=0
    )

    new_indices = set(range(old_count, old_count + len(new_points)))
    state.active_tissue_indices.update(new_indices)
    state.initial_tissue_count = len(state.tissue_points)

    state.rebuild_all_nodes_kdtree()
    state.rebuild_tip_kdtree()
