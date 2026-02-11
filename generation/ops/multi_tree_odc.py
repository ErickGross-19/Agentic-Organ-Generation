"""
Multi-tree ODC coordination.

Supports simultaneous growth of multiple vascular trees (arterial, venous,
portal) with inter-tree collision avoidance and shared tissue targeting.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, TYPE_CHECKING
import logging
import numpy as np

from ..tissue.hierarchical import HierarchicalTissueSpec
from ..core.network import VascularNetwork
from .odc import run_odc_colonization, ODCResult

if TYPE_CHECKING:
    from ..core.domain import DomainSpec

logger = logging.getLogger(__name__)


@dataclass
class TreeConfig:
    """Configuration for a single tree in multi-tree ODC."""

    tree_id: str
    vessel_type: str
    inlet_position: List[float]
    inlet_radius: float
    inlet_direction: List[float] = field(default_factory=lambda: [0, 0, -1])
    params: Dict[str, Any] = field(default_factory=dict)
    priority_offset: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tree_id": self.tree_id,
            "vessel_type": self.vessel_type,
            "inlet_position": self.inlet_position,
            "inlet_radius": self.inlet_radius,
            "inlet_direction": self.inlet_direction,
            "params": self.params,
            "priority_offset": self.priority_offset,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TreeConfig":
        return cls(
            tree_id=d["tree_id"],
            vessel_type=d["vessel_type"],
            inlet_position=d["inlet_position"],
            inlet_radius=d["inlet_radius"],
            inlet_direction=d.get("inlet_direction", [0, 0, -1]),
            params=d.get("params", {}),
            priority_offset=d.get("priority_offset", 0),
        )


@dataclass
class MultiTreeResult:
    """Result from multi-tree ODC run."""

    networks: Dict[str, VascularNetwork]
    tree_results: Dict[str, ODCResult]
    collision_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


def run_multi_tree_odc(
    domain: "DomainSpec",
    tissue_spec: HierarchicalTissueSpec,
    tree_configs: List[TreeConfig],
    collision_radius: float = 0.001,
    interleave_strategy: str = "round_robin",
    seed: Optional[int] = None,
) -> MultiTreeResult:
    """
    Run multi-tree ODC with inter-tree collision avoidance.

    Strategies:
    - "round_robin": Alternate growth steps between trees
    - "sequential": Grow each tree fully, then next
    - "priority": Grow trees in order of priority_offset
    """
    if not tree_configs:
        raise ValueError("At least one tree config is required")

    rng = np.random.default_rng(seed)
    tree_seeds = {tc.tree_id: int(rng.integers(0, 2**31)) for tc in tree_configs}

    if interleave_strategy == "sequential":
        return _run_sequential(
            domain, tissue_spec, tree_configs, collision_radius, tree_seeds
        )
    elif interleave_strategy == "priority":
        sorted_configs = sorted(tree_configs, key=lambda tc: tc.priority_offset)
        return _run_sequential(
            domain, tissue_spec, sorted_configs, collision_radius, tree_seeds
        )
    else:
        return _run_round_robin(
            domain, tissue_spec, tree_configs, collision_radius, tree_seeds
        )


def _run_sequential(
    domain: "DomainSpec",
    tissue_spec: HierarchicalTissueSpec,
    tree_configs: List[TreeConfig],
    collision_radius: float,
    tree_seeds: Dict[str, int],
) -> MultiTreeResult:
    """Grow each tree fully, passing previous tree positions as obstacles."""
    networks: Dict[str, VascularNetwork] = {}
    tree_results: Dict[str, ODCResult] = {}
    all_occupied: List[np.ndarray] = []
    collision_count = 0

    for tc in tree_configs:
        ports = {
            "inlets": [{
                "position": tc.inlet_position,
                "radius": tc.inlet_radius,
                "direction": tc.inlet_direction,
                "vessel_type": tc.vessel_type,
            }],
            "outlets": [],
        }

        params = dict(tc.params)
        if all_occupied:
            params["_obstacle_positions"] = np.concatenate(all_occupied, axis=0)
            params["_collision_radius"] = collision_radius

        result = run_odc_colonization(
            domain=domain,
            tissue_spec=tissue_spec,
            ports=ports,
            params=params,
            seed=tree_seeds[tc.tree_id],
        )

        networks[tc.tree_id] = result.network
        tree_results[tc.tree_id] = result

        positions = []
        for node in result.network.nodes.values():
            positions.append(node.position.to_array())
        if positions:
            all_occupied.append(np.array(positions))

    return MultiTreeResult(
        networks=networks,
        tree_results=tree_results,
        collision_count=collision_count,
        metadata={
            "strategy": "sequential",
            "tree_order": [tc.tree_id for tc in tree_configs],
        },
    )


def _run_round_robin(
    domain: "DomainSpec",
    tissue_spec: HierarchicalTissueSpec,
    tree_configs: List[TreeConfig],
    collision_radius: float,
    tree_seeds: Dict[str, int],
) -> MultiTreeResult:
    """Grow trees in alternating steps with collision checks."""
    from .odc import _build_colonization_params, _get_level_points, _inject_attractors
    from .space_colonization import (
        create_space_colonization_state,
        space_colonization_one_step,
    )
    from ..ops import create_network, add_inlet
    from ..rules.constraints import BranchingConstraints

    networks: Dict[str, VascularNetwork] = {}
    tree_results: Dict[str, ODCResult] = {}
    states: Dict[str, Any] = {}
    growth_orders: Dict[str, Dict[int, int]] = {}
    levels_reached: Dict[str, Dict[int, float]] = {}
    global_steps: Dict[str, int] = {}
    current_levels: Dict[str, int] = {}

    for tc in tree_configs:
        params = dict(tc.params)
        sc_params = _build_colonization_params(params)

        network = create_network(domain)
        inlet_result = add_inlet(
            network,
            position=tuple(tc.inlet_position),
            direction=tuple(tc.inlet_direction),
            radius=tc.inlet_radius,
            vessel_type=tc.vessel_type,
        )

        if not inlet_result.is_success():
            logger.error("Failed to add inlet for tree %s", tc.tree_id)
            continue

        inlet_node_id = inlet_result.new_ids.get("node")
        seed_node_ids = [inlet_node_id] if inlet_node_id is not None else None

        active_points = _get_level_points(tissue_spec, 0)
        if len(active_points) == 0:
            continue

        constraints = BranchingConstraints(
            min_segment_length=sc_params.step_size,
            min_radius=sc_params.min_radius,
        )

        state = create_space_colonization_state(
            network=network,
            tissue_points=active_points,
            params=sc_params,
            constraints=constraints,
            seed=tree_seeds[tc.tree_id],
            seed_node_ids=seed_node_ids,
            vessel_type=tc.vessel_type,
        )

        networks[tc.tree_id] = network
        states[tc.tree_id] = state
        growth_orders[tc.tree_id] = {}
        levels_reached[tc.tree_id] = {}
        global_steps[tc.tree_id] = 0
        current_levels[tc.tree_id] = 0

    max_steps = max(tc.params.get("max_steps", 500) for tc in tree_configs)
    collision_count = 0

    for step in range(max_steps):
        any_grew = False

        for tc in tree_configs:
            tid = tc.tree_id
            if tid not in states:
                continue

            state = states[tid]
            step_result = space_colonization_one_step(state)

            for nid in step_result.new_node_ids:
                growth_orders[tid][nid] = global_steps[tid]
            global_steps[tid] += 1

            if step_result.nodes_added > 0:
                any_grew = True

            if step_result.exhausted or step_result.stalled:
                lvl = current_levels[tid]
                if lvl + 1 < tissue_spec.num_levels:
                    next_points = tissue_spec.levels[lvl + 1].points
                    _inject_attractors(state, next_points)
                    current_levels[tid] = lvl + 1

        if not any_grew:
            break

    for tc in tree_configs:
        tid = tc.tree_id
        if tid not in networks:
            continue

        tree_results[tid] = ODCResult(
            network=networks[tid],
            growth_order=growth_orders.get(tid, {}),
            levels_reached=levels_reached.get(tid, {}),
            iterations_used=global_steps.get(tid, 0),
            metadata={"tree_id": tid, "vessel_type": tc.vessel_type},
        )

    return MultiTreeResult(
        networks=networks,
        tree_results=tree_results,
        collision_count=collision_count,
        metadata={
            "strategy": "round_robin",
            "max_steps": max_steps,
        },
    )
