"""
ODC algorithm state with generation tracking.

Extends SpaceColonizationState with per-node generation depth,
tissue visibility tracking, and exploration/exploitation mode support.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field
from typing import Dict, Set, Optional, List, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from ..core.network import VascularNetwork

from .space_colonization import SpaceColonizationState


@dataclass
class ODCState:
    """Extended state for ODC algorithm with generation tracking."""

    sc_state: SpaceColonizationState
    node_generations: Dict[int, int] = field(default_factory=dict)
    force_bifurcate_nodes: Set[int] = field(default_factory=set)
    exploration_mode_nodes: Set[int] = field(default_factory=set)
    tissue_claimed: Dict[int, int] = field(default_factory=dict)
    current_level_idx: int = 0
    levels_unlocked: Dict[int, bool] = field(default_factory=dict)
    stall_counter: int = 0
    global_step: int = 0
    growth_order: Dict[int, int] = field(default_factory=dict)

    @property
    def network(self) -> "VascularNetwork":
        return self.sc_state.network

    @property
    def active_tissue_mask(self) -> np.ndarray:
        n = len(self.sc_state.tissue_points)
        mask = np.zeros(n, dtype=bool)
        for idx in self.sc_state.active_tissue_indices:
            if idx < n:
                mask[idx] = True
        return mask

    @property
    def all_tissue_points(self) -> np.ndarray:
        return self.sc_state.tissue_points

    def get_node_generation(self, node_id: int) -> int:
        return self.node_generations.get(node_id, 0)

    def set_node_generation(self, node_id: int, generation: int) -> None:
        self.node_generations[node_id] = generation

    def get_children(self, node_id: int) -> List[int]:
        children = []
        for seg in self.network.segments.values():
            if seg.start_node_id == node_id:
                children.append(seg.end_node_id)
        return children

    def get_parent(self, node_id: int) -> Optional[int]:
        for seg in self.network.segments.values():
            if seg.end_node_id == node_id:
                return seg.start_node_id
        return None

    def compute_path_length_to_inlet(self, node_id: int) -> float:
        total = 0.0
        current = node_id
        visited = set()
        while current is not None and current not in visited:
            visited.add(current)
            parent = self.get_parent(current)
            if parent is None:
                break
            parent_node = self.network.get_node(parent)
            current_node = self.network.get_node(current)
            if parent_node is not None and current_node is not None:
                total += float(np.linalg.norm(
                    current_node.position.to_array() - parent_node.position.to_array()
                ))
            current = parent
        return total

    def count_bifurcations_on_path(self, node_id: int) -> int:
        count = 0
        current = node_id
        visited = set()
        while current is not None and current not in visited:
            visited.add(current)
            parent = self.get_parent(current)
            if parent is None:
                break
            children_of_parent = self.get_children(parent)
            if len(children_of_parent) >= 2:
                count += 1
            current = parent
        return count

    def get_sibling_tip_positions(self, node_id: int) -> List[np.ndarray]:
        positions = []
        for tip_id in self.sc_state.active_tip_ids:
            if tip_id != node_id:
                tip_node = self.network.get_node(tip_id)
                if tip_node is not None:
                    positions.append(tip_node.position.to_array())
        return positions
