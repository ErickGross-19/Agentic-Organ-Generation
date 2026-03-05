"""
Discrete topology optimization via subtree swapping.

Implements Jessen-style topology optimization where subtrees are
swapped between different parent segments and accepted/rejected
using simulated annealing.

Swap operator:
1. Select two edges/attachment points
2. Detach subtree A from its parent
3. Detach subtree B from its parent
4. Swap attachments
5. Re-optimize geometry via NLP
6. Accept/reject based on objective change

Pruning rules:
- Reject if creates a cycle
- Reject if new segment length > 2x current length

Acceptance rule (simulated annealing):
- Always accept if delta < 0 (improvement)
- Accept with probability exp(-delta/T) otherwise

Note: The library uses METERS internally for all geometry.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
import numpy as np
import math
import random

from ..core.network import VascularNetwork, Node, VesselSegment
from ..core.types import Point3D
from .nlp_geometry import NLPConfig, optimize_geometry, _compute_total_volume


@dataclass
class TopologyConfig:
    """Configuration for topology optimization."""
    
    initial_temperature: float = 1.0
    final_temperature: float = 0.01
    cooling_rate: float = 0.95
    
    max_swaps: int = 100
    swaps_per_temperature: int = 10
    
    max_length_ratio: float = 2.0
    
    min_subtree_size: int = 1
    max_subtree_size: int = 100
    
    run_nlp_after_swap: bool = True
    nlp_config: Optional[NLPConfig] = None
    
    seed: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "initial_temperature": self.initial_temperature,
            "final_temperature": self.final_temperature,
            "cooling_rate": self.cooling_rate,
            "max_swaps": self.max_swaps,
            "swaps_per_temperature": self.swaps_per_temperature,
            "max_length_ratio": self.max_length_ratio,
            "min_subtree_size": self.min_subtree_size,
            "max_subtree_size": self.max_subtree_size,
            "run_nlp_after_swap": self.run_nlp_after_swap,
        }


@dataclass
class TopologyResult:
    """Result of topology optimization."""
    
    success: bool = False
    
    initial_objective: float = 0.0
    final_objective: float = 0.0
    objective_reduction: float = 0.0
    
    swaps_attempted: int = 0
    swaps_accepted: int = 0
    swaps_rejected_cycle: int = 0
    swaps_rejected_length: int = 0
    swaps_rejected_sa: int = 0
    
    final_temperature: float = 0.0
    
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "initial_objective": self.initial_objective,
            "final_objective": self.final_objective,
            "objective_reduction": self.objective_reduction,
            "swaps_attempted": self.swaps_attempted,
            "swaps_accepted": self.swaps_accepted,
            "swaps_rejected_cycle": self.swaps_rejected_cycle,
            "swaps_rejected_length": self.swaps_rejected_length,
            "swaps_rejected_sa": self.swaps_rejected_sa,
            "final_temperature": self.final_temperature,
            "warnings": self.warnings,
            "errors": self.errors,
        }


class SubtreeInfo:
    """Information about a subtree rooted at a node."""
    
    def __init__(self, root_node_id: int, parent_segment_id: int):
        self.root_node_id = root_node_id
        self.parent_segment_id = parent_segment_id
        self.node_ids: Set[int] = set()
        self.segment_ids: Set[int] = set()
        self.terminal_count: int = 0
    
    @property
    def size(self) -> int:
        return len(self.node_ids)


def _find_subtree(
    network: VascularNetwork,
    root_node_id: int,
    parent_segment_id: int,
) -> SubtreeInfo:
    """
    Find all nodes and segments in subtree rooted at given node.
    
    Traverses downstream from root_node_id, excluding the parent segment.
    """
    subtree = SubtreeInfo(root_node_id, parent_segment_id)
    
    visited = set()
    queue = [root_node_id]
    
    while queue:
        node_id = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)
        subtree.node_ids.add(node_id)
        
        node = network.nodes[node_id]
        if node.node_type in ["terminal", "outlet"]:
            subtree.terminal_count += 1
        
        connected_segs = network.get_connected_segment_ids(node_id)
        
        for seg_id in connected_segs:
            if seg_id == parent_segment_id:
                continue
            
            segment = network.segments[seg_id]
            subtree.segment_ids.add(seg_id)
            
            if segment.start_node_id == node_id:
                other_node = segment.end_node_id
            else:
                other_node = segment.start_node_id
            
            if other_node not in visited:
                queue.append(other_node)
    
    return subtree


def _find_ancestors(
    network: VascularNetwork,
    node_id: int,
) -> Set[int]:
    """Find all ancestor node IDs (path to root)."""
    ancestors = set()
    
    visited = set()
    current = node_id
    
    while current is not None and current not in visited:
        visited.add(current)
        
        node = network.nodes.get(current)
        if node is None:
            break
        
        if node.node_type == "inlet":
            break
        
        connected_segs = network.get_connected_segment_ids(current)
        
        parent_found = False
        for seg_id in connected_segs:
            segment = network.segments[seg_id]
            
            if segment.end_node_id == current:
                ancestors.add(segment.start_node_id)
                current = segment.start_node_id
                parent_found = True
                break
        
        if not parent_found:
            break
    
    return ancestors


def _would_create_cycle(
    network: VascularNetwork,
    subtree_a: SubtreeInfo,
    subtree_b: SubtreeInfo,
) -> bool:
    """
    Check if swapping subtrees would create a cycle.
    
    A cycle is created if one subtree's root is an ancestor of the other.
    """
    ancestors_a = _find_ancestors(network, subtree_a.root_node_id)
    ancestors_b = _find_ancestors(network, subtree_b.root_node_id)
    
    if subtree_b.root_node_id in ancestors_a:
        return True
    if subtree_a.root_node_id in ancestors_b:
        return True
    
    if subtree_a.root_node_id in subtree_b.node_ids:
        return True
    if subtree_b.root_node_id in subtree_a.node_ids:
        return True
    
    return False


def _compute_new_segment_length(
    network: VascularNetwork,
    parent_segment_id: int,
    new_child_node_id: int,
) -> float:
    """Compute length of segment if child node is changed."""
    segment = network.segments[parent_segment_id]
    parent_node_id = segment.start_node_id
    
    parent_pos = network.nodes[parent_node_id].position.to_array()
    child_pos = network.nodes[new_child_node_id].position.to_array()
    
    return float(np.linalg.norm(child_pos - parent_pos))


def perform_subtree_swap(
    network: VascularNetwork,
    subtree_a: SubtreeInfo,
    subtree_b: SubtreeInfo,
) -> bool:
    """
    Perform subtree swap operation.
    
    Swaps the attachment points of two subtrees.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to modify (in place)
    subtree_a : SubtreeInfo
        First subtree
    subtree_b : SubtreeInfo
        Second subtree
        
    Returns
    -------
    bool
        True if swap was successful
    """
    seg_a = network.segments.get(subtree_a.parent_segment_id)
    seg_b = network.segments.get(subtree_b.parent_segment_id)
    
    if seg_a is None or seg_b is None:
        return False
    
    if seg_a.end_node_id == subtree_a.root_node_id:
        seg_a.end_node_id = subtree_b.root_node_id
    else:
        seg_a.start_node_id = subtree_b.root_node_id
    
    if seg_b.end_node_id == subtree_b.root_node_id:
        seg_b.end_node_id = subtree_a.root_node_id
    else:
        seg_b.start_node_id = subtree_a.root_node_id
    
    _update_segment_geometry(network, subtree_a.parent_segment_id)
    _update_segment_geometry(network, subtree_b.parent_segment_id)
    
    return True


def _update_segment_geometry(network: VascularNetwork, segment_id: int) -> None:
    """Update segment geometry after endpoint change."""
    segment = network.segments[segment_id]
    
    start_pos = network.nodes[segment.start_node_id].position
    end_pos = network.nodes[segment.end_node_id].position
    
    segment.geometry.start = start_pos
    segment.geometry.end = end_pos


def _undo_subtree_swap(
    network: VascularNetwork,
    subtree_a: SubtreeInfo,
    subtree_b: SubtreeInfo,
) -> None:
    """Undo a subtree swap by swapping back."""
    perform_subtree_swap(network, subtree_b, subtree_a)


def _select_swap_candidates(
    network: VascularNetwork,
    config: TopologyConfig,
    rng: random.Random,
) -> Optional[Tuple[SubtreeInfo, SubtreeInfo]]:
    """
    Select two subtrees for potential swap.
    
    Returns None if no valid candidates found.
    """
    junction_nodes = [
        n for n in network.nodes.values()
        if n.node_type == "junction"
    ]
    
    if len(junction_nodes) < 2:
        return None
    
    candidates = []
    
    for node in junction_nodes:
        connected_segs = network.get_connected_segment_ids(node.id)
        
        for seg_id in connected_segs:
            segment = network.segments[seg_id]
            
            if segment.start_node_id == node.id:
                subtree = _find_subtree(network, segment.end_node_id, seg_id)
                
                if config.min_subtree_size <= subtree.size <= config.max_subtree_size:
                    candidates.append(subtree)
    
    if len(candidates) < 2:
        return None
    
    rng.shuffle(candidates)
    
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            subtree_a = candidates[i]
            subtree_b = candidates[j]
            
            if subtree_a.parent_segment_id == subtree_b.parent_segment_id:
                continue
            
            if not _would_create_cycle(network, subtree_a, subtree_b):
                return subtree_a, subtree_b
    
    return None


def _check_length_constraint(
    network: VascularNetwork,
    subtree_a: SubtreeInfo,
    subtree_b: SubtreeInfo,
    config: TopologyConfig,
) -> bool:
    """
    Check if swap satisfies length constraint.
    
    Reject if new segment length > max_length_ratio * current length.
    """
    seg_a = network.segments[subtree_a.parent_segment_id]
    seg_b = network.segments[subtree_b.parent_segment_id]
    
    current_length_a = seg_a.geometry.length()
    current_length_b = seg_b.geometry.length()
    
    new_length_a = _compute_new_segment_length(
        network, subtree_a.parent_segment_id, subtree_b.root_node_id
    )
    new_length_b = _compute_new_segment_length(
        network, subtree_b.parent_segment_id, subtree_a.root_node_id
    )
    
    if new_length_a > config.max_length_ratio * current_length_a:
        return False
    if new_length_b > config.max_length_ratio * current_length_b:
        return False
    
    return True


def _simulated_annealing_accept(
    delta: float,
    temperature: float,
    rng: random.Random,
) -> bool:
    """
    Simulated annealing acceptance criterion.
    
    Always accept if delta < 0 (improvement).
    Otherwise accept with probability exp(-delta/T).
    """
    if delta <= 0:
        return True
    
    if temperature <= 0:
        return False
    
    probability = math.exp(-delta / temperature)
    return rng.random() < probability


def optimize_topology(
    network: VascularNetwork,
    config: Optional[TopologyConfig] = None,
) -> TopologyResult:
    """
    Optimize network topology using subtree swapping with simulated annealing.
    
    Iteratively swaps subtrees and accepts/rejects based on objective
    change using simulated annealing acceptance criterion.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to optimize (modified in place)
    config : TopologyConfig, optional
        Optimization configuration
        
    Returns
    -------
    TopologyResult
        Optimization result
    """
    if config is None:
        config = TopologyConfig()
    
    result = TopologyResult()
    
    if config.seed is not None:
        rng = random.Random(config.seed)
    else:
        rng = random.Random()
    
    result.initial_objective = _compute_total_volume(network)
    current_objective = result.initial_objective
    
    temperature = config.initial_temperature
    
    total_swaps = 0
    
    while temperature > config.final_temperature and total_swaps < config.max_swaps:
        for _ in range(config.swaps_per_temperature):
            if total_swaps >= config.max_swaps:
                break
            
            candidates = _select_swap_candidates(network, config, rng)
            
            if candidates is None:
                continue
            
            subtree_a, subtree_b = candidates
            result.swaps_attempted += 1
            total_swaps += 1
            
            if _would_create_cycle(network, subtree_a, subtree_b):
                result.swaps_rejected_cycle += 1
                continue
            
            if not _check_length_constraint(network, subtree_a, subtree_b, config):
                result.swaps_rejected_length += 1
                continue
            
            network_snapshot = network.snapshot()
            
            if not perform_subtree_swap(network, subtree_a, subtree_b):
                network.restore(network_snapshot)
                continue
            
            if config.run_nlp_after_swap:
                nlp_config = config.nlp_config or NLPConfig()
                nlp_result = optimize_geometry(network, nlp_config)
                
                if not nlp_result.success:
                    network.restore(network_snapshot)
                    continue
            
            new_objective = _compute_total_volume(network)
            delta = new_objective - current_objective
            
            if _simulated_annealing_accept(delta, temperature, rng):
                current_objective = new_objective
                result.swaps_accepted += 1
            else:
                network.restore(network_snapshot)
                result.swaps_rejected_sa += 1
        
        temperature *= config.cooling_rate
    
    result.final_objective = current_objective
    result.final_temperature = temperature
    
    if result.initial_objective > 0:
        result.objective_reduction = (
            (result.initial_objective - result.final_objective) / result.initial_objective
        )
    
    result.success = True
    
    return result


def reduce_seed_sensitivity(
    network_generator,
    seeds: List[int],
    config: Optional[TopologyConfig] = None,
) -> Dict[str, Any]:
    """
    Measure seed sensitivity reduction from topology optimization.
    
    Generates multiple networks with different seeds, optimizes each,
    and compares variance before and after optimization.
    
    Parameters
    ----------
    network_generator : callable
        Function that takes a seed and returns a VascularNetwork
    seeds : list
        List of random seeds to test
    config : TopologyConfig, optional
        Optimization configuration
        
    Returns
    -------
    dict
        Sensitivity analysis results
    """
    if config is None:
        config = TopologyConfig()
    
    initial_objectives = []
    final_objectives = []
    
    for seed in seeds:
        network = network_generator(seed)
        
        initial_obj = _compute_total_volume(network)
        initial_objectives.append(initial_obj)
        
        result = optimize_topology(network, config)
        
        final_objectives.append(result.final_objective)
    
    initial_var = np.var(initial_objectives)
    final_var = np.var(final_objectives)
    
    variance_reduction = (initial_var - final_var) / initial_var if initial_var > 0 else 0.0
    
    return {
        "seeds": seeds,
        "initial_objectives": initial_objectives,
        "final_objectives": final_objectives,
        "initial_variance": float(initial_var),
        "final_variance": float(final_var),
        "variance_reduction": float(variance_reduction),
        "initial_mean": float(np.mean(initial_objectives)),
        "final_mean": float(np.mean(final_objectives)),
    }
