"""
Hierarchical coverage computation for ODC.

Computes per-level coverage metrics and ordering compliance
to score how well a vascular network reaches hierarchical tissue targets.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np

from ..core.network import VascularNetwork
from .hierarchical import TissueLevel, HierarchicalTissueSpec


@dataclass
class LevelCoverageResult:
    """Coverage metrics for a single tissue level."""

    priority: int
    total_points: int
    reached_points: int
    coverage_fraction: float
    mean_distance: float
    max_distance: float
    ordering_score: float


@dataclass
class HierarchicalCoverageResult:
    """Aggregate coverage across all priority levels."""

    level_results: List[LevelCoverageResult]
    overall_coverage: float
    ordering_compliance: float
    total_reached: int
    total_points: int


def _compute_distances_to_network(
    points: np.ndarray,
    network: VascularNetwork,
) -> np.ndarray:
    try:
        from ..analysis.distance import compute_tissue_coverage_distances

        result = compute_tissue_coverage_distances(
            points, network, vessel_type=None, use_surface_distance=True
        )
        return result["distances"]
    except Exception:
        node_positions = []
        for node in network.nodes.values():
            node_positions.append(node.position.to_array())
        if not node_positions:
            return np.full(len(points), float("inf"))
        node_positions = np.array(node_positions)

        distances = np.empty(len(points))
        for i, pt in enumerate(points):
            dists = np.linalg.norm(node_positions - pt, axis=1)
            distances[i] = float(np.min(dists))
        return distances


def _compute_ordering_score(
    level: TissueLevel,
    level_distances: np.ndarray,
    next_level: Optional[TissueLevel],
    next_level_distances: Optional[np.ndarray],
    growth_order: Optional[Dict[int, int]],
    network: VascularNetwork,
) -> float:
    if growth_order is None or not growth_order:
        reached = level_distances < level.coverage_threshold
        return float(np.mean(reached)) if len(reached) > 0 else 0.0

    node_positions = {}
    for node in network.nodes.values():
        node_positions[node.id] = node.position.to_array()

    def _earliest_step_reaching(point: np.ndarray, threshold: float) -> int:
        best_step = float("inf")
        for nid, step in growth_order.items():
            if nid in node_positions:
                dist = float(np.linalg.norm(node_positions[nid] - point))
                if dist < threshold and step < best_step:
                    best_step = step
        return int(best_step) if best_step < float("inf") else -1

    if next_level is None or next_level_distances is None:
        reached = level_distances < level.coverage_threshold
        return float(np.mean(reached)) if len(reached) > 0 else 0.0

    level_steps = []
    for pt in level.points:
        step = _earliest_step_reaching(pt, level.coverage_threshold)
        if step >= 0:
            level_steps.append(step)

    next_steps = []
    for pt in next_level.points:
        step = _earliest_step_reaching(pt, next_level.coverage_threshold)
        if step >= 0:
            next_steps.append(step)

    if not level_steps or not next_steps:
        reached = level_distances < level.coverage_threshold
        return float(np.mean(reached)) if len(reached) > 0 else 0.0

    max_current_step = max(level_steps)
    min_next_step = min(next_steps)

    if max_current_step <= min_next_step:
        return 1.0

    violations = sum(1 for s in next_steps if s < max_current_step)
    return max(0.0, 1.0 - violations / len(next_steps))


def compute_hierarchical_coverage(
    network: VascularNetwork,
    tissue_spec: HierarchicalTissueSpec,
    growth_order: Optional[Dict[int, int]] = None,
) -> HierarchicalCoverageResult:
    """
    Compute per-level coverage and ordering compliance.

    For each level, checks:
    1. What fraction of points have a vessel within coverage_threshold
    2. Whether those points were reached BEFORE lower-priority levels

    Parameters
    ----------
    network : VascularNetwork
        The generated vascular network
    tissue_spec : HierarchicalTissueSpec
        Hierarchical tissue point specification
    growth_order : dict, optional
        Maps node IDs to the iteration step they were created.
        Required for ordering_score computation.
    """
    if not tissue_spec.levels:
        return HierarchicalCoverageResult(
            level_results=[],
            overall_coverage=0.0,
            ordering_compliance=0.0,
            total_reached=0,
            total_points=0,
        )

    level_distances: Dict[int, np.ndarray] = {}
    for lv in tissue_spec.levels:
        if lv.num_points > 0:
            level_distances[lv.priority] = _compute_distances_to_network(
                lv.points, network
            )
        else:
            level_distances[lv.priority] = np.array([])

    level_results: List[LevelCoverageResult] = []
    total_reached = 0
    total_points = 0
    ordering_scores: List[float] = []

    for i, lv in enumerate(tissue_spec.levels):
        distances = level_distances[lv.priority]
        if len(distances) == 0:
            level_results.append(
                LevelCoverageResult(
                    priority=lv.priority,
                    total_points=0,
                    reached_points=0,
                    coverage_fraction=0.0,
                    mean_distance=0.0,
                    max_distance=0.0,
                    ordering_score=0.0,
                )
            )
            continue

        reached = distances < lv.coverage_threshold
        reached_count = int(np.sum(reached))
        coverage_frac = float(np.mean(reached))

        next_lv = tissue_spec.levels[i + 1] if i + 1 < len(tissue_spec.levels) else None
        next_distances = (
            level_distances.get(next_lv.priority) if next_lv is not None else None
        )

        ordering = _compute_ordering_score(
            lv, distances, next_lv, next_distances, growth_order, network
        )

        level_results.append(
            LevelCoverageResult(
                priority=lv.priority,
                total_points=lv.num_points,
                reached_points=reached_count,
                coverage_fraction=coverage_frac,
                mean_distance=float(np.mean(distances)),
                max_distance=float(np.max(distances)),
                ordering_score=ordering,
            )
        )
        total_reached += reached_count
        total_points += lv.num_points
        ordering_scores.append(ordering * lv.weight)

    weight_sum = sum(lv.weight for lv in tissue_spec.levels)
    overall_coverage = total_reached / total_points if total_points > 0 else 0.0
    ordering_compliance = (
        sum(ordering_scores) / weight_sum if weight_sum > 0 else 0.0
    )

    return HierarchicalCoverageResult(
        level_results=level_results,
        overall_coverage=overall_coverage,
        ordering_compliance=ordering_compliance,
        total_reached=total_reached,
        total_points=total_points,
    )
