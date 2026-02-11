"""
Priority-aware tissue point sampling for ODC.

Provides functions to validate, augment, and auto-generate hierarchical
tissue point distributions from domain geometry.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from typing import Optional, Tuple, Dict, Any, List, TYPE_CHECKING
import numpy as np
import logging

from .hierarchical import TissueLevel, HierarchicalTissueSpec

if TYPE_CHECKING:
    from ..core.domain import DomainSpec

logger = logging.getLogger(__name__)


def sample_hierarchical_tissue_points(
    domain: "DomainSpec",
    tissue_spec: HierarchicalTissueSpec,
    augment_with_filler: bool = True,
    filler_n_points: int = 500,
    seed: Optional[int] = None,
) -> Tuple[HierarchicalTissueSpec, Dict[str, Any]]:
    """
    Validate and optionally augment hierarchical tissue points.

    If tissue_spec already has explicit points, validates them against domain
    bounds. If augment_with_filler is True, adds uniform filler points at
    lowest priority to improve general coverage in regions not targeted by
    any explicit level.

    Returns validated/augmented spec and metadata dict.
    """
    rng = np.random.default_rng(seed)
    metadata: Dict[str, Any] = {
        "original_levels": tissue_spec.num_levels,
        "original_points": tissue_spec.total_points,
    }

    errors = tissue_spec.validate()
    if errors:
        metadata["validation_errors"] = errors
        logger.warning("Tissue spec validation errors: %s", errors)

    validated_levels: List[TissueLevel] = []
    for lv in tissue_spec.levels:
        if lv.num_points == 0:
            continue
        valid_mask = _points_inside_domain(lv.points, domain)
        valid_points = lv.points[valid_mask]
        n_removed = lv.num_points - len(valid_points)
        if n_removed > 0:
            logger.info(
                "Level %d (%s): removed %d points outside domain",
                lv.priority,
                lv.label,
                n_removed,
            )
        if len(valid_points) > 0:
            validated_levels.append(
                TissueLevel(
                    priority=lv.priority,
                    points=valid_points,
                    label=lv.label,
                    weight=lv.weight,
                    coverage_threshold=lv.coverage_threshold,
                )
            )

    if augment_with_filler and filler_n_points > 0:
        filler_priority = max((lv.priority for lv in validated_levels), default=-1) + 1
        filler_points = _sample_uniform_in_domain(domain, filler_n_points, rng)
        if len(filler_points) > 0:
            validated_levels.append(
                TissueLevel(
                    priority=filler_priority,
                    points=filler_points,
                    label="filler",
                    weight=0.5,
                    coverage_threshold=0.005,
                )
            )
            metadata["filler_points_added"] = len(filler_points)

    result = HierarchicalTissueSpec(levels=validated_levels)
    metadata["final_levels"] = result.num_levels
    metadata["final_points"] = result.total_points
    return result, metadata


def generate_hierarchical_from_strategy(
    domain: "DomainSpec",
    n_levels: int = 3,
    points_per_level: int = 200,
    seed: Optional[int] = None,
) -> HierarchicalTissueSpec:
    """
    Auto-generate hierarchical tissue points from a domain.

    Creates levels with different spatial strategies:
    - Level 0 (highest priority): Deep interior points (near center/bottom)
    - Level 1: Mid-range depth points
    - Level 2+ (lowest priority): Uniform filler

    Useful when the user wants hierarchical growth but doesn't have
    explicit point coordinates.
    """
    rng = np.random.default_rng(seed)
    bounds = _get_bounds(domain)

    levels: List[TissueLevel] = []

    center = np.array([
        (bounds[0] + bounds[1]) / 2,
        (bounds[2] + bounds[3]) / 2,
        (bounds[4] + bounds[5]) / 2,
    ])
    extent = np.array([
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
    ])
    max_dist = float(np.linalg.norm(extent / 2))

    for level_idx in range(n_levels):
        if level_idx == 0:
            points = _sample_near_center(
                domain, bounds, center, max_dist * 0.3, points_per_level, rng
            )
            label = "deep_interior"
        elif level_idx == 1:
            points = _sample_ring(
                domain, bounds, center, max_dist * 0.3, max_dist * 0.6,
                points_per_level, rng,
            )
            label = "mid_range"
        else:
            points = _sample_uniform_in_domain(domain, points_per_level, rng)
            label = f"filler_{level_idx}"

        if len(points) > 0:
            levels.append(
                TissueLevel(
                    priority=level_idx,
                    points=points,
                    label=label,
                    weight=max(0.3, 1.0 - level_idx * 0.25),
                    coverage_threshold=0.005,
                )
            )

    return HierarchicalTissueSpec(levels=levels)


def _get_bounds(domain: "DomainSpec") -> tuple:
    if hasattr(domain, "get_bounds"):
        return domain.get_bounds()
    return (-0.05, 0.05, -0.05, 0.05, -0.05, 0.05)


def _points_inside_domain(
    points: np.ndarray,
    domain: "DomainSpec",
) -> np.ndarray:
    from ..core.types import Point3D

    if not hasattr(domain, "contains"):
        return np.ones(len(points), dtype=bool)
    mask = np.empty(len(points), dtype=bool)
    for i, pt in enumerate(points):
        mask[i] = domain.contains(Point3D(float(pt[0]), float(pt[1]), float(pt[2])))
    return mask


def _sample_uniform_in_domain(
    domain: "DomainSpec",
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    bounds = _get_bounds(domain)
    points = []
    max_attempts = n_points * 10
    attempts = 0
    while len(points) < n_points and attempts < max_attempts:
        pt = np.array([
            rng.uniform(bounds[0], bounds[1]),
            rng.uniform(bounds[2], bounds[3]),
            rng.uniform(bounds[4], bounds[5]),
        ])
        if _points_inside_domain(pt.reshape(1, 3), domain)[0]:
            points.append(pt)
        attempts += 1
    return np.array(points) if points else np.empty((0, 3))


def _sample_near_center(
    domain: "DomainSpec",
    bounds: tuple,
    center: np.ndarray,
    max_radius: float,
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    points = []
    max_attempts = n_points * 10
    attempts = 0
    while len(points) < n_points and attempts < max_attempts:
        offset = rng.normal(0, max_radius / 2, size=3)
        pt = center + offset
        pt[0] = np.clip(pt[0], bounds[0], bounds[1])
        pt[1] = np.clip(pt[1], bounds[2], bounds[3])
        pt[2] = np.clip(pt[2], bounds[4], bounds[5])
        if _points_inside_domain(pt.reshape(1, 3), domain)[0]:
            if float(np.linalg.norm(pt - center)) <= max_radius:
                points.append(pt)
        attempts += 1
    return np.array(points) if points else np.empty((0, 3))


def _sample_ring(
    domain: "DomainSpec",
    bounds: tuple,
    center: np.ndarray,
    r_inner: float,
    r_outer: float,
    n_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    points = []
    max_attempts = n_points * 10
    attempts = 0
    while len(points) < n_points and attempts < max_attempts:
        offset = rng.normal(0, (r_inner + r_outer) / 4, size=3)
        pt = center + offset
        dist = float(np.linalg.norm(pt - center))
        pt[0] = np.clip(pt[0], bounds[0], bounds[1])
        pt[1] = np.clip(pt[1], bounds[2], bounds[3])
        pt[2] = np.clip(pt[2], bounds[4], bounds[5])
        if r_inner <= dist <= r_outer:
            if _points_inside_domain(pt.reshape(1, 3), domain)[0]:
                points.append(pt)
        attempts += 1
    return np.array(points) if points else np.empty((0, 3))
