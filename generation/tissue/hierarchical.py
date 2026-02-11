"""
Hierarchical tissue point specification for ODC.

Defines tissue points organized by priority levels, where the vascular
network must reach higher-priority points before lower-priority ones.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class TissueLevel:
    """A single priority level of tissue points."""

    priority: int
    points: np.ndarray
    label: str = ""
    weight: float = 1.0
    coverage_threshold: float = 0.005

    def __post_init__(self) -> None:
        if not isinstance(self.points, np.ndarray):
            self.points = np.asarray(self.points, dtype=np.float64)
        if self.points.ndim == 1:
            self.points = self.points.reshape(1, -1)

    @property
    def num_points(self) -> int:
        return self.points.shape[0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "priority": self.priority,
            "points": self.points.tolist(),
            "label": self.label,
            "weight": self.weight,
            "coverage_threshold": self.coverage_threshold,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TissueLevel":
        return TissueLevel(
            priority=d["priority"],
            points=np.asarray(d["points"], dtype=np.float64),
            label=d.get("label", ""),
            weight=d.get("weight", 1.0),
            coverage_threshold=d.get("coverage_threshold", 0.005),
        )


@dataclass
class HierarchicalTissueSpec:
    """Ordered collection of tissue levels for ODC."""

    levels: List[TissueLevel] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.levels = sorted(self.levels, key=lambda lv: lv.priority)

    def all_points(self) -> np.ndarray:
        if not self.levels:
            return np.empty((0, 3), dtype=np.float64)
        arrays = [lv.points for lv in self.levels if lv.num_points > 0]
        if not arrays:
            return np.empty((0, 3), dtype=np.float64)
        return np.concatenate(arrays, axis=0)

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    @property
    def total_points(self) -> int:
        return sum(lv.num_points for lv in self.levels)

    def get_level(self, priority: int) -> Optional[TissueLevel]:
        for lv in self.levels:
            if lv.priority == priority:
                return lv
        return None

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.levels:
            errors.append("No tissue levels defined")
            return errors

        priorities = [lv.priority for lv in self.levels]
        if len(set(priorities)) != len(priorities):
            errors.append("Duplicate priority values found")

        for lv in self.levels:
            if lv.points.ndim != 2 or lv.points.shape[1] != 3:
                errors.append(
                    f"Level {lv.priority} ({lv.label}): points must be shape (N, 3), "
                    f"got {lv.points.shape}"
                )
            if lv.weight <= 0:
                errors.append(
                    f"Level {lv.priority} ({lv.label}): weight must be > 0"
                )
            if lv.coverage_threshold <= 0:
                errors.append(
                    f"Level {lv.priority} ({lv.label}): coverage_threshold must be > 0"
                )
            if lv.num_points == 0:
                errors.append(
                    f"Level {lv.priority} ({lv.label}): no points defined"
                )
        return errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "levels": [lv.to_dict() for lv in self.levels],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "HierarchicalTissueSpec":
        levels = [TissueLevel.from_dict(ld) for ld in d.get("levels", [])]
        return HierarchicalTissueSpec(levels=levels)
