"""
Multi-objective scoring function for ODC optimization.

Combines hierarchical coverage, ordering compliance, Murray's law
deviation, and flow efficiency into a single scalar reward.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional

from ..specs.eval_result import EvalResult
from ..tissue.coverage import HierarchicalCoverageResult


@dataclass
class ODCObjective:
    """
    Multi-objective scoring function for ODC optimization.

    Combines hierarchical coverage, ordering compliance, Murray's law
    deviation, and flow efficiency into a single scalar reward.
    """

    coverage_weight: float = 0.35
    ordering_weight: float = 0.30
    murray_weight: float = 0.20
    flow_weight: float = 0.15
    target_coverage: float = 0.95

    def compute_reward(
        self,
        eval_result: EvalResult,
        hierarchical_coverage: HierarchicalCoverageResult,
    ) -> float:
        """
        Compute scalar reward in [0, 1] combining all objectives.
        """
        breakdown = self.compute_reward_breakdown(eval_result, hierarchical_coverage)

        reward = (
            self.coverage_weight * breakdown["coverage_score"]
            + self.ordering_weight * breakdown["ordering_score"]
            + self.murray_weight * breakdown["murray_score"]
            + self.flow_weight * breakdown["flow_score"]
        )
        return max(0.0, min(1.0, reward))

    def compute_reward_breakdown(
        self,
        eval_result: EvalResult,
        hierarchical_coverage: HierarchicalCoverageResult,
    ) -> Dict[str, float]:
        """Return individual score components for debugging/logging."""
        coverage_score = min(
            1.0,
            hierarchical_coverage.overall_coverage / self.target_coverage
            if self.target_coverage > 0
            else 0.0,
        )

        ordering_score = hierarchical_coverage.ordering_compliance

        murray_dev = eval_result.structure.murray_deviation
        murray_score = max(0.0, 1.0 - murray_dev)

        turbulent_frac = eval_result.flow.turbulent_fraction
        flow_score = max(0.0, 1.0 - turbulent_frac)

        return {
            "coverage_score": coverage_score,
            "ordering_score": ordering_score,
            "murray_score": murray_score,
            "flow_score": flow_score,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coverage_weight": self.coverage_weight,
            "ordering_weight": self.ordering_weight,
            "murray_weight": self.murray_weight,
            "flow_weight": self.flow_weight,
            "target_coverage": self.target_coverage,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ODCObjective":
        return ODCObjective(
            coverage_weight=d.get("coverage_weight", 0.35),
            ordering_weight=d.get("ordering_weight", 0.30),
            murray_weight=d.get("murray_weight", 0.20),
            flow_weight=d.get("flow_weight", 0.15),
            target_coverage=d.get("target_coverage", 0.95),
        )
