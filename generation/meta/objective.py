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

    coverage_weight: float = 0.30
    ordering_weight: float = 0.25
    murray_weight: float = 0.15
    flow_weight: float = 0.10
    anti_starburst_weight: float = 0.10
    branching_regularity_weight: float = 0.10
    target_coverage: float = 0.95
    target_min_generations: int = 3
    target_branching_regularity: float = 0.8

    def compute_reward(
        self,
        eval_result: EvalResult,
        hierarchical_coverage: HierarchicalCoverageResult,
        anti_starburst_info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Compute scalar reward in [0, 1] combining all objectives.
        """
        breakdown = self.compute_reward_breakdown(
            eval_result, hierarchical_coverage, anti_starburst_info
        )

        reward = (
            self.coverage_weight * breakdown["coverage_score"]
            + self.ordering_weight * breakdown["ordering_score"]
            + self.murray_weight * breakdown["murray_score"]
            + self.flow_weight * breakdown["flow_score"]
            + self.anti_starburst_weight * breakdown["anti_starburst_score"]
            + self.branching_regularity_weight * breakdown["branching_regularity_score"]
        )
        return max(0.0, min(1.0, reward))

    def compute_reward_breakdown(
        self,
        eval_result: EvalResult,
        hierarchical_coverage: HierarchicalCoverageResult,
        anti_starburst_info: Optional[Dict[str, Any]] = None,
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

        anti_starburst_score = 1.0
        branching_regularity_score = 1.0

        if anti_starburst_info is not None:
            max_gen = anti_starburst_info.get("current_max_generation", 0)
            if self.target_min_generations > 0:
                anti_starburst_score = min(
                    1.0, max_gen / self.target_min_generations
                )

            regularity = anti_starburst_info.get("branching_regularity", 1.0)
            branching_regularity_score = min(
                1.0,
                regularity / self.target_branching_regularity
                if self.target_branching_regularity > 0
                else 1.0,
            )

        return {
            "coverage_score": coverage_score,
            "ordering_score": ordering_score,
            "murray_score": murray_score,
            "flow_score": flow_score,
            "anti_starburst_score": anti_starburst_score,
            "branching_regularity_score": branching_regularity_score,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coverage_weight": self.coverage_weight,
            "ordering_weight": self.ordering_weight,
            "murray_weight": self.murray_weight,
            "flow_weight": self.flow_weight,
            "anti_starburst_weight": self.anti_starburst_weight,
            "branching_regularity_weight": self.branching_regularity_weight,
            "target_coverage": self.target_coverage,
            "target_min_generations": self.target_min_generations,
            "target_branching_regularity": self.target_branching_regularity,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ODCObjective":
        return ODCObjective(
            coverage_weight=d.get("coverage_weight", 0.30),
            ordering_weight=d.get("ordering_weight", 0.25),
            murray_weight=d.get("murray_weight", 0.15),
            flow_weight=d.get("flow_weight", 0.10),
            anti_starburst_weight=d.get("anti_starburst_weight", 0.10),
            branching_regularity_weight=d.get("branching_regularity_weight", 0.10),
            target_coverage=d.get("target_coverage", 0.95),
            target_min_generations=d.get("target_min_generations", 3),
            target_branching_regularity=d.get("target_branching_regularity", 0.8),
        )
