"""
Trained ODC model for inference (generating networks with learned parameters).
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, TYPE_CHECKING
import logging

from .search_space import ODCSearchSpace
from .objective import ODCObjective

if TYPE_CHECKING:
    from ..core.network import VascularNetwork
    from ..core.domain import DomainSpec
    from ..tissue.hierarchical import HierarchicalTissueSpec

logger = logging.getLogger(__name__)


@dataclass
class TrainedODCModel:
    """
    Trained model that predicts optimal ODC parameters.

    After training, this stores the best parameters found by Optuna
    and can generate networks using those parameters.
    """

    best_params: Dict[str, Any]
    best_reward: float
    study_name: str
    n_trials_completed: int
    search_space: ODCSearchSpace
    objective: ODCObjective
    training_metadata: Dict[str, Any] = field(default_factory=dict)

    def generate(
        self,
        domain: "DomainSpec",
        tissue_spec: "HierarchicalTissueSpec",
        ports: Dict[str, Any],
        seed: Optional[int] = None,
    ) -> Tuple["VascularNetwork", Dict[str, Any]]:
        """
        Generate a network using the trained (best) parameters.
        """
        from ..ops.odc import run_odc_colonization
        from ..ops.murray_propagation import propagate_murray_radii

        odc_result = run_odc_colonization(
            domain=domain,
            tissue_spec=tissue_spec,
            ports=ports,
            params=self.best_params,
            seed=seed,
        )

        network = odc_result.network

        murray_result = propagate_murray_radii(
            network,
            terminal_radius=self.best_params.get("terminal_radius", 0.0003),
            gamma=self.best_params.get("murray_exponent", 3.0),
        )

        metadata = {
            "best_params": self.best_params,
            "best_reward": self.best_reward,
            "study_name": self.study_name,
            "levels_reached": odc_result.levels_reached,
            "iterations_used": odc_result.iterations_used,
            "murray_propagation": {
                "nodes_updated": murray_result.nodes_updated,
                "segments_updated": murray_result.segments_updated,
                "deviation_before": murray_result.mean_deviation_before,
                "deviation_after": murray_result.mean_deviation_after,
            },
        }

        return network, metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_reward": self.best_reward,
            "study_name": self.study_name,
            "n_trials_completed": self.n_trials_completed,
            "search_space": self.search_space.to_dict(),
            "objective": self.objective.to_dict(),
            "training_metadata": self.training_metadata,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TrainedODCModel":
        return TrainedODCModel(
            best_params=d["best_params"],
            best_reward=d["best_reward"],
            study_name=d.get("study_name", ""),
            n_trials_completed=d.get("n_trials_completed", 0),
            search_space=ODCSearchSpace.from_dict(d.get("search_space", {})),
            objective=ODCObjective.from_dict(d.get("objective", {})),
            training_metadata=d.get("training_metadata", {}),
        )
