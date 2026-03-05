"""
Optuna-based training loop for ODC parameter optimization.

Runs Bayesian optimization to find optimal space colonization parameters
for a given hierarchical tissue distribution.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, TYPE_CHECKING
import logging
import time

from .search_space import ODCSearchSpace
from .objective import ODCObjective
from .inference import TrainedODCModel

if TYPE_CHECKING:
    from ..core.domain import DomainSpec
    from ..tissue.hierarchical import HierarchicalTissueSpec

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for ODC meta-model training."""

    n_trials: int = 100
    n_startup_trials: int = 10
    timeout_seconds: Optional[float] = None
    seed: Optional[int] = None
    study_name: str = "odc_study"
    storage: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    n_jobs: int = 1
    pruning_enabled: bool = True
    log_interval: int = 10


class ODCTrainer:
    """
    Bayesian optimization trainer for ODC parameters.

    Training loop:
    1. Optuna suggests parameters from ODCSearchSpace
    2. Build SpaceColonizationParams from suggested params
    3. Run run_odc_colonization() to grow network
    4. Apply propagate_murray_radii() post-hoc
    5. Evaluate with evaluate_network() + compute_hierarchical_coverage()
    6. Compute reward via ODCObjective
    7. Report reward to Optuna -> update surrogate model
    8. Repeat
    """

    def __init__(
        self,
        domain: "DomainSpec",
        tissue_spec: "HierarchicalTissueSpec",
        ports: Dict[str, Any],
        search_space: Optional[ODCSearchSpace] = None,
        objective: Optional[ODCObjective] = None,
        config: Optional[TrainingConfig] = None,
    ):
        self.domain = domain
        self.tissue_spec = tissue_spec
        self.ports = ports
        self.search_space = search_space or ODCSearchSpace()
        self.objective = objective or ODCObjective()
        self.config = config or TrainingConfig()
        self._study = None
        self._best_network = None
        self._trial_history: list = []

    def train(self) -> TrainedODCModel:
        """
        Run Optuna study. Returns TrainedODCModel with best parameters.
        """
        try:
            import optuna
        except ImportError:
            raise ImportError(
                "optuna is required for ODC training. "
                "Install with: pip install optuna"
            )

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.TPESampler(
            n_startup_trials=self.config.n_startup_trials,
            seed=self.config.seed,
        )

        self._study = optuna.create_study(
            study_name=self.config.study_name,
            storage=self.config.storage,
            direction="maximize",
            sampler=sampler,
            load_if_exists=True,
        )

        start_time = time.time()

        self._study.optimize(
            self._evaluate_trial,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout_seconds,
            n_jobs=self.config.n_jobs,
            show_progress_bar=False,
        )

        elapsed = time.time() - start_time

        best_trial = self._study.best_trial
        logger.info(
            "Training complete: %d trials in %.1fs. Best reward=%.4f (trial %d)",
            len(self._study.trials),
            elapsed,
            best_trial.value,
            best_trial.number,
        )

        model = TrainedODCModel(
            best_params=best_trial.params,
            best_reward=best_trial.value,
            study_name=self.config.study_name,
            n_trials_completed=len(self._study.trials),
            search_space=self.search_space,
            objective=self.objective,
            training_metadata={
                "elapsed_seconds": elapsed,
                "best_trial_number": best_trial.number,
                "n_startup_trials": self.config.n_startup_trials,
                "seed": self.config.seed,
            },
        )

        if self.config.checkpoint_dir is not None:
            from .checkpoints import save_checkpoint
            import os

            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            path = os.path.join(self.config.checkpoint_dir, "odc_model.json")
            save_checkpoint(model, path)
            logger.info("Saved checkpoint to %s", path)

        return model

    def _evaluate_trial(self, trial: Any) -> float:
        """
        Single trial evaluation.
        """
        from ..ops.odc import run_odc_colonization
        from ..ops.murray_propagation import propagate_murray_radii
        from ..api.evaluate import evaluate_network
        from ..tissue.coverage import compute_hierarchical_coverage

        params = self.search_space.suggest(trial)

        try:
            odc_result = run_odc_colonization(
                domain=self.domain,
                tissue_spec=self.tissue_spec,
                ports=self.ports,
                params=params,
                seed=self.config.seed,
            )

            network = odc_result.network

            propagate_murray_radii(
                network,
                terminal_radius=params.get("terminal_radius", 0.0003),
                gamma=params.get("murray_exponent", 3.0),
            )

            tissue_points = self.tissue_spec.all_points()
            eval_result = evaluate_network(network, tissue_points)

            hier_coverage = compute_hierarchical_coverage(
                network, self.tissue_spec, odc_result.growth_order
            )

            reward = self.objective.compute_reward(eval_result, hier_coverage)

            if trial.number % self.config.log_interval == 0:
                breakdown = self.objective.compute_reward_breakdown(
                    eval_result, hier_coverage
                )
                logger.info(
                    "Trial %d: reward=%.4f (cov=%.3f, ord=%.3f, mur=%.3f, flow=%.3f)",
                    trial.number,
                    reward,
                    breakdown["coverage_score"],
                    breakdown["ordering_score"],
                    breakdown["murray_score"],
                    breakdown["flow_score"],
                )

            self._trial_history.append({
                "trial": trial.number,
                "reward": reward,
                "params": params,
            })

            return reward

        except Exception as e:
            logger.warning("Trial %d failed: %s", trial.number, e)
            return 0.0

    def get_study(self) -> Any:
        """Return the Optuna study for inspection/visualization."""
        return self._study

    def get_trial_history(self) -> list:
        """Return history of all evaluated trials."""
        return self._trial_history
