"""
ODC search space definition for Bayesian optimization.

Defines parameter ranges that the meta-model optimizes over,
including continuous, integer, and categorical parameters with
conditional dependencies.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ODCSearchSpace:
    """
    Defines the parameter ranges the meta-model optimizes over.

    Each field is a tuple (min, max) for continuous params,
    or a list of options for categorical params.
    """

    influence_radius: Tuple[float, float] = (0.005, 0.030)
    kill_radius: Tuple[float, float] = (0.001, 0.010)
    step_size: Tuple[float, float] = (0.001, 0.010)

    bifurcation_angle_threshold: Tuple[float, float] = (20.0, 80.0)
    bifurcation_probability: Tuple[float, float] = (0.3, 0.95)
    max_children_per_node: Tuple[int, int] = (2, 3)
    bifurcation_angle_min: Tuple[float, float] = (20.0, 50.0)
    bifurcation_angle_max: Tuple[float, float] = (50.0, 90.0)
    bifurcation_angle_noise: Tuple[float, float] = (0.0, 15.0)
    min_attractors_for_bifurcation: Tuple[int, int] = (2, 6)

    murray_exponent: Tuple[float, float] = (2.0, 3.5)
    terminal_radius: Tuple[float, float] = (0.0001, 0.001)

    taper_factor: Tuple[float, float] = (0.85, 0.99)
    taper_method: List[str] = field(
        default_factory=lambda: ["geometric", "linear"]
    )
    taper_decrement: Tuple[float, float] = (0.00001, 0.0002)

    min_branch_length: Tuple[float, float] = (0.001, 0.005)
    max_branch_length: Tuple[float, float] = (0.005, 0.030)
    branch_length_variance: Tuple[float, float] = (0.0, 0.5)

    directional_bias: Tuple[float, float] = (0.0, 0.5)
    smoothing_weight: Tuple[float, float] = (0.0, 0.5)
    curvature_limit_deg: Tuple[float, float] = (20.0, 90.0)

    min_generations_before_tissue: Tuple[int, int] = (1, 5)
    reveal_depth_per_generation: Tuple[float, float] = (0.1, 0.5)
    max_initial_branches: Tuple[int, int] = (2, 5)
    branching_quota_per_length: Tuple[float, float] = (0.5, 5.0)
    force_bifurcation_depth: Tuple[int, int] = (2, 6)

    priority_weight_exponent: Tuple[float, float] = (0.5, 3.0)
    priority_decay_rate: Tuple[float, float] = (0.1, 1.0)
    ordering_strictness: Tuple[float, float] = (0.0, 1.0)
    level_completion_threshold: Tuple[float, float] = (0.7, 1.0)

    sampling_strategy: List[str] = field(
        default_factory=lambda: [
            "uniform",
            "depth_biased",
            "radial_biased",
            "poisson_disk",
            "gaussian",
            "mixture",
        ]
    )
    depth_power: Tuple[float, float] = (1.0, 4.0)
    radial_power: Tuple[float, float] = (1.0, 4.0)

    max_steps: Tuple[int, int] = (100, 1000)
    trunk_steps: Tuple[int, int] = (3, 20)
    stall_threshold: Tuple[int, int] = (10, 50)

    def suggest(self, trial: Any) -> Dict[str, Any]:
        """
        Sample a parameter configuration from this search space using an
        Optuna trial object.

        Handles conditional parameters: depth_power only suggested when
        sampling_strategy == "depth_biased", etc.
        """
        params: Dict[str, Any] = {}

        params["influence_radius"] = trial.suggest_float(
            "influence_radius", *self.influence_radius
        )
        params["kill_radius"] = trial.suggest_float(
            "kill_radius", *self.kill_radius
        )
        params["step_size"] = trial.suggest_float(
            "step_size", *self.step_size
        )

        params["bifurcation_angle_threshold"] = trial.suggest_float(
            "bifurcation_angle_threshold", *self.bifurcation_angle_threshold
        )
        params["bifurcation_probability"] = trial.suggest_float(
            "bifurcation_probability", *self.bifurcation_probability
        )
        params["max_children_per_node"] = trial.suggest_int(
            "max_children_per_node", *self.max_children_per_node
        )
        params["bifurcation_angle_min"] = trial.suggest_float(
            "bifurcation_angle_min", *self.bifurcation_angle_min
        )
        params["bifurcation_angle_max"] = trial.suggest_float(
            "bifurcation_angle_max", *self.bifurcation_angle_max
        )
        params["bifurcation_angle_noise"] = trial.suggest_float(
            "bifurcation_angle_noise", *self.bifurcation_angle_noise
        )
        params["min_attractors_for_bifurcation"] = trial.suggest_int(
            "min_attractors_for_bifurcation", *self.min_attractors_for_bifurcation
        )

        params["murray_exponent"] = trial.suggest_float(
            "murray_exponent", *self.murray_exponent
        )
        params["terminal_radius"] = trial.suggest_float(
            "terminal_radius", *self.terminal_radius
        )

        params["taper_factor"] = trial.suggest_float(
            "taper_factor", *self.taper_factor
        )
        params["taper_method"] = trial.suggest_categorical(
            "taper_method", self.taper_method
        )
        if params["taper_method"] == "linear":
            params["taper_decrement"] = trial.suggest_float(
                "taper_decrement", *self.taper_decrement
            )

        params["min_branch_length"] = trial.suggest_float(
            "min_branch_length", *self.min_branch_length
        )
        params["max_branch_length"] = trial.suggest_float(
            "max_branch_length", *self.max_branch_length
        )
        params["branch_length_variance"] = trial.suggest_float(
            "branch_length_variance", *self.branch_length_variance
        )

        params["directional_bias"] = trial.suggest_float(
            "directional_bias", *self.directional_bias
        )
        params["smoothing_weight"] = trial.suggest_float(
            "smoothing_weight", *self.smoothing_weight
        )
        params["curvature_limit_deg"] = trial.suggest_float(
            "curvature_limit_deg", *self.curvature_limit_deg
        )

        params["min_generations_before_tissue"] = trial.suggest_int(
            "min_generations_before_tissue", *self.min_generations_before_tissue
        )
        params["reveal_depth_per_generation"] = trial.suggest_float(
            "reveal_depth_per_generation", *self.reveal_depth_per_generation
        )
        params["max_initial_branches"] = trial.suggest_int(
            "max_initial_branches", *self.max_initial_branches
        )
        params["branching_quota_per_length"] = trial.suggest_float(
            "branching_quota_per_length", *self.branching_quota_per_length
        )
        params["force_bifurcation_depth"] = trial.suggest_int(
            "force_bifurcation_depth", *self.force_bifurcation_depth
        )

        params["priority_weight_exponent"] = trial.suggest_float(
            "priority_weight_exponent", *self.priority_weight_exponent
        )
        params["priority_decay_rate"] = trial.suggest_float(
            "priority_decay_rate", *self.priority_decay_rate
        )
        params["ordering_strictness"] = trial.suggest_float(
            "ordering_strictness", *self.ordering_strictness
        )
        params["level_completion_threshold"] = trial.suggest_float(
            "level_completion_threshold", *self.level_completion_threshold
        )

        params["sampling_strategy"] = trial.suggest_categorical(
            "sampling_strategy", self.sampling_strategy
        )

        if params["sampling_strategy"] == "depth_biased":
            params["depth_power"] = trial.suggest_float(
                "depth_power", *self.depth_power
            )
        if params["sampling_strategy"] == "radial_biased":
            params["radial_power"] = trial.suggest_float(
                "radial_power", *self.radial_power
            )

        params["max_steps"] = trial.suggest_int("max_steps", *self.max_steps)
        params["trunk_steps"] = trial.suggest_int(
            "trunk_steps", *self.trunk_steps
        )
        params["stall_threshold"] = trial.suggest_int(
            "stall_threshold", *self.stall_threshold
        )

        return params

    def to_dict(self) -> Dict[str, Any]:
        return {
            "influence_radius": list(self.influence_radius),
            "kill_radius": list(self.kill_radius),
            "step_size": list(self.step_size),
            "bifurcation_angle_threshold": list(self.bifurcation_angle_threshold),
            "bifurcation_probability": list(self.bifurcation_probability),
            "max_children_per_node": list(self.max_children_per_node),
            "bifurcation_angle_min": list(self.bifurcation_angle_min),
            "bifurcation_angle_max": list(self.bifurcation_angle_max),
            "bifurcation_angle_noise": list(self.bifurcation_angle_noise),
            "min_attractors_for_bifurcation": list(self.min_attractors_for_bifurcation),
            "murray_exponent": list(self.murray_exponent),
            "terminal_radius": list(self.terminal_radius),
            "taper_factor": list(self.taper_factor),
            "taper_method": self.taper_method,
            "taper_decrement": list(self.taper_decrement),
            "min_branch_length": list(self.min_branch_length),
            "max_branch_length": list(self.max_branch_length),
            "branch_length_variance": list(self.branch_length_variance),
            "directional_bias": list(self.directional_bias),
            "smoothing_weight": list(self.smoothing_weight),
            "curvature_limit_deg": list(self.curvature_limit_deg),
            "min_generations_before_tissue": list(self.min_generations_before_tissue),
            "reveal_depth_per_generation": list(self.reveal_depth_per_generation),
            "max_initial_branches": list(self.max_initial_branches),
            "branching_quota_per_length": list(self.branching_quota_per_length),
            "force_bifurcation_depth": list(self.force_bifurcation_depth),
            "priority_weight_exponent": list(self.priority_weight_exponent),
            "priority_decay_rate": list(self.priority_decay_rate),
            "ordering_strictness": list(self.ordering_strictness),
            "level_completion_threshold": list(self.level_completion_threshold),
            "sampling_strategy": self.sampling_strategy,
            "depth_power": list(self.depth_power),
            "radial_power": list(self.radial_power),
            "max_steps": list(self.max_steps),
            "trunk_steps": list(self.trunk_steps),
            "stall_threshold": list(self.stall_threshold),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ODCSearchSpace":
        kwargs: Dict[str, Any] = {}
        float_range_keys = [
            "influence_radius", "kill_radius", "step_size",
            "bifurcation_angle_threshold", "bifurcation_probability",
            "bifurcation_angle_min", "bifurcation_angle_max",
            "bifurcation_angle_noise",
            "murray_exponent", "terminal_radius", "depth_power",
            "radial_power", "smoothing_weight", "taper_factor",
            "taper_decrement",
            "min_branch_length", "max_branch_length", "branch_length_variance",
            "directional_bias", "curvature_limit_deg",
            "reveal_depth_per_generation",
            "branching_quota_per_length",
            "priority_weight_exponent", "priority_decay_rate",
            "ordering_strictness", "level_completion_threshold",
        ]
        for key in float_range_keys:
            if key in d:
                kwargs[key] = tuple(d[key])

        int_range_keys = [
            "max_children_per_node", "max_steps", "trunk_steps",
            "min_attractors_for_bifurcation",
            "min_generations_before_tissue",
            "max_initial_branches", "force_bifurcation_depth",
            "stall_threshold",
        ]
        for key in int_range_keys:
            if key in d:
                kwargs[key] = tuple(d[key])

        for key in ["sampling_strategy", "taper_method"]:
            if key in d:
                kwargs[key] = list(d[key])

        return ODCSearchSpace(**kwargs)
