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

    murray_exponent: Tuple[float, float] = (2.0, 3.5)
    terminal_radius: Tuple[float, float] = (0.0001, 0.001)

    sampling_strategy: List[str] = field(
        default_factory=lambda: [
            "uniform",
            "depth_biased",
            "radial_biased",
            "mixture",
        ]
    )
    depth_power: Tuple[float, float] = (1.0, 4.0)
    radial_power: Tuple[float, float] = (1.0, 4.0)

    max_steps: Tuple[int, int] = (100, 1000)
    trunk_steps: Tuple[int, int] = (3, 20)

    smoothing_weight: Tuple[float, float] = (0.0, 0.5)
    taper_factor: Tuple[float, float] = (0.85, 0.99)

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

        params["murray_exponent"] = trial.suggest_float(
            "murray_exponent", *self.murray_exponent
        )
        params["terminal_radius"] = trial.suggest_float(
            "terminal_radius", *self.terminal_radius
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

        params["smoothing_weight"] = trial.suggest_float(
            "smoothing_weight", *self.smoothing_weight
        )
        params["taper_factor"] = trial.suggest_float(
            "taper_factor", *self.taper_factor
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
            "murray_exponent": list(self.murray_exponent),
            "terminal_radius": list(self.terminal_radius),
            "sampling_strategy": self.sampling_strategy,
            "depth_power": list(self.depth_power),
            "radial_power": list(self.radial_power),
            "max_steps": list(self.max_steps),
            "trunk_steps": list(self.trunk_steps),
            "smoothing_weight": list(self.smoothing_weight),
            "taper_factor": list(self.taper_factor),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ODCSearchSpace":
        kwargs: Dict[str, Any] = {}
        for key in [
            "influence_radius", "kill_radius", "step_size",
            "bifurcation_angle_threshold", "bifurcation_probability",
            "murray_exponent", "terminal_radius", "depth_power",
            "radial_power", "smoothing_weight", "taper_factor",
        ]:
            if key in d:
                kwargs[key] = tuple(d[key])

        for key in ["max_children_per_node", "max_steps", "trunk_steps"]:
            if key in d:
                kwargs[key] = tuple(d[key])

        if "sampling_strategy" in d:
            kwargs["sampling_strategy"] = list(d["sampling_strategy"])

        return ODCSearchSpace(**kwargs)
