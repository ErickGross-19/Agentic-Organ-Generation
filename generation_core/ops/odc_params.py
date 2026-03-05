"""
Complete ODCParams dataclass for Optimized Directed Colonization.

Includes ALL tunable parameters from space colonization plus ODC-specific
additions for priority weighting, ordering enforcement, and anti-starburst.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any


@dataclass
class ODCParams:
    """
    Complete parameters for Optimized Directed Colonization.

    Includes ALL tunable parameters from space colonization plus ODC-specific
    additions.
    """

    influence_radius: float = 0.015
    kill_radius: float = 0.003
    step_size: float = 0.005
    min_radius: float = 0.0003
    max_radius: float = 0.005

    taper_factor: float = 0.95
    taper_method: str = "geometric"
    taper_decrement: float = 0.00005

    enable_bifurcation: bool = True
    bifurcation_probability: float = 0.7
    bifurcation_angle_threshold_deg: float = 40.0
    bifurcation_angle_min_deg: float = 30.0
    bifurcation_angle_max_deg: float = 75.0
    bifurcation_angle_noise_deg: float = 5.0
    min_attractors_for_bifurcation: int = 3
    max_children_per_node: int = 2
    allow_trifurcation: bool = False

    min_branch_length: float = 0.002
    max_branch_length: float = 0.02
    branch_length_variance: float = 0.2

    directional_bias: float = 0.0
    preferred_direction: Optional[Tuple[float, float, float]] = None
    smoothing_weight: float = 0.2
    curvature_limit_deg: float = 45.0
    axis_bias: Optional[Tuple[float, float, float]] = None
    axis_lock: Optional[Tuple[bool, bool, bool]] = None

    collision_detection: bool = True
    collision_radius: float = 0.001
    collision_response: str = "deflect"
    self_collision_check: bool = True

    murray_exponent: float = 3.0
    apply_murray_post_growth: bool = True
    murray_min_children: int = 2

    priority_weight_method: str = "inverse"
    priority_weight_exponent: float = 1.0
    priority_decay_rate: float = 0.5

    enforce_ordering: bool = True
    ordering_strictness: float = 0.5
    level_completion_threshold: float = 0.9
    ordering_mode: str = "soft"

    min_generations_before_tissue: int = 2
    progressive_tissue_reveal: bool = True
    reveal_depth_per_generation: float = 0.3
    max_initial_branches: int = 3
    branching_quota_per_length: float = 2.0
    force_bifurcation_depth: int = 3

    multi_inlet_strategy: str = "blended"
    inlet_blend_exponent: float = 2.0

    max_steps: int = 500
    max_nodes: int = 10000
    stall_threshold: int = 20
    min_coverage_to_stop: float = 0.95

    seed: Optional[int] = None
    noise_scale: float = 0.1
    vessel_type: str = "arterial"
    terminal_radius: float = 0.0003

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ODCParams":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)

    def to_sc_params_dict(self) -> Dict[str, Any]:
        """Convert to dict suitable for SpaceColonizationParams construction."""
        return {
            "influence_radius": self.influence_radius,
            "kill_radius": self.kill_radius,
            "step_size": self.step_size,
            "min_radius": self.min_radius,
            "taper_factor": self.taper_factor,
            "vessel_type": self.vessel_type,
            "max_steps": self.max_steps,
            "smoothing_weight": self.smoothing_weight,
            "encourage_bifurcation": self.enable_bifurcation,
            "min_attractions_for_bifurcation": self.min_attractors_for_bifurcation,
            "max_children_per_node": self.max_children_per_node,
            "bifurcation_angle_threshold_deg": self.bifurcation_angle_threshold_deg,
            "bifurcation_probability": self.bifurcation_probability,
        }
