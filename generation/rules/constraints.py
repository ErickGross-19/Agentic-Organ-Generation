"""
Constraint specifications for vascular network design.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


@dataclass
class BranchingConstraints:
    """
    Constraints for branching behavior.
    
    Controls how vessels can branch and grow.
    
    Units: All spatial parameters are in millimeters.
    """
    
    min_radius: float = 0.3  # 0.3 mm (capillary scale)
    max_radius: float = 10.0  # 10 mm
    max_branch_order: int = 12
    min_segment_length: float = 1.0  # 1 mm
    max_segment_length: float = 50.0  # 50 mm
    max_branch_angle_deg: float = 80.0
    curvature_limit_deg: float = 15.0  # Max curvature per step
    termination_rule: str = "radius_or_order"  # "radius_or_order", "radius_only", "order_only"
    allowed_vessel_types: List[str] = field(default_factory=lambda: ["arterial", "venous"])
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "min_radius": self.min_radius,
            "max_radius": self.max_radius,
            "max_branch_order": self.max_branch_order,
            "min_segment_length": self.min_segment_length,
            "max_segment_length": self.max_segment_length,
            "max_branch_angle_deg": self.max_branch_angle_deg,
            "curvature_limit_deg": self.curvature_limit_deg,
            "termination_rule": self.termination_rule,
            "allowed_vessel_types": self.allowed_vessel_types,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "BranchingConstraints":
        """Create from dictionary."""
        return cls(
            min_radius=d.get("min_radius", 0.3),
            max_radius=d.get("max_radius", 10.0),
            max_branch_order=d.get("max_branch_order", 12),
            min_segment_length=d.get("min_segment_length", 1.0),
            max_segment_length=d.get("max_segment_length", 50.0),
            max_branch_angle_deg=d.get("max_branch_angle_deg", 80.0),
            curvature_limit_deg=d.get("curvature_limit_deg", 15.0),
            termination_rule=d.get("termination_rule", "radius_or_order"),
            allowed_vessel_types=d.get("allowed_vessel_types", ["arterial", "venous"]),
        )


@dataclass
class RadiusRuleSpec:
    """
    Specification for radius calculation rules.
    
    Defines how radii are computed at bifurcations and along segments.
    """
    
    kind: Literal["murray", "fixed", "linear_taper"]
    params: Dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "kind": self.kind,
            "params": self.params,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "RadiusRuleSpec":
        """Create from dictionary."""
        return cls(
            kind=d["kind"],
            params=d.get("params", {}),
        )
    
    @classmethod
    def murray(cls, gamma: float = 3.0, split_ratio_mean: float = 0.6, split_ratio_std: float = 0.1) -> "RadiusRuleSpec":
        """Create Murray's law rule."""
        return cls(
            kind="murray",
            params={
                "gamma": gamma,
                "split_ratio_mean": split_ratio_mean,
                "split_ratio_std": split_ratio_std,
            },
        )
    
    @classmethod
    def fixed(cls, radius: float) -> "RadiusRuleSpec":
        """Create fixed radius rule."""
        return cls(
            kind="fixed",
            params={"radius": radius},
        )
    
    @classmethod
    def linear_taper(cls, taper_factor: float = 0.9) -> "RadiusRuleSpec":
        """Create linear taper rule."""
        return cls(
            kind="linear_taper",
            params={"taper_factor": taper_factor},
        )


@dataclass
class InteractionRuleSpec:
    """
    Rules for interaction between different vessel types.
    
    Controls collision avoidance and connections between arterial/venous trees.
    
    Units: All spatial parameters are in millimeters.
    """
    
    min_distance_between_types: Dict[tuple, float] = field(default_factory=dict)
    anastomosis_allowed: Dict[tuple, bool] = field(default_factory=dict)
    parallel_preference: Dict[tuple, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set default values."""
        if not self.min_distance_between_types:
            self.min_distance_between_types = {
                ("arterial", "venous"): 1.0,  # 1 mm minimum clearance
                ("arterial", "arterial"): 0.5,  # 0.5 mm within same type
                ("venous", "venous"): 0.5,
            }
        
        if not self.anastomosis_allowed:
            self.anastomosis_allowed = {
                ("arterial", "venous"): True,  # Capillary connections allowed
                ("arterial", "arterial"): False,
                ("venous", "venous"): False,
            }
    
    def get_min_distance(self, type1: str, type2: str) -> float:
        """Get minimum distance between two vessel types."""
        key = tuple(sorted([type1, type2]))
        return self.min_distance_between_types.get(key, 1.0)
    
    def is_anastomosis_allowed(self, type1: str, type2: str) -> bool:
        """Check if anastomosis is allowed between two vessel types."""
        key = tuple(sorted([type1, type2]))
        return self.anastomosis_allowed.get(key, False)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "min_distance_between_types": {
                f"{k[0]}-{k[1]}": v for k, v in self.min_distance_between_types.items()
            },
            "anastomosis_allowed": {
                f"{k[0]}-{k[1]}": v for k, v in self.anastomosis_allowed.items()
            },
            "parallel_preference": {
                f"{k[0]}-{k[1]}": v for k, v in self.parallel_preference.items()
            },
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "InteractionRuleSpec":
        """Create from dictionary."""
        min_dist = {}
        for key_str, val in d.get("min_distance_between_types", {}).items():
            types = tuple(key_str.split("-"))
            min_dist[types] = val
        
        anastomosis = {}
        for key_str, val in d.get("anastomosis_allowed", {}).items():
            types = tuple(key_str.split("-"))
            anastomosis[types] = val
        
        parallel = {}
        for key_str, val in d.get("parallel_preference", {}).items():
            types = tuple(key_str.split("-"))
            parallel[types] = val
        
        return cls(
            min_distance_between_types=min_dist,
            anastomosis_allowed=anastomosis,
            parallel_preference=parallel,
        )


@dataclass
class DegradationRuleSpec:
    """
    Rules for radius degradation as branches split.
    
    Controls how vessel radii decrease through successive generations,
    modeling the tapering of vascular trees from large vessels to capillaries.
    
    Units: All spatial parameters are in millimeters.
    """
    
    model: Literal["exponential", "linear", "generation_based", "none"] = "exponential"
    degradation_factor: float = 0.85
    min_terminal_radius: float = 0.1  # 0.1 mm
    max_generation: Optional[int] = None
    
    def __post_init__(self):
        """Validate parameters."""
        if self.degradation_factor <= 0 or self.degradation_factor >= 1:
            raise ValueError(f"degradation_factor must be in (0, 1), got {self.degradation_factor}")
        if self.min_terminal_radius <= 0:
            raise ValueError(f"min_terminal_radius must be positive, got {self.min_terminal_radius}")
    
    def apply_degradation(self, parent_radius: float, generation: int) -> float:
        """
        Apply degradation to compute child radius.
        
        Parameters
        ----------
        parent_radius : float
            Parent vessel radius
        generation : int
            Current generation number (0 = root)
        
        Returns
        -------
        child_radius : float
            Degraded radius for child vessel
        """
        if self.model == "none":
            return parent_radius
        
        elif self.model == "exponential":
            child_radius = parent_radius * (self.degradation_factor ** generation)
        
        elif self.model == "linear":
            decay = 1.0 - (1.0 - self.degradation_factor) * generation
            child_radius = parent_radius * max(decay, 0.1)
        
        elif self.model == "generation_based":
            child_radius = parent_radius * self.degradation_factor
        
        else:
            raise ValueError(f"Unknown degradation model: {self.model}")
        
        return max(child_radius, self.min_terminal_radius)
    
    def should_terminate(self, radius: float, generation: int) -> tuple[bool, Optional[str]]:
        """
        Check if branch should terminate based on degradation rules.
        
        Parameters
        ----------
        radius : float
            Current vessel radius
        generation : int
            Current generation number
        
        Returns
        -------
        should_terminate : bool
            True if branch should terminate
        reason : str or None
            Reason for termination if applicable
        """
        if radius <= self.min_terminal_radius:
            return True, f"Radius {radius:.6f}mm at or below minimum {self.min_terminal_radius:.6f}mm"
        
        if self.max_generation is not None and generation >= self.max_generation:
            return True, f"Generation {generation} reached maximum {self.max_generation}"
        
        return False, None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "model": self.model,
            "degradation_factor": self.degradation_factor,
            "min_terminal_radius": self.min_terminal_radius,
            "max_generation": self.max_generation,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "DegradationRuleSpec":
        """Create from dictionary."""
        return cls(
            model=d.get("model", "exponential"),
            degradation_factor=d.get("degradation_factor", 0.85),
            min_terminal_radius=d.get("min_terminal_radius", 0.1),
            max_generation=d.get("max_generation"),
        )
    
    @classmethod
    def exponential(cls, factor: float = 0.85, min_radius: float = 0.1) -> "DegradationRuleSpec":
        """Create exponential degradation rule."""
        return cls(model="exponential", degradation_factor=factor, min_terminal_radius=min_radius)
    
    @classmethod
    def linear(cls, factor: float = 0.85, min_radius: float = 0.1) -> "DegradationRuleSpec":
        """Create linear degradation rule."""
        return cls(model="linear", degradation_factor=factor, min_terminal_radius=min_radius)
    
    @classmethod
    def generation_based(cls, factor: float = 0.85, max_gen: int = 12) -> "DegradationRuleSpec":
        """Create generation-based degradation rule."""
        return cls(model="generation_based", degradation_factor=factor, max_generation=max_gen)
