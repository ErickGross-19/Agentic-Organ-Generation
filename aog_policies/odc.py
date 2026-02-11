"""
ODC policies for Optimized Directed Colonization.

Controls hierarchical tissue specification and ODC training/inference behavior.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class HierarchicalTissuePolicy:
    """
    Policy for hierarchical tissue point specification.
    """

    enabled: bool = True
    levels: List[Dict[str, Any]] = field(default_factory=list)
    augment_with_filler: bool = True
    filler_n_points: int = 500

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "levels": self.levels,
            "augment_with_filler": self.augment_with_filler,
            "filler_n_points": self.filler_n_points,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "HierarchicalTissuePolicy":
        return HierarchicalTissuePolicy(
            enabled=d.get("enabled", True),
            levels=d.get("levels", []),
            augment_with_filler=d.get("augment_with_filler", True),
            filler_n_points=d.get("filler_n_points", 500),
        )


@dataclass
class ODCPolicy:
    """
    Policy for ODC backend controlling training and inference.
    """

    enabled: bool = True
    training_mode: bool = False
    trained_model_path: Optional[str] = None

    n_trials: int = 100
    timeout_seconds: Optional[float] = None

    coverage_weight: float = 0.35
    ordering_weight: float = 0.30
    murray_weight: float = 0.20
    flow_weight: float = 0.15

    apply_murray_propagation: bool = True
    murray_exponent: float = 3.0
    terminal_radius: float = 0.0003

    tissue: HierarchicalTissuePolicy = field(
        default_factory=HierarchicalTissuePolicy
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "training_mode": self.training_mode,
            "trained_model_path": self.trained_model_path,
            "n_trials": self.n_trials,
            "timeout_seconds": self.timeout_seconds,
            "coverage_weight": self.coverage_weight,
            "ordering_weight": self.ordering_weight,
            "murray_weight": self.murray_weight,
            "flow_weight": self.flow_weight,
            "apply_murray_propagation": self.apply_murray_propagation,
            "murray_exponent": self.murray_exponent,
            "terminal_radius": self.terminal_radius,
            "tissue": self.tissue.to_dict(),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ODCPolicy":
        tissue_dict = d.get("tissue", {})
        tissue = HierarchicalTissuePolicy.from_dict(tissue_dict) if tissue_dict else HierarchicalTissuePolicy()

        return ODCPolicy(
            enabled=d.get("enabled", True),
            training_mode=d.get("training_mode", False),
            trained_model_path=d.get("trained_model_path"),
            n_trials=d.get("n_trials", 100),
            timeout_seconds=d.get("timeout_seconds"),
            coverage_weight=d.get("coverage_weight", 0.35),
            ordering_weight=d.get("ordering_weight", 0.30),
            murray_weight=d.get("murray_weight", 0.20),
            flow_weight=d.get("flow_weight", 0.15),
            apply_murray_propagation=d.get("apply_murray_propagation", True),
            murray_exponent=d.get("murray_exponent", 3.0),
            terminal_radius=d.get("terminal_radius", 0.0003),
            tissue=tissue,
        )
