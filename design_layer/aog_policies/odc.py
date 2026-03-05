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
    distribution_type: str = "uniform"
    distribution_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "levels": self.levels,
            "augment_with_filler": self.augment_with_filler,
            "filler_n_points": self.filler_n_points,
            "distribution_type": self.distribution_type,
            "distribution_params": self.distribution_params,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "HierarchicalTissuePolicy":
        return HierarchicalTissuePolicy(
            enabled=d.get("enabled", True),
            levels=d.get("levels", []),
            augment_with_filler=d.get("augment_with_filler", True),
            filler_n_points=d.get("filler_n_points", 500),
            distribution_type=d.get("distribution_type", "uniform"),
            distribution_params=d.get("distribution_params", {}),
        )


@dataclass
class AntiStarburstPolicy:
    """
    Policy for anti-starburst branching enforcement.
    """

    enabled: bool = True
    min_generations_before_tissue: int = 2
    progressive_tissue_reveal: bool = True
    reveal_depth_per_generation: float = 0.3
    max_initial_branches: int = 3
    branching_quota_per_length: float = 2.0
    force_bifurcation_depth: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "min_generations_before_tissue": self.min_generations_before_tissue,
            "progressive_tissue_reveal": self.progressive_tissue_reveal,
            "reveal_depth_per_generation": self.reveal_depth_per_generation,
            "max_initial_branches": self.max_initial_branches,
            "branching_quota_per_length": self.branching_quota_per_length,
            "force_bifurcation_depth": self.force_bifurcation_depth,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AntiStarburstPolicy":
        return AntiStarburstPolicy(
            enabled=d.get("enabled", True),
            min_generations_before_tissue=d.get("min_generations_before_tissue", 2),
            progressive_tissue_reveal=d.get("progressive_tissue_reveal", True),
            reveal_depth_per_generation=d.get("reveal_depth_per_generation", 0.3),
            max_initial_branches=d.get("max_initial_branches", 3),
            branching_quota_per_length=d.get("branching_quota_per_length", 2.0),
            force_bifurcation_depth=d.get("force_bifurcation_depth", 3),
        )


@dataclass
class MultiTreePolicy:
    """
    Policy for multi-tree ODC coordination.
    """

    enabled: bool = False
    collision_radius: float = 0.001
    interleave_strategy: str = "round_robin"
    tree_configs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "collision_radius": self.collision_radius,
            "interleave_strategy": self.interleave_strategy,
            "tree_configs": self.tree_configs,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MultiTreePolicy":
        return MultiTreePolicy(
            enabled=d.get("enabled", False),
            collision_radius=d.get("collision_radius", 0.001),
            interleave_strategy=d.get("interleave_strategy", "round_robin"),
            tree_configs=d.get("tree_configs", []),
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

    coverage_weight: float = 0.30
    ordering_weight: float = 0.25
    murray_weight: float = 0.15
    flow_weight: float = 0.10
    anti_starburst_weight: float = 0.10
    branching_regularity_weight: float = 0.10

    apply_murray_propagation: bool = True
    murray_exponent: float = 3.0
    terminal_radius: float = 0.0003

    tissue: HierarchicalTissuePolicy = field(
        default_factory=HierarchicalTissuePolicy
    )
    anti_starburst: AntiStarburstPolicy = field(
        default_factory=AntiStarburstPolicy
    )
    multi_tree: MultiTreePolicy = field(
        default_factory=MultiTreePolicy
    )

    preset: Optional[str] = None

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
            "anti_starburst_weight": self.anti_starburst_weight,
            "branching_regularity_weight": self.branching_regularity_weight,
            "apply_murray_propagation": self.apply_murray_propagation,
            "murray_exponent": self.murray_exponent,
            "terminal_radius": self.terminal_radius,
            "tissue": self.tissue.to_dict(),
            "anti_starburst": self.anti_starburst.to_dict(),
            "multi_tree": self.multi_tree.to_dict(),
            "preset": self.preset,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ODCPolicy":
        tissue_dict = d.get("tissue", {})
        tissue = HierarchicalTissuePolicy.from_dict(tissue_dict) if tissue_dict else HierarchicalTissuePolicy()

        anti_starburst_dict = d.get("anti_starburst", {})
        anti_starburst = AntiStarburstPolicy.from_dict(anti_starburst_dict) if anti_starburst_dict else AntiStarburstPolicy()

        multi_tree_dict = d.get("multi_tree", {})
        multi_tree = MultiTreePolicy.from_dict(multi_tree_dict) if multi_tree_dict else MultiTreePolicy()

        return ODCPolicy(
            enabled=d.get("enabled", True),
            training_mode=d.get("training_mode", False),
            trained_model_path=d.get("trained_model_path"),
            n_trials=d.get("n_trials", 100),
            timeout_seconds=d.get("timeout_seconds"),
            coverage_weight=d.get("coverage_weight", 0.30),
            ordering_weight=d.get("ordering_weight", 0.25),
            murray_weight=d.get("murray_weight", 0.15),
            flow_weight=d.get("flow_weight", 0.10),
            anti_starburst_weight=d.get("anti_starburst_weight", 0.10),
            branching_regularity_weight=d.get("branching_regularity_weight", 0.10),
            apply_murray_propagation=d.get("apply_murray_propagation", True),
            murray_exponent=d.get("murray_exponent", 3.0),
            terminal_radius=d.get("terminal_radius", 0.0003),
            tissue=tissue,
            anti_starburst=anti_starburst,
            multi_tree=multi_tree,
            preset=d.get("preset"),
        )


def get_odc_preset(name: str) -> ODCPolicy:
    """Get a named ODC policy preset."""
    presets = {
        "conservative": ODCPolicy(
            anti_starburst=AntiStarburstPolicy(
                min_generations_before_tissue=3,
                max_initial_branches=2,
                force_bifurcation_depth=4,
            ),
            preset="conservative",
        ),
        "aggressive": ODCPolicy(
            anti_starburst=AntiStarburstPolicy(
                min_generations_before_tissue=1,
                max_initial_branches=4,
                force_bifurcation_depth=2,
                branching_quota_per_length=3.0,
            ),
            preset="aggressive",
        ),
        "balanced": ODCPolicy(
            preset="balanced",
        ),
        "liver": ODCPolicy(
            tissue=HierarchicalTissuePolicy(
                distribution_type="liver_lobule",
            ),
            anti_starburst=AntiStarburstPolicy(
                min_generations_before_tissue=2,
                max_initial_branches=3,
            ),
            preset="liver",
        ),
        "lung": ODCPolicy(
            tissue=HierarchicalTissuePolicy(
                distribution_type="lung_bronchiole",
            ),
            anti_starburst=AntiStarburstPolicy(
                min_generations_before_tissue=3,
                force_bifurcation_depth=5,
            ),
            preset="lung",
        ),
        "kidney": ODCPolicy(
            tissue=HierarchicalTissuePolicy(
                distribution_type="kidney_nephron",
            ),
            anti_starburst=AntiStarburstPolicy(
                min_generations_before_tissue=2,
            ),
            preset="kidney",
        ),
    }
    if name not in presets:
        raise ValueError(f"Unknown ODC preset: {name}. Available: {list(presets.keys())}")
    return presets[name]
