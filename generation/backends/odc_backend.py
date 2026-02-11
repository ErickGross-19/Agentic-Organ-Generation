"""
Optimized Directed Colonization (ODC) backend.

Integrates hierarchical tissue targeting with meta-model parameter
optimization. Can operate in two modes:
1. Training mode: Runs Bayesian optimization to find best params
2. Inference mode: Uses pre-trained params from checkpoint

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import logging
import numpy as np

from .base import GenerationBackend, BackendConfig
from ..core.network import VascularNetwork
from ..core.domain import DomainSpec
from ..tissue.hierarchical import TissueLevel, HierarchicalTissueSpec

logger = logging.getLogger(__name__)


@dataclass
class ODCConfig(BackendConfig):
    """Configuration for ODC backend."""

    training_mode: bool = False
    trained_model_path: Optional[str] = None

    n_trials: int = 100
    n_startup_trials: int = 10
    timeout_seconds: Optional[float] = None

    tissue_levels: Optional[List[Dict[str, Any]]] = None

    search_space_overrides: Optional[Dict[str, Any]] = None
    objective_overrides: Optional[Dict[str, Any]] = None

    apply_murray_propagation: bool = True
    murray_exponent: float = 3.0
    terminal_radius: float = 0.0003

    augment_with_filler: bool = True
    filler_n_points: int = 500

    auto_generate_levels: bool = False
    auto_n_levels: int = 3
    auto_points_per_level: int = 200


class ODCBackend(GenerationBackend):
    """
    Optimized Directed Colonization backend.

    Implements GenerationBackend interface for ODC generation.
    """

    @property
    def supports_dual_tree(self) -> bool:
        return False

    @property
    def supports_closed_loop(self) -> bool:
        return False

    def generate(
        self,
        domain: DomainSpec,
        num_outlets: int,
        inlet_position: np.ndarray,
        inlet_radius: float,
        vessel_type: str = "arterial",
        config: Optional[ODCConfig] = None,
        rng_seed: Optional[int] = None,
    ) -> VascularNetwork:
        """
        Generate vascular network using ODC.
        """
        if config is None:
            config = ODCConfig()

        tissue_spec = self._build_tissue_spec(domain, config, rng_seed)

        ports = {
            "inlets": [{
                "position": list(inlet_position) if isinstance(inlet_position, np.ndarray) else inlet_position,
                "radius": inlet_radius,
                "vessel_type": vessel_type,
                "direction": [0, 0, -1],
            }],
            "outlets": [],
        }

        if config.training_mode:
            network = self._generate_training(
                domain, tissue_spec, ports, config, rng_seed
            )
        elif config.trained_model_path is not None:
            network = self._generate_from_checkpoint(
                domain, tissue_spec, ports, config, rng_seed
            )
        else:
            network = self._generate_default(
                domain, tissue_spec, ports, config, rng_seed
            )

        if config.apply_murray_propagation:
            from ..ops.murray_propagation import propagate_murray_radii

            propagate_murray_radii(
                network,
                terminal_radius=config.terminal_radius,
                gamma=config.murray_exponent,
            )

        return network

    def _build_tissue_spec(
        self,
        domain: DomainSpec,
        config: ODCConfig,
        rng_seed: Optional[int],
    ) -> HierarchicalTissueSpec:
        """Build HierarchicalTissueSpec from config."""
        if config.tissue_levels is not None:
            levels = []
            for ld in config.tissue_levels:
                levels.append(TissueLevel.from_dict(ld))
            spec = HierarchicalTissueSpec(levels=levels)
        elif config.auto_generate_levels:
            from ..tissue.samplers import generate_hierarchical_from_strategy

            spec = generate_hierarchical_from_strategy(
                domain,
                n_levels=config.auto_n_levels,
                points_per_level=config.auto_points_per_level,
                seed=rng_seed,
            )
        else:
            from ..tissue.samplers import generate_hierarchical_from_strategy

            spec = generate_hierarchical_from_strategy(
                domain,
                n_levels=3,
                points_per_level=200,
                seed=rng_seed,
            )

        if config.augment_with_filler:
            from ..tissue.samplers import sample_hierarchical_tissue_points

            spec, _ = sample_hierarchical_tissue_points(
                domain,
                spec,
                augment_with_filler=True,
                filler_n_points=config.filler_n_points,
                seed=rng_seed,
            )

        return spec

    def _generate_training(
        self,
        domain: DomainSpec,
        tissue_spec: HierarchicalTissueSpec,
        ports: Dict[str, Any],
        config: ODCConfig,
        rng_seed: Optional[int],
    ) -> VascularNetwork:
        """Generate with training mode (Bayesian optimization)."""
        from ..meta.trainer import ODCTrainer, TrainingConfig
        from ..meta.search_space import ODCSearchSpace
        from ..meta.objective import ODCObjective

        search_space = ODCSearchSpace()
        if config.search_space_overrides:
            search_space = ODCSearchSpace.from_dict(config.search_space_overrides)

        objective = ODCObjective()
        if config.objective_overrides:
            objective = ODCObjective.from_dict(config.objective_overrides)

        training_config = TrainingConfig(
            n_trials=config.n_trials,
            n_startup_trials=config.n_startup_trials,
            timeout_seconds=config.timeout_seconds,
            seed=rng_seed,
        )

        trainer = ODCTrainer(
            domain=domain,
            tissue_spec=tissue_spec,
            ports=ports,
            search_space=search_space,
            objective=objective,
            config=training_config,
        )

        model = trainer.train()
        network, _ = model.generate(domain, tissue_spec, ports, seed=rng_seed)
        return network

    def _generate_from_checkpoint(
        self,
        domain: DomainSpec,
        tissue_spec: HierarchicalTissueSpec,
        ports: Dict[str, Any],
        config: ODCConfig,
        rng_seed: Optional[int],
    ) -> VascularNetwork:
        """Generate from a pre-trained checkpoint."""
        from ..meta.checkpoints import load_checkpoint

        model = load_checkpoint(config.trained_model_path)
        network, _ = model.generate(domain, tissue_spec, ports, seed=rng_seed)
        return network

    def _generate_default(
        self,
        domain: DomainSpec,
        tissue_spec: HierarchicalTissueSpec,
        ports: Dict[str, Any],
        config: ODCConfig,
        rng_seed: Optional[int],
    ) -> VascularNetwork:
        """Generate with default parameters (no training)."""
        from ..ops.odc import run_odc_colonization

        default_params = {
            "influence_radius": 0.015,
            "kill_radius": 0.003,
            "step_size": 0.005,
            "max_steps": 500,
            "bifurcation_probability": 0.7,
            "max_children_per_node": 2,
            "taper_factor": 0.95,
            "murray_exponent": config.murray_exponent,
            "terminal_radius": config.terminal_radius,
        }

        result = run_odc_colonization(
            domain=domain,
            tissue_spec=tissue_spec,
            ports=ports,
            params=default_params,
            seed=rng_seed,
        )

        return result.network
