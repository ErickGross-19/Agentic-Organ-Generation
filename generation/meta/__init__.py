"""
Meta-model infrastructure for Optimized Directed Colonization (ODC).

This package provides Bayesian optimization-based parameter learning
for space colonization, including search space definitions, objective
functions, training loops, and trained model inference.
"""

from .search_space import ODCSearchSpace
from .objective import ODCObjective
from .trainer import ODCTrainer, TrainingConfig
from .inference import TrainedODCModel
from .checkpoints import save_checkpoint, load_checkpoint

__all__ = [
    "ODCSearchSpace",
    "ODCObjective",
    "ODCTrainer",
    "TrainingConfig",
    "TrainedODCModel",
    "save_checkpoint",
    "load_checkpoint",
]
