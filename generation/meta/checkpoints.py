"""
Save/load trained ODC models to/from JSON files.
"""

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .inference import TrainedODCModel

logger = logging.getLogger(__name__)


def save_checkpoint(model: "TrainedODCModel", path: str) -> None:
    """Save TrainedODCModel to JSON file."""
    data = model.to_dict()
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved ODC checkpoint to %s", path)


def load_checkpoint(path: str) -> "TrainedODCModel":
    """Load TrainedODCModel from JSON file."""
    from .inference import TrainedODCModel

    with open(path, "r") as f:
        data = json.load(f)
    model = TrainedODCModel.from_dict(data)
    logger.info(
        "Loaded ODC checkpoint from %s (reward=%.4f, %d trials)",
        path,
        model.best_reward,
        model.n_trials_completed,
    )
    return model
