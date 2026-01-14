"""Operations for building and modifying vascular networks."""

from .build import create_network, add_inlet, add_outlet
from .growth import grow_branch, grow_to_point, bifurcate, grow_kary_tree_to_depths, grow_kary_tree_v2, KaryTreeSpec
from .collision import get_collisions, avoid_collisions
from .space_colonization import space_colonization_step, SpaceColonizationParams
from .anastomosis import create_anastomosis, check_tree_interactions
from .pathfinding import grow_toward_targets, CostWeights
from .embedding import embed_tree_as_negative_space
from .tree_ops import prune, add_branch
from .features import (
    FaceId,
    RidgeSpec,
    create_annular_ridge,
    create_frame_ridge,
    add_raised_ridge,
)

__all__ = [
    "create_network",
    "add_inlet",
    "add_outlet",
    "grow_branch",
    "grow_to_point",
    "bifurcate",
    "grow_kary_tree_to_depths",
    "grow_kary_tree_v2",
    "KaryTreeSpec",
    "get_collisions",
    "avoid_collisions",
    "space_colonization_step",
    "SpaceColonizationParams",
    "create_anastomosis",
    "check_tree_interactions",
    "grow_toward_targets",
    "CostWeights",
    "embed_tree_as_negative_space",
    "prune",
    "add_branch",
    "FaceId",
    "RidgeSpec",
    "create_annular_ridge",
    "create_frame_ridge",
    "add_raised_ridge",
]
