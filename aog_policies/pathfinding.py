"""
Pathfinding policies for vascular network routing.

This module contains policy dataclasses for A* pathfinding configuration,
waypoint handling, and hierarchical coarse-to-fine pathfinding.

All policies are JSON-serializable and support from_dict/to_dict methods.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from .resolution import ResolutionPolicy


@dataclass
class PathfindingPolicy:
    """
    Policy for A* pathfinding configuration.
    
    Controls voxel resolution, clearance requirements, search limits,
    and path smoothing parameters.
    
    JSON Schema:
    {
        "voxel_pitch": float (meters),
        "clearance": float (meters),
        "max_nodes": int,
        "timeout_s": float,
        "turn_penalty": float,
        "heuristic_weight": float,
        "smoothing_enabled": bool,
        "smoothing_iters": int,
        "smoothing_strength": float,
        "allow_partial": bool,
        "diagonal_movement": bool
    }
    """
    voxel_pitch: float = 0.0005  # 0.5mm
    clearance: float = 0.0002  # 0.2mm
    max_nodes: int = 100000
    timeout_s: float = 30.0
    turn_penalty: float = 0.1
    heuristic_weight: float = 1.0
    smoothing_enabled: bool = True
    smoothing_iters: int = 3
    smoothing_strength: float = 0.5
    allow_partial: bool = False
    diagonal_movement: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "voxel_pitch": self.voxel_pitch,
            "clearance": self.clearance,
            "max_nodes": self.max_nodes,
            "timeout_s": self.timeout_s,
            "turn_penalty": self.turn_penalty,
            "heuristic_weight": self.heuristic_weight,
            "smoothing_enabled": self.smoothing_enabled,
            "smoothing_iters": self.smoothing_iters,
            "smoothing_strength": self.smoothing_strength,
            "allow_partial": self.allow_partial,
            "diagonal_movement": self.diagonal_movement,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PathfindingPolicy":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class WaypointPolicy:
    """
    Policy for handling waypoints during pathfinding.
    
    Controls how waypoints are processed and what happens when
    a waypoint is unreachable.
    
    JSON Schema:
    {
        "skip_unreachable": bool,
        "max_skip_count": int,
        "emit_warnings": bool,
        "fallback_direct": bool
    }
    """
    skip_unreachable: bool = True
    max_skip_count: int = 3
    emit_warnings: bool = True
    fallback_direct: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "skip_unreachable": self.skip_unreachable,
            "max_skip_count": self.max_skip_count,
            "emit_warnings": self.emit_warnings,
            "fallback_direct": self.fallback_direct,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WaypointPolicy":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class HierarchicalPathfindingPolicy:
    """
    Policy for hierarchical coarse-to-fine pathfinding.
    
    This extends PathfindingPolicy with parameters for two-stage pathfinding
    that enables fine-pitch routing in large domains.
    
    JSON Schema:
    {
        "pitch_coarse": float (meters),
        "pitch_fine": float (meters),
        "corridor_radius_buffer": float (meters),
        "max_voxels_coarse": int,
        "max_voxels_fine": int,
        "auto_relax_fine_pitch": bool,
        "pitch_step_factor": float,
        "clearance": float (meters),
        "max_nodes_coarse": int,
        "max_nodes_fine": int,
        "timeout_s": float,
        "turn_penalty": float,
        "heuristic_weight": float,
        "smoothing_enabled": bool,
        "smoothing_iters": int,
        "smoothing_strength": float,
        "allow_partial": bool,
        "diagonal_movement": bool,
        "allow_skip_waypoints": bool,
        "max_skip_count": int
    }
    """
    pitch_coarse: float = 0.0001  # 100um coarse pitch
    pitch_fine: float = 0.000005  # 5um fine pitch
    corridor_radius_buffer: float = 0.0002  # 200um buffer around coarse path
    max_voxels_coarse: int = 10_000_000  # 10M voxels for coarse grid
    max_voxels_fine: int = 50_000_000  # 50M voxels for fine corridor
    auto_relax_fine_pitch: bool = True
    pitch_step_factor: float = 1.5
    clearance: float = 0.0002  # 0.2mm
    max_nodes_coarse: int = 100_000
    max_nodes_fine: int = 500_000
    timeout_s: float = 60.0
    turn_penalty: float = 0.1
    heuristic_weight: float = 1.0
    smoothing_enabled: bool = True
    smoothing_iters: int = 3
    smoothing_strength: float = 0.5
    allow_partial: bool = False
    diagonal_movement: bool = True
    allow_skip_waypoints: bool = True
    max_skip_count: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pitch_coarse": self.pitch_coarse,
            "pitch_fine": self.pitch_fine,
            "corridor_radius_buffer": self.corridor_radius_buffer,
            "max_voxels_coarse": self.max_voxels_coarse,
            "max_voxels_fine": self.max_voxels_fine,
            "auto_relax_fine_pitch": self.auto_relax_fine_pitch,
            "pitch_step_factor": self.pitch_step_factor,
            "clearance": self.clearance,
            "max_nodes_coarse": self.max_nodes_coarse,
            "max_nodes_fine": self.max_nodes_fine,
            "timeout_s": self.timeout_s,
            "turn_penalty": self.turn_penalty,
            "heuristic_weight": self.heuristic_weight,
            "smoothing_enabled": self.smoothing_enabled,
            "smoothing_iters": self.smoothing_iters,
            "smoothing_strength": self.smoothing_strength,
            "allow_partial": self.allow_partial,
            "diagonal_movement": self.diagonal_movement,
            "allow_skip_waypoints": self.allow_skip_waypoints,
            "max_skip_count": self.max_skip_count,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HierarchicalPathfindingPolicy":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_resolution_policy(
        cls,
        resolution_policy: "ResolutionPolicy",
        **overrides,
    ) -> "HierarchicalPathfindingPolicy":
        """Create from ResolutionPolicy with optional overrides."""
        return cls(
            pitch_coarse=overrides.get("pitch_coarse", resolution_policy.pathfinding_pitch_coarse),
            pitch_fine=overrides.get("pitch_fine", resolution_policy.pathfinding_pitch_fine),
            max_voxels_coarse=overrides.get("max_voxels_coarse", resolution_policy.max_voxels_pathfinding_coarse),
            max_voxels_fine=overrides.get("max_voxels_fine", resolution_policy.max_voxels_pathfinding_fine),
            auto_relax_fine_pitch=overrides.get("auto_relax_fine_pitch", resolution_policy.auto_relax_pitch),
            pitch_step_factor=overrides.get("pitch_step_factor", resolution_policy.pitch_step_factor),
            **{k: v for k, v in overrides.items() if k not in [
                "pitch_coarse", "pitch_fine", "max_voxels_coarse", "max_voxels_fine",
                "auto_relax_fine_pitch", "pitch_step_factor"
            ]},
        )
    
    def to_coarse_policy(self) -> PathfindingPolicy:
        """Convert to PathfindingPolicy for coarse stage."""
        return PathfindingPolicy(
            voxel_pitch=self.pitch_coarse,
            clearance=self.clearance,
            max_nodes=self.max_nodes_coarse,
            timeout_s=self.timeout_s / 2,
            turn_penalty=self.turn_penalty,
            heuristic_weight=self.heuristic_weight,
            smoothing_enabled=False,
            allow_partial=self.allow_partial,
            diagonal_movement=self.diagonal_movement,
        )
    
    def to_fine_policy(self) -> PathfindingPolicy:
        """Convert to PathfindingPolicy for fine stage."""
        return PathfindingPolicy(
            voxel_pitch=self.pitch_fine,
            clearance=self.clearance,
            max_nodes=self.max_nodes_fine,
            timeout_s=self.timeout_s / 2,
            turn_penalty=self.turn_penalty,
            heuristic_weight=self.heuristic_weight,
            smoothing_enabled=self.smoothing_enabled,
            smoothing_iters=self.smoothing_iters,
            smoothing_strength=self.smoothing_strength,
            allow_partial=self.allow_partial,
            diagonal_movement=self.diagonal_movement,
        )


__all__ = [
    "PathfindingPolicy",
    "WaypointPolicy",
    "HierarchicalPathfindingPolicy",
]
