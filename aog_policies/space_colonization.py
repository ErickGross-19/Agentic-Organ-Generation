"""
Space Colonization Policy for AOG.

This module contains the SpaceColonizationPolicy dataclass that controls
all behavior of the space colonization backend for tree-like vascular
network generation.

DESIGN GOALS
------------
A) Trunk-first + root suppression: Prevent "inlet starburst" where root
   spawns many children immediately.
B) Apical dominance + angular-clustering-based splitting: Prevent "linear
   forever" branches by enabling proper branching when attractor field supports it.

All behavior is controlled via this policy - no hidden constants.
Behavior is reproducible when seed is fixed.
Max split degree per node <= 3.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Literal


@dataclass
class SpaceColonizationPolicy:
    """
    Policy for space colonization backend controlling tree-like growth.

    This policy provides all knobs for:
    A) Trunk/root suppression - prevent inlet starburst
    B) Apical dominance - reduce parallel linear growth
    C) Cluster-based splitting - enable proper branching

    JSON Schema:
    {
        "enabled": bool,
        
        # A) Trunk / root suppression
        "trunk_steps": int,
        "trunk_direction_mode": "inlet_direction" | "dominant_cluster",
        "max_root_children": int,
        "branch_enable_after_steps": int,
        "branch_enable_after_distance": float (meters),
        
        # Apical dominance
        "apical_dominance_alpha": float,
        "active_tip_fraction": float,
        "min_active_tips": int,
        "dominance_mode": "probabilistic" | "topk",
        
        # B) Cluster-based splitting
        "enable_cluster_splitting": bool,
        "cluster_angle_threshold_deg": float,
        "min_attractors_to_split": int,
        "max_children_per_split": int,
        "split_cooldown_steps": int,
        "allow_trifurcation_prob": float,
        "split_strength_mode": "equal" | "proportional_to_cluster_support",
        
        # Randomness + determinism
        "rng_mode": "seeded",
        "noise_angle_deg": float,
        "noise_scale_by_support": bool,
        
        # Safety
        "max_children_per_node_total": int,
        "min_branch_segment_length": float (meters)
    }
    """
    enabled: bool = True
    
    # A) Trunk / root suppression - prevent inlet starburst
    trunk_steps: int = 10
    trunk_direction_mode: Literal["inlet_direction", "dominant_cluster"] = "inlet_direction"
    max_root_children: int = 1
    branch_enable_after_steps: int = 10
    branch_enable_after_distance: float = 0.002  # 2mm minimum distance from inlet before branching
    
    # Apical dominance - reduce parallel linear growth
    apical_dominance_alpha: float = 1.5  # weight = support^alpha
    active_tip_fraction: float = 0.4  # only some tips grow each iteration
    min_active_tips: int = 5
    dominance_mode: Literal["probabilistic", "topk"] = "probabilistic"
    
    # B) Cluster-based splitting - enable proper branching
    enable_cluster_splitting: bool = True
    cluster_angle_threshold_deg: float = 35.0  # degrees
    min_attractors_to_split: int = 8
    max_children_per_split: int = 3
    split_cooldown_steps: int = 8  # avoid rapid repeated splits at same tip
    allow_trifurcation_prob: float = 0.25  # probability of 3-way split (seeded)
    split_strength_mode: Literal["equal", "proportional_to_cluster_support"] = "proportional_to_cluster_support"
    
    # Randomness + determinism
    rng_mode: Literal["seeded"] = "seeded"  # must be seeded for reproducibility
    noise_angle_deg: float = 5.0  # small organic variation
    noise_scale_by_support: bool = True  # optional: scale noise by support
    
    # Safety
    max_children_per_node_total: int = 3  # cap total children per node
    min_branch_segment_length: float = 0.0002  # 0.2mm - derived from ResolutionPolicy if possible

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SpaceColonizationPolicy":
        """Create from dictionary."""
        return SpaceColonizationPolicy(**{
            k: v for k, v in d.items() 
            if k in SpaceColonizationPolicy.__dataclass_fields__
        })
    
    def validate(self) -> List[str]:
        """
        Validate policy parameters.
        
        Returns
        -------
        List[str]
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if self.trunk_steps < 0:
            errors.append(f"trunk_steps must be >= 0, got {self.trunk_steps}")
        
        if self.max_root_children < 1:
            errors.append(f"max_root_children must be >= 1, got {self.max_root_children}")
        
        if self.max_root_children > 3:
            errors.append(f"max_root_children must be <= 3, got {self.max_root_children}")
        
        if self.branch_enable_after_steps < self.trunk_steps:
            errors.append(
                f"branch_enable_after_steps ({self.branch_enable_after_steps}) "
                f"must be >= trunk_steps ({self.trunk_steps})"
            )
        
        if not 0.0 <= self.active_tip_fraction <= 1.0:
            errors.append(f"active_tip_fraction must be in [0, 1], got {self.active_tip_fraction}")
        
        if self.min_active_tips < 1:
            errors.append(f"min_active_tips must be >= 1, got {self.min_active_tips}")
        
        if self.cluster_angle_threshold_deg < 0 or self.cluster_angle_threshold_deg > 180:
            errors.append(
                f"cluster_angle_threshold_deg must be in [0, 180], "
                f"got {self.cluster_angle_threshold_deg}"
            )
        
        if self.min_attractors_to_split < 2:
            errors.append(f"min_attractors_to_split must be >= 2, got {self.min_attractors_to_split}")
        
        if self.max_children_per_split < 2 or self.max_children_per_split > 3:
            errors.append(
                f"max_children_per_split must be in [2, 3], got {self.max_children_per_split}"
            )
        
        if self.split_cooldown_steps < 0:
            errors.append(f"split_cooldown_steps must be >= 0, got {self.split_cooldown_steps}")
        
        if not 0.0 <= self.allow_trifurcation_prob <= 1.0:
            errors.append(
                f"allow_trifurcation_prob must be in [0, 1], got {self.allow_trifurcation_prob}"
            )
        
        if self.max_children_per_node_total < 1 or self.max_children_per_node_total > 3:
            errors.append(
                f"max_children_per_node_total must be in [1, 3], "
                f"got {self.max_children_per_node_total}"
            )
        
        if self.noise_angle_deg < 0:
            errors.append(f"noise_angle_deg must be >= 0, got {self.noise_angle_deg}")
        
        if self.min_branch_segment_length <= 0:
            errors.append(
                f"min_branch_segment_length must be > 0, got {self.min_branch_segment_length}"
            )
        
        if self.rng_mode != "seeded":
            errors.append(f"rng_mode must be 'seeded' for reproducibility, got {self.rng_mode}")
        
        return errors


__all__ = ["SpaceColonizationPolicy"]
