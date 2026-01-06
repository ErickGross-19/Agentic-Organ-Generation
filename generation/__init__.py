"""
Agentic Organ Generation - Generation Library (Part A)

This module provides the core functionality for LLM/agent-driven 3D organ structure generation.
It includes operations for space colonization, vascular network growth, bifurcation, collision
avoidance, anastomosis creation, and embedding structures into domains.

Primary Output: Domain-with-void scaffold mesh + supplementary surface mesh

Main Entry Points:
    - design_from_spec(): Build networks from declarative JSON specifications
    - run_experiment(): One-call orchestration of design -> evaluate -> export
    - create_network(): Low-level network initialization
    - space_colonization_step(): Organic growth algorithm

Example:
    >>> from generation.api import design_from_spec, run_experiment
    >>> from generation.specs import DesignSpec, TreeSpec, EllipsoidSpec
    >>> from generation.params import get_preset
    >>>
    >>> # Define network design as spec
    >>> spec = DesignSpec(
    ...     domain=EllipsoidSpec(center=(0,0,0), semi_axes=(50,50,50)),
    ...     tree=TreeSpec.single_inlet(
    ...         inlet_position=(-50,0,0),
    ...         inlet_radius=5.0,
    ...         colonization=get_preset("liver_arterial_dense")
    ...     )
    ... )
    >>>
    >>> # Run complete experiment
    >>> result = run_experiment(spec, output_dir="./output")
"""

from .api import design_from_spec, evaluate_network, run_experiment
from .ops import (
    create_network,
    add_inlet,
    add_outlet,
    grow_branch,
    bifurcate,
    space_colonization_step,
    get_collisions,
    avoid_collisions,
    create_anastomosis,
    embed_tree_as_negative_space,
)
from .core import VascularNetwork, EllipsoidDomain, BoxDomain
from .params import get_preset

__all__ = [
    # High-level API
    "design_from_spec",
    "evaluate_network", 
    "run_experiment",
    # Operations
    "create_network",
    "add_inlet",
    "add_outlet",
    "grow_branch",
    "bifurcate",
    "space_colonization_step",
    "get_collisions",
    "avoid_collisions",
    "create_anastomosis",
    "embed_tree_as_negative_space",
    # Core types
    "VascularNetwork",
    "EllipsoidDomain",
    "BoxDomain",
    # Params
    "get_preset",
]
