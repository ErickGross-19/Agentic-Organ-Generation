"""
Agentic Organ Generation - Generation Library (Part A)

This module provides the core functionality for LLM/agent-driven 3D organ structure generation.
It includes operations for space colonization, vascular network growth, bifurcation, collision
avoidance, anastomosis creation, and embedding structures into domains.

Primary Output: Domain-with-void scaffold mesh + supplementary surface mesh

IMPORT GUIDE
------------
Import from specific submodules to avoid loading unnecessary dependencies:

    # High-level API (recommended entry points)
    from generation.api import design_from_spec, run_experiment, generate_network
    
    # Specifications
    from generation.specs import DesignSpec, TreeSpec, EllipsoidSpec, CylinderSpec
    
    # Core types
    from generation.core import VascularNetwork, EllipsoidDomain, BoxDomain
    
    # Operations (import only what you need)
    from generation.ops import create_network, add_inlet, add_outlet
    from generation.ops.collision import get_collisions, detect_collisions
    from generation.ops.embedding import embed_tree_as_negative_space
    from generation.ops.features import add_raised_ridge, FaceId
    from generation.ops.pathfinding import find_path, grow_toward_targets
    
    # Backends
    from generation.backends import CCOHybridBackend, SpaceColonizationBackend
    
    # Parameters
    from generation.params import get_preset

Example:
    >>> from generation.api import run_experiment
    >>> from generation.specs import DesignSpec, TreeSpec, EllipsoidSpec
    >>> from generation.params import get_preset
    >>>
    >>> spec = DesignSpec(
    ...     domain=EllipsoidSpec(center=(0,0,0), semi_axes=(50,50,50)),
    ...     tree=TreeSpec.single_inlet(
    ...         inlet_position=(-50,0,0),
    ...         inlet_radius=5.0,
    ...         colonization=get_preset("liver_arterial_dense")
    ...     )
    ... )
    >>> result = run_experiment(spec, output_dir="./output")

NOTE: This __init__.py is intentionally lightweight to prevent import-time errors
when optional dependencies are missing. Import from specific submodules as shown above.
"""

# Only export submodule names for discovery - actual imports should be from submodules
__all__ = [
    "api",
    "backends",
    "core",
    "ops",
    "params",
    "specs",
    "utils",
]
