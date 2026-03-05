"""
Shared helper utilities for scaffold geometry generation.

This module provides common geometric primitives and utilities used across
multiple scaffold generators. Functions are organized into submodules:

- hollow_tubes: Creating hollow tubular structures
- bezier_curves: Smooth curve generation
- bifurcation: Y-junctions and branching structures
- mesh_utils: Enhanced boolean operations
- randomization: Noise and perturbation functions
- statistics: Geometry measurement and statistics
- biological_patterns: Pattern generation (honeycomb, voronoi, etc.)
"""

# Hollow tube creation
from .hollow_tubes import (
    create_hollow_vertical_tube,
    create_hollow_sinusoid,
    create_solid_tube_manifold,
    create_hollow_collector_manifold,
    create_hollow_vessel_network,
)

# Bezier curve generation
from .bezier_curves import (
    bezier_curve,
    bezier_quadratic,
    catmull_rom_spline,
)

# Bifurcation and branching
from .bifurcation import (
    create_bifurcation,
    create_trifurcation,
    create_branching_network,
)

# Mesh utilities
from .mesh_utils import (
    tree_union,
    batch_union,
    tree_union_parallel,
)

# Randomization and noise
from .randomization import (
    perturb_positions,
    perturb_unique_corners,
    generate_randomized_z_positions,
    apply_size_variance,
)

# Statistics calculations
from .statistics import (
    calculate_volume,
    calculate_surface_area,
    calculate_porosity,
    calculate_surface_to_volume_ratio,
    count_features,
    generate_standard_stats,
)

# Biological pattern generation
from .biological_patterns import (
    generate_honeycomb_centers,
    generate_voronoi_pattern,
    generate_fibonacci_spiral,
    get_corner_positions,
    find_unique_corners,
)

__all__ = [
    # Hollow tubes
    "create_hollow_vertical_tube",
    "create_hollow_sinusoid",
    "create_solid_tube_manifold",
    "create_hollow_collector_manifold",
    "create_hollow_vessel_network",
    # Bezier curves
    "bezier_curve",
    "bezier_quadratic",
    "catmull_rom_spline",
    # Bifurcation
    "create_bifurcation",
    "create_trifurcation",
    "create_branching_network",
    # Mesh utils
    "tree_union",
    "batch_union",
    "tree_union_parallel",
    # Randomization
    "perturb_positions",
    "perturb_unique_corners",
    "generate_randomized_z_positions",
    "apply_size_variance",
    # Statistics
    "calculate_volume",
    "calculate_surface_area",
    "calculate_porosity",
    "calculate_surface_to_volume_ratio",
    "count_features",
    "generate_standard_stats",
    # Biological patterns
    "generate_honeycomb_centers",
    "generate_voronoi_pattern",
    "generate_fibonacci_spiral",
    "get_corner_positions",
    "find_unique_corners",
]
