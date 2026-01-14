"""Utility functions for vascular library."""

from .units import (
    CANONICAL_UNIT,
    to_si_length,
    from_si_length,
    convert_length,
    detect_unit,
    warn_if_legacy_units,
)

from .schedules import (
    compute_bifurcation_depths,
    compute_taper_radius,
    compute_child_length_scale,
)

from .layout import (
    compute_inlet_positions,
    compute_outlet_positions,
    compute_grid_positions,
)

__all__ = [
    'CANONICAL_UNIT',
    'to_si_length',
    'from_si_length',
    'convert_length',
    'detect_unit',
    'warn_if_legacy_units',
    'compute_bifurcation_depths',
    'compute_taper_radius',
    'compute_child_length_scale',
    'compute_inlet_positions',
    'compute_outlet_positions',
    'compute_grid_positions',
]
