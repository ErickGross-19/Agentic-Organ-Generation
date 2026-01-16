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

from .faces import (
    CANONICAL_FACES,
    FaceId,
    validate_face,
    face_to_enum,
    face_normal,
    face_center,
    face_frame,
    opposite_face,
)

from .scale import (
    domain_scale,
    domain_extents,
    eps,
    containment_tolerance,
    collision_tolerance,
    snap_tolerance,
    prune_tolerance,
    boundary_mask_tolerance,
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
    'CANONICAL_FACES',
    'FaceId',
    'validate_face',
    'face_to_enum',
    'face_normal',
    'face_center',
    'face_frame',
    'opposite_face',
    'domain_scale',
    'domain_extents',
    'eps',
    'containment_tolerance',
    'collision_tolerance',
    'snap_tolerance',
    'prune_tolerance',
    'boundary_mask_tolerance',
]
