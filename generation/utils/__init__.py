"""Utility functions for vascular library."""

from .units import (
    CANONICAL_UNIT,
    to_si_length,
    from_si_length,
    convert_length,
    detect_unit,
    warn_if_legacy_units,
)

__all__ = [
    'CANONICAL_UNIT',
    'to_si_length',
    'from_si_length',
    'convert_length',
    'detect_unit',
    'warn_if_legacy_units',
]
