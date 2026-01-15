"""
Individual validity checks for mesh and network validation.

This module provides modular check functions that can be enabled/disabled
via policy objects.
"""

from .watertight import check_watertight
from .components import check_components
from .topology import check_topology
from .dimensions import check_dimensions

__all__ = [
    "check_watertight",
    "check_components",
    "check_topology",
    "check_dimensions",
]
