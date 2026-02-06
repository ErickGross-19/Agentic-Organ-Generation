"""
Utility functions for AOG vascular integration.
"""

import numpy as np
from typing import List, Tuple


def normalize_direction(direction: List[float]) -> np.ndarray:
    """
    Normalize a direction vector.

    Parameters
    ----------
    direction : List[float]
        Direction vector [x, y, z]

    Returns
    -------
    np.ndarray
        Normalized direction vector
    """
    vec = np.array(direction)
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        raise ValueError("Direction vector has zero length")
    return vec / norm


def meters_to_mm(value_m: float) -> float:
    """Convert meters to millimeters."""
    return value_m * 1000.0


def mm_to_meters(value_mm: float) -> float:
    """Convert millimeters to meters."""
    return value_mm / 1000.0
