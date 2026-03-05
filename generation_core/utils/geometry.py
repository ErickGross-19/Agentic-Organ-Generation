"""
Canonical geometry utilities for vascular network operations.

This module provides the single source of truth for geometric computations
used across the codebase, including collision detection, spatial indexing,
and distance calculations.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

import numpy as np
from typing import Tuple


def segment_segment_distance(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
) -> float:
    """
    Compute exact minimum distance between two 3D line segments.
    
    This is the canonical implementation used throughout the codebase for
    collision detection and spatial queries. It correctly handles all cases
    including parallel segments, degenerate segments (points), and
    intersecting segments (returns 0.0).
    
    Segment 1: p1 to p2
    Segment 2: p3 to p4
    
    Parameters
    ----------
    p1, p2 : np.ndarray
        Endpoints of segment 1 (shape (3,))
    p3, p4 : np.ndarray
        Endpoints of segment 2 (shape (3,))
        
    Returns
    -------
    float
        Minimum distance between the two segments.
        Returns 0.0 if segments intersect.
        
    Examples
    --------
    >>> import numpy as np
    >>> # Parallel segments
    >>> p1, p2 = np.array([0, 0, 0]), np.array([1, 0, 0])
    >>> p3, p4 = np.array([0, 1, 0]), np.array([1, 1, 0])
    >>> segment_segment_distance(p1, p2, p3, p4)
    1.0
    
    >>> # Intersecting segments (perpendicular cross)
    >>> p1, p2 = np.array([0, 0, 0]), np.array([1, 0, 0])
    >>> p3, p4 = np.array([0.5, -0.5, 0]), np.array([0.5, 0.5, 0])
    >>> segment_segment_distance(p1, p2, p3, p4)
    0.0
    """
    d1 = p2 - p1  # Direction of segment 1
    d2 = p4 - p3  # Direction of segment 2
    r = p1 - p3
    
    a = np.dot(d1, d1)  # |d1|^2
    e = np.dot(d2, d2)  # |d2|^2
    f = np.dot(d2, r)
    
    EPSILON = 1e-10
    
    # Check if both segments are degenerate (points)
    if a < EPSILON and e < EPSILON:
        return float(np.linalg.norm(r))
    
    # Check if segment 1 is degenerate (point)
    if a < EPSILON:
        s = 0.0
        t = np.clip(f / e, 0.0, 1.0)
    else:
        c = np.dot(d1, r)
        
        # Check if segment 2 is degenerate (point)
        if e < EPSILON:
            t = 0.0
            s = np.clip(-c / a, 0.0, 1.0)
        else:
            # General non-degenerate case
            b = np.dot(d1, d2)
            denom = a * e - b * b  # Always >= 0
            
            # If segments are not parallel, compute closest point on line 1 to line 2
            if denom > EPSILON:
                s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
            else:
                # Segments are parallel, pick arbitrary s
                s = 0.0
            
            # Compute point on line 2 closest to S1(s)
            t = (b * s + f) / e
            
            # If t is outside [0,1], clamp and recompute s
            if t < 0.0:
                t = 0.0
                s = np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = np.clip((b - c) / a, 0.0, 1.0)
    
    # Compute closest points
    closest1 = p1 + s * d1
    closest2 = p3 + t * d2
    
    return float(np.linalg.norm(closest1 - closest2))


def capsule_capsule_distance(
    seg1_start: np.ndarray,
    seg1_end: np.ndarray,
    seg1_radius: float,
    seg2_start: np.ndarray,
    seg2_end: np.ndarray,
    seg2_radius: float,
) -> Tuple[float, float]:
    """
    Compute distance between two capsules (swept cylinders).
    
    A capsule is a cylinder with hemispherical caps at each end.
    This function computes the clearance between capsule surfaces.
    
    Parameters
    ----------
    seg1_start, seg1_end : np.ndarray
        Endpoints of first capsule centerline
    seg1_radius : float
        Radius of first capsule
    seg2_start, seg2_end : np.ndarray
        Endpoints of second capsule centerline
    seg2_radius : float
        Radius of second capsule
        
    Returns
    -------
    centerline_distance : float
        Distance between centerlines
    surface_clearance : float
        Clearance between capsule surfaces (negative if overlapping)
    """
    centerline_distance = segment_segment_distance(
        seg1_start, seg1_end, seg2_start, seg2_end
    )
    surface_clearance = centerline_distance - seg1_radius - seg2_radius
    return centerline_distance, surface_clearance


def point_to_segment_distance(
    point: np.ndarray,
    seg_start: np.ndarray,
    seg_end: np.ndarray,
) -> float:
    """
    Compute minimum distance from a point to a line segment.
    
    Parameters
    ----------
    point : np.ndarray
        Query point (shape (3,))
    seg_start, seg_end : np.ndarray
        Endpoints of segment (shape (3,))
        
    Returns
    -------
    float
        Minimum distance from point to segment
    """
    v = seg_end - seg_start
    length_sq = np.dot(v, v)
    
    if length_sq < 1e-10:
        return float(np.linalg.norm(point - seg_start))
    
    t = np.dot(point - seg_start, v) / length_sq
    t = np.clip(t, 0.0, 1.0)
    
    closest = seg_start + t * v
    return float(np.linalg.norm(point - closest))


__all__ = [
    "segment_segment_distance",
    "capsule_capsule_distance",
    "point_to_segment_distance",
]
