"""
Functions for smooth curve generation.

Provides Bezier curves and splines for creating smooth paths
used in tubular structures and vessel networks.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np


def bezier_curve(
    p0: Tuple[float, float, float],
    p1: Tuple[float, float, float],
    p2: Tuple[float, float, float],
    p3: Tuple[float, float, float],
    num_points: int = 20
) -> List[Tuple[float, float, float]]:
    """
    Generate points along a cubic Bezier curve.

    A cubic Bezier curve is defined by 4 control points:
    - p0: Start point (curve passes through)
    - p1: First control point (influences direction from p0)
    - p2: Second control point (influences direction to p3)
    - p3: End point (curve passes through)

    Args:
        p0: Start point (x, y, z)
        p1: First control point (x, y, z)
        p2: Second control point (x, y, z)
        p3: End point (x, y, z)
        num_points: Number of points to generate along the curve

    Returns:
        List of (x, y, z) points along the curve

    Example:
        >>> # Create a curved path from (0,0,0) to (1,0,1)
        >>> points = bezier_curve(
        ...     (0, 0, 0),     # Start
        ...     (0.3, 0, 0),   # Control 1 - pulls curve right
        ...     (0.7, 0, 1),   # Control 2 - pulls curve up
        ...     (1, 0, 1)      # End
        ... )
    """
    points = []
    for i in range(num_points + 1):
        t = i / num_points
        t2 = t * t
        t3 = t2 * t
        mt = 1 - t
        mt2 = mt * mt
        mt3 = mt2 * mt

        x = mt3*p0[0] + 3*mt2*t*p1[0] + 3*mt*t2*p2[0] + t3*p3[0]
        y = mt3*p0[1] + 3*mt2*t*p1[1] + 3*mt*t2*p2[1] + t3*p3[1]
        z = mt3*p0[2] + 3*mt2*t*p1[2] + 3*mt*t2*p2[2] + t3*p3[2]
        points.append((x, y, z))

    return points


def bezier_quadratic(
    p0: Tuple[float, float, float],
    p1: Tuple[float, float, float],
    p2: Tuple[float, float, float],
    num_points: int = 20
) -> List[Tuple[float, float, float]]:
    """
    Generate points along a quadratic Bezier curve.

    A quadratic Bezier curve is defined by 3 control points:
    - p0: Start point (curve passes through)
    - p1: Control point (influences curve shape)
    - p2: End point (curve passes through)

    Simpler than cubic Bezier but less flexible for complex curves.

    Args:
        p0: Start point (x, y, z)
        p1: Control point (x, y, z)
        p2: End point (x, y, z)
        num_points: Number of points to generate along the curve

    Returns:
        List of (x, y, z) points along the curve

    Example:
        >>> # Simple arc from (0,0,0) to (2,0,0) peaking at y=1
        >>> points = bezier_quadratic(
        ...     (0, 0, 0),   # Start
        ...     (1, 1, 0),   # Control - pulls curve up
        ...     (2, 0, 0)    # End
        ... )
    """
    points = []
    for i in range(num_points + 1):
        t = i / num_points
        mt = 1 - t

        x = mt*mt*p0[0] + 2*mt*t*p1[0] + t*t*p2[0]
        y = mt*mt*p0[1] + 2*mt*t*p1[1] + t*t*p2[1]
        z = mt*mt*p0[2] + 2*mt*t*p1[2] + t*t*p2[2]
        points.append((x, y, z))

    return points


def catmull_rom_spline(
    points: List[Tuple[float, float, float]],
    num_points_per_segment: int = 10
) -> List[Tuple[float, float, float]]:
    """
    Generate a smooth Catmull-Rom spline through given points.

    Unlike Bezier curves, a Catmull-Rom spline passes through ALL control
    points, making it ideal for creating smooth paths through known waypoints.

    Args:
        points: List of (x, y, z) points the spline should pass through.
                Minimum 4 points required.
        num_points_per_segment: Number of interpolated points between each
                                pair of original points

    Returns:
        List of (x, y, z) points along the smooth spline

    Raises:
        ValueError: If fewer than 4 points provided

    Example:
        >>> # Create smooth path through 5 waypoints
        >>> waypoints = [(0,0,0), (1,1,0), (2,0,0), (3,1,0), (4,0,0)]
        >>> smooth_path = catmull_rom_spline(waypoints, num_points_per_segment=20)
    """
    if len(points) < 4:
        raise ValueError("Catmull-Rom spline requires at least 4 points")

    pts = np.array(points)
    result = []

    # Tension parameter (0.5 is standard for Catmull-Rom)
    tension = 0.5

    for i in range(len(pts) - 3):
        p0, p1, p2, p3 = pts[i], pts[i+1], pts[i+2], pts[i+3]

        for j in range(num_points_per_segment):
            t = j / num_points_per_segment
            t2 = t * t
            t3 = t2 * t

            # Catmull-Rom matrix coefficients
            c0 = -tension * t3 + 2 * tension * t2 - tension * t
            c1 = (2 - tension) * t3 + (tension - 3) * t2 + 1
            c2 = (tension - 2) * t3 + (3 - 2 * tension) * t2 + tension * t
            c3 = tension * t3 - tension * t2

            point = c0 * p0 + c1 * p1 + c2 * p2 + c3 * p3
            result.append(tuple(point))

    # Add the last point
    result.append(tuple(pts[-2]))

    return result


def arc_points(
    center: Tuple[float, float, float],
    radius: float,
    start_angle: float,
    end_angle: float,
    axis: str = 'z',
    num_points: int = 20
) -> List[Tuple[float, float, float]]:
    """
    Generate points along a circular arc.

    Args:
        center: Center point of the arc (x, y, z)
        radius: Radius of the arc
        start_angle: Starting angle in radians
        end_angle: Ending angle in radians
        axis: Axis perpendicular to the arc plane ('x', 'y', or 'z')
        num_points: Number of points to generate

    Returns:
        List of (x, y, z) points along the arc

    Example:
        >>> # Create a 90-degree arc in XY plane
        >>> arc = arc_points((0, 0, 0), 1.0, 0, np.pi/2, axis='z', num_points=10)
    """
    cx, cy, cz = center
    angles = np.linspace(start_angle, end_angle, num_points)
    points = []

    for angle in angles:
        if axis == 'z':
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            z = cz
        elif axis == 'x':
            x = cx
            y = cy + radius * np.cos(angle)
            z = cz + radius * np.sin(angle)
        else:  # axis == 'y'
            x = cx + radius * np.cos(angle)
            y = cy
            z = cz + radius * np.sin(angle)

        points.append((x, y, z))

    return points


def helix_points(
    center: Tuple[float, float, float],
    radius: float,
    pitch: float,
    num_turns: float,
    num_points_per_turn: int = 20,
    direction: int = 1
) -> List[Tuple[float, float, float]]:
    """
    Generate points along a helix (spiral in 3D).

    Args:
        center: Starting center point (x, y, z)
        radius: Radius of the helix
        pitch: Vertical distance per full turn
        num_turns: Number of complete turns
        num_points_per_turn: Points generated per turn
        direction: 1 for right-handed helix, -1 for left-handed

    Returns:
        List of (x, y, z) points along the helix

    Example:
        >>> # Create a 3-turn helix rising 6mm total
        >>> helix = helix_points((0,0,0), radius=1.0, pitch=2.0, num_turns=3)
    """
    cx, cy, cz = center
    total_points = int(num_turns * num_points_per_turn)
    points = []

    for i in range(total_points + 1):
        t = i / num_points_per_turn  # t in turns
        angle = 2 * np.pi * t * direction

        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        z = cz + pitch * t

        points.append((x, y, z))

    return points
