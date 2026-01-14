"""
Layout utilities for vascular network generation.

This module provides utilities for computing inlet/outlet positions
and other spatial layout calculations.

These utilities are ported from malaria_venule_inserts.py for general use.
"""

from typing import List, Tuple, Optional
import math
import numpy as np


def compute_inlet_positions(
    num_inlets: int,
    domain_radius: float,
    inlet_radius: float,
    ridge_inner_radius: Optional[float] = None,
    placement_fraction: float = 0.5,
    angular_offset: float = 0.0,
) -> List[Tuple[float, float]]:
    """
    Compute inlet positions arranged in a circle within a cylindrical domain.
    
    Positions are computed to ensure inlets fit within the domain while
    accounting for ridge geometry if present.
    
    Parameters
    ----------
    num_inlets : int
        Number of inlets to place
    domain_radius : float
        Radius of the cylindrical domain
    inlet_radius : float
        Radius of each inlet
    ridge_inner_radius : float, optional
        Inner radius of ridge if present. If None, uses domain_radius.
    placement_fraction : float
        Fraction of available radius for placement (default: 0.5)
    angular_offset : float
        Angular offset in radians for first inlet (default: 0.0)
        
    Returns
    -------
    list of (float, float)
        List of (x, y) positions for each inlet
        
    Raises
    ------
    ValueError
        If inlets cannot fit within the domain
        
    Examples
    --------
    >>> positions = compute_inlet_positions(4, 0.001, 0.0005)
    >>> # Returns 4 positions arranged in a circle
    """
    if num_inlets <= 0:
        return []
    
    limiting_radius = domain_radius
    if ridge_inner_radius is not None:
        limiting_radius = min(domain_radius, ridge_inner_radius)
    
    max_placement_radius = limiting_radius - inlet_radius
    
    if max_placement_radius < 0:
        raise ValueError(
            f"Inlet radius ({inlet_radius}) is too large for domain "
            f"(limiting radius: {limiting_radius})"
        )
    
    if num_inlets == 1:
        return [(0.0, 0.0)]
    
    placement_radius = max_placement_radius * placement_fraction
    
    positions = []
    for i in range(num_inlets):
        angle = angular_offset + 2 * math.pi * i / num_inlets
        x = placement_radius * math.cos(angle)
        y = placement_radius * math.sin(angle)
        positions.append((x, y))
    
    return positions


def compute_outlet_positions(
    num_outlets: int,
    domain_radius: float,
    outlet_radius: float,
    placement_fraction: float = 0.7,
    angular_offset: float = 0.0,
) -> List[Tuple[float, float]]:
    """
    Compute outlet positions arranged in a circle at the bottom of a domain.
    
    Parameters
    ----------
    num_outlets : int
        Number of outlets to place
    domain_radius : float
        Radius of the cylindrical domain
    outlet_radius : float
        Radius of each outlet
    placement_fraction : float
        Fraction of available radius for placement (default: 0.7)
    angular_offset : float
        Angular offset in radians for first outlet (default: 0.0)
        
    Returns
    -------
    list of (float, float)
        List of (x, y) positions for each outlet
        
    Examples
    --------
    >>> positions = compute_outlet_positions(8, 0.001, 0.0001)
    >>> # Returns 8 positions arranged in a circle
    """
    if num_outlets <= 0:
        return []
    
    max_placement_radius = domain_radius - outlet_radius
    
    if max_placement_radius < 0:
        max_placement_radius = domain_radius * 0.9
    
    if num_outlets == 1:
        return [(0.0, 0.0)]
    
    placement_radius = max_placement_radius * placement_fraction
    
    positions = []
    for i in range(num_outlets):
        angle = angular_offset + 2 * math.pi * i / num_outlets
        x = placement_radius * math.cos(angle)
        y = placement_radius * math.sin(angle)
        positions.append((x, y))
    
    return positions


def compute_grid_positions(
    num_positions: int,
    width: float,
    height: float,
    margin: float = 0.0,
) -> List[Tuple[float, float]]:
    """
    Compute positions arranged in a grid pattern.
    
    Useful for rectangular domains or when a regular grid is preferred
    over circular arrangement.
    
    Parameters
    ----------
    num_positions : int
        Number of positions to compute
    width : float
        Width of the domain
    height : float
        Height of the domain
    margin : float
        Margin from edges (default: 0.0)
        
    Returns
    -------
    list of (float, float)
        List of (x, y) positions
        
    Examples
    --------
    >>> positions = compute_grid_positions(9, 0.002, 0.002)
    >>> # Returns 9 positions in a 3x3 grid
    """
    if num_positions <= 0:
        return []
    
    if num_positions == 1:
        return [(0.0, 0.0)]
    
    cols = int(math.ceil(math.sqrt(num_positions)))
    rows = int(math.ceil(num_positions / cols))
    
    usable_width = width - 2 * margin
    usable_height = height - 2 * margin
    
    x_start = -usable_width / 2
    y_start = -usable_height / 2
    
    x_step = usable_width / max(cols - 1, 1) if cols > 1 else 0
    y_step = usable_height / max(rows - 1, 1) if rows > 1 else 0
    
    positions = []
    for i in range(num_positions):
        col = i % cols
        row = i // cols
        x = x_start + col * x_step
        y = y_start + row * y_step
        positions.append((x, y))
    
    return positions
