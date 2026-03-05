"""
Schedule utilities for vascular network generation.

This module provides reusable schedule functions for:
- Bifurcation depth spacing (convex, linear, etc.)
- Radius taper schedules (shifted exponential, Murray's law, etc.)
- Child length scaling

These utilities are ported from malaria_venule_inserts.py for general use.
"""

from typing import List, Optional
import math


def compute_bifurcation_depths(
    num_levels: int,
    total_depth: float,
    top_margin_fraction: float = 0.15,
    bottom_margin_fraction: float = 0.05,
    depth_power: float = 1.6,
) -> List[float]:
    """
    Compute bifurcation depths with convex spacing.
    
    Convex spacing (depth_power > 1) gives later levels more room, which helps
    prevent early thick branches from dominating in small domains.
    
    Parameters
    ----------
    num_levels : int
        Number of bifurcation levels
    total_depth : float
        Total depth available (e.g., cylinder height)
    top_margin_fraction : float
        Fraction of depth to reserve at top as trunk (default: 0.15)
    bottom_margin_fraction : float
        Fraction of depth to reserve at bottom (default: 0.05)
    depth_power : float
        Power for convex spacing (default: 1.6)
        - p = 1.0: linear spacing
        - p > 1.0: convex spacing (more room for later levels)
        - p < 1.0: concave spacing (more room for early levels)
        
    Returns
    -------
    list of float
        Bifurcation depths from top (positive values)
        
    Examples
    --------
    >>> depths = compute_bifurcation_depths(7, 0.002, top_margin_fraction=0.15)
    >>> # Returns 7 depths with convex spacing in a 2mm domain
    """
    if num_levels <= 0:
        return []
    
    top_margin = total_depth * top_margin_fraction
    bottom_margin = total_depth * bottom_margin_fraction
    usable_depth = total_depth - top_margin - bottom_margin
    
    if usable_depth <= 0:
        return [top_margin + (total_depth - top_margin) * (i + 1) / (num_levels + 1) 
                for i in range(num_levels)]
    
    depths = []
    for i in range(num_levels):
        t = (i + 1) / (num_levels + 1)
        depth = top_margin + usable_depth * (t ** depth_power)
        depths.append(depth)
    
    return depths


def compute_taper_radius(
    level: int,
    num_levels: int,
    inlet_radius: float,
    terminal_radius: float,
    taper_power: float = 0.8,
    use_shifted_mapping: bool = True,
) -> float:
    """
    Compute radius at a given bifurcation level using shifted exponential taper.
    
    The shifted mapping ensures level 0 is already thinner than inlet radius,
    preventing the first split from being too thick.
    
    Parameters
    ----------
    level : int
        Current bifurcation level (0-indexed)
    num_levels : int
        Total number of levels
    inlet_radius : float
        Radius at inlet (largest)
    terminal_radius : float
        Radius at terminal (smallest)
    taper_power : float
        Power for exponential taper (default: 0.8)
        - p < 1.0: faster early shrinking
        - p = 1.0: linear
        - p > 1.0: slower early shrinking
    use_shifted_mapping : bool
        If True, use (level+1)/(num_levels+1) mapping so level 0 is thinner
        
    Returns
    -------
    float
        Radius at the given level
        
    Examples
    --------
    >>> r = compute_taper_radius(0, 7, 0.0005, 0.00005)
    >>> # Returns radius for level 0 (already thinner than inlet)
    """
    if num_levels <= 0:
        return inlet_radius
    
    if use_shifted_mapping:
        t = (level + 1) / (num_levels + 1)
    else:
        t = level / max(num_levels - 1, 1)
    
    t_powered = t ** taper_power
    
    radius = inlet_radius + (terminal_radius - inlet_radius) * t_powered
    
    return max(radius, terminal_radius)


def compute_child_length_scale(
    level: int,
    num_levels: int,
    min_scale: float = 0.6,
    max_scale: float = 1.0,
) -> float:
    """
    Compute length scale factor for child branches at a given level.
    
    Early branches are shorter (less visual mass), later branches are longer
    (more visible). This helps balance the visual appearance of the tree.
    
    Parameters
    ----------
    level : int
        Current bifurcation level (0-indexed)
    num_levels : int
        Total number of levels
    min_scale : float
        Scale factor for earliest level (default: 0.6)
    max_scale : float
        Scale factor for latest level (default: 1.0)
        
    Returns
    -------
    float
        Length scale factor for the given level
        
    Examples
    --------
    >>> scale = compute_child_length_scale(0, 7)
    >>> # Returns 0.6 (early branches are shorter)
    >>> scale = compute_child_length_scale(6, 7)
    >>> # Returns ~1.0 (later branches are longer)
    """
    if num_levels <= 1:
        return max_scale
    
    t = level / (num_levels - 1)
    
    return min_scale + (max_scale - min_scale) * t
