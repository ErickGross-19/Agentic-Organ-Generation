"""
Radius calculation functions following biophysical rules.
"""

import numpy as np
from typing import Tuple
from .constraints import RadiusRuleSpec


def murray_split(
    parent_radius: float,
    gamma: float = 3.0,
    split_ratio: float = 0.6,
    min_radius: float = 0.0003,
) -> Tuple[float, float]:
    """
    Compute child radii from parent radius using Murray's law.
    
    Murray's law: r_parent^gamma = r_child1^gamma + r_child2^gamma
    
    Parameters
    ----------
    parent_radius : float
        Parent vessel radius
    gamma : float
        Murray's law exponent (typically 3.0)
    split_ratio : float
        Ratio for splitting (0.5 = symmetric, >0.5 = asymmetric)
    min_radius : float
        Minimum allowed radius
    
    Returns
    -------
    r1, r2 : float, float
        Child radii (r1 >= r2)
    """
    max_iters = 20
    alpha = split_ratio
    
    for _ in range(max_iters):
        beta_gamma = 1.0 - alpha**gamma
        if beta_gamma <= 0:
            alpha *= 0.95
            continue
        beta = beta_gamma ** (1.0 / gamma)
        
        total = alpha + beta
        if total < 1e-10:
            break
        alpha = split_ratio * total
    
    r1 = alpha * parent_radius
    r2 = beta * parent_radius
    
    if r1 < min_radius or r2 < min_radius:
        return min_radius, min_radius
    
    return max(r1, r2), min(r1, r2)


def apply_radius_rule(
    parent_radius: float,
    rule_spec: RadiusRuleSpec,
    rng: np.random.Generator = None,
) -> Tuple[float, float]:
    """
    Apply radius rule to compute child radii.
    
    Parameters
    ----------
    parent_radius : float
        Parent vessel radius
    rule_spec : RadiusRuleSpec
        Radius rule specification
    rng : np.random.Generator, optional
        Random number generator for stochastic rules
    
    Returns
    -------
    r1, r2 : float, float
        Child radii
    """
    if rule_spec.kind == "murray":
        gamma = rule_spec.params.get("gamma", 3.0)
        split_ratio_mean = rule_spec.params.get("split_ratio_mean", 0.6)
        split_ratio_std = rule_spec.params.get("split_ratio_std", 0.1)
        
        if rng is not None:
            split_ratio = rng.normal(split_ratio_mean, split_ratio_std)
            split_ratio = np.clip(split_ratio, 0.4, 0.8)
        else:
            split_ratio = split_ratio_mean
        
        return murray_split(parent_radius, gamma, split_ratio)
    
    elif rule_spec.kind == "fixed":
        radius = rule_spec.params.get("radius", parent_radius)
        return radius, radius
    
    elif rule_spec.kind == "linear_taper":
        taper_factor = rule_spec.params.get("taper_factor", 0.9)
        r1 = parent_radius * taper_factor
        r2 = parent_radius * taper_factor
        return r1, r2
    
    else:
        raise ValueError(f"Unknown radius rule kind: {rule_spec.kind}")
