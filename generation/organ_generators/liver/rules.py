"""
Biophysical rules for vascular network generation.

Implements Murray's law, branching angle distributions, and other
physiological constraints.
"""

import numpy as np
from typing import Tuple
from .config import MurrayLawConfig, BranchingConfig


def murray_split_radii(
    parent_radius: float,
    config: MurrayLawConfig,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """
    Compute child radii from parent radius using Murray's law.
    
    Murray's law: r_parent^gamma = r_child1^gamma + r_child2^gamma
    
    Parameters
    ----------
    parent_radius : float
        Parent vessel radius
    config : MurrayLawConfig
        Murray's law configuration
    rng : np.random.Generator
        Random number generator
    
    Returns
    -------
    r1, r2 : float, float
        Child radii (r1 >= r2)
    """
    gamma = config.gamma
    
    split_ratio = rng.normal(config.split_ratio_mean, config.split_ratio_std)
    split_ratio = np.clip(split_ratio, 0.4, 0.8)
    
    
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
    
    if r1 < config.min_radius or r2 < config.min_radius:
        return config.min_radius, config.min_radius
    
    return max(r1, r2), min(r1, r2)


def compute_branching_probability(
    radius: float,
    root_radius: float,
    config: BranchingConfig,
    rng: np.random.Generator,
) -> bool:
    """
    Determine if a tip should branch based on its radius.
    
    Probability decreases as radius decreases (smaller vessels branch less).
    
    Parameters
    ----------
    radius : float
        Current tip radius
    root_radius : float
        Root radius of the tree
    config : BranchingConfig
        Branching configuration
    rng : np.random.Generator
        Random number generator
    
    Returns
    -------
    should_branch : bool
        Whether this tip should branch
    """
    prob = config.base_branching_prob * (radius / root_radius) ** config.prob_exponent
    prob = np.clip(prob, 0.0, 1.0)
    
    return rng.random() < prob


def sample_branch_directions(
    parent_direction: np.ndarray,
    config: BranchingConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample two child branch directions from parent direction.
    
    Uses branching angle distribution and random azimuthal angles.
    
    Parameters
    ----------
    parent_direction : (3,) array
        Parent vessel direction (unit vector)
    config : BranchingConfig
        Branching configuration
    rng : np.random.Generator
        Random number generator
    
    Returns
    -------
    dir1, dir2 : (3,) array, (3,) array
        Two child directions (unit vectors)
    """
    from .geometry import rotate_vector, get_perpendicular_vector
    
    parent_direction = parent_direction / np.linalg.norm(parent_direction)
    
    angle1_deg = rng.normal(config.branch_angle_mean, config.branch_angle_std)
    angle2_deg = rng.normal(config.branch_angle_mean, config.branch_angle_std)
    
    angle1_deg = np.clip(angle1_deg, 20.0, 80.0)
    angle2_deg = np.clip(angle2_deg, 20.0, 80.0)
    
    angle1_rad = np.radians(angle1_deg)
    angle2_rad = np.radians(angle2_deg)
    
    azimuth1 = rng.uniform(0, 2 * np.pi)
    azimuth2 = rng.uniform(0, 2 * np.pi)
    
    perp = get_perpendicular_vector(parent_direction)
    
    axis1 = rotate_vector(perp, parent_direction, azimuth1)
    axis2 = rotate_vector(perp, parent_direction, azimuth2)
    
    dir1 = rotate_vector(parent_direction, axis1, angle1_rad)
    dir2 = rotate_vector(parent_direction, axis2, angle2_rad)
    
    dir1 = dir1 / np.linalg.norm(dir1)
    dir2 = dir2 / np.linalg.norm(dir2)
    
    return dir1, dir2


def perturb_direction(
    direction: np.ndarray,
    noise_std_deg: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add small random perturbation to a direction vector.
    
    Parameters
    ----------
    direction : (3,) array
        Current direction (unit vector)
    noise_std_deg : float
        Standard deviation of perturbation in degrees
    rng : np.random.Generator
        Random number generator
    
    Returns
    -------
    perturbed : (3,) array
        Perturbed direction (unit vector)
    """
    from .geometry import rotate_vector, get_perpendicular_vector
    
    direction = direction / np.linalg.norm(direction)
    
    angle_deg = rng.normal(0.0, noise_std_deg)
    angle_rad = np.radians(angle_deg)
    
    perp = get_perpendicular_vector(direction)
    azimuth = rng.uniform(0, 2 * np.pi)
    axis = rotate_vector(perp, direction, azimuth)
    
    perturbed = rotate_vector(direction, axis, angle_rad)
    perturbed = perturbed / np.linalg.norm(perturbed)
    
    return perturbed
