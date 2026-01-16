"""
Scale-aware tolerance utilities for vascular library.

This module provides helpers for computing scale-dependent tolerances,
replacing hardcoded epsilon values that fail at micron scales.

Problem: Any fixed eps like 0.001 m (1 mm) is disastrous in micron domains.
Solution: Use relative tolerances based on domain scale.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from typing import Union, Tuple, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from aog_policies.resolution import ResolutionPolicy


def domain_scale(domain) -> float:
    """
    Compute the characteristic scale of a domain.
    
    Returns the minimum extent of the domain's bounding box, which represents
    the smallest dimension and is used for computing scale-aware tolerances.
    
    Parameters
    ----------
    domain : Domain
        A domain object with a bounding_box property or method.
        Supported types: EllipsoidDomain, BoxDomain, CylinderDomain, MeshDomain, etc.
    
    Returns
    -------
    float
        Minimum extent of the domain in meters.
    
    Examples
    --------
    >>> from generation.core.domain import CylinderDomain
    >>> domain = CylinderDomain(radius=0.005, height=0.003)  # 5mm radius, 3mm height
    >>> domain_scale(domain)
    0.003  # Height is the smallest dimension
    
    >>> from generation.core.domain import EllipsoidDomain
    >>> domain = EllipsoidDomain(semi_axes=(0.025, 0.025, 0.005))  # 50mm x 50mm x 10mm
    >>> domain_scale(domain)
    0.01  # Z-axis (2 * 0.005) is the smallest dimension
    """
    bbox = _get_bounding_box(domain)
    if bbox is None:
        return 1e-3
    
    extents = bbox[1] - bbox[0]
    return float(np.min(extents))


def domain_extents(domain) -> Tuple[float, float, float]:
    """
    Get the extents (width, height, depth) of a domain's bounding box.
    
    Parameters
    ----------
    domain : Domain
        A domain object with a bounding_box property or method.
    
    Returns
    -------
    Tuple[float, float, float]
        (width, height, depth) extents in meters.
    """
    bbox = _get_bounding_box(domain)
    if bbox is None:
        return (1e-3, 1e-3, 1e-3)
    
    extents = bbox[1] - bbox[0]
    return (float(extents[0]), float(extents[1]), float(extents[2]))


def eps(
    domain,
    resolution_policy: Optional["ResolutionPolicy"] = None,
    rel_epsilon: float = 1e-6,
) -> float:
    """
    Compute scale-aware epsilon tolerance for a domain.
    
    This replaces hardcoded tolerances like 0.001 m (1 mm) that fail at micron scales.
    The epsilon is computed as: max(1e-12, domain_scale * rel_epsilon)
    
    Parameters
    ----------
    domain : Domain
        A domain object with a bounding_box property or method.
    resolution_policy : ResolutionPolicy, optional
        If provided, uses the policy's rel_epsilon. Otherwise uses the default.
    rel_epsilon : float, optional
        Relative epsilon factor. Default 1e-6.
        Only used if resolution_policy is None.
    
    Returns
    -------
    float
        Scale-aware epsilon tolerance in meters.
    
    Examples
    --------
    >>> from generation.core.domain import CylinderDomain
    >>> domain = CylinderDomain(radius=0.005, height=0.003)  # 3mm height
    >>> eps(domain)
    3e-9  # 0.003 * 1e-6
    
    >>> # For a 50mm domain
    >>> domain = CylinderDomain(radius=0.025, height=0.05)
    >>> eps(domain)
    5e-8  # 0.05 * 1e-6
    
    >>> # With ResolutionPolicy
    >>> from aog_policies.resolution import ResolutionPolicy
    >>> policy = ResolutionPolicy(rel_epsilon=1e-5)
    >>> eps(domain, policy)
    5e-7  # 0.05 * 1e-5
    """
    scale = domain_scale(domain)
    
    if resolution_policy is not None:
        rel_eps = resolution_policy.rel_epsilon
    else:
        rel_eps = rel_epsilon
    
    return max(1e-12, scale * rel_eps)


def containment_tolerance(
    domain,
    resolution_policy: Optional["ResolutionPolicy"] = None,
) -> float:
    """
    Compute tolerance for domain containment checks.
    
    Used when checking if a point is inside a domain, allowing for
    small numerical errors at the boundary.
    
    Parameters
    ----------
    domain : Domain
        A domain object.
    resolution_policy : ResolutionPolicy, optional
        If provided, uses the policy's rel_epsilon.
    
    Returns
    -------
    float
        Containment tolerance in meters.
    """
    return eps(domain, resolution_policy) * 10


def collision_tolerance(
    domain,
    resolution_policy: Optional["ResolutionPolicy"] = None,
) -> float:
    """
    Compute tolerance for collision clearance checks.
    
    Used when checking if two objects are too close together.
    
    Parameters
    ----------
    domain : Domain
        A domain object.
    resolution_policy : ResolutionPolicy, optional
        If provided, uses the policy's rel_epsilon.
    
    Returns
    -------
    float
        Collision tolerance in meters.
    """
    return eps(domain, resolution_policy) * 100


def snap_tolerance(
    domain,
    resolution_policy: Optional["ResolutionPolicy"] = None,
) -> float:
    """
    Compute tolerance for node snapping operations.
    
    Used when merging nearby nodes or snapping to grid.
    
    Parameters
    ----------
    domain : Domain
        A domain object.
    resolution_policy : ResolutionPolicy, optional
        If provided, uses the policy's rel_epsilon.
    
    Returns
    -------
    float
        Snap tolerance in meters.
    """
    return eps(domain, resolution_policy) * 1000


def prune_tolerance(
    domain,
    resolution_policy: Optional["ResolutionPolicy"] = None,
) -> float:
    """
    Compute tolerance for segment pruning operations.
    
    Used when removing very short segments.
    
    Parameters
    ----------
    domain : Domain
        A domain object.
    resolution_policy : ResolutionPolicy, optional
        If provided, uses the policy's rel_epsilon.
    
    Returns
    -------
    float
        Prune tolerance in meters.
    """
    return eps(domain, resolution_policy) * 1000


def boundary_mask_tolerance(
    domain,
    resolution_policy: Optional["ResolutionPolicy"] = None,
) -> float:
    """
    Compute tolerance for pathfinding boundary masks.
    
    Used when creating voxel masks near domain boundaries.
    
    Parameters
    ----------
    domain : Domain
        A domain object.
    resolution_policy : ResolutionPolicy, optional
        If provided, uses the policy's rel_epsilon.
    
    Returns
    -------
    float
        Boundary mask tolerance in meters.
    """
    return eps(domain, resolution_policy) * 100


def _get_bounding_box(domain) -> Optional[np.ndarray]:
    """
    Get the bounding box of a domain.
    
    Handles various domain types that may have different APIs.
    
    Parameters
    ----------
    domain : Domain
        A domain object.
    
    Returns
    -------
    np.ndarray or None
        Bounding box as [[min_x, min_y, min_z], [max_x, max_y, max_z]],
        or None if bounding box cannot be determined.
    """
    if hasattr(domain, "get_bounds"):
        bounds = domain.get_bounds()
        if bounds is not None:
            if len(bounds) == 6:
                return np.array([
                    [bounds[0], bounds[2], bounds[4]],
                    [bounds[1], bounds[3], bounds[5]],
                ])
            elif len(bounds) == 2:
                return np.array(bounds)
    
    if hasattr(domain, "bounding_box"):
        bbox = domain.bounding_box
        if callable(bbox):
            bbox = bbox()
        if bbox is not None:
            return np.array(bbox)
    
    if hasattr(domain, "bounds"):
        bounds = domain.bounds
        if callable(bounds):
            bounds = bounds()
        if bounds is not None:
            if hasattr(bounds, '__len__') and len(bounds) == 2:
                return np.array(bounds)
    
    if hasattr(domain, "center") and hasattr(domain, "radius"):
        center = domain.center
        if hasattr(center, 'x'):
            center = np.array([center.x, center.y, center.z])
        else:
            center = np.array(center)
        if hasattr(domain, "height"):
            r = domain.radius
            h = domain.height / 2
            return np.array([
                [center[0] - r, center[1] - r, center[2] - h],
                [center[0] + r, center[1] + r, center[2] + h],
            ])
        else:
            r = domain.radius
            return np.array([
                center - r,
                center + r,
            ])
    
    if hasattr(domain, "semi_axis_a") and hasattr(domain, "semi_axis_b") and hasattr(domain, "semi_axis_c"):
        center = getattr(domain, "center", None)
        if center is None:
            center = np.zeros(3)
        elif hasattr(center, 'x'):
            center = np.array([center.x, center.y, center.z])
        else:
            center = np.array(center)
        semi_axes = np.array([domain.semi_axis_a, domain.semi_axis_b, domain.semi_axis_c])
        return np.array([
            center - semi_axes,
            center + semi_axes,
        ])
    
    if hasattr(domain, "semi_axes"):
        center = getattr(domain, "center", np.zeros(3))
        if hasattr(center, 'x'):
            center = np.array([center.x, center.y, center.z])
        else:
            center = np.array(center)
        semi_axes = np.array(domain.semi_axes)
        return np.array([
            center - semi_axes,
            center + semi_axes,
        ])
    
    return None


__all__ = [
    "domain_scale",
    "domain_extents",
    "eps",
    "containment_tolerance",
    "collision_tolerance",
    "snap_tolerance",
    "prune_tolerance",
    "boundary_mask_tolerance",
]
