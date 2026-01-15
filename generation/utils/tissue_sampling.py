"""
Tissue/attractor point sampling for space colonization.

This module provides functions for sampling tissue points within domains
using various distribution strategies for guiding vascular network growth.

UNIT CONVENTIONS
----------------
All geometric values are in METERS internally.
"""

from typing import Optional, Tuple, Dict, Any, List, TYPE_CHECKING
import numpy as np
import logging

from aog_policies import TissueSamplingPolicy, OperationReport
from ..core.types import Point3D

if TYPE_CHECKING:
    from ..core.domain import DomainSpec

logger = logging.getLogger(__name__)


def sample_tissue_points(
    domain: "DomainSpec",
    ports: Optional[Dict[str, Any]] = None,
    policy: Optional[TissueSamplingPolicy] = None,
    seed: Optional[int] = None,
) -> Tuple[List[Point3D], OperationReport]:
    """
    Sample tissue/attractor points within a domain.
    
    Parameters
    ----------
    domain : DomainSpec
        Domain specification defining the sampling region
    ports : dict, optional
        Port configuration with "inlets" and "outlets" for exclusion zones
    policy : TissueSamplingPolicy, optional
        Policy controlling sampling strategy and parameters
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    points : List[Point3D]
        Sampled tissue points
    report : OperationReport
        Report with sampling statistics and metadata
    """
    if policy is None:
        policy = TissueSamplingPolicy()
    
    if not policy.enabled:
        return [], OperationReport(
            operation="sample_tissue_points",
            success=True,
            requested_policy=policy.to_dict(),
            effective_policy=policy.to_dict(),
            metadata={"n_points": 0, "reason": "sampling disabled"},
        )
    
    # Set random seed
    if seed is not None:
        np.random.seed(seed)
    elif policy.seed is not None:
        np.random.seed(policy.seed)
    
    # Get port positions for exclusion
    port_positions = _get_port_positions(ports)
    
    # Sample based on strategy
    strategy = policy.strategy
    
    if strategy == "uniform":
        points, meta = _sample_uniform(domain, policy, port_positions)
    elif strategy == "depth_biased":
        points, meta = _sample_depth_biased(domain, policy, port_positions)
    elif strategy == "radial_biased":
        points, meta = _sample_radial_biased(domain, policy, port_positions)
    elif strategy == "boundary_shell":
        points, meta = _sample_boundary_shell(domain, policy, port_positions)
    elif strategy == "gaussian":
        points, meta = _sample_gaussian(domain, policy, port_positions)
    elif strategy == "mixture":
        points, meta = _sample_mixture(domain, policy, port_positions)
    else:
        logger.warning(f"Unknown sampling strategy '{strategy}', falling back to uniform")
        points, meta = _sample_uniform(domain, policy, port_positions)
    
    report = OperationReport(
        operation="sample_tissue_points",
        success=True,
        requested_policy=policy.to_dict(),
        effective_policy=policy.to_dict(),
        metadata=meta,
    )
    
    return points, report


def _get_port_positions(ports: Optional[Dict[str, Any]]) -> List[np.ndarray]:
    """Extract port positions from port configuration."""
    if ports is None:
        return []
    
    positions = []
    for inlet in ports.get("inlets", []):
        pos = inlet.get("position", (0, 0, 0))
        positions.append(np.array(pos))
    for outlet in ports.get("outlets", []):
        pos = outlet.get("position", (0, 0, 0))
        positions.append(np.array(pos))
    
    return positions


def _is_valid_point(
    point: np.ndarray,
    domain: "DomainSpec",
    port_positions: List[np.ndarray],
    min_distance_to_ports: float,
    exclude_spheres: List[Dict[str, Any]],
) -> bool:
    """Check if a point is valid (inside domain and not in exclusion zones)."""
    # Check domain containment
    if hasattr(domain, 'contains'):
        if not domain.contains(Point3D(*point)):
            return False
    
    # Check port exclusion
    for port_pos in port_positions:
        if np.linalg.norm(point - port_pos) < min_distance_to_ports:
            return False
    
    # Check exclusion spheres
    for sphere in exclude_spheres:
        center = np.array(sphere.get("center", (0, 0, 0)))
        radius = sphere.get("radius", 0)
        if np.linalg.norm(point - center) < radius:
            return False
    
    return True


def _sample_uniform(
    domain: "DomainSpec",
    policy: TissueSamplingPolicy,
    port_positions: List[np.ndarray],
) -> Tuple[List[Point3D], Dict[str, Any]]:
    """Sample points uniformly within the domain."""
    n_points = policy.n_points
    min_dist = policy.min_distance_to_ports
    exclude_spheres = policy.exclude_spheres
    
    # Get domain bounds
    bounds = _get_domain_bounds(domain)
    
    points = []
    attempts = 0
    max_attempts = n_points * 10
    
    while len(points) < n_points and attempts < max_attempts:
        # Sample random point in bounding box
        point = np.array([
            np.random.uniform(bounds[0], bounds[1]),
            np.random.uniform(bounds[2], bounds[3]),
            np.random.uniform(bounds[4], bounds[5]),
        ])
        
        if _is_valid_point(point, domain, port_positions, min_dist, exclude_spheres):
            points.append(Point3D(*point))
        
        attempts += 1
    
    acceptance_rate = len(points) / max(attempts, 1)
    
    return points, {
        "n_points": len(points),
        "n_requested": n_points,
        "attempts": attempts,
        "acceptance_rate": acceptance_rate,
        "strategy": "uniform",
    }


def _sample_depth_biased(
    domain: "DomainSpec",
    policy: TissueSamplingPolicy,
    port_positions: List[np.ndarray],
) -> Tuple[List[Point3D], Dict[str, Any]]:
    """
    Sample points with depth bias (more points deeper into domain).
    
    Depth is measured from the reference face inward along the face normal.
    Default: "more points deeper" helps fill volume and reduces overcrowding near inlet.
    """
    n_points = policy.n_points
    min_dist = policy.min_distance_to_ports
    exclude_spheres = policy.exclude_spheres
    
    # Get domain bounds and face info
    bounds = _get_domain_bounds(domain)
    face = policy.depth_reference.get("face", "top")
    
    # Determine depth axis and direction based on face
    depth_axis, depth_sign, depth_origin = _get_depth_params(bounds, face)
    
    # Calculate depth range
    if depth_axis == 0:
        depth_range = bounds[1] - bounds[0]
    elif depth_axis == 1:
        depth_range = bounds[3] - bounds[2]
    else:
        depth_range = bounds[5] - bounds[4]
    
    depth_min = policy.depth_min
    depth_max = policy.depth_max if policy.depth_max is not None else depth_range
    
    points = []
    depths = []
    attempts = 0
    max_attempts = n_points * 10
    
    while len(points) < n_points and attempts < max_attempts:
        # Sample depth based on distribution
        depth = _sample_depth_value(policy, depth_min, depth_max)
        
        # Sample random point in the plane perpendicular to depth axis
        point = _sample_point_at_depth(bounds, depth_axis, depth_sign, depth_origin, depth)
        
        if _is_valid_point(point, domain, port_positions, min_dist, exclude_spheres):
            points.append(Point3D(*point))
            depths.append(depth)
        
        attempts += 1
    
    acceptance_rate = len(points) / max(attempts, 1)
    
    return points, {
        "n_points": len(points),
        "n_requested": n_points,
        "attempts": attempts,
        "acceptance_rate": acceptance_rate,
        "strategy": "depth_biased",
        "depth_stats": {
            "mean": float(np.mean(depths)) if depths else 0,
            "min": float(np.min(depths)) if depths else 0,
            "max": float(np.max(depths)) if depths else 0,
        },
    }


def _sample_depth_value(policy: TissueSamplingPolicy, depth_min: float, depth_max: float) -> float:
    """Sample a depth value based on the distribution type."""
    distribution = policy.depth_distribution
    
    # Normalize to [0, 1] then scale
    if distribution == "linear":
        # Linear: uniform distribution
        t = np.random.uniform(0, 1)
    elif distribution == "power":
        # Power: more points deeper (t^p where p > 1)
        p = policy.depth_power
        t = np.random.uniform(0, 1) ** (1.0 / p)  # Inverse CDF
    elif distribution == "exponential":
        # Exponential: more points deeper
        lam = policy.depth_lambda
        t = 1 - np.exp(-lam * np.random.uniform(0, 1))
    elif distribution == "beta":
        # Beta distribution for flexible control
        alpha = policy.depth_alpha
        beta = policy.depth_beta
        t = np.random.beta(alpha, beta)
    else:
        t = np.random.uniform(0, 1)
    
    return depth_min + t * (depth_max - depth_min)


def _get_depth_params(bounds: Tuple, face: str) -> Tuple[int, int, float]:
    """Get depth axis, direction sign, and origin based on face."""
    if face in ("top", "+z"):
        return 2, -1, bounds[5]  # Z axis, going down, from top
    elif face in ("bottom", "-z"):
        return 2, 1, bounds[4]  # Z axis, going up, from bottom
    elif face == "+x":
        return 0, -1, bounds[1]  # X axis, going negative, from +x face
    elif face == "-x":
        return 0, 1, bounds[0]  # X axis, going positive, from -x face
    elif face == "+y":
        return 1, -1, bounds[3]  # Y axis, going negative, from +y face
    elif face == "-y":
        return 1, 1, bounds[2]  # Y axis, going positive, from -y face
    else:
        return 2, -1, bounds[5]  # Default: top face


def _sample_point_at_depth(
    bounds: Tuple,
    depth_axis: int,
    depth_sign: int,
    depth_origin: float,
    depth: float,
) -> np.ndarray:
    """Sample a random point at a given depth."""
    point = np.array([
        np.random.uniform(bounds[0], bounds[1]),
        np.random.uniform(bounds[2], bounds[3]),
        np.random.uniform(bounds[4], bounds[5]),
    ])
    
    # Set the depth coordinate
    point[depth_axis] = depth_origin + depth_sign * depth
    
    return point


def _sample_radial_biased(
    domain: "DomainSpec",
    policy: TissueSamplingPolicy,
    port_positions: List[np.ndarray],
) -> Tuple[List[Point3D], Dict[str, Any]]:
    """Sample points with radial bias from face center."""
    n_points = policy.n_points
    min_dist = policy.min_distance_to_ports
    exclude_spheres = policy.exclude_spheres
    
    bounds = _get_domain_bounds(domain)
    
    # Get face center
    face = policy.radial_reference.get("face", "top")
    face_center = _get_face_center(bounds, face)
    
    # Get radial range
    r_max = policy.r_max
    if r_max is None:
        # Estimate from domain bounds
        r_max = min(bounds[1] - bounds[0], bounds[3] - bounds[2]) / 2
    r_min = policy.r_min
    
    points = []
    radii = []
    attempts = 0
    max_attempts = n_points * 10
    
    while len(points) < n_points and attempts < max_attempts:
        # Sample radius based on distribution
        r = _sample_radial_value(policy, r_min, r_max)
        
        # Sample angle
        theta = np.random.uniform(0, 2 * np.pi)
        
        # Sample depth uniformly
        depth_axis = _get_depth_params(bounds, face)[0]
        if depth_axis == 2:
            z = np.random.uniform(bounds[4], bounds[5])
            point = np.array([
                face_center[0] + r * np.cos(theta),
                face_center[1] + r * np.sin(theta),
                z,
            ])
        elif depth_axis == 0:
            x = np.random.uniform(bounds[0], bounds[1])
            point = np.array([
                x,
                face_center[1] + r * np.cos(theta),
                face_center[2] + r * np.sin(theta),
            ])
        else:
            y = np.random.uniform(bounds[2], bounds[3])
            point = np.array([
                face_center[0] + r * np.cos(theta),
                y,
                face_center[2] + r * np.sin(theta),
            ])
        
        if _is_valid_point(point, domain, port_positions, min_dist, exclude_spheres):
            points.append(Point3D(*point))
            radii.append(r)
        
        attempts += 1
    
    acceptance_rate = len(points) / max(attempts, 1)
    
    return points, {
        "n_points": len(points),
        "n_requested": n_points,
        "attempts": attempts,
        "acceptance_rate": acceptance_rate,
        "strategy": "radial_biased",
        "radial_stats": {
            "mean": float(np.mean(radii)) if radii else 0,
            "min": float(np.min(radii)) if radii else 0,
            "max": float(np.max(radii)) if radii else 0,
        },
    }


def _sample_radial_value(policy: TissueSamplingPolicy, r_min: float, r_max: float) -> float:
    """Sample a radial value based on the distribution type."""
    distribution = policy.radial_distribution
    
    if distribution == "center_heavy":
        # More points near center
        p = policy.radial_power
        t = np.random.uniform(0, 1) ** p
    elif distribution == "edge_heavy":
        # More points near edge
        p = policy.radial_power
        t = 1 - (1 - np.random.uniform(0, 1)) ** p
    elif distribution == "ring":
        # Gaussian ring at r0
        r0 = policy.ring_r0
        sigma = policy.ring_sigma
        r = np.random.normal(r0, sigma)
        return np.clip(r, r_min, r_max)
    else:
        t = np.random.uniform(0, 1)
    
    return r_min + t * (r_max - r_min)


def _get_face_center(bounds: Tuple, face: str) -> np.ndarray:
    """Get the center point of a face."""
    cx = (bounds[0] + bounds[1]) / 2
    cy = (bounds[2] + bounds[3]) / 2
    cz = (bounds[4] + bounds[5]) / 2
    
    if face in ("top", "+z"):
        return np.array([cx, cy, bounds[5]])
    elif face in ("bottom", "-z"):
        return np.array([cx, cy, bounds[4]])
    elif face == "+x":
        return np.array([bounds[1], cy, cz])
    elif face == "-x":
        return np.array([bounds[0], cy, cz])
    elif face == "+y":
        return np.array([cx, bounds[3], cz])
    elif face == "-y":
        return np.array([cx, bounds[2], cz])
    else:
        return np.array([cx, cy, bounds[5]])


def _sample_boundary_shell(
    domain: "DomainSpec",
    policy: TissueSamplingPolicy,
    port_positions: List[np.ndarray],
) -> Tuple[List[Point3D], Dict[str, Any]]:
    """Sample points near domain boundary."""
    n_points = policy.n_points
    min_dist = policy.min_distance_to_ports
    exclude_spheres = policy.exclude_spheres
    shell_thickness = policy.shell_thickness
    shell_mode = policy.shell_mode
    
    bounds = _get_domain_bounds(domain)
    
    points = []
    distances = []
    attempts = 0
    max_attempts = n_points * 10
    
    while len(points) < n_points and attempts < max_attempts:
        # Sample random point in bounding box
        point = np.array([
            np.random.uniform(bounds[0], bounds[1]),
            np.random.uniform(bounds[2], bounds[3]),
            np.random.uniform(bounds[4], bounds[5]),
        ])
        
        # Check distance to boundary
        if hasattr(domain, 'distance_to_boundary'):
            dist = domain.distance_to_boundary(Point3D(*point))
        else:
            # Estimate from bounding box
            dist = min(
                point[0] - bounds[0], bounds[1] - point[0],
                point[1] - bounds[2], bounds[3] - point[1],
                point[2] - bounds[4], bounds[5] - point[2],
            )
        
        # Check shell condition
        if shell_mode == "near_boundary":
            in_shell = dist < shell_thickness
        else:  # near_center
            max_dist = min(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]) / 2
            in_shell = dist > (max_dist - shell_thickness)
        
        if in_shell and _is_valid_point(point, domain, port_positions, min_dist, exclude_spheres):
            points.append(Point3D(*point))
            distances.append(dist)
        
        attempts += 1
    
    acceptance_rate = len(points) / max(attempts, 1)
    
    return points, {
        "n_points": len(points),
        "n_requested": n_points,
        "attempts": attempts,
        "acceptance_rate": acceptance_rate,
        "strategy": "boundary_shell",
        "distance_stats": {
            "mean": float(np.mean(distances)) if distances else 0,
            "min": float(np.min(distances)) if distances else 0,
            "max": float(np.max(distances)) if distances else 0,
        },
    }


def _sample_gaussian(
    domain: "DomainSpec",
    policy: TissueSamplingPolicy,
    port_positions: List[np.ndarray],
) -> Tuple[List[Point3D], Dict[str, Any]]:
    """Sample points from a Gaussian distribution."""
    n_points = policy.n_points
    min_dist = policy.min_distance_to_ports
    exclude_spheres = policy.exclude_spheres
    
    mean = np.array(policy.gaussian_mean)
    sigma = np.array(policy.gaussian_sigma)
    
    points = []
    attempts = 0
    max_attempts = n_points * 10
    
    while len(points) < n_points and attempts < max_attempts:
        # Sample from Gaussian
        point = np.random.normal(mean, sigma)
        
        if _is_valid_point(point, domain, port_positions, min_dist, exclude_spheres):
            points.append(Point3D(*point))
        
        attempts += 1
    
    acceptance_rate = len(points) / max(attempts, 1)
    
    return points, {
        "n_points": len(points),
        "n_requested": n_points,
        "attempts": attempts,
        "acceptance_rate": acceptance_rate,
        "strategy": "gaussian",
    }


def _sample_mixture(
    domain: "DomainSpec",
    policy: TissueSamplingPolicy,
    port_positions: List[np.ndarray],
) -> Tuple[List[Point3D], Dict[str, Any]]:
    """Sample points from a mixture of distributions."""
    components = policy.mixture_components
    
    if not components:
        # Fall back to uniform
        return _sample_uniform(domain, policy, port_positions)
    
    # Normalize weights
    total_weight = sum(c.get("weight", 1.0) for c in components)
    
    all_points = []
    component_stats = []
    
    for comp in components:
        weight = comp.get("weight", 1.0) / total_weight
        n_comp = int(policy.n_points * weight)
        
        # Create sub-policy from component
        sub_policy_dict = comp.get("policy", {})
        sub_policy = TissueSamplingPolicy(
            n_points=n_comp,
            strategy=sub_policy_dict.get("strategy", "uniform"),
            **{k: v for k, v in sub_policy_dict.items() if k != "strategy" and k != "n_points"}
        )
        
        # Sample from component
        if sub_policy.strategy == "uniform":
            points, meta = _sample_uniform(domain, sub_policy, port_positions)
        elif sub_policy.strategy == "depth_biased":
            points, meta = _sample_depth_biased(domain, sub_policy, port_positions)
        elif sub_policy.strategy == "radial_biased":
            points, meta = _sample_radial_biased(domain, sub_policy, port_positions)
        elif sub_policy.strategy == "boundary_shell":
            points, meta = _sample_boundary_shell(domain, sub_policy, port_positions)
        elif sub_policy.strategy == "gaussian":
            points, meta = _sample_gaussian(domain, sub_policy, port_positions)
        else:
            points, meta = _sample_uniform(domain, sub_policy, port_positions)
        
        all_points.extend(points)
        component_stats.append({
            "weight": weight,
            "strategy": sub_policy.strategy,
            "n_points": len(points),
        })
    
    return all_points, {
        "n_points": len(all_points),
        "n_requested": policy.n_points,
        "strategy": "mixture",
        "components": component_stats,
    }


def _get_domain_bounds(domain: "DomainSpec") -> Tuple[float, float, float, float, float, float]:
    """Get bounding box of domain as (x_min, x_max, y_min, y_max, z_min, z_max)."""
    if hasattr(domain, 'get_bounds'):
        return domain.get_bounds()
    
    # Try to infer from domain attributes
    if hasattr(domain, 'x_min'):
        return (domain.x_min, domain.x_max, domain.y_min, domain.y_max, domain.z_min, domain.z_max)
    
    if hasattr(domain, 'center') and hasattr(domain, 'radius'):
        # Cylinder or sphere
        c = domain.center
        r = domain.radius
        h = getattr(domain, 'height', r * 2)
        return (c.x - r, c.x + r, c.y - r, c.y + r, c.z - h/2, c.z + h/2)
    
    # Default bounds
    return (-0.01, 0.01, -0.01, 0.01, -0.01, 0.01)


__all__ = [
    "sample_tissue_points",
]
