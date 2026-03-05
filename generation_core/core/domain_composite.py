"""
CompositeDomain - Boolean operations on domains.

This module provides CompositeDomain for combining multiple domains using
boolean operations (union, intersection, difference).

Example use cases:
- Union: Combine multiple organs into a single domain
- Intersection: Find overlap region between domains
- Difference: Carve out regions from a domain
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal
import numpy as np

from .domain import DomainSpec
from .types import Point3D


@dataclass
class CompositeDomain(DomainSpec):
    """
    Composite domain using boolean operations on child domains.
    
    Parameters
    ----------
    operation : str
        Boolean operation: "union", "intersection", or "difference".
    children : list
        List of child DomainSpec objects.
        For "difference", the first child is the base and subsequent children are subtracted.
    
    Methods
    -------
    contains(point) : bool
        Check if point is inside the composite domain.
    distance_to_boundary(point) : float
        Approximate distance to boundary (uses min/max of child distances).
    sample_points(n_points, seed) : np.ndarray
        Sample points via rejection sampling from bounding box.
    
    Example
    -------
    >>> from generation.core.domain import CylinderDomain, BoxDomain
    >>> cylinder = CylinderDomain(radius=0.005, height=0.01)
    >>> box = BoxDomain.from_center_and_size((0, 0, 0), 0.003, 0.003, 0.003)
    >>> # Create cylinder with box carved out
    >>> composite = CompositeDomain("difference", [cylinder, box])
    """
    
    operation: Literal["union", "intersection", "difference"]
    children: List[DomainSpec] = field(default_factory=list)
    
    def __post_init__(self):
        if len(self.children) < 1:
            raise ValueError("CompositeDomain requires at least one child domain")
        if self.operation == "difference" and len(self.children) < 2:
            raise ValueError("Difference operation requires at least two children")
        if self.operation not in ("union", "intersection", "difference"):
            raise ValueError(f"Unknown operation '{self.operation}'. Valid: union, intersection, difference")
    
    def contains(self, point: Point3D) -> bool:
        """Check if point is inside the composite domain."""
        if self.operation == "union":
            return any(child.contains(point) for child in self.children)
        
        elif self.operation == "intersection":
            return all(child.contains(point) for child in self.children)
        
        elif self.operation == "difference":
            if not self.children[0].contains(point):
                return False
            for child in self.children[1:]:
                if child.contains(point):
                    return False
            return True
        
        return False
    
    def project_inside(self, point: Point3D) -> Point3D:
        """
        Project point to nearest point inside composite domain.
        
        For intersection: projects to the child with largest signed distance (most outside),
        then iterates until all constraints are satisfied. The result is near the boundary
        of the active constraint.
        
        For union: finds the closest projection among all children.
        
        For difference: projects to base domain, avoiding subtracted regions.
        """
        if self.contains(point):
            return point
        
        if self.operation == "union":
            best_point = None
            best_dist = float('inf')
            
            for child in self.children:
                projected = child.project_inside(point)
                if self.contains(projected):
                    dist = self._point_distance(point, projected)
                    if dist < best_dist:
                        best_dist = dist
                        best_point = projected
            
            if best_point is not None:
                return best_point
        
        elif self.operation == "intersection":
            # For intersection, we need to find a point that:
            # 1. Is inside all children (satisfies all constraints)
            # 2. Is near the boundary of the intersection (not deep inside)
            
            # First, find a point inside the intersection using iterative projection
            current = point
            max_iterations = 20
            
            for iteration in range(max_iterations):
                # Find the child with the largest signed distance (most outside)
                max_sd = float('-inf')
                active_child_idx = 0
                
                for i, child in enumerate(self.children):
                    sd = child.signed_distance(current)
                    if sd > max_sd:
                        max_sd = sd
                        active_child_idx = i
                
                # If we're inside all children, we found a valid point
                if max_sd <= 0:
                    break
                
                # Project to the active constraint (the one we're most outside of)
                current = self.children[active_child_idx].project_inside(current)
            
            # If we found a point inside, use binary search to find a point near the boundary
            if self.contains(current):
                # Binary search between original point and current to find boundary
                low, high = 0.0, 1.0
                for _ in range(30):
                    mid = (low + high) / 2
                    test_point = Point3D(
                        point.x + mid * (current.x - point.x),
                        point.y + mid * (current.y - point.y),
                        point.z + mid * (current.z - point.z),
                    )
                    if self.contains(test_point):
                        high = mid
                    else:
                        low = mid
                
                # Return a point just inside the boundary (use high which is inside)
                result = Point3D(
                    point.x + high * (current.x - point.x),
                    point.y + high * (current.y - point.y),
                    point.z + high * (current.z - point.z),
                )
                if self.contains(result):
                    return result
                else:
                    return current
        
        elif self.operation == "difference":
            projected = self.children[0].project_inside(point)
            if self.contains(projected):
                return projected
        
        # Fallback: rejection sampling
        samples = self.sample_points(100, seed=42)
        if len(samples) > 0:
            point_arr = np.array([point.x, point.y, point.z])
            distances = np.linalg.norm(samples - point_arr, axis=1)
            closest_idx = np.argmin(distances)
            return Point3D(samples[closest_idx, 0], samples[closest_idx, 1], samples[closest_idx, 2])
        
        return self.children[0].project_inside(point)
    
    def _point_distance(self, p1: Point3D, p2: Point3D) -> float:
        """Compute Euclidean distance between two points."""
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    
    def distance_to_boundary(self, point: Point3D) -> float:
        """
        Compute distance to composite domain boundary.
        
        For union: min of child distances (closest child boundary)
        For intersection: 
            - If inside: min of child distances (closest constraint boundary)
            - If outside: distance to the most violated constraint
        For difference: distance to base or subtracted boundaries
        """
        child_distances = [child.distance_to_boundary(point) for child in self.children]
        
        if self.operation == "union":
            return min(child_distances)
        
        elif self.operation == "intersection":
            # For intersection, the boundary is where ANY child's boundary is reached
            # When inside, we want the minimum distance (closest boundary)
            # When outside, we want the distance to the most violated constraint
            if self.contains(point):
                # Inside: closest boundary is the minimum distance
                return min(child_distances)
            else:
                # Outside: find the child we're most outside of
                max_sd = float('-inf')
                max_sd_idx = 0
                for i, child in enumerate(self.children):
                    sd = child.signed_distance(point)
                    if sd > max_sd:
                        max_sd = sd
                        max_sd_idx = i
                return child_distances[max_sd_idx]
        
        elif self.operation == "difference":
            base_dist = child_distances[0]
            for i, child in enumerate(self.children[1:], 1):
                if child.contains(point):
                    base_dist = min(base_dist, child_distances[i])
            return base_dist
        
        return min(child_distances)
    
    def sample_points(self, n_points: int, seed: Optional[int] = None) -> np.ndarray:
        """
        Sample random points inside composite domain via rejection sampling.
        
        Samples from the bounding box and rejects points outside the composite.
        """
        rng = np.random.default_rng(seed)
        
        bounds = self.get_bounds()
        x_min, x_max = bounds[0], bounds[1]
        y_min, y_max = bounds[2], bounds[3]
        z_min, z_max = bounds[4], bounds[5]
        
        points = []
        max_attempts = n_points * 100
        attempts = 0
        
        while len(points) < n_points and attempts < max_attempts:
            x = rng.uniform(x_min, x_max)
            y = rng.uniform(y_min, y_max)
            z = rng.uniform(z_min, z_max)
            
            point = Point3D(x, y, z)
            
            if self.contains(point):
                points.append([x, y, z])
            
            attempts += 1
        
        if len(points) < n_points:
            import warnings
            warnings.warn(
                f"CompositeDomain.sample_points: Could only sample {len(points)} points "
                f"after {max_attempts} attempts (requested {n_points})"
            )
        
        return np.array(points) if points else np.zeros((0, 3))
    
    def get_bounds(self) -> tuple:
        """Get bounding box of the composite domain."""
        if self.operation == "union":
            all_bounds = [child.get_bounds() for child in self.children]
            x_min = min(b[0] for b in all_bounds)
            x_max = max(b[1] for b in all_bounds)
            y_min = min(b[2] for b in all_bounds)
            y_max = max(b[3] for b in all_bounds)
            z_min = min(b[4] for b in all_bounds)
            z_max = max(b[5] for b in all_bounds)
            return (x_min, x_max, y_min, y_max, z_min, z_max)
        
        elif self.operation == "intersection":
            all_bounds = [child.get_bounds() for child in self.children]
            x_min = max(b[0] for b in all_bounds)
            x_max = min(b[1] for b in all_bounds)
            y_min = max(b[2] for b in all_bounds)
            y_max = min(b[3] for b in all_bounds)
            z_min = max(b[4] for b in all_bounds)
            z_max = min(b[5] for b in all_bounds)
            return (x_min, x_max, y_min, y_max, z_min, z_max)
        
        elif self.operation == "difference":
            return self.children[0].get_bounds()
        
        return self.children[0].get_bounds()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "type": "composite",
            "operation": self.operation,
            "children": [child.to_dict() for child in self.children],
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "CompositeDomain":
        """Create from dictionary."""
        from .domain import domain_from_dict
        
        children = [domain_from_dict(child_dict) for child_dict in d["children"]]
        return cls(
            operation=d["operation"],
            children=children,
        )
    
    @classmethod
    def union(cls, *domains: DomainSpec) -> "CompositeDomain":
        """Create a union of domains."""
        return cls(operation="union", children=list(domains))
    
    @classmethod
    def intersection(cls, *domains: DomainSpec) -> "CompositeDomain":
        """Create an intersection of domains."""
        return cls(operation="intersection", children=list(domains))
    
    @classmethod
    def difference(cls, base: DomainSpec, *subtracted: DomainSpec) -> "CompositeDomain":
        """Create a difference (base minus subtracted domains)."""
        return cls(operation="difference", children=[base] + list(subtracted))


__all__ = [
    "CompositeDomain",
]
