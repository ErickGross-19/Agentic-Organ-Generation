"""
ImplicitDomain - SDF/JSON AST-based domain definition.

This module provides ImplicitDomain for defining domains using signed distance
functions (SDFs) specified as JSON AST expressions.

Example use cases:
- Define complex shapes using mathematical expressions
- Combine primitives using smooth blending
- Create parametric domains with variable parameters
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
import numpy as np
import math

from .domain import DomainSpec
from .types import Point3D


def _safe_eval_ast(ast: Union[Dict, float, int, str], x: float, y: float, z: float, params: Dict[str, float]) -> float:
    """
    Safely evaluate a JSON AST expression.
    
    Supported operations:
        - Arithmetic: +, -, *, /, neg
        - Math functions: min, max, abs, sqrt, pow, sin, cos, exp, log, clamp
        - Variables: x, y, z
        - Parameters: any key from params dict
        - Constants: numeric literals
    
    Parameters
    ----------
    ast : dict or float or int or str
        The AST node to evaluate.
    x, y, z : float
        Point coordinates.
    params : dict
        Parameter values.
    
    Returns
    -------
    float
        Evaluated result.
    """
    if isinstance(ast, (int, float)):
        return float(ast)
    
    if isinstance(ast, str):
        if ast == "x":
            return x
        elif ast == "y":
            return y
        elif ast == "z":
            return z
        elif ast in params:
            return params[ast]
        else:
            raise ValueError(f"Unknown variable '{ast}'")
    
    if not isinstance(ast, dict):
        raise ValueError(f"Invalid AST node type: {type(ast)}")
    
    op = ast.get("op")
    args = ast.get("args", [])
    
    def eval_arg(arg):
        return _safe_eval_ast(arg, x, y, z, params)
    
    if op == "+":
        return sum(eval_arg(a) for a in args)
    elif op == "-":
        if len(args) == 1:
            return -eval_arg(args[0])
        return eval_arg(args[0]) - eval_arg(args[1])
    elif op == "*":
        result = 1.0
        for a in args:
            result *= eval_arg(a)
        return result
    elif op == "/":
        return eval_arg(args[0]) / eval_arg(args[1])
    elif op == "neg":
        return -eval_arg(args[0])
    elif op == "min":
        return min(eval_arg(a) for a in args)
    elif op == "max":
        return max(eval_arg(a) for a in args)
    elif op == "abs":
        return abs(eval_arg(args[0]))
    elif op == "sqrt":
        return math.sqrt(max(0, eval_arg(args[0])))
    elif op == "pow":
        return math.pow(eval_arg(args[0]), eval_arg(args[1]))
    elif op == "sin":
        return math.sin(eval_arg(args[0]))
    elif op == "cos":
        return math.cos(eval_arg(args[0]))
    elif op == "exp":
        return math.exp(eval_arg(args[0]))
    elif op == "log":
        return math.log(max(1e-10, eval_arg(args[0])))
    elif op == "clamp":
        val = eval_arg(args[0])
        lo = eval_arg(args[1])
        hi = eval_arg(args[2])
        return max(lo, min(hi, val))
    elif op == "length":
        vals = [eval_arg(a) for a in args]
        return math.sqrt(sum(v*v for v in vals))
    elif op == "length2":
        vals = [eval_arg(a) for a in args]
        return sum(v*v for v in vals)
    elif op == "dot":
        a1 = [eval_arg(a) for a in args[0]]
        a2 = [eval_arg(a) for a in args[1]]
        return sum(v1*v2 for v1, v2 in zip(a1, a2))
    else:
        raise ValueError(f"Unknown operation '{op}'")


def sphere_sdf(radius: float = 1.0, center: tuple = (0, 0, 0)) -> Dict:
    """Create AST for sphere SDF: length(p - center) - radius."""
    cx, cy, cz = center
    return {
        "op": "-",
        "args": [
            {
                "op": "length",
                "args": [
                    {"op": "-", "args": ["x", cx]},
                    {"op": "-", "args": ["y", cy]},
                    {"op": "-", "args": ["z", cz]},
                ]
            },
            radius
        ]
    }


def box_sdf(half_extents: tuple = (1, 1, 1), center: tuple = (0, 0, 0)) -> Dict:
    """Create AST for box SDF (approximate)."""
    hx, hy, hz = half_extents
    cx, cy, cz = center
    return {
        "op": "max",
        "args": [
            {"op": "-", "args": [{"op": "abs", "args": [{"op": "-", "args": ["x", cx]}]}, hx]},
            {"op": "-", "args": [{"op": "abs", "args": [{"op": "-", "args": ["y", cy]}]}, hy]},
            {"op": "-", "args": [{"op": "abs", "args": [{"op": "-", "args": ["z", cz]}]}, hz]},
        ]
    }


def cylinder_sdf(radius: float = 1.0, half_height: float = 1.0, center: tuple = (0, 0, 0)) -> Dict:
    """Create AST for Z-aligned cylinder SDF (approximate)."""
    cx, cy, cz = center
    return {
        "op": "max",
        "args": [
            {
                "op": "-",
                "args": [
                    {
                        "op": "length",
                        "args": [
                            {"op": "-", "args": ["x", cx]},
                            {"op": "-", "args": ["y", cy]},
                        ]
                    },
                    radius
                ]
            },
            {"op": "-", "args": [{"op": "abs", "args": [{"op": "-", "args": ["z", cz]}]}, half_height]},
        ]
    }


@dataclass
class ImplicitDomain(DomainSpec):
    """
    Domain defined by a signed distance function (SDF) as JSON AST.
    
    The SDF should return negative values inside the domain, zero on the
    boundary, and positive values outside.
    
    Parameters
    ----------
    bbox : tuple
        Bounding box as (x_min, x_max, y_min, y_max, z_min, z_max).
    sdf_ast : dict
        JSON AST defining the signed distance function.
    params : dict, optional
        Parameter values for the SDF expression.
    
    AST Format
    ----------
    The AST is a nested dictionary with "op" and "args" keys:
    
    - Arithmetic: {"op": "+", "args": [a, b, ...]}
    - Variables: "x", "y", "z"
    - Parameters: any string key from params
    - Constants: numeric literals
    
    Supported operations:
        +, -, *, /, neg, min, max, abs, sqrt, pow, sin, cos, exp, log, clamp, length
    
    Example
    -------
    >>> # Sphere SDF: length(p) - radius
    >>> sdf = {
    ...     "op": "-",
    ...     "args": [
    ...         {"op": "length", "args": ["x", "y", "z"]},
    ...         "radius"
    ...     ]
    ... }
    >>> domain = ImplicitDomain(
    ...     bbox=(-1, 1, -1, 1, -1, 1),
    ...     sdf_ast=sdf,
    ...     params={"radius": 0.5}
    ... )
    """
    
    bbox: tuple
    sdf_ast: Dict[str, Any]
    params: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if len(self.bbox) != 6:
            raise ValueError(f"bbox must have 6 elements, got {len(self.bbox)}")
        if self.bbox[0] >= self.bbox[1]:
            raise ValueError(f"bbox x_min ({self.bbox[0]}) must be less than x_max ({self.bbox[1]})")
        if self.bbox[2] >= self.bbox[3]:
            raise ValueError(f"bbox y_min ({self.bbox[2]}) must be less than y_max ({self.bbox[3]})")
        if self.bbox[4] >= self.bbox[5]:
            raise ValueError(f"bbox z_min ({self.bbox[4]}) must be less than z_max ({self.bbox[5]})")
    
    def sdf(self, point: Point3D) -> float:
        """Evaluate the signed distance function at a point."""
        return _safe_eval_ast(self.sdf_ast, point.x, point.y, point.z, self.params)
    
    def sdf_array(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the SDF at multiple points."""
        return np.array([
            _safe_eval_ast(self.sdf_ast, p[0], p[1], p[2], self.params)
            for p in points
        ])
    
    def contains(self, point: Point3D) -> bool:
        """Check if point is inside the domain (SDF <= 0)."""
        return self.sdf(point) <= 0
    
    def distance_to_boundary(self, point: Point3D) -> float:
        """Compute distance to boundary (absolute SDF value)."""
        return abs(self.sdf(point))
    
    def project_inside(self, point: Point3D) -> Point3D:
        """
        Project point to nearest point inside domain.
        
        Uses gradient descent to find the nearest point with SDF <= 0.
        """
        if self.contains(point):
            return point
        
        p = np.array([point.x, point.y, point.z])
        
        eps = 1e-6
        step_size = 0.1
        max_iterations = 100
        
        for _ in range(max_iterations):
            sdf_val = _safe_eval_ast(self.sdf_ast, p[0], p[1], p[2], self.params)
            
            if sdf_val <= 0:
                return Point3D(p[0], p[1], p[2])
            
            grad = np.zeros(3)
            for i in range(3):
                p_plus = p.copy()
                p_plus[i] += eps
                p_minus = p.copy()
                p_minus[i] -= eps
                
                sdf_plus = _safe_eval_ast(self.sdf_ast, p_plus[0], p_plus[1], p_plus[2], self.params)
                sdf_minus = _safe_eval_ast(self.sdf_ast, p_minus[0], p_minus[1], p_minus[2], self.params)
                
                grad[i] = (sdf_plus - sdf_minus) / (2 * eps)
            
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-10:
                break
            
            grad = grad / grad_norm
            
            p = p - step_size * sdf_val * grad
            
            p[0] = np.clip(p[0], self.bbox[0], self.bbox[1])
            p[1] = np.clip(p[1], self.bbox[2], self.bbox[3])
            p[2] = np.clip(p[2], self.bbox[4], self.bbox[5])
        
        return Point3D(p[0], p[1], p[2])
    
    def sample_points(self, n_points: int, seed: Optional[int] = None) -> np.ndarray:
        """Sample random points inside domain via rejection sampling."""
        rng = np.random.default_rng(seed)
        
        x_min, x_max = self.bbox[0], self.bbox[1]
        y_min, y_max = self.bbox[2], self.bbox[3]
        z_min, z_max = self.bbox[4], self.bbox[5]
        
        points = []
        max_attempts = n_points * 100
        attempts = 0
        
        while len(points) < n_points and attempts < max_attempts:
            x = rng.uniform(x_min, x_max)
            y = rng.uniform(y_min, y_max)
            z = rng.uniform(z_min, z_max)
            
            sdf_val = _safe_eval_ast(self.sdf_ast, x, y, z, self.params)
            
            if sdf_val <= 0:
                points.append([x, y, z])
            
            attempts += 1
        
        if len(points) < n_points:
            import warnings
            warnings.warn(
                f"ImplicitDomain.sample_points: Could only sample {len(points)} points "
                f"after {max_attempts} attempts (requested {n_points})"
            )
        
        return np.array(points) if points else np.zeros((0, 3))
    
    def get_bounds(self) -> tuple:
        """Get bounding box."""
        return self.bbox
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "type": "implicit",
            "bbox": list(self.bbox),
            "sdf_ast": self.sdf_ast,
            "params": self.params,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "ImplicitDomain":
        """Create from dictionary."""
        return cls(
            bbox=tuple(d["bbox"]),
            sdf_ast=d["sdf_ast"],
            params=d.get("params", {}),
        )
    
    @classmethod
    def sphere(cls, radius: float, center: tuple = (0, 0, 0)) -> "ImplicitDomain":
        """Create a sphere domain."""
        cx, cy, cz = center
        bbox = (
            cx - radius * 1.1, cx + radius * 1.1,
            cy - radius * 1.1, cy + radius * 1.1,
            cz - radius * 1.1, cz + radius * 1.1,
        )
        return cls(
            bbox=bbox,
            sdf_ast=sphere_sdf(radius, center),
            params={},
        )
    
    @classmethod
    def box(cls, half_extents: tuple, center: tuple = (0, 0, 0)) -> "ImplicitDomain":
        """Create a box domain."""
        hx, hy, hz = half_extents
        cx, cy, cz = center
        bbox = (
            cx - hx * 1.1, cx + hx * 1.1,
            cy - hy * 1.1, cy + hy * 1.1,
            cz - hz * 1.1, cz + hz * 1.1,
        )
        return cls(
            bbox=bbox,
            sdf_ast=box_sdf(half_extents, center),
            params={},
        )
    
    @classmethod
    def cylinder(cls, radius: float, half_height: float, center: tuple = (0, 0, 0)) -> "ImplicitDomain":
        """Create a Z-aligned cylinder domain."""
        cx, cy, cz = center
        bbox = (
            cx - radius * 1.1, cx + radius * 1.1,
            cy - radius * 1.1, cy + radius * 1.1,
            cz - half_height * 1.1, cz + half_height * 1.1,
        )
        return cls(
            bbox=bbox,
            sdf_ast=cylinder_sdf(radius, half_height, center),
            params={},
        )


__all__ = [
    "ImplicitDomain",
    "sphere_sdf",
    "box_sdf",
    "cylinder_sdf",
    "_safe_eval_ast",
]
