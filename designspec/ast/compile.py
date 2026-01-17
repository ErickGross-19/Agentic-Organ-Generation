"""
AST compiler for aog_ast_v1 format.

This module compiles validated AST nodes into callable evaluators
that can be used for SDF evaluation, taper schedules, etc.

SAFETY
------
The compiler only supports safe numeric operations. No dynamic imports,
no eval(), no exec(). All operations are deterministic.
"""

from typing import Dict, Any, Callable, Optional, Set
import math

from .nodes import (
    ASTNode,
    LiteralNode,
    VariableNode,
    BinaryOpNode,
    UnaryOpNode,
    validate_ast,
    parse_ast,
    ASTValidationError,
)


class ASTCompilationError(Exception):
    """Raised when AST compilation fails."""
    pass


BINARY_OP_FUNCS: Dict[str, Callable[[float, float], float]] = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: a / b if b != 0 else float("inf"),
    "min": lambda a, b: min(a, b),
    "max": lambda a, b: max(a, b),
    "<": lambda a, b: float(a < b),
    "<=": lambda a, b: float(a <= b),
    ">": lambda a, b: float(a > b),
    ">=": lambda a, b: float(a >= b),
    "==": lambda a, b: float(a == b),
    "!=": lambda a, b: float(a != b),
    "and": lambda a, b: float(bool(a) and bool(b)),
    "or": lambda a, b: float(bool(a) or bool(b)),
}

UNARY_OP_FUNCS: Dict[str, Callable[[float], float]] = {
    "neg": lambda a: -a,
    "abs": lambda a: abs(a),
    "sqrt": lambda a: math.sqrt(max(0, a)),
    "sin": lambda a: math.sin(a),
    "cos": lambda a: math.cos(a),
    "tan": lambda a: math.tan(a),
    "exp": lambda a: math.exp(min(a, 700)),
    "log": lambda a: math.log(max(a, 1e-300)),
    "not": lambda a: float(not bool(a)),
}


def _compile_node(node: ASTNode) -> Callable[[Dict[str, float]], float]:
    """
    Compile an AST node into a callable evaluator.
    
    Parameters
    ----------
    node : ASTNode
        The node to compile
        
    Returns
    -------
    callable
        A function that takes a dict of variable values and returns a float
    """
    if isinstance(node, LiteralNode):
        value = node.value
        return lambda env: value
    
    elif isinstance(node, VariableNode):
        name = node.name
        return lambda env: env.get(name, 0.0)
    
    elif isinstance(node, BinaryOpNode):
        op_func = BINARY_OP_FUNCS.get(node.op)
        if op_func is None:
            raise ASTCompilationError(f"Unknown binary op: {node.op}")
        
        left_fn = _compile_node(node.left)
        right_fn = _compile_node(node.right)
        
        return lambda env: op_func(left_fn(env), right_fn(env))
    
    elif isinstance(node, UnaryOpNode):
        op_func = UNARY_OP_FUNCS.get(node.op)
        if op_func is None:
            raise ASTCompilationError(f"Unknown unary op: {node.op}")
        
        arg_fn = _compile_node(node.arg)
        
        return lambda env: op_func(arg_fn(env))
    
    else:
        raise ASTCompilationError(f"Unknown node type: {type(node).__name__}")


def compile_ast(
    ast: Dict[str, Any],
    allowed_vars: Optional[Set[str]] = None,
) -> Callable[[Dict[str, float]], float]:
    """
    Compile an AST dict into a callable evaluator.
    
    Parameters
    ----------
    ast : dict
        The AST dict with "format" and "root" keys
    allowed_vars : set of str, optional
        Set of allowed variable names for validation
        
    Returns
    -------
    callable
        A function that takes a dict of variable values and returns a float
        
    Raises
    ------
    ASTValidationError
        If AST validation fails
    ASTCompilationError
        If compilation fails
        
    Example
    -------
    >>> ast = {
    ...     "format": "aog_ast_v1",
    ...     "root": {
    ...         "type": "binop",
    ...         "op": "-",
    ...         "left": {
    ...             "type": "unop",
    ...             "op": "sqrt",
    ...             "arg": {
    ...                 "type": "binop",
    ...                 "op": "+",
    ...                 "left": {
    ...                     "type": "binop",
    ...                     "op": "*",
    ...                     "left": {"type": "var", "name": "x"},
    ...                     "right": {"type": "var", "name": "x"}
    ...                 },
    ...                 "right": {
    ...                     "type": "binop",
    ...                     "op": "+",
    ...                     "left": {
    ...                         "type": "binop",
    ...                         "op": "*",
    ...                         "left": {"type": "var", "name": "y"},
    ...                         "right": {"type": "var", "name": "y"}
    ...                     },
    ...                     "right": {
    ...                         "type": "binop",
    ...                         "op": "*",
    ...                         "left": {"type": "var", "name": "z"},
    ...                         "right": {"type": "var", "name": "z"}
    ...                     }
    ...                 }
    ...             }
    ...         },
    ...         "right": {"type": "literal", "value": 1.0}
    ...     }
    ... }
    >>> sdf = compile_ast(ast)
    >>> sdf({"x": 0, "y": 0, "z": 0})  # Center of sphere
    -1.0
    >>> sdf({"x": 1, "y": 0, "z": 0})  # On surface
    0.0
    >>> sdf({"x": 2, "y": 0, "z": 0})  # Outside
    1.0
    """
    errors = validate_ast(ast, allowed_vars)
    if errors:
        raise ASTValidationError(f"AST validation failed: {'; '.join(errors)}")
    
    root_node = parse_ast(ast)
    
    return _compile_node(root_node)


def make_sphere_sdf_ast(radius: float = 1.0) -> Dict[str, Any]:
    """
    Create an AST for a sphere SDF centered at origin.
    
    SDF(x, y, z) = sqrt(x^2 + y^2 + z^2) - radius
    
    Parameters
    ----------
    radius : float
        Sphere radius
        
    Returns
    -------
    dict
        AST dict for the sphere SDF
    """
    return {
        "format": "aog_ast_v1",
        "root": {
            "type": "binop",
            "op": "-",
            "left": {
                "type": "unop",
                "op": "sqrt",
                "arg": {
                    "type": "binop",
                    "op": "+",
                    "left": {
                        "type": "binop",
                        "op": "*",
                        "left": {"type": "var", "name": "x"},
                        "right": {"type": "var", "name": "x"},
                    },
                    "right": {
                        "type": "binop",
                        "op": "+",
                        "left": {
                            "type": "binop",
                            "op": "*",
                            "left": {"type": "var", "name": "y"},
                            "right": {"type": "var", "name": "y"},
                        },
                        "right": {
                            "type": "binop",
                            "op": "*",
                            "left": {"type": "var", "name": "z"},
                            "right": {"type": "var", "name": "z"},
                        },
                    },
                },
            },
            "right": {"type": "literal", "value": radius},
        },
    }


def make_box_sdf_ast(half_extents: tuple = (1.0, 1.0, 1.0)) -> Dict[str, Any]:
    """
    Create an AST for a box SDF centered at origin.
    
    This is a simplified version that computes the distance to the
    nearest face (not exact SDF for corners/edges).
    
    Parameters
    ----------
    half_extents : tuple
        Half-extents (hx, hy, hz) of the box
        
    Returns
    -------
    dict
        AST dict for the box SDF
    """
    hx, hy, hz = half_extents
    
    def make_axis_dist(var_name: str, half_extent: float) -> Dict[str, Any]:
        return {
            "type": "binop",
            "op": "-",
            "left": {
                "type": "unop",
                "op": "abs",
                "arg": {"type": "var", "name": var_name},
            },
            "right": {"type": "literal", "value": half_extent},
        }
    
    return {
        "format": "aog_ast_v1",
        "root": {
            "type": "binop",
            "op": "max",
            "left": {
                "type": "binop",
                "op": "max",
                "left": make_axis_dist("x", hx),
                "right": make_axis_dist("y", hy),
            },
            "right": make_axis_dist("z", hz),
        },
    }


__all__ = [
    "compile_ast",
    "ASTCompilationError",
    "make_sphere_sdf_ast",
    "make_box_sdf_ast",
]
