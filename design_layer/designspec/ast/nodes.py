"""
AST node definitions and validation for aog_ast_v1 format.

This module defines the node types and validation logic for the
JSON AST format used in DesignSpec for mathematical expressions.

AST FORMAT (aog_ast_v1)
-----------------------
Each node is a dict with a "type" key and type-specific fields:

Literal: {"type": "literal", "value": <number>}
Variable: {"type": "var", "name": <string>}
BinaryOp: {"type": "binop", "op": <string>, "left": <node>, "right": <node>}
UnaryOp: {"type": "unop", "op": <string>, "arg": <node>}

Supported binary operations: +, -, *, /, min, max, <, <=, >, >=, ==, !=, and, or
Supported unary operations: neg, abs, sqrt, sin, cos, exp, log, not
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union, Set
import math


class ASTValidationError(Exception):
    """Raised when AST validation fails."""
    pass


BINARY_OPS: Set[str] = {
    "+", "-", "*", "/",
    "min", "max",
    "<", "<=", ">", ">=", "==", "!=",
    "and", "or",
}

UNARY_OPS: Set[str] = {
    "neg", "abs", "sqrt",
    "sin", "cos", "tan",
    "exp", "log",
    "not",
}

ALLOWED_VARIABLES: Set[str] = {
    "x", "y", "z",
    "r",
    "t",
}


@dataclass
class ASTNode:
    """Base class for AST nodes."""
    pass


@dataclass
class LiteralNode(ASTNode):
    """A numeric literal."""
    value: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "literal", "value": self.value}


@dataclass
class VariableNode(ASTNode):
    """A variable reference."""
    name: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "var", "name": self.name}


@dataclass
class BinaryOpNode(ASTNode):
    """A binary operation."""
    op: str
    left: ASTNode
    right: ASTNode
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "binop",
            "op": self.op,
            "left": _node_to_dict(self.left),
            "right": _node_to_dict(self.right),
        }


@dataclass
class UnaryOpNode(ASTNode):
    """A unary operation."""
    op: str
    arg: ASTNode
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "unop",
            "op": self.op,
            "arg": _node_to_dict(self.arg),
        }


def _node_to_dict(node: ASTNode) -> Dict[str, Any]:
    """Convert an AST node to dict representation."""
    if hasattr(node, "to_dict"):
        return node.to_dict()
    return {}


def _validate_node(
    node: Dict[str, Any],
    path: str = "root",
    allowed_vars: Optional[Set[str]] = None,
) -> List[str]:
    """
    Validate an AST node recursively.
    
    Parameters
    ----------
    node : dict
        The node dict to validate
    path : str
        Path string for error messages
    allowed_vars : set of str, optional
        Set of allowed variable names (default: ALLOWED_VARIABLES)
        
    Returns
    -------
    list of str
        List of validation error messages (empty if valid)
    """
    errors = []
    
    if allowed_vars is None:
        allowed_vars = ALLOWED_VARIABLES
    
    if not isinstance(node, dict):
        errors.append(f"{path}: node must be a dict, got {type(node).__name__}")
        return errors
    
    node_type = node.get("type")
    if not node_type:
        errors.append(f"{path}: node missing 'type' field")
        return errors
    
    if node_type == "literal":
        value = node.get("value")
        if value is None:
            errors.append(f"{path}: literal node missing 'value' field")
        elif not isinstance(value, (int, float)):
            errors.append(f"{path}: literal value must be a number, got {type(value).__name__}")
    
    elif node_type == "var":
        name = node.get("name")
        if name is None:
            errors.append(f"{path}: var node missing 'name' field")
        elif not isinstance(name, str):
            errors.append(f"{path}: var name must be a string, got {type(name).__name__}")
        elif name not in allowed_vars:
            errors.append(f"{path}: unknown variable '{name}', allowed: {allowed_vars}")
    
    elif node_type == "binop":
        op = node.get("op")
        if op is None:
            errors.append(f"{path}: binop node missing 'op' field")
        elif op not in BINARY_OPS:
            errors.append(f"{path}: unknown binary op '{op}', allowed: {BINARY_OPS}")
        
        left = node.get("left")
        if left is None:
            errors.append(f"{path}: binop node missing 'left' field")
        else:
            errors.extend(_validate_node(left, f"{path}.left", allowed_vars))
        
        right = node.get("right")
        if right is None:
            errors.append(f"{path}: binop node missing 'right' field")
        else:
            errors.extend(_validate_node(right, f"{path}.right", allowed_vars))
    
    elif node_type == "unop":
        op = node.get("op")
        if op is None:
            errors.append(f"{path}: unop node missing 'op' field")
        elif op not in UNARY_OPS:
            errors.append(f"{path}: unknown unary op '{op}', allowed: {UNARY_OPS}")
        
        arg = node.get("arg")
        if arg is None:
            errors.append(f"{path}: unop node missing 'arg' field")
        else:
            errors.extend(_validate_node(arg, f"{path}.arg", allowed_vars))
    
    else:
        errors.append(f"{path}: unknown node type '{node_type}'")
    
    return errors


def validate_ast(
    ast: Dict[str, Any],
    allowed_vars: Optional[Set[str]] = None,
) -> List[str]:
    """
    Validate an AST dict.
    
    Parameters
    ----------
    ast : dict
        The AST dict to validate (should have "format" and "root" keys)
    allowed_vars : set of str, optional
        Set of allowed variable names
        
    Returns
    -------
    list of str
        List of validation error messages (empty if valid)
        
    Raises
    ------
    ASTValidationError
        If the AST structure is fundamentally invalid
    """
    errors = []
    
    if not isinstance(ast, dict):
        raise ASTValidationError(f"AST must be a dict, got {type(ast).__name__}")
    
    ast_format = ast.get("format")
    if ast_format != "aog_ast_v1":
        errors.append(f"AST format must be 'aog_ast_v1', got '{ast_format}'")
    
    root = ast.get("root")
    if root is None:
        errors.append("AST missing 'root' field")
    else:
        errors.extend(_validate_node(root, "root", allowed_vars))
    
    return errors


def parse_ast(ast_dict: Dict[str, Any]) -> ASTNode:
    """
    Parse an AST dict into ASTNode objects.
    
    Parameters
    ----------
    ast_dict : dict
        The AST dict with "format" and "root" keys
        
    Returns
    -------
    ASTNode
        The parsed AST root node
        
    Raises
    ------
    ASTValidationError
        If parsing fails
    """
    errors = validate_ast(ast_dict)
    if errors:
        raise ASTValidationError(f"AST validation failed: {'; '.join(errors)}")
    
    return _parse_node(ast_dict["root"])


def _parse_node(node: Dict[str, Any]) -> ASTNode:
    """Parse a single node dict into an ASTNode."""
    node_type = node["type"]
    
    if node_type == "literal":
        return LiteralNode(value=float(node["value"]))
    
    elif node_type == "var":
        return VariableNode(name=node["name"])
    
    elif node_type == "binop":
        return BinaryOpNode(
            op=node["op"],
            left=_parse_node(node["left"]),
            right=_parse_node(node["right"]),
        )
    
    elif node_type == "unop":
        return UnaryOpNode(
            op=node["op"],
            arg=_parse_node(node["arg"]),
        )
    
    else:
        raise ASTValidationError(f"Unknown node type: {node_type}")


__all__ = [
    "ASTNode",
    "LiteralNode",
    "VariableNode",
    "BinaryOpNode",
    "UnaryOpNode",
    "ASTValidationError",
    "BINARY_OPS",
    "UNARY_OPS",
    "ALLOWED_VARIABLES",
    "validate_ast",
    "parse_ast",
]
