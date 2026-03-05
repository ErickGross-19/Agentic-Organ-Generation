"""
AST subsystem for DesignSpec.

This module provides JSON AST validation and compilation for safe,
deterministic evaluation of mathematical expressions in specs.

Supported use cases:
- Implicit domain definitions (SDF expressions)
- Taper schedules
- Scoring functions

The AST format is "aog_ast_v1" which supports:
- Numeric literals
- Variables (x, y, z for spatial coordinates)
- Binary operations (+, -, *, /, min, max)
- Unary operations (neg, abs, sqrt)
- Comparison operations (<, <=, >, >=)
- Logical operations (and, or, not)
"""

from .nodes import (
    ASTNode,
    LiteralNode,
    VariableNode,
    BinaryOpNode,
    UnaryOpNode,
    validate_ast,
    ASTValidationError,
)

from .compile import (
    compile_ast,
    ASTCompilationError,
)

__all__ = [
    # Nodes
    "ASTNode",
    "LiteralNode",
    "VariableNode",
    "BinaryOpNode",
    "UnaryOpNode",
    "validate_ast",
    "ASTValidationError",
    # Compile
    "compile_ast",
    "ASTCompilationError",
]
