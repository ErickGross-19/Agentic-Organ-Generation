"""Tests for AST compilation and evaluation."""

import pytest
import math
from designspec.ast.compile import (
    compile_ast,
    make_sphere_sdf_ast,
    make_box_sdf_ast,
    ASTCompilationError,
)
from designspec.ast.nodes import ASTValidationError


class TestCompileAST:
    """Tests for AST compilation."""
    
    def test_compile_literal(self):
        ast = {
            "format": "aog_ast_v1",
            "root": {"type": "literal", "value": 42.0},
        }
        fn = compile_ast(ast)
        assert fn({}) == 42.0
    
    def test_compile_variable(self):
        ast = {
            "format": "aog_ast_v1",
            "root": {"type": "var", "name": "x"},
        }
        fn = compile_ast(ast)
        assert fn({"x": 5.0}) == 5.0
        assert fn({"x": -3.0}) == -3.0
    
    def test_compile_addition(self):
        ast = {
            "format": "aog_ast_v1",
            "root": {
                "type": "binop",
                "op": "+",
                "left": {"type": "literal", "value": 1.0},
                "right": {"type": "literal", "value": 2.0},
            },
        }
        fn = compile_ast(ast)
        assert fn({}) == 3.0
    
    def test_compile_subtraction(self):
        ast = {
            "format": "aog_ast_v1",
            "root": {
                "type": "binop",
                "op": "-",
                "left": {"type": "literal", "value": 5.0},
                "right": {"type": "literal", "value": 3.0},
            },
        }
        fn = compile_ast(ast)
        assert fn({}) == 2.0
    
    def test_compile_multiplication(self):
        ast = {
            "format": "aog_ast_v1",
            "root": {
                "type": "binop",
                "op": "*",
                "left": {"type": "literal", "value": 4.0},
                "right": {"type": "literal", "value": 3.0},
            },
        }
        fn = compile_ast(ast)
        assert fn({}) == 12.0
    
    def test_compile_division(self):
        ast = {
            "format": "aog_ast_v1",
            "root": {
                "type": "binop",
                "op": "/",
                "left": {"type": "literal", "value": 10.0},
                "right": {"type": "literal", "value": 2.0},
            },
        }
        fn = compile_ast(ast)
        assert fn({}) == 5.0
    
    def test_compile_negation(self):
        ast = {
            "format": "aog_ast_v1",
            "root": {
                "type": "unop",
                "op": "neg",
                "arg": {"type": "literal", "value": 5.0},
            },
        }
        fn = compile_ast(ast)
        assert fn({}) == -5.0
    
    def test_compile_abs(self):
        ast = {
            "format": "aog_ast_v1",
            "root": {
                "type": "unop",
                "op": "abs",
                "arg": {"type": "literal", "value": -5.0},
            },
        }
        fn = compile_ast(ast)
        assert fn({}) == 5.0
    
    def test_compile_sqrt(self):
        ast = {
            "format": "aog_ast_v1",
            "root": {
                "type": "unop",
                "op": "sqrt",
                "arg": {"type": "literal", "value": 9.0},
            },
        }
        fn = compile_ast(ast)
        assert fn({}) == 3.0
    
    def test_compile_min(self):
        ast = {
            "format": "aog_ast_v1",
            "root": {
                "type": "binop",
                "op": "min",
                "left": {"type": "literal", "value": 3.0},
                "right": {"type": "literal", "value": 7.0},
            },
        }
        fn = compile_ast(ast)
        assert fn({}) == 3.0
    
    def test_compile_max(self):
        ast = {
            "format": "aog_ast_v1",
            "root": {
                "type": "binop",
                "op": "max",
                "left": {"type": "literal", "value": 3.0},
                "right": {"type": "literal", "value": 7.0},
            },
        }
        fn = compile_ast(ast)
        assert fn({}) == 7.0
    
    def test_compile_invalid_ast_raises(self):
        ast = {
            "format": "wrong_format",
            "root": {"type": "literal", "value": 42.0},
        }
        with pytest.raises(ASTValidationError):
            compile_ast(ast)


class TestSphereSDF:
    """Tests for sphere SDF AST."""
    
    def test_sphere_sdf_at_center(self):
        ast = make_sphere_sdf_ast(radius=1.0)
        fn = compile_ast(ast)
        
        result = fn({"x": 0, "y": 0, "z": 0})
        assert result == pytest.approx(-1.0)
    
    def test_sphere_sdf_on_surface(self):
        ast = make_sphere_sdf_ast(radius=1.0)
        fn = compile_ast(ast)
        
        result = fn({"x": 1, "y": 0, "z": 0})
        assert result == pytest.approx(0.0)
        
        result = fn({"x": 0, "y": 1, "z": 0})
        assert result == pytest.approx(0.0)
        
        result = fn({"x": 0, "y": 0, "z": 1})
        assert result == pytest.approx(0.0)
    
    def test_sphere_sdf_outside(self):
        ast = make_sphere_sdf_ast(radius=1.0)
        fn = compile_ast(ast)
        
        result = fn({"x": 2, "y": 0, "z": 0})
        assert result == pytest.approx(1.0)
    
    def test_sphere_sdf_inside(self):
        ast = make_sphere_sdf_ast(radius=1.0)
        fn = compile_ast(ast)
        
        result = fn({"x": 0.5, "y": 0, "z": 0})
        assert result == pytest.approx(-0.5)
    
    def test_sphere_sdf_custom_radius(self):
        ast = make_sphere_sdf_ast(radius=2.0)
        fn = compile_ast(ast)
        
        result = fn({"x": 0, "y": 0, "z": 0})
        assert result == pytest.approx(-2.0)
        
        result = fn({"x": 2, "y": 0, "z": 0})
        assert result == pytest.approx(0.0)
    
    def test_sphere_sdf_diagonal_point(self):
        ast = make_sphere_sdf_ast(radius=1.0)
        fn = compile_ast(ast)
        
        d = 1.0 / math.sqrt(3)
        result = fn({"x": d, "y": d, "z": d})
        assert result == pytest.approx(0.0, abs=1e-10)


class TestBoxSDF:
    """Tests for box SDF AST."""
    
    def test_box_sdf_at_center(self):
        ast = make_box_sdf_ast(half_extents=(1.0, 1.0, 1.0))
        fn = compile_ast(ast)
        
        result = fn({"x": 0, "y": 0, "z": 0})
        assert result == pytest.approx(-1.0)
    
    def test_box_sdf_on_face(self):
        ast = make_box_sdf_ast(half_extents=(1.0, 1.0, 1.0))
        fn = compile_ast(ast)
        
        result = fn({"x": 1, "y": 0, "z": 0})
        assert result == pytest.approx(0.0)
    
    def test_box_sdf_outside(self):
        ast = make_box_sdf_ast(half_extents=(1.0, 1.0, 1.0))
        fn = compile_ast(ast)
        
        result = fn({"x": 2, "y": 0, "z": 0})
        assert result == pytest.approx(1.0)
    
    def test_box_sdf_custom_extents(self):
        ast = make_box_sdf_ast(half_extents=(2.0, 1.0, 0.5))
        fn = compile_ast(ast)
        
        result = fn({"x": 2, "y": 0, "z": 0})
        assert result == pytest.approx(0.0)
        
        result = fn({"x": 0, "y": 1, "z": 0})
        assert result == pytest.approx(0.0)
        
        result = fn({"x": 0, "y": 0, "z": 0.5})
        assert result == pytest.approx(0.0)


class TestComplexExpressions:
    """Tests for complex AST expressions."""
    
    def test_nested_expression(self):
        ast = {
            "format": "aog_ast_v1",
            "root": {
                "type": "binop",
                "op": "+",
                "left": {
                    "type": "binop",
                    "op": "*",
                    "left": {"type": "var", "name": "x"},
                    "right": {"type": "literal", "value": 2.0},
                },
                "right": {"type": "var", "name": "y"},
            },
        }
        fn = compile_ast(ast)
        
        assert fn({"x": 3, "y": 4}) == 10.0
        assert fn({"x": 0, "y": 5}) == 5.0
    
    def test_expression_with_multiple_variables(self):
        ast = {
            "format": "aog_ast_v1",
            "root": {
                "type": "binop",
                "op": "+",
                "left": {
                    "type": "binop",
                    "op": "+",
                    "left": {"type": "var", "name": "x"},
                    "right": {"type": "var", "name": "y"},
                },
                "right": {"type": "var", "name": "z"},
            },
        }
        fn = compile_ast(ast)
        
        assert fn({"x": 1, "y": 2, "z": 3}) == 6.0
