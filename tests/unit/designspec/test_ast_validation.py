"""Tests for AST validation."""

import pytest
from designspec.ast.nodes import (
    validate_ast,
    parse_ast,
    ASTValidationError,
    BINARY_OPS,
    UNARY_OPS,
    ALLOWED_VARIABLES,
)


class TestValidateAST:
    """Tests for AST validation."""
    
    def test_valid_literal_node(self):
        ast = {
            "format": "aog_ast_v1",
            "root": {"type": "literal", "value": 42.0},
        }
        errors = validate_ast(ast)
        assert len(errors) == 0
    
    def test_valid_variable_node(self):
        ast = {
            "format": "aog_ast_v1",
            "root": {"type": "var", "name": "x"},
        }
        errors = validate_ast(ast)
        assert len(errors) == 0
    
    def test_valid_binary_op_node(self):
        ast = {
            "format": "aog_ast_v1",
            "root": {
                "type": "binop",
                "op": "+",
                "left": {"type": "literal", "value": 1.0},
                "right": {"type": "literal", "value": 2.0},
            },
        }
        errors = validate_ast(ast)
        assert len(errors) == 0
    
    def test_valid_unary_op_node(self):
        ast = {
            "format": "aog_ast_v1",
            "root": {
                "type": "unop",
                "op": "neg",
                "arg": {"type": "literal", "value": 5.0},
            },
        }
        errors = validate_ast(ast)
        assert len(errors) == 0
    
    def test_invalid_format_rejected(self):
        ast = {
            "format": "wrong_format",
            "root": {"type": "literal", "value": 42.0},
        }
        errors = validate_ast(ast)
        assert len(errors) > 0
        assert any("format" in e for e in errors)
    
    def test_missing_root_rejected(self):
        ast = {"format": "aog_ast_v1"}
        errors = validate_ast(ast)
        assert len(errors) > 0
        assert any("root" in e for e in errors)
    
    def test_unknown_node_type_rejected(self):
        ast = {
            "format": "aog_ast_v1",
            "root": {"type": "unknown_type"},
        }
        errors = validate_ast(ast)
        assert len(errors) > 0
    
    def test_unknown_variable_rejected(self):
        ast = {
            "format": "aog_ast_v1",
            "root": {"type": "var", "name": "unknown_var"},
        }
        errors = validate_ast(ast)
        assert len(errors) > 0
        assert any("unknown variable" in e for e in errors)
    
    def test_unknown_binary_op_rejected(self):
        ast = {
            "format": "aog_ast_v1",
            "root": {
                "type": "binop",
                "op": "unknown_op",
                "left": {"type": "literal", "value": 1.0},
                "right": {"type": "literal", "value": 2.0},
            },
        }
        errors = validate_ast(ast)
        assert len(errors) > 0
    
    def test_unknown_unary_op_rejected(self):
        ast = {
            "format": "aog_ast_v1",
            "root": {
                "type": "unop",
                "op": "unknown_op",
                "arg": {"type": "literal", "value": 5.0},
            },
        }
        errors = validate_ast(ast)
        assert len(errors) > 0
    
    def test_missing_literal_value_rejected(self):
        ast = {
            "format": "aog_ast_v1",
            "root": {"type": "literal"},
        }
        errors = validate_ast(ast)
        assert len(errors) > 0
        assert any("value" in e for e in errors)
    
    def test_missing_var_name_rejected(self):
        ast = {
            "format": "aog_ast_v1",
            "root": {"type": "var"},
        }
        errors = validate_ast(ast)
        assert len(errors) > 0
        assert any("name" in e for e in errors)
    
    def test_missing_binop_left_rejected(self):
        ast = {
            "format": "aog_ast_v1",
            "root": {
                "type": "binop",
                "op": "+",
                "right": {"type": "literal", "value": 2.0},
            },
        }
        errors = validate_ast(ast)
        assert len(errors) > 0
        assert any("left" in e for e in errors)
    
    def test_missing_unop_arg_rejected(self):
        ast = {
            "format": "aog_ast_v1",
            "root": {
                "type": "unop",
                "op": "neg",
            },
        }
        errors = validate_ast(ast)
        assert len(errors) > 0
        assert any("arg" in e for e in errors)
    
    def test_custom_allowed_vars(self):
        ast = {
            "format": "aog_ast_v1",
            "root": {"type": "var", "name": "custom"},
        }
        errors = validate_ast(ast, allowed_vars={"custom"})
        assert len(errors) == 0


class TestParseAST:
    """Tests for AST parsing."""
    
    def test_parse_literal(self):
        ast = {
            "format": "aog_ast_v1",
            "root": {"type": "literal", "value": 42.0},
        }
        node = parse_ast(ast)
        assert node.value == 42.0
    
    def test_parse_variable(self):
        ast = {
            "format": "aog_ast_v1",
            "root": {"type": "var", "name": "x"},
        }
        node = parse_ast(ast)
        assert node.name == "x"
    
    def test_parse_binary_op(self):
        ast = {
            "format": "aog_ast_v1",
            "root": {
                "type": "binop",
                "op": "+",
                "left": {"type": "literal", "value": 1.0},
                "right": {"type": "literal", "value": 2.0},
            },
        }
        node = parse_ast(ast)
        assert node.op == "+"
        assert node.left.value == 1.0
        assert node.right.value == 2.0
    
    def test_parse_invalid_ast_raises(self):
        ast = {
            "format": "wrong_format",
            "root": {"type": "literal", "value": 42.0},
        }
        with pytest.raises(ASTValidationError):
            parse_ast(ast)


class TestASTConstants:
    """Tests for AST constants."""
    
    def test_binary_ops_includes_arithmetic(self):
        assert "+" in BINARY_OPS
        assert "-" in BINARY_OPS
        assert "*" in BINARY_OPS
        assert "/" in BINARY_OPS
    
    def test_binary_ops_includes_comparison(self):
        assert "<" in BINARY_OPS
        assert "<=" in BINARY_OPS
        assert ">" in BINARY_OPS
        assert ">=" in BINARY_OPS
    
    def test_binary_ops_includes_min_max(self):
        assert "min" in BINARY_OPS
        assert "max" in BINARY_OPS
    
    def test_unary_ops_includes_common(self):
        assert "neg" in UNARY_OPS
        assert "abs" in UNARY_OPS
        assert "sqrt" in UNARY_OPS
    
    def test_allowed_variables_includes_xyz(self):
        assert "x" in ALLOWED_VARIABLES
        assert "y" in ALLOWED_VARIABLES
        assert "z" in ALLOWED_VARIABLES
