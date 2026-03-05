"""
Unit tests for ErrorParser

Tests the error parsing and classification functionality.
"""

import pytest
from automation.designspec_llm.error_taxonomy import ErrorParser, ErrorType, StructuredError


class TestErrorParser:
    """Test ErrorParser functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.parser = ErrorParser()

    def test_parse_pitch_too_large_with_values(self):
        """Test parsing pitch error with explicit values."""
        parser = ErrorParser()
        error_msg = "Error: domain scale is ~0.02 but voxel_pitch is 0.01"
        spec = {}

        result = parser.parse(error_msg, spec)

        assert result.error_type == ErrorType.PITCH_TOO_LARGE
        assert result.confidence > 0.9
        assert result.current_value == 0.01
        assert result.suggested_value is not None
        assert result.json_path == "/policies/mesh_merge/voxel_pitch"

    def test_parse_pitch_too_large_generic(self):
        """Test parsing generic pitch error."""
        parser = ErrorParser()
        error_msg = "The voxel pitch is too large for this domain"
        spec = {}

        result = parser.parse(error_msg, spec)

        assert result.error_type == ErrorType.PITCH_TOO_LARGE
        assert result.confidence > 0.8
        assert "voxel_pitch" in result.suggested_fix.lower()

    def test_parse_component_outside_domain(self):
        """Test parsing component outside domain error."""
        parser = ErrorParser()
        error_msg = "Component exceeds domain boundaries"
        spec = {}

        result = parser.parse(error_msg, spec)

        assert result.error_type == ErrorType.COMPONENT_OUTSIDE_DOMAIN
        assert result.confidence > 0.8

    def test_parse_port_direction_error(self):
        """Test parsing port direction error."""
        parser = ErrorParser()
        error_msg = "Port direction is invalid for domain face"
        spec = {}

        result = parser.parse(error_msg, spec)

        assert result.error_type == ErrorType.PORT_DIRECTION_WRONG
        assert result.confidence > 0.8

    def test_parse_no_domain_error(self):
        """Test parsing no domain error."""
        parser = ErrorParser()
        error_msg = "No domain defined in specification"
        spec = {}

        result = parser.parse(error_msg, spec)

        assert result.error_type == ErrorType.NO_DOMAIN_DEFINED
        assert result.confidence > 0.9

    def test_parse_memory_error(self):
        """Test parsing memory error."""
        parser = ErrorParser()
        error_msg = "Out of memory during mesh generation"
        spec = {}

        result = parser.parse(error_msg, spec)

        assert result.error_type == ErrorType.MEMORY_ERROR
        assert result.confidence > 0.8
        assert "voxel_pitch" in result.suggested_fix.lower()

    def test_parse_unknown_error(self):
        """Test parsing unknown error."""
        parser = ErrorParser()
        error_msg = "Some completely unknown error message"
        spec = {}

        result = parser.parse(error_msg, spec)

        assert result.error_type == ErrorType.UNKNOWN
        assert result.confidence < 0.5

    def test_parse_multiple_errors(self):
        """Test parsing multiple errors."""
        parser = ErrorParser()
        errors = [
            "domain scale is ~0.02 but voxel_pitch is 0.01",
            "Component exceeds domain boundaries",
            "Port direction is invalid",
        ]
        spec = {}

        results = parser.parse_multiple(errors, spec)

        assert len(results) == 3
        assert results[0].error_type == ErrorType.PITCH_TOO_LARGE
        assert results[1].error_type == ErrorType.COMPONENT_OUTSIDE_DOMAIN
        assert results[2].error_type == ErrorType.PORT_DIRECTION_WRONG

    def test_structured_error_to_dict(self):
        """Test structured error serialization."""
        error = StructuredError(
            raw_message="Test error",
            error_type=ErrorType.PITCH_TOO_LARGE,
            affected_policy="mesh_merge",
            json_path="/policies/mesh_merge/voxel_pitch",
            current_value=0.01,
            suggested_value=0.0002,
            suggested_fix="Reduce voxel_pitch",
            confidence=0.95,
        )

        data = error.to_dict()

        assert data["error_type"] == "pitch_too_large"
        assert data["affected_policy"] == "mesh_merge"
        assert data["current_value"] == 0.01
        assert data["suggested_value"] == 0.0002
        assert data["confidence"] == 0.95
