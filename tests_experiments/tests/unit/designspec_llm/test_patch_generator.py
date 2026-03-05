"""
Unit tests for PatchGenerator

Tests the JSON patch generation from structured errors.
"""

import pytest
from automation.designspec_llm.patch_generator import PatchGenerator
from automation.designspec_llm.error_taxonomy import StructuredError, ErrorType


class TestPatchGenerator:
    """Test PatchGenerator functionality."""

    def test_generate_pitch_fix_with_value(self):
        """Test generating pitch fix with suggested value."""
        generator = PatchGenerator()
        error = StructuredError(
            raw_message="pitch too large",
            error_type=ErrorType.PITCH_TOO_LARGE,
            affected_policy="mesh_merge",
            json_path="/policies/mesh_merge/voxel_pitch",
            current_value=0.01,
            suggested_value=0.0002,
            confidence=0.95,
        )
        spec = {}

        patches = generator.generate_fix_patches([error], spec)

        assert len(patches) >= 1
        assert patches[0]["op"] == "replace"
        assert patches[0]["path"] == "/policies/mesh_merge/voxel_pitch"
        assert patches[0]["value"] == 0.0002

    def test_generate_domain_fix(self):
        """Test generating fix for missing domain."""
        generator = PatchGenerator()
        error = StructuredError(
            raw_message="no domain defined",
            error_type=ErrorType.NO_DOMAIN_DEFINED,
            confidence=0.95,
        )
        spec = {}

        patches = generator.generate_fix_patches([error], spec)

        assert len(patches) >= 1
        assert patches[0]["op"] == "add"
        assert "/domains" in patches[0]["path"]

    def test_skip_low_confidence_errors(self):
        """Test that low confidence errors are skipped."""
        generator = PatchGenerator()
        error = StructuredError(
            raw_message="unknown error",
            error_type=ErrorType.UNKNOWN,
            confidence=0.3,
        )
        spec = {}

        patches = generator.generate_fix_patches([error], spec)

        assert len(patches) == 0

    def test_generate_multiple_patches(self):
        """Test generating patches for multiple errors."""
        generator = PatchGenerator()
        errors = [
            StructuredError(
                raw_message="pitch error",
                error_type=ErrorType.PITCH_TOO_LARGE,
                json_path="/policies/mesh_merge/voxel_pitch",
                suggested_value=0.0002,
                confidence=0.95,
            ),
            StructuredError(
                raw_message="domain error",
                error_type=ErrorType.DOMAIN_TOO_SMALL,
                confidence=0.85,
            ),
        ]
        spec = {
            "domains": [{
                "type": "box",
                "box": {"width": 0.01, "height": 0.01, "depth": 0.01}
            }]
        }

        patches = generator.generate_fix_patches(errors, spec)

        # Should have at least one patch per high-confidence error
        assert len(patches) >= 2

    def test_generate_explanation(self):
        """Test generating explanation text."""
        generator = PatchGenerator()
        errors = [
            StructuredError(
                raw_message="pitch too large",
                error_type=ErrorType.PITCH_TOO_LARGE,
                json_path="/policies/mesh_merge/voxel_pitch",
                current_value=0.01,
                suggested_value=0.0002,
                suggested_fix="Set voxel_pitch to 0.0002m",
                confidence=0.95,
            ),
        ]
        patches = [
            {"op": "replace", "path": "/policies/mesh_merge/voxel_pitch", "value": 0.0002}
        ]

        explanation = generator.generate_explanation(errors, patches)

        assert "pitch_too_large" in explanation
        assert "voxel_pitch" in explanation.lower()
        assert "0.01" in explanation
        assert "0.0002" in explanation

    def test_fix_pitch_too_small(self):
        """Test fixing pitch too small error."""
        generator = PatchGenerator()
        error = StructuredError(
            raw_message="pitch too small",
            error_type=ErrorType.PITCH_TOO_SMALL,
            confidence=0.85,
        )
        spec = {
            "policies": {
                "mesh_merge": {
                    "voxel_pitch": 0.00001
                }
            }
        }

        patches = generator.generate_fix_patches([error], spec)

        assert len(patches) >= 1
        assert patches[0]["op"] == "replace"
        assert patches[0]["value"] > 0.00001  # Should be increased

    def test_fix_domain_too_small(self):
        """Test fixing domain too small error."""
        generator = PatchGenerator()
        error = StructuredError(
            raw_message="domain too small",
            error_type=ErrorType.DOMAIN_TOO_SMALL,
            confidence=0.85,
        )
        spec = {
            "domains": [{
                "type": "box",
                "box": {"width": 0.01, "height": 0.01, "depth": 0.01}
            }]
        }

        patches = generator.generate_fix_patches([error], spec)

        # Should have patches for width, height, depth
        assert len(patches) >= 3
        width_patch = next(p for p in patches if "width" in p["path"])
        assert width_patch["value"] > 0.01  # Should be increased
