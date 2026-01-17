"""Tests for schema versioning and compatibility."""

import pytest
from designspec.schema import (
    SCHEMA_NAME,
    SCHEMA_VERSION,
    SUPPORTED_VERSIONS,
    is_version_compatible,
    parse_version,
    validate_schema_block,
    SchemaValidationError,
)


class TestParseVersion:
    """Tests for version parsing."""
    
    def test_parse_simple_version(self):
        major, minor, patch = parse_version("1.0.0")
        assert major == 1
        assert minor == 0
        assert patch == 0
    
    def test_parse_version_with_larger_numbers(self):
        major, minor, patch = parse_version("2.15.3")
        assert major == 2
        assert minor == 15
        assert patch == 3
    
    def test_parse_invalid_version_raises(self):
        with pytest.raises(ValueError):
            parse_version("invalid")
    
    def test_parse_incomplete_version_raises(self):
        with pytest.raises(ValueError):
            parse_version("1.0")


class TestVersionCompatibility:
    """Tests for version compatibility checking."""
    
    def test_exact_version_match_is_compatible(self):
        assert is_version_compatible("1.0.0", "1.0.0") is True
    
    def test_same_major_minor_different_patch_is_compatible(self):
        assert is_version_compatible("1.0.1", "1.0.0") is True
        assert is_version_compatible("1.0.5", "1.0.0") is True
    
    def test_different_major_is_incompatible(self):
        assert is_version_compatible("2.0.0", "1.0.0") is False
    
    def test_different_minor_is_incompatible(self):
        assert is_version_compatible("1.1.0", "1.0.0") is False
    
    def test_current_version_is_compatible(self):
        assert is_version_compatible(SCHEMA_VERSION) is True


class TestValidateSchemaBlock:
    """Tests for schema block validation."""
    
    def test_valid_schema_block(self):
        schema_block = {
            "name": SCHEMA_NAME,
            "version": SCHEMA_VERSION,
        }
        errors = validate_schema_block(schema_block)
        assert len(errors) == 0
    
    def test_missing_name_has_error(self):
        schema_block = {"version": SCHEMA_VERSION}
        errors = validate_schema_block(schema_block)
        assert len(errors) > 0
        assert any("name" in e for e in errors)
    
    def test_missing_version_has_error(self):
        schema_block = {"name": SCHEMA_NAME}
        errors = validate_schema_block(schema_block)
        assert len(errors) > 0
        assert any("version" in e for e in errors)
    
    def test_wrong_schema_name_has_error(self):
        schema_block = {
            "name": "wrong_name",
            "version": SCHEMA_VERSION,
        }
        errors = validate_schema_block(schema_block)
        assert len(errors) > 0
        assert any("name" in e for e in errors)
    
    def test_incompatible_version_has_error(self):
        schema_block = {
            "name": SCHEMA_NAME,
            "version": "2.0.0",
        }
        errors = validate_schema_block(schema_block)
        assert len(errors) > 0
        assert any("version" in e or "compatible" in e for e in errors)
    
    def test_compatible_with_list_is_valid(self):
        schema_block = {
            "name": SCHEMA_NAME,
            "version": SCHEMA_VERSION,
            "compatible_with": ["1.0.x"],
        }
        errors = validate_schema_block(schema_block)
        assert len(errors) == 0


class TestSchemaConstants:
    """Tests for schema constants."""
    
    def test_schema_name_is_correct(self):
        assert SCHEMA_NAME == "aog_designspec"
    
    def test_schema_version_is_semver(self):
        parts = SCHEMA_VERSION.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)
    
    def test_supported_versions_includes_current(self):
        assert SCHEMA_VERSION in SUPPORTED_VERSIONS
