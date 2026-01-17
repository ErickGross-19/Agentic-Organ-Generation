"""Tests for alias application and backward compatibility."""

import pytest
from designspec.compat.v1_aliases import (
    V1_TOP_LEVEL_ALIASES,
    V1_POLICY_ALIASES,
    V1_COMPONENT_ALIASES,
    apply_aliases,
    apply_policy_aliases,
    apply_component_aliases,
    apply_all_aliases,
)


class TestApplyAliases:
    """Tests for basic alias application."""
    
    def test_apply_single_alias(self):
        data = {"old_key": "value"}
        aliases = {"old_key": "new_key"}
        result, applied = apply_aliases(data, aliases)
        
        assert "new_key" in result
        assert "old_key" not in result
        assert result["new_key"] == "value"
        assert len(applied) == 1
    
    def test_apply_multiple_aliases(self):
        data = {"old1": "v1", "old2": "v2", "keep": "v3"}
        aliases = {"old1": "new1", "old2": "new2"}
        result, applied = apply_aliases(data, aliases)
        
        assert "new1" in result
        assert "new2" in result
        assert "keep" in result
        assert len(applied) == 2
    
    def test_no_aliases_applied_when_none_match(self):
        data = {"key1": "v1", "key2": "v2"}
        aliases = {"other": "new"}
        result, applied = apply_aliases(data, aliases)
        
        assert result == data
        assert len(applied) == 0
    
    def test_alias_does_not_overwrite_existing_key(self):
        data = {"old_key": "old_value", "new_key": "existing_value"}
        aliases = {"old_key": "new_key"}
        result, applied = apply_aliases(data, aliases)
        
        assert result["new_key"] == "existing_value"
        assert "old_key" not in result
        assert len(applied) == 0


class TestApplyPolicyAliases:
    """Tests for policy-specific alias application."""
    
    def test_resolution_policy_alias(self):
        policies = {
            "resolution": {
                "voxels_across_min_diameter": 8,
            }
        }
        result, warnings = apply_policy_aliases(policies)
        
        assert "min_voxels_across_feature" in result["resolution"]
        assert len(warnings) > 0
    
    def test_channels_policy_alias(self):
        policies = {
            "channels": {
                "hook_strategy": "reduce_depth",
            }
        }
        result, warnings = apply_policy_aliases(policies)
        
        assert "constraint_strategy" in result["channels"]
        assert len(warnings) > 0
    
    def test_unknown_policy_passes_through(self):
        policies = {
            "unknown_policy": {
                "some_field": "value",
            }
        }
        result, warnings = apply_policy_aliases(policies)
        
        assert "unknown_policy" in result
        assert result["unknown_policy"]["some_field"] == "value"


class TestApplyComponentAliases:
    """Tests for component alias application."""
    
    def test_component_alias_applied(self):
        component = {"generator": "space_colonization"}
        result, applied = apply_component_aliases(component)
        
        if "generator" in V1_COMPONENT_ALIASES:
            assert "build" in result
            assert result["build"]["type"] == "space_colonization"
            assert len(applied) > 0


class TestApplyAllAliases:
    """Tests for full spec alias application."""
    
    def test_apply_all_aliases_to_spec(self):
        spec = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42},
            "policies": {
                "resolution": {
                    "voxels_across_min_diameter": 8,
                }
            },
            "domains": {},
            "components": [],
        }
        result, warnings = apply_all_aliases(spec)
        
        assert "schema" in result
        assert "meta" in result
        assert "policies" in result
    
    def test_warnings_recorded_for_applied_aliases(self):
        spec = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42},
            "policies": {
                "resolution": {
                    "voxels_across_min_diameter": 8,
                }
            },
            "domains": {},
            "components": [],
        }
        result, warnings = apply_all_aliases(spec)
        
        assert any("voxels_across_min_diameter" in w for w in warnings)


class TestAliasConstants:
    """Tests for alias constant definitions."""
    
    def test_policy_aliases_has_resolution(self):
        assert "resolution" in V1_POLICY_ALIASES
    
    def test_policy_aliases_has_channels(self):
        assert "channels" in V1_POLICY_ALIASES
    
    def test_resolution_aliases_include_voxels_across(self):
        resolution_aliases = V1_POLICY_ALIASES.get("resolution", {})
        assert "voxels_across_min_diameter" in resolution_aliases
