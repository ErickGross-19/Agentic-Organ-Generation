"""
Tests for component policy override merging via deep_merge.

Requirement 1: Component policy overrides must work
- Each component may include `policy_overrides` dict
- Runner must deep-merge top-level policies with per-component overrides
- Deep merge rules:
  - dict values merged recursively
  - lists replaced (no concatenation)
  - scalar values override
"""

import pytest
from designspec.runner import deep_merge


class TestDeepMerge:
    """Tests for the deep_merge helper function."""
    
    def test_empty_override_returns_base_copy(self):
        """Empty override should return a copy of base."""
        base = {"a": 1, "b": {"c": 2}}
        result = deep_merge(base, {})
        assert result == base
        assert result is not base
    
    def test_empty_base_returns_override_copy(self):
        """Empty base should return a copy of override."""
        override = {"a": 1, "b": {"c": 2}}
        result = deep_merge({}, override)
        assert result == override
        assert result is not override
    
    def test_scalar_override(self):
        """Scalar values should override."""
        base = {"a": 1, "b": 2}
        override = {"a": 10}
        result = deep_merge(base, override)
        assert result == {"a": 10, "b": 2}
    
    def test_nested_dict_merge(self):
        """Nested dicts should merge recursively."""
        base = {
            "growth": {
                "step_size": 0.001,
                "max_iterations": 1000,
                "params": {"a": 1, "b": 2}
            }
        }
        override = {
            "growth": {
                "step_size": 0.002,
                "params": {"b": 20, "c": 3}
            }
        }
        result = deep_merge(base, override)
        
        assert result["growth"]["step_size"] == 0.002
        assert result["growth"]["max_iterations"] == 1000
        assert result["growth"]["params"]["a"] == 1
        assert result["growth"]["params"]["b"] == 20
        assert result["growth"]["params"]["c"] == 3
    
    def test_list_replacement_not_concatenation(self):
        """Lists should be replaced, not concatenated."""
        base = {"items": [1, 2, 3], "other": "value"}
        override = {"items": [4, 5]}
        result = deep_merge(base, override)
        
        assert result["items"] == [4, 5]
        assert result["other"] == "value"
    
    def test_new_keys_added(self):
        """Keys only in override should be added."""
        base = {"a": 1}
        override = {"b": 2, "c": {"d": 3}}
        result = deep_merge(base, override)
        
        assert result == {"a": 1, "b": 2, "c": {"d": 3}}
    
    def test_none_values(self):
        """None values should override."""
        base = {"a": 1, "b": 2}
        override = {"a": None}
        result = deep_merge(base, override)
        
        assert result["a"] is None
        assert result["b"] == 2
    
    def test_deeply_nested_override(self):
        """Deep nesting should work correctly."""
        base = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": "original"
                    }
                }
            }
        }
        override = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": "overridden"
                    }
                }
            }
        }
        result = deep_merge(base, override)
        
        assert result["level1"]["level2"]["level3"]["value"] == "overridden"


class TestEffectivePolicyDict:
    """Tests for _get_effective_policy_dict method."""
    
    def test_no_overrides_returns_global(self):
        """Component without overrides should use global policy."""
        from designspec.spec import DesignSpec
        from designspec.runner import DesignSpecRunner
        
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m"},
            "domains": {
                "main_domain": {
                                        "type": "cylinder",
                                        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                                        "radius": 0.004,
                                        "height": 0.003,
                }
            },
            "components": [
                {
                    "id": "comp1",
                    "domain_ref": "main_domain",
                    "ports": {"inlets": [], "outlets": []},
                    "build": {"type": "backend_network"}
                }
            ],
            "policies": {
                "growth": {
                    "step_size": 0.001,
                    "max_iterations": 500
                }
            }
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        runner = DesignSpecRunner(spec)
        
        component = runner._get_component("comp1")
        effective = runner._get_effective_policy_dict(component, "growth")
        
        assert effective["step_size"] == 0.001
        assert effective["max_iterations"] == 500
    
    def test_component_override_applied(self):
        """Component with overrides should have them applied."""
        from designspec.spec import DesignSpec
        from designspec.runner import DesignSpecRunner
        
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m"},
            "domains": {
                "main_domain": {
                                        "type": "cylinder",
                                        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                                        "radius": 0.004,
                                        "height": 0.003,
                }
            },
            "components": [
                {
                    "id": "comp1",
                    "domain_ref": "main_domain",
                    "ports": {"inlets": [], "outlets": []},
                    "build": {"type": "backend_network"},
                    "policy_overrides": {
                        "growth": {
                            "step_size": 0.002
                        }
                    }
                }
            ],
            "policies": {
                "growth": {
                    "step_size": 0.001,
                    "max_iterations": 500
                }
            }
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        runner = DesignSpecRunner(spec)
        
        component = runner._get_component("comp1")
        effective = runner._get_effective_policy_dict(component, "growth")
        
        assert effective["step_size"] == 0.002
        assert effective["max_iterations"] == 500
    
    def test_nested_override_merges_correctly(self):
        """Nested policy overrides should merge correctly."""
        from designspec.spec import DesignSpec
        from designspec.runner import DesignSpecRunner
        
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m"},
            "domains": {
                "main_domain": {
                                        "type": "cylinder",
                                        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                                        "radius": 0.004,
                                        "height": 0.003,
                }
            },
            "components": [
                {
                    "id": "comp1",
                    "domain_ref": "main_domain",
                    "ports": {"inlets": [], "outlets": []},
                    "build": {"type": "backend_network"},
                    "policy_overrides": {
                        "growth": {
                            "backend_params": {
                                "custom_param": 42
                            }
                        }
                    }
                }
            ],
            "policies": {
                "growth": {
                    "step_size": 0.001,
                    "backend_params": {
                        "existing_param": 10
                    }
                }
            }
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        runner = DesignSpecRunner(spec)
        
        component = runner._get_component("comp1")
        effective = runner._get_effective_policy_dict(component, "growth")
        
        assert effective["step_size"] == 0.001
        assert effective["backend_params"]["existing_param"] == 10
        assert effective["backend_params"]["custom_param"] == 42
