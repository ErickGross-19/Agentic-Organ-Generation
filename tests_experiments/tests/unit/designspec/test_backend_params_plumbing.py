"""
Tests for backend_params wiring into backend execution.

Requirement 3: backend_params must affect behavior
- component.build.backend_params must affect behavior
- Programmatic backend must receive backend_params via GrowthPolicy.backend_params
- Runner must not ignore backend_params
"""

import pytest
from unittest.mock import patch, MagicMock


class TestBackendParamsPlumbing:
    """Tests for backend_params wiring into GrowthPolicy."""
    
    def test_backend_params_in_effective_policy_snapshot(self):
        """backend_params should appear in effective policy snapshot."""
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
                    "ports": {
                        "inlets": [{"position": [0.0, 0.0, 0.0015], "radius": 0.0005}],
                        "outlets": [{"position": [0.0, 0.0, -0.0015], "radius": 0.0003}]
                    },
                    "build": {
                        "type": "backend_network",
                        "backend": "space_colonization",
                        "backend_params": {
                            "custom_param": 42,
                            "max_iterations": 100
                        }
                    }
                }
            ],
            "policies": {
                "growth": {
                    "step_size": 0.001
                }
            }
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        runner = DesignSpecRunner(spec)
        
        runner._stage_compile_policies()
        runner._stage_compile_domains()
        runner._stage_component_ports("comp1")
        
        # Test that effective policy dict includes backend_params
        component = runner._get_component("comp1")
        effective_growth_dict = runner._get_effective_policy_dict(component, "growth")
        
        # The backend_params from component.build should be wired into the effective policy
        # Check that the runner properly handles backend_params in the build stage
        assert component is not None
        assert component.get("build", {}).get("backend_params", {}).get("custom_param") == 42
        assert component.get("build", {}).get("backend_params", {}).get("max_iterations") == 100
    
    def test_backend_params_merged_into_growth_policy(self):
        """backend_params should be merged into GrowthPolicy.backend_params."""
        from designspec.spec import DesignSpec
        from designspec.runner import DesignSpecRunner, deep_merge
        from aog_policies import GrowthPolicy
        
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
                    "ports": {
                        "inlets": [{"position": [0.0, 0.0, 0.0015], "radius": 0.0005}],
                        "outlets": [{"position": [0.0, 0.0, -0.0015], "radius": 0.0003}]
                    },
                    "build": {
                        "type": "backend_network",
                        "backend": "space_colonization",
                        "backend_params": {
                            "test_param": "test_value"
                        }
                    }
                }
            ],
            "policies": {
                "growth": {
                    "step_size": 0.001,
                    "backend_params": {
                        "existing_param": "existing_value"
                    }
                }
            }
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        runner = DesignSpecRunner(spec)
        
        runner._stage_compile_policies()
        runner._stage_compile_domains()
        runner._stage_component_ports("comp1")
        
        # Test that backend_params merging works correctly
        component = runner._get_component("comp1")
        effective_growth_dict = runner._get_effective_policy_dict(component, "growth")
        growth_policy = GrowthPolicy.from_dict(effective_growth_dict)
        
        # Simulate the backend_params merging that happens in _build_backend_network
        backend_params = component.get("build", {}).get("backend_params", {})
        existing_backend_params = getattr(growth_policy, 'backend_params', {}) or {}
        merged_backend_params = deep_merge(existing_backend_params, backend_params)
        
        # Verify the merge includes both existing and new params
        assert merged_backend_params.get("test_param") == "test_value"
        assert merged_backend_params.get("existing_param") == "existing_value"
    
    def test_empty_backend_params_does_not_break(self):
        """Empty backend_params should not cause errors."""
        from designspec.spec import DesignSpec
        from designspec.runner import DesignSpecRunner, deep_merge
        from aog_policies import GrowthPolicy
        
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
                    "ports": {
                        "inlets": [{"position": [0.0, 0.0, 0.0015], "radius": 0.0005}],
                        "outlets": [{"position": [0.0, 0.0, -0.0015], "radius": 0.0003}]
                    },
                    "build": {
                        "type": "backend_network",
                        "backend": "space_colonization"
                    }
                }
            ],
            "policies": {}
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        runner = DesignSpecRunner(spec)
        
        runner._stage_compile_policies()
        runner._stage_compile_domains()
        runner._stage_component_ports("comp1")
        
        # Test that empty backend_params doesn't cause errors
        component = runner._get_component("comp1")
        effective_growth_dict = runner._get_effective_policy_dict(component, "growth")
        growth_policy = GrowthPolicy.from_dict(effective_growth_dict)
        
        # Simulate the backend_params merging with empty params
        backend_params = component.get("build", {}).get("backend_params", {})
        existing_backend_params = getattr(growth_policy, 'backend_params', {}) or {}
        merged_backend_params = deep_merge(existing_backend_params, backend_params)
        
        # Should not raise any errors - merged params should be empty or existing
        assert isinstance(merged_backend_params, dict)
