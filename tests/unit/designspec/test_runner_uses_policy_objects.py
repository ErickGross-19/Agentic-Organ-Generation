"""
Tests for runner using aog_policies-derived policy objects.

Requirement 4: Use aog_policies-derived policies for mesh synthesis and domain meshing
- Runner must not construct generation.policies defaults directly
- Mesh synthesis policy must come from aog_policies
- Domain meshing must respect DomainMeshingPolicy + ResolutionPolicy budgets
"""

import pytest
from unittest.mock import patch, MagicMock


class TestRunnerUsesPolicyObjects:
    """Tests for runner using aog_policies-derived policy objects."""
    
    def test_mesh_synthesis_uses_aog_policies(self):
        """Mesh synthesis should use aog_policies.MeshSynthesisPolicy."""
        from designspec.spec import DesignSpec
        from designspec.runner import DesignSpecRunner
        from aog_policies import MeshSynthesisPolicy
        
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
                    "build": {"type": "backend_network"}
                }
            ],
            "policies": {
                "mesh_synthesis": {
                    "resolution": 0.0001,
                    "smoothing_iterations": 3
                }
            }
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        runner = DesignSpecRunner(spec)
        
        runner._stage_compile_policies()
        runner._stage_compile_domains()
        runner._stage_component_ports("comp1")
        
        # Test that MeshSynthesisPolicy can be compiled from spec policies
        component = runner._get_component("comp1")
        effective_mesh_dict = runner._get_effective_policy_dict(component, "mesh_synthesis")
        
        # Verify the policy dict contains expected values
        assert effective_mesh_dict.get("resolution") == 0.0001
        assert effective_mesh_dict.get("smoothing_iterations") == 3
        
        # Verify MeshSynthesisPolicy.from_dict works with the effective dict
        mesh_policy = MeshSynthesisPolicy.from_dict(effective_mesh_dict)
        assert mesh_policy is not None
    
    def test_effective_mesh_policy_in_report(self):
        """Effective mesh policy should include component overrides merged with global."""
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
                    "build": {"type": "backend_network"},
                    "policy_overrides": {
                        "mesh_synthesis": {
                            "resolution": 0.00005
                        }
                    }
                }
            ],
            "policies": {
                "mesh_synthesis": {
                    "resolution": 0.0001,
                    "smoothing_iterations": 3
                }
            }
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        runner = DesignSpecRunner(spec)
        
        runner._stage_compile_policies()
        runner._stage_compile_domains()
        runner._stage_component_ports("comp1")
        
        # Test that effective policy dict includes component overrides
        component = runner._get_component("comp1")
        effective_mesh_dict = runner._get_effective_policy_dict(component, "mesh_synthesis")
        
        # Component override should take precedence
        assert effective_mesh_dict.get("resolution") == 0.00005
        # Global policy should be preserved for non-overridden fields
        assert effective_mesh_dict.get("smoothing_iterations") == 3
    
    def test_component_override_applied_to_mesh_policy(self):
        """Component policy overrides should be applied to mesh synthesis policy."""
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
                        "mesh_synthesis": {
                            "custom_override": "value"
                        }
                    }
                }
            ],
            "policies": {
                "mesh_synthesis": {
                    "base_param": "base_value"
                }
            }
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        runner = DesignSpecRunner(spec)
        
        component = runner._get_component("comp1")
        effective = runner._get_effective_policy_dict(component, "mesh_synthesis")
        
        assert effective.get("base_param") == "base_value"
        assert effective.get("custom_override") == "value"
