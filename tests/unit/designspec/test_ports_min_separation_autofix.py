"""
Tests for port min separation validation and autofix.

Requirement 2 (continued): Port validation with min separation
- Two ports too close should generate warning + adjusted positions OR failure if policy says strict
"""

import pytest
from unittest.mock import patch, MagicMock


class TestPortsMinSeparationAutofix:
    """Tests for port min separation validation and autofix."""
    
    def test_ports_with_sufficient_separation_no_warning(self):
        """Ports with sufficient separation should not generate warnings."""
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
                        "inlets": [
                            {"position": [0.0, 0.0, 0.0015], "radius": 0.0005},
                            {"position": [0.002, 0.0, 0.0015], "radius": 0.0005}
                        ],
                        "outlets": []
                    },
                    "build": {"type": "backend_network"}
                }
            ],
            "policies": {
                "ports": {
                    "min_separation": 0.001
                }
            }
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        runner = DesignSpecRunner(spec)
        
        runner._stage_compile_domains()
        
        report = runner._stage_component_ports("comp1")
        
        assert report.success
        separation_warnings = [w for w in report.warnings if "separation" in w.lower()]
        assert len(separation_warnings) == 0
    
    def test_resolved_ports_stored_in_context(self):
        """Resolved ports should be stored in runner context."""
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
                        "inlets": [
                            {"position": [0.0, 0.0, 0.0015], "radius": 0.0005}
                        ],
                        "outlets": [
                            {"position": [0.0, 0.0, -0.0015], "radius": 0.0003}
                        ]
                    },
                    "build": {"type": "backend_network"}
                }
            ],
            "policies": {}
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        runner = DesignSpecRunner(spec)
        
        runner._stage_compile_domains()
        
        assert "comp1" not in runner._resolved_ports
        
        report = runner._stage_component_ports("comp1")
        
        assert report.success
        assert "comp1" in runner._resolved_ports
        
        resolved = runner._resolved_ports["comp1"]
        assert "inlets" in resolved
        assert "outlets" in resolved
        assert len(resolved["inlets"]) == 1
        assert len(resolved["outlets"]) == 1
    
    def test_build_stage_uses_resolved_ports(self):
        """Build stage should use resolved ports from context."""
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
                        "inlets": [
                            {"position": [0.0, 0.0, 0.0015], "radius": 0.0005}
                        ],
                        "outlets": [
                            {"position": [0.0, 0.0, -0.0015], "radius": 0.0003}
                        ]
                    },
                    "build": {"type": "backend_network", "backend": "space_colonization"}
                }
            ],
            "policies": {}
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        runner = DesignSpecRunner(spec)
        
        runner._stage_compile_domains()
        runner._stage_component_ports("comp1")
        
        assert "comp1" in runner._resolved_ports
        
        resolved_ports = runner._resolved_ports["comp1"]
        assert resolved_ports["inlets"][0]["resolved"] is True
