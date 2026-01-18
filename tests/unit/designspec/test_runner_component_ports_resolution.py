"""
Tests for component_ports stage real port resolution.

Requirement 2: component_ports must resolve ports domain-aware
- Uses PortPlacementPolicy + ridge constraints + face rules
- Outputs resolved ports structure to RunnerContext
- Supports layout on reference plane, clamp to face region, etc.
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np


class TestComponentPortsResolution:
    """Tests for _stage_component_ports real port resolution."""
    
    def test_explicit_positions_marked_as_resolved(self):
        """Ports with explicit positions should be marked as resolved."""
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
                            {"position": [0.001, 0.001, -0.0015], "radius": 0.0003}
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
        
        report = runner._stage_component_ports("comp1")
        
        assert report.success
        assert "comp1" in runner._resolved_ports
        
        resolved = runner._resolved_ports["comp1"]
        assert len(resolved["inlets"]) == 1
        assert len(resolved["outlets"]) == 1
        
        assert resolved["inlets"][0]["resolved"] is True
        assert resolved["inlets"][0]["resolution_method"] == "explicit"
        assert resolved["outlets"][0]["resolved"] is True
        assert resolved["outlets"][0]["resolution_method"] == "explicit"
    
    def test_port_counts_in_metadata(self):
        """Stage report should include port counts in metadata."""
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
                            {"position": [0.001, 0.0, 0.0015], "radius": 0.0005}
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
        
        report = runner._stage_component_ports("comp1")
        
        assert report.success
        assert report.metadata["inlet_count"] == 2
        assert report.metadata["outlet_count"] == 1
        assert report.metadata["resolved_inlet_count"] == 2
        assert report.metadata["resolved_outlet_count"] == 1
    
    def test_domain_ref_in_metadata(self):
        """Stage report should include domain_ref in metadata."""
        from designspec.spec import DesignSpec
        from designspec.runner import DesignSpecRunner
        
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m"},
            "domains": {
                "my_custom_domain": {
                                        "type": "cylinder",
                                        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                                        "radius": 0.004,
                                        "height": 0.003,
                }
            },
            "components": [
                {
                    "id": "comp1",
                    "domain_ref": "my_custom_domain",
                    "ports": {
                        "inlets": [{"position": [0.0, 0.0, 0.0015], "radius": 0.0005}],
                        "outlets": []
                    },
                    "build": {"type": "backend_network"}
                }
            ],
            "policies": {}
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        runner = DesignSpecRunner(spec)
        
        runner._stage_compile_domains()
        
        report = runner._stage_component_ports("comp1")
        
        assert report.success
        assert report.metadata["domain_ref"] == "my_custom_domain"
    
    def test_effective_policy_in_report(self):
        """Stage report should include effective policy snapshot."""
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
                        "outlets": []
                    },
                    "build": {"type": "backend_network"},
                    "policy_overrides": {
                        "ports": {
                            "min_separation": 0.002
                        }
                    }
                }
            ],
            "policies": {
                "ports": {
                    "min_separation": 0.001,
                    "clamp_to_face": True
                }
            }
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        runner = DesignSpecRunner(spec)
        
        runner._stage_compile_domains()
        
        report = runner._stage_component_ports("comp1")
        
        assert report.success
        assert report.effective_policy is not None
        assert report.effective_policy.get("min_separation") == 0.002
        assert report.effective_policy.get("clamp_to_face") is True
    
    def test_fails_if_domain_not_found(self):
        """Stage should fail if domain_ref doesn't exist."""
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
                    "domain_ref": "nonexistent_domain",
                    "ports": {
                        "inlets": [{"position": [0.0, 0.0, 0.0015], "radius": 0.0005}],
                        "outlets": []
                    },
                    "build": {"type": "backend_network"}
                }
            ],
            "policies": {}
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        runner = DesignSpecRunner(spec)
        
        runner._stage_compile_domains()
        
        report = runner._stage_component_ports("comp1")
        
        assert not report.success
        assert any("not found" in err.lower() for err in report.errors)
    
    def test_fails_if_component_not_found(self):
        """Stage should fail if component doesn't exist."""
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
            "policies": {}
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        runner = DesignSpecRunner(spec)
        
        report = runner._stage_component_ports("nonexistent_component")
        
        assert not report.success
        assert any("not found" in err.lower() for err in report.errors)
