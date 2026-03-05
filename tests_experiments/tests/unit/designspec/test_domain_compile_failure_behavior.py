"""
Tests for domain compile failure behavior.

Requirement 6: Remove silent dict fallback for domains
- If domain compilation fails, runner must fail loudly (success=False) unless spec explicitly opts into fallback mode
- Add spec/meta flag: meta.allow_domain_compile_fallback = false by default
- If fallback is allowed, report must include warning and downstream stages must check type
"""

import pytest
from unittest.mock import patch, MagicMock


class TestDomainCompileFailureBehavior:
    """Tests for domain compile failure behavior."""
    
    def test_domain_compile_failure_fails_loudly_by_default(self):
        """Domain compile failure should fail loudly by default."""
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
        
        with patch('generation.core.domain.domain_from_dict', side_effect=ValueError("Test error")):
            report = runner._stage_compile_domains()
        
        assert not report.success
        assert len(report.errors) > 0
        assert any("test error" in err.lower() for err in report.errors)
    
    def test_domain_compile_failure_with_fallback_allowed(self):
        """Domain compile failure with allow_domain_compile_fallback=true should succeed with warning."""
        from designspec.spec import DesignSpec
        from designspec.runner import DesignSpecRunner
        
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m", "allow_domain_compile_fallback": True},
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
        
        with patch('generation.core.domain.domain_from_dict', side_effect=ValueError("Test error")):
            report = runner._stage_compile_domains()
        
        assert report.success
        assert len(report.warnings) > 0
        assert any("fallback" in warn.lower() for warn in report.warnings)
    
    def test_fallback_domain_is_dict(self):
        """When fallback is allowed, failed domain should be stored as dict."""
        from designspec.spec import DesignSpec
        from designspec.runner import DesignSpecRunner
        
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m", "allow_domain_compile_fallback": True},
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
        
        with patch('generation.core.domain.domain_from_dict', side_effect=ValueError("Test error")):
            report = runner._stage_compile_domains()
        
        assert report.success
        assert "main_domain" in runner._compiled_domains
        assert isinstance(runner._compiled_domains["main_domain"], dict)
    
    def test_downstream_stage_fails_on_dict_domain(self):
        """Downstream stages should fail when domain is a dict (not compiled)."""
        from designspec.spec import DesignSpec
        from designspec.runner import DesignSpecRunner
        
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m", "allow_domain_compile_fallback": True},
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
                    "build": {"type": "backend_network"}
                }
            ],
            "policies": {}
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        runner = DesignSpecRunner(spec)
        
        with patch('generation.core.domain.domain_from_dict', side_effect=ValueError("Test error")):
            runner._stage_compile_domains()
        
        report = runner._stage_component_ports("comp1")
        
        assert not report.success
        assert any("not compiled" in err.lower() for err in report.errors)
    
    def test_metadata_shows_fallback_status(self):
        """Metadata should show fallback status for each domain."""
        from designspec.spec import DesignSpec
        from designspec.runner import DesignSpecRunner
        
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m", "allow_domain_compile_fallback": True},
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
        
        with patch('generation.core.domain.domain_from_dict', side_effect=ValueError("Test error")):
            report = runner._stage_compile_domains()
        
        assert "main_domain" in report.metadata
        assert "fallback" in report.metadata["main_domain"].lower()
    
    def test_successful_compile_shows_compiled_status(self):
        """Successful domain compile should show 'compiled' in metadata."""
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
        
        report = runner._stage_compile_domains()
        
        assert report.success
        assert "main_domain" in report.metadata
        assert report.metadata["main_domain"] == "compiled"
    
    def test_mesh_domain_fails_on_dict_domain(self):
        """_stage_mesh_domain should fail when domain is a dict."""
        from designspec.spec import DesignSpec
        from designspec.runner import DesignSpecRunner
        
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m", "allow_domain_compile_fallback": True},
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
        
        with patch('generation.core.domain.domain_from_dict', side_effect=ValueError("Test error")):
            runner._stage_compile_domains()
        
        report = runner._stage_mesh_domain()
        
        assert not report.success
        assert any("not compiled" in err.lower() for err in report.errors)
    
    def test_build_stage_fails_on_dict_domain(self):
        """_stage_component_build should fail when domain is a dict."""
        from designspec.spec import DesignSpec
        from designspec.runner import DesignSpecRunner
        
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m", "allow_domain_compile_fallback": True},
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
                    "build": {"type": "backend_network"}
                }
            ],
            "policies": {}
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        runner = DesignSpecRunner(spec)
        
        with patch('generation.core.domain.domain_from_dict', side_effect=ValueError("Test error")):
            runner._stage_compile_domains()
        
        report = runner._stage_component_build("comp1")
        
        assert not report.success
        assert any("not compiled" in err.lower() for err in report.errors)
