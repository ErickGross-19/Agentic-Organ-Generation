"""
Tests for domain selection correctness for embedding/validity.

Requirement 5: Fix domain selection correctness
- Runner must not choose "first domain in dict" for validity and embedding
- Explicit rule:
  - if spec.embedding.domain_ref exists, use it
  - else if single domain, use it
  - else error with clear message
"""

import pytest


class TestDomainRefSelection:
    """Tests for explicit domain selection rule."""
    
    def test_single_domain_selected_automatically(self):
        """Single domain should be selected automatically."""
        from designspec.spec import DesignSpec
        from designspec.runner import DesignSpecRunner
        
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m"},
            "domains": {
                "only_domain": {
                                        "type": "cylinder",
                                        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                                        "radius": 0.004,
                                        "height": 0.003,
                }
            },
            "components": [
                {
                    "id": "comp1",
                    "domain_ref": "only_domain",
                    "ports": {"inlets": [], "outlets": []},
                    "build": {"type": "backend_network"}
                }
            ],
            "policies": {}
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        runner = DesignSpecRunner(spec)
        
        runner._stage_compile_domains()
        
        domain_name, domain, error = runner._select_domain_for_embedding()
        
        assert error is None
        assert domain_name == "only_domain"
        assert domain is not None
    
    def test_explicit_domain_ref_used(self):
        """Explicit embedding.domain_ref should be used."""
        from designspec.spec import DesignSpec
        from designspec.runner import DesignSpecRunner
        
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m"},
            "domains": {
                "domain_a": {
                                        "type": "cylinder",
                                        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                                        "radius": 0.004,
                                        "height": 0.003,
                },
                "domain_b": {
                                        "type": "cylinder",
                                        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                                        "radius": 0.005,
                                        "height": 0.004,
                }
            },
            "components": [
                {
                    "id": "comp1",
                    "domain_ref": "domain_a",
                    "ports": {"inlets": [], "outlets": []},
                    "build": {"type": "backend_network"}
                }
            ],
            "embedding": {
                "domain_ref": "domain_b"
            },
            "policies": {}
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        runner = DesignSpecRunner(spec)
        
        runner._stage_compile_domains()
        
        domain_name, domain, error = runner._select_domain_for_embedding()
        
        assert error is None
        assert domain_name == "domain_b"
    
    def test_multi_domain_without_domain_ref_fails(self):
        """Multiple domains without embedding.domain_ref should fail with clear error."""
        from designspec.spec import DesignSpec
        from designspec.runner import DesignSpecRunner
        
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m"},
            "domains": {
                "domain_a": {
                                        "type": "cylinder",
                                        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                                        "radius": 0.004,
                                        "height": 0.003,
                },
                "domain_b": {
                                        "type": "cylinder",
                                        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                                        "radius": 0.005,
                                        "height": 0.004,
                }
            },
            "components": [
                {
                    "id": "comp1",
                    "domain_ref": "domain_a",
                    "ports": {"inlets": [], "outlets": []},
                    "build": {"type": "backend_network"}
                }
            ],
            "policies": {}
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        runner = DesignSpecRunner(spec)
        
        runner._stage_compile_domains()
        
        domain_name, domain, error = runner._select_domain_for_embedding()
        
        assert error is not None
        assert "multiple domains" in error.lower()
        assert "domain_ref" in error.lower()
    
    def test_invalid_domain_ref_fails(self):
        """Invalid embedding.domain_ref should fail with clear error."""
        from designspec.spec import DesignSpec
        from designspec.runner import DesignSpecRunner
        
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m"},
            "domains": {
                "domain_a": {
                                        "type": "cylinder",
                                        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                                        "radius": 0.004,
                                        "height": 0.003,
                }
            },
            "components": [
                {
                    "id": "comp1",
                    "domain_ref": "domain_a",
                    "ports": {"inlets": [], "outlets": []},
                    "build": {"type": "backend_network"}
                }
            ],
            "embedding": {
                "domain_ref": "nonexistent_domain"
            },
            "policies": {}
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        runner = DesignSpecRunner(spec)
        
        runner._stage_compile_domains()
        
        domain_name, domain, error = runner._select_domain_for_embedding()
        
        assert error is not None
        assert "not found" in error.lower()
    
    def test_mesh_domain_stage_uses_selection_rule(self):
        """_stage_mesh_domain should use the domain selection rule."""
        from designspec.spec import DesignSpec
        from designspec.runner import DesignSpecRunner
        
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m"},
            "domains": {
                "domain_a": {
                                        "type": "cylinder",
                                        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                                        "radius": 0.004,
                                        "height": 0.003,
                },
                "domain_b": {
                                        "type": "cylinder",
                                        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                                        "radius": 0.005,
                                        "height": 0.004,
                }
            },
            "components": [
                {
                    "id": "comp1",
                    "domain_ref": "domain_a",
                    "ports": {"inlets": [], "outlets": []},
                    "build": {"type": "backend_network"}
                }
            ],
            "policies": {}
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        runner = DesignSpecRunner(spec)
        
        runner._stage_compile_domains()
        
        report = runner._stage_mesh_domain()
        
        assert not report.success
        assert any("multiple domains" in err.lower() for err in report.errors)
    
    def test_embed_stage_uses_selection_rule(self):
        """_stage_embed should use the domain selection rule."""
        from designspec.spec import DesignSpec
        from designspec.runner import DesignSpecRunner
        from unittest.mock import MagicMock
        
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m"},
            "domains": {
                "domain_a": {
                                        "type": "cylinder",
                                        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                                        "radius": 0.004,
                                        "height": 0.003,
                },
                "domain_b": {
                                        "type": "cylinder",
                                        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                                        "radius": 0.005,
                                        "height": 0.004,
                }
            },
            "components": [
                {
                    "id": "comp1",
                    "domain_ref": "domain_a",
                    "ports": {"inlets": [], "outlets": []},
                    "build": {"type": "backend_network"}
                }
            ],
            "policies": {}
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        runner = DesignSpecRunner(spec)
        
        runner._stage_compile_domains()
        runner._union_void = MagicMock()
        
        report = runner._stage_embed()
        
        assert not report.success
        assert any("multiple domains" in err.lower() for err in report.errors)
    
    def test_domain_name_in_mesh_domain_metadata(self):
        """_stage_mesh_domain should include domain_name in metadata."""
        from designspec.spec import DesignSpec
        from designspec.runner import DesignSpecRunner
        
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m"},
            "domains": {
                "my_domain": {
                                        "type": "cylinder",
                                        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
                                        "radius": 0.004,
                                        "height": 0.003,
                }
            },
            "components": [
                {
                    "id": "comp1",
                    "domain_ref": "my_domain",
                    "ports": {"inlets": [], "outlets": []},
                    "build": {"type": "backend_network"}
                }
            ],
            "policies": {}
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        runner = DesignSpecRunner(spec)
        
        runner._stage_compile_domains()
        
        report = runner._stage_mesh_domain()
        
        # The stage may fail if domain doesn't support to_mesh, but
        # domain_name should still be in metadata
        assert report.metadata.get("domain_name") == "my_domain"
