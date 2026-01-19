"""Tests for DesignSpec loading and normalization."""

import pytest
import json
from pathlib import Path

from designspec.spec import (
    DesignSpec,
    DesignSpecError,
    DesignSpecValidationError,
    POLICY_LENGTH_FIELDS,
    DOMAIN_LENGTH_FIELDS,
    _get_unit_scale,
    _normalize_policy_to_meters,
    _normalize_domain_to_meters,
)


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "designspec"


class TestUnitScale:
    """Tests for unit scale conversion."""
    
    def test_meters_scale_is_one(self):
        assert _get_unit_scale("m") == 1.0
    
    def test_millimeters_scale(self):
        assert _get_unit_scale("mm") == 0.001
    
    def test_micrometers_scale(self):
        assert _get_unit_scale("um") == 1e-6
    
    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown unit"):
            _get_unit_scale("unknown")


class TestNormalizePolicyToMeters:
    """Tests for policy normalization."""
    
    def test_resolution_policy_normalized(self):
        policy_dict = {
            "min_channel_diameter": 0.02,
            "max_voxels": 5000000,
        }
        result = _normalize_policy_to_meters("resolution", policy_dict, 0.001)
        
        assert result["min_channel_diameter"] == pytest.approx(0.00002)
        assert result["max_voxels"] == 5000000
    
    def test_pathfinding_policy_normalized(self):
        policy_dict = {
            "clearance": 0.2,
            "local_radius": 0.2,
            "max_voxels_coarse": 2000000,
        }
        result = _normalize_policy_to_meters("pathfinding", policy_dict, 0.001)
        
        assert result["clearance"] == pytest.approx(0.0002)
        assert result["local_radius"] == pytest.approx(0.0002)
        assert result["max_voxels_coarse"] == 2000000
    
    def test_unknown_policy_passes_through(self):
        policy_dict = {"some_field": 123}
        result = _normalize_policy_to_meters("unknown_policy", policy_dict, 0.001)
        
        assert result["some_field"] == 123


class TestNormalizeDomainToMeters:
    """Tests for domain normalization."""
    
    def test_box_domain_normalized(self):
        domain_dict = {
            "type": "box",
            "x_min": -15,
            "x_max": 15,
            "y_min": -15,
            "y_max": 15,
            "z_min": -10,
            "z_max": 10,
        }
        result = _normalize_domain_to_meters(domain_dict, 0.001)
        
        assert result["x_min"] == pytest.approx(-0.015)
        assert result["x_max"] == pytest.approx(0.015)
        assert result["z_min"] == pytest.approx(-0.010)
        assert result["z_max"] == pytest.approx(0.010)
    
    def test_cylinder_domain_normalized(self):
        domain_dict = {
            "type": "cylinder",
            "center": [0, 0, 0],
            "radius": 10,
            "height": 20,
        }
        result = _normalize_domain_to_meters(domain_dict, 0.001)
        
        assert result["radius"] == pytest.approx(0.010)
        assert result["height"] == pytest.approx(0.020)
        assert result["center"] == [pytest.approx(0), pytest.approx(0), pytest.approx(0)]


class TestDesignSpecFromDict:
    """Tests for DesignSpec.from_dict()."""
    
    def test_minimal_valid_spec(self):
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m"},
            "policies": {},
            "domains": {"main": {"type": "box", "x_min": -1, "x_max": 1, "y_min": -1, "y_max": 1, "z_min": -1, "z_max": 1}},
            "components": [],
        }
        spec = DesignSpec.from_dict(spec_dict)
        
        assert spec.seed == 42
        assert spec.spec_hash is not None
        assert len(spec.spec_hash) > 0
    
    def test_missing_schema_raises(self):
        spec_dict = {
            "meta": {"seed": 42},
            "policies": {},
            "domains": {},
            "components": [],
        }
        with pytest.raises(DesignSpecValidationError, match="schema"):
            DesignSpec.from_dict(spec_dict)
    
    def test_missing_meta_raises(self):
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "policies": {},
            "domains": {},
            "components": [],
        }
        with pytest.raises(DesignSpecValidationError, match="meta"):
            DesignSpec.from_dict(spec_dict)
    
    def test_mm_units_normalized_to_meters(self):
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "mm"},
            "policies": {
                "resolution": {"min_channel_diameter": 0.02}
            },
            "domains": {
                "main": {"type": "box", "x_min": -15, "x_max": 15, "y_min": -15, "y_max": 15, "z_min": -10, "z_max": 10}
            },
            "components": [],
        }
        spec = DesignSpec.from_dict(spec_dict)
        
        assert spec.domains["main"]["x_min"] == pytest.approx(-0.015)
        assert spec.policies["resolution"]["min_channel_diameter"] == pytest.approx(0.00002)
    
    def test_warnings_recorded_for_aliases(self):
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m"},
            "policies": {
                "resolution": {"voxels_across_min_diameter": 8}
            },
            "domains": {"main": {"type": "box", "x_min": -1, "x_max": 1, "y_min": -1, "y_max": 1, "z_min": -1, "z_max": 1}},
            "components": [],
        }
        spec = DesignSpec.from_dict(spec_dict)
        
        assert any("voxels_across_min_diameter" in w for w in spec.warnings)


class TestDesignSpecFromJson:
    """Tests for DesignSpec.from_json()."""
    
    def test_load_golden_example(self):
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        assert spec.meta["name"] == "golden_example_v1"
        assert spec.seed == 1234
        assert len(spec.components) == 2
    
    def test_golden_example_domains_normalized(self):
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        main_domain = spec.domains["main_domain"]
        assert main_domain["x_min"] == pytest.approx(-0.015)
        assert main_domain["x_max"] == pytest.approx(0.015)
    
    def test_golden_example_port_radii_normalized(self):
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        net_1 = spec.components[0]
        inlet = net_1["ports"]["inlets"][0]
        assert inlet["radius"] == pytest.approx(0.0003)


class TestDesignSpecProperties:
    """Tests for DesignSpec property accessors."""
    
    def test_seed_property(self):
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 12345, "input_units": "m"},
            "policies": {},
            "domains": {"main": {"type": "box", "x_min": -1, "x_max": 1, "y_min": -1, "y_max": 1, "z_min": -1, "z_max": 1}},
            "components": [],
        }
        spec = DesignSpec.from_dict(spec_dict)
        
        assert spec.seed == 12345
    
    def test_input_units_property(self):
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "mm"},
            "policies": {},
            "domains": {"main": {"type": "box", "x_min": -1, "x_max": 1, "y_min": -1, "y_max": 1, "z_min": -1, "z_max": 1}},
            "components": [],
        }
        spec = DesignSpec.from_dict(spec_dict)
        
        assert spec.input_units == "mm"


class TestPolicyLengthFieldsRegistry:
    """Tests for the policy length fields registry."""
    
    def test_resolution_has_length_fields(self):
        assert "resolution" in POLICY_LENGTH_FIELDS
        assert "min_channel_diameter" in POLICY_LENGTH_FIELDS["resolution"]
    
    def test_pathfinding_has_length_fields(self):
        assert "pathfinding" in POLICY_LENGTH_FIELDS
        assert "clearance" in POLICY_LENGTH_FIELDS["pathfinding"]
    
    def test_embedding_has_length_fields(self):
        assert "embedding" in POLICY_LENGTH_FIELDS
        assert "shell_thickness" in POLICY_LENGTH_FIELDS["embedding"]


class TestBackendParamsNormalization:
    """Tests for backend_params unit conversion in growth policy.
    
    Regression tests for the issue where wall_margin_m in backend_params
    was not being converted from mm to meters, causing the k-ary tree
    algorithm to be too restrictive.
    """
    
    def test_growth_policy_backend_params_wall_margin_normalized(self):
        """wall_margin_m in growth.backend_params should be converted from mm to m."""
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "mm"},
            "policies": {
                "growth": {
                    "backend_params": {
                        "wall_margin_m": 0.5,  # 0.5mm
                        "K": 2,  # Not a length field, should not be converted
                    }
                }
            },
            "domains": {"main": {"type": "box", "x_min": -1, "x_max": 1, "y_min": -1, "y_max": 1, "z_min": -1, "z_max": 1}},
            "components": [],
        }
        spec = DesignSpec.from_dict(spec_dict)
        
        # wall_margin_m should be converted from 0.5mm to 0.0005m
        assert spec.policies["growth"]["backend_params"]["wall_margin_m"] == pytest.approx(0.0005)
        # K should remain unchanged (not a length field)
        assert spec.policies["growth"]["backend_params"]["K"] == 2
    
    def test_component_build_backend_params_wall_margin_normalized(self):
        """wall_margin_m in component.build.backend_params should be converted from mm to m."""
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "mm"},
            "policies": {},
            "domains": {"main": {"type": "box", "x_min": -1, "x_max": 1, "y_min": -1, "y_max": 1, "z_min": -1, "z_max": 1}},
            "components": [
                {
                    "id": "tree_1",
                    "domain_ref": "main",
                    "ports": {"inlets": [], "outlets": []},
                    "build": {
                        "type": "backend_network",
                        "backend_params": {
                            "wall_margin_m": 0.5,  # 0.5mm
                            "terminal_radius": 0.1,  # 0.1mm
                            "num_levels": 7,  # Not a length field
                        }
                    }
                }
            ],
        }
        spec = DesignSpec.from_dict(spec_dict)
        
        backend_params = spec.components[0]["build"]["backend_params"]
        # wall_margin_m should be converted from 0.5mm to 0.0005m
        assert backend_params["wall_margin_m"] == pytest.approx(0.0005)
        # terminal_radius should be converted from 0.1mm to 0.0001m
        assert backend_params["terminal_radius"] == pytest.approx(0.0001)
        # num_levels should remain unchanged (not a length field)
        assert backend_params["num_levels"] == 7
