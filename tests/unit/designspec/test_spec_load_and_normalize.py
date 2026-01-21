"""Tests for DesignSpec loading and normalization."""

import pytest
import json
import os
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
from designspec.preflight import (
    fill_policy_defaults,
    run_preflight_checks,
    PreflightValidationError,
    DEFAULT_POLICIES,
    DEBUG_ENV_VAR,
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
    
    def test_scaffold_topdown_backend_params_normalized(self):
        """scaffold_topdown backend_params length fields should be converted from mm to m."""
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "mm"},
            "policies": {
                "growth": {
                    "backend": "scaffold_topdown",
                    "backend_params": {
                        "step_length": 2.0,  # 2.0mm
                        "spread": 1.5,  # 1.5mm
                        "wall_margin_m": 0.1,  # 0.1mm
                        "splits": 3,  # Not a length field
                        "levels": 5,  # Not a length field
                        "ratio": 0.78,  # Not a length field
                    }
                }
            },
            "domains": {"main": {"type": "box", "x_min": -5, "x_max": 5, "y_min": -5, "y_max": 5, "z_min": -3, "z_max": 3}},
            "components": [],
        }
        spec = DesignSpec.from_dict(spec_dict)
        
        backend_params = spec.policies["growth"]["backend_params"]
        # step_length should be converted from 2.0mm to 0.002m
        assert backend_params["step_length"] == pytest.approx(0.002)
        # spread should be converted from 1.5mm to 0.0015m
        assert backend_params["spread"] == pytest.approx(0.0015)
        # wall_margin_m should be converted from 0.1mm to 0.0001m
        assert backend_params["wall_margin_m"] == pytest.approx(0.0001)
        # Non-length fields should remain unchanged
        assert backend_params["splits"] == 3
        assert backend_params["levels"] == 5
        assert backend_params["ratio"] == 0.78
    
    def test_scaffold_topdown_collision_online_params_normalized(self):
        """scaffold_topdown collision_online nested dict length fields should be converted."""
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "mm"},
            "policies": {
                "growth": {
                    "backend": "scaffold_topdown",
                    "backend_params": {
                        "collision_online": {
                            "enabled": True,
                            "buffer_abs_m": 0.02,  # 0.02mm
                            "cell_size_m": 0.5,  # 0.5mm
                            "buffer_rel": 0.05,  # Not a length field (relative)
                            "rotation_attempts": 14,  # Not a length field
                        }
                    }
                }
            },
            "domains": {"main": {"type": "box", "x_min": -5, "x_max": 5, "y_min": -5, "y_max": 5, "z_min": -3, "z_max": 3}},
            "components": [],
        }
        spec = DesignSpec.from_dict(spec_dict)
        
        collision_online = spec.policies["growth"]["backend_params"]["collision_online"]
        # buffer_abs_m should be converted from 0.02mm to 0.00002m
        assert collision_online["buffer_abs_m"] == pytest.approx(0.00002)
        # cell_size_m should be converted from 0.5mm to 0.0005m
        assert collision_online["cell_size_m"] == pytest.approx(0.0005)
        # Non-length fields should remain unchanged
        assert collision_online["enabled"] is True
        assert collision_online["buffer_rel"] == 0.05
        assert collision_online["rotation_attempts"] == 14
    
    def test_scaffold_topdown_collision_postpass_params_normalized(self):
        """scaffold_topdown collision_postpass nested dict length fields should be converted."""
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "mm"},
            "policies": {
                "growth": {
                    "backend": "scaffold_topdown",
                    "backend_params": {
                        "collision_postpass": {
                            "enabled": True,
                            "min_clearance_m": 0.02,  # 0.02mm
                            "shrink_factor": 0.9,  # Not a length field
                            "shrink_max_iterations": 6,  # Not a length field
                        }
                    }
                }
            },
            "domains": {"main": {"type": "box", "x_min": -5, "x_max": 5, "y_min": -5, "y_max": 5, "z_min": -3, "z_max": 3}},
            "components": [],
        }
        spec = DesignSpec.from_dict(spec_dict)
        
        collision_postpass = spec.policies["growth"]["backend_params"]["collision_postpass"]
        # min_clearance_m should be converted from 0.02mm to 0.00002m
        assert collision_postpass["min_clearance_m"] == pytest.approx(0.00002)
        # Non-length fields should remain unchanged
        assert collision_postpass["enabled"] is True
        assert collision_postpass["shrink_factor"] == 0.9
        assert collision_postpass["shrink_max_iterations"] == 6


class TestNestedPolicyNormalization:
    """Tests for nested policy normalization.
    
    Regression tests for the issue where nested policies in composition
    (merge_policy, repair_policy, synthesis_policy) were not being normalized,
    causing mm values to be interpreted as meters.
    """
    
    def test_composition_merge_policy_voxel_pitch_normalized(self):
        """composition.merge_policy.voxel_pitch should be converted from mm to m."""
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "mm"},
            "policies": {
                "composition": {
                    "merge_policy": {
                        "voxel_pitch": 0.05,  # 0.05mm = 50um
                        "min_component_volume": 0.001,  # 0.001 mm³
                    }
                }
            },
            "domains": {"main": {"type": "box", "x_min": -1, "x_max": 1, "y_min": -1, "y_max": 1, "z_min": -1, "z_max": 1}},
            "components": [],
        }
        spec = DesignSpec.from_dict(spec_dict)
        
        merge_policy = spec.policies["composition"]["merge_policy"]
        # voxel_pitch should be converted from 0.05mm to 5e-5m
        assert merge_policy["voxel_pitch"] == pytest.approx(5e-5)
        # min_component_volume should be converted from 0.001mm³ to 1e-12m³ (scale³)
        assert merge_policy["min_component_volume"] == pytest.approx(1e-12)
    
    def test_composition_repair_policy_voxel_pitch_normalized(self):
        """composition.repair_policy.voxel_pitch should be converted from mm to m."""
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "mm"},
            "policies": {
                "composition": {
                    "repair_policy": {
                        "voxel_pitch": 0.1,  # 0.1mm
                        "min_component_volume": 0.001,  # 0.001 mm³
                    }
                }
            },
            "domains": {"main": {"type": "box", "x_min": -1, "x_max": 1, "y_min": -1, "y_max": 1, "z_min": -1, "z_max": 1}},
            "components": [],
        }
        spec = DesignSpec.from_dict(spec_dict)
        
        repair_policy = spec.policies["composition"]["repair_policy"]
        # voxel_pitch should be converted from 0.1mm to 1e-4m
        assert repair_policy["voxel_pitch"] == pytest.approx(1e-4)
        # min_component_volume should be converted from 0.001mm³ to 1e-12m³ (scale³)
        assert repair_policy["min_component_volume"] == pytest.approx(1e-12)
    
    def test_domain_meshing_sub_policies_normalized(self):
        """domain_meshing sub-policies should have their pitches normalized."""
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "mm"},
            "policies": {
                "domain_meshing": {
                    "voxel_pitch": 0.05,  # 0.05mm
                    "mesh_policy": {
                        "repair_voxel_pitch": 0.05,  # 0.05mm
                    },
                    "implicit_policy": {
                        "voxel_pitch": 0.05,  # 0.05mm
                    }
                }
            },
            "domains": {"main": {"type": "box", "x_min": -1, "x_max": 1, "y_min": -1, "y_max": 1, "z_min": -1, "z_max": 1}},
            "components": [],
        }
        spec = DesignSpec.from_dict(spec_dict)
        
        domain_meshing = spec.policies["domain_meshing"]
        # Top-level voxel_pitch should be converted
        assert domain_meshing["voxel_pitch"] == pytest.approx(5e-5)
        # mesh_policy.repair_voxel_pitch should be converted
        assert domain_meshing["mesh_policy"]["repair_voxel_pitch"] == pytest.approx(5e-5)
        # implicit_policy.voxel_pitch should be converted
        assert domain_meshing["implicit_policy"]["voxel_pitch"] == pytest.approx(5e-5)
    
    def test_malaria_venule_spec_normalization(self):
        """
        Regression test: malaria venule spec with input_units=mm should normalize correctly.
        
        This test verifies that:
        - Nested merge voxel pitch is ~5e-5m (from 0.05mm)
        - Domain radius is 0.005m (from 5mm)
        """
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "mm"},
            "policies": {
                "composition": {
                    "merge_policy": {
                        "voxel_pitch": 0.05,  # 0.05mm = 50um
                    }
                }
            },
            "domains": {
                "cylinder_domain": {
                    "type": "cylinder",
                    "center": [0, 0, 0],
                    "radius": 5.0,  # 5mm
                    "height": 2.0,  # 2mm
                }
            },
            "components": [],
        }
        spec = DesignSpec.from_dict(spec_dict)
        
        # Assert normalized merge voxel pitch is ~5e-5m
        merge_policy = spec.policies["composition"]["merge_policy"]
        assert merge_policy["voxel_pitch"] == pytest.approx(5e-5)
        
        # Assert normalized domain radius is 0.005m
        domain = spec.domains["cylinder_domain"]
        assert domain["radius"] == pytest.approx(0.005)
        assert domain["height"] == pytest.approx(0.002)
    
    def test_volume_fields_use_cubic_scale(self):
        """Volume fields should be scaled by scale³, not scale."""
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "mm"},
            "policies": {
                "repair": {
                    "min_component_volume": 1.0,  # 1 mm³
                },
                "composition": {
                    "min_component_volume": 1.0,  # 1 mm³
                },
                "mesh_merge": {
                    "min_component_volume": 1.0,  # 1 mm³
                }
            },
            "domains": {"main": {"type": "box", "x_min": -1, "x_max": 1, "y_min": -1, "y_max": 1, "z_min": -1, "z_max": 1}},
            "components": [],
        }
        spec = DesignSpec.from_dict(spec_dict)
        
        # 1 mm³ = 1e-9 m³ (scale³ = 1e-3³ = 1e-9)
        assert spec.policies["repair"]["min_component_volume"] == pytest.approx(1e-9)
        assert spec.policies["composition"]["min_component_volume"] == pytest.approx(1e-9)
        assert spec.policies["mesh_merge"]["min_component_volume"] == pytest.approx(1e-9)


class TestCentimeterUnits:
    """Tests for centimeter unit support."""
    
    def test_centimeters_scale(self):
        """cm units should have scale factor of 0.01."""
        assert _get_unit_scale("cm") == 0.01
    
    def test_cm_units_normalized_to_meters(self):
        """Spec with cm units should normalize correctly."""
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "cm"},
            "policies": {
                "resolution": {"min_channel_diameter": 0.02}  # 0.02cm = 0.2mm
            },
            "domains": {
                "main": {"type": "cylinder", "center": [0, 0, 0], "radius": 0.5, "height": 0.2}  # 5mm radius, 2mm height
            },
            "components": [],
        }
        spec = DesignSpec.from_dict(spec_dict)
        
        assert spec.domains["main"]["radius"] == pytest.approx(0.005)  # 0.5cm = 0.005m
        assert spec.domains["main"]["height"] == pytest.approx(0.002)  # 0.2cm = 0.002m
        assert spec.policies["resolution"]["min_channel_diameter"] == pytest.approx(0.0002)  # 0.02cm = 0.0002m


class TestPreflightValidation:
    """Tests for preflight validation checks."""
    
    def test_preflight_detects_unnormalized_pitch(self):
        """Preflight should detect voxel_pitch values that appear unnormalized."""
        normalized_spec = {
            "meta": {"seed": 42, "input_units": "m"},
            "policies": {
                "embedding": {
                    "voxel_pitch": 0.05,  # 50mm - way too large, likely not normalized
                }
            },
            "domains": {
                "main": {"type": "cylinder", "center": [0, 0, 0], "radius": 0.005, "height": 0.002}
            },
            "components": [],
        }
        result = run_preflight_checks(normalized_spec)
        
        assert not result.success
        assert any("PITCH_NOT_NORMALIZED" in e.code for e in result.errors)
    
    def test_preflight_warns_on_large_pitch_ratio(self):
        """Preflight should warn when voxel_pitch is too large relative to domain."""
        normalized_spec = {
            "meta": {"seed": 42, "input_units": "m"},
            "policies": {
                "embedding": {
                    "voxel_pitch": 0.002,  # 2mm pitch for 10mm domain - ratio is 1:5
                }
            },
            "domains": {
                "main": {"type": "cylinder", "center": [0, 0, 0], "radius": 0.005, "height": 0.002}
            },
            "components": [],
        }
        result = run_preflight_checks(normalized_spec)
        
        assert any("PITCH_TOO_LARGE" in w.code for w in result.warnings)
    
    def test_preflight_detects_missing_domains(self):
        """Preflight should detect when no domains are defined."""
        normalized_spec = {
            "meta": {"seed": 42, "input_units": "m"},
            "policies": {},
            "domains": {},
            "components": [],
        }
        result = run_preflight_checks(normalized_spec)
        
        assert not result.success
        assert any("NO_DOMAINS" in e.code for e in result.errors)
    
    def test_preflight_detects_invalid_domain_ref(self):
        """Preflight should detect invalid domain references in components."""
        normalized_spec = {
            "meta": {"seed": 42, "input_units": "m"},
            "policies": {},
            "domains": {
                "main": {"type": "cylinder", "center": [0, 0, 0], "radius": 0.005, "height": 0.002}
            },
            "components": [
                {"id": "comp1", "domain_ref": "nonexistent_domain"}
            ],
        }
        result = run_preflight_checks(normalized_spec)
        
        assert not result.success
        assert any("INVALID_DOMAIN_REF" in e.code for e in result.errors)
    
    def test_preflight_strict_raises_exception(self):
        """Preflight with strict mode should raise exception on errors."""
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m"},
            "policies": {
                "embedding": {"voxel_pitch": 0.05}  # Too large
            },
            "domains": {"main": {"type": "box", "x_min": -0.005, "x_max": 0.005, "y_min": -0.005, "y_max": 0.005, "z_min": -0.001, "z_max": 0.001}},
            "components": [],
        }
        with pytest.raises(PreflightValidationError):
            DesignSpec.from_dict(spec_dict, preflight_strict=True)


class TestDefaultPolicyFilling:
    """Tests for default policy filling."""
    
    def test_fill_missing_policy_with_defaults(self):
        """Missing policies should be filled with defaults."""
        policies = {}
        filled, filled_names = fill_policy_defaults(policies)
        
        assert "mesh_merge" in filled
        assert filled["mesh_merge"]["voxel_pitch"] == DEFAULT_POLICIES["mesh_merge"]["voxel_pitch"]
        assert any("mesh_merge" in name for name in filled_names)
    
    def test_fill_missing_keys_in_existing_policy(self):
        """Missing keys in existing policies should be filled."""
        policies = {
            "mesh_merge": {
                "mode": "voxel",  # Explicitly set
            }
        }
        filled, filled_names = fill_policy_defaults(policies)
        
        assert filled["mesh_merge"]["mode"] == "voxel"  # Preserved
        assert "voxel_pitch" in filled["mesh_merge"]  # Filled
    
    def test_fill_nested_policies(self):
        """Nested policies should be filled with defaults."""
        policies = {
            "composition": {}  # Empty composition policy
        }
        filled, filled_names = fill_policy_defaults(policies, fill_nested=True)
        
        assert "merge_policy" in filled["composition"]
        assert "repair_policy" in filled["composition"]
    
    def test_spec_fills_defaults_by_default(self):
        """DesignSpec.from_dict should fill defaults by default."""
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m"},
            "policies": {},
            "domains": {"main": {"type": "box", "x_min": -0.01, "x_max": 0.01, "y_min": -0.01, "y_max": 0.01, "z_min": -0.01, "z_max": 0.01}},
            "components": [],
        }
        spec = DesignSpec.from_dict(spec_dict)
        
        assert "mesh_merge" in spec.policies
        assert len(spec.policies_filled) > 0
    
    def test_spec_can_skip_default_filling(self):
        """DesignSpec.from_dict should allow skipping default filling."""
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m"},
            "policies": {},
            "domains": {"main": {"type": "box", "x_min": -0.01, "x_max": 0.01, "y_min": -0.01, "y_max": 0.01, "z_min": -0.01, "z_max": 0.01}},
            "components": [],
        }
        spec = DesignSpec.from_dict(spec_dict, fill_defaults=False)
        
        assert "mesh_merge" not in spec.policies or spec.policies.get("mesh_merge") is None


class TestMalariaVenuleSpecComprehensive:
    """Comprehensive tests for malaria venule spec normalization and validation."""
    
    EXAMPLES_DIR = Path(__file__).parent.parent.parent.parent / "examples" / "designspec"
    
    def test_load_malaria_venule_cco_spec(self):
        """Load and validate malaria_venule_cco.json spec."""
        spec_path = self.EXAMPLES_DIR / "malaria_venule_cco.json"
        if not spec_path.exists():
            pytest.skip("malaria_venule_cco.json not found")
        
        spec = DesignSpec.from_json(spec_path)
        
        assert spec.seed == 42
        assert spec.input_units == "mm"
        assert "cylinder_domain" in spec.domains
    
    def test_malaria_venule_domain_normalized(self):
        """Malaria venule domain should be normalized to meters."""
        spec_path = self.EXAMPLES_DIR / "malaria_venule_cco.json"
        if not spec_path.exists():
            pytest.skip("malaria_venule_cco.json not found")
        
        spec = DesignSpec.from_json(spec_path)
        
        domain = spec.domains["cylinder_domain"]
        assert domain["radius"] == pytest.approx(0.005)  # 5mm -> 0.005m
        assert domain["height"] == pytest.approx(0.002)  # 2mm -> 0.002m
    
    def test_malaria_venule_voxel_pitches_normalized(self):
        """Malaria venule voxel pitches should be normalized to meters."""
        spec_path = self.EXAMPLES_DIR / "malaria_venule_cco.json"
        if not spec_path.exists():
            pytest.skip("malaria_venule_cco.json not found")
        
        spec = DesignSpec.from_json(spec_path)
        
        merge_policy = spec.policies["composition"]["merge_policy"]
        assert merge_policy["voxel_pitch"] == pytest.approx(5e-5)  # 0.05mm -> 5e-5m
        
        repair_policy = spec.policies["composition"]["repair_policy"]
        assert repair_policy["voxel_pitch"] == pytest.approx(1e-4)  # 0.1mm -> 1e-4m
    
    def test_malaria_venule_volume_fields_normalized(self):
        """Malaria venule volume fields should be normalized with cubic scale."""
        spec_path = self.EXAMPLES_DIR / "malaria_venule_cco.json"
        if not spec_path.exists():
            pytest.skip("malaria_venule_cco.json not found")
        
        spec = DesignSpec.from_json(spec_path)
        
        merge_policy = spec.policies["composition"]["merge_policy"]
        assert merge_policy["min_component_volume"] == pytest.approx(1e-12)  # 0.001mm³ -> 1e-12m³
    
    def test_malaria_venule_port_radii_normalized(self):
        """Malaria venule port radii should be normalized to meters."""
        spec_path = self.EXAMPLES_DIR / "malaria_venule_cco.json"
        if not spec_path.exists():
            pytest.skip("malaria_venule_cco.json not found")
        
        spec = DesignSpec.from_json(spec_path)
        
        component = spec.components[0]
        inlet = component["ports"]["inlets"][0]
        assert inlet["radius"] == pytest.approx(0.0003)  # 0.3mm -> 0.0003m
    
    def test_malaria_venule_policies_not_none(self):
        """Malaria venule resolved policies should not be None."""
        spec_path = self.EXAMPLES_DIR / "malaria_venule_cco.json"
        if not spec_path.exists():
            pytest.skip("malaria_venule_cco.json not found")
        
        spec = DesignSpec.from_json(spec_path)
        
        assert spec.policies.get("composition") is not None
        assert spec.policies["composition"].get("merge_policy") is not None
        assert spec.policies["composition"].get("repair_policy") is not None
        assert spec.policies.get("embedding") is not None
        assert spec.policies.get("mesh_merge") is not None
    
    def test_malaria_venule_preflight_passes(self):
        """Malaria venule spec should pass preflight validation."""
        spec_path = self.EXAMPLES_DIR / "malaria_venule_cco.json"
        if not spec_path.exists():
            pytest.skip("malaria_venule_cco.json not found")
        
        spec = DesignSpec.from_json(spec_path)
        
        assert spec.preflight_result is not None
        assert spec.preflight_result.success, f"Preflight errors: {[e.message for e in spec.preflight_result.errors]}"


class TestUnitAuditReport:
    """Tests for unit audit report generation."""
    
    def test_audit_report_generated_in_debug_mode(self):
        """Unit audit report should be generated when debug mode is enabled."""
        os.environ[DEBUG_ENV_VAR] = "1"
        try:
            spec_dict = {
                "schema": {"name": "aog_designspec", "version": "1.0.0"},
                "meta": {"seed": 42, "input_units": "mm"},
                "policies": {
                    "embedding": {"voxel_pitch": 0.05}
                },
                "domains": {"main": {"type": "box", "x_min": -5, "x_max": 5, "y_min": -5, "y_max": 5, "z_min": -1, "z_max": 1}},
                "components": [],
            }
            spec = DesignSpec.from_dict(spec_dict)
            
            assert spec.unit_audit_report is not None
            assert spec.unit_audit_report.input_units == "mm"
            assert spec.unit_audit_report.scale_factor == 0.001
            assert len(spec.unit_audit_report.entries) > 0
        finally:
            os.environ.pop(DEBUG_ENV_VAR, None)
    
    def test_audit_report_not_generated_without_debug_mode(self):
        """Unit audit report should not be generated without debug mode."""
        os.environ.pop(DEBUG_ENV_VAR, None)
        
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "mm"},
            "policies": {},
            "domains": {"main": {"type": "box", "x_min": -5, "x_max": 5, "y_min": -5, "y_max": 5, "z_min": -1, "z_max": 1}},
            "components": [],
        }
        spec = DesignSpec.from_dict(spec_dict)
        
        assert spec.unit_audit_report is None


class TestKnownBadSpecFailsPreflight:
    """Tests that known-bad specs fail preflight with readable errors."""
    
    def test_missing_required_policy_fails_preflight(self):
        """Spec with missing required policy should fail preflight."""
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m"},
            "policies": {
                "ports": None,  # Explicitly null
            },
            "domains": {"main": {"type": "box", "x_min": -0.01, "x_max": 0.01, "y_min": -0.01, "y_max": 0.01, "z_min": -0.01, "z_max": 0.01}},
            "components": [],
        }
        spec = DesignSpec.from_dict(spec_dict, fill_defaults=False)
        
        result = run_preflight_checks(spec.normalized, stages_to_run=["component_ports"])
        
        assert not result.success
        assert any("MISSING_REQUIRED_POLICY" in e.code for e in result.errors)
        assert any("ports" in e.message for e in result.errors)
    
    def test_unnormalized_domain_warns(self):
        """Spec with unnormalized domain values should generate warnings."""
        normalized_spec = {
            "meta": {"seed": 42, "input_units": "m"},
            "policies": {},
            "domains": {
                "main": {"type": "cylinder", "center": [0, 0, 0], "radius": 5.0, "height": 2.0}  # 5m radius - way too large
            },
            "components": [],
        }
        result = run_preflight_checks(normalized_spec)
        
        assert any("DOMAIN_NOT_NORMALIZED" in w.code for w in result.warnings)
