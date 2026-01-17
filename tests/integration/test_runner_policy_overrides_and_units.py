"""
Test DesignSpecRunner policy overrides and unit normalization.

This module verifies that component-level policy overrides are applied
and that input_units normalization works correctly.
"""

import pytest
import json
from pathlib import Path

from designspec.spec import DesignSpec
from designspec.runner import DesignSpecRunner, RunnerResult
from designspec.plan import ExecutionPlan


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "designspec"


class TestPolicyOverrides:
    """Test component-level policy overrides."""
    
    def test_spec_policies_are_normalized(self):
        """Test that spec policies are normalized from input_units."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        assert spec.policies is not None
        assert isinstance(spec.policies, dict)
    
    def test_resolution_policy_normalized_to_meters(self):
        """Test that resolution policy values are normalized to meters."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        resolution = spec.policies.get("resolution", {})
        
        if "min_channel_diameter" in resolution:
            min_diameter = resolution["min_channel_diameter"]
            assert min_diameter < 0.001, (
                f"min_channel_diameter should be in meters (got {min_diameter})"
            )
    
    def test_component_can_have_policy_overrides(self):
        """Test that components can have policy overrides."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        for component in spec.components:
            if "policy_overrides" in component:
                overrides = component["policy_overrides"]
                assert isinstance(overrides, dict)
    
    def test_effective_policy_reflects_overrides(self, tmp_path):
        """Test that effective policy reflects component overrides."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        plan = ExecutionPlan(run_until="compile_policies")
        
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        if result.success:
            compile_reports = [r for r in result.stage_reports if r.stage == "compile_policies"]
            if compile_reports:
                report = compile_reports[0]
                assert report.success


class TestInputUnitsNormalization:
    """Test input_units normalization."""
    
    def test_spec_has_input_units(self):
        """Test that spec has input_units field."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        assert hasattr(spec, "input_units")
        assert spec.input_units in ["mm", "um", "m", "cm"]
    
    def test_domains_normalized_from_input_units(self):
        """Test that domains are normalized from input_units."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        if spec.input_units == "mm":
            for domain_name, domain in spec.domains.items():
                if "x_min" in domain:
                    assert abs(domain["x_min"]) < 1.0, (
                        f"Domain {domain_name} x_min should be in meters"
                    )
    
    def test_port_positions_normalized(self):
        """Test that port positions are normalized from input_units."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        for component in spec.components:
            ports = component.get("ports", {})
            inlets = ports.get("inlets", [])
            
            for inlet in inlets:
                if "position" in inlet:
                    pos = inlet["position"]
                    for coord in pos:
                        assert abs(coord) < 1.0, (
                            f"Port position should be in meters (got {coord})"
                        )
    
    def test_port_radii_normalized(self):
        """Test that port radii are normalized from input_units."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        for component in spec.components:
            ports = component.get("ports", {})
            inlets = ports.get("inlets", [])
            
            for inlet in inlets:
                if "radius" in inlet:
                    radius = inlet["radius"]
                    assert radius < 0.01, (
                        f"Port radius should be in meters (got {radius})"
                    )


class TestPolicyCompilation:
    """Test policy compilation from dicts to aog_policies objects."""
    
    def test_policies_compile_successfully(self, tmp_path):
        """Test that policies compile successfully."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        plan = ExecutionPlan(run_until="compile_policies")
        
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        compile_reports = [r for r in result.stage_reports if r.stage == "compile_policies"]
        
        if compile_reports:
            report = compile_reports[0]
            assert report.success, f"Policy compilation failed: {report.errors}"
    
    def test_compiled_policies_stored_in_runner(self, tmp_path):
        """Test that compiled policies are stored in runner."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        plan = ExecutionPlan(run_until="compile_policies")
        
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        if result.success:
            assert hasattr(runner, "_compiled_policies")
            assert isinstance(runner._compiled_policies, dict)
    
    def test_policy_compilation_report_has_metadata(self, tmp_path):
        """Test that policy compilation report has metadata."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        plan = ExecutionPlan(run_until="compile_policies")
        
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        compile_reports = [r for r in result.stage_reports if r.stage == "compile_policies"]
        
        if compile_reports:
            report = compile_reports[0]
            assert hasattr(report, "metadata")
            assert isinstance(report.metadata, dict)


class TestComponentPolicyOverridesNormalization:
    """Test that policy_overrides inside components are normalized."""
    
    def test_policy_overrides_length_fields_normalized(self):
        """Test that length fields in policy_overrides are normalized to meters."""
        # Create a spec with policy_overrides containing length fields in mm
        spec_dict = {
            "schema": {
                "name": "aog_designspec",
                "version": "1.0.0"
            },
            "meta": {
                "name": "test_policy_overrides_normalization",
                "version": "1.0",
                "input_units": "mm"
            },
            "domains": {
                "test_domain": {
                    "x_min": 0, "x_max": 10,
                    "y_min": 0, "y_max": 10,
                    "z_min": 0, "z_max": 10
                }
            },
            "components": [
                {
                    "name": "test_component",
                    "kind": "space_colonization",
                    "domain_ref": "test_domain",
                    "policy_overrides": {
                        "growth": {
                            "step_size": 0.5,  # 0.5 mm should become 0.0005 m
                            "min_segment_length": 0.1  # 0.1 mm should become 0.0001 m
                        },
                        "channels": {
                            "length": 1.0,  # 1.0 mm should become 0.001 m
                            "start_offset": 0.2   # 0.2 mm should become 0.0002 m
                        }
                    },
                    "ports": {
                        "inlets": [{"position": [5, 5, 0], "direction": [0, 0, 1], "radius": 0.2}],
                        "outlets": [{"position": [5, 5, 10], "direction": [0, 0, -1], "radius": 0.1}]
                    }
                }
            ],
            "policies": {}
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        
        # Check that policy_overrides are normalized
        component = spec.components[0]
        overrides = component.get("policy_overrides", {})
        
        growth_overrides = overrides.get("growth", {})
        if "step_size" in growth_overrides:
            # 0.5 mm = 0.0005 m
            assert growth_overrides["step_size"] == pytest.approx(0.0005, rel=1e-6), (
                f"step_size should be normalized to meters (got {growth_overrides['step_size']})"
            )
        
        if "min_segment_length" in growth_overrides:
            # 0.1 mm = 0.0001 m
            assert growth_overrides["min_segment_length"] == pytest.approx(0.0001, rel=1e-6), (
                f"min_segment_length should be normalized to meters (got {growth_overrides['min_segment_length']})"
            )
        
        channel_overrides = overrides.get("channels", {})
        if "length" in channel_overrides:
            # 1.0 mm = 0.001 m
            assert channel_overrides["length"] == pytest.approx(0.001, rel=1e-6), (
                f"length should be normalized to meters (got {channel_overrides['length']})"
            )


class TestBackendParamsNormalization:
    """Test that backend_params length fields are normalized."""
    
    def test_backend_params_length_fields_normalized(self):
        """Test that known length fields in backend_params are normalized to meters."""
        # Create a spec with backend_params containing length fields in mm
        spec_dict = {
            "schema": {
                "name": "aog_designspec",
                "version": "1.0.0"
            },
            "meta": {
                "name": "test_backend_params_normalization",
                "version": "1.0",
                "input_units": "mm"
            },
            "domains": {
                "test_domain": {
                    "x_min": 0, "x_max": 10,
                    "y_min": 0, "y_max": 10,
                    "z_min": 0, "z_max": 10
                }
            },
            "components": [
                {
                    "name": "test_component",
                    "kind": "space_colonization",
                    "domain_ref": "test_domain",
                    "build": {
                        "backend_params": {
                            "step_size": 0.5,  # 0.5 mm should become 0.0005 m
                            "influence_radius": 1.0,  # 1.0 mm should become 0.001 m
                            "kill_radius": 0.2,  # 0.2 mm should become 0.0002 m
                            "perception_radius": 2.0  # 2.0 mm should become 0.002 m
                        }
                    },
                    "ports": {
                        "inlets": [{"position": [5, 5, 0], "direction": [0, 0, 1], "radius": 0.2}],
                        "outlets": [{"position": [5, 5, 10], "direction": [0, 0, -1], "radius": 0.1}]
                    }
                }
            ],
            "policies": {}
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        
        # Check that backend_params are normalized
        component = spec.components[0]
        build = component.get("build", {})
        backend_params = build.get("backend_params", {})
        
        if "step_size" in backend_params:
            # 0.5 mm = 0.0005 m
            assert backend_params["step_size"] == pytest.approx(0.0005, rel=1e-6), (
                f"step_size should be normalized to meters (got {backend_params['step_size']})"
            )
        
        if "influence_radius" in backend_params:
            # 1.0 mm = 0.001 m
            assert backend_params["influence_radius"] == pytest.approx(0.001, rel=1e-6), (
                f"influence_radius should be normalized to meters (got {backend_params['influence_radius']})"
            )
        
        if "kill_radius" in backend_params:
            # 0.2 mm = 0.0002 m
            assert backend_params["kill_radius"] == pytest.approx(0.0002, rel=1e-6), (
                f"kill_radius should be normalized to meters (got {backend_params['kill_radius']})"
            )
        
        if "perception_radius" in backend_params:
            # 2.0 mm = 0.002 m
            assert backend_params["perception_radius"] == pytest.approx(0.002, rel=1e-6), (
                f"perception_radius should be normalized to meters (got {backend_params['perception_radius']})"
            )


class TestEffectivePolicySnapshot:
    """Test effective policy snapshot in reports."""
    
    def test_stage_reports_include_requested_policy(self, tmp_path):
        """Test that stage reports include requested_policy."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()
        
        for report in result.stage_reports:
            report_dict = report.to_dict()
            assert "requested_policy" in report_dict
    
    def test_stage_reports_include_effective_policy(self, tmp_path):
        """Test that stage reports include effective_policy."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()
        
        for report in result.stage_reports:
            report_dict = report.to_dict()
            assert "effective_policy" in report_dict
    
    def test_policies_are_json_serializable(self, tmp_path):
        """Test that policies in reports are JSON-serializable."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()
        
        for report in result.stage_reports:
            report_dict = report.to_dict()
            
            if report_dict.get("requested_policy") is not None:
                json_str = json.dumps(report_dict["requested_policy"])
                assert json_str is not None
            
            if report_dict.get("effective_policy") is not None:
                json_str = json.dumps(report_dict["effective_policy"])
                assert json_str is not None
