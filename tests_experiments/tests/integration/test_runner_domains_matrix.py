"""
Test DesignSpecRunner with different domain types.

This module validates that the runner correctly handles various domain
specifications including primitives, transforms, and composites.
"""

import pytest
import json
from pathlib import Path

from designspec.spec import DesignSpec
from designspec.runner import DesignSpecRunner, RunnerResult
from designspec.plan import ExecutionPlan


def make_box_spec(tmp_path):
    """Create a minimal spec with a box domain."""
    return {
        "schema": {"name": "aog_designspec", "version": "1.0.0"},
        "meta": {"name": "box_domain_test", "seed": 42, "input_units": "m"},
        "policies": {
            "resolution": {
                "input_units": "m",
                "min_channel_diameter": 0.1,
                "min_voxels_across_feature": 4,
                "max_voxels": 1000000,
            },
            "growth": {
                "enabled": True,
                "backend": "space_colonization",
                "max_iterations": 10,
                "step_size": 0.5,
            },
        },
        "domains": {
            "main": {
                "type": "box",
                "x_min": -5, "x_max": 5,
                "y_min": -5, "y_max": 5,
                "z_min": -3, "z_max": 3,
            }
        },
        "components": [
            {
                "id": "net_1",
                "domain_ref": "main",
                "ports": {
                    "inlets": [{
                        "name": "inlet",
                        "position": [0, 0, 3],
                        "direction": [0, 0, -1],
                        "radius": 0.3,
                        "vessel_type": "arterial",
                    }],
                    "outlets": [{
                        "name": "outlet",
                        "position": [0, 0, -3],
                        "direction": [0, 0, 1],
                        "radius": 0.25,
                        "vessel_type": "venous",
                    }],
                },
                "build": {"type": "backend_network", "backend": "space_colonization"},
            }
        ],
        "outputs": {"artifacts_dir": str(tmp_path / "artifacts")},
    }


def make_cylinder_spec(tmp_path):
    """Create a minimal spec with a cylinder domain."""
    return {
        "schema": {"name": "aog_designspec", "version": "1.0.0"},
        "meta": {"name": "cylinder_domain_test", "seed": 42, "input_units": "m"},
        "policies": {
            "resolution": {
                "input_units": "m",
                "min_channel_diameter": 0.1,
                "min_voxels_across_feature": 4,
                "max_voxels": 1000000,
            },
            "growth": {
                "enabled": True,
                "backend": "space_colonization",
                "max_iterations": 10,
                "step_size": 0.5,
            },
        },
        "domains": {
            "main": {
                "type": "cylinder",
                "center": [0, 0, 0],
                "radius": 5,
                "height": 6,
                "axis": [0, 0, 1],
            }
        },
        "components": [
            {
                "id": "net_1",
                "domain_ref": "main",
                "ports": {
                    "inlets": [{
                        "name": "inlet",
                        "position": [0, 0, 3],
                        "direction": [0, 0, -1],
                        "radius": 0.3,
                        "vessel_type": "arterial",
                    }],
                    "outlets": [{
                        "name": "outlet",
                        "position": [0, 0, -3],
                        "direction": [0, 0, 1],
                        "radius": 0.25,
                        "vessel_type": "venous",
                    }],
                },
                "build": {"type": "backend_network", "backend": "space_colonization"},
            }
        ],
        "outputs": {"artifacts_dir": str(tmp_path / "artifacts")},
    }


def make_sphere_spec(tmp_path):
    """Create a minimal spec with a sphere domain."""
    return {
        "schema": {"name": "aog_designspec", "version": "1.0.0"},
        "meta": {"name": "sphere_domain_test", "seed": 42, "input_units": "m"},
        "policies": {
            "resolution": {
                "input_units": "m",
                "min_channel_diameter": 0.1,
                "min_voxels_across_feature": 4,
                "max_voxels": 1000000,
            },
            "growth": {
                "enabled": True,
                "backend": "space_colonization",
                "max_iterations": 10,
                "step_size": 0.5,
            },
        },
        "domains": {
            "main": {
                "type": "sphere",
                "center": [0, 0, 0],
                "radius": 5,
            }
        },
        "components": [
            {
                "id": "net_1",
                "domain_ref": "main",
                "ports": {
                    "inlets": [{
                        "name": "inlet",
                        "position": [0, 0, 4.5],
                        "direction": [0, 0, -1],
                        "radius": 0.3,
                        "vessel_type": "arterial",
                    }],
                    "outlets": [{
                        "name": "outlet",
                        "position": [0, 0, -4.5],
                        "direction": [0, 0, 1],
                        "radius": 0.25,
                        "vessel_type": "venous",
                    }],
                },
                "build": {"type": "backend_network", "backend": "space_colonization"},
            }
        ],
        "outputs": {"artifacts_dir": str(tmp_path / "artifacts")},
    }


class TestRunnerDomainsMatrix:
    """Test runner with different domain types."""
    
    def test_box_domain_compiles(self, tmp_path):
        """Test that box domain compiles successfully."""
        spec_dict = make_box_spec(tmp_path)
        spec = DesignSpec.from_dict(spec_dict)
        
        plan = ExecutionPlan(run_until="compile_domains")
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        assert isinstance(result, RunnerResult)
        assert "compile_domains" in result.stages_completed
    
    def test_cylinder_domain_compiles(self, tmp_path):
        """Test that cylinder domain compiles successfully."""
        spec_dict = make_cylinder_spec(tmp_path)
        spec = DesignSpec.from_dict(spec_dict)
        
        plan = ExecutionPlan(run_until="compile_domains")
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        assert isinstance(result, RunnerResult)
        assert "compile_domains" in result.stages_completed
    
    def test_sphere_domain_compiles(self, tmp_path):
        """Test that sphere domain compiles successfully."""
        spec_dict = make_sphere_spec(tmp_path)
        spec = DesignSpec.from_dict(spec_dict)
        
        plan = ExecutionPlan(run_until="compile_domains")
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        assert isinstance(result, RunnerResult)
        assert "compile_domains" in result.stages_completed
    
    def test_box_domain_has_bounds(self, tmp_path):
        """Test that compiled box domain has bounds."""
        spec_dict = make_box_spec(tmp_path)
        spec = DesignSpec.from_dict(spec_dict)
        
        plan = ExecutionPlan(run_until="compile_domains")
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        if result.success:
            domain = runner._compiled_domains.get("main")
            assert domain is not None
            # BoxDomain has x_min, x_max, y_min, y_max, z_min, z_max attributes
            assert hasattr(domain, "x_min")
            assert hasattr(domain, "x_max")
            assert hasattr(domain, "y_min")
            assert hasattr(domain, "y_max")
            assert hasattr(domain, "z_min")
            assert hasattr(domain, "z_max")
    
    def test_result_reports_are_json_safe(self, tmp_path):
        """Test that all stage reports are JSON-serializable."""
        spec_dict = make_box_spec(tmp_path)
        spec = DesignSpec.from_dict(spec_dict)
        
        plan = ExecutionPlan(run_until="compile_domains")
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        result_dict = result.to_dict()
        json_str = json.dumps(result_dict)
        
        assert json_str is not None
        decoded = json.loads(json_str)
        assert "stage_reports" in decoded


class TestDomainCompilationMetadata:
    """Test domain compilation metadata."""
    
    def test_compile_domains_report_exists(self, tmp_path):
        """Test that compile_domains stage report exists."""
        spec_dict = make_box_spec(tmp_path)
        spec = DesignSpec.from_dict(spec_dict)
        
        plan = ExecutionPlan(run_until="compile_domains")
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        compile_reports = [r for r in result.stage_reports if r.stage == "compile_domains"]
        assert len(compile_reports) >= 1
    
    def test_compile_domains_report_has_metadata(self, tmp_path):
        """Test that compile_domains stage report has metadata."""
        spec_dict = make_box_spec(tmp_path)
        spec = DesignSpec.from_dict(spec_dict)
        
        plan = ExecutionPlan(run_until="compile_domains")
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        compile_reports = [r for r in result.stage_reports if r.stage == "compile_domains"]
        if compile_reports:
            report = compile_reports[0]
            assert hasattr(report, "metadata")
            assert isinstance(report.metadata, dict)
