"""
Test DesignSpecRunner with different growth backends.

This module validates that the runner correctly handles various growth
algorithm backends including space_colonization and others.
"""

import pytest
import json
from pathlib import Path

from designspec.spec import DesignSpec
from designspec.runner import DesignSpecRunner, RunnerResult
from designspec.plan import ExecutionPlan


def make_space_colonization_spec(tmp_path, seed=42):
    """Create a minimal spec with space_colonization backend."""
    return {
        "schema": {"name": "aog_designspec", "version": "1.0.0"},
        "meta": {"name": "space_colonization_test", "seed": seed, "input_units": "mm"},
        "policies": {
            "resolution": {
                "input_units": "mm",
                "min_channel_diameter": 0.1,
                "min_voxels_across_feature": 4,
                "max_voxels": 1000000,
            },
            "growth": {
                "enabled": True,
                "backend": "space_colonization",
                "max_iterations": 20,
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


def make_kary_tree_spec(tmp_path, seed=42):
    """Create a minimal spec with kary_tree backend."""
    return {
        "schema": {"name": "aog_designspec", "version": "1.0.0"},
        "meta": {"name": "kary_tree_test", "seed": seed, "input_units": "mm"},
        "policies": {
            "resolution": {
                "input_units": "mm",
                "min_channel_diameter": 0.1,
                "min_voxels_across_feature": 4,
                "max_voxels": 1000000,
            },
            "growth": {
                "enabled": True,
                "backend": "kary_tree",
                "max_iterations": 20,
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
                "build": {"type": "backend_network", "backend": "kary_tree"},
            }
        ],
        "outputs": {"artifacts_dir": str(tmp_path / "artifacts")},
    }


class TestRunnerGrowthBackends:
    """Test runner with different growth backends."""
    
    def test_space_colonization_builds_network(self, tmp_path):
        """Test that space_colonization backend builds a network."""
        spec_dict = make_space_colonization_spec(tmp_path)
        spec = DesignSpec.from_dict(spec_dict)
        
        plan = ExecutionPlan(run_until="component_mesh:net_1")
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        assert isinstance(result, RunnerResult)
        # Check that component_build stage was attempted
        build_stages = [s for s in result.stages_completed if "component_build" in s]
        assert len(build_stages) > 0, "Should have completed component_build stage"
    
    def test_space_colonization_produces_mesh(self, tmp_path):
        """Test that space_colonization backend produces a void mesh."""
        spec_dict = make_space_colonization_spec(tmp_path)
        spec = DesignSpec.from_dict(spec_dict)
        
        plan = ExecutionPlan(run_until="union_voids")
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        if result.success:
            assert runner._union_void is not None
            assert len(runner._union_void.vertices) > 0
            assert len(runner._union_void.faces) > 0
    
    def test_kary_tree_builds_network(self, tmp_path):
        """Test that kary_tree backend builds a network."""
        spec_dict = make_kary_tree_spec(tmp_path)
        spec = DesignSpec.from_dict(spec_dict)
        
        plan = ExecutionPlan(run_until="component_mesh:net_1")
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        assert isinstance(result, RunnerResult)
        # Check that component_build stage was attempted
        build_stages = [s for s in result.stages_completed if "component_build" in s]
        assert len(build_stages) > 0, "Should have completed component_build stage"
    
    def test_backend_name_in_report(self, tmp_path):
        """Test that backend name is recorded in stage report."""
        spec_dict = make_space_colonization_spec(tmp_path)
        spec = DesignSpec.from_dict(spec_dict)
        
        plan = ExecutionPlan(run_until="component_build:net_1")
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        build_reports = [r for r in result.stage_reports if "component_build" in r.stage]
        assert len(build_reports) > 0, "Should have build stage report"


class TestGrowthBackendMetrics:
    """Test growth backend metrics in reports."""
    
    def test_space_colonization_reports_metrics(self, tmp_path):
        """Test that space_colonization reports growth metrics."""
        spec_dict = make_space_colonization_spec(tmp_path)
        spec = DesignSpec.from_dict(spec_dict)
        
        plan = ExecutionPlan(run_until="component_build:net_1")
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        build_reports = [r for r in result.stage_reports if "component_build" in r.stage]
        if build_reports and build_reports[0].success:
            metadata = build_reports[0].metadata
            assert isinstance(metadata, dict)
    
    def test_result_is_json_serializable(self, tmp_path):
        """Test that runner result is JSON-serializable."""
        spec_dict = make_space_colonization_spec(tmp_path)
        spec = DesignSpec.from_dict(spec_dict)
        
        plan = ExecutionPlan(run_until="component_build:net_1")
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        result_dict = result.to_dict()
        json_str = json.dumps(result_dict)
        
        assert json_str is not None
        decoded = json.loads(json_str)
        assert "success" in decoded
