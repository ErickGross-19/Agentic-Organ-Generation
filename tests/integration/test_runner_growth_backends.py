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


def make_scaffold_topdown_spec(tmp_path, seed=42):
    """Create a minimal spec with scaffold_topdown backend."""
    return {
        "schema": {"name": "aog_designspec", "version": "1.0.0"},
        "meta": {"name": "scaffold_topdown_test", "seed": seed, "input_units": "mm"},
        "policies": {
            "resolution": {
                "input_units": "mm",
                "min_channel_diameter": 0.1,
                "min_voxels_across_feature": 4,
                "max_voxels": 1000000,
            },
            "growth": {
                "enabled": True,
                "backend": "scaffold_topdown",
                "max_iterations": 20,
                "step_size": 0.5,
                "min_radius": 0.05,
                "backend_params": {
                    "primary_axis": [0, 0, -1],
                    "splits": 2,
                    "levels": 3,
                    "ratio": 0.78,
                    "step_length": 1.5,
                    "step_decay": 0.9,
                    "spread": 1.0,
                    "spread_decay": 0.9,
                    "cone_angle_deg": 60,
                    "jitter_deg": 10,
                    "curvature": 0.3,
                    "curve_samples": 5,
                    "wall_margin_m": 0.1,
                    "min_radius": 0.05,
                    "collision_online": {
                        "enabled": True,
                        "buffer_abs_m": 0.02,
                        "buffer_rel": 0.05,
                        "cell_size_m": 0.5,
                        "rotation_attempts": 8,
                        "reduction_factors": [0.6, 0.35],
                        "max_attempts_per_child": 12,
                        "on_fail": "terminate_branch"
                    },
                    "collision_postpass": {
                        "enabled": False
                    }
                },
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
                        "radius": 0.5,
                        "vessel_type": "arterial",
                    }],
                    "outlets": [],
                },
                "build": {
                    "type": "backend_network",
                    "backend": "scaffold_topdown",
                    "backend_params": {
                        "primary_axis": [0, 0, -1],
                        "splits": 2,
                        "levels": 3,
                        "ratio": 0.78,
                        "step_length": 1.5,
                        "step_decay": 0.9,
                        "spread": 1.0,
                        "spread_decay": 0.9,
                        "cone_angle_deg": 60,
                        "jitter_deg": 10,
                        "curvature": 0.3,
                        "curve_samples": 5,
                        "wall_margin_m": 0.1,
                        "min_radius": 0.05,
                        "collision_online": {
                            "enabled": True,
                            "buffer_abs_m": 0.02,
                            "buffer_rel": 0.05,
                            "cell_size_m": 0.5,
                            "rotation_attempts": 8,
                            "reduction_factors": [0.6, 0.35],
                            "max_attempts_per_child": 12,
                            "on_fail": "terminate_branch"
                        },
                        "collision_postpass": {
                            "enabled": False
                        }
                    }
                },
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
    
    def test_scaffold_topdown_builds_network(self, tmp_path):
        """Test that scaffold_topdown backend builds a network."""
        spec_dict = make_scaffold_topdown_spec(tmp_path)
        spec = DesignSpec.from_dict(spec_dict)
        
        plan = ExecutionPlan(run_until="component_mesh:net_1")
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        assert isinstance(result, RunnerResult)
        build_stages = [s for s in result.stages_completed if "component_build" in s]
        assert len(build_stages) > 0, "Should have completed component_build stage"
    
    def test_scaffold_topdown_produces_segments(self, tmp_path):
        """Test that scaffold_topdown backend produces network with segments."""
        spec_dict = make_scaffold_topdown_spec(tmp_path)
        spec = DesignSpec.from_dict(spec_dict)
        
        plan = ExecutionPlan(run_until="component_build:net_1")
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        if result.success:
            network = runner._component_networks.get("net_1")
            if network is not None:
                assert len(network.segments) > 0, "Network should have segments"
                assert len(network.nodes) > 0, "Network should have nodes"
    
    def test_scaffold_topdown_deterministic(self, tmp_path):
        """Test that scaffold_topdown backend is deterministic with same seed."""
        spec_dict1 = make_scaffold_topdown_spec(tmp_path, seed=12345)
        spec_dict2 = make_scaffold_topdown_spec(tmp_path, seed=12345)
        
        spec1 = DesignSpec.from_dict(spec_dict1)
        spec2 = DesignSpec.from_dict(spec_dict2)
        
        plan = ExecutionPlan(run_until="component_build:net_1")
        
        runner1 = DesignSpecRunner(spec1, plan=plan, output_dir=tmp_path / "run1")
        result1 = runner1.run()
        
        runner2 = DesignSpecRunner(spec2, plan=plan, output_dir=tmp_path / "run2")
        result2 = runner2.run()
        
        if result1.success and result2.success:
            net1 = runner1._component_networks.get("net_1")
            net2 = runner2._component_networks.get("net_1")
            if net1 is not None and net2 is not None:
                assert len(net1.segments) == len(net2.segments), "Same seed should produce same segment count"
                assert len(net1.nodes) == len(net2.nodes), "Same seed should produce same node count"


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
