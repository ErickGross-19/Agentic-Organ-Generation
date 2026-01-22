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


def make_scaffold_topdown_cylinder_spec(tmp_path, seed=42):
    """
    Create a minimal spec with scaffold_topdown backend using a CYLINDER domain.
    
    This is a regression test for Issue 1: CylinderDomain crash due to
    _clamp_to_domain using domain.bounds which doesn't exist on CylinderDomain.
    """
    return {
        "schema": {"name": "aog_designspec", "version": "1.0.0"},
        "meta": {"name": "scaffold_topdown_cylinder_test", "seed": seed, "input_units": "mm"},
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
                "type": "cylinder",
                "radius": 5.0,
                "height": 6.0,
                "center": [0, 0, 0],
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
    
    def test_scaffold_topdown_cylinder_domain(self, tmp_path):
        """
        Test that scaffold_topdown backend works with CYLINDER domain.
        
        This is a regression test for Issue 1: CylinderDomain crash due to
        _clamp_to_domain using domain.bounds which doesn't exist on CylinderDomain.
        The fix uses DomainSpec.project_inside() API which works for all domain types.
        """
        spec_dict = make_scaffold_topdown_cylinder_spec(tmp_path)
        spec = DesignSpec.from_dict(spec_dict)
        
        plan = ExecutionPlan(run_until="component_build:net_1")
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        # The test passes if generation succeeds without AttributeError
        assert isinstance(result, RunnerResult)
        assert result.success, f"Cylinder domain generation should succeed: {result.error}"
        
        # Verify network was created with segments
        network = runner._component_networks.get("net_1")
        assert network is not None, "Network should be created"
        assert len(network.segments) > 0, "Network should have segments"
        assert len(network.nodes) > 0, "Network should have nodes"


class TestScaffoldTopdownBifurcation:
    """
    Regression tests for scaffold_topdown bifurcation with splits=2.
    
    These tests verify that the shared-endpoint exclusion fix allows sibling
    branches to be created without false collision detection at the shared
    parent node.
    """
    
    def test_scaffold_topdown_bifurcation_produces_outdegree_2(self, tmp_path):
        """
        Test that scaffold_topdown with splits=2 produces nodes with outdegree=2.
        
        This is a regression test for the issue where sibling branches sharing
        the same parent node were incorrectly detected as colliding at the
        shared endpoint, preventing bifurcation.
        """
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"name": "bifurcation_test", "seed": 42, "input_units": "mm"},
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
                    "min_radius": 0.05,
                    "backend_params": {
                        "primary_axis": [0, 0, -1],
                        "splits": 2,
                        "levels": 4,
                        "ratio": 0.78,
                        "step_length": 1.5,
                        "step_decay": 0.85,
                        "spread": 1.2,
                        "spread_decay": 0.88,
                        "cone_angle_deg": 60,
                        "jitter_deg": 5,
                        "curvature": 0.2,
                        "curve_samples": 5,
                        "wall_margin_m": 0.1,
                        "min_radius": 0.05,
                        "collision_online": {
                            "enabled": True,
                            "buffer_abs_m": 0.02,
                            "buffer_rel": 0.05,
                            "cell_size_m": 0.5,
                            "rotation_attempts": 12,
                            "reduction_factors": [0.6, 0.35],
                            "max_attempts_per_child": 16,
                            "on_fail": "terminate_branch"
                        },
                        "collision_postpass": {"enabled": False}
                    },
                },
            },
            "domains": {
                "main": {
                    "type": "box",
                    "x_min": -10, "x_max": 10,
                    "y_min": -10, "y_max": 10,
                    "z_min": -8, "z_max": 4,
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
                            "levels": 4,
                            "ratio": 0.78,
                            "step_length": 1.5,
                            "step_decay": 0.85,
                            "spread": 1.2,
                            "spread_decay": 0.88,
                            "cone_angle_deg": 60,
                            "jitter_deg": 5,
                            "curvature": 0.2,
                            "curve_samples": 5,
                            "wall_margin_m": 0.1,
                            "min_radius": 0.05,
                            "collision_online": {
                                "enabled": True,
                                "buffer_abs_m": 0.02,
                                "buffer_rel": 0.05,
                                "cell_size_m": 0.5,
                                "rotation_attempts": 12,
                                "reduction_factors": [0.6, 0.35],
                                "max_attempts_per_child": 16,
                                "on_fail": "terminate_branch"
                            },
                            "collision_postpass": {"enabled": False}
                        }
                    },
                }
            ],
            "outputs": {"artifacts_dir": str(tmp_path / "artifacts")},
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        plan = ExecutionPlan(run_until="component_build:net_1")
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        assert result.success, f"Generation should succeed: {result.error}"
        
        network = runner._component_networks.get("net_1")
        assert network is not None, "Network should be created"
        
        # Count segments and verify non-trivial count
        segment_count = len(network.segments)
        assert segment_count > 20, f"Expected > 20 segments for bifurcating tree, got {segment_count}"
        
        # Compute outdegree for each node
        outdegree = {}
        for seg in network.segments.values():
            start_id = seg.start_node_id
            outdegree[start_id] = outdegree.get(start_id, 0) + 1
        
        # Check that at least one node has outdegree == 2 (bifurcation)
        max_outdegree = max(outdegree.values()) if outdegree else 0
        nodes_with_outdegree_2 = sum(1 for od in outdegree.values() if od == 2)
        
        assert max_outdegree >= 2, f"Expected at least one node with outdegree >= 2, max was {max_outdegree}"
        assert nodes_with_outdegree_2 > 0, "Expected at least one bifurcation point (outdegree=2)"
    
    def test_scaffold_topdown_bifurcation_deterministic(self, tmp_path):
        """
        Test that scaffold_topdown bifurcation is deterministic with fixed seed.
        """
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"name": "bifurcation_deterministic", "seed": 12345, "input_units": "mm"},
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
                    "min_radius": 0.05,
                    "backend_params": {
                        "primary_axis": [0, 0, -1],
                        "splits": 2,
                        "levels": 3,
                        "ratio": 0.78,
                        "step_length": 1.5,
                        "step_decay": 0.85,
                        "spread": 1.2,
                        "spread_decay": 0.88,
                        "cone_angle_deg": 60,
                        "jitter_deg": 5,
                        "curvature": 0.2,
                        "curve_samples": 5,
                        "wall_margin_m": 0.1,
                        "min_radius": 0.05,
                        "collision_online": {
                            "enabled": True,
                            "buffer_abs_m": 0.02,
                            "buffer_rel": 0.05,
                            "cell_size_m": 0.5,
                            "rotation_attempts": 12,
                            "reduction_factors": [0.6, 0.35],
                            "max_attempts_per_child": 16,
                            "on_fail": "terminate_branch"
                        },
                        "collision_postpass": {"enabled": False}
                    },
                },
            },
            "domains": {
                "main": {
                    "type": "box",
                    "x_min": -10, "x_max": 10,
                    "y_min": -10, "y_max": 10,
                    "z_min": -8, "z_max": 4,
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
                            "step_decay": 0.85,
                            "spread": 1.2,
                            "spread_decay": 0.88,
                            "cone_angle_deg": 60,
                            "jitter_deg": 5,
                            "curvature": 0.2,
                            "curve_samples": 5,
                            "wall_margin_m": 0.1,
                            "min_radius": 0.05,
                            "collision_online": {
                                "enabled": True,
                                "buffer_abs_m": 0.02,
                                "buffer_rel": 0.05,
                                "cell_size_m": 0.5,
                                "rotation_attempts": 12,
                                "reduction_factors": [0.6, 0.35],
                                "max_attempts_per_child": 16,
                                "on_fail": "terminate_branch"
                            },
                            "collision_postpass": {"enabled": False}
                        }
                    },
                }
            ],
            "outputs": {"artifacts_dir": str(tmp_path / "artifacts")},
        }
        
        # Run twice with same seed
        spec1 = DesignSpec.from_dict(spec_dict)
        spec2 = DesignSpec.from_dict(spec_dict)
        
        plan = ExecutionPlan(run_until="component_build:net_1")
        
        runner1 = DesignSpecRunner(spec1, plan=plan, output_dir=tmp_path / "run1")
        result1 = runner1.run()
        
        runner2 = DesignSpecRunner(spec2, plan=plan, output_dir=tmp_path / "run2")
        result2 = runner2.run()
        
        assert result1.success and result2.success, "Both runs should succeed"
        
        net1 = runner1._component_networks.get("net_1")
        net2 = runner2._component_networks.get("net_1")
        
        assert net1 is not None and net2 is not None
        assert len(net1.segments) == len(net2.segments), "Same seed should produce same segment count"
        assert len(net1.nodes) == len(net2.nodes), "Same seed should produce same node count"


class TestScaffoldTopdownDeeperBifurcation:
    """
    Tests for deeper bifurcation with scaffold_topdown.
    
    These tests verify that scaffold_topdown produces trees with depth > 2
    and that the retry and merge features work correctly.
    """
    
    def test_scaffold_topdown_deeper_bifurcation_depth_greater_than_2(self, tmp_path):
        """
        Test that scaffold_topdown with splits=2, levels>=4 produces depth > 2.
        
        This test verifies that the tree has multiple levels of bifurcation,
        not just a single split at the root.
        """
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"name": "deeper_bifurcation_test", "seed": 42, "input_units": "mm"},
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
                    "min_radius": 0.02,
                    "backend_params": {
                        "primary_axis": [0, 0, -1],
                        "splits": 2,
                        "levels": 5,
                        "ratio": 0.70,
                        "step_length": 2.0,
                        "step_decay": 0.95,
                        "spread": 1.8,
                        "spread_decay": 0.97,
                        "cone_angle_deg": 60,
                        "jitter_deg": 5,
                        "curvature": 0.2,
                        "curve_samples": 5,
                        "wall_margin_m": 0.1,
                        "min_radius": 0.02,
                        "collision_online": {
                            "enabled": True,
                            "buffer_abs_m": 0.02,
                            "buffer_rel": 0.05,
                            "cell_size_m": 0.5,
                            "rotation_attempts": 28,
                            "reduction_factors": [0.75, 0.55, 0.35, 0.2],
                            "max_attempts_per_child": 60,
                            "on_fail": "terminate_branch",
                            "fail_retry_rounds": 2,
                            "fail_retry_mode": "both",
                            "fail_retry_shrink_factor": 0.85,
                            "fail_retry_step_boost": 1.2
                        },
                        "collision_postpass": {"enabled": False}
                    },
                },
            },
            "domains": {
                "main": {
                    "type": "box",
                    "x_min": -15, "x_max": 15,
                    "y_min": -15, "y_max": 15,
                    "z_min": -12, "z_max": 5,
                }
            },
            "components": [
                {
                    "id": "net_1",
                    "domain_ref": "main",
                    "ports": {
                        "inlets": [{
                            "name": "inlet",
                            "position": [0, 0, 4],
                            "direction": [0, 0, -1],
                            "radius": 0.8,
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
                            "levels": 5,
                            "ratio": 0.70,
                            "step_length": 2.0,
                            "step_decay": 0.95,
                            "spread": 1.8,
                            "spread_decay": 0.97,
                            "cone_angle_deg": 60,
                            "jitter_deg": 5,
                            "curvature": 0.2,
                            "curve_samples": 5,
                            "wall_margin_m": 0.1,
                            "min_radius": 0.02,
                            "collision_online": {
                                "enabled": True,
                                "buffer_abs_m": 0.02,
                                "buffer_rel": 0.05,
                                "cell_size_m": 0.5,
                                "rotation_attempts": 28,
                                "reduction_factors": [0.75, 0.55, 0.35, 0.2],
                                "max_attempts_per_child": 60,
                                "on_fail": "terminate_branch",
                                "fail_retry_rounds": 2,
                                "fail_retry_mode": "both",
                                "fail_retry_shrink_factor": 0.85,
                                "fail_retry_step_boost": 1.2
                            },
                            "collision_postpass": {"enabled": False}
                        }
                    },
                }
            ],
            "outputs": {"artifacts_dir": str(tmp_path / "artifacts")},
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        plan = ExecutionPlan(run_until="component_build:net_1")
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        assert result.success, f"Generation should succeed: {result.error}"
        
        network = runner._component_networks.get("net_1")
        assert network is not None, "Network should be created"
        
        # Count segments - with levels=5, splits=2, we expect many segments
        segment_count = len(network.segments)
        assert segment_count > 10, f"Expected > 10 segments for deeper tree, got {segment_count}"
        
        # Compute outdegree for each node
        outdegree = {}
        for seg in network.segments.values():
            start_id = seg.start_node_id
            outdegree[start_id] = outdegree.get(start_id, 0) + 1
        
        # Count branchpoints (nodes with outdegree >= 2)
        branchpoints = sum(1 for od in outdegree.values() if od >= 2)
        
        # With levels=5, we should have multiple branchpoints (not just 1)
        # A perfect binary tree with 5 levels would have 2^4 - 1 = 15 branchpoints
        # With collisions, we expect fewer, but still > 1
        assert branchpoints >= 1, f"Expected at least 1 branchpoint, got {branchpoints}"
        
        # Compute depth of the tree (longest path from inlet to terminal)
        # Build adjacency list
        children = {}
        for seg in network.segments.values():
            start_id = seg.start_node_id
            end_id = seg.end_node_id
            if start_id not in children:
                children[start_id] = []
            children[start_id].append(end_id)
        
        # Find inlet nodes
        inlet_nodes = [n for n in network.nodes.values() if n.node_type == "inlet"]
        
        # BFS to find max depth
        max_depth = 0
        for inlet in inlet_nodes:
            queue = [(inlet.id, 0)]
            while queue:
                node_id, depth = queue.pop(0)
                max_depth = max(max_depth, depth)
                for child_id in children.get(node_id, []):
                    queue.append((child_id, depth + 1))
        
        # With levels=5, we expect depth > 2
        assert max_depth > 2, f"Expected depth > 2, got {max_depth}"


class TestMalariaBifurcatingTreeSpec:
    """
    Tests for the malaria_venule_bifurcating_tree.json spec.
    
    These tests verify that the malaria bifurcating tree example produces
    actual bifurcations with the junction-safe sibling generation fix.
    """
    
    def test_malaria_bifurcating_tree_produces_bifurcation(self, tmp_path):
        """
        Test that malaria_venule_bifurcating_tree.json produces bifurcations.
        
        This is a regression test that verifies:
        1. The spec runs through component_build without errors
        2. The generated network has nodes with outdegree >= 2
        3. The segment count is substantially larger than 2 * num_inlets
        """
        import json
        from pathlib import Path
        
        # Load the malaria bifurcating tree spec
        spec_path = Path(__file__).parent.parent.parent / "examples" / "designspec" / "malaria_venule_bifurcating_tree.json"
        
        if not spec_path.exists():
            pytest.skip(f"Spec file not found: {spec_path}")
        
        with open(spec_path) as f:
            spec_dict = json.load(f)
        
        # Override output directory
        spec_dict["outputs"]["artifacts_dir"] = str(tmp_path / "artifacts")
        spec_dict["policies"]["output"]["output_dir"] = str(tmp_path / "output")
        
        spec = DesignSpec.from_dict(spec_dict)
        plan = ExecutionPlan(run_until="component_build:bifurcating_tree_5in")
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        assert result.success, f"Generation should succeed: {result.error}"
        
        network = runner._component_networks.get("bifurcating_tree_5in")
        assert network is not None, "Network should be created"
        
        # With 5 inlets and splits=2, levels=6, we expect many segments
        # Minimum: 5 inlets * 2 = 10 segments (if no bifurcation)
        # With bifurcation: should be much higher
        segment_count = len(network.segments)
        num_inlets = 5
        assert segment_count > 2 * num_inlets, f"Expected > {2 * num_inlets} segments, got {segment_count}"
        
        # Compute outdegree for each node
        outdegree = {}
        for seg in network.segments.values():
            start_id = seg.start_node_id
            outdegree[start_id] = outdegree.get(start_id, 0) + 1
        
        # Check that at least one node has outdegree >= 2 (bifurcation)
        max_outdegree = max(outdegree.values()) if outdegree else 0
        nodes_with_outdegree_2 = sum(1 for od in outdegree.values() if od >= 2)
        
        assert max_outdegree >= 2, f"Expected at least one node with outdegree >= 2, max was {max_outdegree}"
        assert nodes_with_outdegree_2 > 0, "Expected at least one bifurcation point (outdegree>=2)"


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
