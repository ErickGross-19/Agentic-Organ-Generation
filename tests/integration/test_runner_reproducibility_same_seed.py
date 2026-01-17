"""
Test DesignSpecRunner reproducibility with same seed.

This module validates that running the same spec twice with the same seed
produces deterministic results at the report level.
"""

import pytest
import json
from pathlib import Path

from designspec.spec import DesignSpec
from designspec.runner import DesignSpecRunner, RunnerResult
from designspec.plan import ExecutionPlan


def make_deterministic_spec(tmp_path, seed=42):
    """Create a minimal spec for reproducibility testing."""
    return {
        "schema": {"name": "aog_designspec", "version": "1.0.0"},
        "meta": {"name": "reproducibility_test", "seed": seed, "input_units": "mm"},
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


class TestRunnerReproducibility:
    """Test runner reproducibility with same seed."""
    
    def test_same_seed_produces_same_stages_completed(self, tmp_path):
        """Test that same seed produces same stages completed."""
        seed = 12345
        
        # First run
        spec_dict1 = make_deterministic_spec(tmp_path / "run1", seed=seed)
        spec1 = DesignSpec.from_dict(spec_dict1)
        plan1 = ExecutionPlan(run_until="component_build:net_1")
        runner1 = DesignSpecRunner(spec1, plan=plan1, output_dir=tmp_path / "run1")
        result1 = runner1.run()
        
        # Second run
        spec_dict2 = make_deterministic_spec(tmp_path / "run2", seed=seed)
        spec2 = DesignSpec.from_dict(spec_dict2)
        plan2 = ExecutionPlan(run_until="component_build:net_1")
        runner2 = DesignSpecRunner(spec2, plan=plan2, output_dir=tmp_path / "run2")
        result2 = runner2.run()
        
        # Compare stages completed
        assert result1.stages_completed == result2.stages_completed
    
    def test_same_seed_produces_same_success_status(self, tmp_path):
        """Test that same seed produces same success status."""
        seed = 12345
        
        # First run
        spec_dict1 = make_deterministic_spec(tmp_path / "run1", seed=seed)
        spec1 = DesignSpec.from_dict(spec_dict1)
        plan1 = ExecutionPlan(run_until="component_build:net_1")
        runner1 = DesignSpecRunner(spec1, plan=plan1, output_dir=tmp_path / "run1")
        result1 = runner1.run()
        
        # Second run
        spec_dict2 = make_deterministic_spec(tmp_path / "run2", seed=seed)
        spec2 = DesignSpec.from_dict(spec_dict2)
        plan2 = ExecutionPlan(run_until="component_build:net_1")
        runner2 = DesignSpecRunner(spec2, plan=plan2, output_dir=tmp_path / "run2")
        result2 = runner2.run()
        
        # Compare success status
        assert result1.success == result2.success
    
    def test_different_seeds_may_differ(self, tmp_path):
        """Test that different seeds may produce different results."""
        # First run with seed 1
        spec_dict1 = make_deterministic_spec(tmp_path / "run1", seed=1)
        spec1 = DesignSpec.from_dict(spec_dict1)
        plan1 = ExecutionPlan(run_until="component_build:net_1")
        runner1 = DesignSpecRunner(spec1, plan=plan1, output_dir=tmp_path / "run1")
        result1 = runner1.run()
        
        # Second run with seed 2
        spec_dict2 = make_deterministic_spec(tmp_path / "run2", seed=2)
        spec2 = DesignSpec.from_dict(spec_dict2)
        plan2 = ExecutionPlan(run_until="component_build:net_1")
        runner2 = DesignSpecRunner(spec2, plan=plan2, output_dir=tmp_path / "run2")
        result2 = runner2.run()
        
        # Both should complete (may or may not have same metrics)
        assert isinstance(result1, RunnerResult)
        assert isinstance(result2, RunnerResult)
    
    def test_spec_hash_exists_and_is_string(self, tmp_path):
        """Test that spec hash exists and is a string."""
        seed = 12345
        
        spec_dict = make_deterministic_spec(tmp_path, seed=seed)
        spec = DesignSpec.from_dict(spec_dict)
        plan = ExecutionPlan(run_until="compile_policies")
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        # Verify spec hash exists and is a non-empty string
        assert result.spec_hash is not None
        assert isinstance(result.spec_hash, str)
        assert len(result.spec_hash) > 0


class TestReportLevelDeterminism:
    """Test report-level determinism."""
    
    def test_stage_report_count_is_deterministic(self, tmp_path):
        """Test that stage report count is deterministic."""
        seed = 12345
        
        # First run
        spec_dict1 = make_deterministic_spec(tmp_path / "run1", seed=seed)
        spec1 = DesignSpec.from_dict(spec_dict1)
        plan1 = ExecutionPlan(run_until="component_build:net_1")
        runner1 = DesignSpecRunner(spec1, plan=plan1, output_dir=tmp_path / "run1")
        result1 = runner1.run()
        
        # Second run
        spec_dict2 = make_deterministic_spec(tmp_path / "run2", seed=seed)
        spec2 = DesignSpec.from_dict(spec_dict2)
        plan2 = ExecutionPlan(run_until="component_build:net_1")
        runner2 = DesignSpecRunner(spec2, plan=plan2, output_dir=tmp_path / "run2")
        result2 = runner2.run()
        
        # Compare stage report counts
        assert len(result1.stage_reports) == len(result2.stage_reports)
    
    def test_stage_report_stages_are_deterministic(self, tmp_path):
        """Test that stage report stage names are deterministic."""
        seed = 12345
        
        # First run
        spec_dict1 = make_deterministic_spec(tmp_path / "run1", seed=seed)
        spec1 = DesignSpec.from_dict(spec_dict1)
        plan1 = ExecutionPlan(run_until="component_build:net_1")
        runner1 = DesignSpecRunner(spec1, plan=plan1, output_dir=tmp_path / "run1")
        result1 = runner1.run()
        
        # Second run
        spec_dict2 = make_deterministic_spec(tmp_path / "run2", seed=seed)
        spec2 = DesignSpec.from_dict(spec_dict2)
        plan2 = ExecutionPlan(run_until="component_build:net_1")
        runner2 = DesignSpecRunner(spec2, plan=plan2, output_dir=tmp_path / "run2")
        result2 = runner2.run()
        
        # Compare stage names
        stages1 = [r.stage for r in result1.stage_reports]
        stages2 = [r.stage for r in result2.stage_reports]
        assert stages1 == stages2
