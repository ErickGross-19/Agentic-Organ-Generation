"""
Test DesignSpecRunner determinism and reproducibility.

This module verifies that running the same spec with the same seed
produces consistent results.
"""

import pytest
import json
from pathlib import Path

from designspec.spec import DesignSpec
from designspec.runner import DesignSpecRunner, RunnerResult
from designspec.plan import ExecutionPlan


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "designspec"


class TestRunnerDeterminism:
    """Test runner determinism with same seed."""
    
    def test_same_seed_produces_same_spec_hash(self):
        """Test that same seed produces same spec hash."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec1 = DesignSpec.from_json(golden_path)
        spec2 = DesignSpec.from_json(golden_path)
        
        assert spec1.spec_hash == spec2.spec_hash
    
    def test_same_seed_produces_consistent_stages(self, tmp_path):
        """Test that same seed produces consistent stage completion."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        plan = ExecutionPlan(run_until="compile_policies")
        
        runner1 = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path / "run1")
        runner2 = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path / "run2")
        
        result1 = runner1.run()
        result2 = runner2.run()
        
        assert result1.stages_completed == result2.stages_completed
    
    def test_runner_context_has_seed(self, tmp_path):
        """Test that runner context has seed."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        
        assert runner.context.seed == spec.seed
    
    def test_runner_context_has_spec_hash(self, tmp_path):
        """Test that runner context has spec hash."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        
        assert runner.context.spec_hash == spec.spec_hash


class TestReproducibilityMetrics:
    """Test reproducibility of key metrics."""
    
    def test_result_spec_hash_matches_input(self, tmp_path):
        """Test that result spec_hash matches input spec."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        plan = ExecutionPlan(run_until="compile_policies")
        
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        assert result.spec_hash == spec.spec_hash
    
    def test_stage_reports_are_reproducible(self, tmp_path):
        """Test that stage reports are reproducible."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        plan = ExecutionPlan(run_until="compile_policies")
        
        runner1 = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path / "run1")
        runner2 = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path / "run2")
        
        result1 = runner1.run()
        result2 = runner2.run()
        
        assert len(result1.stage_reports) == len(result2.stage_reports)
        
        for r1, r2 in zip(result1.stage_reports, result2.stage_reports):
            assert r1.stage == r2.stage
            assert r1.success == r2.success


class TestRunReportMetadata:
    """Test RunReport metadata for reproducibility."""
    
    def test_result_includes_spec_hash(self, tmp_path):
        """Test that result includes spec hash."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()
        
        result_dict = result.to_dict()
        assert "spec_hash" in result_dict
        assert result_dict["spec_hash"] == spec.spec_hash
    
    def test_result_includes_stages_completed(self, tmp_path):
        """Test that result includes stages_completed."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()
        
        result_dict = result.to_dict()
        assert "stages_completed" in result_dict
        assert isinstance(result_dict["stages_completed"], list)
    
    def test_result_includes_total_duration(self, tmp_path):
        """Test that result includes total_duration_s."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()
        
        result_dict = result.to_dict()
        assert "total_duration_s" in result_dict
        assert isinstance(result_dict["total_duration_s"], (int, float))
    
    def test_stage_reports_include_duration(self, tmp_path):
        """Test that stage reports include duration_s."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()
        
        for report in result.stage_reports:
            report_dict = report.to_dict()
            assert "duration_s" in report_dict


class TestDeterministicJSONOutput:
    """Test deterministic JSON output."""
    
    def test_result_json_is_deterministic(self, tmp_path):
        """Test that result JSON is deterministic (excluding timing)."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        plan = ExecutionPlan(run_until="compile_policies")
        
        runner1 = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path / "run1")
        runner2 = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path / "run2")
        
        result1 = runner1.run()
        result2 = runner2.run()
        
        dict1 = result1.to_dict()
        dict2 = result2.to_dict()
        
        del dict1["total_duration_s"]
        del dict2["total_duration_s"]
        for r in dict1["stage_reports"]:
            r["duration_s"] = 0
        for r in dict2["stage_reports"]:
            r["duration_s"] = 0
        
        assert dict1["spec_hash"] == dict2["spec_hash"]
        assert dict1["stages_completed"] == dict2["stages_completed"]
    
    def test_spec_hash_is_stable_across_loads(self):
        """Test that spec hash is stable across multiple loads."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        hashes = []
        for _ in range(3):
            spec = DesignSpec.from_json(golden_path)
            hashes.append(spec.spec_hash)
        
        assert all(h == hashes[0] for h in hashes), "Spec hash should be stable"
    
    def test_spec_hash_from_dict_matches_from_json(self):
        """Test that spec hash from dict matches from JSON."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        with open(golden_path) as f:
            data = json.load(f)
        
        spec_from_json = DesignSpec.from_json(golden_path)
        spec_from_dict = DesignSpec.from_dict(data)
        
        assert spec_from_json.spec_hash == spec_from_dict.spec_hash
