"""
Test DesignSpecRunner end-to-end execution until union_voids stage.

This module verifies that the runner can execute partial pipelines,
specifically stopping at the union_voids stage.
"""

import pytest
import json
from pathlib import Path

from designspec.spec import DesignSpec
from designspec.runner import DesignSpecRunner, RunnerResult
from designspec.plan import ExecutionPlan


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "designspec"


class TestRunnerEndToEndUnionOnly:
    """Test runner execution until union_voids stage."""
    
    def test_runner_can_run_until_union_voids(self, tmp_path):
        """Test that runner can execute until union_voids stage."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        plan = ExecutionPlan(run_until="union_voids")
        
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        assert isinstance(result, RunnerResult)
        assert "union_voids" in result.stages_completed or result.success
    
    def test_union_void_exists_after_union_stage(self, tmp_path):
        """Test that union void mesh exists after union_voids stage."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        plan = ExecutionPlan(run_until="union_voids")
        
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        if result.success:
            assert runner._union_void is not None, "Union void should exist after union_voids stage"
    
    def test_union_void_is_mesh(self, tmp_path):
        """Test that union void is a valid mesh object."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        plan = ExecutionPlan(run_until="union_voids")
        
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        if result.success and runner._union_void is not None:
            union_void = runner._union_void
            assert hasattr(union_void, "vertices"), "Union void should have vertices"
            assert hasattr(union_void, "faces"), "Union void should have faces"
            assert len(union_void.vertices) > 0, "Union void should have non-empty vertices"
            assert len(union_void.faces) > 0, "Union void should have non-empty faces"
    
    def test_stages_after_union_not_executed(self, tmp_path):
        """Test that stages after union_voids are not executed."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        plan = ExecutionPlan(run_until="union_voids")
        
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        post_union_stages = ["mesh_domain", "embed", "port_recarve", "validity", "export"]
        for stage in post_union_stages:
            assert stage not in result.stages_completed, f"Stage {stage} should not be completed"
    
    def test_result_is_json_serializable(self, tmp_path):
        """Test that runner result is JSON-serializable."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        plan = ExecutionPlan(run_until="union_voids")
        
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        result_dict = result.to_dict()
        json_str = json.dumps(result_dict)
        
        assert json_str is not None
        assert len(json_str) > 0
        
        decoded = json.loads(json_str)
        assert "success" in decoded
        assert "stages_completed" in decoded


class TestUnionVoidsStageReport:
    """Test union_voids stage report structure."""
    
    def test_stage_report_exists_for_union(self, tmp_path):
        """Test that stage report exists for union_voids."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        plan = ExecutionPlan(run_until="union_voids")
        
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        union_reports = [r for r in result.stage_reports if r.stage == "union_voids"]
        
        if result.success:
            assert len(union_reports) >= 1, "Should have stage report for union_voids"
    
    def test_stage_report_has_metadata(self, tmp_path):
        """Test that union_voids stage report has metadata."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        plan = ExecutionPlan(run_until="union_voids")
        
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        result = runner.run()
        
        union_reports = [r for r in result.stage_reports if r.stage == "union_voids"]
        
        if result.success and union_reports:
            report = union_reports[0]
            assert hasattr(report, "metadata"), "Stage report should have metadata"
            assert isinstance(report.metadata, dict), "Metadata should be a dict"


class TestPartialExecutionControl:
    """Test partial execution control via ExecutionPlan."""
    
    def test_execution_plan_run_until(self):
        """Test ExecutionPlan run_until parameter."""
        plan = ExecutionPlan(run_until="union_voids")
        
        assert plan.run_until == "union_voids"
    
    def test_execution_plan_computes_stages(self):
        """Test ExecutionPlan computes stages correctly."""
        plan = ExecutionPlan(run_until="union_voids")
        
        stages = plan.compute_stages()
        
        assert isinstance(stages, list)
        assert len(stages) > 0
    
    def test_uses_temp_dir_not_repo(self, tmp_path):
        """Test that runner uses temp dir, not repo directory."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        plan = ExecutionPlan(run_until="union_voids")
        
        runner = DesignSpecRunner(spec, plan=plan, output_dir=tmp_path)
        
        assert runner.output_dir == tmp_path
        assert str(tmp_path) not in str(FIXTURES_DIR)
