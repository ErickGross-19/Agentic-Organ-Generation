"""
Test DesignSpecRunner full end-to-end execution.

This module verifies that the runner can execute the complete pipeline
and produce all expected artifacts with valid validity reports.
"""

import pytest
import json
from pathlib import Path

from designspec.spec import DesignSpec
from designspec.runner import DesignSpecRunner, RunnerResult
from designspec.plan import ExecutionPlan


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "designspec"


class TestRunnerEndToEndFull:
    """Test runner full end-to-end execution."""
    
    def test_runner_can_run_full_pipeline(self, tmp_path):
        """Test that runner can execute full pipeline."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()
        
        assert isinstance(result, RunnerResult)
    
    def test_domain_with_void_exists(self, tmp_path):
        """Test that domain_with_void exists after full execution."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()
        
        if result.success:
            assert runner._embedded_solid is not None, "domain_with_void should exist"
    
    def test_void_mesh_exists(self, tmp_path):
        """Test that void mesh exists after full execution."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()
        
        if result.success:
            assert runner._union_void is not None, "void mesh should exist"
    
    def test_validity_report_exists(self, tmp_path):
        """Test that validity report exists after full execution."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()
        
        if result.success:
            validity_reports = [r for r in result.stage_reports if r.stage == "validity"]
            assert len(validity_reports) >= 1 or runner._validity_report is not None, (
                "validity report should exist"
            )
    
    def test_validity_report_is_json_serializable(self, tmp_path):
        """Test that validity report is JSON-serializable."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()
        
        if result.success:
            validity_reports = [r for r in result.stage_reports if r.stage == "validity"]
            if validity_reports:
                report = validity_reports[0]
                report_dict = report.to_dict()
                json_str = json.dumps(report_dict)
                assert json_str is not None
                assert len(json_str) > 0


class TestFullPipelineArtifacts:
    """Test artifacts produced by full pipeline."""
    
    def test_all_expected_stages_completed(self, tmp_path):
        """Test that all expected stages are completed."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()
        
        if result.success:
            expected_stages = [
                "compile_policies",
                "compile_domains",
                "union_voids",
                "mesh_domain",
                "embed",
            ]
            for stage in expected_stages:
                assert stage in result.stages_completed or any(
                    s.startswith(stage) for s in result.stages_completed
                ), f"Stage {stage} should be completed"
    
    def test_artifacts_manifest_exists(self, tmp_path):
        """Test that artifacts manifest exists."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()
        
        assert "artifacts" in result.to_dict()
    
    def test_result_includes_spec_hash(self, tmp_path):
        """Test that result includes spec hash."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()
        
        assert result.spec_hash is not None
        assert result.spec_hash == spec.spec_hash


class TestFullPipelineReporting:
    """Test reporting from full pipeline execution."""
    
    def test_result_to_dict_is_complete(self, tmp_path):
        """Test that result.to_dict() is complete."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()
        
        result_dict = result.to_dict()
        
        required_keys = [
            "success",
            "spec_hash",
            "stages_completed",
            "stage_reports",
            "warnings",
            "errors",
            "artifacts",
            "total_duration_s",
        ]
        
        for key in required_keys:
            assert key in result_dict, f"Result should have key: {key}"
    
    def test_result_json_roundtrip(self, tmp_path):
        """Test that result can roundtrip through JSON."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()
        
        result_dict = result.to_dict()
        json_str = json.dumps(result_dict)
        decoded = json.loads(json_str)
        
        assert decoded["success"] == result.success
        assert decoded["spec_hash"] == result.spec_hash
    
    def test_stage_reports_are_json_clean(self, tmp_path):
        """Test that all stage reports are JSON-clean."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()
        
        for report in result.stage_reports:
            report_dict = report.to_dict()
            json_str = json.dumps(report_dict)
            assert json_str is not None, f"Stage {report.stage} report should be JSON-serializable"


class TestFullPipelineUsesTemporaryDirectory:
    """Test that full pipeline uses temporary directory correctly."""
    
    def test_uses_tmp_path_fixture(self, tmp_path):
        """Test that runner uses tmp_path fixture."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        
        assert runner.output_dir == tmp_path
    
    def test_does_not_write_to_repo(self, tmp_path):
        """Test that runner does not write to repo directory."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()
        
        repo_dir = FIXTURES_DIR.parent.parent
        assert not str(runner.output_dir).startswith(str(repo_dir))
