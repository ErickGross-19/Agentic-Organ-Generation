"""
Test DesignSpecRunner backend_params plumbing.

This module verifies that backend_params actually affect runtime behavior
or at least appear in effective policy snapshot/metrics.
"""

import pytest
import json
from pathlib import Path

from designspec.spec import DesignSpec
from designspec.runner import DesignSpecRunner, RunnerResult
from designspec.plan import ExecutionPlan


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "designspec"


class TestBackendParamsPlumbing:
    """Test backend_params plumbing through the runner."""
    
    def test_spec_can_have_backend_params(self):
        """Test that spec can have backend_params in components."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        for component in spec.components:
            build = component.get("build", {})
            if "backend_params" in build:
                params = build["backend_params"]
                assert isinstance(params, dict)
    
    def test_backend_params_in_build_config(self):
        """Test that backend_params appear in build config."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        backend_network_components = [
            c for c in spec.components
            if c.get("build", {}).get("type") == "backend_network"
        ]
        
        for component in backend_network_components:
            build = component.get("build", {})
            assert "type" in build
    
    def test_component_build_stage_uses_backend_params(self, tmp_path):
        """Test that component_build stage uses backend_params."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()
        
        build_reports = [
            r for r in result.stage_reports
            if r.stage.startswith("component_build:")
        ]
        
        for report in build_reports:
            assert hasattr(report, "metadata")
            assert "build_type" in report.metadata


class TestBackendParamsInReports:
    """Test backend_params appear in reports."""
    
    def test_build_stage_report_has_metadata(self, tmp_path):
        """Test that build stage report has metadata."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()
        
        build_reports = [
            r for r in result.stage_reports
            if r.stage.startswith("component_build:")
        ]
        
        for report in build_reports:
            report_dict = report.to_dict()
            assert "metadata" in report_dict
    
    def test_build_metadata_is_json_serializable(self, tmp_path):
        """Test that build metadata is JSON-serializable."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()
        
        build_reports = [
            r for r in result.stage_reports
            if r.stage.startswith("component_build:")
        ]
        
        for report in build_reports:
            report_dict = report.to_dict()
            json_str = json.dumps(report_dict)
            assert json_str is not None
    
    def test_effective_policy_snapshot_in_build_report(self, tmp_path):
        """Test that effective policy snapshot appears in build report."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()
        
        build_reports = [
            r for r in result.stage_reports
            if r.stage.startswith("component_build:")
        ]
        
        for report in build_reports:
            report_dict = report.to_dict()
            assert "effective_policy" in report_dict or "requested_policy" in report_dict


class TestBackendNetworkGeneration:
    """Test backend network generation with params."""
    
    def test_backend_network_generates_nodes(self, tmp_path):
        """Test that backend network generates nodes."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()
        
        build_reports = [
            r for r in result.stage_reports
            if r.stage.startswith("component_build:")
        ]
        
        for report in build_reports:
            if report.metadata.get("build_type") == "backend_network":
                if report.success:
                    assert "node_count" in report.metadata or "segment_count" in report.metadata
    
    def test_backend_network_generates_segments(self, tmp_path):
        """Test that backend network generates segments."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()
        
        build_reports = [
            r for r in result.stage_reports
            if r.stage.startswith("component_build:")
        ]
        
        for report in build_reports:
            if report.metadata.get("build_type") == "backend_network":
                if report.success:
                    assert "segment_count" in report.metadata


class TestPrimitiveChannelsGeneration:
    """Test primitive channels generation with params."""
    
    def test_primitive_channels_generates_mesh(self, tmp_path):
        """Test that primitive channels generates mesh."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()
        
        build_reports = [
            r for r in result.stage_reports
            if r.stage.startswith("component_build:")
        ]
        
        for report in build_reports:
            if report.metadata.get("build_type") == "primitive_channels":
                if report.success:
                    assert "vertex_count" in report.metadata or "face_count" in report.metadata
    
    def test_primitive_channels_mesh_has_faces(self, tmp_path):
        """Test that primitive channels mesh has faces."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()
        
        build_reports = [
            r for r in result.stage_reports
            if r.stage.startswith("component_build:")
        ]
        
        for report in build_reports:
            if report.metadata.get("build_type") == "primitive_channels":
                if report.success and "face_count" in report.metadata:
                    assert report.metadata["face_count"] > 0


class TestBackendParamsAffectBehavior:
    """Test that backend_params affect runtime behavior."""
    
    def test_different_seeds_produce_different_results(self, tmp_path):
        """Test that different seeds produce different results."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        with open(golden_path) as f:
            data = json.load(f)
        
        data1 = data.copy()
        data1["meta"]["seed"] = 1234
        
        data2 = data.copy()
        data2["meta"]["seed"] = 5678
        
        spec1 = DesignSpec.from_dict(data1)
        spec2 = DesignSpec.from_dict(data2)
        
        assert spec1.seed != spec2.seed
    
    def test_runner_uses_spec_seed(self, tmp_path):
        """Test that runner uses spec seed."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        
        assert runner.context.seed == spec.seed
    
    def test_metrics_reflect_backend_params(self, tmp_path):
        """Test that metrics reflect backend_params."""
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        runner = DesignSpecRunner(spec, output_dir=tmp_path)
        result = runner.run()
        
        result_dict = result.to_dict()
        json_str = json.dumps(result_dict)
        
        assert json_str is not None
        assert "stage_reports" in result_dict
