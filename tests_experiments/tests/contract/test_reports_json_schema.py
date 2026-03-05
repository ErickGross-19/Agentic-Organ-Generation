"""
Test reports JSON schema (H3).

This module verifies that reports are JSON serializable and include
requested vs effective policies.
"""

import pytest
import json
import numpy as np

from aog_policies import (
    OperationReport,
    ValidationPolicy,
    EmbeddingPolicy,
    ResolutionPolicy,
)


class TestReportsJSONSchema:
    """H3: Reports are JSON serializable and include requested vs effective policies."""
    
    def test_operation_report_json_serializable(self):
        """Test that OperationReport is JSON-serializable."""
        report = OperationReport(
            operation="test_operation",
            success=True,
            requested_policy={"key": "value"},
            effective_policy={"key": "modified_value"},
            warnings=["test warning"],
            errors=[],
            metadata={"metric": 123},
        )
        
        report_dict = report.to_dict()
        
        try:
            json_str = json.dumps(report_dict)
            assert isinstance(json_str, str)
        except TypeError as e:
            pytest.fail(f"OperationReport is not JSON-serializable: {e}")
    
    def test_report_includes_operation_name(self):
        """Test that report includes operation name."""
        report = OperationReport(
            operation="embedding",
            success=True,
        )
        
        report_dict = report.to_dict()
        assert "operation" in report_dict
        assert report_dict["operation"] == "embedding"
    
    def test_report_includes_requested_policy(self):
        """Test that report includes requested_policy."""
        requested = {
            "voxel_pitch": 1e-5,
            "max_voxels": 10_000_000,
        }
        
        report = OperationReport(
            operation="embedding",
            success=True,
            requested_policy=requested,
        )
        
        report_dict = report.to_dict()
        assert "requested_policy" in report_dict
        assert report_dict["requested_policy"]["voxel_pitch"] == 1e-5
    
    def test_report_includes_effective_policy(self):
        """Test that report includes effective_policy."""
        effective = {
            "voxel_pitch": 2e-5,
            "max_voxels": 10_000_000,
        }
        
        report = OperationReport(
            operation="embedding",
            success=True,
            effective_policy=effective,
        )
        
        report_dict = report.to_dict()
        assert "effective_policy" in report_dict
        assert report_dict["effective_policy"]["voxel_pitch"] == 2e-5
    
    def test_report_includes_warnings(self):
        """Test that report includes warnings."""
        report = OperationReport(
            operation="embedding",
            success=True,
            warnings=[
                "Pitch relaxed from 1e-5 to 2e-5",
                "Budget exceeded, adjusted parameters",
            ],
        )
        
        report_dict = report.to_dict()
        assert "warnings" in report_dict
        assert len(report_dict["warnings"]) == 2
    
    def test_report_includes_errors(self):
        """Test that report includes errors."""
        report = OperationReport(
            operation="embedding",
            success=False,
            errors=[
                "Mesh is not watertight",
                "Port validation failed",
            ],
        )
        
        report_dict = report.to_dict()
        assert "errors" in report_dict
        assert len(report_dict["errors"]) == 2
    
    def test_report_includes_key_metrics(self):
        """Test that report includes key metrics."""
        report = OperationReport(
            operation="embedding",
            success=True,
            metadata={
                "pitch_relax_factor": 2.0,
                "component_count": 1,
                "voxel_count": 5_000_000,
                "time_s": 10.5,
            },
        )
        
        report_dict = report.to_dict()
        assert "metadata" in report_dict
        assert report_dict["metadata"]["pitch_relax_factor"] == 2.0
        assert report_dict["metadata"]["component_count"] == 1


class TestGenerationReportSchema:
    """Test generation report schema."""
    
    def test_generation_report_serializable(self):
        """Test that generation report is JSON-serializable."""
        report = OperationReport(
            operation="generation",
            success=True,
            requested_policy={
                "backend": "cco_hybrid",
                "target_terminals": 100,
                "seed": 42,
            },
            effective_policy={
                "backend": "cco_hybrid",
                "target_terminals": 100,
                "seed": 42,
            },
            metadata={
                "terminals_generated": 98,
                "iterations": 500,
                "time_s": 30.0,
            },
        )
        
        json_str = json.dumps(report.to_dict())
        decoded = json.loads(json_str)
        
        assert decoded["operation"] == "generation"
        assert decoded["metadata"]["terminals_generated"] == 98


class TestEmbeddingReportSchema:
    """Test embedding report schema."""
    
    def test_embedding_report_serializable(self):
        """Test that embedding report is JSON-serializable."""
        policy = EmbeddingPolicy(
            preserve_ports_enabled=True,
            max_voxels=10_000_000,
        )
        
        report = OperationReport(
            operation="embedding",
            success=True,
            requested_policy=policy.to_dict(),
            effective_policy=policy.to_dict(),
            metadata={
                "voxel_count": 8_000_000,
                "pitch": 2e-5,
                "outputs": {
                    "domain_with_void": {"exists": True},
                    "void_mesh": {"exists": True},
                },
            },
        )
        
        json_str = json.dumps(report.to_dict())
        decoded = json.loads(json_str)
        
        assert decoded["operation"] == "embedding"
        assert decoded["metadata"]["voxel_count"] == 8_000_000


class TestValidityReportSchema:
    """Test validity report schema."""
    
    def test_validity_report_serializable(self):
        """Test that validity report is JSON-serializable."""
        policy = ValidationPolicy(
            check_watertight=True,
            check_components=True,
        )
        
        report = OperationReport(
            operation="validity",
            success=True,
            requested_policy=policy.to_dict(),
            effective_policy=policy.to_dict(),
            metadata={
                "checks": {
                    "watertight": {"passed": True},
                    "components": {"passed": True, "count": 1},
                },
            },
        )
        
        json_str = json.dumps(report.to_dict())
        decoded = json.loads(json_str)
        
        assert decoded["operation"] == "validity"
        assert decoded["metadata"]["checks"]["watertight"]["passed"]


class TestRequestedVsEffectivePolicy:
    """Test requested vs effective policy pattern."""
    
    def test_requested_and_effective_differ_when_relaxed(self):
        """Test that requested and effective differ when relaxation occurs."""
        report = OperationReport(
            operation="embedding",
            success=True,
            requested_policy={
                "voxel_pitch": 1e-5,
                "max_voxels": 10_000_000,
            },
            effective_policy={
                "voxel_pitch": 2e-5,
                "max_voxels": 10_000_000,
            },
            warnings=["Pitch relaxed due to voxel budget"],
        )
        
        report_dict = report.to_dict()
        
        assert report_dict["requested_policy"]["voxel_pitch"] != report_dict["effective_policy"]["voxel_pitch"]
        assert len(report_dict["warnings"]) > 0
    
    def test_requested_and_effective_same_when_no_relaxation(self):
        """Test that requested and effective are same when no relaxation."""
        policy = {
            "voxel_pitch": 1e-5,
            "max_voxels": 100_000_000,
        }
        
        report = OperationReport(
            operation="embedding",
            success=True,
            requested_policy=policy,
            effective_policy=policy,
        )
        
        report_dict = report.to_dict()
        
        assert report_dict["requested_policy"] == report_dict["effective_policy"]
    
    def test_effective_policy_includes_derived_values(self):
        """Test that effective policy includes derived values."""
        report = OperationReport(
            operation="embedding",
            success=True,
            requested_policy={
                "voxel_pitch": 1e-5,
            },
            effective_policy={
                "voxel_pitch": 2e-5,
                "relax_factor": 2.0,
                "voxel_count": 8_000_000,
            },
        )
        
        effective = report.to_dict()["effective_policy"]
        
        assert "relax_factor" in effective
        assert "voxel_count" in effective


class TestReportJSONPrimitives:
    """Test that reports contain only JSON primitives."""
    
    def test_no_numpy_types_in_report(self):
        """Test that report contains no numpy types."""
        report = OperationReport(
            operation="test",
            success=True,
            metadata={
                "int_value": 123,
                "float_value": 1.5,
                "list_value": [1, 2, 3],
                "dict_value": {"a": 1},
            },
        )
        
        report_dict = report.to_dict()
        
        def check_no_numpy(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    check_no_numpy(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    check_no_numpy(v, f"{path}[{i}]")
            else:
                assert not isinstance(obj, (np.ndarray, np.integer, np.floating)), (
                    f"Found numpy type at {path}: {type(obj)}"
                )
        
        check_no_numpy(report_dict)
    
    def test_only_json_primitives(self):
        """Test that report contains only JSON primitives."""
        report = OperationReport(
            operation="test",
            success=True,
            metadata={
                "string": "value",
                "int": 123,
                "float": 1.5,
                "bool": True,
                "null": None,
                "list": [1, 2, 3],
                "dict": {"a": 1},
            },
        )
        
        report_dict = report.to_dict()
        
        def check_json_primitives(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    assert isinstance(k, str), f"Dict key at {path} is not string: {type(k)}"
                    check_json_primitives(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    check_json_primitives(v, f"{path}[{i}]")
            else:
                assert obj is None or isinstance(obj, (str, int, float, bool)), (
                    f"Non-JSON primitive at {path}: {type(obj)}"
                )
        
        check_json_primitives(report_dict)
    
    def test_report_to_json_method(self):
        """Test OperationReport.to_json() method."""
        report = OperationReport(
            operation="test",
            success=True,
            metadata={"key": "value"},
        )
        
        json_str = report.to_json()
        
        assert isinstance(json_str, str)
        
        decoded = json.loads(json_str)
        assert decoded["operation"] == "test"
