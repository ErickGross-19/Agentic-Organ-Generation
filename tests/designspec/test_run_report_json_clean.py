"""Tests for RunReport JSON serialization cleanliness."""

import pytest
import json
from designspec.reports.run_report import (
    RunReport,
    MetaInfo,
    EnvInfo,
    HashInfo,
    StageReportEntry,
    ArtifactEntry,
)
from designspec.reports.serializers import make_json_safe, to_json, compute_content_hash


class TestMakeJsonSafe:
    """Tests for make_json_safe function."""
    
    def test_primitives_pass_through(self):
        assert make_json_safe(42) == 42
        assert make_json_safe(3.14) == 3.14
        assert make_json_safe("hello") == "hello"
        assert make_json_safe(True) is True
        assert make_json_safe(None) is None
    
    def test_dict_converted(self):
        result = make_json_safe({"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}
    
    def test_list_converted(self):
        result = make_json_safe([1, 2, 3])
        assert result == [1, 2, 3]
    
    def test_nested_structure_converted(self):
        result = make_json_safe({"a": [1, {"b": 2}]})
        assert result == {"a": [1, {"b": 2}]}
    
    def test_numpy_scalar_converted(self):
        try:
            import numpy as np
            
            result = make_json_safe(np.int64(42))
            assert result == 42
            assert isinstance(result, int)
            
            result = make_json_safe(np.float64(3.14))
            assert result == pytest.approx(3.14)
            assert isinstance(result, float)
            
        except ImportError:
            pytest.skip("numpy not available")
    
    def test_numpy_array_converted(self):
        try:
            import numpy as np
            
            result = make_json_safe(np.array([1, 2, 3]))
            assert result == [1, 2, 3]
            assert isinstance(result, list)
            
        except ImportError:
            pytest.skip("numpy not available")
    
    def test_numpy_bool_converted(self):
        try:
            import numpy as np
            
            result = make_json_safe(np.bool_(True))
            assert result is True
            assert isinstance(result, bool)
            
        except ImportError:
            pytest.skip("numpy not available")
    
    def test_bytes_converted_to_base64(self):
        result = make_json_safe(b"hello")
        assert isinstance(result, str)
    
    def test_object_with_to_dict_converted(self):
        class MockObject:
            def to_dict(self):
                return {"key": "value"}
        
        result = make_json_safe(MockObject())
        assert result == {"key": "value"}


class TestToJson:
    """Tests for to_json function."""
    
    def test_simple_dict_serialized(self):
        result = to_json({"a": 1, "b": 2})
        parsed = json.loads(result)
        assert parsed == {"a": 1, "b": 2}
    
    def test_sorted_keys(self):
        result = to_json({"z": 1, "a": 2})
        assert result.index('"a"') < result.index('"z"')
    
    def test_numpy_values_serialized(self):
        try:
            import numpy as np
            
            data = {"value": np.int64(42), "array": np.array([1, 2, 3])}
            result = to_json(data)
            parsed = json.loads(result)
            
            assert parsed["value"] == 42
            assert parsed["array"] == [1, 2, 3]
            
        except ImportError:
            pytest.skip("numpy not available")


class TestComputeContentHash:
    """Tests for compute_content_hash function."""
    
    def test_same_content_same_hash(self):
        data = {"a": 1, "b": 2}
        hash1 = compute_content_hash(data)
        hash2 = compute_content_hash(data)
        assert hash1 == hash2
    
    def test_different_key_order_same_hash(self):
        data1 = {"a": 1, "b": 2}
        data2 = {"b": 2, "a": 1}
        hash1 = compute_content_hash(data1)
        hash2 = compute_content_hash(data2)
        assert hash1 == hash2
    
    def test_different_content_different_hash(self):
        data1 = {"a": 1}
        data2 = {"a": 2}
        hash1 = compute_content_hash(data1)
        hash2 = compute_content_hash(data2)
        assert hash1 != hash2
    
    def test_hash_is_hex_string(self):
        data = {"a": 1}
        result = compute_content_hash(data)
        assert all(c in "0123456789abcdef" for c in result)


class TestRunReportToDict:
    """Tests for RunReport.to_dict() method."""
    
    def test_run_report_to_dict(self):
        report = RunReport(
            success=True,
            meta=MetaInfo(seed=42, input_units="m"),
            env=EnvInfo.capture(),
            hashes=HashInfo(spec_hash="abc123"),
        )
        
        result = report.to_dict()
        
        assert isinstance(result, dict)
        assert result["success"] is True
        assert result["meta"]["seed"] == 42
    
    def test_run_report_to_dict_no_numpy_scalars(self):
        try:
            import numpy as np
            
            report = RunReport(
                success=True,
                meta=MetaInfo(seed=42),
            )
            report.stages.append(StageReportEntry(
                stage="test",
                success=True,
                metrics={"count": np.int64(100)},
            ))
            
            result = report.to_dict()
            
            assert isinstance(result["stages"][0]["metrics"]["count"], int)
            assert not isinstance(result["stages"][0]["metrics"]["count"], np.integer)
            
        except ImportError:
            pytest.skip("numpy not available")


class TestRunReportToJson:
    """Tests for RunReport.to_json() method."""
    
    def test_run_report_to_json_succeeds(self):
        report = RunReport(
            success=True,
            meta=MetaInfo(seed=42, input_units="m"),
            env=EnvInfo.capture(),
            hashes=HashInfo(spec_hash="abc123"),
        )
        
        result = report.to_json()
        
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["success"] is True
    
    def test_run_report_to_json_no_numpy_leak(self):
        try:
            import numpy as np
            
            report = RunReport(
                success=True,
                meta=MetaInfo(seed=42),
            )
            report.stages.append(StageReportEntry(
                stage="test",
                success=True,
                duration_s=np.float64(1.5),
                metrics={
                    "count": np.int64(100),
                    "values": np.array([1, 2, 3]),
                },
            ))
            
            result = report.to_json()
            parsed = json.loads(result)
            
            assert parsed["stages"][0]["duration_s"] == 1.5
            assert parsed["stages"][0]["metrics"]["count"] == 100
            assert parsed["stages"][0]["metrics"]["values"] == [1, 2, 3]
            
        except ImportError:
            pytest.skip("numpy not available")
    
    def test_run_report_json_dumps_works(self):
        report = RunReport(
            success=True,
            meta=MetaInfo(seed=42),
            env=EnvInfo.capture(),
            hashes=HashInfo(spec_hash="abc123"),
            stages=[
                StageReportEntry(stage="compile_policies", success=True),
                StageReportEntry(stage="compile_domains", success=True),
            ],
            artifacts={
                "void_mesh": ArtifactEntry(name="void_mesh", stage="union_voids"),
            },
        )
        
        result = json.dumps(report.to_dict())
        
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed["success"] is True


class TestRunReportRoundtrip:
    """Tests for RunReport serialization roundtrip."""
    
    def test_run_report_roundtrip(self):
        original = RunReport(
            success=True,
            meta=MetaInfo(seed=42, input_units="mm", spec_name="test"),
            env=EnvInfo.capture(),
            hashes=HashInfo(spec_hash="abc123"),
            stages=[
                StageReportEntry(stage="compile_policies", success=True, duration_s=0.5),
            ],
            warnings=["test warning"],
        )
        
        json_str = original.to_json()
        restored = RunReport.from_json(json_str)
        
        assert restored.success == original.success
        assert restored.meta.seed == original.meta.seed
        assert restored.meta.spec_name == original.meta.spec_name
        assert len(restored.stages) == len(original.stages)
        assert restored.warnings == original.warnings


class TestEnvInfoCapture:
    """Tests for EnvInfo.capture() method."""
    
    def test_env_info_capture_has_python_version(self):
        env = EnvInfo.capture()
        assert len(env.python_version) > 0
    
    def test_env_info_capture_has_platform(self):
        env = EnvInfo.capture()
        assert len(env.platform_system) > 0
    
    def test_env_info_capture_has_package_versions(self):
        env = EnvInfo.capture()
        assert isinstance(env.package_versions, dict)


class TestHashInfoCapture:
    """Tests for HashInfo.capture() method."""
    
    def test_hash_info_capture_has_spec_hash(self):
        hash_info = HashInfo.capture(spec_hash="test123")
        assert hash_info.spec_hash == "test123"
    
    def test_hash_info_to_dict(self):
        hash_info = HashInfo(spec_hash="abc123", repo_commit="def456")
        result = hash_info.to_dict()
        
        assert result["spec_hash"] == "abc123"
        assert result["repo_commit"] == "def456"
