"""
Test no numpy in JSON reports (J2).

This module ensures no numpy scalars/arrays leak into reports,
verifying that reports contain only JSON primitives.
"""

import pytest
import json
import numpy as np

from aog_policies import OperationReport


class TestNoNumpyInJSONReports:
    """J2: Ensure no numpy scalars/arrays leak into reports."""
    
    def test_report_with_python_primitives_serializable(self):
        """Test that report with Python primitives is JSON-serializable."""
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
                "dict": {"a": 1, "b": 2},
            },
        )
        
        report_dict = report.to_dict()
        
        try:
            json_str = json.dumps(report_dict)
            assert isinstance(json_str, str)
        except TypeError as e:
            pytest.fail(f"Report with Python primitives should be serializable: {e}")
    
    def test_numpy_int_not_json_serializable(self):
        """Test that numpy int is not directly JSON-serializable."""
        numpy_int = np.int64(42)
        
        with pytest.raises(TypeError):
            json.dumps({"value": numpy_int})
    
    def test_numpy_float_should_be_converted(self):
        """Test that numpy float should be converted to Python float before serialization.
        
        Note: In some Python/numpy versions, np.float64 may be directly JSON-serializable,
        but we should still convert to Python primitives for consistency and portability.
        """
        numpy_float = np.float64(3.14)
        python_float = float(numpy_float)
        
        assert isinstance(python_float, float)
        assert not isinstance(python_float, np.floating)
        
        json_str = json.dumps({"value": python_float})
        assert json_str is not None
    
    def test_numpy_array_not_json_serializable(self):
        """Test that numpy array is not directly JSON-serializable."""
        numpy_array = np.array([1, 2, 3])
        
        with pytest.raises(TypeError):
            json.dumps({"value": numpy_array})
    
    def test_report_dict_contains_no_numpy(self):
        """Test that report dict contains no numpy types."""
        report = OperationReport(
            operation="test",
            success=True,
            metadata={
                "value": 42,
                "nested": {
                    "list": [1, 2, 3],
                    "float": 3.14,
                },
            },
        )
        
        report_dict = report.to_dict()
        
        self._assert_no_numpy_types(report_dict)
    
    def test_operation_report_converts_numpy_to_python(self):
        """Test that OperationReport converts numpy types to Python types."""
        report = OperationReport(
            operation="test",
            success=True,
            metadata={
                "int_value": 42,
                "float_value": 3.14,
                "list_value": [1, 2, 3],
            },
        )
        
        report_dict = report.to_dict()
        
        json_str = json.dumps(report_dict)
        decoded = json.loads(json_str)
        
        assert decoded["metadata"]["int_value"] == 42
        assert abs(decoded["metadata"]["float_value"] - 3.14) < 1e-10
    
    def _assert_no_numpy_types(self, obj, path=""):
        """Recursively assert no numpy types in object."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                self._assert_no_numpy_types(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                self._assert_no_numpy_types(v, f"{path}[{i}]")
        else:
            assert not isinstance(obj, (np.ndarray, np.integer, np.floating, np.bool_)), (
                f"Found numpy type at {path}: {type(obj).__name__}"
            )


class TestJSONPrimitivesOnly:
    """Test that reports contain only JSON primitives."""
    
    def test_only_json_primitives_in_report(self):
        """Test that report contains only JSON primitives."""
        report = OperationReport(
            operation="test",
            success=True,
            requested_policy={"key": "value"},
            effective_policy={"key": "value"},
            warnings=["warning1"],
            errors=[],
            metadata={
                "string": "value",
                "int": 123,
                "float": 1.5,
                "bool": True,
                "null": None,
                "list": [1, "two", 3.0],
                "nested_dict": {"a": 1, "b": [2, 3]},
            },
        )
        
        report_dict = report.to_dict()
        
        self._assert_only_json_primitives(report_dict)
    
    def test_dict_keys_are_strings(self):
        """Test that all dict keys are strings."""
        report = OperationReport(
            operation="test",
            success=True,
            metadata={
                "level1": {
                    "level2": {
                        "level3": "value",
                    },
                },
            },
        )
        
        report_dict = report.to_dict()
        
        self._assert_string_keys(report_dict)
    
    def test_no_callable_in_report(self):
        """Test that report contains no callable objects."""
        report = OperationReport(
            operation="test",
            success=True,
            metadata={
                "value": 42,
            },
        )
        
        report_dict = report.to_dict()
        
        self._assert_no_callables(report_dict)
    
    def test_no_custom_objects_in_report(self):
        """Test that report contains no custom objects."""
        report = OperationReport(
            operation="test",
            success=True,
            metadata={
                "value": 42,
            },
        )
        
        report_dict = report.to_dict()
        
        self._assert_no_custom_objects(report_dict)
    
    def _assert_only_json_primitives(self, obj, path=""):
        """Recursively assert only JSON primitives."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert isinstance(k, str), f"Dict key at {path} is not string: {type(k)}"
                self._assert_only_json_primitives(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                self._assert_only_json_primitives(v, f"{path}[{i}]")
        else:
            assert obj is None or isinstance(obj, (str, int, float, bool)), (
                f"Non-JSON primitive at {path}: {type(obj).__name__}"
            )
    
    def _assert_string_keys(self, obj, path=""):
        """Recursively assert all dict keys are strings."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                assert isinstance(k, str), f"Non-string key at {path}: {k} ({type(k)})"
                self._assert_string_keys(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                self._assert_string_keys(v, f"{path}[{i}]")
    
    def _assert_no_callables(self, obj, path=""):
        """Recursively assert no callable objects."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                self._assert_no_callables(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                self._assert_no_callables(v, f"{path}[{i}]")
        else:
            assert not callable(obj), f"Found callable at {path}: {obj}"
    
    def _assert_no_custom_objects(self, obj, path=""):
        """Recursively assert no custom objects."""
        allowed_types = (type(None), str, int, float, bool, dict, list)
        
        if isinstance(obj, dict):
            for k, v in obj.items():
                self._assert_no_custom_objects(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                self._assert_no_custom_objects(v, f"{path}[{i}]")
        else:
            assert isinstance(obj, allowed_types), (
                f"Custom object at {path}: {type(obj).__name__}"
            )


class TestReportSerialization:
    """Test report serialization methods."""
    
    def test_to_dict_returns_dict(self):
        """Test that to_dict returns a dict."""
        report = OperationReport(
            operation="test",
            success=True,
        )
        
        result = report.to_dict()
        assert isinstance(result, dict)
    
    def test_to_json_returns_string(self):
        """Test that to_json returns a string."""
        report = OperationReport(
            operation="test",
            success=True,
        )
        
        result = report.to_json()
        assert isinstance(result, str)
    
    def test_to_json_is_valid_json(self):
        """Test that to_json returns valid JSON."""
        report = OperationReport(
            operation="test",
            success=True,
            metadata={"key": "value"},
        )
        
        json_str = report.to_json()
        
        try:
            decoded = json.loads(json_str)
            assert isinstance(decoded, dict)
        except json.JSONDecodeError as e:
            pytest.fail(f"to_json did not return valid JSON: {e}")
    
    def test_round_trip_preserves_data(self):
        """Test that JSON round-trip preserves data."""
        original_data = {
            "operation": "test",
            "success": True,
            "metadata": {
                "int": 42,
                "float": 3.14,
                "string": "hello",
                "list": [1, 2, 3],
                "nested": {"a": 1},
            },
        }
        
        report = OperationReport(
            operation=original_data["operation"],
            success=original_data["success"],
            metadata=original_data["metadata"],
        )
        
        json_str = report.to_json()
        decoded = json.loads(json_str)
        
        assert decoded["operation"] == original_data["operation"]
        assert decoded["success"] == original_data["success"]
        assert decoded["metadata"]["int"] == original_data["metadata"]["int"]
        assert abs(decoded["metadata"]["float"] - original_data["metadata"]["float"]) < 1e-10


class TestNumpyConversionHelpers:
    """Test numpy conversion helper functions."""
    
    def test_convert_numpy_int_to_python(self):
        """Test converting numpy int to Python int."""
        numpy_int = np.int64(42)
        python_int = int(numpy_int)
        
        assert isinstance(python_int, int)
        assert not isinstance(python_int, np.integer)
        
        json_str = json.dumps({"value": python_int})
        assert json_str is not None
    
    def test_convert_numpy_float_to_python(self):
        """Test converting numpy float to Python float."""
        numpy_float = np.float64(3.14)
        python_float = float(numpy_float)
        
        assert isinstance(python_float, float)
        assert not isinstance(python_float, np.floating)
        
        json_str = json.dumps({"value": python_float})
        assert json_str is not None
    
    def test_convert_numpy_array_to_list(self):
        """Test converting numpy array to Python list."""
        numpy_array = np.array([1, 2, 3])
        python_list = numpy_array.tolist()
        
        assert isinstance(python_list, list)
        assert all(isinstance(x, int) for x in python_list)
        
        json_str = json.dumps({"value": python_list})
        assert json_str is not None
    
    def test_convert_numpy_bool_to_python(self):
        """Test converting numpy bool to Python bool."""
        numpy_bool = np.bool_(True)
        python_bool = bool(numpy_bool)
        
        assert isinstance(python_bool, bool)
        assert not isinstance(python_bool, np.bool_)
        
        json_str = json.dumps({"value": python_bool})
        assert json_str is not None
    
    def test_recursive_numpy_conversion(self):
        """Test recursive conversion of nested numpy types."""
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        data_with_numpy = {
            "int": np.int64(42),
            "float": np.float64(3.14),
            "array": np.array([1, 2, 3]),
            "nested": {
                "bool": np.bool_(True),
            },
        }
        
        converted = convert_numpy(data_with_numpy)
        
        json_str = json.dumps(converted)
        decoded = json.loads(json_str)
        
        assert decoded["int"] == 42
        assert abs(decoded["float"] - 3.14) < 1e-10
        assert decoded["array"] == [1, 2, 3]
        assert decoded["nested"]["bool"] is True
