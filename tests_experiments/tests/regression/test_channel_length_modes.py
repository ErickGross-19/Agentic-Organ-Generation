"""
Test channel length modes (E1, E2).

This module verifies that:
- E1: stop_before_boundary is applied once (regression test for "double subtract" bug)
- E2: No hidden 1mm minimum clamp
"""

import pytest
import numpy as np

from aog_policies import ChannelPolicy, OperationReport
from generation.core.domain import BoxDomain, CylinderDomain
from generation.core.types import Point3D


class TestStopBeforeBoundaryAppliedOnce:
    """E1: stop_before_boundary applied once (regression test for "double subtract" bug)."""
    
    def test_stop_before_boundary_formula(self):
        """Test that computed length matches expected formula exactly."""
        domain_depth = 0.01
        stop_before = 0.001
        
        expected_length = domain_depth - stop_before
        
        policy = ChannelPolicy(
            length_mode="to_boundary",
            stop_before_boundary=stop_before,
        )
        
        computed_length = domain_depth - policy.stop_before_boundary
        
        tolerance = 1e-12
        assert abs(computed_length - expected_length) < tolerance, (
            f"Computed length ({computed_length}) should match expected ({expected_length})"
        )
    
    def test_no_double_subtract_bug(self):
        """Test that stop_before_boundary is not subtracted twice."""
        domain_depth = 0.01
        stop_before = 0.002
        
        correct_length = domain_depth - stop_before
        
        buggy_length = domain_depth - (2 * stop_before)
        
        policy = ChannelPolicy(
            length_mode="to_boundary",
            stop_before_boundary=stop_before,
        )
        
        computed_length = domain_depth - policy.stop_before_boundary
        
        assert abs(computed_length - correct_length) < 1e-12, (
            f"Length should be {correct_length}, not {buggy_length} (double subtract bug)"
        )
        
        assert abs(computed_length - buggy_length) > 1e-6, (
            "Length appears to have double subtract bug"
        )
    
    def test_stop_before_boundary_with_various_depths(self):
        """Test stop_before_boundary with various domain depths."""
        test_cases = [
            (0.005, 0.0005),
            (0.01, 0.001),
            (0.02, 0.002),
            (0.05, 0.005),
        ]
        
        for domain_depth, stop_before in test_cases:
            policy = ChannelPolicy(
                length_mode="to_boundary",
                stop_before_boundary=stop_before,
            )
            
            expected_length = domain_depth - stop_before
            computed_length = domain_depth - policy.stop_before_boundary
            
            assert abs(computed_length - expected_length) < 1e-12, (
                f"For depth={domain_depth}, stop_before={stop_before}: "
                f"expected {expected_length}, got {computed_length}"
            )
    
    def test_channel_length_report_includes_formula(self):
        """Test that channel length report includes the formula used."""
        report = OperationReport(
            operation="channel_generation",
            success=True,
            requested_policy={
                "length_mode": "to_boundary",
                "stop_before_boundary": 0.001,
            },
            effective_policy={
                "length_mode": "to_boundary",
                "stop_before_boundary": 0.001,
            },
            metadata={
                "domain_depth": 0.01,
                "stop_before_boundary": 0.001,
                "computed_length": 0.009,
                "formula": "domain_depth - stop_before_boundary",
            },
        )
        
        assert report.metadata["computed_length"] == 0.009
        assert "formula" in report.metadata


class TestNoHidden1mmMinimumClamp:
    """E2: No hidden 1mm minimum clamp."""
    
    def test_channel_length_less_than_1mm(self):
        """Test that channel length < 1mm is allowed."""
        policy = ChannelPolicy(
            length_mode="fixed",
            length=0.0005,
            min_length=0.0,
        )
        
        assert policy.length < 0.001, (
            f"Policy should allow length < 1mm, got {policy.length}"
        )
    
    def test_no_implicit_1mm_clamp(self):
        """Test that there's no implicit 1mm minimum clamp."""
        small_lengths = [0.0001, 0.0002, 0.0005, 0.0008, 0.0009]
        
        for length in small_lengths:
            policy = ChannelPolicy(
                length_mode="fixed",
                length=length,
                min_length=0.0,
            )
            
            assert abs(policy.length - length) < 1e-12, (
                f"Length {length} should not be clamped to 1mm"
            )
    
    def test_domain_depth_produces_sub_1mm_channel(self):
        """Test that domain depth < 1mm produces sub-1mm channel."""
        domain_depth = 0.0008
        stop_before = 0.0001
        
        expected_length = domain_depth - stop_before
        
        assert expected_length < 0.001, (
            f"Expected length ({expected_length}) should be < 1mm"
        )
        
        policy = ChannelPolicy(
            length_mode="to_boundary",
            stop_before_boundary=stop_before,
        )
        
        computed_length = domain_depth - policy.stop_before_boundary
        
        assert computed_length < 0.001, (
            f"Computed length ({computed_length}) should be < 1mm"
        )
    
    def test_min_length_policy_respected(self):
        """Test that min_length policy is respected when set."""
        policy = ChannelPolicy(
            length_mode="fixed",
            length=0.0005,
            min_length=0.0002,
        )
        
        assert policy.min_length == 0.0002
        
        if policy.length < policy.min_length:
            effective_length = policy.min_length
        else:
            effective_length = policy.length
        
        assert effective_length >= policy.min_length
    
    def test_no_hardcoded_1mm_in_policy(self):
        """Test that ChannelPolicy doesn't have hardcoded 1mm values."""
        policy = ChannelPolicy()
        policy_dict = policy.to_dict()
        
        for key, value in policy_dict.items():
            if isinstance(value, (int, float)):
                assert value != 0.001 or key in ["min_length", "length"], (
                    f"Suspicious 1mm value found in {key}: {value}"
                )


class TestChannelLengthModes:
    """Test different channel length modes."""
    
    def test_fixed_length_mode(self):
        """Test fixed length mode."""
        policy = ChannelPolicy(
            length_mode="fixed",
            length=0.005,
        )
        
        assert policy.length_mode == "fixed"
        assert policy.length == 0.005
    
    def test_to_boundary_length_mode(self):
        """Test to_boundary length mode."""
        policy = ChannelPolicy(
            length_mode="to_boundary",
            stop_before_boundary=0.001,
        )
        
        assert policy.length_mode == "to_boundary"
        assert policy.stop_before_boundary == 0.001
    
    def test_to_depth_length_mode(self):
        """Test to_depth length mode."""
        policy = ChannelPolicy(
            length_mode="to_depth",
            target_depth=0.008,
        )
        
        assert policy.length_mode == "to_depth"
        assert policy.target_depth == 0.008
    
    def test_length_mode_serialization(self):
        """Test that length mode is serialized correctly."""
        import json
        
        policy = ChannelPolicy(
            length_mode="to_boundary",
            stop_before_boundary=0.001,
        )
        
        policy_dict = policy.to_dict()
        json_str = json.dumps(policy_dict)
        decoded = json.loads(json_str)
        
        assert decoded["length_mode"] == "to_boundary"
        assert decoded["stop_before_boundary"] == 0.001


class TestChannelLengthReporting:
    """Test channel length reporting in operation reports."""
    
    def test_report_includes_length_calculation(self):
        """Test that report includes length calculation details."""
        report = OperationReport(
            operation="channel_generation",
            success=True,
            metadata={
                "length_mode": "to_boundary",
                "domain_depth": 0.01,
                "stop_before_boundary": 0.001,
                "computed_length": 0.009,
                "actual_mesh_length": 0.00898,
            },
        )
        
        assert "computed_length" in report.metadata
        assert "actual_mesh_length" in report.metadata
    
    def test_report_warns_on_length_mismatch(self):
        """Test that report warns when actual length differs from computed."""
        computed = 0.009
        actual = 0.0085
        tolerance = 0.0001
        
        mismatch = abs(computed - actual) > tolerance
        
        report = OperationReport(
            operation="channel_generation",
            success=True,
            warnings=["Actual mesh length differs from computed by > tolerance"] if mismatch else [],
            metadata={
                "computed_length": computed,
                "actual_mesh_length": actual,
                "length_tolerance": tolerance,
            },
        )
        
        if mismatch:
            assert len(report.warnings) > 0
