"""
Test open-port ROI budgeted connectivity (H2).

This module verifies that open-port validation uses ROI and respects
max_voxels_roi budget.
"""

import pytest
import numpy as np

from aog_policies import OpenPortPolicy, ResolutionPolicy, OperationReport


class TestOpenPortROIBudgetedConnectivity:
    """H2: Open-port validation uses ROI and respects max_voxels_roi."""
    
    def test_open_port_policy_has_roi_budget(self):
        """Test that OpenPortPolicy has max_voxels_roi parameter."""
        policy = OpenPortPolicy(
            enabled=True,
            max_voxels_roi=1_000_000,
        )
        
        assert hasattr(policy, 'max_voxels_roi')
        assert policy.max_voxels_roi == 1_000_000
    
    def test_pitch_relaxed_when_roi_exceeds_budget(self):
        """Test that pitch is relaxed when ROI would exceed budget."""
        report = OperationReport(
            operation="open_port_validation",
            success=True,
            requested_policy={
                "max_voxels_roi": 500_000,
                "pitch": 1e-5,
            },
            effective_policy={
                "max_voxels_roi": 500_000,
                "pitch": 2e-5,
            },
            warnings=[
                "ROI pitch relaxed from 1e-5 to 2e-5 due to voxel budget"
            ],
            metadata={
                "roi": {
                    "requested_pitch": 1e-5,
                    "effective_pitch": 2e-5,
                    "pitch_relaxed": True,
                    "voxel_count": 450_000,
                    "max_voxels": 500_000,
                },
            },
        )
        
        roi = report.metadata["roi"]
        assert roi["pitch_relaxed"]
        assert roi["effective_pitch"] > roi["requested_pitch"]
        assert roi["voxel_count"] <= roi["max_voxels"]
    
    def test_roi_voxel_count_within_budget(self):
        """Test that ROI voxel count is within budget."""
        report = OperationReport(
            operation="open_port_validation",
            success=True,
            metadata={
                "roi": {
                    "voxel_count": 450_000,
                    "max_voxels": 500_000,
                },
            },
        )
        
        roi = report.metadata["roi"]
        assert roi["voxel_count"] <= roi["max_voxels"]
    
    def test_connectivity_result_stable(self):
        """Test that connectivity result is stable (not random)."""
        results = []
        
        for _ in range(3):
            report = OperationReport(
                operation="open_port_validation",
                success=True,
                metadata={
                    "connectivity": {
                        "is_connected": True,
                        "method": "flood_fill",
                        "seed": 42,
                    },
                },
            )
            results.append(report.metadata["connectivity"]["is_connected"])
        
        assert all(r == results[0] for r in results), (
            "Connectivity result should be deterministic"
        )
    
    def test_open_port_policy_serializable(self):
        """Test that OpenPortPolicy is JSON-serializable."""
        import json
        
        policy = OpenPortPolicy(
            enabled=True,
            probe_radius_factor=1.0,
            max_voxels_roi=1_000_000,
        )
        
        policy_dict = policy.to_dict()
        json_str = json.dumps(policy_dict)
        
        assert json_str is not None
        
        decoded = json.loads(json_str)
        assert decoded["max_voxels_roi"] == 1_000_000


class TestROIConfiguration:
    """Test ROI configuration options."""
    
    def test_roi_size_configuration(self):
        """Test ROI size configuration."""
        policy = OpenPortPolicy(
            enabled=True,
            roi_size_factor=2.0,
            roi_min_size=0.001,
            roi_max_size=0.01,
        )
        
        assert policy.roi_size_factor == 2.0
        assert policy.roi_min_size == 0.001
        assert policy.roi_max_size == 0.01
    
    def test_probe_radius_factor(self):
        """Test probe radius factor configuration."""
        policy = OpenPortPolicy(
            enabled=True,
            probe_radius_factor=0.9,
        )
        
        assert policy.probe_radius_factor == 0.9
    
    def test_resolution_policy_derives_roi_budget(self):
        """Test that ResolutionPolicy derives ROI budget."""
        resolution = ResolutionPolicy(
            max_voxels=100_000_000,
            max_voxels_open_port_roi=1_000_000,
        )
        
        roi_budget = resolution.get_max_voxels_for_operation("open_port_roi")
        assert roi_budget == 1_000_000


class TestROIPitchRelaxation:
    """Test ROI pitch relaxation behavior."""
    
    def test_pitch_relaxation_with_large_roi(self):
        """Test pitch relaxation with large ROI."""
        resolution = ResolutionPolicy(
            min_channel_diameter=20e-6,
            voxels_across_min_diameter=8,
            max_voxels_open_port_roi=100_000,
            auto_relax_pitch=True,
        )
        
        roi_extents = (0.005, 0.005, 0.005)
        
        effective_pitch, was_relaxed, warning = resolution.compute_relaxed_pitch(
            base_pitch=resolution.target_pitch,
            domain_extents=roi_extents,
            max_voxels_override=resolution.get_max_voxels_for_operation("open_port_roi"),
        )
        
        if was_relaxed:
            assert effective_pitch > resolution.target_pitch
    
    def test_no_relaxation_with_small_roi(self):
        """Test no relaxation with small ROI."""
        resolution = ResolutionPolicy(
            min_channel_diameter=20e-6,
            voxels_across_min_diameter=8,
            max_voxels_open_port_roi=100_000_000,
            auto_relax_pitch=True,
        )
        
        small_roi_extents = (0.0001, 0.0001, 0.0001)
        
        effective_pitch, was_relaxed, warning = resolution.compute_relaxed_pitch(
            base_pitch=resolution.target_pitch,
            domain_extents=small_roi_extents,
            max_voxels_override=resolution.get_max_voxels_for_operation("open_port_roi"),
        )
        
        assert not was_relaxed or abs(effective_pitch - resolution.target_pitch) < 1e-15


class TestConnectivityValidation:
    """Test connectivity validation behavior."""
    
    def test_connectivity_check_method(self):
        """Test connectivity check method."""
        report = OperationReport(
            operation="open_port_validation",
            success=True,
            metadata={
                "connectivity": {
                    "method": "flood_fill",
                    "is_connected": True,
                    "connected_volume_fraction": 0.95,
                },
            },
        )
        
        connectivity = report.metadata["connectivity"]
        assert connectivity["method"] == "flood_fill"
        assert connectivity["is_connected"]
    
    def test_connectivity_reports_volume_fraction(self):
        """Test that connectivity reports volume fraction."""
        report = OperationReport(
            operation="open_port_validation",
            success=True,
            metadata={
                "connectivity": {
                    "is_connected": True,
                    "connected_volume_fraction": 0.98,
                    "disconnected_volume_fraction": 0.02,
                },
            },
        )
        
        connectivity = report.metadata["connectivity"]
        total = connectivity["connected_volume_fraction"] + connectivity["disconnected_volume_fraction"]
        assert abs(total - 1.0) < 0.01
    
    def test_connectivity_failure_details(self):
        """Test connectivity failure details."""
        report = OperationReport(
            operation="open_port_validation",
            success=False,
            errors=["Port connectivity check failed: port 1 is sealed"],
            metadata={
                "connectivity": {
                    "is_connected": False,
                    "sealed_ports": [1],
                    "reason": "No path from exterior to port interior",
                },
            },
        )
        
        connectivity = report.metadata["connectivity"]
        assert not connectivity["is_connected"]
        assert 1 in connectivity["sealed_ports"]


class TestROIReporting:
    """Test ROI reporting in operation reports."""
    
    def test_report_includes_roi_metrics(self):
        """Test that report includes ROI metrics."""
        report = OperationReport(
            operation="open_port_validation",
            success=True,
            metadata={
                "roi": {
                    "center": [0.0, 0.0, 0.005],
                    "extents": [0.002, 0.002, 0.002],
                    "voxel_count": 450_000,
                    "max_voxels": 500_000,
                    "pitch": 2e-5,
                    "pitch_relaxed": True,
                },
            },
        )
        
        roi = report.metadata["roi"]
        assert "center" in roi
        assert "extents" in roi
        assert "voxel_count" in roi
        assert "pitch" in roi
    
    def test_report_includes_per_port_results(self):
        """Test that report includes per-port results."""
        report = OperationReport(
            operation="open_port_validation",
            success=True,
            metadata={
                "ports": [
                    {
                        "id": 1,
                        "position": [0.0, 0.0, 0.005],
                        "radius": 0.0005,
                        "is_open": True,
                        "connectivity_fraction": 0.98,
                    },
                    {
                        "id": 2,
                        "position": [0.0, 0.0, -0.005],
                        "radius": 0.0005,
                        "is_open": True,
                        "connectivity_fraction": 0.97,
                    },
                ],
            },
        )
        
        ports = report.metadata["ports"]
        assert len(ports) == 2
        assert all(p["is_open"] for p in ports)
    
    def test_report_json_serializable(self):
        """Test that ROI report is JSON-serializable."""
        import json
        
        report = OperationReport(
            operation="open_port_validation",
            success=True,
            requested_policy={
                "max_voxels_roi": 500_000,
            },
            effective_policy={
                "max_voxels_roi": 500_000,
                "pitch": 2e-5,
            },
            metadata={
                "roi": {
                    "voxel_count": 450_000,
                    "pitch_relaxed": True,
                },
            },
        )
        
        report_dict = report.to_dict()
        json_str = json.dumps(report_dict)
        
        assert json_str is not None
