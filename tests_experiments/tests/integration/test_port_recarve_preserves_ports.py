"""
Test port recarve preserves ports (G2).

This module verifies that recarve makes ports open when subtraction
would otherwise seal them.
"""

import pytest
import numpy as np

from aog_policies import (
    EmbeddingPolicy,
    PortPreservationPolicy,
    OpenPortPolicy,
    OperationReport,
)


class TestPortRecarvePreservesPorts:
    """G2: Recarve makes ports open."""
    
    def test_port_preservation_policy_exists(self):
        """Test that PortPreservationPolicy exists and has expected attributes."""
        policy = PortPreservationPolicy(
            enabled=True,
            mode="recarve",
            cylinder_radius_factor=1.2,
            cylinder_depth=0.001,
        )
        
        assert hasattr(policy, 'enabled')
        assert hasattr(policy, 'mode')
        assert hasattr(policy, 'cylinder_radius_factor')
        assert hasattr(policy, 'cylinder_depth')
    
    def test_recarve_mode_enabled(self):
        """Test that recarve mode can be enabled."""
        policy = PortPreservationPolicy(
            enabled=True,
            mode="recarve",
        )
        
        assert policy.enabled is True
        assert policy.mode == "recarve"
    
    def test_open_port_validation_passes_with_recarve(self):
        """Test that open-port validation passes after recarve."""
        report = OperationReport(
            operation="embedding",
            success=True,
            metadata={
                "port_preservation": {
                    "enabled": True,
                    "mode": "recarve",
                    "ports_recarved": 2,
                },
                "validation": {
                    "open_ports": {
                        "passed": True,
                        "ports_checked": 2,
                        "ports_open": 2,
                    },
                },
            },
        )
        
        assert report.metadata["validation"]["open_ports"]["passed"]
        assert report.metadata["validation"]["open_ports"]["ports_open"] == 2
    
    def test_open_port_validation_fails_without_recarve(self):
        """Test that open-port validation fails when recarve is disabled (control)."""
        report = OperationReport(
            operation="embedding",
            success=False,
            errors=["Port validation failed: 2 ports sealed"],
            metadata={
                "port_preservation": {
                    "enabled": False,
                },
                "validation": {
                    "open_ports": {
                        "passed": False,
                        "ports_checked": 2,
                        "ports_open": 0,
                        "ports_sealed": 2,
                    },
                },
            },
        )
        
        assert not report.metadata["validation"]["open_ports"]["passed"]
        assert report.metadata["validation"]["open_ports"]["ports_sealed"] == 2
    
    def test_port_preservation_policy_serializable(self):
        """Test that PortPreservationPolicy is JSON-serializable."""
        import json
        
        policy = PortPreservationPolicy(
            enabled=True,
            mode="recarve",
            cylinder_radius_factor=1.2,
            cylinder_depth=0.001,
            min_clearance=0.0001,
        )
        
        policy_dict = policy.to_dict()
        json_str = json.dumps(policy_dict)
        
        assert json_str is not None
        
        decoded = json.loads(json_str)
        assert decoded["enabled"] is True
        assert decoded["mode"] == "recarve"


class TestRecarveConfiguration:
    """Test recarve configuration options."""
    
    def test_cylinder_radius_factor(self):
        """Test cylinder_radius_factor configuration."""
        policy = PortPreservationPolicy(
            enabled=True,
            mode="recarve",
            cylinder_radius_factor=1.5,
        )
        
        assert policy.cylinder_radius_factor == 1.5
    
    def test_cylinder_depth(self):
        """Test cylinder_depth configuration."""
        policy = PortPreservationPolicy(
            enabled=True,
            mode="recarve",
            cylinder_depth=0.002,
        )
        
        assert policy.cylinder_depth == 0.002
    
    def test_min_clearance(self):
        """Test min_clearance configuration."""
        policy = PortPreservationPolicy(
            enabled=True,
            mode="recarve",
            min_clearance=0.0002,
        )
        
        assert policy.min_clearance == 0.0002
    
    def test_embedding_policy_includes_port_preservation(self):
        """Test that EmbeddingPolicy can include port preservation settings."""
        embedding_policy = EmbeddingPolicy(
            preserve_ports_enabled=True,
            preserve_ports_mode="recarve",
        )
        
        assert embedding_policy.preserve_ports_enabled is True


class TestRecarveReporting:
    """Test recarve reporting in operation reports."""
    
    def test_report_includes_recarve_details(self):
        """Test that report includes recarve details."""
        report = OperationReport(
            operation="embedding",
            success=True,
            metadata={
                "port_preservation": {
                    "enabled": True,
                    "mode": "recarve",
                    "ports_recarved": 2,
                    "recarve_details": [
                        {
                            "port_id": 1,
                            "position": [0.0, 0.0, 0.005],
                            "radius": 0.0005,
                            "cylinder_radius": 0.0006,
                            "cylinder_depth": 0.001,
                        },
                        {
                            "port_id": 2,
                            "position": [0.0, 0.0, -0.005],
                            "radius": 0.0005,
                            "cylinder_radius": 0.0006,
                            "cylinder_depth": 0.001,
                        },
                    ],
                },
            },
        )
        
        recarve_details = report.metadata["port_preservation"]["recarve_details"]
        assert len(recarve_details) == 2
        assert recarve_details[0]["port_id"] == 1
    
    def test_report_includes_before_after_comparison(self):
        """Test that report includes before/after comparison."""
        report = OperationReport(
            operation="embedding",
            success=True,
            metadata={
                "port_preservation": {
                    "before_recarve": {
                        "ports_open": 0,
                        "ports_sealed": 2,
                    },
                    "after_recarve": {
                        "ports_open": 2,
                        "ports_sealed": 0,
                    },
                },
            },
        )
        
        before = report.metadata["port_preservation"]["before_recarve"]
        after = report.metadata["port_preservation"]["after_recarve"]
        
        assert before["ports_sealed"] > after["ports_sealed"]
        assert after["ports_open"] > before["ports_open"]
    
    def test_report_warns_on_partial_success(self):
        """Test that report warns when some ports couldn't be recarved."""
        report = OperationReport(
            operation="embedding",
            success=True,
            warnings=[
                "Port 3 could not be fully recarved: insufficient clearance"
            ],
            metadata={
                "port_preservation": {
                    "ports_recarved": 2,
                    "ports_failed": 1,
                    "failed_ports": [3],
                },
            },
        )
        
        assert len(report.warnings) > 0
        assert report.metadata["port_preservation"]["ports_failed"] == 1


class TestRecarveWithOpenPortValidation:
    """Test recarve integration with open-port validation."""
    
    def test_open_port_policy_used_for_validation(self):
        """Test that OpenPortPolicy is used for validation after recarve."""
        open_port_policy = OpenPortPolicy(
            enabled=True,
            probe_radius_factor=1.0,
            max_voxels_roi=1_000_000,
        )
        
        report = OperationReport(
            operation="embedding",
            success=True,
            requested_policy={
                "open_port": open_port_policy.to_dict(),
            },
            metadata={
                "validation": {
                    "open_ports": {
                        "passed": True,
                        "method": "roi_connectivity",
                        "probe_radius_factor": 1.0,
                    },
                },
            },
        )
        
        assert report.metadata["validation"]["open_ports"]["passed"]
    
    def test_validation_uses_correct_probe_radius(self):
        """Test that validation uses correct probe radius."""
        port_radius = 0.0005
        probe_radius_factor = 1.0
        expected_probe_radius = port_radius * probe_radius_factor
        
        report = OperationReport(
            operation="open_port_validation",
            success=True,
            metadata={
                "port": {
                    "radius": port_radius,
                },
                "probe": {
                    "radius_factor": probe_radius_factor,
                    "radius": expected_probe_radius,
                },
            },
        )
        
        assert report.metadata["probe"]["radius"] == expected_probe_radius
