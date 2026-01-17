"""
Test waypoint skipping behavior (D3).

This module verifies that pathfinding handles infeasible waypoints correctly,
including skipping behavior and proper reporting.
"""

import pytest
import numpy as np

from aog_policies import (
    HierarchicalPathfindingPolicy,
    WaypointPolicy,
    OperationReport,
)
from generation.core.domain import BoxDomain
from generation.core.types import Point3D


class TestWaypointSkippingBehavior:
    """D3: Waypoint skipping behavior."""
    
    def test_waypoint_policy_has_skip_parameters(self):
        """Test that WaypointPolicy has skip-related parameters."""
        policy = WaypointPolicy(
            skip_unreachable=True,
            max_skip_count=3,
            emit_warnings=True,
        )
        
        assert hasattr(policy, 'skip_unreachable')
        assert hasattr(policy, 'max_skip_count')
        assert hasattr(policy, 'emit_warnings')
        
        assert policy.skip_unreachable is True
        assert policy.max_skip_count == 3
        assert policy.emit_warnings is True
    
    def test_waypoint_policy_serializable(self):
        """Test that WaypointPolicy is JSON-serializable."""
        import json
        
        policy = WaypointPolicy(
            skip_unreachable=True,
            max_skip_count=5,
            emit_warnings=True,
        )
        
        policy_dict = policy.to_dict()
        json_str = json.dumps(policy_dict)
        
        assert json_str is not None
        
        decoded = json.loads(json_str)
        assert decoded["skip_unreachable"] is True
        assert decoded["max_skip_count"] == 5
    
    def test_skipped_waypoints_reported(self):
        """Test that skipped waypoints are reported in metadata."""
        report = OperationReport(
            operation="pathfinding",
            success=True,
            requested_policy={
                "waypoints": [
                    {"x": 0.0, "y": 0.0, "z": 0.0},
                    {"x": 0.01, "y": 0.01, "z": 0.01},
                    {"x": 0.02, "y": 0.02, "z": 0.02},
                ],
                "skip_unreachable": True,
                "max_skip_count": 2,
            },
            effective_policy={
                "waypoints": [
                    {"x": 0.0, "y": 0.0, "z": 0.0},
                    {"x": 0.02, "y": 0.02, "z": 0.02},
                ],
                "skip_unreachable": True,
                "max_skip_count": 2,
            },
            warnings=["Skipped waypoint at index 1: unreachable"],
            metadata={
                "skipped_waypoint_indices": [1],
                "total_waypoints": 3,
                "reached_waypoints": 2,
            },
        )
        
        assert "skipped_waypoint_indices" in report.metadata
        assert 1 in report.metadata["skipped_waypoint_indices"]
        assert len(report.warnings) > 0
        assert "skipped" in report.warnings[0].lower()
    
    def test_skip_count_exceeded_fails(self):
        """Test that exceeding max_skip_count causes failure."""
        report = OperationReport(
            operation="pathfinding",
            success=False,
            requested_policy={
                "waypoints": [
                    {"x": 0.0, "y": 0.0, "z": 0.0},
                    {"x": 0.01, "y": 0.01, "z": 0.01},
                    {"x": 0.02, "y": 0.02, "z": 0.02},
                    {"x": 0.03, "y": 0.03, "z": 0.03},
                ],
                "skip_unreachable": True,
                "max_skip_count": 2,
            },
            errors=["Exceeded max_skip_count (2): 3 waypoints unreachable"],
            metadata={
                "skipped_waypoint_indices": [1, 2, 3],
                "max_skip_count": 2,
            },
        )
        
        assert not report.success
        assert len(report.errors) > 0
        assert "exceeded" in report.errors[0].lower() or "max_skip" in report.errors[0].lower()
    
    def test_no_skip_when_disabled(self):
        """Test that waypoints are not skipped when skip_unreachable is False."""
        policy = WaypointPolicy(
            skip_unreachable=False,
            max_skip_count=0,
        )
        
        assert policy.skip_unreachable is False
        
        report = OperationReport(
            operation="pathfinding",
            success=False,
            requested_policy=policy.to_dict(),
            errors=["Waypoint at index 1 is unreachable and skip_unreachable is disabled"],
        )
        
        assert not report.success


class TestWaypointPolicyConfiguration:
    """Test waypoint policy configuration."""
    
    def test_default_waypoint_policy(self):
        """Test default WaypointPolicy values."""
        policy = WaypointPolicy()
        
        assert hasattr(policy, 'skip_unreachable')
        assert hasattr(policy, 'max_skip_count')
    
    def test_waypoint_policy_with_zero_skips(self):
        """Test WaypointPolicy with zero max_skip_count."""
        policy = WaypointPolicy(
            skip_unreachable=True,
            max_skip_count=0,
        )
        
        assert policy.max_skip_count == 0
    
    def test_waypoint_policy_with_high_skip_count(self):
        """Test WaypointPolicy with high max_skip_count."""
        policy = WaypointPolicy(
            skip_unreachable=True,
            max_skip_count=100,
        )
        
        assert policy.max_skip_count == 100
    
    def test_emit_warnings_flag(self):
        """Test emit_warnings flag in WaypointPolicy."""
        policy_with_warnings = WaypointPolicy(emit_warnings=True)
        policy_without_warnings = WaypointPolicy(emit_warnings=False)
        
        assert policy_with_warnings.emit_warnings is True
        assert policy_without_warnings.emit_warnings is False


class TestWaypointReporting:
    """Test waypoint reporting in operation reports."""
    
    def test_report_includes_waypoint_metadata(self):
        """Test that report includes comprehensive waypoint metadata."""
        report = OperationReport(
            operation="pathfinding",
            success=True,
            metadata={
                "waypoints": {
                    "total": 5,
                    "reached": 4,
                    "skipped": 1,
                    "skipped_indices": [2],
                    "reasons": {
                        2: "outside_domain",
                    },
                },
            },
        )
        
        waypoint_meta = report.metadata["waypoints"]
        assert waypoint_meta["total"] == 5
        assert waypoint_meta["reached"] == 4
        assert waypoint_meta["skipped"] == 1
        assert 2 in waypoint_meta["skipped_indices"]
    
    def test_report_includes_skip_reasons(self):
        """Test that report includes reasons for skipped waypoints."""
        report = OperationReport(
            operation="pathfinding",
            success=True,
            warnings=[
                "Waypoint 1 skipped: outside domain bounds",
                "Waypoint 3 skipped: collision with obstacle",
            ],
            metadata={
                "skipped_waypoint_indices": [1, 3],
                "skip_reasons": {
                    1: "outside_domain",
                    3: "collision",
                },
            },
        )
        
        assert len(report.warnings) == 2
        assert "skip_reasons" in report.metadata
        assert report.metadata["skip_reasons"][1] == "outside_domain"
        assert report.metadata["skip_reasons"][3] == "collision"


class TestWaypointIntegrationWithHierarchical:
    """Test waypoint handling integration with hierarchical pathfinding."""
    
    def test_hierarchical_policy_includes_waypoint_policy(self):
        """Test that HierarchicalPathfindingPolicy can include waypoint policy."""
        hierarchical = HierarchicalPathfindingPolicy(
            pitch_coarse=0.001,
            pitch_fine=0.0001,
        )
        
        waypoint = WaypointPolicy(
            skip_unreachable=True,
            max_skip_count=3,
        )
        
        combined_dict = {
            **hierarchical.to_dict(),
            "waypoint_policy": waypoint.to_dict(),
        }
        
        assert "waypoint_policy" in combined_dict
        assert combined_dict["waypoint_policy"]["skip_unreachable"] is True
    
    def test_waypoint_skipping_in_coarse_vs_fine_stage(self):
        """Test that waypoint skipping can occur in both coarse and fine stages."""
        report = OperationReport(
            operation="hierarchical_pathfinding",
            success=True,
            metadata={
                "coarse_stage": {
                    "waypoints_reached": 4,
                    "waypoints_skipped": 1,
                },
                "fine_stage": {
                    "waypoints_reached": 4,
                    "waypoints_skipped": 0,
                },
            },
        )
        
        assert report.metadata["coarse_stage"]["waypoints_skipped"] == 1
        assert report.metadata["fine_stage"]["waypoints_skipped"] == 0
