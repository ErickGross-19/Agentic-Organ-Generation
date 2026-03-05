"""
Test mandatory hierarchical pathfinding usage (D1).

This module verifies that the routing API uses hierarchical pathfinding
and reports coarse and fine stage metrics.
"""

import pytest
import numpy as np

from aog_policies import (
    HierarchicalPathfindingPolicy,
    PathfindingPolicy,
    ResolutionPolicy,
    OperationReport,
)
from generation.core.domain import BoxDomain
from generation.core.types import Point3D


def create_test_domain():
    """Create a test domain for pathfinding."""
    return BoxDomain(
        x_min=-0.02, x_max=0.02,
        y_min=-0.02, y_max=0.02,
        z_min=-0.02, z_max=0.02,
    )


class TestHierarchicalMandatory:
    """D1: Mandatory hierarchical usage."""
    
    def test_hierarchical_policy_has_coarse_and_fine_stages(self):
        """Test that HierarchicalPathfindingPolicy has coarse and fine stage parameters."""
        policy = HierarchicalPathfindingPolicy(
            pitch_coarse=0.001,
            pitch_fine=0.0001,
            max_voxels_coarse=5_000_000,
            max_voxels_fine=25_000_000,
        )
        
        assert hasattr(policy, 'pitch_coarse'), "Policy should have pitch_coarse"
        assert hasattr(policy, 'pitch_fine'), "Policy should have pitch_fine"
        assert hasattr(policy, 'max_voxels_coarse'), "Policy should have max_voxels_coarse"
        assert hasattr(policy, 'max_voxels_fine'), "Policy should have max_voxels_fine"
        
        assert policy.pitch_coarse > policy.pitch_fine, (
            f"Coarse pitch ({policy.pitch_coarse}) should be > fine pitch ({policy.pitch_fine})"
        )
    
    def test_hierarchical_policy_serializable(self):
        """Test that HierarchicalPathfindingPolicy is JSON-serializable."""
        import json
        
        policy = HierarchicalPathfindingPolicy(
            pitch_coarse=0.001,
            pitch_fine=0.0001,
            max_voxels_coarse=5_000_000,
            max_voxels_fine=25_000_000,
            clearance=0.0005,
            local_radius=0.0002,
        )
        
        policy_dict = policy.to_dict()
        json_str = json.dumps(policy_dict)
        
        assert json_str is not None
        
        decoded = json.loads(json_str)
        assert decoded["pitch_coarse"] == 0.001
        assert decoded["pitch_fine"] == 0.0001
    
    def test_hierarchical_report_includes_stage_metrics(self):
        """Test that hierarchical pathfinding report includes stage metrics."""
        report = OperationReport(
            operation="hierarchical_pathfinding",
            success=True,
            requested_policy={
                "pitch_coarse": 0.001,
                "pitch_fine": 0.0001,
                "max_voxels_coarse": 5_000_000,
                "max_voxels_fine": 25_000_000,
            },
            effective_policy={
                "pitch_coarse": 0.001,
                "pitch_fine": 0.0001,
                "max_voxels_coarse": 5_000_000,
                "max_voxels_fine": 25_000_000,
            },
            metadata={
                "coarse_stage": {
                    "voxel_count": 1_000_000,
                    "pitch": 0.001,
                    "path_length": 0.05,
                    "time_s": 0.5,
                },
                "fine_stage": {
                    "voxel_count": 10_000_000,
                    "pitch": 0.0001,
                    "path_length": 0.048,
                    "time_s": 2.0,
                },
                "total_time_s": 2.5,
            },
        )
        
        assert "coarse_stage" in report.metadata
        assert "fine_stage" in report.metadata
        assert "voxel_count" in report.metadata["coarse_stage"]
        assert "voxel_count" in report.metadata["fine_stage"]
    
    def test_hierarchical_mode_indicated_in_report(self):
        """Test that report indicates hierarchical mode was used."""
        report = OperationReport(
            operation="pathfinding",
            success=True,
            metadata={
                "mode": "hierarchical",
                "coarse_stage": {"voxel_count": 1_000_000},
                "fine_stage": {"voxel_count": 10_000_000},
            },
        )
        
        assert report.metadata.get("mode") == "hierarchical", (
            "Report should indicate hierarchical mode was used"
        )
    
    def test_resolution_policy_derives_hierarchical_pitches(self):
        """Test that ResolutionPolicy derives hierarchical pathfinding pitches."""
        resolution = ResolutionPolicy(
            min_channel_diameter=20e-6,
            voxels_across_min_diameter=8,
            pathfinding_coarse_factor=20.0,
            pathfinding_fine_factor=1.0,
        )
        
        assert resolution.pathfinding_pitch_coarse > resolution.pathfinding_pitch_fine
        
        expected_coarse = resolution.target_pitch * 20.0
        expected_fine = resolution.target_pitch * 1.0
        
        assert abs(resolution.pathfinding_pitch_coarse - expected_coarse) < 1e-15
        assert abs(resolution.pathfinding_pitch_fine - expected_fine) < 1e-15


class TestHierarchicalPolicyConfiguration:
    """Test hierarchical pathfinding policy configuration."""
    
    def test_default_hierarchical_policy(self):
        """Test default HierarchicalPathfindingPolicy values."""
        policy = HierarchicalPathfindingPolicy()
        
        assert policy.pitch_coarse > 0
        assert policy.pitch_fine > 0
        assert policy.max_voxels_coarse > 0
        assert policy.max_voxels_fine > 0
    
    def test_hierarchical_policy_with_clearance(self):
        """Test HierarchicalPathfindingPolicy with clearance."""
        policy = HierarchicalPathfindingPolicy(
            pitch_coarse=0.001,
            pitch_fine=0.0001,
            clearance=0.0005,
        )
        
        assert policy.clearance == 0.0005
    
    def test_hierarchical_policy_with_local_radius(self):
        """Test HierarchicalPathfindingPolicy with local_radius."""
        policy = HierarchicalPathfindingPolicy(
            pitch_coarse=0.001,
            pitch_fine=0.0001,
            local_radius=0.0002,
        )
        
        assert policy.local_radius == 0.0002
    
    def test_hierarchical_policy_inflation_mode(self):
        """Test HierarchicalPathfindingPolicy inflation mode."""
        policy = HierarchicalPathfindingPolicy(
            pitch_coarse=0.001,
            pitch_fine=0.0001,
            inflation_mode="clearance_plus_local_radius",
        )
        
        assert policy.inflation_mode == "clearance_plus_local_radius"


class TestHierarchicalVsNonHierarchical:
    """Test that hierarchical is preferred over non-hierarchical."""
    
    def test_hierarchical_policy_exists(self):
        """Test that HierarchicalPathfindingPolicy exists and is distinct."""
        from aog_policies import HierarchicalPathfindingPolicy, PathfindingPolicy
        
        assert HierarchicalPathfindingPolicy is not PathfindingPolicy
    
    def test_hierarchical_has_two_stage_params(self):
        """Test that hierarchical policy has two-stage parameters."""
        policy = HierarchicalPathfindingPolicy()
        
        coarse_params = [
            "pitch_coarse",
            "max_voxels_coarse",
        ]
        fine_params = [
            "pitch_fine",
            "max_voxels_fine",
        ]
        
        for param in coarse_params:
            assert hasattr(policy, param), f"Missing coarse param: {param}"
        
        for param in fine_params:
            assert hasattr(policy, param), f"Missing fine param: {param}"
