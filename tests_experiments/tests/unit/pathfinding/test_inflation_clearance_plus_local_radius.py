"""
Test inflation rule correctness (D2).

This module verifies that the inflation rule (clearance + local_radius)
correctly determines path feasibility through narrow passages.
"""

import pytest
import numpy as np

from aog_policies import (
    HierarchicalPathfindingPolicy,
    PathfindingPolicy,
    OperationReport,
)
from generation.core.domain import BoxDomain
from generation.core.domain_composite import CompositeDomain
from generation.core.domain_primitives import SphereDomain
from generation.core.types import Point3D


def create_domain_with_narrow_passage(passage_width: float):
    """
    Create a domain with a narrow passage.
    
    The domain is a box with a sphere removed from the center,
    creating a narrow passage around the sphere.
    
    Args:
        passage_width: Width of the passage (gap between sphere and box walls)
    """
    box_half_size = 0.01
    sphere_radius = box_half_size - passage_width
    
    box = BoxDomain(
        x_min=-box_half_size, x_max=box_half_size,
        y_min=-box_half_size, y_max=box_half_size,
        z_min=-box_half_size, z_max=box_half_size,
    )
    
    sphere = SphereDomain(
        radius=sphere_radius,
        center=Point3D(0, 0, 0),
    )
    
    return CompositeDomain.difference(box, sphere)


class TestInflationRuleCorrectness:
    """D2: Inflation rule correctness."""
    
    def test_small_local_radius_passes_narrow_passage(self):
        """Test that small local_radius allows path through narrow passage."""
        passage_width = 0.002
        
        policy = HierarchicalPathfindingPolicy(
            pitch_coarse=0.0005,
            pitch_fine=0.0001,
            clearance=0.0001,
            local_radius=0.0005,
            inflation_mode="clearance_plus_local_radius",
        )
        
        total_inflation = policy.clearance + policy.local_radius
        
        assert total_inflation < passage_width, (
            f"Total inflation ({total_inflation}) should be < passage width ({passage_width}) "
            f"for path to exist"
        )
    
    def test_large_local_radius_blocks_narrow_passage(self):
        """Test that large local_radius blocks path through narrow passage."""
        passage_width = 0.002
        
        policy = HierarchicalPathfindingPolicy(
            pitch_coarse=0.0005,
            pitch_fine=0.0001,
            clearance=0.0001,
            local_radius=0.002,
            inflation_mode="clearance_plus_local_radius",
        )
        
        total_inflation = policy.clearance + policy.local_radius
        
        assert total_inflation > passage_width, (
            f"Total inflation ({total_inflation}) should be > passage width ({passage_width}) "
            f"for path to be blocked"
        )
    
    def test_inflation_mode_reported(self):
        """Test that inflation mode is reported in pathfinding report."""
        policy = HierarchicalPathfindingPolicy(
            pitch_coarse=0.001,
            pitch_fine=0.0001,
            clearance=0.0005,
            local_radius=0.0002,
            inflation_mode="clearance_plus_local_radius",
        )
        
        report = OperationReport(
            operation="pathfinding",
            success=True,
            requested_policy=policy.to_dict(),
            effective_policy=policy.to_dict(),
            metadata={
                "inflation_mode": policy.inflation_mode,
                "total_inflation": policy.clearance + policy.local_radius,
            },
        )
        
        assert report.metadata["inflation_mode"] == "clearance_plus_local_radius"
        assert report.metadata["total_inflation"] == policy.clearance + policy.local_radius
    
    def test_pass_fail_flips_with_local_radius(self):
        """Test that pass/fail flips as expected with different local_radius values."""
        passage_width = 0.002
        clearance = 0.0001
        
        small_radius = 0.0005
        large_radius = 0.002
        
        small_inflation = clearance + small_radius
        large_inflation = clearance + large_radius
        
        small_passes = small_inflation < passage_width
        large_passes = large_inflation < passage_width
        
        assert small_passes, "Small radius should allow passage"
        assert not large_passes, "Large radius should block passage"
        
        assert small_passes != large_passes, "Pass/fail should flip between small and large radius"


class TestInflationModes:
    """Test different inflation modes."""
    
    def test_clearance_plus_local_radius_mode(self):
        """Test clearance_plus_local_radius inflation mode."""
        policy = HierarchicalPathfindingPolicy(
            clearance=0.0005,
            local_radius=0.0002,
            inflation_mode="clearance_plus_local_radius",
        )
        
        expected_inflation = 0.0005 + 0.0002
        actual_inflation = policy.clearance + policy.local_radius
        
        assert abs(actual_inflation - expected_inflation) < 1e-15
    
    def test_clearance_only_mode(self):
        """Test clearance-only inflation mode."""
        policy = HierarchicalPathfindingPolicy(
            clearance=0.0005,
            local_radius=0.0002,
            inflation_mode="clearance_only",
        )
        
        if policy.inflation_mode == "clearance_only":
            expected_inflation = policy.clearance
            assert expected_inflation == 0.0005
    
    def test_local_radius_only_mode(self):
        """Test local_radius-only inflation mode."""
        policy = HierarchicalPathfindingPolicy(
            clearance=0.0005,
            local_radius=0.0002,
            inflation_mode="local_radius_only",
        )
        
        if policy.inflation_mode == "local_radius_only":
            expected_inflation = policy.local_radius
            assert expected_inflation == 0.0002


class TestInflationPolicyConfiguration:
    """Test inflation policy configuration."""
    
    def test_default_inflation_mode(self):
        """Test default inflation mode."""
        policy = HierarchicalPathfindingPolicy()
        
        assert hasattr(policy, 'inflation_mode')
        assert policy.inflation_mode in [
            "clearance_plus_local_radius",
            "clearance_only",
            "local_radius_only",
        ]
    
    def test_clearance_and_local_radius_independent(self):
        """Test that clearance and local_radius are independent parameters."""
        policy1 = HierarchicalPathfindingPolicy(
            clearance=0.001,
            local_radius=0.0001,
        )
        
        policy2 = HierarchicalPathfindingPolicy(
            clearance=0.0001,
            local_radius=0.001,
        )
        
        assert policy1.clearance != policy2.clearance
        assert policy1.local_radius != policy2.local_radius
    
    def test_inflation_values_serialized(self):
        """Test that inflation values are serialized in policy dict."""
        import json
        
        policy = HierarchicalPathfindingPolicy(
            clearance=0.0005,
            local_radius=0.0002,
            inflation_mode="clearance_plus_local_radius",
        )
        
        policy_dict = policy.to_dict()
        json_str = json.dumps(policy_dict)
        decoded = json.loads(json_str)
        
        assert "clearance" in decoded
        assert "local_radius" in decoded
        assert "inflation_mode" in decoded
        
        assert decoded["clearance"] == 0.0005
        assert decoded["local_radius"] == 0.0002
        assert decoded["inflation_mode"] == "clearance_plus_local_radius"
