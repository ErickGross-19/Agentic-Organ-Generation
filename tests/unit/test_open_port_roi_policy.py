"""
Tests for OpenPortPolicy ROI-first reduction and prefer fine pitch features.

These tests cover Task E:
- Prefer fine pitch policy
- ROI-first reduction before pitch relaxation
- Debug disable with warning
"""

import pytest
import math
from unittest.mock import MagicMock

from aog_policies.validity import OpenPortPolicy, ValidationPolicy
from aog_policies.resolution import ResolutionPolicy


class TestOpenPortPolicyConfig:
    """Tests for OpenPortPolicy configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        policy = OpenPortPolicy()
        
        assert policy.enabled is True
        assert policy.prefer_fine_pitch is True
        assert policy.roi_first_reduction is True
        assert policy.min_local_region_size == 0.001
        assert policy.validation_pitch is None
        assert policy.local_region_size == 0.004
        assert policy.max_voxels_roi == 2_000_000
        assert policy.auto_relax_pitch is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        policy = OpenPortPolicy(
            enabled=True,
            prefer_fine_pitch=False,
            roi_first_reduction=False,
            min_local_region_size=0.002,
            validation_pitch=0.00005,
            local_region_size=0.005,
            max_voxels_roi=1_000_000,
        )
        
        assert policy.prefer_fine_pitch is False
        assert policy.roi_first_reduction is False
        assert policy.min_local_region_size == 0.002
        assert policy.validation_pitch == 0.00005
        assert policy.local_region_size == 0.005
        assert policy.max_voxels_roi == 1_000_000
    
    def test_to_dict_includes_new_fields(self):
        """Test that to_dict includes new fields."""
        policy = OpenPortPolicy(
            prefer_fine_pitch=True,
            roi_first_reduction=True,
            min_local_region_size=0.0015,
        )
        
        d = policy.to_dict()
        
        assert "prefer_fine_pitch" in d
        assert "roi_first_reduction" in d
        assert "min_local_region_size" in d
        assert d["prefer_fine_pitch"] is True
        assert d["roi_first_reduction"] is True
        assert d["min_local_region_size"] == 0.0015
    
    def test_from_dict_with_new_fields(self):
        """Test that from_dict handles new fields."""
        d = {
            "enabled": True,
            "prefer_fine_pitch": False,
            "roi_first_reduction": False,
            "min_local_region_size": 0.002,
        }
        
        policy = OpenPortPolicy.from_dict(d)
        
        assert policy.prefer_fine_pitch is False
        assert policy.roi_first_reduction is False
        assert policy.min_local_region_size == 0.002


class TestPreferFinePitchPolicy:
    """Tests for prefer fine pitch policy."""
    
    def test_compute_adaptive_pitch_with_explicit_validation_pitch(self):
        """Test that explicit validation_pitch is used when set."""
        policy = OpenPortPolicy(validation_pitch=0.00005)
        
        pitch = policy.compute_adaptive_pitch(port_radius=0.0005)
        
        assert pitch == 0.00005
    
    def test_compute_adaptive_pitch_from_port_radius(self):
        """Test adaptive pitch computation from port radius."""
        policy = OpenPortPolicy(
            validation_pitch=None,
            adaptive_pitch=True,
            min_voxels_across_radius=8,
        )
        
        port_radius = 0.0004
        pitch = policy.compute_adaptive_pitch(port_radius)
        
        expected_pitch = port_radius / 8
        assert pitch == expected_pitch
    
    def test_compute_pitch_with_resolution_policy(self):
        """Test pitch computation using resolution policy."""
        policy = OpenPortPolicy(
            validation_pitch=None,
            prefer_fine_pitch=True,
        )
        
        resolution_policy = ResolutionPolicy(
            min_channel_diameter=0.00002,
            min_voxels_across_feature=8,
        )
        
        pitch = policy.compute_pitch_with_resolution_policy(
            port_radius=0.0005,
            resolution_policy=resolution_policy,
        )
        
        assert pitch == resolution_policy.target_pitch
    
    def test_compute_pitch_without_resolution_policy(self):
        """Test pitch computation falls back to adaptive when no resolution policy."""
        policy = OpenPortPolicy(
            validation_pitch=None,
            prefer_fine_pitch=True,
            adaptive_pitch=True,
            min_voxels_across_radius=8,
        )
        
        port_radius = 0.0004
        pitch = policy.compute_pitch_with_resolution_policy(
            port_radius=port_radius,
            resolution_policy=None,
        )
        
        expected_pitch = port_radius / 8
        assert pitch == expected_pitch
    
    def test_compute_pitch_prefer_fine_pitch_disabled(self):
        """Test pitch computation when prefer_fine_pitch is disabled."""
        policy = OpenPortPolicy(
            validation_pitch=None,
            prefer_fine_pitch=False,
            adaptive_pitch=True,
            min_voxels_across_radius=8,
        )
        
        resolution_policy = ResolutionPolicy(
            min_channel_diameter=0.00002,
            min_voxels_across_feature=8,
        )
        
        port_radius = 0.0004
        pitch = policy.compute_pitch_with_resolution_policy(
            port_radius=port_radius,
            resolution_policy=resolution_policy,
        )
        
        expected_pitch = port_radius / 8
        assert pitch == expected_pitch


class TestROIFirstReduction:
    """Tests for ROI-first reduction policy."""
    
    def test_roi_within_budget_no_changes(self):
        """Test that ROI and pitch are unchanged when within budget."""
        policy = OpenPortPolicy(
            local_region_size=0.002,
            max_voxels_roi=10_000_000,
            roi_first_reduction=True,
            auto_relax_pitch=True,
        )
        
        pitch = 0.0001
        
        roi_size, eff_pitch, roi_reduced, pitch_relaxed = policy.compute_roi_with_budget(pitch)
        
        assert roi_size == 0.002
        assert eff_pitch == pitch
        assert roi_reduced is False
        assert pitch_relaxed is False
    
    def test_roi_reduced_before_pitch_relaxed(self):
        """Test that ROI is reduced before pitch is relaxed."""
        policy = OpenPortPolicy(
            local_region_size=0.004,
            min_local_region_size=0.001,
            max_voxels_roi=100_000,
            roi_first_reduction=True,
            auto_relax_pitch=True,
        )
        
        pitch = 0.00005
        
        roi_size, eff_pitch, roi_reduced, pitch_relaxed = policy.compute_roi_with_budget(pitch)
        
        assert roi_size < 0.004
        assert roi_size >= 0.001
        assert roi_reduced is True
    
    def test_pitch_relaxed_when_roi_at_minimum(self):
        """Test that pitch is relaxed when ROI is at minimum."""
        policy = OpenPortPolicy(
            local_region_size=0.004,
            min_local_region_size=0.001,
            max_voxels_roi=1000,
            roi_first_reduction=True,
            auto_relax_pitch=True,
        )
        
        pitch = 0.00001
        
        roi_size, eff_pitch, roi_reduced, pitch_relaxed = policy.compute_roi_with_budget(pitch)
        
        assert eff_pitch > pitch
        assert pitch_relaxed is True
    
    def test_roi_first_reduction_disabled(self):
        """Test behavior when roi_first_reduction is disabled."""
        policy = OpenPortPolicy(
            local_region_size=0.004,
            min_local_region_size=0.001,
            max_voxels_roi=100_000,
            roi_first_reduction=False,
            auto_relax_pitch=True,
        )
        
        pitch = 0.00005
        
        roi_size, eff_pitch, roi_reduced, pitch_relaxed = policy.compute_roi_with_budget(pitch)
        
        assert roi_size == 0.004
        assert roi_reduced is False
        if eff_pitch > pitch:
            assert pitch_relaxed is True


class TestValidationPolicyDebugDisable:
    """Tests for debug disable with warning."""
    
    def test_check_open_ports_disabled(self):
        """Test that check_open_ports can be disabled."""
        policy = ValidationPolicy(check_open_ports=False)
        
        assert policy.check_open_ports is False
    
    def test_open_port_policy_disabled(self):
        """Test that OpenPortPolicy can be disabled."""
        policy = OpenPortPolicy(enabled=False)
        
        assert policy.enabled is False
    
    def test_validation_policy_to_dict(self):
        """Test ValidationPolicy serialization."""
        policy = ValidationPolicy(
            check_open_ports=False,
            check_void_inside_domain=True,
            allow_boundary_intersections_at_ports=True,
        )
        
        d = policy.to_dict()
        
        assert d["check_open_ports"] is False
        assert d["check_void_inside_domain"] is True
        assert d["allow_boundary_intersections_at_ports"] is True


class TestResolutionPolicyIntegration:
    """Tests for ResolutionPolicy integration with OpenPortPolicy."""
    
    def test_resolution_policy_target_pitch(self):
        """Test that ResolutionPolicy provides target_pitch."""
        resolution_policy = ResolutionPolicy(
            min_channel_diameter=0.00002,
            min_voxels_across_feature=8,
        )
        
        expected_pitch = 0.00002 / 8
        assert resolution_policy.target_pitch == expected_pitch
    
    def test_resolution_policy_pathfinding_fine_pitch(self):
        """Test that ResolutionPolicy provides pathfinding_pitch_fine."""
        resolution_policy = ResolutionPolicy(
            min_channel_diameter=0.00002,
            min_voxels_across_feature=8,
            pathfinding_fine_factor=1.0,
        )
        
        expected_pitch = 0.00002 / 8
        assert resolution_policy.pathfinding_pitch_fine == expected_pitch
