"""
Test pitch resolution budgeting (C1, C2, C3).

This module verifies that:
- C1: Enforces ">= 8 voxels across min diameter" rule
- C2: Relaxes pitch under max_voxels pressure
- C3: Deterministic relaxation
"""

import pytest
import numpy as np

from aog_policies import ResolutionPolicy, PitchLimits


class TestMinVoxelsAcrossRule:
    """C1: Enforces ">= 8 voxels across min diameter" rule."""
    
    def test_target_pitch_from_min_diameter_and_voxels(self):
        """Test that target pitch is derived from min_diameter / voxels_across."""
        policy = ResolutionPolicy(
            min_channel_diameter=20e-6,
            voxels_across_min_diameter=8,
        )
        
        expected_pitch = 20e-6 / 8
        
        assert abs(policy.target_pitch - expected_pitch) < 1e-12, (
            f"target_pitch should be {expected_pitch}, got {policy.target_pitch}"
        )
    
    def test_min_diameter_20um_8_voxels_gives_2_5um_pitch(self):
        """Test specific case: 20um diameter, 8 voxels -> 2.5um pitch."""
        policy = ResolutionPolicy(
            min_channel_diameter=20e-6,
            voxels_across_min_diameter=8,
        )
        
        expected_pitch = 2.5e-6
        
        assert abs(policy.target_pitch - expected_pitch) < 1e-12, (
            f"target_pitch should be {expected_pitch} (2.5um), got {policy.target_pitch}"
        )
    
    def test_different_voxels_across_values(self):
        """Test that different voxels_across values produce correct pitches."""
        test_cases = [
            (20e-6, 4, 5e-6),
            (20e-6, 10, 2e-6),
            (40e-6, 8, 5e-6),
            (100e-6, 10, 10e-6),
        ]
        
        for min_diameter, voxels_across, expected_pitch in test_cases:
            policy = ResolutionPolicy(
                min_channel_diameter=min_diameter,
                voxels_across_min_diameter=voxels_across,
            )
            
            assert abs(policy.target_pitch - expected_pitch) < 1e-12, (
                f"For min_diameter={min_diameter}, voxels_across={voxels_across}: "
                f"expected pitch {expected_pitch}, got {policy.target_pitch}"
            )
    
    def test_pitch_clamped_to_limits(self):
        """Test that target pitch is clamped to pitch_limits."""
        policy = ResolutionPolicy(
            min_channel_diameter=1e-9,
            voxels_across_min_diameter=8,
            pitch_limits=PitchLimits(min_pitch=1e-6, max_pitch=1e-3),
        )
        
        assert policy.target_pitch >= 1e-6, (
            f"target_pitch should be >= min_pitch (1e-6), got {policy.target_pitch}"
        )
        
        policy2 = ResolutionPolicy(
            min_channel_diameter=1.0,
            voxels_across_min_diameter=8,
            pitch_limits=PitchLimits(min_pitch=1e-6, max_pitch=1e-3),
        )
        
        assert policy2.target_pitch <= 1e-3, (
            f"target_pitch should be <= max_pitch (1e-3), got {policy2.target_pitch}"
        )


class TestPitchRelaxationUnderBudget:
    """C2: Relax pitch under max_voxels pressure."""
    
    def test_pitch_relaxed_when_budget_exceeded(self):
        """Test that pitch is relaxed when voxel budget would be exceeded."""
        policy = ResolutionPolicy(
            min_channel_diameter=20e-6,
            voxels_across_min_diameter=8,
            max_voxels=1_000_000,
            auto_relax_pitch=True,
            pitch_step_factor=1.5,
        )
        
        large_extents = (0.05, 0.05, 0.05)
        
        effective_pitch, was_relaxed, warning = policy.compute_relaxed_pitch(
            base_pitch=policy.target_pitch,
            domain_extents=large_extents,
        )
        
        assert effective_pitch >= policy.target_pitch, (
            f"effective_pitch ({effective_pitch}) should be >= target_pitch ({policy.target_pitch})"
        )
        
        nx = int(np.ceil(large_extents[0] / effective_pitch))
        ny = int(np.ceil(large_extents[1] / effective_pitch))
        nz = int(np.ceil(large_extents[2] / effective_pitch))
        voxel_count = nx * ny * nz
        
        assert voxel_count <= policy.max_voxels * 1.1, (
            f"voxel_count ({voxel_count}) should be <= max_voxels ({policy.max_voxels})"
        )
    
    def test_warning_present_when_relaxed(self):
        """Test that warning is present when pitch is relaxed."""
        policy = ResolutionPolicy(
            min_channel_diameter=20e-6,
            voxels_across_min_diameter=8,
            max_voxels=100_000,
            auto_relax_pitch=True,
            pitch_step_factor=1.5,
        )
        
        large_extents = (0.05, 0.05, 0.05)
        
        effective_pitch, was_relaxed, warning = policy.compute_relaxed_pitch(
            base_pitch=policy.target_pitch,
            domain_extents=large_extents,
        )
        
        if was_relaxed:
            assert len(warning) > 0, "Warning should be present when pitch is relaxed"
            assert "relax" in warning.lower() or "budget" in warning.lower(), (
                f"Warning should mention 'relax' or 'budget': {warning}"
            )
    
    def test_no_relaxation_when_budget_sufficient(self):
        """Test that pitch is not relaxed when budget is sufficient."""
        policy = ResolutionPolicy(
            min_channel_diameter=20e-6,
            voxels_across_min_diameter=8,
            max_voxels=1_000_000_000,
            auto_relax_pitch=True,
        )
        
        small_extents = (0.001, 0.001, 0.001)
        
        effective_pitch, was_relaxed, warning = policy.compute_relaxed_pitch(
            base_pitch=policy.target_pitch,
            domain_extents=small_extents,
        )
        
        assert abs(effective_pitch - policy.target_pitch) < 1e-12, (
            f"effective_pitch should equal target_pitch when budget is sufficient"
        )
        assert not was_relaxed, "was_relaxed should be False when budget is sufficient"
    
    def test_no_relaxation_when_disabled(self):
        """Test that pitch is not relaxed when auto_relax_pitch is False."""
        policy = ResolutionPolicy(
            min_channel_diameter=20e-6,
            voxels_across_min_diameter=8,
            max_voxels=100,
            auto_relax_pitch=False,
        )
        
        large_extents = (0.05, 0.05, 0.05)
        
        effective_pitch, was_relaxed, warning = policy.compute_relaxed_pitch(
            base_pitch=policy.target_pitch,
            domain_extents=large_extents,
        )
        
        assert abs(effective_pitch - policy.target_pitch) < 1e-12, (
            f"effective_pitch should equal target_pitch when auto_relax_pitch is False"
        )
        assert not was_relaxed, "was_relaxed should be False when auto_relax_pitch is False"


class TestDeterministicRelaxation:
    """C3: Deterministic relaxation."""
    
    def test_same_inputs_same_output(self):
        """Test that same inputs produce same effective pitch."""
        policy = ResolutionPolicy(
            min_channel_diameter=20e-6,
            voxels_across_min_diameter=8,
            max_voxels=500_000,
            auto_relax_pitch=True,
            pitch_step_factor=1.5,
        )
        
        extents = (0.03, 0.03, 0.03)
        
        results = []
        for _ in range(5):
            effective_pitch, was_relaxed, warning = policy.compute_relaxed_pitch(
                base_pitch=policy.target_pitch,
                domain_extents=extents,
            )
            results.append(effective_pitch)
        
        for i in range(1, len(results)):
            assert abs(results[i] - results[0]) < 1e-15, (
                f"Relaxation should be deterministic: run 0 = {results[0]}, run {i} = {results[i]}"
            )
    
    def test_deterministic_across_policy_instances(self):
        """Test that different policy instances with same params produce same results."""
        params = {
            "min_channel_diameter": 20e-6,
            "voxels_across_min_diameter": 8,
            "max_voxels": 500_000,
            "auto_relax_pitch": True,
            "pitch_step_factor": 1.5,
        }
        
        extents = (0.03, 0.03, 0.03)
        
        results = []
        for _ in range(3):
            policy = ResolutionPolicy(**params)
            effective_pitch, _, _ = policy.compute_relaxed_pitch(
                base_pitch=policy.target_pitch,
                domain_extents=extents,
            )
            results.append(effective_pitch)
        
        for i in range(1, len(results)):
            assert abs(results[i] - results[0]) < 1e-15, (
                f"Relaxation should be deterministic across instances"
            )


class TestOperationSpecificPitches:
    """Test operation-specific pitch derivation."""
    
    def test_embed_pitch_derived_from_target(self):
        """Test that embed_pitch is derived from target_pitch."""
        policy = ResolutionPolicy(
            min_channel_diameter=20e-6,
            voxels_across_min_diameter=8,
            embed_pitch_factor=1.0,
        )
        
        assert abs(policy.embed_pitch - policy.target_pitch) < 1e-12
    
    def test_merge_pitch_derived_from_target(self):
        """Test that merge_pitch is derived from target_pitch."""
        policy = ResolutionPolicy(
            min_channel_diameter=20e-6,
            voxels_across_min_diameter=8,
            merge_pitch_factor=1.0,
        )
        
        assert abs(policy.merge_pitch - policy.target_pitch) < 1e-12
    
    def test_pathfinding_coarse_pitch_larger_than_fine(self):
        """Test that coarse pathfinding pitch is larger than fine."""
        policy = ResolutionPolicy(
            min_channel_diameter=20e-6,
            voxels_across_min_diameter=8,
            pathfinding_coarse_factor=20.0,
            pathfinding_fine_factor=1.0,
        )
        
        assert policy.pathfinding_pitch_coarse > policy.pathfinding_pitch_fine, (
            f"coarse pitch ({policy.pathfinding_pitch_coarse}) should be > "
            f"fine pitch ({policy.pathfinding_pitch_fine})"
        )
    
    def test_operation_specific_max_voxels(self):
        """Test that operation-specific max_voxels can be set."""
        policy = ResolutionPolicy(
            max_voxels=100_000_000,
            max_voxels_embed=50_000_000,
            max_voxels_merge=25_000_000,
            max_voxels_pathfinding_coarse=10_000_000,
            max_voxels_pathfinding_fine=50_000_000,
        )
        
        assert policy.get_max_voxels_for_operation("embed") == 50_000_000
        assert policy.get_max_voxels_for_operation("merge") == 25_000_000
        assert policy.get_max_voxels_for_operation("pathfinding_coarse") == 10_000_000
        assert policy.get_max_voxels_for_operation("pathfinding_fine") == 50_000_000
        
        assert policy.get_max_voxels_for_operation("unknown") == 100_000_000


class TestInputUnitsConversion:
    """Test input units conversion."""
    
    def test_meters_input_units(self):
        """Test that meters input units work correctly."""
        policy = ResolutionPolicy(
            input_units="m",
            min_channel_diameter=20e-6,
        )
        
        assert abs(policy.min_channel_diameter_m - 20e-6) < 1e-15
    
    def test_mm_input_units(self):
        """Test that mm input units are converted to meters."""
        policy = ResolutionPolicy(
            input_units="mm",
            min_channel_diameter=0.02,
        )
        
        assert abs(policy.min_channel_diameter_m - 20e-6) < 1e-15
    
    def test_um_input_units(self):
        """Test that um input units are converted to meters."""
        policy = ResolutionPolicy(
            input_units="um",
            min_channel_diameter=20.0,
        )
        
        assert abs(policy.min_channel_diameter_m - 20e-6) < 1e-15
