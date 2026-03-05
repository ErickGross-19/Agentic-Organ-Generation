"""
Test operation pitch resolver shared behavior (C4).

This module verifies that all voxel-based operations use the same resolver
and respect max_voxels budgets consistently.
"""

import pytest
import numpy as np

from aog_policies import (
    ResolutionPolicy,
    MeshMergePolicy,
    EmbeddingPolicy,
    HierarchicalPathfindingPolicy,
    OpenPortPolicy,
)


class TestSharedResolverBehavior:
    """C4: All voxel-based operations use the same resolver."""
    
    def test_mesh_merge_uses_resolution_policy(self):
        """Test that mesh merge respects ResolutionPolicy pitch."""
        resolution = ResolutionPolicy(
            min_channel_diameter=20e-6,
            voxels_across_min_diameter=8,
            max_voxels=10_000_000,
            merge_pitch_factor=1.0,
        )
        
        merge_policy = MeshMergePolicy(
            voxel_pitch=resolution.merge_pitch,
            max_voxels=resolution.get_max_voxels_for_operation("merge"),
        )
        
        assert abs(merge_policy.voxel_pitch - resolution.merge_pitch) < 1e-15, (
            f"MeshMergePolicy pitch should match ResolutionPolicy merge_pitch"
        )
    
    def test_embedding_uses_resolution_policy(self):
        """Test that embedding respects ResolutionPolicy pitch."""
        resolution = ResolutionPolicy(
            min_channel_diameter=20e-6,
            voxels_across_min_diameter=8,
            max_voxels=10_000_000,
            embed_pitch_factor=1.0,
        )
        
        embed_policy = EmbeddingPolicy(
            voxel_pitch=resolution.embed_pitch,
            max_voxels=resolution.get_max_voxels_for_operation("embed"),
        )
        
        assert abs(embed_policy.voxel_pitch - resolution.embed_pitch) < 1e-15, (
            f"EmbeddingPolicy pitch should match ResolutionPolicy embed_pitch"
        )
    
    def test_pathfinding_coarse_uses_resolution_policy(self):
        """Test that pathfinding coarse stage respects ResolutionPolicy."""
        resolution = ResolutionPolicy(
            min_channel_diameter=20e-6,
            voxels_across_min_diameter=8,
            max_voxels=10_000_000,
            pathfinding_coarse_factor=20.0,
            max_voxels_pathfinding_coarse=5_000_000,
        )
        
        pathfinding_policy = HierarchicalPathfindingPolicy(
            pitch_coarse=resolution.pathfinding_pitch_coarse,
            max_voxels_coarse=resolution.get_max_voxels_for_operation("pathfinding_coarse"),
        )
        
        assert abs(pathfinding_policy.pitch_coarse - resolution.pathfinding_pitch_coarse) < 1e-15
        assert pathfinding_policy.max_voxels_coarse == 5_000_000
    
    def test_pathfinding_fine_uses_resolution_policy(self):
        """Test that pathfinding fine stage respects ResolutionPolicy."""
        resolution = ResolutionPolicy(
            min_channel_diameter=20e-6,
            voxels_across_min_diameter=8,
            max_voxels=10_000_000,
            pathfinding_fine_factor=1.0,
            max_voxels_pathfinding_fine=25_000_000,
        )
        
        pathfinding_policy = HierarchicalPathfindingPolicy(
            pitch_fine=resolution.pathfinding_pitch_fine,
            max_voxels_fine=resolution.get_max_voxels_for_operation("pathfinding_fine"),
        )
        
        assert abs(pathfinding_policy.pitch_fine - resolution.pathfinding_pitch_fine) < 1e-15
        assert pathfinding_policy.max_voxels_fine == 25_000_000
    
    def test_open_port_roi_uses_resolution_policy(self):
        """Test that open-port ROI respects ResolutionPolicy."""
        resolution = ResolutionPolicy(
            min_channel_diameter=20e-6,
            voxels_across_min_diameter=8,
            max_voxels=10_000_000,
            max_voxels_open_port_roi=1_000_000,
        )
        
        open_port_policy = OpenPortPolicy(
            max_voxels_roi=resolution.get_max_voxels_for_operation("open_port_roi"),
        )
        
        assert open_port_policy.max_voxels_roi == 1_000_000


class TestConsistentEffectivePitchRules:
    """Test that all operations emit consistent effective_pitch rules."""
    
    def test_all_operations_respect_max_voxels(self):
        """Test that all operations respect their max_voxels budgets."""
        resolution = ResolutionPolicy(
            min_channel_diameter=20e-6,
            voxels_across_min_diameter=8,
            max_voxels=10_000_000,
            max_voxels_embed=5_000_000,
            max_voxels_merge=2_500_000,
            max_voxels_pathfinding_coarse=1_000_000,
            max_voxels_pathfinding_fine=10_000_000,
            max_voxels_open_port_roi=500_000,
            auto_relax_pitch=True,
        )
        
        large_extents = (0.05, 0.05, 0.05)
        
        operations = [
            ("embed", resolution.embed_pitch, resolution.get_max_voxels_for_operation("embed")),
            ("merge", resolution.merge_pitch, resolution.get_max_voxels_for_operation("merge")),
            ("pathfinding_coarse", resolution.pathfinding_pitch_coarse, resolution.get_max_voxels_for_operation("pathfinding_coarse")),
            ("pathfinding_fine", resolution.pathfinding_pitch_fine, resolution.get_max_voxels_for_operation("pathfinding_fine")),
            ("open_port_roi", resolution.target_pitch, resolution.get_max_voxels_for_operation("open_port_roi")),
        ]
        
        for op_name, base_pitch, max_voxels in operations:
            effective_pitch, was_relaxed, warning = resolution.compute_relaxed_pitch(
                base_pitch=base_pitch,
                domain_extents=large_extents,
                max_voxels_override=max_voxels,
            )
            
            nx = int(np.ceil(large_extents[0] / effective_pitch))
            ny = int(np.ceil(large_extents[1] / effective_pitch))
            nz = int(np.ceil(large_extents[2] / effective_pitch))
            voxel_count = nx * ny * nz
            
            assert voxel_count <= max_voxels * 1.1, (
                f"{op_name}: voxel_count ({voxel_count}) exceeds max_voxels ({max_voxels})"
            )
    
    def test_relaxation_factor_consistent(self):
        """Test that relaxation factor is consistent across operations."""
        resolution = ResolutionPolicy(
            min_channel_diameter=20e-6,
            voxels_across_min_diameter=8,
            max_voxels=1_000_000,
            auto_relax_pitch=True,
            pitch_step_factor=1.5,
        )
        
        extents = (0.03, 0.03, 0.03)
        
        effective_pitch_1, _, _ = resolution.compute_relaxed_pitch(
            base_pitch=resolution.target_pitch,
            domain_extents=extents,
        )
        
        effective_pitch_2, _, _ = resolution.compute_relaxed_pitch(
            base_pitch=resolution.target_pitch,
            domain_extents=extents,
        )
        
        assert abs(effective_pitch_1 - effective_pitch_2) < 1e-15, (
            f"Relaxation should be consistent: {effective_pitch_1} vs {effective_pitch_2}"
        )
    
    def test_pitch_step_factor_applied_correctly(self):
        """Test that pitch_step_factor is applied correctly during relaxation."""
        resolution = ResolutionPolicy(
            min_channel_diameter=20e-6,
            voxels_across_min_diameter=8,
            max_voxels=100,
            auto_relax_pitch=True,
            pitch_step_factor=2.0,
        )
        
        extents = (0.01, 0.01, 0.01)
        
        effective_pitch, was_relaxed, _ = resolution.compute_relaxed_pitch(
            base_pitch=resolution.target_pitch,
            domain_extents=extents,
        )
        
        if was_relaxed:
            ratio = effective_pitch / resolution.target_pitch
            
            expected_steps = np.log(ratio) / np.log(2.0)
            assert expected_steps >= 0, "Relaxation should increase pitch"
            
            assert abs(expected_steps - round(expected_steps)) < 0.01, (
                f"Relaxation should be in steps of pitch_step_factor (2.0): ratio={ratio}"
            )


class TestOperationPitchDerivation:
    """Test that operation-specific pitches are derived correctly."""
    
    def test_embed_pitch_factor(self):
        """Test that embed_pitch uses embed_pitch_factor."""
        resolution = ResolutionPolicy(
            min_channel_diameter=20e-6,
            voxels_across_min_diameter=8,
            embed_pitch_factor=2.0,
        )
        
        expected = resolution.target_pitch * 2.0
        assert abs(resolution.embed_pitch - expected) < 1e-15
    
    def test_merge_pitch_factor(self):
        """Test that merge_pitch uses merge_pitch_factor."""
        resolution = ResolutionPolicy(
            min_channel_diameter=20e-6,
            voxels_across_min_diameter=8,
            merge_pitch_factor=0.5,
        )
        
        expected = resolution.target_pitch * 0.5
        assert abs(resolution.merge_pitch - expected) < 1e-15
    
    def test_pathfinding_coarse_factor(self):
        """Test that pathfinding_pitch_coarse uses pathfinding_coarse_factor."""
        resolution = ResolutionPolicy(
            min_channel_diameter=20e-6,
            voxels_across_min_diameter=8,
            pathfinding_coarse_factor=20.0,
        )
        
        expected = resolution.target_pitch * 20.0
        assert abs(resolution.pathfinding_pitch_coarse - expected) < 1e-15
    
    def test_pathfinding_fine_factor(self):
        """Test that pathfinding_pitch_fine uses pathfinding_fine_factor."""
        resolution = ResolutionPolicy(
            min_channel_diameter=20e-6,
            voxels_across_min_diameter=8,
            pathfinding_fine_factor=1.0,
        )
        
        expected = resolution.target_pitch * 1.0
        assert abs(resolution.pathfinding_pitch_fine - expected) < 1e-15


class TestBudgetInheritance:
    """Test that operation-specific budgets inherit from global when not set."""
    
    def test_embed_inherits_global_max_voxels(self):
        """Test that embed uses global max_voxels when not specified."""
        resolution = ResolutionPolicy(
            max_voxels=50_000_000,
        )
        
        assert resolution.get_max_voxels_for_operation("embed") == 50_000_000
    
    def test_embed_uses_specific_when_set(self):
        """Test that embed uses specific max_voxels when set."""
        resolution = ResolutionPolicy(
            max_voxels=50_000_000,
            max_voxels_embed=25_000_000,
        )
        
        assert resolution.get_max_voxels_for_operation("embed") == 25_000_000
    
    def test_unknown_operation_uses_global(self):
        """Test that unknown operations use global max_voxels."""
        resolution = ResolutionPolicy(
            max_voxels=50_000_000,
        )
        
        assert resolution.get_max_voxels_for_operation("unknown_op") == 50_000_000
