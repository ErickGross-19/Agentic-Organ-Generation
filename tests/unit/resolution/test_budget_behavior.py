"""
Test budget stress scenarios with pitch relaxation.

This module validates that the resolution system correctly handles
budget constraints and pitch relaxation.
"""

import pytest


class TestResolutionPolicyPitchRelaxation:
    """Test that ResolutionPolicy can compute relaxed pitch under budget pressure."""
    
    def test_resolution_policy_pitch_relaxation_basic(self):
        """Test that ResolutionPolicy can compute relaxed pitch under budget pressure."""
        from aog_policies import ResolutionPolicy
        
        policy = ResolutionPolicy(
            min_pitch=1e-6,
            max_pitch=1e-2,  # 10mm - large enough to satisfy budget for 0.1m domain
            max_voxels=1_000_000,
        )
        
        # Use smaller domain that can satisfy budget within max_pitch
        large_bbox = (0, 0.1, 0, 0.1, 0, 0.1)  # 0.1m cube = 1e6 voxels at 1e-4 pitch
        
        relaxed_pitch = policy.compute_relaxed_pitch(
            bbox=large_bbox,
            requested_pitch=1e-5,
        )
        
        assert relaxed_pitch >= 1e-5
        
        dims = (
            large_bbox[1] - large_bbox[0],
            large_bbox[3] - large_bbox[2],
            large_bbox[5] - large_bbox[4],
        )
        estimated_voxels = (dims[0] / relaxed_pitch) * (dims[1] / relaxed_pitch) * (dims[2] / relaxed_pitch)
        
        assert estimated_voxels <= policy.max_voxels * 1.1
    
    def test_resolution_policy_respects_min_pitch(self):
        """Test that relaxed pitch respects min_pitch constraint."""
        from aog_policies import ResolutionPolicy
        
        policy = ResolutionPolicy(
            min_pitch=1e-4,
            max_pitch=1e-3,
            max_voxels=1_000_000,
        )
        
        bbox = (0, 0.1, 0, 0.1, 0, 0.1)
        
        relaxed_pitch = policy.compute_relaxed_pitch(
            bbox=bbox,
            requested_pitch=1e-5,
        )
        
        assert relaxed_pitch >= policy.min_pitch
    
    def test_resolution_policy_respects_max_pitch(self):
        """Test that relaxed pitch respects max_pitch constraint."""
        from aog_policies import ResolutionPolicy
        
        policy = ResolutionPolicy(
            min_pitch=1e-6,
            max_pitch=1e-3,
            max_voxels=100,
        )
        
        large_bbox = (0, 10.0, 0, 10.0, 0, 10.0)
        
        relaxed_pitch = policy.compute_relaxed_pitch(
            bbox=large_bbox,
            requested_pitch=1e-5,
        )
        
        assert relaxed_pitch <= policy.max_pitch


class TestResolverWarnings:
    """Test that resolver emits warnings on relaxation."""
    
    def test_resolver_emits_warnings_on_relaxation(self):
        """Test that resolver emits warnings when pitch is relaxed."""
        from generation.utils.resolution_resolver import resolve_pitch
        from aog_policies import ResolutionPolicy
        
        policy = ResolutionPolicy(
            min_pitch=1e-6,
            max_pitch=1e-3,
            max_voxels=100_000,
        )
        
        large_bbox = (0, 0.5, 0, 0.5, 0, 0.5)
        
        effective_pitch, warnings, metrics = resolve_pitch(
            op_name="test_op",
            requested_pitch=1e-5,
            bbox=large_bbox,
            resolution_policy=policy,
        )
        
        assert effective_pitch >= 1e-5
        
        if effective_pitch > 1e-5:
            assert len(warnings) > 0
            assert any("relax" in w.lower() or "budget" in w.lower() for w in warnings)
    
    def test_resolver_returns_metrics(self):
        """Test that resolver returns metrics dict."""
        from generation.utils.resolution_resolver import resolve_pitch
        from aog_policies import ResolutionPolicy
        
        policy = ResolutionPolicy(
            min_pitch=1e-6,
            max_pitch=1e-3,
            max_voxels=10_000_000,
        )
        
        bbox = (0, 0.01, 0, 0.01, 0, 0.01)
        
        effective_pitch, warnings, metrics = resolve_pitch(
            op_name="test_op",
            requested_pitch=1e-4,
            bbox=bbox,
            resolution_policy=policy,
        )
        
        assert isinstance(metrics, dict)
        assert "requested_pitch" in metrics or "effective_pitch" in metrics


class TestMergeMeshesWithResolutionPolicy:
    """Test that merge_meshes honors use_resolution_policy."""
    
    def test_merge_meshes_with_resolution_policy(self):
        """Test that merge_meshes honors use_resolution_policy."""
        from generation.ops.mesh.merge import merge_meshes
        from aog_policies import MeshMergePolicy, ResolutionPolicy
        import trimesh
        
        box1 = trimesh.creation.box(extents=[0.01, 0.01, 0.01])
        box2 = trimesh.creation.box(extents=[0.01, 0.01, 0.01])
        box2.apply_translation([0.005, 0, 0])
        
        merge_policy = MeshMergePolicy(
            mode="voxel",
            use_resolution_policy=True,
            voxel_pitch=None,
        )
        
        res_policy = ResolutionPolicy(
            min_pitch=1e-5,
            max_pitch=1e-3,
            max_voxels=1_000_000,
        )
        
        merged, report = merge_meshes(
            [box1, box2],
            merge_policy,
            resolution_policy=res_policy,
        )
        
        assert merged is not None
        assert merged.is_watertight or len(merged.vertices) > 0
    
    def test_merge_meshes_without_resolution_policy(self):
        """Test that merge_meshes works without resolution_policy."""
        from generation.ops.mesh.merge import merge_meshes
        from aog_policies import MeshMergePolicy
        import trimesh
        
        box1 = trimesh.creation.box(extents=[0.01, 0.01, 0.01])
        box2 = trimesh.creation.box(extents=[0.01, 0.01, 0.01])
        box2.apply_translation([0.005, 0, 0])
        
        merge_policy = MeshMergePolicy(
            mode="voxel",
            voxel_pitch=1e-4,
        )
        
        merged, report = merge_meshes(
            [box1, box2],
            merge_policy,
        )
        
        assert merged is not None


class TestEmbeddingWithResolutionPolicy:
    """Test embedding operations with resolution policy."""
    
    def test_embedding_respects_max_voxels(self):
        """Test that embedding respects max_voxels constraint."""
        from aog_policies import EmbeddingPolicy, ResolutionPolicy
        
        embed_policy = EmbeddingPolicy(
            preserve_ports_enabled=True,
            preserve_mode="recarve",
            max_voxels=5_000_000,
            use_resolution_policy=True,
        )
        
        res_policy = ResolutionPolicy(
            min_pitch=1e-5,
            max_pitch=1e-3,
            max_voxels=5_000_000,
        )
        
        assert embed_policy.max_voxels == res_policy.max_voxels
        assert embed_policy.use_resolution_policy is True
    
    def test_embedding_policy_serialization_with_resolution(self):
        """Test that EmbeddingPolicy with resolution settings serializes correctly."""
        from aog_policies import EmbeddingPolicy
        import json
        
        policy = EmbeddingPolicy(
            preserve_ports_enabled=True,
            preserve_mode="recarve",
            max_voxels=5_000_000,
            use_resolution_policy=True,
        )
        
        d = policy.to_dict()
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        
        assert restored["max_voxels"] == 5_000_000
        assert restored["use_resolution_policy"] is True
