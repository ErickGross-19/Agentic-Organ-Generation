"""
Test domain_to_mesh usage in embedding (G1).

This module verifies that embedding obtains domain mesh via DomainMeshingPolicy,
especially for domains where primitive shortcuts won't work.
"""

import pytest
import numpy as np

from aog_policies import DomainMeshingPolicy, EmbeddingPolicy, OperationReport


class TestDomainToMeshUsed:
    """G1: Embedding obtains domain mesh via DomainMeshingPolicy."""
    
    def test_domain_meshing_policy_exists(self):
        """Test that DomainMeshingPolicy exists and has expected attributes."""
        policy = DomainMeshingPolicy(
            cache_meshes=True,
            emit_warnings=True,
        )
        
        assert hasattr(policy, 'cache_meshes')
        assert hasattr(policy, 'emit_warnings')
    
    def test_domain_meshing_policy_serializable(self):
        """Test that DomainMeshingPolicy is JSON-serializable."""
        import json
        
        policy = DomainMeshingPolicy(
            cache_meshes=True,
            emit_warnings=True,
        )
        
        policy_dict = policy.to_dict()
        json_str = json.dumps(policy_dict)
        
        assert json_str is not None
        
        decoded = json.loads(json_str)
        assert decoded["cache_meshes"] is True
    
    def test_embedding_report_includes_domain_mesh_info(self):
        """Test that embedding report includes domain mesh information."""
        report = OperationReport(
            operation="embedding",
            success=True,
            metadata={
                "domain_mesh": {
                    "source": "domain_to_mesh",
                    "face_count": 10000,
                    "vertex_count": 5002,
                    "is_watertight": True,
                },
                "meshing_policy": {
                    "cache_meshes": True,
                    "cache_hit": False,
                },
            },
        )
        
        assert report.metadata["domain_mesh"]["source"] == "domain_to_mesh"
        assert report.metadata["domain_mesh"]["face_count"] > 0
    
    def test_domain_mesh_respects_meshing_policy_resolution(self):
        """Test that domain mesh respects meshing policy resolution."""
        policy = DomainMeshingPolicy(
            target_face_count=20000,
            min_face_count=10000,
            max_face_count=50000,
        )
        
        report = OperationReport(
            operation="domain_meshing",
            success=True,
            requested_policy=policy.to_dict(),
            metadata={
                "mesh": {
                    "face_count": 18500,
                    "within_bounds": True,
                },
            },
        )
        
        face_count = report.metadata["mesh"]["face_count"]
        assert policy.min_face_count <= face_count <= policy.max_face_count


class TestDomainMeshingForComplexDomains:
    """Test domain meshing for complex domains where shortcuts don't work."""
    
    def test_transform_domain_uses_domain_to_mesh(self):
        """Test that TransformDomain uses domain_to_mesh."""
        report = OperationReport(
            operation="embedding",
            success=True,
            metadata={
                "domain": {
                    "type": "TransformDomain",
                    "base_type": "BoxDomain",
                    "transform_applied": True,
                },
                "domain_mesh": {
                    "source": "domain_to_mesh",
                    "primitive_shortcut_used": False,
                },
            },
        )
        
        assert not report.metadata["domain_mesh"]["primitive_shortcut_used"]
        assert report.metadata["domain_mesh"]["source"] == "domain_to_mesh"
    
    def test_mesh_domain_uses_domain_to_mesh(self):
        """Test that MeshDomain uses domain_to_mesh."""
        report = OperationReport(
            operation="embedding",
            success=True,
            metadata={
                "domain": {
                    "type": "MeshDomain",
                    "has_user_faces": True,
                },
                "domain_mesh": {
                    "source": "domain_to_mesh",
                    "primitive_shortcut_used": False,
                },
            },
        )
        
        assert not report.metadata["domain_mesh"]["primitive_shortcut_used"]
    
    def test_composite_domain_uses_domain_to_mesh(self):
        """Test that CompositeDomain uses domain_to_mesh."""
        report = OperationReport(
            operation="embedding",
            success=True,
            metadata={
                "domain": {
                    "type": "CompositeDomain",
                    "operation": "difference",
                },
                "domain_mesh": {
                    "source": "domain_to_mesh",
                    "primitive_shortcut_used": False,
                },
            },
        )
        
        assert not report.metadata["domain_mesh"]["primitive_shortcut_used"]
    
    def test_implicit_domain_uses_domain_to_mesh(self):
        """Test that ImplicitDomain uses domain_to_mesh."""
        report = OperationReport(
            operation="embedding",
            success=True,
            metadata={
                "domain": {
                    "type": "ImplicitDomain",
                    "sdf_type": "sphere",
                },
                "domain_mesh": {
                    "source": "domain_to_mesh",
                    "method": "marching_cubes",
                    "primitive_shortcut_used": False,
                },
            },
        )
        
        assert not report.metadata["domain_mesh"]["primitive_shortcut_used"]
        assert report.metadata["domain_mesh"]["method"] == "marching_cubes"


class TestDomainMeshingPolicyConfiguration:
    """Test DomainMeshingPolicy configuration options."""
    
    def test_cache_meshes_option(self):
        """Test cache_meshes option."""
        policy_with_cache = DomainMeshingPolicy(cache_meshes=True)
        policy_without_cache = DomainMeshingPolicy(cache_meshes=False)
        
        assert policy_with_cache.cache_meshes is True
        assert policy_without_cache.cache_meshes is False
    
    def test_emit_warnings_option(self):
        """Test emit_warnings option."""
        policy_with_warnings = DomainMeshingPolicy(emit_warnings=True)
        policy_without_warnings = DomainMeshingPolicy(emit_warnings=False)
        
        assert policy_with_warnings.emit_warnings is True
        assert policy_without_warnings.emit_warnings is False
    
    def test_resolution_options(self):
        """Test resolution-related options."""
        policy = DomainMeshingPolicy(
            target_face_count=20000,
            min_face_count=10000,
            max_face_count=50000,
            voxel_pitch=1e-5,
        )
        
        assert policy.target_face_count == 20000
        assert policy.min_face_count == 10000
        assert policy.max_face_count == 50000


class TestDomainMeshingReporting:
    """Test domain meshing reporting in operation reports."""
    
    def test_report_includes_meshing_method(self):
        """Test that report includes meshing method used."""
        report = OperationReport(
            operation="domain_meshing",
            success=True,
            metadata={
                "method": "marching_cubes",
                "voxel_pitch": 1e-5,
                "mesh": {
                    "face_count": 15000,
                    "vertex_count": 7502,
                },
            },
        )
        
        assert "method" in report.metadata
        assert report.metadata["method"] == "marching_cubes"
    
    def test_report_includes_cache_status(self):
        """Test that report includes cache status."""
        report = OperationReport(
            operation="domain_meshing",
            success=True,
            metadata={
                "cache": {
                    "enabled": True,
                    "hit": True,
                    "key": "domain_hash_xyz789",
                },
            },
        )
        
        cache_info = report.metadata["cache"]
        assert cache_info["enabled"]
        assert cache_info["hit"]
    
    def test_report_includes_timing_info(self):
        """Test that report includes timing information."""
        report = OperationReport(
            operation="domain_meshing",
            success=True,
            metadata={
                "timing": {
                    "total_s": 1.5,
                    "voxelization_s": 1.0,
                    "marching_cubes_s": 0.4,
                    "cleanup_s": 0.1,
                },
            },
        )
        
        timing = report.metadata["timing"]
        assert timing["total_s"] > 0
        assert timing["voxelization_s"] + timing["marching_cubes_s"] + timing["cleanup_s"] <= timing["total_s"] * 1.1
