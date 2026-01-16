"""
DesignSpecRunner readiness test suite (Z1).

This integration test suite validates the runner contract:
- Policies-only control surface
- JSON-friendly inputs
- Budget behavior with pitch relaxation
- Effective policy snapshots in reports
"""

import pytest
import numpy as np
import ast
import os


class TestPolicyOwnership:
    """A1: Verify canonical modules don't import policy dataclasses from non-aog_policies modules."""
    
    def test_generate_api_no_ops_policy_imports(self):
        """Test that generation/api/generate.py doesn't import policies from ops."""
        import generation.api.generate as generate_module
        
        source_file = generate_module.__file__
        with open(source_file, 'r') as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and 'generation.ops' in node.module:
                    for alias in node.names:
                        assert 'Policy' not in alias.name, (
                            f"generate.py imports {alias.name} from {node.module}"
                        )
    
    def test_embed_api_no_ops_policy_imports(self):
        """Test that generation/api/embed.py doesn't import policies from ops."""
        import generation.api.embed as embed_module
        
        source_file = embed_module.__file__
        with open(source_file, 'r') as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and 'generation.ops' in node.module:
                    for alias in node.names:
                        assert 'Policy' not in alias.name, (
                            f"embed.py imports {alias.name} from {node.module}"
                        )
    
    def test_validity_runner_uses_aog_policies(self):
        """Test that validity/runner.py imports policies from aog_policies."""
        import validity.runner as runner_module
        
        source_file = runner_module.__file__
        with open(source_file, 'r') as f:
            source = f.read()
        
        assert 'from aog_policies' in source
        assert 'ValidationPolicy' in source or 'ResolutionPolicy' in source
    
    def test_policy_round_trip_serialization(self):
        """Test policy round-trip: serialize → reconstruct → identical behavior."""
        from aog_policies import (
            ResolutionPolicy,
            EmbeddingPolicy,
            MeshMergePolicy,
            ValidationPolicy,
        )
        
        res_policy = ResolutionPolicy(
            min_pitch=1e-6,
            max_pitch=1e-3,
            min_voxels_across_feature=8,
            max_voxels=10_000_000,
        )
        res_dict = res_policy.to_dict()
        res_reconstructed = ResolutionPolicy.from_dict(res_dict)
        
        assert res_reconstructed.min_pitch == res_policy.min_pitch
        assert res_reconstructed.max_pitch == res_policy.max_pitch
        assert res_reconstructed.min_voxels_across_feature == res_policy.min_voxels_across_feature
        assert res_reconstructed.max_voxels == res_policy.max_voxels
        
        embed_policy = EmbeddingPolicy(
            preserve_ports_enabled=True,
            preserve_mode="recarve",
            max_voxels=5_000_000,
            use_resolution_policy=True,
        )
        embed_dict = embed_policy.to_dict()
        embed_reconstructed = EmbeddingPolicy.from_dict(embed_dict)
        
        assert embed_reconstructed.preserve_ports_enabled == embed_policy.preserve_ports_enabled
        assert embed_reconstructed.max_voxels == embed_policy.max_voxels
        assert embed_reconstructed.use_resolution_policy == embed_policy.use_resolution_policy


class TestBudgetBehavior:
    """Test budget stress scenarios with pitch relaxation."""
    
    def test_resolution_policy_pitch_relaxation(self):
        """Test that ResolutionPolicy can compute relaxed pitch under budget pressure."""
        from aog_policies import ResolutionPolicy
        
        policy = ResolutionPolicy(
            min_pitch=1e-6,
            max_pitch=1e-3,
            max_voxels=1_000_000,
        )
        
        large_bbox = (0, 1.0, 0, 1.0, 0, 1.0)
        
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


class TestValidityRunner:
    """Test canonical validity runner integration."""
    
    def test_validity_runner_import(self):
        """Test that validity runner can be imported."""
        from validity.runner import run_validity_checks, ValidityReport, CheckResult
        
        assert run_validity_checks is not None
        assert ValidityReport is not None
        assert CheckResult is not None
    
    def test_validity_report_json_serializable(self):
        """Test that ValidityReport is JSON serializable."""
        from validity.runner import ValidityReport, CheckResult
        import json
        
        check = CheckResult(
            name="test_check",
            passed=True,
            message="Test passed",
            details={"key": "value"},
        )
        
        report = ValidityReport(
            success=True,
            checks=[check],
            warnings=["test warning"],
            requested_policies={"validation": {}},
            effective_policies={"validation": {}},
        )
        
        report_dict = report.to_dict()
        json_str = json.dumps(report_dict)
        
        assert json_str is not None
        
        parsed = json.loads(json_str)
        assert parsed["success"] is True
        assert len(parsed["checks"]) == 1
    
    def test_validity_runner_with_simple_mesh(self):
        """Test validity runner with a simple mesh."""
        from validity.runner import run_validity_checks
        from aog_policies import ValidationPolicy, ResolutionPolicy
        import trimesh
        
        void_mesh = trimesh.creation.box(extents=[0.01, 0.01, 0.01])
        domain_mesh = trimesh.creation.box(extents=[0.02, 0.02, 0.02])
        
        validation_policy = ValidationPolicy()
        resolution_policy = ResolutionPolicy()
        
        report = run_validity_checks(
            void_mesh=void_mesh,
            domain_mesh=domain_mesh,
            validation_policy=validation_policy,
            resolution_policy=resolution_policy,
        )
        
        assert report is not None
        assert hasattr(report, 'success')
        assert hasattr(report, 'checks')


class TestDomainMeshing:
    """Test domain meshing with policy control."""
    
    def test_domain_to_mesh_import(self):
        """Test that domain_to_mesh can be imported."""
        from generation.ops.domain_meshing import domain_to_mesh
        
        assert domain_to_mesh is not None
    
    def test_domain_to_mesh_box(self):
        """Test domain_to_mesh with BoxDomain."""
        from generation.ops.domain_meshing import domain_to_mesh
        from generation.core.domain import BoxDomain
        from aog_policies import DomainMeshingPolicy, ResolutionPolicy
        
        domain = BoxDomain(
            x_min=0, x_max=0.01,
            y_min=0, y_max=0.01,
            z_min=0, z_max=0.01,
        )
        
        meshing_policy = DomainMeshingPolicy()
        resolution_policy = ResolutionPolicy()
        
        mesh, report = domain_to_mesh(
            domain=domain,
            meshing_policy=meshing_policy,
            resolution_policy=resolution_policy,
        )
        
        assert mesh is not None
        assert len(mesh.vertices) > 0
        assert mesh.is_watertight


class TestEmbeddingPipeline:
    """Test unified embedding pipeline."""
    
    def test_embed_void_mesh_import(self):
        """Test that embed_void_mesh_as_negative_space can be imported."""
        from generation.ops.embedding import embed_void_mesh_as_negative_space
        
        assert embed_void_mesh_as_negative_space is not None
    
    def test_embedding_with_resolution_policy(self):
        """Test embedding with resolution policy."""
        from generation.ops.embedding import embed_void_mesh_as_negative_space
        from aog_policies import EmbeddingPolicy, ResolutionPolicy
        import trimesh
        
        domain_mesh = trimesh.creation.box(extents=[0.02, 0.02, 0.02])
        void_mesh = trimesh.creation.box(extents=[0.005, 0.005, 0.01])
        
        embedding_policy = EmbeddingPolicy(
            use_resolution_policy=True,
            max_voxels=500_000,
        )
        resolution_policy = ResolutionPolicy(
            max_voxels=500_000,
        )
        
        result = embed_void_mesh_as_negative_space(
            domain_mesh=domain_mesh,
            void_mesh=void_mesh,
            embedding_policy=embedding_policy,
            resolution_policy=resolution_policy,
        )
        
        assert result is not None


class TestPathfindingHierarchical:
    """Test hierarchical pathfinding is mandatory."""
    
    def test_hierarchical_pathfinding_import(self):
        """Test that hierarchical pathfinding can be imported."""
        from generation.ops.pathfinding import hierarchical_astar
        
        assert hierarchical_astar is not None
    
    def test_hierarchical_policy_fields(self):
        """Test HierarchicalPathfindingPolicy fields."""
        from aog_policies import HierarchicalPathfindingPolicy
        
        policy = HierarchicalPathfindingPolicy(
            coarse_pitch_factor=4.0,
            max_voxels_coarse=1_000_000,
            max_voxels_fine=5_000_000,
            corridor_radius_factor=2.0,
        )
        
        assert policy.coarse_pitch_factor == 4.0
        assert policy.max_voxels_coarse == 1_000_000
        assert policy.max_voxels_fine == 5_000_000


class TestChannelPrimitives:
    """Test channel primitives with policy control."""
    
    def test_create_channel_from_policy_import(self):
        """Test that create_channel_from_policy can be imported."""
        from generation.ops.primitives.channels import create_channel_from_policy
        
        assert create_channel_from_policy is not None
    
    def test_channel_length_modes(self):
        """Test channel length modes work correctly."""
        from aog_policies import ChannelPolicy
        
        explicit = ChannelPolicy(
            length_mode="explicit",
            length=0.005,
            stop_before_boundary=0.001,
        )
        assert explicit.length == 0.005
        
        to_depth = ChannelPolicy(
            length_mode="to_depth",
            stop_before_boundary=0.001,
        )
        assert to_depth.length_mode == "to_depth"
        assert to_depth.stop_before_boundary == 0.001


class TestOpenPortValidation:
    """Test open-port validation with ROI budgeting."""
    
    def test_open_port_check_import(self):
        """Test that open port check can be imported."""
        from validity.checks.open_ports import check_open_ports, check_port_open
        
        assert check_open_ports is not None
        assert check_port_open is not None
    
    def test_port_check_result_structure(self):
        """Test PortCheckResult structure."""
        from validity.checks.open_ports import PortCheckResult
        
        result = PortCheckResult(
            port_name="test_port",
            is_open=True,
            connectivity_ratio=0.95,
            voxel_pitch=1e-4,
            roi_voxels=10000,
            warnings=[],
        )
        
        assert result.port_name == "test_port"
        assert result.is_open is True
        assert result.connectivity_ratio == 0.95


class TestEffectivePolicySnapshots:
    """Test that reports include effective policy snapshots."""
    
    def test_operation_report_has_effective_policy(self):
        """Test that OperationReport includes effective_policy."""
        from generation.core.report import OperationReport
        
        report = OperationReport(
            operation="test_op",
            success=True,
            requested_policy={"pitch": 1e-5},
            effective_policy={"pitch": 2e-5},
            warnings=["Pitch relaxed due to budget"],
        )
        
        assert report.requested_policy["pitch"] == 1e-5
        assert report.effective_policy["pitch"] == 2e-5
        assert len(report.warnings) == 1
    
    def test_validity_report_has_policy_snapshots(self):
        """Test that ValidityReport includes policy snapshots."""
        from validity.runner import ValidityReport
        
        report = ValidityReport(
            success=True,
            checks=[],
            warnings=[],
            requested_policies={"validation": {"max_components": 1}},
            effective_policies={"validation": {"max_components": 1}},
        )
        
        assert "validation" in report.requested_policies
        assert "validation" in report.effective_policies
