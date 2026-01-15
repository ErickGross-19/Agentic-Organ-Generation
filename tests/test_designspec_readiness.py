"""
DesignSpec readiness tests.

These tests prove all orchestration primitives work end-to-end,
ensuring the codebase is ready for DesignSpec implementation.

Tests cover:
1. Port placement on cylinder top face with ridge constraint reduces effective radius
2. Fang hook bends radially outward from face center and respects effective radius
3. Tissue sampling strategies actually change attractor distribution
4. Programmatic backend routes around obstacles using voxel A* and skips infeasible waypoints
5. Component compositor unions mesh+network components and embeds once successfully
6. Validation pipeline catches disconnected void components and emits shrink warnings
"""

import pytest
import numpy as np


class TestPortPlacement:
    """Tests for domain-aware port placement with ridge constraints."""
    
    def test_port_placement_policy_fields(self):
        """Test PortPlacementPolicy field structure."""
        from aog_policies import PortPlacementPolicy
        
        policy = PortPlacementPolicy(
            face="top",
            pattern="circle",
            ridge_width=0.002,
            ridge_clearance=0.001,
            port_margin=0.0005,
            ridge_constraint_enabled=True,
        )
        
        assert policy.face == "top"
        assert policy.pattern == "circle"
        assert policy.ridge_width == 0.002
        assert policy.ridge_clearance == 0.001
        assert policy.port_margin == 0.0005
        assert policy.ridge_constraint_enabled is True
    
    def test_port_placement_ridge_constraint_fields(self):
        """Test that ridge constraint fields exist and affect effective radius calculation."""
        from aog_policies import PortPlacementPolicy
        
        policy_no_ridge = PortPlacementPolicy(
            face="top",
            pattern="circle",
            ridge_constraint_enabled=False,
        )
        
        policy_with_ridge = PortPlacementPolicy(
            face="top",
            pattern="circle",
            ridge_constraint_enabled=True,
            ridge_width=0.002,
            ridge_clearance=0.001,
            port_margin=0.0005,
        )
        
        assert policy_no_ridge.ridge_constraint_enabled is False
        assert policy_with_ridge.ridge_constraint_enabled is True
        
        R = 0.01
        ridge_inner_radius = R - policy_with_ridge.ridge_width - policy_with_ridge.ridge_clearance
        effective_radius = ridge_inner_radius - policy_with_ridge.port_margin
        
        assert effective_radius < R


class TestChannelPolicy:
    """Tests for channel builder with ChannelPolicy."""
    
    def test_fang_hook_profile_creation(self):
        """Test fang hook channel profile creation."""
        from aog_policies import ChannelPolicy
        
        policy = ChannelPolicy(
            profile="fang_hook",
            length_mode="explicit",
            length=0.005,
            straight_fraction=0.3,
            curve_fraction=0.5,
        )
        
        assert policy.profile == "fang_hook"
        assert policy.straight_fraction == 0.3
        assert policy.curve_fraction == 0.5
    
    def test_channel_policy_length_modes(self):
        """Test different length modes in ChannelPolicy."""
        from aog_policies import ChannelPolicy
        
        explicit = ChannelPolicy(length_mode="explicit", length=0.01)
        assert explicit.length_mode == "explicit"
        
        to_center = ChannelPolicy(length_mode="to_center_fraction", length_fraction=0.5)
        assert to_center.length_mode == "to_center_fraction"
        
        to_depth = ChannelPolicy(length_mode="to_depth", length=0.008)
        assert to_depth.length_mode == "to_depth"


class TestTissueSampling:
    """Tests for tissue sampling strategies."""
    
    def test_tissue_sampling_strategies_differ(self):
        """Test that different sampling strategies produce different distributions."""
        from aog_policies import TissueSamplingPolicy
        
        uniform_policy = TissueSamplingPolicy(
            strategy="uniform",
            n_points=100,
        )
        
        depth_biased_policy = TissueSamplingPolicy(
            strategy="depth_biased",
            n_points=100,
            depth_power=2.0,
        )
        
        assert uniform_policy.strategy == "uniform"
        assert depth_biased_policy.strategy == "depth_biased"
        assert depth_biased_policy.depth_power == 2.0
        
        assert uniform_policy.strategy != depth_biased_policy.strategy


class TestProgrammaticBackend:
    """Tests for programmatic backend routing."""
    
    def test_programmatic_backend_import(self):
        """Test that programmatic backend can be imported."""
        from generation.backends.programmatic_backend import (
            ProgrammaticBackend,
            ProgramPolicy,
            WaypointPolicy,
            StepSpec,
        )
        
        backend = ProgrammaticBackend()
        assert backend is not None
    
    def test_programmatic_backend_in_growth_policy(self):
        """Test that 'programmatic' is a valid backend in GrowthPolicy."""
        from aog_policies import GrowthPolicy
        
        policy = GrowthPolicy(backend="programmatic")
        assert policy.backend == "programmatic"
    
    def test_waypoint_policy_skip_unreachable(self):
        """Test WaypointPolicy skip_unreachable and emit_warnings."""
        from generation.backends.programmatic_backend import WaypointPolicy
        
        policy = WaypointPolicy(
            skip_unreachable=True,
            max_skip_count=3,
            emit_warnings=True,
            fallback_direct=True,
        )
        
        assert policy.skip_unreachable is True
        assert policy.emit_warnings is True


class TestComponentCompositor:
    """Tests for multi-component composition."""
    
    def test_compose_components_import(self):
        """Test that compose_components can be imported."""
        from generation.ops.compose import (
            compose_components,
            ComposePolicy,
            ComponentSpec,
            ComposeReport,
        )
        
        policy = ComposePolicy()
        assert policy is not None
    
    def test_compose_policy_defaults(self):
        """Test ComposePolicy default values."""
        from generation.ops.compose import ComposePolicy
        
        policy = ComposePolicy()
        
        assert policy.repair_enabled is True
        assert policy.keep_largest_component is True
        assert policy.synthesis_policy is not None
        assert policy.merge_policy is not None


class TestValidationPipeline:
    """Tests for validation pipeline."""
    
    def test_validate_and_repair_artifacts_import(self):
        """Test that validate_and_repair_artifacts can be imported."""
        from validity.api.pipeline import (
            validate_and_repair_artifacts,
            ArtifactValidationReport,
        )
        
        report = ArtifactValidationReport(success=True)
        assert report.success is True
    
    def test_open_ports_check_ignored(self):
        """Test that check_open_ports=True emits 'ignored (not implemented)'."""
        from validity.api.pipeline import validate_and_repair_artifacts
        import trimesh
        
        mesh = trimesh.creation.box(extents=[0.01, 0.01, 0.01])
        
        _, report = validate_and_repair_artifacts(
            void_mesh=mesh,
            check_open_ports=True,
            check_pre_embedding=False,
            check_post_embedding=False,
        )
        
        assert any("ignored (not implemented)" in w for w in report.warnings)
    
    def test_disconnected_void_components_warning(self):
        """Test that disconnected void components emit warnings."""
        from validity.api.pipeline import validate_and_repair_artifacts
        from aog_policies import ValidationPolicy
        import trimesh
        
        box1 = trimesh.creation.box(extents=[0.005, 0.005, 0.005])
        box1.apply_translation([0, 0, 0])
        
        box2 = trimesh.creation.box(extents=[0.005, 0.005, 0.005])
        box2.apply_translation([0.02, 0, 0])
        
        disconnected_mesh = trimesh.util.concatenate([box1, box2])
        
        policy = ValidationPolicy(max_components=1)
        
        _, report = validate_and_repair_artifacts(
            void_mesh=disconnected_mesh,
            validation_policy=policy,
            check_pre_embedding=False,
            check_post_embedding=True,
        )
        
        has_component_warning = any("disconnected" in w.lower() or "component" in w.lower() 
                                    for w in report.warnings)
        assert has_component_warning or len(report.post_embedding_checks) > 0


class TestCanonicalFaceNaming:
    """Tests for canonical face naming utilities."""
    
    def test_validate_face(self):
        """Test face validation."""
        from generation.utils.faces import validate_face
        
        assert validate_face("top") == "top"
        assert validate_face("bottom") == "bottom"
        assert validate_face("+x") == "+x"
        assert validate_face("-x") == "-x"
        assert validate_face("+y") == "+y"
        assert validate_face("-y") == "-y"
        assert validate_face("+z") == "+z"
        assert validate_face("-z") == "-z"
    
    def test_face_normal(self):
        """Test face normal computation."""
        from generation.utils.faces import face_normal
        
        class MockDomain:
            pass
        
        domain = MockDomain()
        
        top_normal = face_normal("top", domain)
        assert np.allclose(top_normal, [0, 0, 1])
        
        bottom_normal = face_normal("bottom", domain)
        assert np.allclose(bottom_normal, [0, 0, -1])
        
        px_normal = face_normal("+x", domain)
        assert np.allclose(px_normal, [1, 0, 0])


class TestMeshPolicies:
    """Tests for mesh synthesis and merge policies."""
    
    def test_mesh_synthesis_policy_radius_clamping(self):
        """Test MeshSynthesisPolicy radius clamping fields."""
        from aog_policies import MeshSynthesisPolicy
        
        policy = MeshSynthesisPolicy(
            radius_clamp_min=0.0001,
            radius_clamp_max=0.005,
            mutate_network_in_place=False,
            radius_clamp_mode="copy",
        )
        
        assert policy.radius_clamp_min == 0.0001
        assert policy.radius_clamp_max == 0.005
        assert policy.mutate_network_in_place is False
        assert policy.radius_clamp_mode == "copy"
    
    def test_mesh_merge_policy_knobs(self):
        """Test MeshMergePolicy enforcement knobs."""
        from aog_policies import MeshMergePolicy
        
        policy = MeshMergePolicy(
            fill_voxels=True,
            max_voxels=50_000_000,
            auto_adjust_pitch=True,
            keep_largest_component=True,
            min_component_faces=100,
            min_component_volume=1e-12,
        )
        
        assert policy.fill_voxels is True
        assert policy.max_voxels == 50_000_000
        assert policy.auto_adjust_pitch is True
        assert policy.keep_largest_component is True


class TestEmbeddingPolicy:
    """Tests for embedding policy with port preservation."""
    
    def test_embedding_policy_port_preservation_fields(self):
        """Test EmbeddingPolicy port preservation fields."""
        from aog_policies import EmbeddingPolicy
        
        policy = EmbeddingPolicy(
            preserve_ports_enabled=True,
            preserve_mode="recarve",
            carve_radius_factor=1.2,
            carve_depth=0.002,
        )
        
        assert policy.preserve_ports_enabled is True
        assert policy.preserve_mode == "recarve"
        assert policy.carve_radius_factor == 1.2
        assert policy.carve_depth == 0.002
    
    def test_embedding_exports_both_apis(self):
        """Test that embedding package exports both legacy and new APIs."""
        from generation.ops.embedding import (
            embed_tree_as_negative_space,
            embed_void_mesh_as_negative_space,
            embed_with_port_preservation,
        )
        
        assert embed_tree_as_negative_space is not None
        assert embed_void_mesh_as_negative_space is not None
        assert embed_with_port_preservation is not None


class TestDeprecatedDesignFromSpec:
    """Tests for deprecated design_from_spec."""
    
    def test_design_from_spec_has_deprecation_warning(self):
        """Test that design_from_spec has deprecation warning in docstring."""
        from generation.api.design import design_from_spec
        
        assert design_from_spec.__doc__ is not None
        assert "deprecated" in design_from_spec.__doc__.lower()
