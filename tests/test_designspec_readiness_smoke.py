"""
DesignSpec readiness smoke tests.

This test module ensures no regressions before adding new capability.
It validates:
1. All policies are JSON-serializable
2. No module/package name collisions exist
3. All public API functions return OperationReport with requested/effective policy
4. A tiny end-to-end: place ports -> channels -> merge -> embed -> validate
"""

import pytest
import json
import numpy as np


class TestPolicyJSONSerializable:
    """Test that all policies are JSON-serializable."""
    
    def test_port_placement_policy_serializable(self):
        """Test PortPlacementPolicy is JSON-serializable."""
        from aog_policies import PortPlacementPolicy
        
        policy = PortPlacementPolicy(
            face="top",
            pattern="circle",
            ridge_width=0.002,
            ridge_clearance=0.001,
            port_margin=0.0005,
            ridge_constraint_enabled=True,
        )
        
        d = policy.to_dict()
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        
        assert restored["face"] == "top"
        assert restored["pattern"] == "circle"
        assert restored["ridge_width"] == 0.002
    
    def test_channel_policy_serializable(self):
        """Test ChannelPolicy is JSON-serializable."""
        from aog_policies import ChannelPolicy
        
        policy = ChannelPolicy(
            profile="fang_hook",
            length_mode="explicit",
            length=0.005,
            straight_fraction=0.3,
            curve_fraction=0.5,
        )
        
        d = policy.to_dict()
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        
        assert restored["profile"] == "fang_hook"
        assert restored["length_mode"] == "explicit"
    
    def test_growth_policy_serializable(self):
        """Test GrowthPolicy is JSON-serializable."""
        from aog_policies import GrowthPolicy
        
        policy = GrowthPolicy(
            backend="programmatic",
            target_terminals=100,
            max_iterations=500,
            backend_params={"mode": "network", "path_algorithm": "astar_voxel"},
        )
        
        d = policy.to_dict()
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        
        assert restored["backend"] == "programmatic"
        assert restored["backend_params"]["mode"] == "network"
    
    def test_tissue_sampling_policy_serializable(self):
        """Test TissueSamplingPolicy is JSON-serializable."""
        from aog_policies import TissueSamplingPolicy
        
        policy = TissueSamplingPolicy(
            strategy="depth_biased",
            n_points=100,
            depth_power=2.0,
        )
        
        d = policy.to_dict()
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        
        assert restored["strategy"] == "depth_biased"
        assert restored["n_points"] == 100
    
    def test_collision_policy_serializable(self):
        """Test CollisionPolicy is JSON-serializable."""
        from aog_policies import CollisionPolicy
        
        policy = CollisionPolicy(
            enabled=True,
            check_collisions=True,
            collision_clearance=0.0002,
        )
        
        d = policy.to_dict()
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        
        assert restored["enabled"] is True
        assert restored["collision_clearance"] == 0.0002
    
    def test_mesh_synthesis_policy_serializable(self):
        """Test MeshSynthesisPolicy is JSON-serializable."""
        from aog_policies import MeshSynthesisPolicy
        
        policy = MeshSynthesisPolicy(
            add_node_spheres=True,
            cap_ends=True,
            radius_clamp_min=0.0001,
            radius_clamp_max=0.005,
        )
        
        d = policy.to_dict()
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        
        assert restored["add_node_spheres"] is True
        assert restored["radius_clamp_min"] == 0.0001
    
    def test_mesh_merge_policy_serializable(self):
        """Test MeshMergePolicy is JSON-serializable."""
        from aog_policies import MeshMergePolicy
        
        policy = MeshMergePolicy(
            mode="auto",
            voxel_pitch=5e-5,
            auto_adjust_pitch=True,
            max_voxels=100_000_000,
        )
        
        d = policy.to_dict()
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        
        assert restored["mode"] == "auto"
        assert restored["voxel_pitch"] == 5e-5
    
    def test_embedding_policy_serializable(self):
        """Test EmbeddingPolicy is JSON-serializable."""
        from aog_policies import EmbeddingPolicy
        
        policy = EmbeddingPolicy(
            voxel_pitch=3e-4,
            shell_thickness=2e-3,
            preserve_ports_enabled=True,
            carve_radius_factor=1.2,
            carve_depth=0.002,
        )
        
        d = policy.to_dict()
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        
        assert restored["voxel_pitch"] == 3e-4
        assert restored["preserve_ports_enabled"] is True
    
    def test_validation_policy_serializable(self):
        """Test ValidationPolicy is JSON-serializable."""
        from aog_policies import ValidationPolicy
        
        policy = ValidationPolicy(
            max_components=1,
        )
        
        d = policy.to_dict()
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        
        assert restored["max_components"] == 1
    
    def test_repair_policy_serializable(self):
        """Test RepairPolicy is JSON-serializable."""
        from aog_policies import RepairPolicy
        
        policy = RepairPolicy()
        
        d = policy.to_dict()
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        
        assert isinstance(restored, dict)


class TestModuleImports:
    """Test that all modules can be imported without collisions."""
    
    def test_aog_policies_import(self):
        """Test aog_policies package imports cleanly."""
        import aog_policies
        
        assert hasattr(aog_policies, "PortPlacementPolicy")
        assert hasattr(aog_policies, "ChannelPolicy")
        assert hasattr(aog_policies, "GrowthPolicy")
        assert hasattr(aog_policies, "TissueSamplingPolicy")
        assert hasattr(aog_policies, "CollisionPolicy")
        assert hasattr(aog_policies, "MeshSynthesisPolicy")
        assert hasattr(aog_policies, "MeshMergePolicy")
        assert hasattr(aog_policies, "EmbeddingPolicy")
        assert hasattr(aog_policies, "ValidationPolicy")
        assert hasattr(aog_policies, "RepairPolicy")
        assert hasattr(aog_policies, "OperationReport")
    
    def test_generation_api_import(self):
        """Test generation.api package imports cleanly."""
        from generation.api import (
            generate_network,
            generate_void_mesh,
            build_component,
            embed_void,
            embed_void_mesh_as_negative_space,
        )
        
        assert callable(generate_network)
        assert callable(generate_void_mesh)
        assert callable(build_component)
        assert callable(embed_void)
        assert callable(embed_void_mesh_as_negative_space)
    
    def test_validity_api_import(self):
        """Test validity.api package imports cleanly."""
        from validity.api import (
            validate_mesh,
            validate_network,
            validate_artifacts,
            repair_mesh,
            validate_repair_validate,
            run_full_pipeline,
        )
        
        assert callable(validate_mesh)
        assert callable(validate_network)
        assert callable(validate_artifacts)
        assert callable(repair_mesh)
        assert callable(validate_repair_validate)
        assert callable(run_full_pipeline)
    
    def test_programmatic_backend_import(self):
        """Test programmatic backend imports cleanly."""
        from generation.backends.programmatic_backend import (
            ProgrammaticBackend,
            StepSpec,
            GenerationReport,
        )
        from aog_policies import ProgramPolicy, WaypointPolicy
        
        assert ProgrammaticBackend is not None
        assert ProgramPolicy is not None
        assert WaypointPolicy is not None
        assert StepSpec is not None
        assert GenerationReport is not None
    
    def test_embedding_ops_import(self):
        """Test embedding ops imports cleanly."""
        from generation.ops.embedding import (
            embed_with_port_preservation,
            embed_void_mesh_as_negative_space,
        )
        
        assert callable(embed_with_port_preservation)
        assert callable(embed_void_mesh_as_negative_space)
    
    def test_voxel_recarve_import(self):
        """Test voxel recarve imports cleanly."""
        from generation.ops.embedding.enhanced_embedding import (
            voxel_recarve_ports,
            RecarveReport,
            PortRecarveResult,
        )
        
        assert callable(voxel_recarve_ports)
        assert RecarveReport is not None
        assert PortRecarveResult is not None


class TestOperationReportPattern:
    """Test that public API functions return OperationReport with requested/effective policy."""
    
    def test_generate_network_returns_operation_report(self):
        """Test generate_network returns OperationReport."""
        from generation.api import generate_network
        from generation.core.domain import CylinderDomain
        from aog_policies import GrowthPolicy, OperationReport
        
        domain = CylinderDomain(radius=0.005, height=0.003)
        ports = {
            "inlets": [{"position": (0, 0, 0.0015), "radius": 0.0005, "direction": (0, 0, -1), "vessel_type": "arterial"}],
            "outlets": [{"position": (0, 0, -0.0015), "radius": 0.0003, "direction": (0, 0, 1), "vessel_type": "venous"}],
        }
        growth_policy = GrowthPolicy(backend="cco_hybrid", max_iterations=10, target_terminals=5)
        
        network, report = generate_network(
            generator_kind="cco_hybrid",
            domain=domain,
            ports=ports,
            growth_policy=growth_policy,
        )
        
        assert isinstance(report, OperationReport)
        assert report.operation == "generate_network"
        assert report.requested_policy is not None
        assert report.effective_policy is not None
        assert isinstance(report.requested_policy, dict)
        assert isinstance(report.effective_policy, dict)
    
    def test_embed_void_returns_operation_report(self):
        """Test embed_void returns OperationReport."""
        from generation.api import embed_void
        from generation.core.domain import CylinderDomain
        from aog_policies import EmbeddingPolicy, OperationReport
        import trimesh
        
        domain = CylinderDomain(radius=0.005, height=0.003)
        void_mesh = trimesh.creation.cylinder(radius=0.001, height=0.002)
        
        policy = EmbeddingPolicy(
            voxel_pitch=5e-4,
            preserve_ports_enabled=False,
        )
        
        solid, void_out, shell, report = embed_void(
            domain=domain,
            void_mesh=void_mesh,
            embedding_policy=policy,
        )
        
        assert isinstance(report, OperationReport)
        assert report.operation == "embed_void"
        assert report.requested_policy is not None
        assert report.effective_policy is not None


class TestEndToEndSmoke:
    """Tiny end-to-end test: place ports -> channels -> merge -> embed -> validate."""
    
    def test_minimal_end_to_end_pipeline(self):
        """Test minimal end-to-end pipeline runs without errors."""
        from generation.core.domain import CylinderDomain
        from generation.ops.primitives.channels import create_channel_from_policy
        from generation.api import embed_void
        from validity.api import validate_mesh
        from aog_policies import (
            ChannelPolicy,
            EmbeddingPolicy,
            OperationReport,
        )
        import trimesh
        
        domain = CylinderDomain(radius=0.005, height=0.003)
        
        port_position = (0, 0, 0.0015)
        port_direction = (0, 0, -1)
        port_radius = 0.0005
        
        channel_policy = ChannelPolicy(
            profile="straight",
            length_mode="explicit",
            length=0.002,
        )
        
        channel_mesh, channel_report = create_channel_from_policy(
            start=port_position,
            direction=port_direction,
            radius=port_radius,
            policy=channel_policy,
            domain_depth=0.003,
        )
        
        assert isinstance(channel_report, OperationReport)
        assert channel_mesh is not None
        assert len(channel_mesh.vertices) > 0
        
        embedding_policy = EmbeddingPolicy(
            voxel_pitch=5e-4,
            preserve_ports_enabled=False,
        )
        
        solid, void_out, shell, embed_report = embed_void(
            domain=domain,
            void_mesh=channel_mesh,
            embedding_policy=embedding_policy,
        )
        
        assert isinstance(embed_report, OperationReport)
        assert embed_report.operation == "embed_void"
        assert solid is not None
        
        if solid is not None and len(solid.vertices) > 0:
            validation_result = validate_mesh(solid)
            assert validation_result is not None
    
    def test_port_preservation_with_voxel_recarve(self):
        """Test port preservation using voxel recarve (no boolean backend dependency)."""
        from generation.core.domain import CylinderDomain
        from generation.api import embed_void
        from aog_policies import EmbeddingPolicy, OperationReport
        import trimesh
        
        domain = CylinderDomain(radius=0.005, height=0.003)
        void_mesh = trimesh.creation.cylinder(radius=0.001, height=0.002)
        
        ports = [
            {
                "position": (0, 0, 0.0015),
                "direction": (0, 0, 1),
                "radius": 0.0005,
            }
        ]
        
        policy = EmbeddingPolicy(
            voxel_pitch=5e-4,
            preserve_ports_enabled=True,
            carve_radius_factor=1.2,
            carve_depth=0.001,
        )
        
        solid, void_out, shell, report = embed_void(
            domain=domain,
            void_mesh=void_mesh,
            embedding_policy=policy,
            ports=ports,
        )
        
        assert isinstance(report, OperationReport)
        assert report.operation == "embed_void"
        
        if "recarve_report" in report.metadata:
            recarve_report = report.metadata["recarve_report"]
            assert "ports_carved" in recarve_report
            assert "voxel_pitch_used" in recarve_report


class TestResolutionPolicy:
    """Test ResolutionPolicy for scale-aware tolerances."""
    
    def test_resolution_policy_serializable(self):
        """Test ResolutionPolicy is JSON-serializable."""
        from aog_policies import ResolutionPolicy, PitchLimits
        import json
        
        policy = ResolutionPolicy(
            input_units="um",
            min_channel_diameter=20.0,  # 20 µm
            voxels_across_min_diameter=8,
            max_voxels=100_000_000,
            pitch_limits=PitchLimits(min_pitch=1e-6, max_pitch=1e-3),
            auto_relax_pitch=True,
            rel_epsilon=1e-6,
        )
        
        d = policy.to_dict()
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        
        assert restored["input_units"] == "um"
        assert restored["min_channel_diameter"] == 20.0
        assert restored["voxels_across_min_diameter"] == 8
    
    def test_resolution_policy_target_pitch_computation(self):
        """Test target_pitch is computed correctly from min_channel_diameter."""
        from aog_policies import ResolutionPolicy
        
        policy = ResolutionPolicy(
            input_units="um",
            min_channel_diameter=20.0,  # 20 µm
            voxels_across_min_diameter=8,
        )
        
        expected_pitch = 2.5e-6  # 20 µm / 8 = 2.5 µm = 2.5e-6 m
        assert abs(policy.target_pitch - expected_pitch) < 1e-12
    
    def test_resolution_policy_derived_pitches(self):
        """Test derived pitches are computed correctly."""
        from aog_policies import ResolutionPolicy
        
        policy = ResolutionPolicy(
            input_units="um",
            min_channel_diameter=20.0,  # 20 µm
            voxels_across_min_diameter=8,
            embed_pitch_factor=1.0,
            merge_pitch_factor=1.0,
            pathfinding_coarse_factor=20.0,
            pathfinding_fine_factor=1.0,
        )
        
        target = 2.5e-6  # 2.5 µm
        
        assert abs(policy.embed_pitch - target) < 1e-12
        assert abs(policy.merge_pitch - target) < 1e-12
        assert abs(policy.pathfinding_pitch_coarse - target * 20) < 1e-12
        assert abs(policy.pathfinding_pitch_fine - target) < 1e-12
    
    def test_resolution_policy_pitch_relaxation(self):
        """Test pitch relaxation when voxel budget exceeded."""
        from aog_policies import ResolutionPolicy
        
        policy = ResolutionPolicy(
            input_units="m",
            min_channel_diameter=2e-5,  # 20 µm
            voxels_across_min_diameter=8,
            max_voxels=1_000_000,  # 1M voxels - small budget
            auto_relax_pitch=True,
            pitch_step_factor=1.5,
        )
        
        domain_extents = (0.05, 0.05, 0.05)  # 50mm cube
        
        effective_pitch, was_relaxed, warning = policy.compute_relaxed_pitch(
            policy.target_pitch,
            domain_extents,
        )
        
        assert was_relaxed
        assert effective_pitch > policy.target_pitch
        assert len(warning) > 0
    
    def test_resolution_policy_eps_scaling(self):
        """Test eps scales correctly with domain size."""
        from aog_policies import ResolutionPolicy
        
        policy = ResolutionPolicy(rel_epsilon=1e-6)
        
        small_scale = 0.005  # 5mm domain
        large_scale = 0.05  # 50mm domain
        
        eps_small = policy.eps(small_scale)
        eps_large = policy.eps(large_scale)
        
        assert eps_small < eps_large
        assert abs(eps_small - 5e-9) < 1e-15  # 0.005 * 1e-6
        assert abs(eps_large - 5e-8) < 1e-14  # 0.05 * 1e-6


class TestScaleUtilities:
    """Test scale-aware tolerance utilities."""
    
    def test_domain_scale_cylinder(self):
        """Test domain_scale for CylinderDomain."""
        from generation.core.domain import CylinderDomain
        from generation.utils.scale import domain_scale
        
        domain = CylinderDomain(radius=0.005, height=0.003)  # 5mm radius, 3mm height
        
        scale = domain_scale(domain)
        
        assert abs(scale - 0.003) < 1e-10
    
    def test_domain_scale_ellipsoid(self):
        """Test domain_scale for EllipsoidDomain."""
        from generation.core.domain import EllipsoidDomain
        from generation.utils.scale import domain_scale
        
        domain = EllipsoidDomain(
            semi_axis_a=0.025,
            semi_axis_b=0.025,
            semi_axis_c=0.005,
        )  # 50mm x 50mm x 10mm
        
        scale = domain_scale(domain)
        
        assert abs(scale - 0.01) < 1e-10
    
    def test_eps_with_domain(self):
        """Test eps computation with domain object."""
        from generation.core.domain import CylinderDomain
        from generation.utils.scale import eps
        
        domain = CylinderDomain(radius=0.005, height=0.003)  # 3mm height
        
        epsilon = eps(domain)
        
        assert epsilon > 0
        assert epsilon < 1e-6
    
    def test_eps_with_resolution_policy(self):
        """Test eps computation with ResolutionPolicy."""
        from generation.core.domain import CylinderDomain
        from generation.utils.scale import eps
        from aog_policies import ResolutionPolicy
        
        domain = CylinderDomain(radius=0.025, height=0.05)  # 50mm height
        policy = ResolutionPolicy(rel_epsilon=1e-5)
        
        epsilon = eps(domain, policy)
        
        expected = 0.05 * 1e-5  # 5e-7
        assert abs(epsilon - expected) < 1e-12
    
    def test_scale_tolerances(self):
        """Test various scale-aware tolerances."""
        from generation.core.domain import CylinderDomain
        from generation.utils.scale import (
            containment_tolerance,
            collision_tolerance,
            snap_tolerance,
        )
        
        domain = CylinderDomain(radius=0.005, height=0.003)
        
        contain_tol = containment_tolerance(domain)
        collision_tol = collision_tolerance(domain)
        snap_tol = snap_tolerance(domain)
        
        assert contain_tol > 0
        assert collision_tol > contain_tol
        assert snap_tol > collision_tol


class TestProgrammaticBackendUnifiedAPI:
    """Test programmatic backend works through unified API with backend_params."""
    
    def test_programmatic_backend_via_growth_policy(self):
        """Test programmatic backend runs through unified API with backend_params."""
        from generation.api import generate_network
        from generation.core.domain import CylinderDomain
        from aog_policies import GrowthPolicy, OperationReport
        
        domain = CylinderDomain(radius=0.005, height=0.003)
        
        ports = {
            "inlets": [{"position": (0, 0, 0.0015), "radius": 0.0005}],
            "outlets": [{"position": (0, 0, -0.0015), "radius": 0.0003}],
        }
        
        backend_params = {
            "mode": "network",
            "path_algorithm": "straight",
            "waypoint_policy": {"allow_skip": True},
        }
        
        growth_policy = GrowthPolicy(
            backend="programmatic",
            backend_params=backend_params,
        )
        
        network, report = generate_network(
            generator_kind="programmatic",
            domain=domain,
            ports=ports,
            growth_policy=growth_policy,
        )
        
        assert isinstance(report, OperationReport)
        assert report.operation == "generate_network"
        assert network is not None
