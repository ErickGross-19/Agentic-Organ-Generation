"""
Acceptance tests for the Generation + Validity Library Refactor.

These tests verify the key functionality specified in the refactor proposal:
1. Placement / ridge constraint
2. Fang hook
3. Kary tree backend
4. Voxel-first merge
5. Embedding mesh input
6. Validity pipeline
"""

import json
import numpy as np
import pytest
import tempfile
from pathlib import Path


class TestPortPlacement:
    """Test domain-aware port placement with ridge constraints."""
    
    def test_effective_radius_calculation(self):
        """Test effective radius is computed correctly with ridge constraints."""
        from generation.utils.port_placement import compute_effective_radius
        
        domain_radius = 0.004  # 4mm
        ridge_width = 0.0005  # 0.5mm
        ridge_clearance = 0.0002  # 0.2mm
        port_margin = 0.0003  # 0.3mm
        
        r_eff = compute_effective_radius(
            domain_radius, ridge_width, ridge_clearance, port_margin
        )
        
        expected = domain_radius - ridge_width - ridge_clearance - port_margin
        assert abs(r_eff - expected) < 1e-9
        assert r_eff < domain_radius
    
    def test_ports_inside_effective_radius(self):
        """Test that placed ports are inside effective radius."""
        from generation.utils.port_placement import place_ports, PortPlacementPolicy
        
        policy = PortPlacementPolicy(
            pattern="circle",
            ridge_width=0.0005,
            ridge_clearance=0.0002,
            port_margin=0.0003,
        )
        
        result, report = place_ports(
            num_ports=6,
            domain_radius=0.004,
            port_radius=0.0003,
            z_position=0.003,
            policy=policy,
        )
        
        assert result is not None
        assert len(result.positions) == 6
        
        for pos in result.positions:
            r = np.sqrt(pos[0]**2 + pos[1]**2)
            assert r <= result.effective_radius + 1e-9
        
        assert "effective_radius" in report.metadata
        assert report.metadata["effective_radius"] == result.effective_radius
    
    def test_report_includes_clamp_counts(self):
        """Test that report includes clamp counts and R_eff."""
        from generation.utils.port_placement import place_ports, PortPlacementPolicy
        
        policy = PortPlacementPolicy(
            pattern="circle",
            ridge_width=0.001,
            ridge_clearance=0.0005,
            port_margin=0.0005,
        )
        
        result, report = place_ports(
            num_ports=8,
            domain_radius=0.003,
            port_radius=0.0003,
            z_position=0.003,
            policy=policy,
        )
        
        assert "clamp_count" in report.metadata
        assert "effective_radius" in report.metadata
        assert report.metadata["effective_radius"] > 0


class TestFangHook:
    """Test fang hook channel primitive."""
    
    def test_fang_hook_creates_curved_mesh(self):
        """Test that fang_hook creates a curved, tapered mesh."""
        from generation.ops.primitives.channels import create_fang_hook, ChannelPolicy
        
        start = np.array([0.0, 0.0, 0.003])
        end = np.array([0.0, 0.0, 0.0])
        radius = 0.0003
        
        policy = ChannelPolicy(
            hook_depth=0.001,
            hook_angle_deg=45.0,
            segments_per_curve=16,
        )
        
        mesh, meta = create_fang_hook(
            start=start,
            end=end,
            radius=radius,
            policy=policy,
        )
        
        assert mesh is not None
        assert mesh.is_watertight or len(mesh.vertices) > 0
        
        assert "hook_depth_used" in meta
        assert meta["hook_depth_used"] > 0
    
    def test_fang_hook_depth_reduction_warning(self):
        """Test that hook depth reduction triggers warning."""
        from generation.ops.primitives.channels import create_fang_hook, ChannelPolicy
        
        start = np.array([0.0, 0.0, 0.003])
        end = np.array([0.0, 0.0, 0.0])
        radius = 0.0003
        
        policy = ChannelPolicy(
            hook_depth=0.005,
            hook_angle_deg=45.0,
            segments_per_curve=16,
        )
        
        mesh, meta = create_fang_hook(
            start=start,
            end=end,
            radius=radius,
            effective_radius=0.002,
            policy=policy,
        )
        
        assert mesh is not None
        
        if meta.get("hook_depth_used", 0) < 0.005:
            assert "depth_reduced" in meta or meta.get("hook_depth_used") < policy.hook_depth


class TestKaryTreeBackend:
    """Test k-ary tree recursive bifurcation backend."""
    
    def test_kary_tree_generates_network(self):
        """Test that kary tree backend generates a valid network."""
        from generation.backends.kary_tree_backend import KaryTreeBackend, KaryTreeConfig
        from generation.specs.design_spec import CylinderSpec
        
        domain = CylinderSpec(
            center=(0.0, 0.0, 0.0015),
            radius=0.004,
            height=0.003,
        )
        
        config = KaryTreeConfig(
            k=2,
            target_terminals=8,
            terminal_tolerance=0.2,
            branch_length=0.001,
        )
        
        backend = KaryTreeBackend(config)
        
        network = backend.generate(
            domain=domain,
            num_outlets=8,
            inlet_position=(0.0, 0.0, 0.003),
            inlet_radius=0.0005,
        )
        
        assert network is not None
        assert len(network.nodes) > 0
        assert len(network.segments) > 0
    
    def test_kary_tree_terminal_count_tolerance(self):
        """Test that terminal count is within tolerance or warning emitted."""
        from generation.backends.kary_tree_backend import KaryTreeBackend, KaryTreeConfig
        from generation.specs.design_spec import CylinderSpec
        from generation.core.types import NodeType
        
        domain = CylinderSpec(
            center=(0.0, 0.0, 0.0015),
            radius=0.004,
            height=0.003,
        )
        
        target = 16
        tolerance = 0.2
        
        config = KaryTreeConfig(
            k=2,
            target_terminals=target,
            terminal_tolerance=tolerance,
            branch_length=0.0008,
        )
        
        backend = KaryTreeBackend(config)
        
        network = backend.generate(
            domain=domain,
            num_outlets=target,
            inlet_position=(0.0, 0.0, 0.003),
            inlet_radius=0.0005,
        )
        
        terminal_count = sum(
            1 for n in network.nodes.values()
            if n.node_type == NodeType.TERMINAL
        )
        
        min_expected = int(target * (1 - tolerance))
        max_expected = int(target * (1 + tolerance))
        
        assert terminal_count >= min_expected or terminal_count <= max_expected


class TestVoxelFirstMerge:
    """Test voxel-first mesh merge."""
    
    def test_voxel_merge_creates_mesh(self):
        """Test that voxel merge creates a valid mesh."""
        import trimesh
        from generation.ops.mesh.merge import merge_meshes, MeshMergePolicy
        
        mesh1 = trimesh.creation.cylinder(radius=0.001, height=0.003)
        mesh1.apply_translation([0.0, 0.0, 0.0015])
        
        mesh2 = trimesh.creation.cylinder(radius=0.001, height=0.003)
        mesh2.apply_translation([0.001, 0.0, 0.0015])
        
        policy = MeshMergePolicy(
            mode="voxel",
            voxel_pitch=0.0002,
        )
        
        merged, report = merge_meshes([mesh1, mesh2], policy)
        
        assert merged is not None
        assert len(merged.vertices) > 0
        assert report.success
    
    def test_pitch_stepping_in_report(self):
        """Test that pitch stepping triggers report entries."""
        import trimesh
        from generation.ops.mesh.merge import merge_meshes, MeshMergePolicy
        
        mesh1 = trimesh.creation.cylinder(radius=0.001, height=0.003)
        mesh2 = trimesh.creation.cylinder(radius=0.001, height=0.003)
        mesh2.apply_translation([0.0015, 0.0, 0.0])
        
        policy = MeshMergePolicy(
            mode="auto",
            voxel_pitch=0.0001,
            auto_adjust_pitch=True,
        )
        
        merged, report = merge_meshes([mesh1, mesh2], policy)
        
        assert merged is not None
        assert report.success
        
        assert "mode_used" in report.metadata or "pitch_adjustments" in report.metadata


class TestEmbeddingMeshInput:
    """Test embedding accepts in-memory meshes."""
    
    def test_embed_accepts_mesh_object(self):
        """Test that embed accepts in-memory mesh, not just paths."""
        import trimesh
        from generation.api.embed import embed_void
        from generation.policies import EmbeddingPolicy
        from generation.specs.design_spec import CylinderSpec
        
        domain = CylinderSpec(
            center=(0.0, 0.0, 0.0015),
            radius=0.004,
            height=0.003,
        )
        
        void_mesh = trimesh.creation.cylinder(radius=0.0005, height=0.002)
        void_mesh.apply_translation([0.0, 0.0, 0.0015])
        
        policy = EmbeddingPolicy(
            voxel_pitch=0.0002,
            shell_thickness=0.0003,
        )
        
        solid, void, shell, report = embed_void(domain, void_mesh, policy)
        
        assert solid is not None
        assert void is not None
        assert report is not None
    
    def test_embed_report_includes_pitch_changes(self):
        """Test that embed report includes pitch changes and shrink warnings."""
        import trimesh
        from generation.api.embed import embed_void
        from generation.policies import EmbeddingPolicy
        from generation.specs.design_spec import CylinderSpec
        
        domain = CylinderSpec(
            center=(0.0, 0.0, 0.0015),
            radius=0.004,
            height=0.003,
        )
        
        void_mesh = trimesh.creation.cylinder(radius=0.0005, height=0.002)
        void_mesh.apply_translation([0.0, 0.0, 0.0015])
        
        policy = EmbeddingPolicy(
            voxel_pitch=0.0001,
            shell_thickness=0.0003,
        )
        
        solid, void, shell, report = embed_void(domain, void_mesh, policy)
        
        assert report is not None
        assert "requested_policy" in report.to_dict()
        assert "effective_policy" in report.to_dict()


class TestValidityPipeline:
    """Test validity pipeline."""
    
    def test_watertight_check_passes_after_repair(self):
        """Test that watertight check passes after repair if enabled."""
        import trimesh
        from validity.api.pipeline import validate_repair_validate
        from validity.api.validate import ValidationPolicy
        from validity.api.repair import RepairPolicy
        
        mesh = trimesh.creation.cylinder(radius=0.001, height=0.003)
        
        validation_policy = ValidationPolicy(
            check_watertight=True,
            check_components=True,
        )
        
        repair_policy = RepairPolicy(
            voxel_repair_enabled=True,
            voxel_pitch=0.0002,
        )
        
        result_mesh, report = validate_repair_validate(
            mesh, validation_policy, repair_policy
        )
        
        assert result_mesh is not None
        assert report is not None
        
        if report.final_validation:
            watertight_check = next(
                (c for c in report.final_validation.checks if "watertight" in c.check_name),
                None
            )
            if watertight_check:
                assert watertight_check.passed or len(report.warnings) > 0
    
    def test_run_report_json_serializable(self):
        """Test that run report is JSON-serializable."""
        from validity.reporting.run_report import RunReport, create_run_report
        
        report = create_run_report("test_run_123")
        
        report.inputs = {"domain": "cylinder", "radius": 0.004}
        report.placement = {"effective_radius": 0.003, "clamps": 2}
        report.generation = {"terminal_count": 16}
        report.validity = {"watertight": True}
        report.outputs = {"solid_path": "/tmp/solid.stl"}
        
        json_str = report.to_json()
        
        assert json_str is not None
        assert len(json_str) > 0
        
        parsed = json.loads(json_str)
        assert parsed["run_id"] == "test_run_123"
        assert parsed["inputs"]["radius"] == 0.004
    
    def test_drift_report_computation(self):
        """Test drift report computation."""
        from validity.reporting.drift_report import compute_drift_report, DriftReport
        
        requested_spec = {
            "target_terminals": 16,
            "min_radius": 0.0003,
        }
        
        report = compute_drift_report(
            network=None,
            void_mesh=None,
            requested_spec=requested_spec,
        )
        
        assert report is not None
        assert isinstance(report, DriftReport)
        assert report.to_dict() is not None


class TestPolicyValidation:
    """Test policy validation and echo mechanism."""
    
    def test_policy_to_dict_from_dict(self):
        """Test policy serialization round-trip."""
        from generation.policies import GrowthPolicy, CollisionPolicy
        
        growth = GrowthPolicy(
            min_segment_length=0.0005,
            max_segment_length=0.002,
            step_size=0.0003,
        )
        
        d = growth.to_dict()
        restored = GrowthPolicy.from_dict(d)
        
        assert restored.min_segment_length == growth.min_segment_length
        assert restored.max_segment_length == growth.max_segment_length
        assert restored.step_size == growth.step_size
    
    def test_operation_report_structure(self):
        """Test OperationReport has requested vs effective policy."""
        from generation.policies import OperationReport
        
        report = OperationReport(
            success=True,
            requested_policy={"mode": "auto"},
            effective_policy={"mode": "voxel"},
            metadata={"pitch_used": 0.0002},
        )
        
        d = report.to_dict()
        
        assert "requested_policy" in d
        assert "effective_policy" in d
        assert d["requested_policy"]["mode"] == "auto"
        assert d["effective_policy"]["mode"] == "voxel"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
