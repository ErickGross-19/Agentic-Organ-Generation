"""
Tests for bifurcating tree pipeline fixes.

These tests verify:
1. Port field names are correctly set (type vs port_type, id vs name)
2. Face projection is applied to explicit positions when clamp_to_face is set
3. Domain-aware tree scaling works correctly
4. Network artifacts are saved when requested
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np


class TestPortFieldNormalization:
    """Tests for port field name normalization in _collect_all_ports."""
    
    def test_port_type_field_is_set_correctly(self):
        """Verify ports have 'type' field set to 'inlet' or 'outlet'."""
        from designspec.runner import DesignSpecRunner
        from designspec.spec import DesignSpec
        
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"name": "test", "input_units": "mm"},
            "policies": {},
            "domains": {
                "main": {"type": "cylinder", "center": [0, 0, 0], "radius": 5.0, "height": 2.0}
            },
            "components": [{
                "id": "comp1",
                "domain_ref": "main",
                "ports": {
                    "inlets": [{"name": "inlet_1", "position": [0, 0, 1], "radius": 0.5}],
                    "outlets": [{"name": "outlet_1", "position": [0, 0, -1], "radius": 0.3}]
                },
                "build": {"type": "backend_network", "backend": "kary_tree"}
            }]
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = DesignSpecRunner(spec, output_dir=tmpdir)
            runner._resolved_ports["comp1"] = {
                "inlets": [{"name": "inlet_1", "position": [0, 0, 0.001], "radius": 0.0005}],
                "outlets": [{"name": "outlet_1", "position": [0, 0, -0.001], "radius": 0.0003}]
            }
            
            ports = runner._collect_all_ports()
            
            for port in ports:
                assert "type" in port, "Port should have 'type' field"
                assert port["type"] in ("inlet", "outlet"), f"Port type should be 'inlet' or 'outlet', got {port['type']}"
                assert "port_type" not in port or port.get("type") == port.get("port_type", port["type"]), \
                    "If port_type exists, it should match type"
    
    def test_port_id_preserves_name(self):
        """Verify port 'id' field preserves the original 'name' field."""
        from designspec.runner import DesignSpecRunner
        from designspec.spec import DesignSpec
        
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"name": "test", "input_units": "mm"},
            "policies": {},
            "domains": {
                "main": {"type": "cylinder", "center": [0, 0, 0], "radius": 5.0, "height": 2.0}
            },
            "components": [{
                "id": "comp1",
                "domain_ref": "main",
                "ports": {
                    "inlets": [{"name": "inlet_center", "position": [0, 0, 1], "radius": 0.5}],
                    "outlets": []
                },
                "build": {"type": "backend_network", "backend": "kary_tree"}
            }]
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = DesignSpecRunner(spec, output_dir=tmpdir)
            runner._resolved_ports["comp1"] = {
                "inlets": [{"name": "inlet_center", "position": [0, 0, 0.001], "radius": 0.0005}],
                "outlets": []
            }
            
            ports = runner._collect_all_ports()
            
            inlet_ports = [p for p in ports if p["type"] == "inlet"]
            assert len(inlet_ports) == 1
            assert inlet_ports[0].get("id") == "inlet_center", \
                f"Port id should be 'inlet_center', got {inlet_ports[0].get('id')}"


class TestFaceProjection:
    """Tests for face projection of explicit positions."""
    
    def test_project_position_to_face_top(self):
        """Test projecting a position onto the top face of a cylinder."""
        from generation.utils.port_placement import project_position_to_face
        from generation.core.domain import CylinderDomain
        from generation.core.types import Point3D
        
        domain = CylinderDomain(
            center=Point3D(0, 0, 0),
            radius=0.005,
            height=0.002
        )
        
        original_pos = (0, 0, 0.0005)
        projected_pos, direction, proj_dist = project_position_to_face(
            position=original_pos,
            face="top",
            domain=domain,
        )
        
        assert projected_pos[2] == pytest.approx(0.001, abs=1e-6), \
            f"Projected z should be at top face (0.001m), got {projected_pos[2]}"
        assert direction[2] == pytest.approx(-1.0, abs=1e-6), \
            f"Direction should point inward (-z), got {direction}"
        assert proj_dist == pytest.approx(0.0005, abs=1e-6), \
            f"Projection distance should be 0.5mm, got {proj_dist}"
    
    def test_project_position_to_face_bottom(self):
        """Test projecting a position onto the bottom face of a cylinder."""
        from generation.utils.port_placement import project_position_to_face
        from generation.core.domain import CylinderDomain
        from generation.core.types import Point3D
        
        domain = CylinderDomain(
            center=Point3D(0, 0, 0),
            radius=0.005,
            height=0.002
        )
        
        original_pos = (0, 0, -0.0005)
        projected_pos, direction, proj_dist = project_position_to_face(
            position=original_pos,
            face="bottom",
            domain=domain,
        )
        
        assert projected_pos[2] == pytest.approx(-0.001, abs=1e-6), \
            f"Projected z should be at bottom face (-0.001m), got {projected_pos[2]}"
        assert direction[2] == pytest.approx(1.0, abs=1e-6), \
            f"Direction should point inward (+z), got {direction}"
    
    def test_explicit_position_projected_when_clamp_to_face(self):
        """Test that explicit positions are projected when projection_mode is clamp_to_face."""
        from designspec.runner import DesignSpecRunner
        from designspec.spec import DesignSpec
        
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"name": "test", "input_units": "mm"},
            "policies": {
                "ports": {
                    "face": "top",
                    "projection_mode": "clamp_to_face"
                }
            },
            "domains": {
                "main": {"type": "cylinder", "center": [0, 0, 0], "radius": 5.0, "height": 2.0}
            },
            "components": [{
                "id": "comp1",
                "domain_ref": "main",
                "ports": {
                    "inlets": [{
                        "name": "inlet_center",
                        "position": [0, 0, 0.5],
                        "direction": [0, 0, -1],
                        "radius": 0.5
                    }],
                    "outlets": []
                },
                "build": {"type": "backend_network", "backend": "kary_tree"}
            }]
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = DesignSpecRunner(spec, output_dir=tmpdir)
            runner._stage_compile_policies()
            runner._stage_compile_domains()
            
            report = runner._stage_component_ports("comp1")
            
            assert report.success, f"Port resolution should succeed: {report.errors}"
            
            resolved = runner._resolved_ports.get("comp1", {})
            inlets = resolved.get("inlets", [])
            assert len(inlets) == 1
            
            inlet = inlets[0]
            assert inlet.get("resolution_method") == "explicit_projected", \
                f"Resolution method should be 'explicit_projected', got {inlet.get('resolution_method')}"
            
            assert inlet["position"][2] == pytest.approx(0.001, abs=1e-6), \
                f"Inlet z should be projected to top face (0.001m), got {inlet['position'][2]}"


class TestKaryTreeDomainScaling:
    """Tests for domain-aware kary tree scaling."""
    
    def test_domain_characteristic_size_cylinder(self):
        """Test characteristic size calculation for cylinder domain."""
        from generation.backends.kary_tree_backend import KaryTreeBackend
        from generation.core.domain import CylinderDomain
        from generation.core.types import Point3D
        
        backend = KaryTreeBackend()
        
        domain = CylinderDomain(
            center=Point3D(0, 0, 0),
            radius=0.005,
            height=0.002
        )
        
        size = backend._get_domain_characteristic_size(domain)
        
        assert size == pytest.approx(0.001, abs=1e-6), \
            f"Characteristic size should be min(radius, height/2) = 0.001m, got {size}"
    
    def test_domain_characteristic_size_box(self):
        """Test characteristic size calculation for box domain."""
        from generation.backends.kary_tree_backend import KaryTreeBackend
        from generation.core.domain import BoxDomain
        
        backend = KaryTreeBackend()
        
        domain = BoxDomain(
            x_min=-0.005, x_max=0.005,
            y_min=-0.003, y_max=0.003,
            z_min=-0.001, z_max=0.001
        )
        
        size = backend._get_domain_characteristic_size(domain)
        
        assert size == pytest.approx(0.001, abs=1e-6), \
            f"Characteristic size should be min(dx, dy, dz)/2 = 0.001m, got {size}"
    
    def test_branch_length_computed_from_domain_size(self):
        """Test that branch_length is computed from domain size when not explicitly set."""
        from generation.backends.kary_tree_backend import KaryTreeBackend, KaryTreeConfig
        from generation.core.domain import CylinderDomain
        from generation.core.types import Point3D
        
        domain = CylinderDomain(
            center=Point3D(0, 0, 0),
            radius=0.005,
            height=0.002
        )
        
        config = KaryTreeConfig(
            k=2,
            target_terminals=8,
            branch_length=None,
            tree_extent_fraction=0.4,
            use_domain_scaling=True,
        )
        
        backend = KaryTreeBackend()
        network = backend.generate(
            domain=domain,
            num_outlets=8,
            inlet_position=np.array([0, 0, 0.001]),
            inlet_radius=0.0005,
            config=config,
            rng_seed=42,
        )
        
        assert len(network.nodes) > 1, "Network should have multiple nodes"
        assert len(network.segments) > 0, "Network should have segments"
        
        positions = [np.array([n.position.x, n.position.y, n.position.z]) 
                     for n in network.nodes.values()]
        positions = np.array(positions)
        bbox_size = positions.max(axis=0) - positions.min(axis=0)
        
        domain_size = backend._get_domain_characteristic_size(domain)
        tree_extent = max(bbox_size)
        
        assert tree_extent > domain_size * 0.1, \
            f"Tree extent ({tree_extent}) should be meaningful relative to domain ({domain_size})"


class TestArtifactPersistence:
    """Tests for network artifact persistence."""
    
    def test_network_artifact_saved_when_requested(self):
        """Test that network artifacts are saved to disk when requested."""
        from designspec.context import ArtifactStore
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(tmpdir)
            
            store.request_artifact("comp1_network", "artifacts/comp1_network.json")
            
            mock_network = MagicMock()
            mock_network.to_dict.return_value = {
                "nodes": [{"id": "n1", "position": [0, 0, 0]}],
                "segments": []
            }
            
            store.register("comp1_network", "component_build:comp1", mock_network)
            
            saved_path = store.save_json("comp1_network", mock_network)
            
            assert saved_path is not None, "save_json should return a path"
            assert saved_path.exists(), f"Artifact file should exist at {saved_path}"
            
            with open(saved_path) as f:
                data = json.load(f)
            assert "nodes" in data, "Saved JSON should contain network data"
            
            entry = store.get("comp1_network")
            assert entry.saved is True, "Artifact entry should be marked as saved"
    
    def test_artifact_manifest_reflects_saved_status(self):
        """Test that artifact manifest correctly reports saved status."""
        from designspec.context import ArtifactStore
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(tmpdir)
            
            store.request_artifact("saved_artifact", "artifacts/saved.json")
            
            mock_data = MagicMock()
            mock_data.to_dict.return_value = {"key": "value"}
            
            store.register("saved_artifact", "test_stage", mock_data)
            store.save_json("saved_artifact", mock_data)
            
            store.register("unsaved_artifact", "test_stage", {"key": "value2"})
            
            manifest = store.build_manifest()
            
            assert manifest["saved_artifact"]["saved"] is True, \
                "Saved artifact should have saved=True in manifest"
            assert manifest["unsaved_artifact"]["saved"] is False, \
                "Unsaved artifact should have saved=False in manifest"
    
    def test_persist_saves_json_artifact(self):
        """Test that persist() saves JSON artifacts correctly."""
        from designspec.context import ArtifactStore
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(tmpdir)
            
            store.request_artifact("test_network", "artifacts/test_network.json")
            
            mock_network = MagicMock()
            mock_network.to_dict.return_value = {
                "nodes": [{"id": "n1", "position": [0, 0, 0]}],
                "segments": [{"id": "s1", "start": "n1", "end": "n2"}]
            }
            
            store.register("test_network", "component_build:test", mock_network)
            
            saved_path = store.persist("test_network", mock_network)
            
            assert saved_path.exists(), f"Artifact file should exist at {saved_path}"
            
            with open(saved_path) as f:
                data = json.load(f)
            assert "nodes" in data, "Saved JSON should contain network data"
            assert "segments" in data, "Saved JSON should contain segment data"
            
            entry = store.get("test_network")
            assert entry.saved is True, "Artifact entry should be marked as saved"
    
    def test_persist_raises_on_unregistered_artifact(self):
        """Test that persist() raises ArtifactSaveError for unregistered artifacts."""
        from designspec.context import ArtifactStore, ArtifactSaveError
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(tmpdir)
            
            with pytest.raises(ArtifactSaveError) as exc_info:
                store.persist("nonexistent_artifact", {"data": "test"})
            
            assert "nonexistent_artifact" in str(exc_info.value)
            assert "not registered" in str(exc_info.value)
    
    def test_persist_raises_on_unrequested_artifact(self):
        """Test that persist() raises ArtifactSaveError for unrequested artifacts."""
        from designspec.context import ArtifactStore, ArtifactSaveError
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(tmpdir)
            
            store.register("unrequested_artifact", "test_stage", {"data": "test"})
            
            with pytest.raises(ArtifactSaveError) as exc_info:
                store.persist("unrequested_artifact", {"data": "test"})
            
            assert "unrequested_artifact" in str(exc_info.value)
            assert "not requested" in str(exc_info.value)
    
    def test_persist_raises_on_serialization_failure(self):
        """Test that persist() raises ArtifactSaveError when serialization fails."""
        from designspec.context import ArtifactStore, ArtifactSaveError
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(tmpdir)
            
            store.request_artifact("bad_network", "artifacts/bad_network.json")
            
            mock_network = MagicMock()
            mock_network.to_dict.side_effect = ValueError("Serialization error")
            
            store.register("bad_network", "component_build:bad", mock_network)
            
            with pytest.raises(ArtifactSaveError) as exc_info:
                store.persist("bad_network", mock_network)
            
            assert "bad_network" in str(exc_info.value)
            assert "artifacts/bad_network.json" in str(exc_info.value)
    
    def test_is_requested_returns_correct_status(self):
        """Test that is_requested() correctly identifies requested artifacts."""
        from designspec.context import ArtifactStore
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(tmpdir)
            
            store.request_artifact("requested_artifact", "artifacts/requested.json")
            
            assert store.is_requested("requested_artifact") is True
            assert store.is_requested("unrequested_artifact") is False


class TestBifurcatingTreeSpecIntegration:
    """Integration tests using the actual bifurcating tree spec."""
    
    @pytest.fixture
    def bifurcating_tree_spec_path(self):
        """Path to the bifurcating tree example spec."""
        return Path(__file__).parent.parent.parent.parent / "examples" / "designspec" / "malaria_venule_bifurcating_tree.json"
    
    def test_spec_loads_and_normalizes(self, bifurcating_tree_spec_path):
        """Test that the bifurcating tree spec loads and normalizes correctly."""
        if not bifurcating_tree_spec_path.exists():
            pytest.skip(f"Spec file not found: {bifurcating_tree_spec_path}")
        
        from designspec.spec import DesignSpec
        
        spec = DesignSpec.from_json(bifurcating_tree_spec_path)
        
        assert spec.meta["name"] == "malaria_venule_bifurcating_tree"
        assert spec.meta["input_units"] == "m", \
            "After normalization, input_units should be 'm' (meters)"
        
        domain = spec.domains.get("cylinder_domain", {})
        assert domain["radius"] == pytest.approx(0.005, abs=1e-6), \
            f"Domain radius should be normalized to 0.005m, got {domain['radius']}"
        assert domain["height"] == pytest.approx(0.002, abs=1e-6), \
            f"Domain height should be normalized to 0.002m, got {domain['height']}"
    
    def test_port_policy_specifies_clamp_to_face(self, bifurcating_tree_spec_path):
        """Test that the spec's port policy specifies clamp_to_face projection."""
        if not bifurcating_tree_spec_path.exists():
            pytest.skip(f"Spec file not found: {bifurcating_tree_spec_path}")
        
        from designspec.spec import DesignSpec
        
        spec = DesignSpec.from_json(bifurcating_tree_spec_path)
        
        ports_policy = spec.policies.get("ports", {})
        assert ports_policy.get("face") == "top", \
            f"Port policy should specify face='top', got {ports_policy.get('face')}"
        assert ports_policy.get("projection_mode") == "clamp_to_face", \
            f"Port policy should specify projection_mode='clamp_to_face', got {ports_policy.get('projection_mode')}"
    
    def test_component_requests_network_artifact(self, bifurcating_tree_spec_path):
        """Test that the component requests a network artifact."""
        if not bifurcating_tree_spec_path.exists():
            pytest.skip(f"Spec file not found: {bifurcating_tree_spec_path}")
        
        from designspec.spec import DesignSpec
        
        spec = DesignSpec.from_json(bifurcating_tree_spec_path)
        
        component = spec.components[0]
        save_artifacts = component.get("save_artifacts", {})
        
        assert "network" in save_artifacts, \
            "Component should request network artifact to be saved"
        assert save_artifacts["network"] == "artifacts/bifurcating_tree_network.json", \
            f"Network artifact path should be 'artifacts/bifurcating_tree_network.json', got {save_artifacts['network']}"


class TestSurfaceOpeningSemantics:
    """Tests for surface opening semantics in validity checks."""
    
    def test_surface_opening_port_class(self):
        """Test SurfaceOpeningPort class correctly identifies points in neighborhood."""
        from generation.ops.validity.void_checks import SurfaceOpeningPort
        
        port = SurfaceOpeningPort(
            position=np.array([0.0, 0.0, 0.001]),
            direction=np.array([0.0, 0.0, -1.0]),
            radius=0.0005,
            tolerance=0.001,
            port_id="test_port",
        )
        
        # Point at port center should be in neighborhood
        assert port.is_point_in_neighborhood(np.array([0.0, 0.0, 0.001])) is True
        
        # Point slightly inside port (along direction) should be in neighborhood
        assert port.is_point_in_neighborhood(np.array([0.0, 0.0, 0.0005])) is True
        
        # Point far from port should not be in neighborhood
        assert port.is_point_in_neighborhood(np.array([0.0, 0.0, -0.005])) is False
        
        # Point outside port radius should not be in neighborhood
        assert port.is_point_in_neighborhood(np.array([0.005, 0.0, 0.001])) is False
    
    def test_validation_policy_surface_opening_fields(self):
        """Test ValidationPolicy has surface opening fields."""
        from aog_policies.validity import ValidationPolicy
        
        policy = ValidationPolicy()
        
        assert hasattr(policy, 'allow_boundary_intersections_at_ports')
        assert hasattr(policy, 'surface_opening_tolerance')
        assert policy.allow_boundary_intersections_at_ports is False  # Default
        assert policy.surface_opening_tolerance == 0.001  # 1mm default
    
    def test_is_surface_opening_field_preserved_in_ports(self):
        """Test that is_surface_opening field is preserved when collecting ports."""
        from designspec.runner import DesignSpecRunner
        from designspec.spec import DesignSpec
        
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"name": "test", "input_units": "mm"},
            "policies": {},
            "domains": {
                "main": {"type": "cylinder", "center": [0, 0, 0], "radius": 5.0, "height": 2.0}
            },
            "components": [{
                "id": "comp1",
                "domain_ref": "main",
                "ports": {
                    "inlets": [{
                        "name": "inlet_1",
                        "position": [0, 0, 1],
                        "radius": 0.5,
                        "is_surface_opening": True
                    }],
                    "outlets": [{"name": "outlet_1", "position": [0, 0, -1], "radius": 0.3}]
                },
                "build": {"type": "backend_network", "backend": "kary_tree"}
            }]
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = DesignSpecRunner(spec, output_dir=tmpdir)
            runner._resolved_ports["comp1"] = {
                "inlets": [{
                    "name": "inlet_1",
                    "position": [0, 0, 0.001],
                    "radius": 0.0005,
                    "is_surface_opening": True
                }],
                "outlets": [{"name": "outlet_1", "position": [0, 0, -0.001], "radius": 0.0003}]
            }
            
            ports = runner._collect_all_ports()
            
            inlet_ports = [p for p in ports if p["type"] == "inlet"]
            assert len(inlet_ports) == 1
            assert inlet_ports[0].get("is_surface_opening") is True, \
                "is_surface_opening field should be preserved"


class TestUnionDetailPreservation:
    """Tests for union detail preservation during merge operations."""
    
    def test_mesh_merge_policy_detail_fields(self):
        """Test MeshMergePolicy has detail preservation fields."""
        from aog_policies.generation import MeshMergePolicy
        
        policy = MeshMergePolicy()
        
        assert hasattr(policy, 'min_voxels_per_diameter')
        assert hasattr(policy, 'min_channel_diameter')
        assert hasattr(policy, 'detail_loss_threshold')
        assert hasattr(policy, 'detail_loss_strictness')
        
        assert policy.min_voxels_per_diameter == 4  # Default
        assert policy.detail_loss_threshold == 0.5  # 50% default
        assert policy.detail_loss_strictness == "warn"  # Default
    
    def test_detect_detail_loss_no_loss(self):
        """Test detect_detail_loss returns False when no significant loss."""
        from generation.ops.mesh.merge import detect_detail_loss
        from aog_policies.generation import MeshMergePolicy
        
        # Create mock meshes with similar properties
        class MockMesh:
            def __init__(self, faces, vertices, volume):
                self.faces = np.zeros((faces, 3), dtype=int)
                self.vertices = np.zeros((vertices, 3))
                self._volume = volume
            
            @property
            def volume(self):
                return self._volume
        
        input_meshes = [
            MockMesh(faces=1000, vertices=500, volume=1e-9),
            MockMesh(faces=1000, vertices=500, volume=1e-9),
        ]
        output_mesh = MockMesh(faces=1800, vertices=900, volume=1.8e-9)
        
        policy = MeshMergePolicy(detail_loss_threshold=0.5)
        
        detail_lost, report = detect_detail_loss(input_meshes, output_mesh, policy)
        
        assert detail_lost is False, "Should not detect detail loss for similar meshes"
        assert report["volume_ratio"] > 0.5, "Volume ratio should be > 0.5"
    
    def test_detect_detail_loss_significant_loss(self):
        """Test detect_detail_loss returns True when significant loss detected."""
        from generation.ops.mesh.merge import detect_detail_loss
        from aog_policies.generation import MeshMergePolicy
        
        class MockMesh:
            def __init__(self, faces, vertices, volume):
                self.faces = np.zeros((faces, 3), dtype=int)
                self.vertices = np.zeros((vertices, 3))
                self._volume = volume
            
            @property
            def volume(self):
                return self._volume
        
        # Input has 100k faces, output has 1k faces (100x collapse)
        input_meshes = [
            MockMesh(faces=50000, vertices=25000, volume=1e-9),
            MockMesh(faces=50000, vertices=25000, volume=1e-9),
        ]
        output_mesh = MockMesh(faces=1000, vertices=500, volume=0.2e-9)
        
        policy = MeshMergePolicy(detail_loss_threshold=0.5)
        
        detail_lost, report = detect_detail_loss(input_meshes, output_mesh, policy)
        
        assert detail_lost is True, "Should detect detail loss for collapsed mesh"
        assert report["face_ratio"] < 0.1, "Face ratio should indicate collapse"
    
    def test_compute_pitch_from_diameter(self):
        """Test pitch computation from minimum channel diameter."""
        from generation.ops.mesh.merge import compute_pitch_from_diameter
        
        # 100um diameter with 4 voxels per diameter = 25um pitch
        pitch = compute_pitch_from_diameter(
            min_channel_diameter=100e-6,
            min_voxels_per_diameter=4,
        )
        
        assert pitch == pytest.approx(25e-6, abs=1e-9), \
            f"Pitch should be 25um, got {pitch*1e6}um"


class TestPreflightChecks:
    """Tests for preflight validation checks."""
    
    def test_surface_opening_port_on_boundary_passes(self):
        """Test that surface opening port on boundary passes preflight check."""
        from validity.pre_embedding.preflight_checks import check_surface_opening_port_placement
        
        # Port at top of domain (z=0.001)
        result = check_surface_opening_port_placement(
            port_position=np.array([0.0, 0.0, 0.001]),
            port_direction=np.array([0.0, 0.0, -1.0]),
            port_radius=0.0005,
            domain_bounds=(np.array([-0.005, -0.005, -0.001]), np.array([0.005, 0.005, 0.001])),
            port_id="test_inlet",
            tolerance=0.001,
        )
        
        assert result.passed is True, f"Port on boundary should pass: {result.message}"
    
    def test_surface_opening_port_far_from_boundary_fails(self):
        """Test that surface opening port far from boundary fails preflight check."""
        from validity.pre_embedding.preflight_checks import check_surface_opening_port_placement
        
        # Port in center of domain (far from boundary)
        result = check_surface_opening_port_placement(
            port_position=np.array([0.0, 0.0, 0.0]),
            port_direction=np.array([0.0, 0.0, -1.0]),
            port_radius=0.0005,
            domain_bounds=(np.array([-0.005, -0.005, -0.001]), np.array([0.005, 0.005, 0.001])),
            port_id="test_inlet",
            tolerance=0.0001,  # Tight tolerance
        )
        
        assert result.passed is False, f"Port far from boundary should fail: {result.message}"
        assert "test_inlet" in result.message, "Error message should include port ID"
    
    def test_union_pitch_too_coarse_warns(self):
        """Test that union pitch too coarse for channel diameter triggers warning."""
        from validity.pre_embedding.preflight_checks import check_union_pitch_sensibility
        
        # 50um pitch for 100um diameter = only 2 voxels (need 4)
        result = check_union_pitch_sensibility(
            voxel_pitch=50e-6,
            min_channel_diameter=100e-6,
            domain_size=np.array([0.01, 0.01, 0.002]),
            min_voxels_per_diameter=4,
        )
        
        assert result.passed is False, f"Coarse pitch should fail: {result.message}"
        assert result.severity == "warning", "Should be a warning, not error"
    
    def test_union_pitch_sensible_passes(self):
        """Test that sensible union pitch passes preflight check."""
        from validity.pre_embedding.preflight_checks import check_union_pitch_sensibility
        
        # 25um pitch for 100um diameter = 4 voxels (meets requirement)
        result = check_union_pitch_sensibility(
            voxel_pitch=25e-6,
            min_channel_diameter=100e-6,
            domain_size=np.array([0.01, 0.01, 0.002]),
            min_voxels_per_diameter=4,
        )
        
        assert result.passed is True, f"Sensible pitch should pass: {result.message}"
    
    def test_port_radius_too_large_warns(self):
        """Test that port radius too large relative to domain triggers warning."""
        from validity.pre_embedding.preflight_checks import check_port_radius_compatibility
        
        # Port radius is 30% of smallest domain dimension
        result = check_port_radius_compatibility(
            port_radius=0.0003,
            wall_thickness=None,
            domain_size=np.array([0.01, 0.01, 0.001]),  # z is smallest at 1mm
            port_id="large_port",
        )
        
        assert result.passed is False, f"Large port should fail: {result.message}"
        assert "large_port" in result.message, "Error message should include port ID"
    
    def test_network_bbox_coverage_low_warns(self):
        """Test that low network bbox coverage triggers warning."""
        from validity.pre_embedding.preflight_checks import check_network_bbox_coverage
        
        # Network is tiny compared to domain
        result = check_network_bbox_coverage(
            network_bbox=(np.array([0.0, 0.0, 0.0]), np.array([0.001, 0.001, 0.001])),
            domain_bbox=(np.array([-0.005, -0.005, -0.001]), np.array([0.005, 0.005, 0.001])),
            min_coverage_ratio=0.1,
        )
        
        assert result.passed is False, f"Low coverage should fail: {result.message}"
        assert result.severity == "warning", "Should be a warning"
    
    def test_run_preflight_checks_aggregates_results(self):
        """Test that run_preflight_checks aggregates all check results."""
        from validity.pre_embedding.preflight_checks import run_preflight_checks
        
        ports = [
            {
                "id": "inlet_1",
                "position": [0.0, 0.0, 0.001],
                "direction": [0.0, 0.0, -1.0],
                "radius": 0.0005,
                "is_surface_opening": True,
            }
        ]
        
        report = run_preflight_checks(
            ports=ports,
            domain_bounds=(np.array([-0.005, -0.005, -0.001]), np.array([0.005, 0.005, 0.001])),
            voxel_pitch=25e-6,
            min_channel_diameter=100e-6,
        )
        
        assert len(report.results) > 0, "Report should have results"
        assert report.passed is True, "All checks should pass for valid config"
    
    def test_preflight_report_to_dict(self):
        """Test that PreflightReport.to_dict() returns correct structure."""
        from validity.pre_embedding.preflight_checks import PreflightReport, PreflightResult
        
        report = PreflightReport()
        report.add(PreflightResult(
            check_name="test_check",
            passed=True,
            message="Test passed",
            details={"key": "value"},
            severity="info",
        ))
        
        result_dict = report.to_dict()
        
        assert "passed" in result_dict
        assert "has_warnings" in result_dict
        assert "results" in result_dict
        assert len(result_dict["results"]) == 1
        assert result_dict["results"][0]["check_name"] == "test_check"


class TestUnitNormalization:
    """Tests for unit normalization of new fields."""
    
    def test_surface_opening_tolerance_normalized(self):
        """Test that surface_opening_tolerance is normalized from mm to meters."""
        from designspec.spec import DesignSpec
        
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"name": "test", "input_units": "mm"},
            "policies": {
                "validity": {
                    "surface_opening_tolerance": 1.0  # 1mm
                }
            },
            "domains": {
                "main": {"type": "cylinder", "center": [0, 0, 0], "radius": 5.0, "height": 2.0}
            },
            "components": []
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        
        validity_policy = spec.policies.get("validity", {})
        assert validity_policy.get("surface_opening_tolerance") == pytest.approx(0.001, abs=1e-9), \
            f"surface_opening_tolerance should be normalized to 0.001m, got {validity_policy.get('surface_opening_tolerance')}"
    
    def test_min_channel_diameter_normalized(self):
        """Test that min_channel_diameter in mesh_merge is normalized from mm to meters."""
        from designspec.spec import DesignSpec
        
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"name": "test", "input_units": "mm"},
            "policies": {
                "mesh_merge": {
                    "min_channel_diameter": 0.1  # 0.1mm = 100um
                }
            },
            "domains": {
                "main": {"type": "cylinder", "center": [0, 0, 0], "radius": 5.0, "height": 2.0}
            },
            "components": []
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        
        mesh_merge_policy = spec.policies.get("mesh_merge", {})
        assert mesh_merge_policy.get("min_channel_diameter") == pytest.approx(0.0001, abs=1e-9), \
            f"min_channel_diameter should be normalized to 0.0001m, got {mesh_merge_policy.get('min_channel_diameter')}"
