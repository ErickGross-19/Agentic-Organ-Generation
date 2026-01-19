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
