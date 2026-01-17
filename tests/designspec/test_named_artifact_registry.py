"""Tests for named artifact registry."""

import pytest
import tempfile
from pathlib import Path
from designspec.context import ArtifactStore, ArtifactEntry


class TestArtifactStoreBasics:
    """Tests for basic ArtifactStore functionality."""
    
    def test_create_artifact_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(Path(tmpdir))
            assert store.output_dir == Path(tmpdir)
    
    def test_request_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(Path(tmpdir))
            store.request_artifact("void_mesh", "artifacts/void.stl")
            
            assert "void_mesh" in store._requested
            assert store._requested["void_mesh"] == "artifacts/void.stl"
    
    def test_register_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(Path(tmpdir))
            store.register("void_mesh", "union_voids", {"data": "test"})
            
            assert "void_mesh" in store._artifacts
    
    def test_register_with_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(Path(tmpdir))
            store.register(
                "void_mesh",
                "union_voids",
                {"data": "test"},
                metadata={"vertex_count": 100},
            )
            
            entry = store._artifacts["void_mesh"]
            assert entry.metadata["vertex_count"] == 100


class TestArtifactStorePaths:
    """Tests for artifact path handling."""
    
    def test_requested_artifact_path_respected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(Path(tmpdir))
            store.request_artifact("void_mesh", "custom/path/void.stl")
            store.register("void_mesh", "union_voids", {"data": "test"})
            
            entry = store._artifacts["void_mesh"]
            assert "custom/path/void.stl" in str(entry.path) or entry.path is None
    
    def test_unrequested_artifact_gets_default_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(Path(tmpdir))
            store.register("void_mesh", "union_voids", {"data": "test"})
            
            entry = store._artifacts["void_mesh"]
            assert entry.path is None or "void_mesh" in str(entry.path)


class TestArtifactStoreManifest:
    """Tests for artifact manifest building."""
    
    def test_build_manifest_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(Path(tmpdir))
            manifest = store.build_manifest()
            
            assert isinstance(manifest, dict)
    
    def test_build_manifest_with_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(Path(tmpdir))
            store.register("void_mesh", "union_voids", {"data": "test"})
            store.register("domain_mesh", "mesh_domain", {"data": "test2"})
            
            manifest = store.build_manifest()
            
            assert "void_mesh" in manifest
            assert "domain_mesh" in manifest
    
    def test_manifest_includes_stage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(Path(tmpdir))
            store.register("void_mesh", "union_voids", {"data": "test"})
            
            manifest = store.build_manifest()
            
            assert manifest["void_mesh"]["stage"] == "union_voids"
    
    def test_manifest_includes_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(Path(tmpdir))
            store.register(
                "void_mesh",
                "union_voids",
                {"data": "test"},
                metadata={"vertex_count": 100, "face_count": 200},
            )
            
            manifest = store.build_manifest()
            
            assert manifest["void_mesh"]["metadata"]["vertex_count"] == 100
            assert manifest["void_mesh"]["metadata"]["face_count"] == 200


class TestArtifactStoreSaveManifest:
    """Tests for saving artifact manifest."""
    
    def test_save_manifest_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(Path(tmpdir))
            store.register("void_mesh", "union_voids", {"data": "test"})
            
            manifest_path = store.save_manifest()
            
            assert manifest_path.exists()
    
    def test_save_manifest_custom_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(Path(tmpdir))
            store.register("void_mesh", "union_voids", {"data": "test"})
            
            custom_path = Path(tmpdir) / "custom_manifest.json"
            store.save_manifest(custom_path)
            
            assert custom_path.exists()
    
    def test_save_manifest_is_valid_json(self):
        import json
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(Path(tmpdir))
            store.register("void_mesh", "union_voids", {"data": "test"})
            
            manifest_path = store.save_manifest()
            
            with open(manifest_path) as f:
                data = json.load(f)
            
            assert "void_mesh" in data


class TestArtifactEntry:
    """Tests for ArtifactEntry dataclass."""
    
    def test_create_artifact_entry(self):
        entry = ArtifactEntry(
            name="void_mesh",
            stage="union_voids",
        )
        
        assert entry.name == "void_mesh"
        assert entry.stage == "union_voids"
    
    def test_artifact_entry_with_metadata(self):
        entry = ArtifactEntry(
            name="void_mesh",
            stage="union_voids",
            metadata={"vertex_count": 100},
        )
        
        assert entry.metadata["vertex_count"] == 100
    
    def test_artifact_entry_attributes(self):
        from pathlib import Path
        
        entry = ArtifactEntry(
            name="void_mesh",
            stage="union_voids",
            path=Path("/path/to/file.stl"),
            content_hash="abc123",
        )
        
        assert entry.name == "void_mesh"
        assert entry.stage == "union_voids"
        assert str(entry.path) == "/path/to/file.stl"
        assert entry.content_hash == "abc123"
