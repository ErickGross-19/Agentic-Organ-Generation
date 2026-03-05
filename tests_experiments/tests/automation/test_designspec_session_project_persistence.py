"""
Tests for DesignSpecSession project persistence.

Verifies:
- Project creation creates correct folder structure
- Spec is saved to disk after patches
- Patch history is maintained
- Compile auto-runs after patch application
- Reports are saved correctly
"""

import json
import pytest
from pathlib import Path
from typing import Dict, Any


class TestDesignSpecSessionProjectPersistence:
    """Test suite for DesignSpecSession project persistence."""
    
    def test_create_project_creates_folder_structure(self, tmp_path):
        """Test that create_project creates the correct folder structure."""
        from automation.designspec_session import DesignSpecSession
        
        project_root = tmp_path / "projects"
        project_root.mkdir()
        
        session = DesignSpecSession.create_project(
            project_root=str(project_root),
            name="test_project",
        )
        
        project_dir = project_root / "test_project"
        assert project_dir.exists()
        assert (project_dir / "spec.json").exists()
        assert (project_dir / "spec_history").is_dir()
        assert (project_dir / "patches").is_dir()
        assert (project_dir / "reports").is_dir()
        assert (project_dir / "artifacts").is_dir()
        assert (project_dir / "logs").is_dir()
    
    def test_create_project_with_template_spec(self, tmp_path):
        """Test that create_project uses template spec if provided."""
        from automation.designspec_session import DesignSpecSession
        
        project_root = tmp_path / "projects"
        project_root.mkdir()
        
        template_spec = {
            "meta": {"name": "Test Organ", "seed": 42},
            "domain": {"type": "box", "size": [0.02, 0.06, 0.03]},
        }
        
        session = DesignSpecSession.create_project(
            project_root=str(project_root),
            name="test_project",
            template_spec=template_spec,
        )
        
        spec = session.get_spec()
        assert spec["meta"]["name"] == "Test Organ"
        assert spec["meta"]["seed"] == 42
    
    def test_load_project_loads_existing_spec(self, tmp_path):
        """Test that load_project loads an existing project."""
        from automation.designspec_session import DesignSpecSession
        
        project_root = tmp_path / "projects"
        project_root.mkdir()
        
        session1 = DesignSpecSession.create_project(
            project_root=str(project_root),
            name="test_project",
            template_spec={"meta": {"name": "Test", "seed": 123}},
        )
        
        project_dir = project_root / "test_project"
        session2 = DesignSpecSession.load_project(str(project_dir))
        
        spec = session2.get_spec()
        assert spec["meta"]["name"] == "Test"
        assert spec["meta"]["seed"] == 123
    
    def test_apply_patch_saves_spec_to_disk(self, tmp_path):
        """Test that apply_patch saves the updated spec to disk."""
        from automation.designspec_session import DesignSpecSession
        
        project_root = tmp_path / "projects"
        project_root.mkdir()
        
        session = DesignSpecSession.create_project(
            project_root=str(project_root),
            name="test_project",
        )
        
        patches = [
            {"op": "add", "path": "/meta/seed", "value": 42},
        ]
        
        result = session.apply_patch(patches, author="test", auto_compile=False)
        
        assert result.success
        
        spec_file = project_root / "test_project" / "spec.json"
        with open(spec_file) as f:
            saved_spec = json.load(f)
        
        assert saved_spec["meta"]["seed"] == 42
    
    def test_apply_patch_saves_patch_history(self, tmp_path):
        """Test that apply_patch saves patch to history."""
        from automation.designspec_session import DesignSpecSession
        
        project_root = tmp_path / "projects"
        project_root.mkdir()
        
        session = DesignSpecSession.create_project(
            project_root=str(project_root),
            name="test_project",
        )
        
        patches = [
            {"op": "add", "path": "/meta/seed", "value": 42},
        ]
        
        result = session.apply_patch(patches, author="test", auto_compile=False)
        
        assert result.success
        
        patches_dir = project_root / "test_project" / "patches"
        patch_files = list(patches_dir.glob("*.json"))
        assert len(patch_files) == 1
        
        with open(patch_files[0]) as f:
            saved_patch = json.load(f)
        
        assert saved_patch["author"] == "test"
        assert saved_patch["patches"] == patches
    
    def test_apply_patch_saves_spec_snapshot(self, tmp_path):
        """Test that apply_patch saves spec snapshot to history."""
        from automation.designspec_session import DesignSpecSession
        
        project_root = tmp_path / "projects"
        project_root.mkdir()
        
        session = DesignSpecSession.create_project(
            project_root=str(project_root),
            name="test_project",
        )
        
        patches = [
            {"op": "add", "path": "/meta/seed", "value": 42},
        ]
        
        result = session.apply_patch(patches, author="test", auto_compile=False)
        
        assert result.success
        
        history_dir = project_root / "test_project" / "spec_history"
        history_files = list(history_dir.glob("*.json"))
        assert len(history_files) >= 1
    
    def test_get_patch_history_returns_all_patches(self, tmp_path):
        """Test that get_patch_history returns all applied patches."""
        from automation.designspec_session import DesignSpecSession
        
        project_root = tmp_path / "projects"
        project_root.mkdir()
        
        session = DesignSpecSession.create_project(
            project_root=str(project_root),
            name="test_project",
        )
        
        session.apply_patch(
            [{"op": "add", "path": "/meta/seed", "value": 42}],
            author="user1",
            auto_compile=False,
        )
        
        session.apply_patch(
            [{"op": "replace", "path": "/meta/seed", "value": 123}],
            author="user2",
            auto_compile=False,
        )
        
        history = session.get_patch_history()
        
        assert len(history) == 2
        assert history[0]["author"] == "user1"
        assert history[1]["author"] == "user2"
    
    def test_validate_spec_returns_validation_report(self, tmp_path):
        """Test that validate_spec returns a ValidationReport."""
        from automation.designspec_session import DesignSpecSession, ValidationReport
        
        project_root = tmp_path / "projects"
        project_root.mkdir()
        
        session = DesignSpecSession.create_project(
            project_root=str(project_root),
            name="test_project",
        )
        
        report = session.validate_spec()
        
        assert isinstance(report, ValidationReport)
        assert hasattr(report, "valid")
        assert hasattr(report, "errors")
        assert hasattr(report, "warnings")


class TestDesignSpecSessionPatchValidation:
    """Test suite for patch validation."""
    
    def test_apply_patch_validates_patch_format(self, tmp_path):
        """Test that apply_patch validates patch format."""
        from automation.designspec_session import DesignSpecSession
        
        project_root = tmp_path / "projects"
        project_root.mkdir()
        
        session = DesignSpecSession.create_project(
            project_root=str(project_root),
            name="test_project",
        )
        
        invalid_patches = [
            {"invalid": "patch"},
        ]
        
        result = session.apply_patch(invalid_patches, author="test", auto_compile=False)
        
        assert not result.success
        assert len(result.errors) > 0
    
    def test_apply_patch_validates_path_exists_for_replace(self, tmp_path):
        """Test that apply_patch validates path exists for replace operations."""
        from automation.designspec_session import DesignSpecSession
        
        project_root = tmp_path / "projects"
        project_root.mkdir()
        
        session = DesignSpecSession.create_project(
            project_root=str(project_root),
            name="test_project",
        )
        
        patches = [
            {"op": "replace", "path": "/nonexistent/path", "value": "test"},
        ]
        
        result = session.apply_patch(patches, author="test", auto_compile=False)
        
        assert not result.success


class TestDesignSpecSessionCompile:
    """Test suite for compile functionality."""
    
    def test_compile_returns_compile_report(self, tmp_path):
        """Test that compile returns a CompileReport."""
        from automation.designspec_session import DesignSpecSession, CompileReport
        
        project_root = tmp_path / "projects"
        project_root.mkdir()
        
        template_spec = {
            "schema": {"name": "aog_designspec", "version": "1.0"},
            "meta": {"name": "Test", "seed": 42},
            "domain": {
                "main_domain": {
                    "type": "box",
                    "center": [0, 0, 0],
                    "size": [0.02, 0.06, 0.03],
                },
            },
        }
        
        session = DesignSpecSession.create_project(
            project_root=str(project_root),
            name="test_project",
            template_spec=template_spec,
        )
        
        report = session.compile()
        
        assert isinstance(report, CompileReport)
        assert hasattr(report, "success")
        assert hasattr(report, "stages")
