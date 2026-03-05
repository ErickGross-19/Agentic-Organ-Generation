"""
Tests for DesignSpec LLM Context Builder.

Tests cover:
1. Compact pack contains required fields
2. Can locate last run and summarize it
3. Context building with various artifact states
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime

import pytest

from automation.designspec_llm.context_builder import (
    ContextBuilder,
    ContextPack,
    SpecSummary,
    RunSummary,
    ValidationSummary,
    PatchHistoryEntry,
)


class TestSpecSummary:
    """Tests for SpecSummary dataclass."""
    
    def test_spec_summary_creation(self):
        summary = SpecSummary(
            has_domains=True,
            domain_count=2,
            domain_names=["main", "secondary"],
            has_components=True,
            component_count=3,
            component_ids=["inlet", "outlet", "network"],
            has_policies=True,
            policy_names=["mesh_policy", "validity_policy"],
            meta_seed=42,
        )
        
        assert summary.has_domains is True
        assert summary.domain_count == 2
        assert "main" in summary.domain_names
        assert summary.component_count == 3
    
    def test_spec_summary_to_dict(self):
        summary = SpecSummary(
            has_domains=True,
            domain_count=1,
            domain_names=["main"],
        )
        
        d = summary.to_dict()
        assert d["domains"]["present"] is True
        assert d["domains"]["count"] == 1
        assert "main" in d["domains"]["names"]
    
    def test_spec_summary_defaults(self):
        summary = SpecSummary()
        
        assert summary.has_domains is False
        assert summary.domain_count == 0
        assert summary.has_components is False
        assert summary.has_policies is False


class TestRunSummary:
    """Tests for RunSummary dataclass."""
    
    def test_run_summary_creation(self):
        summary = RunSummary(
            run_id="run_20240101_120000",
            timestamp="2024-01-01T12:00:00",
            success=True,
            stages_completed=["component_build", "component_mesh"],
            stages_failed=[],
            errors=[],
        )
        
        assert summary.run_id == "run_20240101_120000"
        assert summary.success is True
        assert len(summary.stages_completed) == 2
    
    def test_run_summary_to_dict(self):
        summary = RunSummary(
            run_id="test_run",
            timestamp="2024-01-01T12:00:00",
            success=False,
            stages_completed=["component_build"],
            stages_failed=["component_mesh"],
            errors=["Mesh generation failed"],
        )
        
        d = summary.to_dict()
        assert d["run_id"] == "test_run"
        assert d["success"] is False
        assert "component_mesh" in d["stages_failed"]
        assert "Mesh generation failed" in d["errors"]
    
    def test_run_summary_with_mesh_stats(self):
        summary = RunSummary(
            run_id="test",
            success=True,
            mesh_faces=1000,
            mesh_vertices=500,
            mesh_watertight=True,
        )
        
        d = summary.to_dict()
        assert "mesh_stats" in d
        assert d["mesh_stats"]["faces"] == 1000


class TestValidationSummary:
    """Tests for ValidationSummary dataclass."""
    
    def test_validation_summary_creation(self):
        summary = ValidationSummary(
            valid=True,
            error_count=0,
            warning_count=0,
            errors=[],
            warnings=[],
        )
        
        assert summary.valid is True
        assert summary.error_count == 0
    
    def test_validation_summary_with_failures(self):
        summary = ValidationSummary(
            valid=False,
            error_count=2,
            warning_count=1,
            errors=["watertight failed", "port_connectivity failed"],
            warnings=["Low face count"],
        )
        
        assert summary.valid is False
        assert "watertight failed" in summary.errors
        assert len(summary.warnings) == 1


class TestPatchHistoryEntry:
    """Tests for PatchHistoryEntry dataclass."""
    
    def test_patch_history_entry_creation(self):
        entry = PatchHistoryEntry(
            patch_id="patch_001",
            timestamp="2024-01-01T12:00:00",
            author="user",
            operation_count=1,
            paths_modified=["/domains/main"],
        )
        
        assert entry.patch_id == "patch_001"
        assert entry.operation_count == 1
        assert len(entry.paths_modified) == 1
    
    def test_patch_history_entry_to_dict(self):
        entry = PatchHistoryEntry(
            patch_id="patch_001",
            timestamp="2024-01-01T12:00:00",
            author="user",
            operation_count=2,
            paths_modified=["/domains/main", "/components/inlet"],
        )
        
        d = entry.to_dict()
        assert d["patch_id"] == "patch_001"
        assert d["operation_count"] == 2


class TestContextPack:
    """Tests for ContextPack dataclass."""
    
    def test_context_pack_creation(self):
        pack = ContextPack(
            spec_summary=SpecSummary(has_domains=True, domain_count=1),
            last_run=None,
            recent_runs=[],
            validation_summary=None,
            recent_patches=[],
            compile_success=True,
        )
        
        assert pack.spec_summary.has_domains is True
        assert pack.last_run is None
        assert pack.compile_success is True
    
    def test_context_pack_to_dict(self):
        pack = ContextPack(
            spec_summary=SpecSummary(has_domains=True, domain_count=1, domain_names=["main"]),
            last_run=RunSummary(
                run_id="test",
                timestamp="2024-01-01",
                success=True,
                stages_completed=["full"],
            ),
            recent_runs=[],
            validation_summary=ValidationSummary(valid=True, error_count=0),
            recent_patches=[],
            compile_success=True,
        )
        
        d = pack.to_dict()
        assert "spec_summary" in d
        assert "last_run" in d
        assert d["last_run"]["success"] is True
    
    def test_context_pack_to_prompt_text(self):
        pack = ContextPack(
            spec_summary=SpecSummary(
                has_domains=True,
                domain_count=1,
                domain_names=["main"],
                has_components=True,
                component_count=2,
                component_ids=["inlet", "outlet"],
            ),
            compile_success=True,
        )
        
        text = pack.to_prompt_text()
        
        assert "Spec Summary" in text
        assert "Domains" in text
        assert "Components" in text


class TestContextBuilder:
    """Tests for ContextBuilder class."""
    
    def test_context_builder_creation(self):
        mock_session = MagicMock()
        mock_session.get_spec.return_value = {"meta": {"seed": 42}}
        mock_session.project_dir = Path("/tmp/test_project")
        
        builder = ContextBuilder(session=mock_session)
        
        assert builder.session == mock_session
    
    def test_build_run_summary(self):
        mock_session = MagicMock()
        mock_session.get_spec.return_value = {}
        mock_session.project_dir = Path("/tmp/test_project")
        
        run_result = {
            "run_id": "test_run",
            "timestamp": "2024-01-01T12:00:00",
            "success": True,
            "errors": [],
            "warnings": [],
            "output_dir": "/tmp/output",
            "stage_reports": [
                {"stage": "component_build", "success": True},
                {"stage": "component_mesh", "success": True},
            ],
            "mesh_stats": {
                "faces": 1000,
                "vertices": 500,
                "watertight": True,
            },
        }
        
        builder = ContextBuilder(session=mock_session)
        summary = builder.build_run_summary(run_result)
        
        assert summary.run_id == "test_run"
        assert summary.success is True
        assert "component_build" in summary.stages_completed
        assert summary.mesh_faces == 1000
    
    def test_get_compile_state_compiled(self):
        mock_session = MagicMock()
        mock_session.get_spec.return_value = {}
        mock_session.project_dir = Path("/tmp/test_project")
        mock_session.get_last_compile_report.return_value = {
            "success": True,
            "errors": [],
        }
        
        builder = ContextBuilder(session=mock_session)
        success, errors = builder.get_compile_state()
        
        assert success is True
    
    def test_get_compile_state_failed(self):
        mock_session = MagicMock()
        mock_session.get_spec.return_value = {}
        mock_session.project_dir = Path("/tmp/test_project")
        mock_session.get_last_compile_report.return_value = {
            "success": False,
            "errors": ["Invalid domain"],
        }
        
        builder = ContextBuilder(session=mock_session)
        success, errors = builder.get_compile_state()
        
        assert success is False
        assert "Invalid domain" in errors
    
    def test_get_compile_state_no_report(self):
        mock_session = MagicMock()
        mock_session.get_spec.return_value = {}
        mock_session.project_dir = Path("/tmp/test_project")
        mock_session.get_last_compile_report.return_value = None
        
        builder = ContextBuilder(session=mock_session)
        success, errors = builder.get_compile_state()
        
        assert success is False
        assert len(errors) == 1  # "No compile report available"


class TestContextBuilderWithFileSystem:
    """Tests for ContextBuilder with actual file system operations."""
    
    def test_get_recent_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_dir = Path(tmpdir) / "artifacts"
            
            for i in range(5):
                run_dir = artifacts_dir / f"run_2024010{i}_120000"
                run_dir.mkdir(parents=True)
                
                run_report = {
                    "run_id": f"run_2024010{i}_120000",
                    "timestamp": f"2024-01-0{i}T12:00:00",
                    "success": i % 2 == 0,
                    "stages": {},
                }
                
                with open(run_dir / "run_report.json", "w") as f:
                    json.dump(run_report, f)
            
            mock_session = MagicMock()
            mock_session.get_spec.return_value = {}
            mock_session.project_dir = Path(tmpdir)
            mock_session.get_last_runner_result.return_value = None
            
            builder = ContextBuilder(session=mock_session)
            runs = builder.get_recent_runs(limit=3)
            
            assert len(runs) == 3
    
    def test_get_last_successful_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_dir = Path(tmpdir) / "artifacts"
            
            run1_dir = artifacts_dir / "run_20240101_120000"
            run1_dir.mkdir(parents=True)
            with open(run1_dir / "run_report.json", "w") as f:
                json.dump({"run_id": "run_20240101_120000", "success": True, "stages": {}}, f)
            
            run2_dir = artifacts_dir / "run_20240102_120000"
            run2_dir.mkdir(parents=True)
            with open(run2_dir / "run_report.json", "w") as f:
                json.dump({"run_id": "run_20240102_120000", "success": False, "stages": {}}, f)
            
            mock_session = MagicMock()
            mock_session.get_spec.return_value = {}
            mock_session.project_dir = Path(tmpdir)
            mock_session.get_last_runner_result.return_value = None
            
            builder = ContextBuilder(session=mock_session)
            successful = builder.get_last_successful_run()
            
            assert successful is not None
            assert successful.success is True
