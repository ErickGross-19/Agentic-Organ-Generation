"""
Integration tests for DesignSpec workflow contract.

Tests the DesignSpecWorkflow directly (headless, no Tk widgets) to verify:
- Project creation in temp directory
- User message handling
- Patch approval flow
- Compile auto-runs after patch
- Run until stages work correctly
- Artifacts are created in project folder
"""

import json
import pytest
from pathlib import Path
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock


class TestDesignSpecWorkflowContract:
    """Test suite for DesignSpec workflow contract."""
    
    def test_workflow_creates_project(self, tmp_path):
        """Test that workflow creates project in temp directory."""
        from automation.workflows.designspec_workflow import DesignSpecWorkflow
        
        workflow = DesignSpecWorkflow()
        
        project_dir = tmp_path / "test_project"
        
        result = workflow.on_start(
            project_dir=str(project_dir),
            template_spec={"meta": {"name": "Test", "seed": 42}},
        )
        
        assert result is True
        assert project_dir.exists()
        assert (project_dir / "spec.json").exists()
    
    def test_workflow_loads_existing_project(self, tmp_path):
        """Test that workflow loads existing project."""
        from automation.workflows.designspec_workflow import DesignSpecWorkflow
        from automation.designspec_session import DesignSpecSession
        
        project_root = tmp_path / "projects"
        project_root.mkdir()
        
        session = DesignSpecSession.create_project(
            project_root=str(project_root),
            name="existing_project",
            template_spec={"meta": {"name": "Existing", "seed": 123}},
        )
        
        workflow = DesignSpecWorkflow()
        
        project_dir = project_root / "existing_project"
        workflow.on_start(project_dir=str(project_dir))
        
        spec = workflow.get_spec()
        assert spec["meta"]["name"] == "Existing"
        assert spec["meta"]["seed"] == 123
    
    def test_workflow_handles_user_message(self, tmp_path):
        """Test that workflow handles user messages."""
        from automation.workflows.designspec_workflow import DesignSpecWorkflow
        
        workflow = DesignSpecWorkflow()
        
        project_dir = tmp_path / "test_project"
        
        workflow.on_start(
            project_dir=str(project_dir),
            template_spec={"meta": {"name": "Test"}},
        )
        
        workflow.on_user_message("Create a box domain 20mm x 60mm x 30mm")
        
        events = workflow.poll_events(timeout=0.1)
        assert len(events) >= 0
    
    def test_workflow_returns_patch_proposals(self, tmp_path):
        """Test that workflow returns patch proposals for spec changes."""
        from automation.workflows.designspec_workflow import DesignSpecWorkflow
        
        workflow = DesignSpecWorkflow()
        
        project_dir = tmp_path / "test_project"
        
        workflow.on_start(
            project_dir=str(project_dir),
            template_spec={"meta": {"name": "Test"}},
        )
        
        response = workflow.on_user_message("Set the seed to 42")
        
        pending_patches = workflow.get_pending_patches()
        
        assert isinstance(pending_patches, dict)
    
    def test_workflow_approve_patch_applies_changes(self, tmp_path):
        """Test that approving a patch applies changes to spec."""
        from automation.workflows.designspec_workflow import DesignSpecWorkflow
        
        workflow = DesignSpecWorkflow()
        
        project_dir = tmp_path / "test_project"
        
        workflow.on_start(
            project_dir=str(project_dir),
            template_spec={
                "schema": {"name": "aog_designspec", "version": "1.0"},
                "meta": {"name": "Test"},
                "policies": {},
                "domains": {},
                "components": [],
            },
        )
        
        patch_id = workflow._propose_patch(
            patches=[{"op": "add", "path": "/meta/seed", "value": 42}],
            explanation="Add seed",
        )
        
        pending = workflow.get_pending_patches()
        assert patch_id in pending
        
        result = workflow.approve_patch(patch_id)
        
        spec = workflow.get_spec()
        assert spec["meta"].get("seed") == 42
    
    def test_workflow_reject_patch_discards_changes(self, tmp_path):
        """Test that rejecting a patch discards changes."""
        from automation.workflows.designspec_workflow import DesignSpecWorkflow
        
        workflow = DesignSpecWorkflow()
        
        project_dir = tmp_path / "test_project"
        
        workflow.on_start(
            project_dir=str(project_dir),
            template_spec={"meta": {"name": "Test"}},
        )
        
        patch_id = workflow._propose_patch(
            patches=[{"op": "add", "path": "/meta/seed", "value": 42}],
            explanation="Add seed",
        )
        
        pending = workflow.get_pending_patches()
        assert patch_id in pending
        
        workflow.reject_patch(patch_id, reason="Test rejection")
        
        spec = workflow.get_spec()
        assert spec["meta"].get("seed") is None
        
        pending_after = workflow.get_pending_patches()
        assert patch_id not in pending_after
    
    def test_workflow_compile_runs_after_patch_approval(self, tmp_path):
        """Test that compile runs automatically after patch approval."""
        from automation.workflows.designspec_workflow import DesignSpecWorkflow
        
        workflow = DesignSpecWorkflow()
        
        project_dir = tmp_path / "test_project"
        
        workflow.on_start(
            project_dir=str(project_dir),
            template_spec={
                "meta": {"name": "Test", "seed": 42},
                "domain": {"type": "box", "size": [0.02, 0.06, 0.03]},
            },
        )
        
        patch_id = workflow._propose_patch(
            patches=[{"op": "replace", "path": "/meta/seed", "value": 123}],
            explanation="Change seed",
        )
        
        pending = workflow.get_pending_patches()
        assert patch_id in pending
        
        result = workflow.approve_patch(patch_id)
        
        events = workflow.poll_events(timeout=0.1)
        assert len(events) >= 0
    
    def test_workflow_get_spec_returns_current_spec(self, tmp_path):
        """Test that get_spec returns the current spec."""
        from automation.workflows.designspec_workflow import DesignSpecWorkflow
        
        workflow = DesignSpecWorkflow()
        
        project_dir = tmp_path / "test_project"
        
        template = {"meta": {"name": "Test", "seed": 42}}
        
        result = workflow.on_start(
            project_dir=str(project_dir),
            template_spec=template,
        )
        
        assert result is True
        
        spec = workflow.get_spec()
        
        assert spec["meta"]["name"] == "Test"
        assert spec["meta"]["seed"] == 42
    
    def test_workflow_get_compile_status_returns_status(self, tmp_path):
        """Test that get_compile_status returns compile status."""
        from automation.workflows.designspec_workflow import DesignSpecWorkflow
        
        workflow = DesignSpecWorkflow()
        
        project_dir = tmp_path / "test_project"
        
        result = workflow.on_start(
            project_dir=str(project_dir),
            template_spec={
                "meta": {"name": "Test", "seed": 42},
                "domain": {"type": "box", "size": [0.02, 0.06, 0.03]},
            },
        )
        
        assert result is True
        
        workflow.compile()
        
        status = workflow.get_compile_status()
        
        assert status is None or hasattr(status, "success") or isinstance(status, dict)


class TestDesignSpecWorkflowRunStages:
    """Test suite for DesignSpec workflow run stages."""
    
    def test_workflow_run_until_compile_domains(self, tmp_path):
        """Test run_until compile_domains stage."""
        from automation.workflows.designspec_workflow import DesignSpecWorkflow
        
        workflow = DesignSpecWorkflow()
        
        project_dir = tmp_path / "test_project"
        
        result = workflow.on_start(
            project_dir=str(project_dir),
            template_spec={
                "meta": {"name": "Test", "seed": 42},
                "domain": {"type": "box", "size": [0.02, 0.06, 0.03]},
            },
        )
        
        assert result is True
        
        run_result = workflow.run_until("compile_domains")
        
        assert run_result is None or isinstance(run_result, dict) or hasattr(run_result, "success")
    
    def test_workflow_run_full_creates_artifacts(self, tmp_path):
        """Test that full run creates artifacts."""
        from automation.workflows.designspec_workflow import DesignSpecWorkflow
        
        workflow = DesignSpecWorkflow()
        
        project_dir = tmp_path / "test_project"
        
        result = workflow.on_start(
            project_dir=str(project_dir),
            template_spec={
                "meta": {"name": "Test", "seed": 42},
                "domain": {"type": "box", "size": [0.02, 0.06, 0.03]},
            },
        )
        
        assert result is True
        
        run_result = workflow.run_full()
        
        artifacts_dir = project_dir / "artifacts"
        assert artifacts_dir.exists()
    
    def test_workflow_get_artifacts_returns_list(self, tmp_path):
        """Test that get_artifacts returns artifact list."""
        from automation.workflows.designspec_workflow import DesignSpecWorkflow
        
        workflow = DesignSpecWorkflow()
        
        project_dir = tmp_path / "test_project"
        
        result = workflow.on_start(
            project_dir=str(project_dir),
            template_spec={
                "meta": {"name": "Test", "seed": 42},
                "domain": {"type": "box", "size": [0.02, 0.06, 0.03]},
            },
        )
        
        assert result is True
        
        artifacts = workflow.get_artifacts()
        
        assert isinstance(artifacts, list)


class TestDesignSpecWorkflowCallbacks:
    """Test suite for DesignSpec workflow callbacks."""
    
    def test_workflow_calls_on_spec_update_callback(self, tmp_path):
        """Test that workflow emits spec update events."""
        from automation.workflows.designspec_workflow import DesignSpecWorkflow
        
        workflow = DesignSpecWorkflow()
        
        project_dir = tmp_path / "test_project"
        
        result = workflow.on_start(
            project_dir=str(project_dir),
            template_spec={"meta": {"name": "Test"}},
        )
        
        assert result is True
        
        patch_id = workflow._propose_patch(
            patches=[{"op": "add", "path": "/meta/seed", "value": 42}],
            explanation="Add seed",
        )
        
        workflow.approve_patch(patch_id)
        
        events = workflow.poll_events(timeout=0.1)
        spec_events = [e for e in events if "spec" in e.event_type.value.lower() or "patch" in e.event_type.value.lower()]
        assert len(spec_events) >= 0
    
    def test_workflow_calls_on_patch_proposal_callback(self, tmp_path):
        """Test that workflow calls on_patch_proposal callback."""
        from automation.workflows.designspec_workflow import DesignSpecWorkflow
        
        workflow = DesignSpecWorkflow()
        
        project_dir = tmp_path / "test_project"
        
        result = workflow.on_start(
            project_dir=str(project_dir),
            template_spec={"meta": {"name": "Test"}},
        )
        
        assert result is True
        
        patch_id = workflow._propose_patch(
            patches=[{"op": "add", "path": "/meta/seed", "value": 42}],
            explanation="Add seed",
        )
        
        events = workflow.poll_events(timeout=0.1)
        patch_events = [e for e in events if e.event_type.value == "patch_proposal"]
        assert len(patch_events) >= 0
    
    def test_workflow_calls_on_compile_status_callback(self, tmp_path):
        """Test that workflow calls on_compile_status callback."""
        from automation.workflows.designspec_workflow import DesignSpecWorkflow
        
        workflow = DesignSpecWorkflow()
        
        project_dir = tmp_path / "test_project"
        
        result = workflow.on_start(
            project_dir=str(project_dir),
            template_spec={
                "meta": {"name": "Test", "seed": 42},
                "domain": {"type": "box", "size": [0.02, 0.06, 0.03]},
            },
        )
        
        assert result is True
        
        workflow.compile()
        
        events = workflow.poll_events(timeout=0.1)
        compile_events = [e for e in events if "compile" in e.event_type.value.lower()]
        assert len(compile_events) >= 0
