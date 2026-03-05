"""
Unit tests for DesignSpec workflow command routing.

Tests that local diagnostic commands are properly detected and handled
before being passed to the agent.
"""

import json
import pytest
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

from automation.workflows.designspec_workflow import (
    DesignSpecWorkflow,
    WorkflowEvent,
    WorkflowEventType,
    WorkflowStatus,
)


class TestDesignSpecCommandRouter:
    """Tests for local command routing in DesignSpecWorkflow."""

    @pytest.fixture
    def tmp_project(self, tmp_path: Path) -> Path:
        """Create a temporary project directory with minimal spec."""
        project_dir = tmp_path / "test_project"
        project_dir.mkdir()
        
        spec = {
            "version": "1.0",
            "domains": {},
            "components": {},
            "features": {},
        }
        spec_path = project_dir / "spec.json"
        spec_path.write_text(json.dumps(spec, indent=2))
        
        return project_dir

    @pytest.fixture
    def workflow_with_events(self, tmp_project: Path) -> tuple:
        """Create a workflow and capture events."""
        events: List[WorkflowEvent] = []
        
        def event_callback(event: WorkflowEvent):
            events.append(event)
        
        workflow = DesignSpecWorkflow()
        workflow.on_event = event_callback
        
        return workflow, events, tmp_project

    def test_show_spec_command_emits_message(self, workflow_with_events, tmp_project):
        """Test that 'show spec' command emits MESSAGE event with spec.json path."""
        workflow, events, project_dir = workflow_with_events
        
        with patch.object(workflow, '_session') as mock_session:
            mock_session.get_spec.return_value = {"version": "1.0", "domains": {}}
            mock_session.project_dir = project_dir
            
            workflow._agent = MagicMock()
            
            result = workflow._handle_local_command("show spec")
            
            assert result is True, "Command should be handled"
            
            message_events = [e for e in events if e.event_type == WorkflowEventType.MESSAGE]
            assert len(message_events) > 0, "Expected MESSAGE event"
            
            event = message_events[0]
            assert "spec.json" in event.message, "Message should include spec.json path"

    def test_show_spec_variants(self, workflow_with_events, tmp_project):
        """Test various 'show spec' command variants."""
        workflow, events, project_dir = workflow_with_events
        
        with patch.object(workflow, '_session') as mock_session:
            mock_session.get_spec.return_value = {"version": "1.0"}
            mock_session.project_dir = project_dir
            workflow._agent = MagicMock()
            
            variants = [
                "show spec",
                "print spec",
                "output current design spec",
                "display spec",
                "Show the spec",
                "SHOW CURRENT SPEC",
            ]
            
            for variant in variants:
                events.clear()
                result = workflow._handle_local_command(variant)
                assert result is True, f"'{variant}' should be handled as show spec command"

    def test_what_is_missing_command_includes_domains_when_missing(self, workflow_with_events, tmp_project):
        """Test that 'what is missing' command shows checklist with domains when missing."""
        workflow, events, project_dir = workflow_with_events
        
        with patch.object(workflow, '_session') as mock_session:
            mock_session.get_spec.return_value = {"version": "1.0", "domains": {}}
            
            mock_validation = MagicMock()
            mock_validation.valid = False
            mock_validation.errors = ["Missing required domain"]
            mock_validation.warnings = []
            mock_validation.to_dict.return_value = {"valid": False, "errors": ["Missing required domain"]}
            mock_session.validate_spec.return_value = mock_validation
            
            mock_agent = MagicMock()
            mock_agent._build_missing_fields_checklist.return_value = "[ ] domains - No domains defined"
            workflow._agent = mock_agent
            
            result = workflow._handle_local_command("what is missing")
            
            assert result is True, "Command should be handled"
            
            message_events = [e for e in events if e.event_type == WorkflowEventType.MESSAGE]
            assert len(message_events) > 0, "Expected MESSAGE event"
            
            event = message_events[0]
            assert "domains" in event.message.lower(), "Checklist should mention domains"

    def test_missing_command_variants(self, workflow_with_events, tmp_project):
        """Test various 'what is missing' command variants."""
        workflow, events, project_dir = workflow_with_events
        
        with patch.object(workflow, '_session') as mock_session:
            mock_session.get_spec.return_value = {"version": "1.0"}
            mock_validation = MagicMock()
            mock_validation.valid = True
            mock_validation.errors = []
            mock_validation.warnings = []
            mock_validation.to_dict.return_value = {"valid": True}
            mock_session.validate_spec.return_value = mock_validation
            
            mock_agent = MagicMock()
            mock_agent._build_missing_fields_checklist.return_value = "All fields complete"
            workflow._agent = mock_agent
            
            variants = [
                "what is missing",
                "what's missing",
                "what is left",
                "status",
                "show status",
                "missing fields",
            ]
            
            for variant in variants:
                events.clear()
                result = workflow._handle_local_command(variant)
                assert result is True, f"'{variant}' should be handled as missing fields command"

    def test_last_error_command_with_no_previous_run(self, workflow_with_events, tmp_project):
        """Test 'show last error' when no previous run exists."""
        workflow, events, project_dir = workflow_with_events
        
        with patch.object(workflow, '_session') as mock_session:
            mock_session.get_last_runner_result.return_value = None
            workflow._agent = MagicMock()
            
            result = workflow._handle_local_command("show last error")
            
            assert result is True, "Command should be handled"
            
            message_events = [e for e in events if e.event_type == WorkflowEventType.MESSAGE]
            assert len(message_events) > 0, "Expected MESSAGE event"
            
            event = message_events[0]
            assert "no previous run" in event.message.lower(), "Should indicate no previous run"

    def test_last_error_command_with_failed_run(self, workflow_with_events, tmp_project):
        """Test 'show last error' with a failed run result."""
        workflow, events, project_dir = workflow_with_events
        
        with patch.object(workflow, '_session') as mock_session:
            mock_session.get_last_runner_result.return_value = {
                "success": False,
                "errors": ["Error 1", "Error 2", "Error 3", "Error 4"],
                "stage_reports": [
                    {"stage": "compile", "success": True},
                    {"stage": "union_voids", "success": False, "errors": ["Stage error"]},
                ],
                "output_dir": "/tmp/output",
                "run_id": "test-run-123",
            }
            workflow._agent = MagicMock()
            
            result = workflow._handle_local_command("why did it fail")
            
            assert result is True, "Command should be handled"
            
            message_events = [e for e in events if e.event_type == WorkflowEventType.MESSAGE]
            assert len(message_events) > 0, "Expected MESSAGE event"
            
            event = message_events[0]
            assert "FAILED" in event.message, "Should indicate failure"
            assert "Error 1" in event.message, "Should include first error"
            assert "union_voids" in event.message, "Should include failing stage"
            assert "/tmp/output" in event.message, "Should include output_dir"

    def test_error_command_variants(self, workflow_with_events, tmp_project):
        """Test various 'show error' command variants."""
        workflow, events, project_dir = workflow_with_events
        
        with patch.object(workflow, '_session') as mock_session:
            mock_session.get_last_runner_result.return_value = None
            workflow._agent = MagicMock()
            
            variants = [
                "why did it fail",
                "what is the issue",
                "show last error",
                "what went wrong",
                "show error",
                "last error",
            ]
            
            for variant in variants:
                events.clear()
                result = workflow._handle_local_command(variant)
                assert result is True, f"'{variant}' should be handled as error command"

    def test_non_command_returns_false(self, workflow_with_events, tmp_project):
        """Test that non-command messages return False."""
        workflow, events, project_dir = workflow_with_events
        
        workflow._agent = MagicMock()
        
        non_commands = [
            "I want a box 20mm on all sides",
            "add a channel",
            "make the domain larger",
            "hello",
            "run the pipeline",
        ]
        
        for message in non_commands:
            result = workflow._handle_local_command(message)
            assert result is False, f"'{message}' should NOT be handled as a command"

    def test_show_spec_truncates_large_spec(self, workflow_with_events, tmp_project):
        """Test that large specs are truncated in show spec output."""
        workflow, events, project_dir = workflow_with_events
        
        with patch.object(workflow, '_session') as mock_session:
            large_spec = {"data": "x" * 3000}
            mock_session.get_spec.return_value = large_spec
            mock_session.project_dir = project_dir
            workflow._agent = MagicMock()
            
            workflow._handle_local_command("show spec")
            
            message_events = [e for e in events if e.event_type == WorkflowEventType.MESSAGE]
            assert len(message_events) > 0
            
            event = message_events[0]
            assert "truncated" in event.message.lower(), "Large spec should be truncated"

    def test_command_sets_status_to_waiting_input(self, workflow_with_events, tmp_project):
        """Test that handling a command sets status back to WAITING_INPUT."""
        workflow, events, project_dir = workflow_with_events
        
        status_changes = []
        
        def track_status(status, message=""):
            status_changes.append(status)
        
        workflow._set_status = track_status
        
        with patch.object(workflow, '_session') as mock_session:
            mock_session.get_spec.return_value = {"version": "1.0"}
            mock_session.project_dir = project_dir
            workflow._agent = MagicMock()
            workflow._lock = MagicMock()
            workflow._lock.__enter__ = MagicMock(return_value=None)
            workflow._lock.__exit__ = MagicMock(return_value=None)
            
            workflow.on_user_message("show spec")
