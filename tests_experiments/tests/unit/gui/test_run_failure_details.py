"""
Unit tests for DesignSpecWorkflowManager run failure display.

Tests that run failures are displayed with detailed error information
including errors list, failing stage, and output directory.
"""

import pytest
from typing import List
from unittest.mock import MagicMock

from gui.designspec_workflow_manager import DesignSpecWorkflowManager
from gui.workflow_manager import WorkflowMessage, WorkflowStatus


class TestRunFailureDetails:
    """Tests for run failure detail formatting in DesignSpecWorkflowManager."""

    def test_format_run_failure_with_errors(self):
        """Test that run failure includes error strings."""
        manager = DesignSpecWorkflowManager()
        
        result = {
            "success": False,
            "errors": ["Error 1: Something went wrong", "Error 2: Another issue"],
            "stage_reports": [],
            "output_dir": "/tmp/output/run_123",
        }
        
        formatted = manager._format_run_failure_details(result)
        
        assert "Run failed" in formatted
        assert "Error 1" in formatted
        assert "Error 2" in formatted

    def test_format_run_failure_with_output_dir(self):
        """Test that run failure includes output_dir path."""
        manager = DesignSpecWorkflowManager()
        
        result = {
            "success": False,
            "errors": ["Some error"],
            "stage_reports": [],
            "output_dir": "/tmp/output/run_456",
        }
        
        formatted = manager._format_run_failure_details(result)
        
        assert "/tmp/output/run_456" in formatted

    def test_format_run_failure_with_failing_stage(self):
        """Test that run failure includes failing stage name."""
        manager = DesignSpecWorkflowManager()
        
        result = {
            "success": False,
            "errors": [],
            "stage_reports": [
                {"stage": "compile", "success": True},
                {"stage": "union_voids", "success": False, "errors": ["Stage failed"]},
                {"stage": "export", "success": True},
            ],
            "output_dir": "/tmp/output",
        }
        
        formatted = manager._format_run_failure_details(result)
        
        assert "union_voids" in formatted
        assert "Failing stage" in formatted

    def test_format_run_failure_with_stage_errors(self):
        """Test that run failure includes stage-specific errors."""
        manager = DesignSpecWorkflowManager()
        
        result = {
            "success": False,
            "errors": [],
            "stage_reports": [
                {
                    "stage": "mesh_generation",
                    "success": False,
                    "errors": ["Mesh error 1", "Mesh error 2"],
                },
            ],
            "output_dir": "/tmp/output",
        }
        
        formatted = manager._format_run_failure_details(result)
        
        assert "mesh_generation" in formatted
        assert "Mesh error 1" in formatted

    def test_format_run_failure_empty_result(self):
        """Test that run failure handles empty result gracefully."""
        manager = DesignSpecWorkflowManager()
        
        result = {
            "success": False,
        }
        
        formatted = manager._format_run_failure_details(result)
        
        assert "Run failed" in formatted

    def test_run_completed_event_calls_format_on_failure(self):
        """Test that RUN_COMPLETED event with failure uses detailed formatting."""
        messages: List[WorkflowMessage] = []
        
        def message_callback(msg: WorkflowMessage):
            messages.append(msg)
        
        manager = DesignSpecWorkflowManager(message_callback=message_callback)
        
        from automation.workflows.designspec_workflow import WorkflowEvent, WorkflowEventType
        
        event = WorkflowEvent(
            event_type=WorkflowEventType.RUN_COMPLETED,
            data={
                "result": {
                    "success": False,
                    "errors": ["Test error message"],
                    "stage_reports": [
                        {"stage": "test_stage", "success": False, "errors": ["Stage error"]},
                    ],
                    "output_dir": "/test/output/dir",
                }
            },
            message="Run completed",
        )
        
        manager._on_workflow_event(event)
        
        error_messages = [m for m in messages if m.type == "error"]
        assert len(error_messages) > 0
        
        error_content = error_messages[0].content
        assert "Test error message" in error_content
        assert "test_stage" in error_content
        assert "/test/output/dir" in error_content

    def test_run_completed_event_success_does_not_show_errors(self):
        """Test that successful RUN_COMPLETED event shows success message."""
        messages: List[WorkflowMessage] = []
        
        def message_callback(msg: WorkflowMessage):
            messages.append(msg)
        
        manager = DesignSpecWorkflowManager(message_callback=message_callback)
        
        from automation.workflows.designspec_workflow import WorkflowEvent, WorkflowEventType
        
        event = WorkflowEvent(
            event_type=WorkflowEventType.RUN_COMPLETED,
            data={
                "result": {
                    "success": True,
                    "artifacts": [],
                }
            },
            message="Run completed successfully",
        )
        
        manager._on_workflow_event(event)
        
        success_messages = [m for m in messages if m.type == "success"]
        assert len(success_messages) > 0
        assert "Run completed successfully" in success_messages[0].content

    def test_format_limits_errors_to_five(self):
        """Test that error list is limited to first 5 errors."""
        manager = DesignSpecWorkflowManager()
        
        result = {
            "success": False,
            "errors": [f"Error {i}" for i in range(10)],
            "stage_reports": [],
            "output_dir": "",
        }
        
        formatted = manager._format_run_failure_details(result)
        
        assert "Error 0" in formatted
        assert "Error 4" in formatted
        assert "Error 5" not in formatted

    def test_format_limits_stage_errors_to_three(self):
        """Test that stage error list is limited to first 3 errors."""
        manager = DesignSpecWorkflowManager()
        
        result = {
            "success": False,
            "errors": [],
            "stage_reports": [
                {
                    "stage": "test_stage",
                    "success": False,
                    "errors": [f"Stage error {i}" for i in range(10)],
                },
            ],
            "output_dir": "",
        }
        
        formatted = manager._format_run_failure_details(result)
        
        assert "Stage error 0" in formatted
        assert "Stage error 2" in formatted
        assert "Stage error 3" not in formatted
