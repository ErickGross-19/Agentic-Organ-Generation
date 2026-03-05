"""
Unit tests for WorkflowManager DESIGNSPEC workflow type handling.

Tests that DESIGNSPEC workflow type is properly blocked with a friendly message
when started via the generic WorkflowManager (should use DesignSpec Project dialog instead).
"""

import pytest
import time
from unittest.mock import MagicMock, patch
from typing import List

from gui.workflow_manager import (
    WorkflowManager,
    WorkflowConfig,
    WorkflowType,
    WorkflowStatus,
    WorkflowMessage,
)


class TestWorkflowManagerDesignspecBlocked:
    """Tests for DESIGNSPEC workflow type blocking in WorkflowManager."""

    def test_designspec_workflow_emits_friendly_message(self):
        """Test that DESIGNSPEC workflow emits friendly message instead of raising."""
        messages: List[WorkflowMessage] = []
        statuses: List[tuple] = []
        
        def message_callback(msg: WorkflowMessage):
            messages.append(msg)
        
        def status_callback(status: WorkflowStatus, message: str):
            statuses.append((status, message))
        
        manager = WorkflowManager(
            message_callback=message_callback,
            status_callback=status_callback,
        )
        
        with patch.object(manager, '_agent', MagicMock()):
            config = WorkflowConfig(workflow_type=WorkflowType.DESIGNSPEC)
            result = manager.start_workflow(config)
            
            assert result is True, "start_workflow should return True"
            
            time.sleep(0.5)
            
            system_messages = [m for m in messages if m.type == "system"]
            assert len(system_messages) > 0, "Expected at least one system message"
            
            friendly_message_found = any(
                "DesignSpec Project dialog" in m.content
                for m in system_messages
            )
            assert friendly_message_found, (
                "Expected friendly message about DesignSpec Project dialog, "
                f"got messages: {[m.content for m in system_messages]}"
            )

    def test_designspec_workflow_does_not_raise(self):
        """Test that DESIGNSPEC workflow does not raise an exception."""
        manager = WorkflowManager()
        
        with patch.object(manager, '_agent', MagicMock()):
            config = WorkflowConfig(workflow_type=WorkflowType.DESIGNSPEC)
            
            try:
                result = manager.start_workflow(config)
                time.sleep(0.5)
            except Exception as e:
                pytest.fail(f"DESIGNSPEC workflow should not raise, but got: {e}")

    def test_designspec_workflow_sets_waiting_input_status(self):
        """Test that DESIGNSPEC workflow sets status to WAITING_INPUT."""
        statuses: List[tuple] = []
        
        def status_callback(status: WorkflowStatus, message: str):
            statuses.append((status, message))
        
        manager = WorkflowManager(status_callback=status_callback)
        
        with patch.object(manager, '_agent', MagicMock()):
            config = WorkflowConfig(workflow_type=WorkflowType.DESIGNSPEC)
            manager.start_workflow(config)
            
            time.sleep(0.5)
            
            final_status = statuses[-1][0] if statuses else None
            assert final_status == WorkflowStatus.WAITING_INPUT, (
                f"Expected final status WAITING_INPUT, got {final_status}"
            )

    def test_designspec_workflow_message_mentions_file_menu(self):
        """Test that the friendly message mentions File menu options."""
        messages: List[WorkflowMessage] = []
        
        def message_callback(msg: WorkflowMessage):
            messages.append(msg)
        
        manager = WorkflowManager(message_callback=message_callback)
        
        with patch.object(manager, '_agent', MagicMock()):
            config = WorkflowConfig(workflow_type=WorkflowType.DESIGNSPEC)
            manager.start_workflow(config)
            
            time.sleep(0.5)
            
            system_messages = [m for m in messages if m.type == "system"]
            file_menu_mentioned = any(
                "File" in m.content or "New DesignSpec" in m.content or "Open DesignSpec" in m.content
                for m in system_messages
            )
            assert file_menu_mentioned, (
                "Expected message to mention File menu options for DesignSpec projects"
            )
