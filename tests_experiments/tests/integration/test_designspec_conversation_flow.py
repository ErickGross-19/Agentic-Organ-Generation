"""
Integration test for DesignSpec conversation flow.

Tests the full conversation loop without Tk GUI:
1. Create project
2. Send message describing desired structure
3. Receive PATCH_PROPOSAL event
4. Approve patch
5. Verify compile is successful
6. Send "what is missing" command
7. Verify checklist is shown without looping
"""

import json
import pytest
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

from automation.workflows.designspec_workflow import (
    DesignSpecWorkflow,
    WorkflowEvent,
    WorkflowEventType,
    WorkflowStatus,
)


class TestDesignSpecConversationFlow:
    """Integration tests for DesignSpec conversation flow."""

    @pytest.fixture
    def tmp_project_root(self, tmp_path: Path) -> Path:
        """Create a temporary project root directory."""
        return tmp_path

    @pytest.fixture
    def event_collector(self):
        """Create an event collector for capturing workflow events."""
        events: List[WorkflowEvent] = []
        
        def collector(event: WorkflowEvent):
            events.append(event)
        
        return events, collector

    def test_conversation_flow_box_channel_ridge(self, tmp_project_root, event_collector):
        """
        Test full conversation flow for box + channel + ridge request.
        
        Steps:
        1. Create project
        2. Send message: "I want a box 20mm on all sides with a straight channel through it and a ridge on the left side"
        3. Expect PATCH_PROPOSAL event with domain + component + ridge patches
        4. Approve patch
        5. Verify spec is updated
        """
        events, collector = event_collector
        
        workflow = DesignSpecWorkflow(
            llm_client=None,
            event_callback=collector,
        )
        
        success = workflow.on_start(
            project_root=str(tmp_project_root),
            project_name="test_project",
        )
        assert success, "Failed to start workflow"
        
        events.clear()
        
        workflow.on_user_message(
            "I want a box 20mm on all sides with a straight channel through it and a ridge on the left side"
        )
        
        patch_events = [
            e for e in events
            if e.event_type == WorkflowEventType.PATCH_PROPOSAL
        ]
        
        assert len(patch_events) > 0, "Expected PATCH_PROPOSAL event"
        
        patch_event = patch_events[0]
        patches = patch_event.data.get("patches", [])
        
        domain_patch = next(
            (p for p in patches if "/domains" in p.get("path", "")),
            None
        )
        assert domain_patch is not None, "Expected domain patch"
        
        component_patch = next(
            (p for p in patches if "/components" in p.get("path", "")),
            None
        )
        assert component_patch is not None, "Expected component patch"
        
        ridge_patch = next(
            (p for p in patches if "/features" in p.get("path", "")),
            None
        )
        assert ridge_patch is not None, "Expected ridge patch"
        
        patch_id = patch_event.data.get("patch_id")
        assert patch_id is not None, "Expected patch_id in event"
        
        events.clear()
        workflow.approve_patch(patch_id)
        
        spec = workflow.get_spec()
        assert "domains" in spec and spec["domains"], "Spec should have domains after patch"

    def test_what_is_missing_command_does_not_loop(self, tmp_project_root, event_collector):
        """
        Test that "what is missing" command shows checklist without looping.
        
        Steps:
        1. Create project
        2. Send "what is missing" command
        3. Verify MESSAGE event with checklist is emitted
        4. Verify no infinite loop (only one response)
        """
        events, collector = event_collector
        
        workflow = DesignSpecWorkflow(
            llm_client=None,
            event_callback=collector,
        )
        
        success = workflow.on_start(
            project_root=str(tmp_project_root),
            project_name="test_project_missing",
        )
        assert success, "Failed to start workflow"
        
        events.clear()
        
        workflow.on_user_message("what is missing")
        
        message_events = [
            e for e in events
            if e.event_type == WorkflowEventType.MESSAGE
        ]
        
        assert len(message_events) >= 1, "Expected at least one MESSAGE event"
        
        checklist_event = next(
            (e for e in message_events if "checklist" in e.message.lower() or "domains" in e.message.lower()),
            None
        )
        assert checklist_event is not None, "Expected checklist in message"
        
        assert len(message_events) < 5, "Too many MESSAGE events - possible loop detected"

    def test_show_spec_command(self, tmp_project_root, event_collector):
        """
        Test that "show spec" command displays current spec.
        
        Steps:
        1. Create project
        2. Send "show spec" command
        3. Verify MESSAGE event with spec.json path is emitted
        """
        events, collector = event_collector
        
        workflow = DesignSpecWorkflow(
            llm_client=None,
            event_callback=collector,
        )
        
        success = workflow.on_start(
            project_root=str(tmp_project_root),
            project_name="test_project_show",
        )
        assert success, "Failed to start workflow"
        
        events.clear()
        
        workflow.on_user_message("show spec")
        
        message_events = [
            e for e in events
            if e.event_type == WorkflowEventType.MESSAGE
        ]
        
        assert len(message_events) >= 1, "Expected MESSAGE event"
        
        spec_event = next(
            (e for e in message_events if "spec.json" in e.message),
            None
        )
        assert spec_event is not None, "Expected spec.json path in message"

    def test_patch_approval_updates_spec(self, tmp_project_root, event_collector):
        """
        Test that approving a patch updates the spec correctly.
        """
        events, collector = event_collector
        
        workflow = DesignSpecWorkflow(
            llm_client=None,
            event_callback=collector,
        )
        
        success = workflow.on_start(
            project_root=str(tmp_project_root),
            project_name="test_project_approval",
        )
        assert success, "Failed to start workflow"
        
        initial_spec = workflow.get_spec()
        initial_domains = initial_spec.get("domains", {})
        
        events.clear()
        
        workflow.on_user_message("box should be 20mm on all sides")
        
        patch_events = [
            e for e in events
            if e.event_type == WorkflowEventType.PATCH_PROPOSAL
        ]
        
        if patch_events:
            patch_id = patch_events[0].data.get("patch_id")
            workflow.approve_patch(patch_id)
            
            updated_spec = workflow.get_spec()
            updated_domains = updated_spec.get("domains", {})
            
            assert updated_domains != initial_domains or "main_domain" in updated_domains, \
                "Spec domains should be updated after patch approval"

    def test_sequential_messages_build_spec(self, tmp_project_root, event_collector):
        """
        Test that sequential messages correctly build up the spec.
        """
        events, collector = event_collector
        
        workflow = DesignSpecWorkflow(
            llm_client=None,
            event_callback=collector,
        )
        
        success = workflow.on_start(
            project_root=str(tmp_project_root),
            project_name="test_project_sequential",
        )
        assert success, "Failed to start workflow"
        
        events.clear()
        workflow.on_user_message("box should be 20mm on all sides")
        
        patch_events = [e for e in events if e.event_type == WorkflowEventType.PATCH_PROPOSAL]
        if patch_events:
            patch_id = patch_events[0].data.get("patch_id")
            workflow.approve_patch(patch_id)
        
        spec_after_domain = workflow.get_spec()
        assert "domains" in spec_after_domain and spec_after_domain["domains"], \
            "Spec should have domains after first message"
        
        events.clear()
        workflow.on_user_message("add a straight channel through it")
        
        patch_events = [e for e in events if e.event_type == WorkflowEventType.PATCH_PROPOSAL]
        if patch_events:
            patch_id = patch_events[0].data.get("patch_id")
            workflow.approve_patch(patch_id)
        
        spec_after_channel = workflow.get_spec()
        components = spec_after_channel.get("components", [])
        assert len(components) > 0, "Spec should have components after second message"
        
        events.clear()
        workflow.on_user_message("add a ridge on the left side")
        
        patch_events = [e for e in events if e.event_type == WorkflowEventType.PATCH_PROPOSAL]
        if patch_events:
            patch_id = patch_events[0].data.get("patch_id")
            workflow.approve_patch(patch_id)
        
        spec_after_ridge = workflow.get_spec()
        features = spec_after_ridge.get("features", {})
        ridges = features.get("ridges", [])
        assert len(ridges) > 0, "Spec should have ridges after third message"

    def test_error_command_with_no_previous_run(self, tmp_project_root, event_collector):
        """
        Test that "show last error" command handles no previous run gracefully.
        """
        events, collector = event_collector
        
        workflow = DesignSpecWorkflow(
            llm_client=None,
            event_callback=collector,
        )
        
        success = workflow.on_start(
            project_root=str(tmp_project_root),
            project_name="test_project_error",
        )
        assert success, "Failed to start workflow"
        
        events.clear()
        
        workflow.on_user_message("show last error")
        
        message_events = [
            e for e in events
            if e.event_type == WorkflowEventType.MESSAGE
        ]
        
        assert len(message_events) >= 1, "Expected MESSAGE event"
        
        no_run_event = next(
            (e for e in message_events if "no previous run" in e.message.lower()),
            None
        )
        assert no_run_event is not None, "Expected 'no previous run' message"
