"""
Integration tests for the DesignSpec GUI workflow loop.

Tests the cube + channel + ridge workflow via conversation.
"""

import json
import pytest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from automation.workflows.designspec_workflow import (
    DesignSpecWorkflow,
    WorkflowEventType,
    WorkflowEvent,
)


class TestDesignSpecGuiWorkflowLoop:
    """Integration tests for the DesignSpec GUI workflow loop."""

    def test_cube_request_emits_patch_proposal(self, tmp_path: Path):
        """Test that a cube request emits a patch proposal."""
        events: List[WorkflowEvent] = []
        
        def event_handler(event: WorkflowEvent):
            events.append(event)
        
        workflow = DesignSpecWorkflow(event_callback=event_handler)
        
        workflow.on_start(project_root=tmp_path, project_name="test_cube_project")
        
        workflow.on_user_message("Create a box with 20 mm sides")
        
        patch_events = [e for e in events if e.event_type == WorkflowEventType.PATCH_PROPOSAL]
        assert len(patch_events) > 0, "Expected at least one PATCH_PROPOSAL event"
        
        patch_data = patch_events[0].data
        patches = patch_data.get("patches", [])
        
        domain_patch = next(
            (p for p in patches if "/domains" in p.get("path", "")),
            None
        )
        assert domain_patch is not None, "Expected a domain patch"

    def test_approve_patch_updates_spec_and_compiles(self, tmp_path: Path):
        """Test that approving a patch updates spec.json and triggers compile."""
        events: List[WorkflowEvent] = []
        
        def event_handler(event: WorkflowEvent):
            events.append(event)
        
        workflow = DesignSpecWorkflow(event_callback=event_handler)
        
        workflow.on_start(project_root=tmp_path, project_name="test_approve_project")
        
        workflow.on_user_message("Create a box with 20 mm sides")
        
        patch_events = [e for e in events if e.event_type == WorkflowEventType.PATCH_PROPOSAL]
        assert len(patch_events) > 0, "Expected PATCH_PROPOSAL event"
        
        patch_id = patch_events[0].data.get("patch_id", "")
        
        events.clear()
        workflow.approve_patch(patch_id)
        
        compile_events = [e for e in events if e.event_type == WorkflowEventType.COMPILE_COMPLETED]
        assert len(compile_events) > 0, "Expected COMPILE_COMPLETED event after patch approval"
        
        spec_path = tmp_path / "test_approve_project" / "spec.json"
        assert spec_path.exists(), "spec.json should exist after patch approval"
        
        with open(spec_path) as f:
            spec = json.load(f)
        
        assert "domains" in spec, "Spec should have domains after patch"
        assert "main_domain" in spec["domains"], "Spec should have main_domain"
        assert spec["domains"]["main_domain"]["type"] == "box"
        assert spec["domains"]["main_domain"]["x_min"] == -10.0
        assert spec["domains"]["main_domain"]["x_max"] == 10.0

    def test_channel_request_creates_component(self, tmp_path: Path):
        """Test that a channel request creates a primitive_channels component."""
        events: List[WorkflowEvent] = []
        
        def event_handler(event: WorkflowEvent):
            events.append(event)
        
        workflow = DesignSpecWorkflow(event_callback=event_handler)
        
        workflow.on_start(project_root=tmp_path, project_name="test_channel_project")
        
        workflow.on_user_message("Create a box with 20 mm sides")
        patch_events = [e for e in events if e.event_type == WorkflowEventType.PATCH_PROPOSAL]
        if patch_events:
            patch_id = patch_events[0].data.get("patch_id", "")
            workflow.approve_patch(patch_id)
        
        events.clear()
        workflow.on_user_message("Add a straight channel through it")
        
        patch_events = [e for e in events if e.event_type == WorkflowEventType.PATCH_PROPOSAL]
        assert len(patch_events) > 0, "Expected PATCH_PROPOSAL for channel"
        
        patches = patch_events[0].data.get("patches", [])
        component_patch = next(
            (p for p in patches if "/components" in p.get("path", "")),
            None
        )
        assert component_patch is not None, "Expected a component patch"
        
        component_value = component_patch.get("value", {})
        assert component_value.get("build", {}).get("type") == "primitive_channels"

    def test_ridge_request_adds_features(self, tmp_path: Path):
        """Test that a ridge request adds ridge features."""
        events: List[WorkflowEvent] = []
        
        def event_handler(event: WorkflowEvent):
            events.append(event)
        
        workflow = DesignSpecWorkflow(event_callback=event_handler)
        
        workflow.on_start(project_root=tmp_path, project_name="test_ridge_project")
        
        workflow.on_user_message("Create a box with 20 mm sides")
        patch_events = [e for e in events if e.event_type == WorkflowEventType.PATCH_PROPOSAL]
        if patch_events:
            patch_id = patch_events[0].data.get("patch_id", "")
            workflow.approve_patch(patch_id)
        
        events.clear()
        workflow.on_user_message("Add ridges on left face and right face")
        
        patch_events = [e for e in events if e.event_type == WorkflowEventType.PATCH_PROPOSAL]
        assert len(patch_events) > 0, "Expected PATCH_PROPOSAL for ridges"
        
        patches = patch_events[0].data.get("patches", [])
        ridge_patch = next(
            (p for p in patches if "/features" in p.get("path", "")),
            None
        )
        assert ridge_patch is not None, "Expected a features patch"
        
        features_value = ridge_patch.get("value", {})
        ridges = features_value.get("ridges", features_value)
        if isinstance(ridges, list):
            faces = [r.get("face") for r in ridges]
            assert "-x" in faces, "Expected left face (-x) ridge"
            assert "+x" in faces, "Expected right face (+x) ridge"

    def test_full_cube_channel_ridge_workflow(self, tmp_path: Path):
        """Test the full cube + channel + ridge workflow."""
        events: List[WorkflowEvent] = []
        
        def event_handler(event: WorkflowEvent):
            events.append(event)
        
        workflow = DesignSpecWorkflow(event_callback=event_handler)
        
        workflow.on_start(project_root=tmp_path, project_name="test_full_workflow")
        
        workflow.on_user_message("Create a box with 20 mm sides")
        patch_events = [e for e in events if e.event_type == WorkflowEventType.PATCH_PROPOSAL]
        assert len(patch_events) > 0, "Expected PATCH_PROPOSAL for cube"
        patch_id = patch_events[0].data.get("patch_id", "")
        workflow.approve_patch(patch_id)
        
        events.clear()
        workflow.on_user_message("Add a straight channel through it")
        patch_events = [e for e in events if e.event_type == WorkflowEventType.PATCH_PROPOSAL]
        assert len(patch_events) > 0, "Expected PATCH_PROPOSAL for channel"
        patch_id = patch_events[0].data.get("patch_id", "")
        workflow.approve_patch(patch_id)
        
        events.clear()
        workflow.on_user_message("Add ridges on left face and right face")
        patch_events = [e for e in events if e.event_type == WorkflowEventType.PATCH_PROPOSAL]
        assert len(patch_events) > 0, "Expected PATCH_PROPOSAL for ridges"
        patch_id = patch_events[0].data.get("patch_id", "")
        workflow.approve_patch(patch_id)
        
        spec_path = tmp_path / "test_full_workflow" / "spec.json"
        assert spec_path.exists(), "spec.json should exist"
        
        with open(spec_path) as f:
            spec = json.load(f)
        
        assert "domains" in spec
        assert "main_domain" in spec["domains"]
        assert spec["domains"]["main_domain"]["type"] == "box"
        assert spec["domains"]["main_domain"]["x_min"] == -10.0
        assert spec["domains"]["main_domain"]["x_max"] == 10.0
        
        assert "components" in spec
        assert len(spec["components"]) > 0
        channel_comp = next(
            (c for c in spec["components"] if c.get("build", {}).get("type") == "primitive_channels"),
            None
        )
        assert channel_comp is not None, "Expected primitive_channels component"
        
        assert "features" in spec
        assert "ridges" in spec["features"]
        ridges = spec["features"]["ridges"]
        faces = [r.get("face") for r in ridges]
        assert "-x" in faces
        assert "+x" in faces

    def test_compile_event_emitted_after_patch_approval(self, tmp_path: Path):
        """Test that compile event is emitted after patch approval."""
        events: List[WorkflowEvent] = []
        
        def event_handler(event: WorkflowEvent):
            events.append(event)
        
        workflow = DesignSpecWorkflow(event_callback=event_handler)
        
        workflow.on_start(project_root=tmp_path, project_name="test_compile_event")
        
        workflow.on_user_message("Create a box with 20 mm sides")
        patch_events = [e for e in events if e.event_type == WorkflowEventType.PATCH_PROPOSAL]
        assert len(patch_events) > 0
        patch_id = patch_events[0].data.get("patch_id", "")
        
        events.clear()
        workflow.approve_patch(patch_id)
        
        compile_events = [e for e in events if e.event_type == WorkflowEventType.COMPILE_COMPLETED]
        assert len(compile_events) > 0, "Expected COMPILE_COMPLETED event"
        
        compile_data = compile_events[-1].data
        compile_report = compile_data.get("compile_report", {})
        assert "success" in compile_report, "Compile report should have success field"
