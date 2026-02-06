"""
Tests for the DesignSpec REST API bridge (Phase 2).

Tests cover:
1. DesignSpecBridge class initialization and state
2. DesignSpecBridge project creation and loading
3. DesignSpecBridge event collection from workflow callbacks
4. DesignSpecBridge message sending (async)
5. DesignSpecBridge patch approval/rejection
6. DesignSpecBridge pipeline run
7. REST endpoint routing and response shapes
8. Integration with DesignSpecWorkflow event types
"""

import importlib.util
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_backend_dir = os.path.join(_repo_root, "backend")
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

_designspec_file = os.path.join(
    _repo_root, "backend", "app", "api", "designspec.py"
)


def _load_designspec_module():
    """Load designspec.py directly via importlib, bypassing __init__.py."""
    spec = importlib.util.spec_from_file_location(
        "designspec_mod", _designspec_file
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ds = _load_designspec_module()
DesignSpecBridge = _ds.DesignSpecBridge
get_bridge = _ds.get_bridge
CreateProjectRequest = _ds.CreateProjectRequest
LoadProjectRequest = _ds.LoadProjectRequest
MessageRequest = _ds.MessageRequest
PatchActionRequest = _ds.PatchActionRequest
RunUntilRequest = _ds.RunUntilRequest
LLMConfigRequest = _ds.LLMConfigRequest
router = _ds.router


class TestDesignSpecBridgeInit:
    """Tests for DesignSpecBridge initialization."""

    def test_bridge_import(self):
        bridge = DesignSpecBridge()
        assert bridge._workflow is None
        assert bridge._event_buffer == []
        assert bridge._project_dir is None

    def test_bridge_get_status_no_project(self):
        bridge = DesignSpecBridge()
        status = bridge.get_status()
        assert status["status"] == "no_project"
        assert status["project_dir"] is None

    def test_bridge_ensure_workflow_raises_without_project(self):
        from fastapi import HTTPException
        bridge = DesignSpecBridge()
        with pytest.raises(HTTPException) as exc_info:
            bridge._ensure_workflow()
        assert exc_info.value.status_code == 400
        assert "No project loaded" in exc_info.value.detail


class TestDesignSpecBridgeEventCollection:
    """Tests for event collection from workflow callbacks."""

    def test_on_workflow_event_collects_events(self):
        bridge = DesignSpecBridge()

        mock_event = MagicMock()
        mock_event.to_dict.return_value = {
            "event_type": "message",
            "data": {},
            "message": "Hello",
            "timestamp": 1234567890.0,
        }

        bridge._on_workflow_event(mock_event)

        assert len(bridge._event_buffer) == 1
        assert bridge._event_buffer[0]["event_type"] == "message"
        assert bridge._event_buffer[0]["message"] == "Hello"

    def test_poll_events_clears_buffer(self):
        bridge = DesignSpecBridge()

        mock_event = MagicMock()
        mock_event.to_dict.return_value = {
            "event_type": "message",
            "data": {},
            "message": "test",
            "timestamp": 0.0,
        }

        bridge._on_workflow_event(mock_event)
        bridge._on_workflow_event(mock_event)

        events = bridge.poll_events()
        assert len(events) == 2

        events2 = bridge.poll_events()
        assert len(events2) == 0

    def test_multiple_event_types_collected(self):
        bridge = DesignSpecBridge()

        event_types = ["message", "patch_proposal", "run_progress", "error"]
        for etype in event_types:
            mock_event = MagicMock()
            mock_event.to_dict.return_value = {
                "event_type": etype,
                "data": {},
                "message": f"Event: {etype}",
                "timestamp": 0.0,
            }
            bridge._on_workflow_event(mock_event)

        events = bridge.poll_events()
        assert len(events) == 4
        collected_types = [e["event_type"] for e in events]
        assert collected_types == event_types


class TestDesignSpecBridgeProjectManagement:
    """Tests for project creation and loading."""

    def test_create_project_initializes_workflow(self):
        bridge = DesignSpecBridge()

        with patch.object(
            bridge, "create_project", wraps=bridge.create_project
        ):
            with patch(
                "automation.workflows.designspec_workflow.DesignSpecWorkflow"
            ) as MockWorkflow:
                mock_workflow_instance = MagicMock()
                mock_workflow_instance.on_start.return_value = True
                MockWorkflow.return_value = mock_workflow_instance

                success = bridge.create_project("test_project")

                assert success is True
                assert bridge._workflow is not None
                assert bridge._project_dir is not None

    def test_create_project_failure(self):
        bridge = DesignSpecBridge()

        with patch(
            "automation.workflows.designspec_workflow.DesignSpecWorkflow"
        ) as MockWorkflow:
            mock_workflow_instance = MagicMock()
            mock_workflow_instance.on_start.return_value = False
            MockWorkflow.return_value = mock_workflow_instance

            success = bridge.create_project("test_project")

            assert success is False
            assert bridge._project_dir is None

    def test_load_project_initializes_workflow(self):
        bridge = DesignSpecBridge()
        project_dir = "/tmp/test_project"

        with patch(
            "automation.workflows.designspec_workflow.DesignSpecWorkflow"
        ) as MockWorkflow:
            mock_workflow_instance = MagicMock()
            mock_workflow_instance.on_start.return_value = True
            MockWorkflow.return_value = mock_workflow_instance

            success = bridge.load_project(project_dir)

            assert success is True
            assert bridge._project_dir == Path(project_dir)


class TestDesignSpecBridgePatchOperations:
    """Tests for patch approval/rejection."""

    def test_approve_patch_delegates_to_workflow(self):
        bridge = DesignSpecBridge()
        bridge._workflow = MagicMock()
        bridge._workflow.approve_patch.return_value = True
        bridge._workflow.get_spec.return_value = {"domain": {}}

        result = bridge.approve_patch("patch-123")

        assert result["success"] is True
        assert result["patch_id"] == "patch-123"
        assert result["spec"] == {"domain": {}}
        bridge._workflow.approve_patch.assert_called_once_with("patch-123")

    def test_reject_patch_delegates_to_workflow(self):
        bridge = DesignSpecBridge()
        bridge._workflow = MagicMock()

        result = bridge.reject_patch("patch-456", "not needed")

        assert result["patch_id"] == "patch-456"
        assert result["reason"] == "not needed"
        bridge._workflow.reject_patch.assert_called_once_with("patch-456", "not needed")


class TestDesignSpecBridgeSpecAndArtifacts:
    """Tests for spec and artifact retrieval."""

    def test_get_spec_delegates_to_workflow(self):
        bridge = DesignSpecBridge()
        bridge._workflow = MagicMock()
        bridge._workflow.get_spec.return_value = {"domain": {"type": "cylinder"}}

        spec = bridge.get_spec()
        assert spec == {"domain": {"type": "cylinder"}}

    def test_get_pending_patches_delegates_to_workflow(self):
        bridge = DesignSpecBridge()
        bridge._workflow = MagicMock()
        bridge._workflow.get_pending_patches.return_value = {
            "patch-1": {"description": "Add domain"}
        }

        patches = bridge.get_pending_patches()
        assert "patch-1" in patches

    def test_get_artifacts_delegates_to_workflow(self):
        bridge = DesignSpecBridge()
        bridge._workflow = MagicMock()
        bridge._workflow.get_artifacts.return_value = [
            {"type": "stl", "path": "/tmp/out.stl"}
        ]

        artifacts = bridge.get_artifacts()
        assert len(artifacts) == 1
        assert artifacts[0]["type"] == "stl"

    def test_get_status_with_workflow(self):
        from automation.workflows.designspec_workflow import WorkflowStatus

        bridge = DesignSpecBridge()
        bridge._workflow = MagicMock()
        bridge._workflow.status = WorkflowStatus.WAITING_INPUT
        bridge._project_dir = Path("/tmp/test")

        status = bridge.get_status()
        assert status["status"] == "waiting_input"
        assert status["project_dir"] == "/tmp/test"


class TestDesignSpecBridgeAsync:
    """Tests for async message sending and pipeline run."""

    @pytest.mark.asyncio
    async def test_send_message_collects_events(self):
        from automation.workflows.designspec_workflow import WorkflowStatus

        bridge = DesignSpecBridge()
        bridge._workflow = MagicMock()
        bridge._workflow.status = WorkflowStatus.WAITING_INPUT

        def fake_on_user_message(text):
            mock_event = MagicMock()
            mock_event.to_dict.return_value = {
                "event_type": "message",
                "data": {},
                "message": f"I'll help with: {text}",
                "timestamp": 0.0,
            }
            bridge._on_workflow_event(mock_event)

        bridge._workflow.on_user_message.side_effect = fake_on_user_message

        result = await bridge.send_message("Create a box domain")

        assert "messages" in result
        assert "events" in result
        assert result["status"] == "waiting_input"
        bridge._workflow.on_user_message.assert_called_once_with("Create a box domain")

    @pytest.mark.asyncio
    async def test_send_message_collects_patch_proposals(self):
        from automation.workflows.designspec_workflow import WorkflowStatus

        bridge = DesignSpecBridge()
        bridge._workflow = MagicMock()
        bridge._workflow.status = WorkflowStatus.WAITING_PATCH_APPROVAL

        def fake_on_user_message(text):
            msg_event = MagicMock()
            msg_event.to_dict.return_value = {
                "event_type": "message",
                "data": {},
                "message": "I'll add a domain",
                "timestamp": 0.0,
            }
            bridge._on_workflow_event(msg_event)

            patch_event = MagicMock()
            patch_event.to_dict.return_value = {
                "event_type": "patch_proposal",
                "data": {
                    "patch_id": "p-1",
                    "description": "Add cylinder domain",
                    "diff": {"domain": {"type": "cylinder"}},
                },
                "message": "Proposed patch",
                "timestamp": 0.0,
            }
            bridge._on_workflow_event(patch_event)

        bridge._workflow.on_user_message.side_effect = fake_on_user_message

        result = await bridge.send_message("Add a cylinder domain")

        assert len(result["patches"]) == 1
        assert result["patches"][0]["patch_id"] == "p-1"
        assert result["status"] == "waiting_patch_approval"

    @pytest.mark.asyncio
    async def test_run_pipeline_delegates_to_workflow(self):
        from automation.workflows.designspec_workflow import WorkflowStatus

        bridge = DesignSpecBridge()
        bridge._workflow = MagicMock()
        bridge._workflow.status = WorkflowStatus.WAITING_INPUT
        bridge._workflow.get_runner_result.return_value = {"success": True}

        result = await bridge.run_pipeline()

        assert result["success"] is True
        bridge._workflow.run_full.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_pipeline_until_stage(self):
        from automation.workflows.designspec_workflow import WorkflowStatus

        bridge = DesignSpecBridge()
        bridge._workflow = MagicMock()
        bridge._workflow.status = WorkflowStatus.WAITING_INPUT
        bridge._workflow.get_runner_result.return_value = {"success": True}

        await bridge.run_pipeline(run_until="component_build")

        bridge._workflow.run_until.assert_called_once_with("component_build")


class TestDesignSpecBridgeCompile:
    """Tests for compilation."""

    def test_compile_spec_delegates_to_workflow(self):
        bridge = DesignSpecBridge()
        bridge._workflow = MagicMock()
        bridge._workflow.get_spec.return_value = {"domain": {}}

        result = bridge.compile_spec()

        assert "events" in result
        assert result["spec"] == {"domain": {}}
        bridge._workflow.compile.assert_called_once()


class TestWorkflowEventTypes:
    """Tests verifying integration with DesignSpecWorkflow event types."""

    def test_all_workflow_event_types_exist(self):
        from automation.workflows.designspec_workflow import WorkflowEventType

        expected_types = [
            "MESSAGE", "QUESTION", "PATCH_PROPOSAL", "PATCH_APPLIED",
            "PATCH_REJECTED", "COMPILE_STARTED", "COMPILE_COMPLETED",
            "RUN_STARTED", "RUN_COMPLETED", "RUN_PROGRESS",
            "RUN_APPROVAL_REQUIRED", "RUN_APPROVED", "RUN_REJECTED",
            "AUTO_FIX_PROPOSED", "SPEC_UPDATED", "VALIDATION_RESULT",
            "ERROR", "STATUS_CHANGE",
        ]
        for etype in expected_types:
            assert hasattr(WorkflowEventType, etype), f"Missing event type: {etype}"

    def test_workflow_event_to_dict(self):
        from automation.workflows.designspec_workflow import (
            WorkflowEvent,
            WorkflowEventType,
        )

        event = WorkflowEvent(
            event_type=WorkflowEventType.MESSAGE,
            data={"key": "value"},
            message="test message",
            timestamp=1234567890.0,
        )

        d = event.to_dict()
        assert d["event_type"] == "message"
        assert d["data"] == {"key": "value"}
        assert d["message"] == "test message"
        assert d["timestamp"] == 1234567890.0

    def test_workflow_status_values(self):
        from automation.workflows.designspec_workflow import WorkflowStatus

        expected_statuses = [
            "IDLE", "INITIALIZING", "WAITING_INPUT", "PROCESSING",
            "WAITING_APPROVAL", "WAITING_PATCH_APPROVAL",
            "WAITING_RUN_APPROVAL", "COMPILING", "RUNNING",
            "COMPLETED", "FAILED", "ERROR", "CANCELLED",
        ]
        for status in expected_statuses:
            assert hasattr(WorkflowStatus, status), f"Missing status: {status}"

    def test_pipeline_stages_list(self):
        from automation.workflows.designspec_workflow import PIPELINE_STAGES

        expected_stages = [
            "compile_policies", "compile_domains", "component_ports",
            "component_build", "component_mesh", "union_voids",
            "mesh_domain", "embed", "port_recarve", "validity", "export",
        ]
        assert PIPELINE_STAGES == expected_stages


class TestDesignSpecRouterEndpoints:
    """Tests for the FastAPI router endpoint definitions."""

    def test_router_import(self):
        assert router is not None
        assert router.prefix == "/api/designspec"

    def test_router_has_expected_routes(self):
        route_paths = [r.path for r in router.routes]
        prefix = router.prefix
        expected_suffixes = [
            "/llm/init",
            "/projects",
            "/projects/load",
            "/message",
            "/patches/{patch_id}/approve",
            "/patches/{patch_id}/reject",
            "/run",
            "/run-until",
            "/compile",
            "/spec",
            "/patches",
            "/status",
            "/artifacts",
            "/events",
        ]
        for suffix in expected_suffixes:
            full_path = prefix + suffix
            assert full_path in route_paths, f"Missing route: {full_path}"

    def test_request_models_validate(self):
        proj = CreateProjectRequest(project_name="test")
        assert proj.project_name == "test"
        assert proj.template_spec is None

        load = LoadProjectRequest(project_dir="/tmp/test")
        assert load.project_dir == "/tmp/test"

        msg = MessageRequest(message="hello")
        assert msg.message == "hello"

        patch_action = PatchActionRequest()
        assert patch_action.reason == ""

        run = RunUntilRequest(stage="validity")
        assert run.stage == "validity"

        llm = LLMConfigRequest(provider="openai", api_key="sk-test")
        assert llm.provider == "openai"
        assert llm.api_key == "sk-test"


class TestGetBridgeSingleton:
    """Tests for the bridge singleton pattern."""

    def test_get_bridge_returns_same_instance(self):
        _ds._bridge_instance = None

        bridge1 = get_bridge()
        bridge2 = get_bridge()
        assert bridge1 is bridge2

        _ds._bridge_instance = None
