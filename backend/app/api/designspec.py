"""
DesignSpec REST API endpoints.

Bridges the web API to AOG's DesignSpecWorkflow for conversation-driven
scaffold design using the full 11-stage DesignSpec pipeline.

Endpoints:
- POST /api/designspec/projects          — Create a new DesignSpec project
- POST /api/designspec/projects/load     — Load an existing project
- POST /api/designspec/message           — Send a user message to the workflow
- POST /api/designspec/patches/{id}/approve — Approve a pending patch
- POST /api/designspec/patches/{id}/reject  — Reject a pending patch
- POST /api/designspec/run               — Run the full pipeline
- POST /api/designspec/run-until         — Run pipeline until a specific stage
- POST /api/designspec/compile           — Manually trigger compilation
- GET  /api/designspec/spec              — Get current DesignSpec JSON
- GET  /api/designspec/patches           — Get pending patches
- GET  /api/designspec/status            — Get workflow status
- GET  /api/designspec/artifacts         — Get generated artifacts
- GET  /api/designspec/events            — Poll for workflow events
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/designspec", tags=["designspec"])


class CreateProjectRequest(BaseModel):
    project_name: str = Field(description="Name for the new project")
    template_spec: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Initial spec template (optional)",
    )


class LoadProjectRequest(BaseModel):
    project_dir: str = Field(description="Path to existing project directory")


class MessageRequest(BaseModel):
    message: str = Field(description="User message text")


class PatchActionRequest(BaseModel):
    reason: str = Field(default="", description="Reason for rejection (optional)")


class RunUntilRequest(BaseModel):
    stage: str = Field(description="Pipeline stage to run until")


class LLMConfigRequest(BaseModel):
    provider: str = Field(description="LLM provider name (e.g. 'anthropic', 'openai')")
    api_key: Optional[str] = Field(default=None, description="API key")
    model: Optional[str] = Field(default=None, description="Model name")
    api_base: Optional[str] = Field(default=None, description="Custom API base URL")


class WorkflowEventResponse(BaseModel):
    event_type: str
    data: Dict[str, Any] = Field(default_factory=dict)
    message: str = ""
    timestamp: float = 0.0


class DesignSpecBridge:
    """
    Bridges the web API to AOG's DesignSpecWorkflow.

    Translates HTTP request/response to the callback-based workflow.
    Uses asyncio.Queue to collect workflow events from the background thread.
    Each project gets its own DesignSpecWorkflow instance.
    """

    def __init__(self) -> None:
        self._workflow = None
        self._event_buffer: List[Dict[str, Any]] = []
        self._project_dir: Optional[Path] = None
        self._llm_initialized: bool = False

    def _on_workflow_event(self, event) -> None:
        """Collect workflow events into the buffer (called from background thread)."""
        self._event_buffer.append(event.to_dict())

    def _ensure_workflow(self) -> None:
        if self._workflow is None:
            raise HTTPException(
                status_code=400,
                detail="No project loaded. Create or load a project first.",
            )

    def initialize_llm(
        self,
        provider: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> bool:
        from automation.llm_client import LLMClient, LLMConfig

        llm_config = LLMConfig(
            provider=provider,
            api_key=api_key,
            model=model or "default",
            api_base=api_base,
        )
        self._llm_client = LLMClient(config=llm_config)
        self._llm_initialized = True
        return True

    def create_project(
        self,
        project_name: str,
        template_spec: Optional[Dict[str, Any]] = None,
    ) -> bool:
        from automation.workflows.designspec_workflow import DesignSpecWorkflow

        project_root = Path(tempfile.mkdtemp(prefix="aog_"))
        self._workflow = DesignSpecWorkflow(
            llm_client=getattr(self, "_llm_client", None),
            event_callback=self._on_workflow_event,
        )
        success = self._workflow.on_start(
            project_root=str(project_root),
            project_name=project_name,
            template_spec=template_spec,
        )
        if success:
            self._project_dir = project_root / project_name
        return success

    def load_project(self, project_dir: str) -> bool:
        from automation.workflows.designspec_workflow import DesignSpecWorkflow

        self._workflow = DesignSpecWorkflow(
            llm_client=getattr(self, "_llm_client", None),
            event_callback=self._on_workflow_event,
        )
        success = self._workflow.on_start(project_dir=project_dir)
        if success:
            self._project_dir = Path(project_dir)
        return success

    async def send_message(self, message: str) -> Dict[str, Any]:
        self._ensure_workflow()
        self._event_buffer.clear()

        self._workflow.on_user_message(message)

        await asyncio.sleep(0.5)
        max_wait = 30.0
        waited = 0.5
        while waited < max_wait:
            from automation.workflows.designspec_workflow import WorkflowStatus
            status = self._workflow.status
            if status in (
                WorkflowStatus.WAITING_INPUT,
                WorkflowStatus.WAITING_PATCH_APPROVAL,
                WorkflowStatus.WAITING_RUN_APPROVAL,
                WorkflowStatus.FAILED,
                WorkflowStatus.ERROR,
            ):
                break
            await asyncio.sleep(0.3)
            waited += 0.3

        events = list(self._event_buffer)
        self._event_buffer.clear()

        assistant_messages = []
        patches = []
        questions = []
        run_request = None
        spec = None

        for event in events:
            etype = event.get("event_type", "")
            emsg = event.get("message", "")
            edata = event.get("data", {})

            if etype == "message" and emsg:
                assistant_messages.append(emsg)
            elif etype == "question":
                questions.append(edata)
                if emsg:
                    assistant_messages.append(emsg)
            elif etype == "patch_proposal":
                patches.append(edata)
            elif etype == "run_approval_required":
                run_request = edata
            elif etype == "spec_updated":
                spec = edata.get("spec")
            elif etype == "error":
                assistant_messages.append(f"Error: {emsg}")

        return {
            "messages": assistant_messages,
            "patches": patches,
            "questions": questions,
            "run_request": run_request,
            "spec": spec,
            "status": self._workflow.status.value,
            "events": events,
        }

    def approve_patch(self, patch_id: str) -> Dict[str, Any]:
        self._ensure_workflow()
        self._event_buffer.clear()
        success = self._workflow.approve_patch(patch_id)
        events = list(self._event_buffer)
        self._event_buffer.clear()
        return {
            "success": success,
            "patch_id": patch_id,
            "spec": self._workflow.get_spec(),
            "events": events,
        }

    def reject_patch(self, patch_id: str, reason: str = "") -> Dict[str, Any]:
        self._ensure_workflow()
        self._event_buffer.clear()
        self._workflow.reject_patch(patch_id, reason)
        events = list(self._event_buffer)
        self._event_buffer.clear()
        return {
            "patch_id": patch_id,
            "reason": reason,
            "events": events,
        }

    async def run_pipeline(self, run_until: Optional[str] = None) -> Dict[str, Any]:
        self._ensure_workflow()
        self._event_buffer.clear()

        if run_until:
            self._workflow.run_until(run_until)
        else:
            self._workflow.run_full()

        max_wait = 300.0
        waited = 0.0
        while waited < max_wait:
            from automation.workflows.designspec_workflow import WorkflowStatus
            status = self._workflow.status
            if status in (
                WorkflowStatus.WAITING_INPUT,
                WorkflowStatus.FAILED,
                WorkflowStatus.ERROR,
                WorkflowStatus.COMPLETED,
            ):
                break
            await asyncio.sleep(1.0)
            waited += 1.0

        events = list(self._event_buffer)
        self._event_buffer.clear()

        result = self._workflow.get_runner_result()
        return {
            "success": result.get("success", False) if result else False,
            "result": result,
            "status": self._workflow.status.value,
            "events": events,
        }

    def compile_spec(self) -> Dict[str, Any]:
        self._ensure_workflow()
        self._event_buffer.clear()
        self._workflow.compile()
        import time as _time
        _time.sleep(1.0)
        events = list(self._event_buffer)
        self._event_buffer.clear()
        return {
            "events": events,
            "spec": self._workflow.get_spec(),
        }

    def get_spec(self) -> Optional[Dict[str, Any]]:
        self._ensure_workflow()
        return self._workflow.get_spec()

    def get_pending_patches(self) -> Dict[str, Dict[str, Any]]:
        self._ensure_workflow()
        return self._workflow.get_pending_patches()

    def get_status(self) -> Dict[str, Any]:
        if self._workflow is None:
            return {"status": "no_project", "project_dir": None}
        return {
            "status": self._workflow.status.value,
            "project_dir": str(self._project_dir) if self._project_dir else None,
            "llm_initialized": self._llm_initialized,
        }

    def get_artifacts(self) -> List[Dict[str, Any]]:
        self._ensure_workflow()
        return self._workflow.get_artifacts()

    def poll_events(self) -> List[Dict[str, Any]]:
        events = list(self._event_buffer)
        self._event_buffer.clear()
        return events


_bridge_instance: Optional[DesignSpecBridge] = None


def get_bridge() -> DesignSpecBridge:
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = DesignSpecBridge()
    return _bridge_instance


@router.post("/llm/init")
async def init_llm(request: LLMConfigRequest) -> Dict[str, Any]:
    bridge = get_bridge()
    try:
        success = bridge.initialize_llm(
            provider=request.provider,
            api_key=request.api_key,
            model=request.model,
            api_base=request.api_base,
        )
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects")
async def create_project(request: CreateProjectRequest) -> Dict[str, Any]:
    bridge = get_bridge()
    try:
        success = bridge.create_project(
            project_name=request.project_name,
            template_spec=request.template_spec,
        )
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create project")
        return {
            "success": True,
            "project_dir": str(bridge._project_dir),
            "spec": bridge.get_spec(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/load")
async def load_project(request: LoadProjectRequest) -> Dict[str, Any]:
    bridge = get_bridge()
    try:
        success = bridge.load_project(request.project_dir)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load project")
        return {
            "success": True,
            "project_dir": str(bridge._project_dir),
            "spec": bridge.get_spec(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/message")
async def send_message(request: MessageRequest) -> Dict[str, Any]:
    bridge = get_bridge()
    try:
        return await bridge.send_message(request.message)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to process message")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/patches/{patch_id}/approve")
async def approve_patch(patch_id: str) -> Dict[str, Any]:
    bridge = get_bridge()
    try:
        return bridge.approve_patch(patch_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/patches/{patch_id}/reject")
async def reject_patch(patch_id: str, request: PatchActionRequest) -> Dict[str, Any]:
    bridge = get_bridge()
    try:
        return bridge.reject_patch(patch_id, request.reason)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run")
async def run_pipeline() -> Dict[str, Any]:
    bridge = get_bridge()
    try:
        return await bridge.run_pipeline()
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Pipeline run failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run-until")
async def run_until(request: RunUntilRequest) -> Dict[str, Any]:
    bridge = get_bridge()
    try:
        return await bridge.run_pipeline(run_until=request.stage)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compile")
async def compile_spec() -> Dict[str, Any]:
    bridge = get_bridge()
    try:
        return bridge.compile_spec()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/spec")
async def get_spec() -> Dict[str, Any]:
    bridge = get_bridge()
    try:
        spec = bridge.get_spec()
        return {"spec": spec}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patches")
async def get_patches() -> Dict[str, Any]:
    bridge = get_bridge()
    try:
        return {"patches": bridge.get_pending_patches()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_status() -> Dict[str, Any]:
    bridge = get_bridge()
    return bridge.get_status()


@router.get("/artifacts")
async def get_artifacts() -> Dict[str, Any]:
    bridge = get_bridge()
    try:
        return {"artifacts": bridge.get_artifacts()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events")
async def poll_events() -> Dict[str, Any]:
    bridge = get_bridge()
    return {"events": bridge.poll_events()}
