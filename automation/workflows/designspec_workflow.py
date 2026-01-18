"""
DesignSpec Workflow for conversation-driven spec editing.

This module provides the DesignSpecWorkflow class that integrates:
- DesignSpecSession for project management
- DesignSpecAgent for conversation handling
- Event-based communication with GUI

The workflow is conversation-first: users describe what they want,
the agent proposes patches, and the runner executes the spec.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union
from pathlib import Path
from enum import Enum
import json
import logging
import threading
import queue
import time

from ..designspec_session import (
    DesignSpecSession,
    ValidationReport,
    CompileReport,
    PatchProposal,
    OperationReport,
)
from ..designspec_agent import (
    DesignSpecAgent,
    AgentResponse,
    AgentResponseType,
    Question,
    RunRequest,
)

logger = logging.getLogger(__name__)


class WorkflowEventType(str, Enum):
    """Types of events emitted by the workflow."""
    MESSAGE = "message"
    QUESTION = "question"
    PATCH_PROPOSAL = "patch_proposal"
    PATCH_APPLIED = "patch_applied"
    PATCH_REJECTED = "patch_rejected"
    COMPILE_STARTED = "compile_started"
    COMPILE_COMPLETED = "compile_completed"
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_PROGRESS = "run_progress"
    SPEC_UPDATED = "spec_updated"
    VALIDATION_RESULT = "validation_result"
    ERROR = "error"
    STATUS_CHANGE = "status_change"


class WorkflowStatus(str, Enum):
    """Status of the workflow."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    WAITING_INPUT = "waiting_input"
    PROCESSING = "processing"
    WAITING_APPROVAL = "waiting_approval"
    COMPILING = "compiling"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkflowEvent:
    """Event emitted by the workflow."""
    event_type: WorkflowEventType
    data: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "data": self.data,
            "message": self.message,
            "timestamp": self.timestamp,
        }


class DesignSpecWorkflow:
    """
    Workflow for conversation-driven DesignSpec editing.
    
    Integrates session management, agent interaction, and GUI communication
    through an event-based interface.
    
    Usage:
        workflow = DesignSpecWorkflow()
        workflow.on_event = lambda event: print(event)
        workflow.on_start(project_dir="/path/to/project")
        workflow.on_user_message("Create a box domain 20mm x 60mm x 30mm")
        workflow.approve_patch(patch_id)
        workflow.run_until("union_voids")
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        event_callback: Optional[Callable[[WorkflowEvent], None]] = None,
    ):
        """
        Initialize the workflow.
        
        Parameters
        ----------
        llm_client : LLMClient, optional
            LLM client for agent natural language understanding
        event_callback : callable, optional
            Callback function for workflow events
        """
        self.llm_client = llm_client
        self._event_callback = event_callback
        
        self._session: Optional[DesignSpecSession] = None
        self._agent: Optional[DesignSpecAgent] = None
        self._status = WorkflowStatus.IDLE
        
        self._pending_patches: Dict[str, PatchProposal] = {}
        self._pending_run_request: Optional[RunRequest] = None
        
        self._event_queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
    
    @property
    def status(self) -> WorkflowStatus:
        """Get current workflow status."""
        return self._status
    
    @property
    def session(self) -> Optional[DesignSpecSession]:
        """Get the current session."""
        return self._session
    
    @property
    def on_event(self) -> Optional[Callable[[WorkflowEvent], None]]:
        """Get the event callback."""
        return self._event_callback
    
    @on_event.setter
    def on_event(self, callback: Callable[[WorkflowEvent], None]) -> None:
        """Set the event callback."""
        self._event_callback = callback
    
    def _emit_event(self, event: WorkflowEvent) -> None:
        """Emit an event to the callback."""
        if self._event_callback:
            try:
                self._event_callback(event)
            except Exception as e:
                logger.exception(f"Event callback failed: {e}")
        self._event_queue.put(event)
    
    def _set_status(self, status: WorkflowStatus) -> None:
        """Set the workflow status and emit event."""
        old_status = self._status
        self._status = status
        self._emit_event(WorkflowEvent(
            event_type=WorkflowEventType.STATUS_CHANGE,
            data={"old_status": old_status.value, "new_status": status.value},
            message=f"Status changed from {old_status.value} to {status.value}",
        ))
    
    def on_start(
        self,
        project_dir: Optional[Union[str, Path]] = None,
        project_root: Optional[Union[str, Path]] = None,
        project_name: Optional[str] = None,
        template_spec: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Start the workflow with a project.
        
        Either provide project_dir to load an existing project,
        or provide project_root + project_name to create a new one.
        
        Parameters
        ----------
        project_dir : str or Path, optional
            Path to existing project directory
        project_root : str or Path, optional
            Parent directory for new project
        project_name : str, optional
            Name for new project
        template_spec : dict, optional
            Initial spec for new project
            
        Returns
        -------
        bool
            True if started successfully
        """
        self._set_status(WorkflowStatus.INITIALIZING)
        
        try:
            if project_dir:
                project_path = Path(project_dir)
                if project_path.exists() and (project_path / "spec.json").exists():
                    self._session = DesignSpecSession.load_project(project_dir)
                    self._emit_event(WorkflowEvent(
                        event_type=WorkflowEventType.MESSAGE,
                        message=f"Loaded project from {project_dir}",
                    ))
                else:
                    project_root_path = project_path.parent
                    project_name_str = project_path.name
                    self._session = DesignSpecSession.create_project(
                        str(project_root_path), project_name_str, template_spec
                    )
                    self._emit_event(WorkflowEvent(
                        event_type=WorkflowEventType.MESSAGE,
                        message=f"Created new project: {project_name_str}",
                    ))
            elif project_root and project_name:
                self._session = DesignSpecSession.create_project(
                    project_root, project_name, template_spec
                )
                self._emit_event(WorkflowEvent(
                    event_type=WorkflowEventType.MESSAGE,
                    message=f"Created new project: {project_name}",
                ))
            else:
                raise ValueError(
                    "Must provide either project_dir or (project_root + project_name)"
                )
            
            self._agent = DesignSpecAgent(llm_client=self.llm_client)
            
            validation = self._session.validate_spec()
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.VALIDATION_RESULT,
                data=validation.to_dict(),
                message="Initial validation complete",
            ))
            
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.SPEC_UPDATED,
                data={"spec": self._session.get_spec()},
                message="Spec loaded",
            ))
            
            self._set_status(WorkflowStatus.WAITING_INPUT)
            
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.MESSAGE,
                message="Ready! Describe what you want to create, or ask questions about the spec.",
            ))
            
            return True
            
        except Exception as e:
            logger.exception("Failed to start workflow")
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.ERROR,
                data={"error": str(e)},
                message=f"Failed to start: {str(e)}",
            ))
            self._set_status(WorkflowStatus.FAILED)
            return False
    
    def on_user_message(self, text: str) -> None:
        """
        Process a user message.
        
        Parameters
        ----------
        text : str
            The user's message
        """
        if not self._session or not self._agent:
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.ERROR,
                message="Workflow not started. Call on_start() first.",
            ))
            return
        
        with self._lock:
            self._set_status(WorkflowStatus.PROCESSING)
            
            try:
                spec = self._session.get_spec()
                validation = self._session.validate_spec()
                compile_report = self._session.get_last_compile_report()
                
                response = self._agent.process_message(
                    user_message=text,
                    spec=spec,
                    validation_report=validation,
                    compile_report=compile_report,
                )
                
                self._handle_agent_response(response)
                
            except Exception as e:
                logger.exception("Failed to process message")
                self._emit_event(WorkflowEvent(
                    event_type=WorkflowEventType.ERROR,
                    data={"error": str(e)},
                    message=f"Error processing message: {str(e)}",
                ))
                self._set_status(WorkflowStatus.WAITING_INPUT)
    
    def _handle_agent_response(self, response: AgentResponse) -> None:
        """Handle an agent response."""
        self._agent.add_assistant_message(response.message)
        
        if response.response_type == AgentResponseType.QUESTION:
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.QUESTION,
                data={
                    "questions": [q.to_dict() for q in response.questions],
                },
                message=response.message,
            ))
            self._set_status(WorkflowStatus.WAITING_INPUT)
        
        elif response.response_type == AgentResponseType.PATCH_PROPOSAL:
            patch = response.patch_proposal
            self._pending_patches[patch.patch_id] = patch
            
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.PATCH_PROPOSAL,
                data={
                    "patch_id": patch.patch_id,
                    "explanation": patch.explanation,
                    "patches": patch.patches,
                    "confidence": patch.confidence,
                    "requires_confirmation": patch.requires_confirmation,
                },
                message=response.message,
            ))
            
            if patch.requires_confirmation:
                self._set_status(WorkflowStatus.WAITING_APPROVAL)
            else:
                self.approve_patch(patch.patch_id)
        
        elif response.response_type == AgentResponseType.RUN_REQUEST:
            run_req = response.run_request
            self._pending_run_request = run_req
            
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.MESSAGE,
                data={
                    "run_request": run_req.to_dict(),
                },
                message=response.message,
            ))
            
            if run_req.full_run:
                self.run_full()
            else:
                self.run_until(run_req.run_until)
        
        elif response.response_type == AgentResponseType.MESSAGE:
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.MESSAGE,
                message=response.message,
            ))
            self._set_status(WorkflowStatus.WAITING_INPUT)
        
        elif response.response_type == AgentResponseType.ERROR:
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.ERROR,
                message=response.message,
            ))
            self._set_status(WorkflowStatus.WAITING_INPUT)
    
    def approve_patch(self, patch_id: str) -> bool:
        """
        Approve and apply a pending patch.
        
        Parameters
        ----------
        patch_id : str
            ID of the patch to approve
            
        Returns
        -------
        bool
            True if patch was applied successfully
        """
        if not self._session:
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.ERROR,
                message="No active session",
            ))
            return False
        
        patch = self._pending_patches.get(patch_id)
        if not patch:
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.ERROR,
                message=f"Patch not found: {patch_id}",
            ))
            return False
        
        self._set_status(WorkflowStatus.COMPILING)
        self._emit_event(WorkflowEvent(
            event_type=WorkflowEventType.COMPILE_STARTED,
            message="Applying patch and compiling...",
        ))
        
        try:
            result = self._session.apply_patch(
                patches=patch.patches,
                author="agent",
                auto_compile=True,
            )
            
            del self._pending_patches[patch_id]
            
            if result.success:
                self._emit_event(WorkflowEvent(
                    event_type=WorkflowEventType.PATCH_APPLIED,
                    data={
                        "patch_id": patch_id,
                        "result": result.to_dict(),
                    },
                    message="Patch applied successfully",
                ))
                
                self._emit_event(WorkflowEvent(
                    event_type=WorkflowEventType.SPEC_UPDATED,
                    data={"spec": self._session.get_spec()},
                    message="Spec updated",
                ))
                
                compile_report = result.metadata.get("compile_report", {})
                self._emit_event(WorkflowEvent(
                    event_type=WorkflowEventType.COMPILE_COMPLETED,
                    data={"compile_report": compile_report},
                    message="Compile completed" if compile_report.get("success") else "Compile had issues",
                ))
                
                self._set_status(WorkflowStatus.WAITING_INPUT)
                return True
            else:
                self._emit_event(WorkflowEvent(
                    event_type=WorkflowEventType.ERROR,
                    data={"errors": result.errors},
                    message=f"Patch failed: {'; '.join(result.errors)}",
                ))
                self._set_status(WorkflowStatus.WAITING_INPUT)
                return False
                
        except Exception as e:
            logger.exception("Failed to apply patch")
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.ERROR,
                data={"error": str(e)},
                message=f"Failed to apply patch: {str(e)}",
            ))
            self._set_status(WorkflowStatus.WAITING_INPUT)
            return False
    
    def reject_patch(self, patch_id: str, reason: str = "") -> None:
        """
        Reject a pending patch.
        
        Parameters
        ----------
        patch_id : str
            ID of the patch to reject
        reason : str, optional
            Reason for rejection
        """
        patch = self._pending_patches.pop(patch_id, None)
        
        if patch:
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.PATCH_REJECTED,
                data={
                    "patch_id": patch_id,
                    "reason": reason,
                },
                message=f"Patch rejected{': ' + reason if reason else ''}",
            ))
        
        self._set_status(WorkflowStatus.WAITING_INPUT)
    
    def run_until(self, stage: str) -> None:
        """
        Run the pipeline until a specific stage.
        
        Parameters
        ----------
        stage : str
            Stage to run until (e.g., "union_voids", "embed", "validity")
        """
        if not self._session:
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.ERROR,
                message="No active session",
            ))
            return
        
        self._set_status(WorkflowStatus.RUNNING)
        self._emit_event(WorkflowEvent(
            event_type=WorkflowEventType.RUN_STARTED,
            data={"run_until": stage},
            message=f"Running pipeline until {stage}...",
        ))
        
        try:
            result = self._session.run(run_until=stage)
            
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.RUN_COMPLETED,
                data={"result": result},
                message="Run completed" if result.get("success") else "Run failed",
            ))
            
            self._set_status(WorkflowStatus.WAITING_INPUT)
            
        except Exception as e:
            logger.exception("Run failed")
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.ERROR,
                data={"error": str(e)},
                message=f"Run failed: {str(e)}",
            ))
            self._set_status(WorkflowStatus.WAITING_INPUT)
    
    def run_full(self) -> None:
        """Run the full pipeline."""
        if not self._session:
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.ERROR,
                message="No active session",
            ))
            return
        
        self._set_status(WorkflowStatus.RUNNING)
        self._emit_event(WorkflowEvent(
            event_type=WorkflowEventType.RUN_STARTED,
            data={"full_run": True},
            message="Running full pipeline...",
        ))
        
        try:
            result = self._session.run()
            
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.RUN_COMPLETED,
                data={"result": result},
                message="Full run completed" if result.get("success") else "Full run failed",
            ))
            
            self._set_status(WorkflowStatus.WAITING_INPUT)
            
        except Exception as e:
            logger.exception("Full run failed")
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.ERROR,
                data={"error": str(e)},
                message=f"Full run failed: {str(e)}",
            ))
            self._set_status(WorkflowStatus.WAITING_INPUT)
    
    def get_spec(self) -> Optional[Dict[str, Any]]:
        """Get the current spec."""
        if self._session:
            return self._session.get_spec()
        return None
    
    def get_validation_report(self) -> Optional[ValidationReport]:
        """Get the latest validation report."""
        if self._session:
            return self._session.validate_spec()
        return None
    
    def get_compile_report(self) -> Optional[Dict[str, Any]]:
        """Get the latest compile report."""
        if self._session:
            return self._session.get_last_compile_report()
        return None
    
    def get_runner_result(self) -> Optional[Dict[str, Any]]:
        """Get the latest runner result."""
        if self._session:
            return self._session.get_last_runner_result()
        return None
    
    def get_artifacts(self) -> List[Dict[str, Any]]:
        """Get list of generated artifacts."""
        if self._session:
            return self._session.get_artifacts()
        return []
    
    def get_pending_patches(self) -> Dict[str, Dict[str, Any]]:
        """Get pending patches awaiting approval."""
        return {
            pid: patch.to_dict()
            for pid, patch in self._pending_patches.items()
        }
    
    def get_compile_status(self) -> Optional[CompileReport]:
        """Get the last compile status."""
        if self._session:
            return self._session.get_last_compile_report()
        return None
    
    def _propose_patch(
        self,
        patches: List[Dict[str, Any]],
        explanation: str,
        confidence: float = 0.8,
    ) -> str:
        """
        Propose a patch for testing purposes.
        
        Parameters
        ----------
        patches : list of dict
            JSON Patch operations
        explanation : str
            Explanation of the patch
        confidence : float
            Confidence level (0-1)
            
        Returns
        -------
        str
            The patch ID
        """
        from automation.designspec_agent import PatchProposal
        
        patch_id = f"patch_{len(self._pending_patches) + 1}"
        
        proposal = PatchProposal(
            explanation=explanation,
            patches=patches,
            confidence=confidence,
            requires_confirmation=True,
        )
        
        self._pending_patches[patch_id] = proposal
        
        self._emit_event(WorkflowEvent(
            event_type=WorkflowEventType.PATCH_PROPOSAL,
            data={
                "patch_id": patch_id,
                "explanation": explanation,
                "patches": patches,
                "confidence": confidence,
            },
            message=f"Proposed patch: {explanation}",
        ))
        
        return patch_id
    
    def compile(self) -> Optional[CompileReport]:
        """Manually trigger compilation."""
        if not self._session:
            return None
        
        self._set_status(WorkflowStatus.COMPILING)
        self._emit_event(WorkflowEvent(
            event_type=WorkflowEventType.COMPILE_STARTED,
            message="Compiling...",
        ))
        
        try:
            report = self._session.compile()
            
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.COMPILE_COMPLETED,
                data={"compile_report": report.to_dict()},
                message="Compile completed" if report.success else "Compile failed",
            ))
            
            self._set_status(WorkflowStatus.WAITING_INPUT)
            return report
            
        except Exception as e:
            logger.exception("Compile failed")
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.ERROR,
                data={"error": str(e)},
                message=f"Compile failed: {str(e)}",
            ))
            self._set_status(WorkflowStatus.WAITING_INPUT)
            return None
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        if self._agent:
            return self._agent.get_conversation_history()
        return []
    
    def poll_events(self, timeout: float = 0.1) -> List[WorkflowEvent]:
        """
        Poll for pending events.
        
        Parameters
        ----------
        timeout : float
            Timeout in seconds
            
        Returns
        -------
        list of WorkflowEvent
            Pending events
        """
        events = []
        try:
            while True:
                event = self._event_queue.get(timeout=timeout)
                events.append(event)
        except queue.Empty:
            pass
        return events


__all__ = [
    "DesignSpecWorkflow",
    "WorkflowEvent",
    "WorkflowEventType",
    "WorkflowStatus",
]
