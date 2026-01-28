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
import traceback

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
from ..designspec_llm import (
    DesignSpecLLMAgent,
    create_llm_agent,
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
    RUN_APPROVAL_REQUIRED = "run_approval_required"
    RUN_APPROVED = "run_approved"
    RUN_REJECTED = "run_rejected"
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
    WAITING_PATCH_APPROVAL = "waiting_patch_approval"
    WAITING_RUN_APPROVAL = "waiting_run_approval"
    COMPILING = "compiling"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ERROR = "error"
    CANCELLED = "cancelled"


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


PIPELINE_STAGES = [
    "compile_policies",
    "compile_domains",
    "component_ports",
    "component_build",
    "component_mesh",
    "union_voids",
    "mesh_domain",
    "embed",
    "port_recarve",
    "validity",
    "export",
]


class AsyncDesignSpecRunner:
    """
    Async wrapper for DesignSpec pipeline execution.
    
    Runs the pipeline in a background thread with:
    - Progress callbacks via RUN_PROGRESS events
    - Cancellation support via threading.Event
    - Proper error handling and logging
    """
    
    def __init__(
        self,
        session: "DesignSpecSession",
        event_callback: Callable[[WorkflowEvent], None],
    ):
        """
        Initialize the async runner.
        
        Parameters
        ----------
        session : DesignSpecSession
            The active design spec session
        event_callback : callable
            Callback for emitting workflow events
        """
        self._session = session
        self._event_callback = event_callback
        self._cancel_flag = threading.Event()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._result: Optional[Dict[str, Any]] = None
        self._error: Optional[str] = None
        self._start_time: Optional[float] = None
    
    @property
    def is_running(self) -> bool:
        """Check if the runner is currently executing."""
        return self._running
    
    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._cancel_flag.is_set()
    
    def cancel(self) -> None:
        """Request cancellation of the running pipeline."""
        self._cancel_flag.set()
        logger.info("Pipeline cancellation requested")
    
    def _emit_progress(
        self,
        stage: str,
        stage_index: int,
        total_stages: int,
        status: str = "running",
        message: str = "",
        elapsed_time: float = 0.0,
    ) -> None:
        """Emit a progress event."""
        progress_pct = (stage_index / total_stages) * 100 if total_stages > 0 else 0
        
        event = WorkflowEvent(
            event_type=WorkflowEventType.RUN_PROGRESS,
            data={
                "stage": stage,
                "stage_index": stage_index,
                "total_stages": total_stages,
                "progress_percent": progress_pct,
                "status": status,
                "elapsed_time": elapsed_time,
                "estimated_remaining": self._estimate_remaining(stage_index, total_stages, elapsed_time),
            },
            message=message or f"Stage {stage_index + 1}/{total_stages}: {stage}",
        )
        
        if self._event_callback:
            try:
                self._event_callback(event)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    def _estimate_remaining(
        self,
        stage_index: int,
        total_stages: int,
        elapsed_time: float,
    ) -> float:
        """Estimate remaining time based on elapsed time and progress."""
        if stage_index <= 0 or elapsed_time <= 0:
            return 0.0
        
        avg_time_per_stage = elapsed_time / stage_index
        remaining_stages = total_stages - stage_index
        return avg_time_per_stage * remaining_stages
    
    def run_async(
        self,
        run_until: Optional[str] = None,
        full_run: bool = False,
    ) -> None:
        """
        Start the pipeline execution in a background thread.
        
        Parameters
        ----------
        run_until : str, optional
            Stage to run until (if not full_run)
        full_run : bool
            If True, run all stages
        """
        if self._running:
            logger.warning("Runner is already executing")
            return
        
        self._cancel_flag.clear()
        self._result = None
        self._error = None
        self._running = True
        self._start_time = time.time()
        
        self._thread = threading.Thread(
            target=self._run_pipeline,
            args=(run_until, full_run),
            daemon=True,
        )
        self._thread.start()
    
    def _run_pipeline(
        self,
        run_until: Optional[str],
        full_run: bool,
    ) -> None:
        """
        Execute the pipeline (runs in background thread).
        
        Parameters
        ----------
        run_until : str, optional
            Stage to run until
        full_run : bool
            If True, run all stages
        """
        try:
            stages_to_run = PIPELINE_STAGES.copy()
            
            if not full_run and run_until:
                try:
                    stop_index = stages_to_run.index(run_until) + 1
                    stages_to_run = stages_to_run[:stop_index]
                except ValueError:
                    logger.warning(f"Unknown stage '{run_until}', running all stages")
            
            total_stages = len(stages_to_run)
            
            if self._cancel_flag.is_set():
                logger.info("Pipeline cancelled before starting")
                self._emit_progress(
                    stage="cancelled",
                    stage_index=0,
                    total_stages=total_stages,
                    status="cancelled",
                    message="Pipeline cancelled before starting",
                    elapsed_time=time.time() - self._start_time,
                )
                self._error = "Pipeline cancelled by user"
                return
            
            self._emit_progress(
                stage=stages_to_run[0] if stages_to_run else "starting",
                stage_index=0,
                total_stages=total_stages,
                status="running",
                message=f"Starting pipeline ({total_stages} stages)...",
                elapsed_time=time.time() - self._start_time,
            )
            
            if full_run:
                self._result = self._session.run()
            else:
                self._result = self._session.run(run_until=run_until)
            
            if self._cancel_flag.is_set():
                self._emit_progress(
                    stage="cancelled",
                    stage_index=total_stages,
                    total_stages=total_stages,
                    status="cancelled",
                    message="Pipeline cancelled",
                    elapsed_time=time.time() - self._start_time,
                )
                return
            
            success = self._result.get("success", False) if self._result else False
            
            self._emit_progress(
                stage=stages_to_run[-1] if stages_to_run else "complete",
                stage_index=total_stages,
                total_stages=total_stages,
                status="completed" if success else "failed",
                message="Pipeline completed" if success else "Pipeline completed with errors",
                elapsed_time=time.time() - self._start_time,
            )
            
        except Exception as e:
            logger.exception("Pipeline execution failed")
            self._error = str(e)
            self._emit_progress(
                stage="error",
                stage_index=0,
                total_stages=1,
                status="failed",
                message=f"Pipeline failed: {str(e)}",
                elapsed_time=time.time() - self._start_time if self._start_time else 0,
            )
        finally:
            self._running = False
    
    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the pipeline to complete.
        
        Parameters
        ----------
        timeout : float, optional
            Maximum time to wait in seconds
            
        Returns
        -------
        bool
            True if completed, False if timed out
        """
        if self._thread:
            self._thread.join(timeout=timeout)
            return not self._thread.is_alive()
        return True
    
    def get_result(self) -> Optional[Dict[str, Any]]:
        """Get the pipeline result (None if not completed or failed)."""
        return self._result
    
    def get_error(self) -> Optional[str]:
        """Get the error message if pipeline failed."""
        return self._error


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
        use_legacy_agent: bool = False,
    ):
        """
        Initialize the workflow.
        
        Parameters
        ----------
        llm_client : LLMClient, optional
            LLM client for agent natural language understanding
        event_callback : callable, optional
            Callback function for workflow events
        use_legacy_agent : bool
            If True, use the legacy rule-based agent instead of the LLM-first agent.
            Default is False (use LLM-first agent).
        """
        self.llm_client = llm_client
        self._event_callback = event_callback
        self._use_legacy_agent = use_legacy_agent
        
        self._session: Optional[DesignSpecSession] = None
        self._agent: Optional[DesignSpecAgent] = None
        self._llm_agent: Optional[DesignSpecLLMAgent] = None
        self._status = WorkflowStatus.IDLE
        
        self._pending_patches: Dict[str, PatchProposal] = {}
        self._pending_run_request: Optional[RunRequest] = None
        
        self._event_queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        
        self._async_runner: Optional[AsyncDesignSpecRunner] = None
    
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
            
            if self._use_legacy_agent:
                self._agent = DesignSpecAgent(llm_client=self.llm_client)
                self._llm_agent = None
            else:
                self._agent = DesignSpecAgent(llm_client=None)
                self._llm_agent = create_llm_agent(
                    session=self._session,
                    llm_client=self.llm_client,
                    use_legacy_fallback=True,
                )
            
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
        if not self._session:
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.ERROR,
                message="Workflow not started. Call on_start() first.",
            ))
            return
        
        if not self._agent and not self._llm_agent:
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.ERROR,
                message="No agent available. Call on_start() first.",
            ))
            return
        
        with self._lock:
            self._set_status(WorkflowStatus.PROCESSING)
            
            try:
                if self._handle_local_command(text):
                    self._set_status(WorkflowStatus.WAITING_INPUT)
                    return
                
                spec = self._session.get_spec()
                validation = self._session.validate_spec()
                compile_report = self._session.get_last_compile_report()
                
                if self._llm_agent and not self._use_legacy_agent:
                    response = self._llm_agent.process_message(
                        user_message=text,
                        validation_report=validation,
                        compile_report=compile_report,
                    )
                else:
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
    
    def _handle_local_command(self, text: str) -> bool:
        """
        Handle local diagnostic commands before passing to agent.
        
        Detects and handles commands like:
        - "show spec", "print spec", "output current design spec"
        - "what is missing", "what is left", "status"
        - "why did it fail", "what is the issue", "show last error"
        
        Parameters
        ----------
        text : str
            The user's message
            
        Returns
        -------
        bool
            True if the message was handled as a command, False otherwise
        """
        text_lower = text.lower().strip()
        
        show_spec_patterns = [
            "show spec", "print spec", "output current design spec",
            "display spec", "show the spec", "show current spec",
            "what is the spec", "what's the spec",
        ]
        if any(pattern in text_lower for pattern in show_spec_patterns):
            self._handle_show_spec_command()
            return True
        
        missing_patterns = [
            "what is missing", "what's missing", "what is left",
            "what's left", "status", "show status", "missing fields",
            "what do i need", "what else is needed",
        ]
        if any(pattern in text_lower for pattern in missing_patterns):
            self._handle_missing_fields_command()
            return True
        
        error_patterns = [
            "why did it fail", "what is the issue", "show last error",
            "what went wrong", "show error", "last error",
            "what is the error", "what's the error", "show errors",
        ]
        if any(pattern in text_lower for pattern in error_patterns):
            self._handle_last_error_command()
            return True
        
        # Handle "run" command - provides confirmation feedback
        if text_lower == "run":
            self._handle_run_command()
            return True
        
        # Handle "cancel" / "stop" command - cancels running process
        cancel_patterns = ["cancel", "stop", "abort"]
        if text_lower in cancel_patterns:
            self._handle_cancel_command()
            return True
        
        defaults_patterns = [
            "explain defaults", "show defaults", "what are the defaults",
            "default values", "list defaults", "defaults",
            "what defaults", "show default values", "policy defaults",
        ]
        if any(pattern in text_lower for pattern in defaults_patterns):
            self._handle_explain_defaults_command()
            return True
        
        return False
    
    def _handle_show_spec_command(self) -> None:
        """Handle the 'show spec' command."""
        spec = self._session.get_spec()
        spec_path = self._session.project_dir / "spec.json"
        
        spec_json = json.dumps(spec, indent=2)
        if len(spec_json) > 2000:
            spec_json = spec_json[:2000] + "\n... (truncated, see full spec at path below)"
        
        message = f"Current spec:\n```json\n{spec_json}\n```\n\nSpec file: {spec_path}"
        
        self._emit_event(WorkflowEvent(
            event_type=WorkflowEventType.MESSAGE,
            data={"spec": spec, "spec_path": str(spec_path)},
            message=message,
        ))
    
    def _handle_missing_fields_command(self) -> None:
        """Handle the 'what is missing' / 'status' command."""
        spec = self._session.get_spec()
        validation = self._session.validate_spec()
        
        checklist = self._agent._build_missing_fields_checklist(spec)
        
        status_parts = []
        
        if validation.valid:
            status_parts.append("Spec validation: PASSED")
        else:
            status_parts.append("Spec validation: FAILED")
            if validation.errors:
                status_parts.append("Errors:")
                for error in validation.errors[:5]:
                    status_parts.append(f"  - {error}")
        
        if validation.warnings:
            status_parts.append("Warnings:")
            for warning in validation.warnings[:3]:
                status_parts.append(f"  - {warning}")
        
        status_parts.append("\nSpec checklist:")
        status_parts.append(checklist)
        
        message = "\n".join(status_parts)
        
        self._emit_event(WorkflowEvent(
            event_type=WorkflowEventType.MESSAGE,
            data={
                "validation": validation.to_dict(),
                "checklist": checklist,
            },
            message=message,
        ))
    
    def _handle_last_error_command(self) -> None:
        """Handle the 'show last error' / 'why did it fail' command."""
        runner_result = self._session.get_last_runner_result()
        
        if not runner_result:
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.MESSAGE,
                message="No previous run found. Use 'run' to execute the pipeline first.",
            ))
            return
        
        success = runner_result.get("success", False)
        errors = runner_result.get("errors", [])
        stage_reports = runner_result.get("stage_reports", [])
        output_dir = runner_result.get("output_dir", "")
        run_id = runner_result.get("run_id", "")
        
        message_parts = []
        
        if success:
            message_parts.append("Last run: SUCCESS")
        else:
            message_parts.append("Last run: FAILED")
        
        if errors:
            message_parts.append("\nErrors (first 3):")
            for error in errors[:3]:
                message_parts.append(f"  - {error}")
        
        failing_stage = None
        for report in stage_reports:
            if isinstance(report, dict) and not report.get("success", True):
                failing_stage = report.get("stage", "unknown")
                stage_errors = report.get("errors", [])
                if stage_errors:
                    message_parts.append(f"\nFailing stage: {failing_stage}")
                    message_parts.append("Stage errors:")
                    for err in stage_errors[:3]:
                        message_parts.append(f"  - {err}")
                break
        
        if output_dir:
            message_parts.append(f"\nOutput directory: {output_dir}")
        
        if run_id:
            message_parts.append(f"Run ID: {run_id}")
        
        message = "\n".join(message_parts)
        
        self._emit_event(WorkflowEvent(
            event_type=WorkflowEventType.MESSAGE,
            data={
                "success": success,
                "errors": errors[:3] if errors else [],
                "failing_stage": failing_stage,
                "output_dir": output_dir,
                "run_id": run_id,
            },
            message=message,
        ))
    
    def _handle_run_command(self) -> None:
        """Handle the 'run' command - starts full pipeline with confirmation."""
        if self._status == WorkflowStatus.RUNNING:
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.MESSAGE,
                message="A run is already in progress. Type 'cancel' to stop it.",
            ))
            return
        
        # Emit confirmation that run is starting
        self._emit_event(WorkflowEvent(
            event_type=WorkflowEventType.MESSAGE,
            message="Starting full run... Type 'cancel' to stop.",
        ))
        
        # Start the full run
        self.run_full()
    
    def _handle_cancel_command(self) -> None:
        """Handle the 'cancel' / 'stop' command - cancels running process."""
        if not self.is_run_in_progress() and self._status != WorkflowStatus.RUNNING:
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.MESSAGE,
                message="No run is currently in progress.",
            ))
            return
        
        if self.cancel_run():
            return
        
        self._set_status(WorkflowStatus.CANCELLED)
        
        self._emit_event(WorkflowEvent(
            event_type=WorkflowEventType.MESSAGE,
            message="Run cancelled. The current operation will stop at the next safe point.",
        ))
    
    def _handle_explain_defaults_command(self) -> None:
        """
        Handle the 'explain defaults' command.
        
        Shows a table of policy parameters with current values, default values,
        units, and descriptions.
        """
        try:
            from designspec.preflight import DEFAULT_POLICIES
            
            spec = self._session.get_spec() if self._session else {}
            current_policies = spec.get("policies", {})
            
            lines = ["## Policy Defaults Comparison", ""]
            lines.append("The following table shows current spec values vs. recommended defaults:")
            lines.append("")
            
            key_policies = [
                ("mesh_merge", "voxel_pitch", "m", "Voxel size for mesh union operations"),
                ("mesh_merge", "keep_largest_component", "", "Keep only largest mesh component"),
                ("embedding", "voxel_pitch", "m", "Voxel size for embedding operations"),
                ("embedding", "shell_thickness", "m", "Minimum wall thickness"),
                ("growth", "min_segment_length", "m", "Minimum vessel segment length"),
                ("growth", "max_segment_length", "m", "Maximum vessel segment length"),
                ("growth", "min_radius", "m", "Minimum vessel radius"),
                ("growth", "step_size", "m", "Growth step size"),
                ("collision", "collision_clearance", "m", "Minimum clearance between vessels"),
                ("radius", "min_radius", "m", "Minimum vessel radius"),
                ("radius", "max_radius", "m", "Maximum vessel radius"),
                ("resolution", "min_channel_diameter", "m", "Minimum channel diameter"),
                ("resolution", "min_pitch", "m", "Minimum voxel pitch"),
                ("resolution", "max_pitch", "m", "Maximum voxel pitch"),
            ]
            
            lines.append("| Policy | Parameter | Current | Default | Units | Description |")
            lines.append("|--------|-----------|---------|---------|-------|-------------|")
            
            for policy_name, param_name, units, description in key_policies:
                default_policy = DEFAULT_POLICIES.get(policy_name, {})
                default_value = default_policy.get(param_name, "N/A")
                
                current_policy = current_policies.get(policy_name, {})
                if isinstance(current_policy, dict):
                    current_value = current_policy.get(param_name, "not set")
                else:
                    current_value = "not set"
                
                if isinstance(default_value, float):
                    default_str = f"{default_value:.6f}"
                else:
                    default_str = str(default_value)
                
                if isinstance(current_value, float):
                    current_str = f"{current_value:.6f}"
                elif current_value == "not set":
                    current_str = "*not set*"
                else:
                    current_str = str(current_value)
                
                lines.append(
                    f"| {policy_name} | {param_name} | {current_str} | {default_str} | {units} | {description} |"
                )
            
            lines.append("")
            lines.append("### Scale-Appropriate Recommendations")
            lines.append("")
            
            domain_scale = self._get_domain_scale_from_spec(spec)
            if domain_scale:
                recommended_pitch = domain_scale / 100
                lines.append(f"Based on your domain scale (~{domain_scale:.4f}m):")
                lines.append(f"- Recommended voxel_pitch: {recommended_pitch:.6f}m (domain_scale / 100)")
                lines.append(f"- Recommended min_segment_length: {domain_scale / 50:.6f}m")
                lines.append(f"- Recommended collision_clearance: {domain_scale / 100:.6f}m")
            else:
                lines.append("No domain defined yet. Define a domain to get scale-appropriate recommendations.")
            
            message = "\n".join(lines)
            
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.MESSAGE,
                data={
                    "defaults": {k: v for k, v in DEFAULT_POLICIES.items()},
                    "current_policies": current_policies,
                },
                message=message,
            ))
            
        except Exception as e:
            logger.exception("Failed to explain defaults")
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.ERROR,
                data={"error": str(e)},
                message=f"Failed to explain defaults: {str(e)}",
            ))
    
    def _get_domain_scale_from_spec(self, spec: dict) -> float:
        """
        Get the approximate scale of the domain in meters.
        
        Parameters
        ----------
        spec : dict
            The spec dictionary
            
        Returns
        -------
        float
            Domain scale in meters, or 0.0 if no domain defined
        """
        domains = spec.get("domains", {})
        if not domains:
            return 0.0
        
        domain = next(iter(domains.values()))
        if not isinstance(domain, dict):
            return 0.0
        
        domain_type = domain.get("type", "").lower()
        
        if domain_type == "cylinder":
            radius = domain.get("radius", 0.0)
            height = domain.get("height", 0.0)
            return max(radius * 2, height)
        elif domain_type == "box":
            x_size = abs(domain.get("x_max", 0.0) - domain.get("x_min", 0.0))
            y_size = abs(domain.get("y_max", 0.0) - domain.get("y_min", 0.0))
            z_size = abs(domain.get("z_max", 0.0) - domain.get("z_min", 0.0))
            return max(x_size, y_size, z_size)
        elif domain_type == "ellipsoid":
            radii = domain.get("radii", [0.0, 0.0, 0.0])
            if isinstance(radii, list) and len(radii) >= 3:
                return max(radii) * 2
            return 0.0
        
        return 0.0
    
    def _apply_validation_aware_fixes(self, patches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply validation-aware fixes to patches before they are applied.
        
        This method checks for validation warnings (like PITCH_TOO_LARGE) and
        automatically adjusts patch values to resolve them.
        
        Parameters
        ----------
        patches : list of dict
            The original JSON Patch operations
            
        Returns
        -------
        list of dict
            The adjusted patches with validation-aware fixes
        """
        if not self._session:
            return patches
        
        try:
            from designspec.preflight import run_preflight_checks, _get_domain_scale
            
            spec = self._session.get_spec()
            if not spec:
                return patches
            
            preflight_result = run_preflight_checks(spec)
            domain_scale = _get_domain_scale(spec)
            
            if domain_scale <= 0:
                return patches
            
            pitch_warnings = [
                w for w in preflight_result.warnings
                if w.code == "PITCH_TOO_LARGE"
            ]
            
            if not pitch_warnings:
                return patches
            
            recommended_pitch = domain_scale / 100
            
            adjusted_patches = []
            pitch_paths_to_fix = set()
            
            for warning in pitch_warnings:
                if warning.path:
                    pitch_paths_to_fix.add(warning.path)
            
            for patch in patches:
                adjusted_patch = dict(patch)
                path = patch.get("path", "")
                op = patch.get("op", "")
                
                if op in ("add", "replace") and "voxel_pitch" in path:
                    current_value = patch.get("value")
                    if isinstance(current_value, (int, float)):
                        if current_value > recommended_pitch * 2:
                            adjusted_patch["value"] = recommended_pitch
                            logger.info(
                                f"Validation-aware fix: Adjusted voxel_pitch from "
                                f"{current_value} to {recommended_pitch} at {path}"
                            )
                
                adjusted_patches.append(adjusted_patch)
            
            for pitch_path in pitch_paths_to_fix:
                path_already_patched = any(
                    p.get("path", "").startswith(pitch_path.rsplit(".", 1)[0])
                    and "voxel_pitch" in p.get("path", "")
                    for p in adjusted_patches
                )
                
                if not path_already_patched:
                    parts = pitch_path.split(".")
                    if len(parts) >= 2:
                        policy_name = parts[1]
                        fix_path = f"/policies/{policy_name}/voxel_pitch"
                        
                        already_has_fix = any(
                            p.get("path") == fix_path for p in adjusted_patches
                        )
                        
                        if not already_has_fix:
                            adjusted_patches.append({
                                "op": "replace",
                                "path": fix_path,
                                "value": recommended_pitch,
                            })
                            logger.info(
                                f"Validation-aware fix: Added patch to fix "
                                f"PITCH_TOO_LARGE at {fix_path}"
                            )
            
            return adjusted_patches
            
        except Exception as e:
            logger.warning(f"Failed to apply validation-aware fixes: {e}")
            return patches
    
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
                event_type=WorkflowEventType.RUN_APPROVAL_REQUIRED,
                data={
                    "run_request": run_req.to_dict(),
                    "run_until": run_req.run_until,
                    "full_run": run_req.full_run,
                    "reason": run_req.reason,
                    "expected_signal": getattr(run_req, "expected_signal", ""),
                },
                message=response.message,
            ))
            self._set_status(WorkflowStatus.WAITING_RUN_APPROVAL)
        
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
            adjusted_patches = self._apply_validation_aware_fixes(patch.patches)
            
            result = self._session.apply_patch(
                patches=adjusted_patches,
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
    
    def approve_pending_run(self) -> bool:
        """
        Approve and execute the pending run request.
        
        Returns
        -------
        bool
            True if run was executed successfully
        """
        if not self._pending_run_request:
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.ERROR,
                message="No pending run request to approve",
            ))
            return False
        
        if self._status != WorkflowStatus.WAITING_RUN_APPROVAL:
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.ERROR,
                message=f"Cannot approve run in status: {self._status.value}",
            ))
            return False
        
        run_req = self._pending_run_request
        self._pending_run_request = None
        
        self._emit_event(WorkflowEvent(
            event_type=WorkflowEventType.RUN_APPROVED,
            data={
                "run_until": run_req.run_until,
                "full_run": run_req.full_run,
                "reason": run_req.reason,
            },
            message="Run approved by user",
        ))
        
        if run_req.full_run:
            self.run_full()
        else:
            self.run_until(run_req.run_until)
        
        return True
    
    def reject_pending_run(self, reason: str = "") -> None:
        """
        Reject the pending run request.
        
        Parameters
        ----------
        reason : str, optional
            Reason for rejection
        """
        if not self._pending_run_request:
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.ERROR,
                message="No pending run request to reject",
            ))
            return
        
        run_req = self._pending_run_request
        self._pending_run_request = None
        
        self._emit_event(WorkflowEvent(
            event_type=WorkflowEventType.RUN_REJECTED,
            data={
                "run_until": run_req.run_until,
                "full_run": run_req.full_run,
                "reason": reason,
            },
            message=f"Run rejected{': ' + reason if reason else ''}",
        ))
        
        self._set_status(WorkflowStatus.WAITING_INPUT)
    
    def get_pending_run_request(self) -> Optional[RunRequest]:
        """Get the pending run request if any."""
        return self._pending_run_request
    
    def run_until(self, stage: str, async_mode: bool = True) -> None:
        """
        Run the pipeline until a specific stage.
        
        Parameters
        ----------
        stage : str
            Stage to run until (e.g., "union_voids", "embed", "validity")
        async_mode : bool
            If True, run in background thread (default). If False, run synchronously.
        """
        if not self._session:
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.ERROR,
                message="No active session",
            ))
            return
        
        if self._async_runner and self._async_runner.is_running:
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.MESSAGE,
                message="A run is already in progress. Type 'cancel' to stop it.",
            ))
            return
        
        self._set_status(WorkflowStatus.RUNNING)
        self._emit_event(WorkflowEvent(
            event_type=WorkflowEventType.RUN_STARTED,
            data={"run_until": stage},
            message=f"Running pipeline until {stage}...",
        ))
        
        if async_mode:
            self._async_runner = AsyncDesignSpecRunner(
                session=self._session,
                event_callback=self._on_async_runner_event,
            )
            self._async_runner.run_async(run_until=stage, full_run=False)
        else:
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
    
    def run_full(self, async_mode: bool = True) -> None:
        """
        Run the full pipeline.
        
        Parameters
        ----------
        async_mode : bool
            If True, run in background thread (default). If False, run synchronously.
        """
        if not self._session:
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.ERROR,
                message="No active session",
            ))
            return
        
        if self._async_runner and self._async_runner.is_running:
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.MESSAGE,
                message="A run is already in progress. Type 'cancel' to stop it.",
            ))
            return
        
        self._set_status(WorkflowStatus.RUNNING)
        self._emit_event(WorkflowEvent(
            event_type=WorkflowEventType.RUN_STARTED,
            data={"full_run": True},
            message="Running full pipeline...",
        ))
        
        if async_mode:
            self._async_runner = AsyncDesignSpecRunner(
                session=self._session,
                event_callback=self._on_async_runner_event,
            )
            self._async_runner.run_async(full_run=True)
        else:
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
    
    def _on_async_runner_event(self, event: WorkflowEvent) -> None:
        """
        Handle events from the async runner.
        
        Parameters
        ----------
        event : WorkflowEvent
            Event from the async runner
        """
        self._emit_event(event)
        
        if event.event_type == WorkflowEventType.RUN_PROGRESS:
            status = event.data.get("status", "")
            if status in ("completed", "failed", "cancelled"):
                if status == "completed":
                    result = self._async_runner.get_result() if self._async_runner else None
                    self._emit_event(WorkflowEvent(
                        event_type=WorkflowEventType.RUN_COMPLETED,
                        data={"result": result},
                        message="Run completed" if result and result.get("success") else "Run completed with issues",
                    ))
                    self._set_status(WorkflowStatus.WAITING_INPUT)
                elif status == "failed":
                    error = self._async_runner.get_error() if self._async_runner else "Unknown error"
                    self._emit_event(WorkflowEvent(
                        event_type=WorkflowEventType.ERROR,
                        data={"error": error},
                        message=f"Run failed: {error}",
                    ))
                    self._set_status(WorkflowStatus.FAILED)
                elif status == "cancelled":
                    self._emit_event(WorkflowEvent(
                        event_type=WorkflowEventType.MESSAGE,
                        message="Run cancelled by user",
                    ))
                    self._set_status(WorkflowStatus.CANCELLED)
    
    def cancel_run(self) -> bool:
        """
        Cancel the currently running pipeline.
        
        Returns
        -------
        bool
            True if cancellation was requested, False if no run in progress
        """
        if self._async_runner and self._async_runner.is_running:
            self._async_runner.cancel()
            self._emit_event(WorkflowEvent(
                event_type=WorkflowEventType.MESSAGE,
                message="Cancellation requested. Waiting for current stage to complete...",
            ))
            return True
        return False
    
    def is_run_in_progress(self) -> bool:
        """Check if a pipeline run is currently in progress."""
        return self._async_runner is not None and self._async_runner.is_running
    
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
        if self._llm_agent and not self._use_legacy_agent:
            return self._llm_agent.get_conversation_history()
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
    "AsyncDesignSpecRunner",
    "DesignSpecWorkflow",
    "PIPELINE_STAGES",
    "WorkflowEvent",
    "WorkflowEventType",
    "WorkflowStatus",
]
