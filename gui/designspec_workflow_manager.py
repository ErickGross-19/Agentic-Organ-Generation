"""
DesignSpec Workflow Manager for GUI

Provides GUI integration for the DesignSpec-first workflow.
Manages project creation/loading, patch approval, and run controls.
"""

import os
import json
import threading
import queue
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from pathlib import Path
from enum import Enum
from datetime import datetime

from .workflow_manager import WorkflowMessage, WorkflowStatus


class DesignSpecWorkflowManager:
    """
    Manages DesignSpec workflow execution and communication with GUI.

    Provides:
    - Project creation and loading
    - Patch proposal and approval flow
    - Compile and run controls
    - Event-based communication with GUI

    Parameters
    ----------
    message_callback : Callable
        Callback for sending messages to GUI
    status_callback : Callable
        Callback for status updates
    output_callback : Callable
        Callback for output/artifact updates
    spec_callback : Callable
        Callback for spec updates
    patch_callback : Callable
        Callback for patch proposals
    compile_callback : Callable
        Callback for compile status updates
    """

    def __init__(
        self,
        message_callback: Optional[Callable[[WorkflowMessage], None]] = None,
        status_callback: Optional[Callable[[WorkflowStatus, str], None]] = None,
        output_callback: Optional[Callable[[str, Any], None]] = None,
        spec_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        patch_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        compile_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        use_legacy_agent: bool = False,
    ):
        """
        Initialize the DesignSpec workflow manager.

        Parameters
        ----------
        message_callback : Callable, optional
            Callback for sending messages to GUI
        status_callback : Callable, optional
            Callback for status updates
        output_callback : Callable, optional
            Callback for output/artifact updates
        spec_callback : Callable, optional
            Callback for spec updates
        patch_callback : Callable, optional
            Callback for patch proposals
        compile_callback : Callable, optional
            Callback for compile status updates
        use_legacy_agent : bool
            If True, use the legacy rule-based agent instead of the LLM-first agent.
            Default is False (use LLM-first agent, recommended).
        """
        self.message_callback = message_callback
        self.status_callback = status_callback
        self.output_callback = output_callback
        self.spec_callback = spec_callback
        self.patch_callback = patch_callback
        self.compile_callback = compile_callback
        self._use_legacy_agent = use_legacy_agent

        self._status = WorkflowStatus.IDLE
        self._workflow = None
        self._workflow_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._input_queue: queue.Queue = queue.Queue()

        self._project_dir: Optional[Path] = None
        self._llm_client = None
        
        # LLM initialization state tracking
        self._llm_ready: bool = False
        self._last_llm_init_error: Optional[str] = None

        self._conversation_history: List[Dict[str, str]] = []
        self._artifacts: Dict[str, str] = {}

    @property
    def status(self) -> WorkflowStatus:
        """Get current workflow status."""
        return self._status

    @property
    def is_running(self) -> bool:
        """Check if workflow is currently running."""
        return self._status in (WorkflowStatus.RUNNING, WorkflowStatus.WAITING_INPUT)

    @property
    def project_dir(self) -> Optional[Path]:
        """Get current project directory."""
        return self._project_dir

    @property
    def llm_ready(self) -> bool:
        """Check if LLM client was successfully initialized."""
        return self._llm_ready

    @property
    def last_llm_init_error(self) -> Optional[str]:
        """Get the last LLM initialization error message, if any."""
        return self._last_llm_init_error

    def _set_status(self, status: WorkflowStatus, message: str = ""):
        """Update status and notify callback."""
        self._status = status
        if self.status_callback:
            self.status_callback(status, message)

    def _send_message(self, msg_type: str, content: str, data: Optional[Dict] = None):
        """Send message to GUI."""
        msg = WorkflowMessage(type=msg_type, content=content, data=data)
        if self.message_callback:
            self.message_callback(msg)

    def _send_output(self, output_type: str, data: Any):
        """Send output/artifact to GUI."""
        if self.output_callback:
            self.output_callback(output_type, data)

    def _send_spec_update(self, spec: Dict[str, Any]):
        """Send spec update to GUI."""
        if self.spec_callback:
            self.spec_callback(spec)

    def _send_patch_proposal(self, patch_data: Dict[str, Any]):
        """Send patch proposal to GUI."""
        if self.patch_callback:
            self.patch_callback(patch_data)

    def _send_compile_status(self, compile_data: Dict[str, Any]):
        """Send compile status to GUI."""
        if self.compile_callback:
            self.compile_callback(compile_data)

    def _format_run_failure_details(self, result: Dict[str, Any]) -> str:
        """
        Format detailed error message for run failures.

        Includes errors list, failing stage name, and output directory path.

        Parameters
        ----------
        result : dict
            The run result dictionary containing success, errors, stage_reports, etc.

        Returns
        -------
        str
            Formatted error message with details
        """
        message_parts = ["Run failed"]

        errors = result.get("errors", [])
        if errors:
            message_parts.append("\nErrors:")
            for error in errors[:5]:
                message_parts.append(f"  - {error}")

        stage_reports = result.get("stage_reports", [])
        failing_stage = None
        stage_errors = []
        for report in stage_reports:
            if isinstance(report, dict) and not report.get("success", True):
                failing_stage = report.get("stage", "unknown")
                stage_errors = report.get("errors", [])
                break

        if failing_stage:
            message_parts.append(f"\nFailing stage: {failing_stage}")
            if stage_errors:
                message_parts.append("Stage errors:")
                for err in stage_errors[:3]:
                    message_parts.append(f"  - {err}")

        output_dir = result.get("output_dir", "")
        if output_dir:
            message_parts.append(f"\nOutput directory: {output_dir}")

        return "\n".join(message_parts)

    def initialize_llm(
        self,
        provider: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> bool:
        """
        Initialize the LLM client for the agent.

        Parameters
        ----------
        provider : str
            LLM provider name
        api_key : str, optional
            API key for the provider
        model : str, optional
            Model name
        api_base : str, optional
            Custom API base URL
        temperature : float
            Sampling temperature
        max_tokens : int
            Maximum tokens in response

        Returns
        -------
        bool
            True if initialization was successful
        """
        try:
            from automation.llm_client import LLMClient, LLMConfig

            self._send_message("system", f"Initializing {provider} LLM client...")

            llm_config = LLMConfig(
                provider=provider,
                api_key=api_key,
                model=model or "default",
                api_base=api_base,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            self._llm_client = LLMClient(config=llm_config)

            self._send_message(
                "system",
                f"LLM initialized: {provider}/{model or 'default'}"
            )
            
            # Mark LLM as ready and clear any previous error
            self._llm_ready = True
            self._last_llm_init_error = None
            return True

        except Exception as e:
            # Store the error for the GUI to display
            error_str = str(e)
            # Sanitize: truncate and remove potential secrets
            if len(error_str) > 300:
                error_str = error_str[:300] + "..."
            self._last_llm_init_error = error_str
            self._llm_ready = False
            
            self._send_message("error", f"Failed to initialize LLM: {e}")
            return False

    def create_project(
        self,
        project_root: str,
        project_name: str,
        template_spec: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Create a new DesignSpec project.

        Parameters
        ----------
        project_root : str
            Parent directory for the project
        project_name : str
            Name of the project
        template_spec : dict, optional
            Initial spec template

        Returns
        -------
        bool
            True if project was created successfully
        """
        try:
            from automation.designspec_session import DesignSpecSession
            from automation.designspec_agent import DesignSpecAgent
            from automation.workflows.designspec_workflow import DesignSpecWorkflow

            self._send_message("system", f"Creating project: {project_name}...")
            self._set_status(WorkflowStatus.INITIALIZING, "Creating project...")

            self._workflow = DesignSpecWorkflow(
                llm_client=self._llm_client,
                event_callback=self._on_workflow_event,
                use_legacy_agent=self._use_legacy_agent,
            )

            agent_mode = "legacy rule-based" if self._use_legacy_agent else "LLM-first"
            self._send_message("system", f"Using {agent_mode} agent mode")

            success = self._workflow.on_start(
                project_root=project_root,
                project_name=project_name,
                template_spec=template_spec,
            )

            if success:
                self._project_dir = Path(project_root) / project_name
                self._set_status(WorkflowStatus.WAITING_INPUT, "Project created")
                self._send_message("success", f"Project created: {self._project_dir}")

                spec = self._workflow.get_spec()
                if spec:
                    self._send_spec_update(spec)

                return True
            else:
                self._set_status(WorkflowStatus.FAILED, "Failed to create project")
                return False

        except Exception as e:
            self._send_message("error", f"Failed to create project: {e}")
            self._set_status(WorkflowStatus.FAILED, str(e))
            return False

    def load_project(self, project_dir: str) -> bool:
        """
        Load an existing DesignSpec project.

        Parameters
        ----------
        project_dir : str
            Path to the project directory

        Returns
        -------
        bool
            True if project was loaded successfully
        """
        try:
            from automation.workflows.designspec_workflow import DesignSpecWorkflow

            self._send_message("system", f"Loading project: {project_dir}...")
            self._set_status(WorkflowStatus.INITIALIZING, "Loading project...")

            self._workflow = DesignSpecWorkflow(
                llm_client=self._llm_client,
                event_callback=self._on_workflow_event,
                use_legacy_agent=self._use_legacy_agent,
            )

            agent_mode = "legacy rule-based" if self._use_legacy_agent else "LLM-first"
            self._send_message("system", f"Using {agent_mode} agent mode")

            success = self._workflow.on_start(project_dir=project_dir)

            if success:
                self._project_dir = Path(project_dir)
                self._set_status(WorkflowStatus.WAITING_INPUT, "Project loaded")
                self._send_message("success", f"Project loaded: {self._project_dir}")

                spec = self._workflow.get_spec()
                if spec:
                    self._send_spec_update(spec)

                return True
            else:
                self._set_status(WorkflowStatus.FAILED, "Failed to load project")
                return False

        except Exception as e:
            self._send_message("error", f"Failed to load project: {e}")
            self._set_status(WorkflowStatus.FAILED, str(e))
            return False

    def _on_workflow_event(self, event):
        """Handle workflow events."""
        from automation.workflows.designspec_workflow import WorkflowEventType

        event_type = event.event_type
        data = event.data
        message = event.message

        if event_type == WorkflowEventType.MESSAGE:
            self._send_message("assistant", message)

        elif event_type == WorkflowEventType.QUESTION:
            questions = data.get("questions", [])
            if questions:
                question_text = "\n".join(
                    f"- {q.get('question_text', q.get('text', q.get('question', '')))}"
                    for q in questions
                )
                self._send_message("prompt", f"{message}\n{question_text}")
            else:
                self._send_message("prompt", message)

        elif event_type == WorkflowEventType.PATCH_PROPOSAL:
            self._send_patch_proposal(data)
            self._send_message("assistant", message)

        elif event_type == WorkflowEventType.PATCH_APPLIED:
            self._send_message("success", message)

        elif event_type == WorkflowEventType.PATCH_REJECTED:
            self._send_message("system", message)

        elif event_type == WorkflowEventType.COMPILE_STARTED:
            self._send_compile_status({"status": "running", "message": message})

        elif event_type == WorkflowEventType.COMPILE_COMPLETED:
            compile_report = data.get("compile_report", {})
            success = compile_report.get("success", False)
            self._send_compile_status({
                "status": "success" if success else "failed",
                "report": compile_report,
                "message": message,
            })

        elif event_type == WorkflowEventType.RUN_STARTED:
            self._send_message("system", message)
            self._set_status(WorkflowStatus.RUNNING, message)

        elif event_type == WorkflowEventType.RUN_COMPLETED:
            result = data.get("result", {})
            success = result.get("success", False)
            if success:
                self._send_message("success", message)
                artifacts = result.get("artifacts", [])
                for artifact in artifacts:
                    if artifact.get("type") == "stl":
                        self._send_output("stl_file", artifact.get("path"))
            else:
                error_details = self._format_run_failure_details(result)
                self._send_message("error", error_details)
            self._set_status(WorkflowStatus.WAITING_INPUT, "Run completed")

        elif event_type == WorkflowEventType.SPEC_UPDATED:
            spec = data.get("spec", {})
            self._send_spec_update(spec)

        elif event_type == WorkflowEventType.VALIDATION_RESULT:
            valid = data.get("valid", False)
            errors = data.get("errors", [])
            warnings = data.get("warnings", [])
            if not valid:
                error_text = "\n".join(f"- {e}" for e in errors[:5])
                self._send_message("warning", f"Validation issues:\n{error_text}")

        elif event_type == WorkflowEventType.ERROR:
            self._send_message("error", message)

        elif event_type == WorkflowEventType.STATUS_CHANGE:
            new_status = data.get("new_status", "")
            if new_status == "waiting_input":
                self._set_status(WorkflowStatus.WAITING_INPUT, "Waiting for input")
            elif new_status == "processing":
                self._set_status(WorkflowStatus.RUNNING, "Processing...")
            elif new_status == "compiling":
                self._set_status(WorkflowStatus.RUNNING, "Compiling...")
            elif new_status == "running":
                self._set_status(WorkflowStatus.RUNNING, "Running pipeline...")

    def send_message(self, text: str):
        """
        Send a user message to the workflow.

        Parameters
        ----------
        text : str
            User message text
        """
        if not self._workflow:
            self._send_message("error", "No project loaded. Create or open a project first.")
            return

        self._conversation_history.append({"role": "user", "content": text})

        self._workflow_thread = threading.Thread(
            target=self._process_message_thread,
            args=(text,),
            daemon=True,
        )
        self._workflow_thread.start()

    def _process_message_thread(self, text: str):
        """Process user message in background thread."""
        try:
            self._workflow.on_user_message(text)
        except Exception as e:
            self._send_message("error", f"Error processing message: {e}")

    def approve_patch(self, patch_id: str) -> bool:
        """
        Approve a pending patch.

        Parameters
        ----------
        patch_id : str
            ID of the patch to approve

        Returns
        -------
        bool
            True if patch was approved successfully
        """
        if not self._workflow:
            self._send_message("error", "No project loaded")
            return False

        return self._workflow.approve_patch(patch_id)

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
        if not self._workflow:
            self._send_message("error", "No project loaded")
            return

        self._workflow.reject_patch(patch_id, reason)

    def approve_run(self) -> bool:
        """
        Approve the pending run request.

        Returns
        -------
        bool
            True if run was approved and started successfully
        """
        if not self._workflow:
            self._send_message("error", "No project loaded")
            return False

        self._workflow_thread = threading.Thread(
            target=self._approve_run_thread,
            daemon=True,
        )
        self._workflow_thread.start()
        return True

    def _approve_run_thread(self):
        """Approve run in background thread."""
        try:
            self._workflow.approve_pending_run()
        except Exception as e:
            self._send_message("error", f"Run approval failed: {e}")

    def reject_run(self, reason: str = "") -> None:
        """
        Reject the pending run request.

        Parameters
        ----------
        reason : str, optional
            Reason for rejection
        """
        if not self._workflow:
            self._send_message("error", "No project loaded")
            return

        self._workflow.reject_pending_run(reason)

    def get_pending_run_request(self) -> Optional[Dict[str, Any]]:
        """
        Get the pending run request if any.

        Returns
        -------
        dict or None
            The pending run request data, or None if no pending request
        """
        if not self._workflow:
            return None

        run_req = self._workflow.get_pending_run_request()
        if run_req:
            return run_req.to_dict()
        return None

    def run_until(self, stage: str) -> None:
        """
        Run the pipeline until a specific stage.

        Parameters
        ----------
        stage : str
            Stage to run until
        """
        if not self._workflow:
            self._send_message("error", "No project loaded")
            return

        self._workflow_thread = threading.Thread(
            target=self._run_until_thread,
            args=(stage,),
            daemon=True,
        )
        self._workflow_thread.start()

    def _run_until_thread(self, stage: str):
        """Run until stage in background thread."""
        try:
            self._workflow.run_until(stage)
        except Exception as e:
            self._send_message("error", f"Run failed: {e}")

    def run_full(self) -> None:
        """Run the full pipeline."""
        if not self._workflow:
            self._send_message("error", "No project loaded")
            return

        self._workflow_thread = threading.Thread(
            target=self._run_full_thread,
            daemon=True,
        )
        self._workflow_thread.start()

    def _run_full_thread(self):
        """Run full pipeline in background thread."""
        try:
            self._workflow.run_full()
        except Exception as e:
            self._send_message("error", f"Full run failed: {e}")

    def compile(self) -> None:
        """Manually trigger compilation."""
        if not self._workflow:
            self._send_message("error", "No project loaded")
            return

        self._workflow_thread = threading.Thread(
            target=self._compile_thread,
            daemon=True,
        )
        self._workflow_thread.start()

    def _compile_thread(self):
        """Compile in background thread."""
        try:
            self._workflow.compile()
        except Exception as e:
            self._send_message("error", f"Compile failed: {e}")

    def get_spec(self) -> Optional[Dict[str, Any]]:
        """Get the current spec."""
        if self._workflow:
            return self._workflow.get_spec()
        return None

    def get_pending_patches(self) -> Dict[str, Dict[str, Any]]:
        """Get pending patches awaiting approval."""
        if self._workflow:
            return self._workflow.get_pending_patches()
        return {}

    def get_artifacts(self) -> List[Dict[str, Any]]:
        """Get list of generated artifacts."""
        if self._workflow:
            return self._workflow.get_artifacts()
        return []

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        if self._workflow:
            return self._workflow.get_conversation_history()
        return self._conversation_history.copy()

    def stop(self):
        """Stop the workflow."""
        self._stop_event.set()
        self._set_status(WorkflowStatus.CANCELLED, "Workflow stopped")
        self._send_message("system", "Workflow stopped")


__all__ = ["DesignSpecWorkflowManager"]
