"""
Workflow Manager

Manages the execution of Single Agent workflows.
Provides a unified interface for workflow control and status monitoring.

V5 Update:
- Added support for V5 goal-driven controller
- Uses IO adapters instead of input() monkeypatching
- Supports modal approvals for generation and postprocess
- Shows trace timeline for workflow progress
"""

import io
import os
import sys
import threading
import queue
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, List
from enum import Enum
from pathlib import Path


class OutputCapture(io.StringIO):
    """Capture stdout/stderr and forward to a callback."""
    
    def __init__(self, callback: Callable[[str], None], original_stream):
        super().__init__()
        self._callback = callback
        self._original = original_stream
    
    def write(self, text: str) -> int:
        if text and text.strip():
            self._callback(text.rstrip())
        if self._original:
            self._original.write(text)
        return len(text)
    
    def flush(self):
        if self._original:
            self._original.flush()


class WorkflowType(Enum):
    """Available workflow types."""
    SINGLE_AGENT = "single_agent"
    DESIGNSPEC = "designspec"


class WorkflowStatus(Enum):
    """Workflow execution status."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    WAITING_INPUT = "waiting_input"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowMessage:
    """Message from workflow to GUI."""
    type: str
    content: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            from datetime import datetime
            self.timestamp = datetime.now().isoformat()


@dataclass
class WorkflowConfig:
    """Configuration for workflow execution."""
    workflow_type: WorkflowType = WorkflowType.SINGLE_AGENT
    output_dir: str = "./projects"
    execution_mode: str = "review_then_run"
    timeout_seconds: float = 300.0
    verbose: bool = True
    auto_approve: bool = False
    llm_first_mode: bool = False  # Enable LLM-first agentic mode for V5


class WorkflowManager:
    """
    Manages workflow execution and communication with GUI.
    
    Provides:
    - Workflow initialization and execution
    - Thread-safe message passing to GUI
    - User input handling
    - Status monitoring
    
    Parameters
    ----------
    message_callback : Callable
        Callback for sending messages to GUI
    status_callback : Callable
        Callback for status updates
    output_callback : Callable
        Callback for output/artifact updates
    """
    
    def __init__(
        self,
        message_callback: Optional[Callable[[WorkflowMessage], None]] = None,
        status_callback: Optional[Callable[[WorkflowStatus, str], None]] = None,
        output_callback: Optional[Callable[[str, Any], None]] = None,
    ):
        self.message_callback = message_callback
        self.status_callback = status_callback
        self.output_callback = output_callback
        
        self._status = WorkflowStatus.IDLE
        self._workflow_thread: Optional[threading.Thread] = None
        self._input_queue: queue.Queue = queue.Queue()
        self._approval_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._current_workflow = None
        self._config: Optional[WorkflowConfig] = None
        self._agent = None
        self._llm_client = None
        
        self._artifacts: Dict[str, str] = {}
        self._conversation_history: List[Dict[str, str]] = []
    
    @property
    def status(self) -> WorkflowStatus:
        """Get current workflow status."""
        return self._status
    
    @property
    def is_running(self) -> bool:
        """Check if workflow is currently running."""
        return self._status in (WorkflowStatus.RUNNING, WorkflowStatus.WAITING_INPUT)
    
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
    
    def initialize_agent(
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
        Initialize the LLM agent.
        
        Parameters
        ----------
        provider : str
            LLM provider name
        api_key : str, optional
            API key for the provider
        model : str, optional
            Model name
        api_base : str, optional
            Custom API base URL (required for local provider)
        temperature : float
            Sampling temperature (0.0-1.0)
        max_tokens : int
            Maximum tokens in response
        **kwargs
            Additional configuration options
            
        Returns
        -------
        bool
            True if initialization was successful
        """
        try:
            from automation.llm_client import LLMClient, LLMConfig
            from automation.agent_runner import AgentRunner, AgentConfig, create_agent
            
            self._send_message("system", f"Initializing {provider} agent...")
            
            llm_config = LLMConfig(
                provider=provider,
                api_key=api_key,
                model=model or "default",
                api_base=api_base,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            self._llm_client = LLMClient(config=llm_config)
            
            self._agent = create_agent(
                provider=provider,
                api_key=api_key,
                model=model,
                api_base=api_base,
                temperature=temperature,
                max_tokens=max_tokens,
                verbose=kwargs.get("verbose", True),
            )
            
            self._send_message("system", f"Agent initialized: {provider}/{model or 'default'} (temp={temperature}, max_tokens={max_tokens})")
            return True
            
        except Exception as e:
            self._send_message("error", f"Failed to initialize agent: {e}")
            return False
    
    def start_workflow(self, config: WorkflowConfig) -> bool:
        """
        Start a workflow execution.
        
        Parameters
        ----------
        config : WorkflowConfig
            Workflow configuration
            
        Returns
        -------
        bool
            True if workflow started successfully
        """
        if self.is_running:
            self._send_message("error", "A workflow is already running")
            return False
        
        if self._agent is None:
            self._send_message("error", "Agent not initialized. Configure agent first.")
            return False
        
        self._config = config
        self._stop_event.clear()
        self._artifacts.clear()
        self._conversation_history.clear()
        
        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                break
        
        self._workflow_thread = threading.Thread(
            target=self._run_workflow_thread,
            daemon=True,
        )
        self._workflow_thread.start()
        
        return True
    
    def _run_workflow_thread(self):
        """Run workflow in background thread."""
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        def on_output(text: str):
            self._send_message("output", text)
        
        stdout_capture = OutputCapture(on_output, original_stdout)
        stderr_capture = OutputCapture(on_output, original_stderr)
        
        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            self._set_status(WorkflowStatus.INITIALIZING, "Starting workflow...")
            
            if self._config.workflow_type == WorkflowType.SINGLE_AGENT:
                self._run_single_agent_workflow_v5()
            elif self._config.workflow_type == WorkflowType.DESIGNSPEC:
                self._send_message(
                    "system",
                    "DesignSpec projects are started via the DesignSpec Project dialog, "
                    "not Start Workflow. Please use File > New DesignSpec Project or "
                    "File > Open DesignSpec Project instead."
                )
                self._set_status(WorkflowStatus.WAITING_INPUT, "Use DesignSpec Project dialog")
                return
            else:
                raise ValueError(f"Unknown workflow type: {self._config.workflow_type}")
            
        except Exception as e:
            self._set_status(WorkflowStatus.FAILED, str(e))
            self._send_message("error", f"Workflow failed: {e}")
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
    
    def _run_single_agent_workflow(self):
        """
        Run Single Agent Organ Generator workflow.
        
        Note: This method temporarily replaces builtins.input() with a GUI-based
        input function to intercept workflow prompts. This is a known limitation
        that affects any code using input() during workflow execution. For better
        isolation, consider running workflows in a subprocess in future versions.
        """
        try:
            from automation.workflow import SingleAgentOrganGeneratorV4
            from automation.execution_modes import parse_execution_mode
            
            self._send_message("system", "Starting Single Agent Organ Generator V4...")
            self._set_status(WorkflowStatus.RUNNING, "Running Single Agent workflow")
            
            execution_mode = parse_execution_mode(self._config.execution_mode)
            
            os.makedirs(self._config.output_dir, exist_ok=True)
            
            # IMPORTANT: Apply monkeypatch BEFORE creating workflow object
            # The workflow's ReactivePromptSession captures the reference to input()
            # at construction time, so we must patch it first.
            original_input = __builtins__["input"] if isinstance(__builtins__, dict) else __builtins__.input
            
            def gui_input(prompt=""):
                self._send_message("prompt", prompt)
                self._set_status(WorkflowStatus.WAITING_INPUT, "Waiting for user input")
                
                while not self._stop_event.is_set():
                    try:
                        response = self._input_queue.get(timeout=0.5)
                        self._set_status(WorkflowStatus.RUNNING, "Processing input")
                        return response
                    except queue.Empty:
                        continue
                
                raise KeyboardInterrupt("Workflow cancelled")
            
            if isinstance(__builtins__, dict):
                __builtins__["input"] = gui_input
            else:
                import builtins
                builtins.input = gui_input
            
            # Use try/finally to ensure we always restore the original input function
            try:
                # Now create the workflow - it will capture the patched input function
                workflow = SingleAgentOrganGeneratorV4(
                    agent=self._agent,
                    base_output_dir=self._config.output_dir,
                    verbose=self._config.verbose,
                    execution_mode=execution_mode,
                    timeout_seconds=self._config.timeout_seconds,
                )
                
                self._current_workflow = workflow
                
                context = workflow.run()
                
                self._artifacts = {
                    "project_name": context.project_name,
                    "output_dir": context.output_dir,
                }
                
                if hasattr(context, "objects"):
                    for obj in context.objects:
                        if hasattr(obj, "output_dir"):
                            stl_files = list(Path(obj.output_dir).glob("**/*.stl"))
                            for stl_file in stl_files:
                                self._artifacts[f"stl_{stl_file.stem}"] = str(stl_file)
                                self._send_output("stl_file", str(stl_file))
                
                self._set_status(WorkflowStatus.COMPLETED, "Workflow completed successfully")
                self._send_message("success", f"Workflow completed! Output: {context.output_dir}")
                
            finally:
                # Always restore the original input function
                if isinstance(__builtins__, dict):
                    __builtins__["input"] = original_input
                else:
                    import builtins
                    builtins.input = original_input
            
        except KeyboardInterrupt:
            self._set_status(WorkflowStatus.CANCELLED, "Workflow cancelled by user")
            self._send_message("system", "Workflow cancelled")
        except Exception as e:
            self._set_status(WorkflowStatus.FAILED, str(e))
            self._send_message("error", f"Single Agent workflow failed: {e}")
    
    def send_input(self, text: str):
        """
        Send user input to the running workflow.
        
        For V5 workflows, this method routes messages appropriately:
        - If the workflow thread is alive and waiting on a queue (blocking callback),
          the message goes to the input queue.
        - If the workflow thread has ended (controller returned WAITING), the message
          is routed to process_user_message() and the workflow run loop is restarted.
        
        The key distinction is whether the workflow thread is still alive:
        - Thread alive + WAITING_INPUT = blocked on queue.get() → put in queue
        - Thread dead + WAITING_INPUT = controller paused → process_user_message + restart
        
        Parameters
        ----------
        text : str
            User input text
        """
        self._conversation_history.append({"role": "user", "content": text})
        
        # Check if this is a V5 workflow that has truly paused (thread ended, controller
        # returned WAITING) vs just waiting on the input queue (thread alive, blocked on
        # queue.get()). Only call process_user_message() if the thread has ended.
        if (self._current_workflow is not None and 
            hasattr(self._current_workflow, 'process_user_message') and
            self._status == WorkflowStatus.WAITING_INPUT and
            self._workflow_thread is not None and
            not self._workflow_thread.is_alive()):
            # Controller has paused (run() returned WAITING) - process message and restart
            self._current_workflow.process_user_message(text)
            self._resume_v5_workflow()
        else:
            # Thread is alive and waiting on queue, or not a V5 workflow
            self._input_queue.put(text)
    
    def _resume_v5_workflow(self):
        """
        Resume a paused V5 workflow by restarting the run loop.
        
        This is called after process_user_message() has been invoked to
        continue the workflow from where it left off.
        """
        if self._current_workflow is None:
            return
        
        if not hasattr(self._current_workflow, 'run'):
            return
        
        self._workflow_thread = threading.Thread(
            target=self._continue_v5_workflow,
            daemon=True,
        )
        self._workflow_thread.start()
    
    def _continue_v5_workflow(self):
        """
        Continue a V5 workflow after user input has been processed.
        
        This runs in a separate thread and calls workflow.run() to
        continue from where the workflow left off.
        """
        try:
            from automation.single_agent_organ_generation.v5.controller import RunResult
            
            self._set_status(WorkflowStatus.RUNNING, "Resuming V5 workflow")
            
            result = self._current_workflow.run()
            
            if result == RunResult.COMPLETED:
                status = self._current_workflow.get_status()
                self._artifacts = {
                    "spec_hash": status.get("spec_hash", ""),
                    "output_dir": self._config.output_dir,
                }
                
                self._set_status(WorkflowStatus.COMPLETED, "V5 workflow completed successfully")
                self._send_message("success", f"V5 workflow completed! Output: {self._config.output_dir}")
            elif result == RunResult.WAITING:
                self._set_status(WorkflowStatus.WAITING_INPUT, "Workflow paused - waiting for input")
            else:
                self._set_status(WorkflowStatus.FAILED, "V5 workflow failed")
                self._send_message("error", "V5 workflow failed")
                
        except KeyboardInterrupt:
            self._set_status(WorkflowStatus.CANCELLED, "Workflow cancelled by user")
            self._send_message("system", "Workflow cancelled")
        except Exception as e:
            self._set_status(WorkflowStatus.FAILED, str(e))
            self._send_message("error", f"V5 workflow failed: {e}")
    
    def stop_workflow(self):
        """Stop the currently running workflow."""
        if self.is_running:
            self._stop_event.set()
            self._send_message("system", "Stopping workflow...")
    
    def get_artifacts(self) -> Dict[str, str]:
        """Get generated artifacts."""
        return self._artifacts.copy()
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self._conversation_history.copy()
    
    def _run_single_agent_workflow_v5(self, initial_message: Optional[str] = None):
        """
        Run Single Agent Organ Generator V5 workflow.
        
        V5 uses IO adapters instead of input() monkeypatching, providing:
        - Modal approvals for generation and postprocess
        - Trace timeline for workflow progress
        - Clean separation between controller and UI
        
        Parameters
        ----------
        initial_message : str, optional
            Initial user message to start the workflow
        """
        try:
            from automation.workflow import (
                SingleAgentOrganGeneratorV5,
                ControllerConfig,
                GUIIOAdapter,
            )
            from automation.single_agent_organ_generation.v5.io.base_io import (
                IOMessageKind,
                TraceEvent,
            )
            from automation.single_agent_organ_generation.v5.controller import RunResult
            
            self._send_message("system", "Starting Single Agent Organ Generator V5...")
            self._set_status(WorkflowStatus.RUNNING, "Running V5 workflow")
            
            os.makedirs(self._config.output_dir, exist_ok=True)
            
            def on_message(message: str, kind: IOMessageKind, payload: Optional[Dict] = None):
                msg_type = "assistant"
                if kind == IOMessageKind.ERROR:
                    msg_type = "error"
                elif kind == IOMessageKind.WARNING:
                    msg_type = "warning"
                elif kind == IOMessageKind.SUCCESS:
                    msg_type = "success"
                elif kind == IOMessageKind.SYSTEM:
                    msg_type = "system"
                elif kind == IOMessageKind.TRACE:
                    msg_type = "trace"
                
                self._send_message(msg_type, message, payload)
            
            def on_approval(approval_data: Dict[str, Any]) -> bool:
                self._send_message("approval_request", approval_data.get("prompt", "Approve?"), approval_data)
                self._set_status(WorkflowStatus.WAITING_INPUT, "Waiting for approval")
                
                while not self._stop_event.is_set():
                    try:
                        approved = self._approval_queue.get(timeout=0.5)
                        self._set_status(WorkflowStatus.RUNNING, "Processing approval")
                        return approved
                    except queue.Empty:
                        continue
                
                raise KeyboardInterrupt("Workflow cancelled")
            
            def on_text_input(prompt: str, suggestions: Optional[List[str]], default: Optional[str]) -> str:
                self._send_message("prompt", prompt, {"suggestions": suggestions, "default": default})
                self._set_status(WorkflowStatus.WAITING_INPUT, "Waiting for user input")
                
                while not self._stop_event.is_set():
                    try:
                        response = self._input_queue.get(timeout=0.5)
                        self._set_status(WorkflowStatus.RUNNING, "Processing input")
                        if not response and default:
                            return default
                        return response
                    except queue.Empty:
                        continue
                
                raise KeyboardInterrupt("Workflow cancelled")
            
            def on_trace(event: TraceEvent):
                self._send_message("trace", event.message, {
                    "event_type": event.event_type,
                    "data": event.data,
                })
            
            def on_spec_display(spec_summary: Dict[str, Any]):
                self._send_message("spec_update", "Living spec updated", spec_summary)
            
            def on_plans_display(plans: List[Dict[str, Any]], recommended_id: Optional[str]):
                self._send_message("plans", "Plans proposed", {
                    "plans": plans,
                    "recommended_id": recommended_id,
                })
            
            def on_plan_selection(plans: List[Dict[str, Any]]) -> Optional[str]:
                self._send_message("plan_selection", "Select a plan", {"plans": plans})
                self._set_status(WorkflowStatus.WAITING_INPUT, "Waiting for plan selection")
                
                while not self._stop_event.is_set():
                    try:
                        response = self._input_queue.get(timeout=0.5)
                        self._set_status(WorkflowStatus.RUNNING, "Processing selection")
                        if not response:
                            return None
                        return response
                    except queue.Empty:
                        continue
                
                raise KeyboardInterrupt("Workflow cancelled")
            
            def on_safe_fix(field: str, before: Any, after: Any, reason: str):
                self._send_message("safe_fix", f"Applied safe fix: {field}", {
                    "field": field,
                    "before": before,
                    "after": after,
                    "reason": reason,
                })
            
            def on_generation_ready(runtime: str, outputs: List[str], assumptions: List[str], risks: List[str]):
                self._send_message("generation_ready", "Ready to generate", {
                    "runtime_estimate": runtime,
                    "expected_outputs": outputs,
                    "assumptions": assumptions,
                    "risk_flags": risks,
                })
            
            def on_postprocess_ready(voxel_pitch: float, settings: Dict, steps: List[str], runtime: str, outputs: List[str]):
                self._send_message("postprocess_ready", "Ready to postprocess", {
                    "voxel_pitch": voxel_pitch,
                    "embedding_settings": settings,
                    "repair_steps": steps,
                    "runtime_estimate": runtime,
                    "expected_outputs": outputs,
                })
            
            def on_stl_viewer(stl_path: str):
                self._send_message("stl_ready", f"STL file ready: {stl_path}", {"stl_path": stl_path})
                self._send_output("stl_file", stl_path)
            
            io_adapter = GUIIOAdapter(
                message_callback=on_message,
                approval_callback=on_approval,
                text_input_callback=on_text_input,
                trace_callback=on_trace,
                spec_display_callback=on_spec_display,
                plans_display_callback=on_plans_display,
                plan_selection_callback=on_plan_selection,
                safe_fix_callback=on_safe_fix,
                generation_ready_callback=on_generation_ready,
                postprocess_ready_callback=on_postprocess_ready,
                stl_viewer_callback=on_stl_viewer,
            )
            
            config = ControllerConfig(
                verbose=self._config.verbose,
                auto_select_plan_if_confident=not self._config.auto_approve,
                llm_first_mode=self._config.llm_first_mode,
                output_dir=self._config.output_dir,
            )
            
            # Pass LLM client to V5 controller for LLM-first mode
            workflow = SingleAgentOrganGeneratorV5(
                io_adapter=io_adapter,
                config=config,
                llm_client=self._llm_client if self._config.llm_first_mode else None,
            )
            
            self._current_workflow = workflow
            
            result = workflow.run(initial_message=initial_message)
            
            if result == RunResult.COMPLETED:
                status = workflow.get_status()
                self._artifacts = {
                    "spec_hash": status.get("spec_hash", ""),
                    "output_dir": self._config.output_dir,
                }
                
                self._set_status(WorkflowStatus.COMPLETED, "V5 workflow completed successfully")
                self._send_message("success", f"V5 workflow completed! Output: {self._config.output_dir}")
            elif result == RunResult.WAITING:
                self._set_status(WorkflowStatus.WAITING_INPUT, "Workflow paused - waiting for input")
            else:
                self._set_status(WorkflowStatus.FAILED, "V5 workflow failed")
                self._send_message("error", "V5 workflow failed")
            
        except KeyboardInterrupt:
            self._set_status(WorkflowStatus.CANCELLED, "Workflow cancelled by user")
            self._send_message("system", "Workflow cancelled")
        except Exception as e:
            self._set_status(WorkflowStatus.FAILED, str(e))
            self._send_message("error", f"V5 workflow failed: {e}")
    
    def start_workflow_v5(self, config: WorkflowConfig, initial_message: Optional[str] = None) -> bool:
        """
        Start a V5 workflow execution.
        
        Parameters
        ----------
        config : WorkflowConfig
            Workflow configuration
        initial_message : str, optional
            Initial user message to start the workflow
            
        Returns
        -------
        bool
            True if workflow started successfully
        """
        if self.is_running:
            self._send_message("error", "A workflow is already running")
            return False
        
        self._config = config
        self._stop_event.clear()
        self._artifacts.clear()
        self._conversation_history.clear()
        
        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                break
        
        self._workflow_thread = threading.Thread(
            target=lambda: self._run_single_agent_workflow_v5(initial_message),
            daemon=True,
        )
        self._workflow_thread.start()
        
        return True
    
    def send_approval(self, approved: bool):
        """
        Send approval response to the running V5 workflow.
        
        Uses a dedicated approval queue to ensure modal-safe behavior,
        separate from the text input queue.
        
        Parameters
        ----------
        approved : bool
            Whether the user approved the action
        """
        self._approval_queue.put(approved)
        self._conversation_history.append({
            "role": "user",
            "content": f"Approval: {'Yes' if approved else 'No'}",
        })
