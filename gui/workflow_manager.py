"""
Workflow Manager

Manages the execution of Single Agent and MOGS workflows.
Provides a unified interface for workflow control and status monitoring.
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
    MOGS = "mogs"


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
                self._run_single_agent_workflow()
            elif self._config.workflow_type == WorkflowType.MOGS:
                self._run_mogs_workflow()
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
    
    def _run_mogs_workflow(self):
        """Run MOGS (Multi-Agent Organ Generation System) workflow."""
        try:
            from automation.mogs import MOGSRunner, create_mogs_runner
            from automation.mogs.gates import GateContext, GateResult, ApprovalChoice
            
            self._send_message("system", "Starting MOGS workflow...")
            self._set_status(WorkflowStatus.RUNNING, "Running MOGS workflow")
            
            os.makedirs(self._config.output_dir, exist_ok=True)
            
            def approval_callback(context: GateContext) -> GateResult:
                """Handle approval gates via GUI."""
                gate_info = {
                    "gate_type": context.gate_type.value if hasattr(context.gate_type, "value") else str(context.gate_type),
                    "spec_version": context.spec_version,
                    "summary": getattr(context, "summary", ""),
                    "artifacts": getattr(context, "artifacts", {}),
                }
                
                self._send_message("approval_gate", f"Approval required: {gate_info['gate_type']}", gate_info)
                self._set_status(WorkflowStatus.WAITING_INPUT, f"Waiting for approval: {gate_info['gate_type']}")
                
                while not self._stop_event.is_set():
                    try:
                        response = self._input_queue.get(timeout=0.5)
                        self._set_status(WorkflowStatus.RUNNING, "Processing approval")
                        
                        response_lower = response.lower().strip()
                        if response_lower in ("approve", "yes", "y", "ok", "approved"):
                            return GateResult(
                                gate_type=context.gate_type,
                                spec_version=context.spec_version,
                                choice=ApprovalChoice.APPROVE,
                                comments="Approved via GUI",
                            )
                        elif response_lower.startswith("refine") or response_lower.startswith("revise"):
                            refinement_notes = response
                            if ":" in response:
                                refinement_notes = response.split(":", 1)[1].strip()
                            elif " " in response:
                                refinement_notes = response.split(" ", 1)[1].strip()
                            return GateResult(
                                gate_type=context.gate_type,
                                spec_version=context.spec_version,
                                choice=ApprovalChoice.REFINE,
                                refinement_notes=refinement_notes or response,
                            )
                        elif response_lower in ("reject", "no", "n", "rejected"):
                            return GateResult(
                                gate_type=context.gate_type,
                                spec_version=context.spec_version,
                                choice=ApprovalChoice.REJECT,
                                comments="Rejected via GUI",
                            )
                        else:
                            return GateResult(
                                gate_type=context.gate_type,
                                spec_version=context.spec_version,
                                choice=ApprovalChoice.REFINE,
                                refinement_notes=response,
                            )
                    except queue.Empty:
                        continue
                
                raise KeyboardInterrupt("Workflow cancelled")
            
            if self._config.auto_approve:
                approval_cb = None
            else:
                approval_cb = approval_callback
            
            runner = MOGSRunner(
                objects_base_dir=self._config.output_dir,
                llm_client=self._llm_client,
                approval_callback=approval_cb,
                auto_approve=self._config.auto_approve,
            )
            
            self._current_workflow = runner
            
            self._send_message("prompt", "Enter object name for MOGS workflow:")
            self._set_status(WorkflowStatus.WAITING_INPUT, "Waiting for object name")
            
            object_name = None
            while not self._stop_event.is_set():
                try:
                    object_name = self._input_queue.get(timeout=0.5)
                    break
                except queue.Empty:
                    continue
            
            if object_name is None:
                raise KeyboardInterrupt("Workflow cancelled")
            
            self._set_status(WorkflowStatus.RUNNING, "Creating MOGS object")
            folder_manager = runner.create_object(object_name)
            object_uuid = folder_manager.object_uuid
            
            self._send_message("system", f"Created object: {object_name} ({object_uuid})")
            
            self._send_message("prompt", "Describe the organ structure you want to generate:")
            self._set_status(WorkflowStatus.WAITING_INPUT, "Waiting for requirements")
            
            user_description = None
            while not self._stop_event.is_set():
                try:
                    user_description = self._input_queue.get(timeout=0.5)
                    break
                except queue.Empty:
                    continue
            
            if user_description is None:
                raise KeyboardInterrupt("Workflow cancelled")
            
            self._set_status(WorkflowStatus.RUNNING, "Running MOGS workflow")
            
            requirements = {
                "description": user_description,
                "user_intent": user_description,
            }
            
            result = runner.run_workflow(
                object_uuid=object_uuid,
                requirements=requirements,
                user_description=user_description,
            )
            
            if result.success:
                self._artifacts = result.final_outputs
                
                for key, path in result.final_outputs.items():
                    if path.endswith(".stl"):
                        self._send_output("stl_file", path)
                
                self._set_status(WorkflowStatus.COMPLETED, "MOGS workflow completed successfully")
                self._send_message("success", f"MOGS workflow completed! {result.message}")
            else:
                self._set_status(WorkflowStatus.FAILED, result.message)
                self._send_message("error", f"MOGS workflow failed: {result.message}")
            
        except KeyboardInterrupt:
            self._set_status(WorkflowStatus.CANCELLED, "Workflow cancelled by user")
            self._send_message("system", "Workflow cancelled")
        except Exception as e:
            self._set_status(WorkflowStatus.FAILED, str(e))
            self._send_message("error", f"MOGS workflow failed: {e}")
    
    def send_input(self, text: str):
        """
        Send user input to the running workflow.
        
        Parameters
        ----------
        text : str
            User input text
        """
        self._input_queue.put(text)
        self._conversation_history.append({"role": "user", "content": text})
    
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
