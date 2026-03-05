"""
Workflow Manager

Provides shared utilities for workflow execution in the GUI.
The primary workflow is DesignSpec, managed by DesignSpecWorkflowManager.
"""

import io
import threading
import queue
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable, List
from enum import Enum


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
    workflow_type: WorkflowType = WorkflowType.DESIGNSPEC
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
            from automation.agent_runner import create_agent
            
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
    
    def send_input(self, text: str):
        """
        Send user input to the running workflow.
        
        Parameters
        ----------
        text : str
            User input text
        """
        self._conversation_history.append({"role": "user", "content": text})
        self._input_queue.put(text)
    
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
