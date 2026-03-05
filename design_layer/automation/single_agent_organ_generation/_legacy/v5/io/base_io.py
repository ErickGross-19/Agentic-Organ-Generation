"""
Base IO Adapter

Abstract interface for IO operations that works with both CLI and GUI.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

from ..world_model import TraceEvent


class IOMessageKind(Enum):
    """Kind of IO message."""
    ASSISTANT = "assistant"
    TRACE = "trace"
    SYSTEM = "system"
    ERROR = "error"
    WARNING = "warning"
    SUCCESS = "success"


@dataclass
class IOMessage:
    """A message to display to the user."""
    content: str
    kind: IOMessageKind = IOMessageKind.ASSISTANT
    payload: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "kind": self.kind.value,
            "payload": self.payload,
        }


@dataclass
class ApprovalRequest:
    """A request for user approval."""
    prompt: str
    details: Dict[str, Any]
    runtime_estimate: Optional[str] = None
    expected_outputs: Optional[List[str]] = None
    assumptions: Optional[List[str]] = None
    risk_flags: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "details": self.details,
            "runtime_estimate": self.runtime_estimate,
            "expected_outputs": self.expected_outputs,
            "assumptions": self.assumptions,
            "risk_flags": self.risk_flags,
        }


@dataclass
class TextPrompt:
    """A prompt for text input."""
    prompt: str
    suggestions: Optional[List[str]] = None
    default: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "suggestions": self.suggestions,
            "default": self.default,
        }


class BaseIOAdapter(ABC):
    """
    Abstract base class for IO adapters.
    
    Provides a unified interface for:
    - say(message, kind, payload) - Display a message
    - ask_confirm(prompt, details, modal) - Ask for yes/no confirmation
    - ask_text(prompt, suggestions) - Ask for text input
    - emit_trace(event) - Emit a trace event for GUI timeline
    """
    
    @abstractmethod
    def say(
        self,
        message: str,
        kind: IOMessageKind = IOMessageKind.ASSISTANT,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Display a message to the user.
        
        Parameters
        ----------
        message : str
            The message content
        kind : IOMessageKind
            The kind of message (assistant, trace, system, error, etc.)
        payload : dict, optional
            Additional data to include with the message
        """
        pass
    
    @abstractmethod
    def ask_confirm(
        self,
        prompt: str,
        details: Optional[Dict[str, Any]] = None,
        modal: bool = True,
        runtime_estimate: Optional[str] = None,
        expected_outputs: Optional[List[str]] = None,
        assumptions: Optional[List[str]] = None,
        risk_flags: Optional[List[str]] = None,
    ) -> bool:
        """
        Ask for yes/no confirmation.
        
        Parameters
        ----------
        prompt : str
            The confirmation prompt
        details : dict, optional
            Additional details to show
        modal : bool
            Whether to show as a modal dialog (GUI) or inline (CLI)
        runtime_estimate : str, optional
            Estimated runtime
        expected_outputs : list, optional
            Expected output files/types
        assumptions : list, optional
            Current assumptions
        risk_flags : list, optional
            Risk flags to highlight
            
        Returns
        -------
        bool
            True if user confirmed, False otherwise
        """
        pass
    
    @abstractmethod
    def ask_text(
        self,
        prompt: str,
        suggestions: Optional[List[str]] = None,
        default: Optional[str] = None,
    ) -> str:
        """
        Ask for text input.
        
        Parameters
        ----------
        prompt : str
            The input prompt
        suggestions : list, optional
            Suggested responses
        default : str, optional
            Default value if user provides empty input
            
        Returns
        -------
        str
            The user's input
        """
        pass
    
    @abstractmethod
    def emit_trace(self, event: TraceEvent) -> None:
        """
        Emit a trace event for GUI timeline.
        
        Parameters
        ----------
        event : TraceEvent
            The trace event to emit
        """
        pass
    
    @abstractmethod
    def show_living_spec(self, spec_summary: Dict[str, Any]) -> None:
        """
        Show the current living spec summary.
        
        Parameters
        ----------
        spec_summary : dict
            The spec summary from WorldModel.get_living_spec_summary()
        """
        pass
    
    @abstractmethod
    def show_plans(self, plans: List[Dict[str, Any]], recommended_id: Optional[str] = None) -> None:
        """
        Show proposed plans for user selection.
        
        Parameters
        ----------
        plans : list
            List of plan dictionaries
        recommended_id : str, optional
            ID of the recommended plan
        """
        pass
    
    @abstractmethod
    def prompt_plan_selection(self, plans: List[Dict[str, Any]]) -> Optional[str]:
        """
        Prompt user to select a plan.
        
        Parameters
        ----------
        plans : list
            List of plan dictionaries
            
        Returns
        -------
        str or None
            Selected plan ID, or None if user wants to continue with recommendation
        """
        pass
    
    @abstractmethod
    def show_safe_fix(
        self,
        field: str,
        before: Any,
        after: Any,
        reason: str,
    ) -> None:
        """
        Show a safe fix that was applied.
        
        Parameters
        ----------
        field : str
            The field that was changed
        before : Any
            The value before the fix
        after : Any
            The value after the fix
        reason : str
            Why the fix was applied
        """
        pass
    
    @abstractmethod
    def show_generation_ready(
        self,
        runtime_estimate: str,
        expected_outputs: List[str],
        assumptions: List[str],
        risk_flags: List[str],
    ) -> None:
        """
        Show "Ready to generate" card.
        
        Parameters
        ----------
        runtime_estimate : str
            Estimated runtime
        expected_outputs : list
            Expected output files
        assumptions : list
            Current assumptions
        risk_flags : list
            Risk flags
        """
        pass
    
    @abstractmethod
    def show_postprocess_ready(
        self,
        voxel_pitch: float,
        embedding_settings: Dict[str, Any],
        repair_steps: List[str],
        runtime_estimate: str,
        expected_outputs: List[str],
    ) -> None:
        """
        Show "Ready to postprocess" card.
        
        Parameters
        ----------
        voxel_pitch : float
            Voxel pitch setting
        embedding_settings : dict
            Embedding configuration
        repair_steps : list
            Repair steps to be run
        runtime_estimate : str
            Estimated runtime
        expected_outputs : list
            Expected output files
        """
        pass
    
    @abstractmethod
    def prompt_stl_viewer(self, stl_path: str) -> None:
        """
        Prompt user to switch to STL viewer after generation.
        
        Parameters
        ----------
        stl_path : str
            Path to the generated STL file
        """
        pass
    
    def say_assistant(self, message: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Convenience method for assistant messages."""
        self.say(message, IOMessageKind.ASSISTANT, payload)
    
    def say_trace(self, message: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Convenience method for trace messages."""
        self.say(message, IOMessageKind.TRACE, payload)
    
    def say_system(self, message: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Convenience method for system messages."""
        self.say(message, IOMessageKind.SYSTEM, payload)
    
    def say_error(self, message: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Convenience method for error messages."""
        self.say(message, IOMessageKind.ERROR, payload)
    
    def say_warning(self, message: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Convenience method for warning messages."""
        self.say(message, IOMessageKind.WARNING, payload)
    
    def say_success(self, message: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Convenience method for success messages."""
        self.say(message, IOMessageKind.SUCCESS, payload)
