"""
GUI IO Adapter

GUI implementation of the IO adapter using callbacks and queues.
Supports modal approvals and trace timeline.
"""

import queue
import threading
from typing import Any, Callable, Dict, List, Optional

from .base_io import BaseIOAdapter, IOMessageKind, TraceEvent


class GUIIOAdapter(BaseIOAdapter):
    """
    GUI implementation of the IO adapter.
    
    Uses callbacks for output and queues for input.
    Approval uses modal dialogs.
    """
    
    def __init__(
        self,
        message_callback: Optional[Callable[[str, IOMessageKind, Optional[Dict[str, Any]]], None]] = None,
        approval_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
        text_input_callback: Optional[Callable[[str, Optional[List[str]], Optional[str]], str]] = None,
        trace_callback: Optional[Callable[[TraceEvent], None]] = None,
        spec_display_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        plans_display_callback: Optional[Callable[[List[Dict[str, Any]], Optional[str]], None]] = None,
        plan_selection_callback: Optional[Callable[[List[Dict[str, Any]]], Optional[str]]] = None,
        safe_fix_callback: Optional[Callable[[str, Any, Any, str], None]] = None,
        generation_ready_callback: Optional[Callable[[str, List[str], List[str], List[str]], None]] = None,
        postprocess_ready_callback: Optional[Callable[[float, Dict[str, Any], List[str], str, List[str]], None]] = None,
        stl_viewer_callback: Optional[Callable[[str], None]] = None,
    ):
        self.message_callback = message_callback
        self.approval_callback = approval_callback
        self.text_input_callback = text_input_callback
        self.trace_callback = trace_callback
        self.spec_display_callback = spec_display_callback
        self.plans_display_callback = plans_display_callback
        self.plan_selection_callback = plan_selection_callback
        self.safe_fix_callback = safe_fix_callback
        self.generation_ready_callback = generation_ready_callback
        self.postprocess_ready_callback = postprocess_ready_callback
        self.stl_viewer_callback = stl_viewer_callback
        
        self._input_queue: queue.Queue = queue.Queue()
        self._approval_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
    
    def set_message_callback(
        self,
        callback: Callable[[str, IOMessageKind, Optional[Dict[str, Any]]], None],
    ) -> None:
        """Set the message callback."""
        self.message_callback = callback
    
    def set_approval_callback(
        self,
        callback: Callable[[Dict[str, Any]], bool],
    ) -> None:
        """Set the approval callback."""
        self.approval_callback = callback
    
    def set_text_input_callback(
        self,
        callback: Callable[[str, Optional[List[str]], Optional[str]], str],
    ) -> None:
        """Set the text input callback."""
        self.text_input_callback = callback
    
    def set_trace_callback(
        self,
        callback: Callable[[TraceEvent], None],
    ) -> None:
        """Set the trace callback."""
        self.trace_callback = callback
    
    def set_spec_display_callback(
        self,
        callback: Callable[[Dict[str, Any]], None],
    ) -> None:
        """Set the spec display callback."""
        self.spec_display_callback = callback
    
    def set_plans_display_callback(
        self,
        callback: Callable[[List[Dict[str, Any]], Optional[str]], None],
    ) -> None:
        """Set the plans display callback."""
        self.plans_display_callback = callback
    
    def set_plan_selection_callback(
        self,
        callback: Callable[[List[Dict[str, Any]]], Optional[str]],
    ) -> None:
        """Set the plan selection callback."""
        self.plan_selection_callback = callback
    
    def set_safe_fix_callback(
        self,
        callback: Callable[[str, Any, Any, str], None],
    ) -> None:
        """Set the safe fix callback."""
        self.safe_fix_callback = callback
    
    def set_generation_ready_callback(
        self,
        callback: Callable[[str, List[str], List[str], List[str]], None],
    ) -> None:
        """Set the generation ready callback."""
        self.generation_ready_callback = callback
    
    def set_postprocess_ready_callback(
        self,
        callback: Callable[[float, Dict[str, Any], List[str], str, List[str]], None],
    ) -> None:
        """Set the postprocess ready callback."""
        self.postprocess_ready_callback = callback
    
    def set_stl_viewer_callback(
        self,
        callback: Callable[[str], None],
    ) -> None:
        """Set the STL viewer callback."""
        self.stl_viewer_callback = callback
    
    def stop(self) -> None:
        """Signal to stop waiting for input."""
        self._stop_event.set()
    
    def provide_text_input(self, text: str) -> None:
        """Provide text input from GUI."""
        self._input_queue.put(text)
    
    def provide_approval(self, approved: bool) -> None:
        """Provide approval response from GUI."""
        self._approval_queue.put(approved)
    
    def say(
        self,
        message: str,
        kind: IOMessageKind = IOMessageKind.ASSISTANT,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Display a message to the user."""
        if self.message_callback:
            self.message_callback(message, kind, payload)
    
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
        """Ask for yes/no confirmation using modal dialog."""
        if self.approval_callback:
            approval_data = {
                "prompt": prompt,
                "details": details or {},
                "modal": modal,
                "runtime_estimate": runtime_estimate,
                "expected_outputs": expected_outputs or [],
                "assumptions": assumptions or [],
                "risk_flags": risk_flags or [],
            }
            return self.approval_callback(approval_data)
        
        while not self._stop_event.is_set():
            try:
                return self._approval_queue.get(timeout=0.5)
            except queue.Empty:
                continue
        
        raise KeyboardInterrupt("Workflow cancelled")
    
    def ask_text(
        self,
        prompt: str,
        suggestions: Optional[List[str]] = None,
        default: Optional[str] = None,
    ) -> str:
        """Ask for text input."""
        if self.text_input_callback:
            return self.text_input_callback(prompt, suggestions, default)
        
        self.say(prompt, IOMessageKind.ASSISTANT, {
            "suggestions": suggestions,
            "default": default,
        })
        
        while not self._stop_event.is_set():
            try:
                response = self._input_queue.get(timeout=0.5)
                if not response and default:
                    return default
                return response
            except queue.Empty:
                continue
        
        raise KeyboardInterrupt("Workflow cancelled")
    
    def emit_trace(self, event: TraceEvent) -> None:
        """Emit a trace event for GUI timeline."""
        if self.trace_callback:
            self.trace_callback(event)
    
    def show_living_spec(self, spec_summary: Dict[str, Any]) -> None:
        """Show the current living spec summary."""
        if self.spec_display_callback:
            self.spec_display_callback(spec_summary)
        else:
            self.say(
                "Living spec updated",
                IOMessageKind.SYSTEM,
                {"spec_summary": spec_summary},
            )
    
    def show_plans(self, plans: List[Dict[str, Any]], recommended_id: Optional[str] = None) -> None:
        """Show proposed plans for user selection."""
        if self.plans_display_callback:
            self.plans_display_callback(plans, recommended_id)
        else:
            self.say(
                "Plans proposed",
                IOMessageKind.ASSISTANT,
                {"plans": plans, "recommended_id": recommended_id},
            )
    
    def prompt_plan_selection(self, plans: List[Dict[str, Any]]) -> Optional[str]:
        """Prompt user to select a plan."""
        if self.plan_selection_callback:
            return self.plan_selection_callback(plans)
        
        self.say(
            "Please select a plan",
            IOMessageKind.ASSISTANT,
            {"plans": plans, "awaiting_selection": True},
        )
        
        while not self._stop_event.is_set():
            try:
                response = self._input_queue.get(timeout=0.5)
                if not response:
                    return None
                return response
            except queue.Empty:
                continue
        
        raise KeyboardInterrupt("Workflow cancelled")
    
    def show_safe_fix(
        self,
        field: str,
        before: Any,
        after: Any,
        reason: str,
    ) -> None:
        """Show a safe fix that was applied."""
        if self.safe_fix_callback:
            self.safe_fix_callback(field, before, after, reason)
        else:
            self.say(
                f"Applied safe fix: {field}: {before} -> {after}",
                IOMessageKind.SYSTEM,
                {
                    "field": field,
                    "before": before,
                    "after": after,
                    "reason": reason,
                },
            )
    
    def show_generation_ready(
        self,
        runtime_estimate: str,
        expected_outputs: List[str],
        assumptions: List[str],
        risk_flags: List[str],
    ) -> None:
        """Show 'Ready to generate' card."""
        if self.generation_ready_callback:
            self.generation_ready_callback(
                runtime_estimate,
                expected_outputs,
                assumptions,
                risk_flags,
            )
        else:
            self.say(
                "Ready to generate",
                IOMessageKind.SYSTEM,
                {
                    "runtime_estimate": runtime_estimate,
                    "expected_outputs": expected_outputs,
                    "assumptions": assumptions,
                    "risk_flags": risk_flags,
                },
            )
    
    def show_postprocess_ready(
        self,
        voxel_pitch: float,
        embedding_settings: Dict[str, Any],
        repair_steps: List[str],
        runtime_estimate: str,
        expected_outputs: List[str],
    ) -> None:
        """Show 'Ready to postprocess' card."""
        if self.postprocess_ready_callback:
            self.postprocess_ready_callback(
                voxel_pitch,
                embedding_settings,
                repair_steps,
                runtime_estimate,
                expected_outputs,
            )
        else:
            self.say(
                "Ready to postprocess",
                IOMessageKind.SYSTEM,
                {
                    "voxel_pitch": voxel_pitch,
                    "embedding_settings": embedding_settings,
                    "repair_steps": repair_steps,
                    "runtime_estimate": runtime_estimate,
                    "expected_outputs": expected_outputs,
                },
            )
    
    def prompt_stl_viewer(self, stl_path: str) -> None:
        """Prompt user to switch to STL viewer after generation."""
        if self.stl_viewer_callback:
            self.stl_viewer_callback(stl_path)
        else:
            self.say(
                f"Generation complete! STL file: {stl_path}",
                IOMessageKind.SUCCESS,
                {"stl_path": stl_path, "prompt_viewer": True},
            )
