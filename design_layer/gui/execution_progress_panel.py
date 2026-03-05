"""
Execution Progress Panel

Devin-style progress tracker showing:
- Current stage (e.g., "Stage 3/11: component_build")
- Progress bar (per-stage and overall)
- Time elapsed / estimated remaining
- Stage status indicators (completed, running, failed, pending)
- Real-time log output (last 10 lines)
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import time


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


@dataclass
class StageStatus:
    """Status of a pipeline stage."""
    name: str
    status: str = "pending"
    duration_s: Optional[float] = None
    error: Optional[str] = None
    
    @property
    def status_icon(self) -> str:
        icons = {
            "pending": "○",
            "running": "⟳",
            "completed": "✓",
            "failed": "✗",
            "skipped": "−",
        }
        return icons.get(self.status, "?")


class ExecutionProgressPanel(ttk.Frame):
    """
    Progress tracker showing pipeline execution status.
    
    Features:
    - Current stage indicator
    - Overall progress bar
    - Stage-by-stage status list
    - Time elapsed / estimated remaining
    - Real-time log output
    - Cancel button
    
    Parameters
    ----------
    parent : tk.Widget
        Parent widget
    on_cancel : Callable, optional
        Callback when cancel is requested
    """
    
    def __init__(
        self,
        parent: tk.Widget,
        on_cancel: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        super().__init__(parent, **kwargs)
        
        self.on_cancel = on_cancel
        
        self._stages: List[StageStatus] = [
            StageStatus(name=stage) for stage in PIPELINE_STAGES
        ]
        self._current_stage_index: int = -1
        self._start_time: Optional[float] = None
        self._is_running: bool = False
        self._log_lines: List[str] = []
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the panel UI."""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        
        header_frame = ttk.Frame(self)
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        header_frame.columnconfigure(0, weight=1)
        
        self.title_label = ttk.Label(
            header_frame,
            text="Pipeline Execution",
            font=("TkDefaultFont", 12, "bold"),
        )
        self.title_label.grid(row=0, column=0, sticky="w")
        
        self.status_label = ttk.Label(
            header_frame,
            text="Ready",
            foreground="gray",
        )
        self.status_label.grid(row=0, column=1, sticky="e")
        
        progress_frame = ttk.Frame(self)
        progress_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode="determinate",
            maximum=len(PIPELINE_STAGES),
        )
        self.progress_bar.grid(row=0, column=0, sticky="ew")
        
        self.progress_label = ttk.Label(
            progress_frame,
            text="0%",
            width=6,
        )
        self.progress_label.grid(row=0, column=1, padx=(10, 0))
        
        self.time_label = ttk.Label(
            progress_frame,
            text="Elapsed: 0s",
            foreground="gray",
        )
        self.time_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(5, 0))
        
        content_frame = ttk.Frame(self)
        content_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
        content_frame.columnconfigure(0, weight=1)
        content_frame.columnconfigure(1, weight=2)
        content_frame.rowconfigure(0, weight=1)
        
        stages_frame = ttk.LabelFrame(content_frame, text="Stages", padding=5)
        stages_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        stages_frame.columnconfigure(0, weight=1)
        stages_frame.rowconfigure(0, weight=1)
        
        self.stages_canvas = tk.Canvas(stages_frame, highlightthickness=0)
        self.stages_canvas.grid(row=0, column=0, sticky="nsew")
        
        stages_scrollbar = ttk.Scrollbar(
            stages_frame,
            orient="vertical",
            command=self.stages_canvas.yview,
        )
        stages_scrollbar.grid(row=0, column=1, sticky="ns")
        self.stages_canvas.configure(yscrollcommand=stages_scrollbar.set)
        
        self.stages_inner_frame = ttk.Frame(self.stages_canvas)
        self.stages_canvas.create_window(
            (0, 0),
            window=self.stages_inner_frame,
            anchor="nw",
        )
        
        self.stage_labels: List[ttk.Label] = []
        for i, stage in enumerate(self._stages):
            label = ttk.Label(
                self.stages_inner_frame,
                text=f"{stage.status_icon} {stage.name}",
                font=("TkDefaultFont", 9),
            )
            label.grid(row=i, column=0, sticky="w", pady=2)
            self.stage_labels.append(label)
        
        self.stages_inner_frame.update_idletasks()
        self.stages_canvas.configure(
            scrollregion=self.stages_canvas.bbox("all")
        )
        
        log_frame = ttk.LabelFrame(content_frame, text="Log Output", padding=5)
        log_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap="word",
            state="disabled",
            font=("TkFixedFont", 8),
            height=10,
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")
        
        button_frame = ttk.Frame(self)
        button_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))
        
        self.cancel_btn = ttk.Button(
            button_frame,
            text="Cancel",
            command=self._on_cancel_click,
            state="disabled",
        )
        self.cancel_btn.pack(side="right")
    
    def _on_cancel_click(self):
        """Handle cancel button click."""
        if self.on_cancel:
            self.on_cancel()
    
    def start(self):
        """Start the progress tracking."""
        self._is_running = True
        self._start_time = time.time()
        self._current_stage_index = -1
        
        for stage in self._stages:
            stage.status = "pending"
            stage.duration_s = None
            stage.error = None
        
        self._log_lines = []
        
        self.status_label.config(text="Running...", foreground="blue")
        self.cancel_btn.config(state="normal")
        self.progress_bar["value"] = 0
        self.progress_label.config(text="0%")
        
        self._update_stage_display()
        self._clear_log()
        self._update_time()
    
    def stop(self, success: bool = True, error: Optional[str] = None):
        """Stop the progress tracking."""
        self._is_running = False
        
        if success:
            self.status_label.config(text="Completed", foreground="green")
            self.progress_bar["value"] = len(PIPELINE_STAGES)
            self.progress_label.config(text="100%")
        else:
            self.status_label.config(text="Failed", foreground="red")
            if error:
                self.add_log(f"Error: {error}")
        
        self.cancel_btn.config(state="disabled")
        self._update_time()
    
    def update_progress(
        self,
        stage: str,
        stage_index: int,
        total_stages: int,
        status: str = "running",
        elapsed_s: Optional[float] = None,
        estimated_remaining_s: Optional[float] = None,
        message: Optional[str] = None,
    ):
        """
        Update the progress display.
        
        Parameters
        ----------
        stage : str
            Current stage name
        stage_index : int
            Current stage index (0-based)
        total_stages : int
            Total number of stages
        status : str
            Stage status ("running", "completed", "failed")
        elapsed_s : float, optional
            Elapsed time in seconds
        estimated_remaining_s : float, optional
            Estimated remaining time in seconds
        message : str, optional
            Optional message to display
        """
        if stage_index < len(self._stages):
            if self._current_stage_index >= 0 and self._current_stage_index < len(self._stages):
                if self._stages[self._current_stage_index].status == "running":
                    self._stages[self._current_stage_index].status = "completed"
            
            self._current_stage_index = stage_index
            self._stages[stage_index].status = status
        
        completed = sum(1 for s in self._stages if s.status == "completed")
        progress_pct = int((completed / len(PIPELINE_STAGES)) * 100)
        
        self.progress_bar["value"] = completed
        self.progress_label.config(text=f"{progress_pct}%")
        
        if elapsed_s is not None:
            time_text = f"Elapsed: {self._format_time(elapsed_s)}"
            if estimated_remaining_s is not None and estimated_remaining_s > 0:
                time_text += f" / Est. remaining: {self._format_time(estimated_remaining_s)}"
            self.time_label.config(text=time_text)
        
        if message:
            self.add_log(message)
        
        self._update_stage_display()
    
    def set_stage_completed(self, stage: str, duration_s: Optional[float] = None):
        """Mark a stage as completed."""
        for i, s in enumerate(self._stages):
            if s.name == stage:
                s.status = "completed"
                s.duration_s = duration_s
                break
        
        self._update_stage_display()
    
    def set_stage_failed(self, stage: str, error: Optional[str] = None):
        """Mark a stage as failed."""
        for i, s in enumerate(self._stages):
            if s.name == stage:
                s.status = "failed"
                s.error = error
                break
        
        self._update_stage_display()
    
    def add_log(self, message: str):
        """Add a message to the log output."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"
        
        self._log_lines.append(line)
        if len(self._log_lines) > 100:
            self._log_lines = self._log_lines[-100:]
        
        self.log_text.config(state="normal")
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")
    
    def _clear_log(self):
        """Clear the log output."""
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")
    
    def _update_stage_display(self):
        """Update the stage status display."""
        for i, (stage, label) in enumerate(zip(self._stages, self.stage_labels)):
            icon = stage.status_icon
            text = f"{icon} {stage.name}"
            
            if stage.duration_s is not None:
                text += f" ({stage.duration_s:.1f}s)"
            
            label.config(text=text)
            
            if stage.status == "completed":
                label.config(foreground="green")
            elif stage.status == "running":
                label.config(foreground="blue")
            elif stage.status == "failed":
                label.config(foreground="red")
            else:
                label.config(foreground="gray")
    
    def _update_time(self):
        """Update the elapsed time display."""
        if self._start_time is not None:
            elapsed = time.time() - self._start_time
            self.time_label.config(text=f"Elapsed: {self._format_time(elapsed)}")
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as a human-readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def reset(self):
        """Reset the panel to initial state."""
        self._is_running = False
        self._start_time = None
        self._current_stage_index = -1
        
        for stage in self._stages:
            stage.status = "pending"
            stage.duration_s = None
            stage.error = None
        
        self.status_label.config(text="Ready", foreground="gray")
        self.cancel_btn.config(state="disabled")
        self.progress_bar["value"] = 0
        self.progress_label.config(text="0%")
        self.time_label.config(text="Elapsed: 0s")
        
        self._update_stage_display()
        self._clear_log()


__all__ = [
    "ExecutionProgressPanel",
    "PIPELINE_STAGES",
    "StageStatus",
]
