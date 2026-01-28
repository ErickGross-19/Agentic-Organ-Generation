"""
DesignSpec GUI Panels

Provides specialized panels for the DesignSpec-first workflow:
- SpecPanel: Shows current spec with validation status
- PatchPanel: Shows proposed patches with diff view
- CompilePanel: Shows compile status and metrics
- RunPanel: Run controls with stage selection
- ArtifactsPanel: Lists generated artifacts
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
from typing import Any, Callable, Dict, List, Optional
import json


class SpecPanel(ttk.LabelFrame):
    """
    Panel showing the current DesignSpec JSON with validation status.
    
    Features:
    - Pretty-printed JSON display
    - Validation status indicator (green/yellow/red)
    - Collapsible sections for large specs
    """
    
    def __init__(
        self,
        parent: tk.Widget,
        on_edit_request: Optional[Callable[[str], None]] = None,
        **kwargs,
    ):
        super().__init__(parent, text="Spec", **kwargs)
        
        self.on_edit_request = on_edit_request
        self._spec: Dict[str, Any] = {}
        self._validation_status = "unknown"
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the panel UI."""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        
        status_frame = ttk.Frame(self)
        status_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        ttk.Label(status_frame, text="Status:").pack(side="left")
        
        self.status_label = ttk.Label(
            status_frame,
            text="Unknown",
            foreground="gray",
        )
        self.status_label.pack(side="left", padx=5)
        
        self.refresh_btn = ttk.Button(
            status_frame,
            text="Refresh",
            command=self._on_refresh,
            width=8,
        )
        self.refresh_btn.pack(side="right")
        
        self.spec_text = scrolledtext.ScrolledText(
            self,
            wrap="none",
            state="disabled",
            font=("TkFixedFont", 9),
        )
        self.spec_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        self.spec_text.tag_configure("key", foreground="blue")
        self.spec_text.tag_configure("string", foreground="green")
        self.spec_text.tag_configure("number", foreground="orange")
        self.spec_text.tag_configure("boolean", foreground="purple")
        self.spec_text.tag_configure("null", foreground="gray")
    
    def _on_refresh(self):
        """Handle refresh button click."""
        if self.on_edit_request:
            self.on_edit_request("refresh")
    
    def update_spec(self, spec: Dict[str, Any]):
        """Update the displayed spec."""
        self._spec = spec
        
        self.spec_text.config(state="normal")
        self.spec_text.delete("1.0", "end")
        
        try:
            formatted = json.dumps(spec, indent=2, default=str)
            self.spec_text.insert("1.0", formatted)
        except Exception as e:
            self.spec_text.insert("1.0", f"Error formatting spec: {e}")
        
        self.spec_text.config(state="disabled")
    
    def update_validation_status(
        self,
        valid: bool,
        errors: List[str] = None,
        warnings: List[str] = None,
    ):
        """Update the validation status display."""
        errors = errors or []
        warnings = warnings or []
        
        if valid and not errors:
            if warnings:
                self._validation_status = "warnings"
                self.status_label.config(
                    text=f"Valid ({len(warnings)} warnings)",
                    foreground="orange",
                )
            else:
                self._validation_status = "valid"
                self.status_label.config(text="Valid", foreground="green")
        else:
            self._validation_status = "invalid"
            self.status_label.config(
                text=f"Invalid ({len(errors)} errors)",
                foreground="red",
            )
    
    def get_spec(self) -> Dict[str, Any]:
        """Get the current spec."""
        return self._spec.copy()


class PatchPanel(ttk.LabelFrame):
    """
    Panel showing proposed patches with diff view.
    
    Features:
    - Patch explanation display
    - JSON diff view (before/after)
    - Approve/Reject buttons
    """
    
    def __init__(
        self,
        parent: tk.Widget,
        on_approve: Optional[Callable[[str], None]] = None,
        on_reject: Optional[Callable[[str, str], None]] = None,
        **kwargs,
    ):
        super().__init__(parent, text="Patch Proposal", **kwargs)
        
        self.on_approve = on_approve
        self.on_reject = on_reject
        self._current_patch_id: Optional[str] = None
        self._patches: Dict[str, Dict[str, Any]] = {}
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the panel UI."""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        
        header_frame = ttk.Frame(self)
        header_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        ttk.Label(header_frame, text="Pending:").pack(side="left")
        
        self.patch_count_label = ttk.Label(
            header_frame,
            text="0 patches",
            foreground="gray",
        )
        self.patch_count_label.pack(side="left", padx=5)
        
        self.patch_selector = ttk.Combobox(
            header_frame,
            state="readonly",
            width=20,
        )
        self.patch_selector.pack(side="left", padx=5)
        self.patch_selector.bind("<<ComboboxSelected>>", self._on_patch_selected)
        
        content_frame = ttk.Frame(self)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        content_frame.columnconfigure(0, weight=1)
        content_frame.rowconfigure(1, weight=1)
        
        self.explanation_label = ttk.Label(
            content_frame,
            text="No patch selected",
            wraplength=300,
        )
        self.explanation_label.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        
        self.diff_text = scrolledtext.ScrolledText(
            content_frame,
            wrap="none",
            state="disabled",
            font=("TkFixedFont", 9),
            height=10,
        )
        self.diff_text.grid(row=1, column=0, sticky="nsew")
        
        self.diff_text.tag_configure("add", foreground="green")
        self.diff_text.tag_configure("remove", foreground="red")
        self.diff_text.tag_configure("header", foreground="blue")
        
        button_frame = ttk.Frame(self)
        button_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        self.approve_btn = ttk.Button(
            button_frame,
            text="Approve",
            command=self._on_approve_click,
            state="disabled",
        )
        self.approve_btn.pack(side="left", padx=5)
        
        self.reject_btn = ttk.Button(
            button_frame,
            text="Reject",
            command=self._on_reject_click,
            state="disabled",
        )
        self.reject_btn.pack(side="left", padx=5)
        
        self.confidence_label = ttk.Label(
            button_frame,
            text="",
            foreground="gray",
        )
        self.confidence_label.pack(side="right", padx=5)
    
    def _on_patch_selected(self, event):
        """Handle patch selection."""
        selection = self.patch_selector.get()
        if selection and selection in self._patches:
            self._current_patch_id = selection
            self._display_patch(self._patches[selection])
    
    def _display_patch(self, patch_data: Dict[str, Any]):
        """Display a patch in the panel."""
        explanation = patch_data.get("explanation", "No explanation provided")
        self.explanation_label.config(text=explanation)
        
        confidence = patch_data.get("confidence", 0.0)
        self.confidence_label.config(text=f"Confidence: {confidence:.0%}")
        
        self.diff_text.config(state="normal")
        self.diff_text.delete("1.0", "end")
        
        patches = patch_data.get("patches", [])
        for i, patch in enumerate(patches):
            op = patch.get("op", "unknown")
            path = patch.get("path", "")
            value = patch.get("value", "")
            
            self.diff_text.insert("end", f"--- Patch {i+1} ---\n", "header")
            self.diff_text.insert("end", f"Operation: {op}\n")
            self.diff_text.insert("end", f"Path: {path}\n")
            
            if op == "add":
                value_str = json.dumps(value, indent=2, default=str)
                self.diff_text.insert("end", f"+ {value_str}\n", "add")
            elif op == "remove":
                self.diff_text.insert("end", f"- (removed)\n", "remove")
            elif op == "replace":
                from_value = patch.get("from", "")
                value_str = json.dumps(value, indent=2, default=str)
                self.diff_text.insert("end", f"- {from_value}\n", "remove")
                self.diff_text.insert("end", f"+ {value_str}\n", "add")
            
            self.diff_text.insert("end", "\n")
        
        self.diff_text.config(state="disabled")
        
        self.approve_btn.config(state="normal")
        self.reject_btn.config(state="normal")
    
    def _on_approve_click(self):
        """Handle approve button click."""
        if self._current_patch_id and self.on_approve:
            self.on_approve(self._current_patch_id)
            self._remove_patch(self._current_patch_id)
    
    def _on_reject_click(self):
        """Handle reject button click."""
        if self._current_patch_id and self.on_reject:
            self.on_reject(self._current_patch_id, "User rejected")
            self._remove_patch(self._current_patch_id)
    
    def _remove_patch(self, patch_id: str):
        """Remove a patch from the pending list."""
        if patch_id in self._patches:
            del self._patches[patch_id]
        
        self._update_patch_list()
        
        if not self._patches:
            self._clear_display()
    
    def _update_patch_list(self):
        """Update the patch selector dropdown."""
        patch_ids = list(self._patches.keys())
        self.patch_selector["values"] = patch_ids
        self.patch_count_label.config(text=f"{len(patch_ids)} patches")
        
        if patch_ids:
            if self._current_patch_id not in patch_ids:
                self._current_patch_id = patch_ids[0]
                self.patch_selector.set(self._current_patch_id)
                self._display_patch(self._patches[self._current_patch_id])
        else:
            self._current_patch_id = None
            self.patch_selector.set("")
    
    def _clear_display(self):
        """Clear the patch display."""
        self.explanation_label.config(text="No patch selected")
        self.confidence_label.config(text="")
        
        self.diff_text.config(state="normal")
        self.diff_text.delete("1.0", "end")
        self.diff_text.config(state="disabled")
        
        self.approve_btn.config(state="disabled")
        self.reject_btn.config(state="disabled")
    
    def add_patch(self, patch_data: Dict[str, Any]):
        """Add a new patch proposal."""
        patch_id = patch_data.get("patch_id", f"patch_{len(self._patches)}")
        self._patches[patch_id] = patch_data
        self._update_patch_list()
        
        if len(self._patches) == 1:
            self._current_patch_id = patch_id
            self.patch_selector.set(patch_id)
            self._display_patch(patch_data)
    
    def clear_patches(self):
        """Clear all pending patches."""
        self._patches.clear()
        self._current_patch_id = None
        self._update_patch_list()
        self._clear_display()


class CompilePanel(ttk.LabelFrame):
    """
    Panel showing compile status and metrics.
    
    Features:
    - Stage status indicators
    - Warnings and errors display
    - Key metrics summary
    """
    
    def __init__(
        self,
        parent: tk.Widget,
        on_compile: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        super().__init__(parent, text="Compile Status", **kwargs)
        
        self.on_compile = on_compile
        self._status = "idle"
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the panel UI."""
        self.columnconfigure(0, weight=1)
        
        status_frame = ttk.Frame(self)
        status_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        ttk.Label(status_frame, text="Status:").pack(side="left")
        
        self.status_label = ttk.Label(
            status_frame,
            text="Idle",
            foreground="gray",
        )
        self.status_label.pack(side="left", padx=5)
        
        self.compile_btn = ttk.Button(
            status_frame,
            text="Compile",
            command=self._on_compile_click,
            width=8,
        )
        self.compile_btn.pack(side="right")
        
        stages_frame = ttk.LabelFrame(self, text="Stages")
        stages_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        self.stage_labels = {}
        stages = ["compile_policies", "compile_domains"]
        for i, stage in enumerate(stages):
            frame = ttk.Frame(stages_frame)
            frame.pack(fill="x", padx=5, pady=2)
            
            indicator = ttk.Label(frame, text="[ ]", width=3)
            indicator.pack(side="left")
            
            label = ttk.Label(frame, text=stage)
            label.pack(side="left", padx=5)
            
            self.stage_labels[stage] = indicator
        
        metrics_frame = ttk.LabelFrame(self, text="Metrics")
        metrics_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        self.metrics_text = tk.Text(
            metrics_frame,
            height=4,
            state="disabled",
            font=("TkFixedFont", 9),
        )
        self.metrics_text.pack(fill="x", padx=5, pady=5)
    
    def _on_compile_click(self):
        """Handle compile button click."""
        if self.on_compile:
            self.on_compile()
    
    def update_status(self, status: str, message: str = ""):
        """Update the compile status."""
        self._status = status
        
        if status == "running":
            self.status_label.config(text="Compiling...", foreground="blue")
            self.compile_btn.config(state="disabled")
        elif status == "success":
            self.status_label.config(text="Success", foreground="green")
            self.compile_btn.config(state="normal")
        elif status == "failed":
            self.status_label.config(text=f"Failed: {message}", foreground="red")
            self.compile_btn.config(state="normal")
        else:
            self.status_label.config(text="Idle", foreground="gray")
            self.compile_btn.config(state="normal")
    
    def update_stage(self, stage: str, success: bool):
        """Update a stage status indicator."""
        if stage in self.stage_labels:
            if success:
                self.stage_labels[stage].config(text="[+]", foreground="green")
            else:
                self.stage_labels[stage].config(text="[X]", foreground="red")
    
    def update_metrics(self, metrics: Dict[str, Any]):
        """Update the metrics display."""
        self.metrics_text.config(state="normal")
        self.metrics_text.delete("1.0", "end")
        
        for key, value in metrics.items():
            self.metrics_text.insert("end", f"{key}: {value}\n")
        
        self.metrics_text.config(state="disabled")
    
    def reset(self):
        """Reset the panel to initial state."""
        self._status = "idle"
        self.status_label.config(text="Idle", foreground="gray")
        self.compile_btn.config(state="normal")
        
        for stage, label in self.stage_labels.items():
            label.config(text="[ ]", foreground="gray")
        
        self.metrics_text.config(state="normal")
        self.metrics_text.delete("1.0", "end")
        self.metrics_text.config(state="disabled")


class RunPanel(ttk.LabelFrame):
    """
    Panel for run controls with stage selection and approval buttons.
    
    Features:
    - Stage selector dropdown
    - Run Until button
    - Full Run button
    - Approve Run / Reject Run buttons for pending run approvals
    - Progress indicator
    - Pending run request display
    """
    
    def __init__(
        self,
        parent: tk.Widget,
        on_run_until: Optional[Callable[[str], None]] = None,
        on_run_full: Optional[Callable[[], None]] = None,
        on_approve_run: Optional[Callable[[], None]] = None,
        on_reject_run: Optional[Callable[[str], None]] = None,
        **kwargs,
    ):
        super().__init__(parent, text="Run", **kwargs)
        
        self.on_run_until = on_run_until
        self.on_run_full = on_run_full
        self.on_approve_run = on_approve_run
        self.on_reject_run = on_reject_run
        self._is_running = False
        self._waiting_approval = False
        self._pending_run_request: Optional[Dict[str, Any]] = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the panel UI."""
        self.columnconfigure(0, weight=1)
        
        # Stage selector frame
        stage_frame = ttk.Frame(self)
        stage_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        ttk.Label(stage_frame, text="Run until:").pack(side="left")
        
        self.stage_selector = ttk.Combobox(
            stage_frame,
            state="readonly",
            width=20,
        )
        self.stage_selector["values"] = [
            "compile_domains",
            "component_build",
            "union_voids",
            "embed",
            "validity",
            "export",
        ]
        self.stage_selector.set("union_voids")
        self.stage_selector.pack(side="left", padx=5)
        
        # Manual run buttons frame
        button_frame = ttk.Frame(self)
        button_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        self.run_until_btn = ttk.Button(
            button_frame,
            text="Run Until",
            command=self._on_run_until_click,
        )
        self.run_until_btn.pack(side="left", padx=5)
        
        self.run_full_btn = ttk.Button(
            button_frame,
            text="Full Run",
            command=self._on_run_full_click,
        )
        self.run_full_btn.pack(side="left", padx=5)
        
        # Separator
        ttk.Separator(self, orient="horizontal").grid(
            row=2, column=0, sticky="ew", padx=5, pady=5
        )
        
        # Pending run approval section
        approval_label_frame = ttk.LabelFrame(self, text="Pending Run Approval")
        approval_label_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        approval_label_frame.columnconfigure(0, weight=1)
        
        # Pending run info display
        self.pending_info_frame = ttk.Frame(approval_label_frame)
        self.pending_info_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        self.pending_stage_label = ttk.Label(
            self.pending_info_frame,
            text="Stage: -",
            foreground="gray",
        )
        self.pending_stage_label.pack(anchor="w")
        
        self.pending_reason_label = ttk.Label(
            self.pending_info_frame,
            text="Reason: -",
            foreground="gray",
            wraplength=300,
        )
        self.pending_reason_label.pack(anchor="w")
        
        self.pending_signal_label = ttk.Label(
            self.pending_info_frame,
            text="Expected: -",
            foreground="gray",
            wraplength=300,
        )
        self.pending_signal_label.pack(anchor="w")
        
        # Approve/Reject buttons frame
        approval_button_frame = ttk.Frame(approval_label_frame)
        approval_button_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        self.approve_run_btn = ttk.Button(
            approval_button_frame,
            text="Approve Run",
            command=self._on_approve_run_click,
            state="disabled",
        )
        self.approve_run_btn.pack(side="left", padx=5)
        
        self.reject_run_btn = ttk.Button(
            approval_button_frame,
            text="Reject Run",
            command=self._on_reject_run_click,
            state="disabled",
        )
        self.reject_run_btn.pack(side="left", padx=5)
        
        # Progress indicator
        self.progress = ttk.Progressbar(
            self,
            mode="indeterminate",
            length=200,
        )
        self.progress.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
        
        # Status label
        self.status_label = ttk.Label(
            self,
            text="Ready",
            foreground="gray",
        )
        self.status_label.grid(row=5, column=0, sticky="w", padx=5, pady=5)
    
    def _on_run_until_click(self):
        """Handle run until button click."""
        if self.on_run_until:
            stage = self.stage_selector.get()
            self.on_run_until(stage)
    
    def _on_run_full_click(self):
        """Handle full run button click."""
        if self.on_run_full:
            self.on_run_full()
    
    def _on_approve_run_click(self):
        """Handle approve run button click."""
        if self.on_approve_run:
            self.on_approve_run()
    
    def _on_reject_run_click(self):
        """Handle reject run button click."""
        if self.on_reject_run:
            self.on_reject_run("")
    
    def set_running(self, is_running: bool, message: str = ""):
        """Set the running state."""
        self._is_running = is_running
        
        if is_running:
            self.run_until_btn.config(state="disabled")
            self.run_full_btn.config(state="disabled")
            self.approve_run_btn.config(state="disabled")
            self.reject_run_btn.config(state="disabled")
            self.progress.start()
            self.status_label.config(text=message or "Running...", foreground="blue")
        else:
            self.run_until_btn.config(state="normal")
            self.run_full_btn.config(state="normal")
            # Only enable approval buttons if waiting for approval
            if self._waiting_approval:
                self.approve_run_btn.config(state="normal")
                self.reject_run_btn.config(state="normal")
            self.progress.stop()
            self.status_label.config(text=message or "Ready", foreground="gray")
    
    def set_waiting_approval(
        self,
        waiting: bool,
        run_request: Optional[Dict[str, Any]] = None,
    ):
        """
        Set the waiting for approval state.
        
        Parameters
        ----------
        waiting : bool
            Whether waiting for run approval
        run_request : dict, optional
            The pending run request data
        """
        self._waiting_approval = waiting
        self._pending_run_request = run_request
        
        if waiting and run_request:
            # Enable approval buttons
            self.approve_run_btn.config(state="normal")
            self.reject_run_btn.config(state="normal")
            
            # Update pending run info display
            run_until = run_request.get("run_until", "full")
            reason = run_request.get("reason", "")
            expected_signal = run_request.get("expected_signal", "")
            
            self.pending_stage_label.config(
                text=f"Stage: {run_until or 'full'}",
                foreground="blue",
            )
            self.pending_reason_label.config(
                text=f"Reason: {reason or 'Not specified'}",
                foreground="blue",
            )
            self.pending_signal_label.config(
                text=f"Expected: {expected_signal or 'Not specified'}",
                foreground="blue",
            )
            
            self.status_label.config(
                text="Waiting for run approval",
                foreground="orange",
            )
        else:
            # Disable approval buttons
            self.approve_run_btn.config(state="disabled")
            self.reject_run_btn.config(state="disabled")
            
            # Clear pending run info display
            self.pending_stage_label.config(text="Stage: -", foreground="gray")
            self.pending_reason_label.config(text="Reason: -", foreground="gray")
            self.pending_signal_label.config(text="Expected: -", foreground="gray")
            
            if not self._is_running:
                self.status_label.config(text="Ready", foreground="gray")
    
    def update_status(self, message: str, is_error: bool = False):
        """Update the status message."""
        color = "red" if is_error else "gray"
        self.status_label.config(text=message, foreground=color)


class ArtifactsPanel(ttk.LabelFrame):
    """
    Panel listing generated artifacts.
    
    Features:
    - Artifact list with types
    - Load STL button
    - Open folder button
    """
    
    def __init__(
        self,
        parent: tk.Widget,
        on_load_stl: Optional[Callable[[str], None]] = None,
        on_open_folder: Optional[Callable[[str], None]] = None,
        **kwargs,
    ):
        super().__init__(parent, text="Artifacts", **kwargs)
        
        self.on_load_stl = on_load_stl
        self.on_open_folder = on_open_folder
        self._artifacts: List[Dict[str, Any]] = []
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the panel UI."""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        
        list_frame = ttk.Frame(self)
        list_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        self.artifact_list = tk.Listbox(
            list_frame,
            selectmode="single",
            font=("TkFixedFont", 9),
        )
        self.artifact_list.grid(row=0, column=0, sticky="nsew")
        
        scrollbar = ttk.Scrollbar(
            list_frame,
            orient="vertical",
            command=self.artifact_list.yview,
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.artifact_list.config(yscrollcommand=scrollbar.set)
        
        button_frame = ttk.Frame(self)
        button_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        self.load_stl_btn = ttk.Button(
            button_frame,
            text="Load STL",
            command=self._on_load_stl_click,
            state="disabled",
        )
        self.load_stl_btn.pack(side="left", padx=5)
        
        self.open_folder_btn = ttk.Button(
            button_frame,
            text="Open Folder",
            command=self._on_open_folder_click,
            state="disabled",
        )
        self.open_folder_btn.pack(side="left", padx=5)
        
        self.artifact_list.bind("<<ListboxSelect>>", self._on_selection_change)
    
    def _on_selection_change(self, event):
        """Handle selection change."""
        selection = self.artifact_list.curselection()
        if selection:
            idx = selection[0]
            artifact = self._artifacts[idx]
            
            if artifact.get("type") == "stl":
                self.load_stl_btn.config(state="normal")
            else:
                self.load_stl_btn.config(state="disabled")
            
            self.open_folder_btn.config(state="normal")
        else:
            self.load_stl_btn.config(state="disabled")
            self.open_folder_btn.config(state="disabled")
    
    def _on_load_stl_click(self):
        """Handle load STL button click."""
        selection = self.artifact_list.curselection()
        if selection and self.on_load_stl:
            idx = selection[0]
            artifact = self._artifacts[idx]
            path = artifact.get("path", "")
            if path:
                self.on_load_stl(path)
    
    def _on_open_folder_click(self):
        """Handle open folder button click."""
        selection = self.artifact_list.curselection()
        if selection and self.on_open_folder:
            idx = selection[0]
            artifact = self._artifacts[idx]
            path = artifact.get("path", "")
            if path:
                import os
                folder = os.path.dirname(path)
                self.on_open_folder(folder)
    
    def update_artifacts(self, artifacts: List[Dict[str, Any]]):
        """Update the artifacts list."""
        self._artifacts = artifacts
        
        self.artifact_list.delete(0, "end")
        
        for artifact in artifacts:
            name = artifact.get("name", "Unknown")
            artifact_type = artifact.get("type", "unknown")
            self.artifact_list.insert("end", f"[{artifact_type}] {name}")
    
    def clear(self):
        """Clear the artifacts list."""
        self._artifacts = []
        self.artifact_list.delete(0, "end")
        self.load_stl_btn.config(state="disabled")
        self.open_folder_btn.config(state="disabled")


class LiveSpecViewer(ttk.Frame):
    """
    Hierarchical tree view of current DesignSpec with:
    - Expandable sections (domains, components, policies)
    - Color-coded status (complete, missing, invalid)
    - Diff highlighting when patches applied
    - Click to expand/collapse
    
    Parameters
    ----------
    parent : tk.Widget
        Parent widget
    on_section_click : Callable, optional
        Callback when a section is clicked
    """
    
    STATUS_COLORS = {
        "valid": "#2ecc71",
        "warning": "#f39c12",
        "error": "#e74c3c",
        "default": "#95a5a6",
        "added": "#27ae60",
        "removed": "#c0392b",
        "changed": "#f1c40f",
    }
    
    def __init__(
        self,
        parent: tk.Widget,
        on_section_click: Optional[Callable[[str], None]] = None,
        **kwargs,
    ):
        super().__init__(parent, **kwargs)
        
        self.on_section_click = on_section_click
        self._spec: Dict[str, Any] = {}
        self._validation_errors: Dict[str, List[str]] = {}
        self._validation_warnings: Dict[str, List[str]] = {}
        self._diff_paths: Dict[str, str] = {}
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the viewer UI."""
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        
        header_frame = ttk.Frame(self)
        header_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        ttk.Label(
            header_frame,
            text="DesignSpec",
            font=("TkDefaultFont", 10, "bold"),
        ).pack(side="left")
        
        self.status_indicator = ttk.Label(
            header_frame,
            text="",
            width=3,
        )
        self.status_indicator.pack(side="right")
        
        tree_frame = ttk.Frame(self)
        tree_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 5))
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        
        self.tree = ttk.Treeview(
            tree_frame,
            selectmode="browse",
            show="tree",
        )
        self.tree.grid(row=0, column=0, sticky="nsew")
        
        scrollbar = ttk.Scrollbar(
            tree_frame,
            orient="vertical",
            command=self.tree.yview,
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.tag_configure("valid", foreground=self.STATUS_COLORS["valid"])
        self.tree.tag_configure("warning", foreground=self.STATUS_COLORS["warning"])
        self.tree.tag_configure("error", foreground=self.STATUS_COLORS["error"])
        self.tree.tag_configure("default", foreground=self.STATUS_COLORS["default"])
        self.tree.tag_configure("added", foreground=self.STATUS_COLORS["added"])
        self.tree.tag_configure("removed", foreground=self.STATUS_COLORS["removed"])
        self.tree.tag_configure("changed", foreground=self.STATUS_COLORS["changed"])
        self.tree.tag_configure("section", font=("TkDefaultFont", 9, "bold"))
        
        self.tree.bind("<<TreeviewSelect>>", self._on_selection)
        self.tree.bind("<Double-1>", self._on_double_click)
    
    def _on_selection(self, event):
        """Handle tree selection."""
        selection = self.tree.selection()
        if selection and self.on_section_click:
            item_id = selection[0]
            path = self._get_item_path(item_id)
            self.on_section_click(path)
    
    def _on_double_click(self, event):
        """Handle double-click to expand/collapse."""
        item_id = self.tree.identify_row(event.y)
        if item_id:
            if self.tree.item(item_id, "open"):
                self.tree.item(item_id, open=False)
            else:
                self.tree.item(item_id, open=True)
    
    def _get_item_path(self, item_id: str) -> str:
        """Get the JSON path for a tree item."""
        path_parts = []
        current = item_id
        
        while current:
            text = self.tree.item(current, "text")
            if ":" in text:
                text = text.split(":")[0].strip()
            path_parts.insert(0, text)
            current = self.tree.parent(current)
        
        return "/".join(path_parts)
    
    def update_spec(self, spec: Dict[str, Any]):
        """Update the displayed spec."""
        self._spec = spec
        self._rebuild_tree()
    
    def update_validation(
        self,
        errors: Dict[str, List[str]] = None,
        warnings: Dict[str, List[str]] = None,
    ):
        """Update validation status for sections."""
        self._validation_errors = errors or {}
        self._validation_warnings = warnings or {}
        self._update_status_colors()
    
    def show_diff(self, diff_paths: Dict[str, str]):
        """
        Show diff highlighting for changed paths.
        
        Parameters
        ----------
        diff_paths : dict
            Mapping of JSON paths to diff type ("added", "removed", "changed")
        """
        self._diff_paths = diff_paths
        self._update_status_colors()
    
    def clear_diff(self):
        """Clear diff highlighting."""
        self._diff_paths = {}
        self._update_status_colors()
    
    def _rebuild_tree(self):
        """Rebuild the tree from the spec."""
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        if not self._spec:
            return
        
        sections = [
            ("meta", "Meta"),
            ("domains", "Domains"),
            ("components", "Components"),
            ("policies", "Policies"),
            ("features", "Features"),
        ]
        
        for key, label in sections:
            if key in self._spec:
                self._add_section(key, label, self._spec[key])
        
        self._update_status_colors()
    
    def _add_section(self, key: str, label: str, data: Any, parent: str = ""):
        """Add a section to the tree."""
        if data is None:
            return
        
        section_id = self.tree.insert(
            parent,
            "end",
            text=label,
            tags=("section",),
            open=True,
        )
        
        if isinstance(data, dict):
            for sub_key, sub_value in data.items():
                self._add_item(section_id, sub_key, sub_value, f"{key}.{sub_key}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict) and "id" in item:
                    item_label = f"{i}: {item['id']}"
                else:
                    item_label = f"[{i}]"
                self._add_item(section_id, item_label, item, f"{key}[{i}]")
    
    def _add_item(self, parent: str, key: str, value: Any, path: str):
        """Add an item to the tree."""
        if isinstance(value, dict):
            item_id = self.tree.insert(
                parent,
                "end",
                text=key,
                open=False,
            )
            for sub_key, sub_value in value.items():
                self._add_item(item_id, sub_key, sub_value, f"{path}.{sub_key}")
        elif isinstance(value, list):
            if len(value) <= 3 and all(not isinstance(v, (dict, list)) for v in value):
                display_value = str(value)
                self.tree.insert(parent, "end", text=f"{key}: {display_value}")
            else:
                item_id = self.tree.insert(
                    parent,
                    "end",
                    text=f"{key} [{len(value)} items]",
                    open=False,
                )
                for i, item in enumerate(value):
                    self._add_item(item_id, f"[{i}]", item, f"{path}[{i}]")
        else:
            display_value = self._format_value(value)
            self.tree.insert(parent, "end", text=f"{key}: {display_value}")
    
    def _format_value(self, value: Any) -> str:
        """Format a value for display."""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, float):
            if abs(value) < 0.0001 or abs(value) > 10000:
                return f"{value:.6e}"
            return f"{value:.6f}".rstrip("0").rstrip(".")
        elif isinstance(value, str):
            if len(value) > 50:
                return f'"{value[:47]}..."'
            return f'"{value}"'
        return str(value)
    
    def _update_status_colors(self):
        """Update status colors based on validation and diff."""
        has_errors = bool(self._validation_errors)
        has_warnings = bool(self._validation_warnings)
        
        if has_errors:
            self.status_indicator.config(
                text="X",
                foreground=self.STATUS_COLORS["error"],
            )
        elif has_warnings:
            self.status_indicator.config(
                text="!",
                foreground=self.STATUS_COLORS["warning"],
            )
        else:
            self.status_indicator.config(
                text="",
                foreground=self.STATUS_COLORS["valid"],
            )
    
    def scroll_to_path(self, path: str):
        """Scroll to and select a specific path in the tree."""
        pass
    
    def expand_all(self):
        """Expand all tree nodes."""
        def expand_recursive(item):
            self.tree.item(item, open=True)
            for child in self.tree.get_children(item):
                expand_recursive(child)
        
        for item in self.tree.get_children():
            expand_recursive(item)
    
    def collapse_all(self):
        """Collapse all tree nodes."""
        def collapse_recursive(item):
            self.tree.item(item, open=False)
            for child in self.tree.get_children(item):
                collapse_recursive(child)
        
        for item in self.tree.get_children():
            collapse_recursive(item)


__all__ = [
    "ArtifactsPanel",
    "CompilePanel",
    "LiveSpecViewer",
    "PatchPanel",
    "RunPanel",
    "SpecPanel",
]
