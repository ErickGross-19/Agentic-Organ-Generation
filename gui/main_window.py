"""
Main Window

Primary application window for the Organ Generator GUI.
Provides workflow selection, agent configuration, and three-panel layout
with chat, output, and STL viewer.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
from typing import Optional, Dict, Any
import os
import sys
from pathlib import Path
from datetime import datetime

from .security import SecureConfig
from .agent_config import AgentConfigPanel, AgentConfiguration
from .workflow_manager import WorkflowManager, WorkflowConfig, WorkflowType, WorkflowStatus, WorkflowMessage
from .stl_viewer import STLViewer


class WorkflowSelectionDialog(tk.Toplevel):
    """
    Dialog for selecting workflow type and configuration.
    
    Parameters
    ----------
    parent : tk.Widget
        Parent widget
    callback : Callable
        Callback with selected workflow type
    """
    
    def __init__(self, parent: tk.Widget, callback):
        super().__init__(parent)
        
        self.callback = callback
        self.result: Optional[WorkflowType] = None
        
        self.title("Select Workflow")
        self.geometry("400x300")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        
        self._setup_ui()
        
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")
    
    def _setup_ui(self):
        """Set up dialog UI."""
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill="both", expand=True)
        
        ttk.Label(
            main_frame,
            text="Select Workflow Type",
            font=("TkDefaultFont", 14, "bold"),
        ).pack(pady=(0, 20))
        
        self.workflow_var = tk.StringVar(value="single_agent")
        
        single_frame = ttk.Frame(main_frame)
        single_frame.pack(fill="x", pady=5)
        
        ttk.Radiobutton(
            single_frame,
            text="Single Agent Organ Generator",
            variable=self.workflow_var,
            value="single_agent",
        ).pack(anchor="w")
        
        ttk.Label(
            single_frame,
            text="Interactive workflow with topology-first questioning.\nBest for guided organ structure design.",
            foreground="gray",
        ).pack(anchor="w", padx=20)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(30, 0))
        
        ttk.Button(
            button_frame,
            text="Cancel",
            command=self._on_cancel,
        ).pack(side="right", padx=5)
        
        ttk.Button(
            button_frame,
            text="Start",
            command=self._on_start,
        ).pack(side="right", padx=5)
    
    def _on_start(self):
        """Handle start button click."""
        workflow_type = WorkflowType(self.workflow_var.get())
        self.result = workflow_type
        self.callback(workflow_type)
        self.destroy()
    
    def _on_cancel(self):
        """Handle cancel button click."""
        self.result = None
        self.destroy()


class MainWindow:
    """
    Main application window for Organ Generator GUI.
    
    Provides:
    - Workflow selection and control
    - Agent configuration panel
    - Three-panel layout: Chat, Output, STL Viewer
    - Status bar with workflow status
    
    Parameters
    ----------
    title : str
        Window title
    width : int
        Window width
    height : int
        Window height
    """
    
    def __init__(
        self,
        title: str = "Organ Generator v1.0",
        width: int = 1200,
        height: int = 800,
    ):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(800, 600)
        
        self.secure_config = SecureConfig()
        
        self.workflow_manager = WorkflowManager(
            message_callback=self._on_workflow_message,
            status_callback=self._on_status_change,
            output_callback=self._on_output,
        )
        
        self._current_workflow_type: Optional[WorkflowType] = None
        self._agent_config: Optional[AgentConfiguration] = None
        
        # Plan selection state (for in-chat selection)
        self._awaiting_plan_selection: bool = False
        self._pending_plan_selection: set = set()
        
        self._setup_menu()
        self._setup_ui()
        self._setup_bindings()
        
        self._load_window_state()
        
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _setup_menu(self):
        """Set up menu bar."""
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)
        
        file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Workflow", command=self._show_workflow_selection)
        file_menu.add_command(label="Open Project...", command=self._open_project)
        file_menu.add_separator()
        file_menu.add_command(label="Load STL...", command=self._load_stl)
        file_menu.add_command(label="Export View...", command=self._export_view)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        
        workflow_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Workflow", menu=workflow_menu)
        workflow_menu.add_command(label="Start", command=self._start_workflow)
        workflow_menu.add_command(label="Stop", command=self._stop_workflow)
        workflow_menu.add_separator()
        workflow_menu.add_command(label="Clear Chat", command=self._clear_chat)
        workflow_menu.add_command(label="Clear Output", command=self._clear_output)
        
        help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Documentation", command=self._show_docs)
        help_menu.add_command(label="About", command=self._show_about)
    
    def _setup_ui(self):
        """Set up main UI layout."""
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        
        self._setup_toolbar()
        self._setup_main_panels()
        self._setup_status_bar()
    
    def _setup_toolbar(self):
        """Set up toolbar."""
        toolbar = ttk.Frame(self.main_frame)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        
        self.workflow_btn = ttk.Button(
            toolbar,
            text="Select Workflow",
            command=self._show_workflow_selection,
        )
        self.workflow_btn.pack(side="left", padx=2)
        
        self.start_btn = ttk.Button(
            toolbar,
            text="Start",
            command=self._start_workflow,
            state="disabled",
        )
        self.start_btn.pack(side="left", padx=2)
        
        self.stop_btn = ttk.Button(
            toolbar,
            text="Stop",
            command=self._stop_workflow,
            state="disabled",
        )
        self.stop_btn.pack(side="left", padx=2)
        
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=10)
        
        self.workflow_label = ttk.Label(toolbar, text="No workflow selected")
        self.workflow_label.pack(side="left", padx=5)
        
        self.config_btn = ttk.Button(
            toolbar,
            text="Agent Config",
            command=self._show_agent_config,
        )
        self.config_btn.pack(side="right", padx=2)
        
        # P3 #26: View master script button
        self.view_script_btn = ttk.Button(
            toolbar,
            text="View Script",
            command=self._view_master_script,
            state="disabled",
        )
        self.view_script_btn.pack(side="right", padx=2)
        
        # P3 #29: Run history selector
        self.run_history_btn = ttk.Button(
            toolbar,
            text="Run History",
            command=self._show_run_history,
            state="disabled",
        )
        self.run_history_btn.pack(side="right", padx=2)
        
        # P3 #31: Verification report button
        self.verification_btn = ttk.Button(
            toolbar,
            text="Verification",
            command=self._show_verification_report,
            state="disabled",
        )
        self.verification_btn.pack(side="right", padx=2)
    
    def _setup_main_panels(self):
        """Set up three-panel layout."""
        paned = ttk.PanedWindow(self.main_frame, orient="horizontal")
        paned.grid(row=1, column=0, sticky="nsew")
        
        left_paned = ttk.PanedWindow(paned, orient="vertical")
        paned.add(left_paned, weight=2)
        
        chat_frame = ttk.LabelFrame(left_paned, text="Chat")
        left_paned.add(chat_frame, weight=1)
        
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        self.chat_text = scrolledtext.ScrolledText(
            chat_frame,
            wrap="word",
            state="disabled",
            font=("TkFixedFont", 10),
        )
        self.chat_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.chat_text.tag_configure("system", foreground="blue")
        self.chat_text.tag_configure("user", foreground="green")
        self.chat_text.tag_configure("assistant", foreground="purple")
        self.chat_text.tag_configure("error", foreground="red")
        self.chat_text.tag_configure("success", foreground="darkgreen")
        self.chat_text.tag_configure("prompt", foreground="orange")
        
        input_frame = ttk.Frame(chat_frame)
        input_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        input_frame.columnconfigure(0, weight=1)
        
        self.input_var = tk.StringVar()
        self.input_entry = ttk.Entry(
            input_frame,
            textvariable=self.input_var,
            font=("TkFixedFont", 10),
        )
        self.input_entry.grid(row=0, column=0, sticky="ew")
        
        self.send_btn = ttk.Button(
            input_frame,
            text="Send",
            command=self._send_input,
            state="normal",
        )
        self.send_btn.grid(row=0, column=1, padx=(5, 0))
        
        output_frame = ttk.LabelFrame(left_paned, text="Output")
        left_paned.add(output_frame, weight=1)
        
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        
        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            wrap="word",
            state="disabled",
            font=("TkFixedFont", 9),
        )
        self.output_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        output_toolbar = ttk.Frame(output_frame)
        output_toolbar.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        self.current_file_var = tk.StringVar(value="No file")
        ttk.Label(output_toolbar, textvariable=self.current_file_var).pack(side="left")
        
        ttk.Button(
            output_toolbar,
            text="Open Folder",
            command=self._open_output_folder,
        ).pack(side="right")
        
        viewer_frame = ttk.LabelFrame(paned, text="STL Viewer")
        paned.add(viewer_frame, weight=1)
        
        self.stl_viewer = STLViewer(viewer_frame, width=400, height=400)
        self.stl_viewer.pack(fill="both", expand=True, padx=5, pady=5)
    
    def _setup_status_bar(self):
        """Set up status bar."""
        status_frame = ttk.Frame(self.main_frame)
        status_frame.grid(row=2, column=0, sticky="ew", pady=(5, 0))
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.status_label.pack(side="left")
        
        self.progress = ttk.Progressbar(
            status_frame,
            mode="indeterminate",
            length=100,
        )
        self.progress.pack(side="right", padx=5)
    
    def _setup_bindings(self):
        """Set up keyboard bindings."""
        self.input_entry.bind("<Return>", lambda e: self._send_input())
        self.root.bind("<Control-n>", lambda e: self._show_workflow_selection())
        self.root.bind("<Control-q>", lambda e: self._on_close())
    
    def _show_workflow_selection(self):
        """Show workflow selection dialog."""
        dialog = WorkflowSelectionDialog(self.root, self._on_workflow_selected)
        self.root.wait_window(dialog)
    
    def _on_workflow_selected(self, workflow_type: WorkflowType):
        """Handle workflow selection."""
        self._current_workflow_type = workflow_type
        
        if workflow_type == WorkflowType.SINGLE_AGENT:
            self.workflow_label.config(text="Single Agent Organ Generator")
        
        self.start_btn.config(state="normal")
        self._append_chat("system", f"Selected workflow: {workflow_type.value}")
    
    def _show_agent_config(self):
        """Show agent configuration dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Agent Configuration")
        dialog.geometry("450x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        config_panel = AgentConfigPanel(
            dialog,
            self.secure_config,
            on_config_change=self._on_agent_config_change,
        )
        config_panel.pack(fill="both", expand=True, padx=10, pady=10)
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill="x", padx=10, pady=10)
        
        def on_save():
            config_panel.save_current_config()
            is_valid, error = config_panel.validate_configuration()
            if is_valid:
                self._agent_config = config_panel.get_configuration()
                self._append_chat("system", f"Agent configured: {self._agent_config.provider}/{self._agent_config.model}")
                dialog.destroy()
            else:
                messagebox.showwarning("Validation Error", error)
        
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side="right", padx=5)
        ttk.Button(button_frame, text="Save", command=on_save).pack(side="right", padx=5)
        
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
    
    def _on_agent_config_change(self, config: AgentConfiguration):
        """Handle agent configuration change."""
        self._agent_config = config
    
    def _start_workflow(self):
        """Start the selected workflow."""
        if self._current_workflow_type is None:
            messagebox.showwarning("Warning", "Please select a workflow first.")
            return
        
        if self._agent_config is None:
            messagebox.showwarning("Warning", "Please configure the agent first.")
            self._show_agent_config()
            return
        
        is_valid, error = self._validate_agent_config()
        if not is_valid:
            messagebox.showerror("Configuration Error", error)
            return
        
        if not self.workflow_manager.initialize_agent(
            provider=self._agent_config.provider,
            api_key=self._agent_config.api_key,
            model=self._agent_config.model,
            api_base=self._agent_config.api_base,
            temperature=self._agent_config.temperature,
            max_tokens=self._agent_config.max_tokens,
        ):
            return
        
        output_dir = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=os.path.expanduser("~/projects"),
        )
        
        if not output_dir:
            return
        
        config = WorkflowConfig(
            workflow_type=self._current_workflow_type,
            output_dir=output_dir,
            execution_mode="review_then_run",
            timeout_seconds=300.0,
            verbose=True,
            llm_first_mode=True,  # Default to LLM-first mode in GUI
        )
        
        if self.workflow_manager.start_workflow(config):
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            # Send button is always enabled - _send_input handles the no-workflow case
            self.progress.start()
            # P3: Enable workspace-related buttons when workflow starts
            self.view_script_btn.config(state="normal")
            self.run_history_btn.config(state="normal")
            self.verification_btn.config(state="normal")
    
    def _validate_agent_config(self) -> tuple:
        """Validate agent configuration."""
        if self._agent_config is None:
            return False, "Agent not configured"
        
        if self._agent_config.provider == "local":
            if not self._agent_config.api_base:
                return False, "API Base URL required for local provider"
        else:
            if not self._agent_config.api_key:
                env_vars = {
                    "openai": "OPENAI_API_KEY",
                    "anthropic": "ANTHROPIC_API_KEY",
                    "google": "GOOGLE_API_KEY",
                    "mistral": "MISTRAL_API_KEY",
                    "xai": "XAI_API_KEY",
                    "groq": "GROQ_API_KEY",
                }
                env_var = env_vars.get(self._agent_config.provider)
                if env_var and not os.environ.get(env_var):
                    return False, f"API key required. Enter key or set {env_var}"
        
        return True, ""
    
    def _stop_workflow(self):
        """Stop the running workflow."""
        self.workflow_manager.stop_workflow()
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        # Send button stays enabled - _send_input handles the no-workflow case
        # P3: Disable workspace-related buttons when workflow stops
        self.view_script_btn.config(state="disabled")
        self.run_history_btn.config(state="disabled")
        self.verification_btn.config(state="disabled")
        self.progress.stop()
    
    def _send_input(self):
        """Send user input to workflow."""
        try:
            # Use entry.get() directly as primary method (more reliable than StringVar)
            # Fall back to StringVar if entry.get() fails
            try:
                text = self.input_entry.get().strip()
            except Exception:
                text = self.input_var.get().strip()
            
            # Check if workflow is running
            if not self.workflow_manager.is_running:
                self._append_chat("system", "Please start a workflow first (File > New Workflow or Ctrl+N)")
                return
            
            # Handle plan selection validation (in-chat selection)
            if self._awaiting_plan_selection:
                # Empty input means use recommended plan
                if not text:
                    self._append_chat("user", "(using recommended)")
                    self.workflow_manager.send_input("")
                    self._awaiting_plan_selection = False
                    self._pending_plan_selection = set()
                # Validate the plan ID
                elif text in self._pending_plan_selection:
                    self._append_chat("user", text)
                    self.workflow_manager.send_input(text)
                    self._awaiting_plan_selection = False
                    self._pending_plan_selection = set()
                else:
                    # Invalid plan ID - show error and re-prompt without returning to controller
                    self._append_chat("error", f"Invalid plan ID: '{text}'")
                    valid_ids = ", ".join(sorted(self._pending_plan_selection))
                    self._append_chat("system", f"Valid plan IDs are: {valid_ids}")
                    self._append_chat("system", "Please enter a valid plan ID or press Enter for recommended.")
                    # Don't clear awaiting state - let user try again
                
                # Clear the input field
                self.input_var.set("")
                self.input_entry.delete(0, "end")
                return
            
            # Normal input handling
            # Provide feedback even if text is empty (never fail silently)
            if not text:
                self._append_chat("system", "Please enter some text to send.")
                return
            
            # Send the input
            self._append_chat("user", text)
            self.workflow_manager.send_input(text)
            
            # Clear the input field using both methods for reliability
            self.input_var.set("")
            self.input_entry.delete(0, "end")
            
        except Exception as e:
            self._append_chat("error", f"Failed to send input: {e}")
    
    def _on_workflow_message(self, message: WorkflowMessage):
        """
        Handle message from workflow.
        
        V5 workflows emit structured messages with payloads that require
        special handling:
        - approval_request: Show modal approval dialog
        - plan_selection: Show plan selection dialog
        - generation_ready: Show generation approval card
        - postprocess_ready: Show postprocess approval card
        - spec_update: Update living spec display (if panel exists)
        - plans: Display proposed plans
        - safe_fix: Show safe fix notification
        """
        def handle_message():
            msg_type = message.type
            content = message.content
            data = message.data
            
            if msg_type == "approval_request" and data:
                self._show_approval_dialog(content, data)
            elif msg_type == "plan_selection" and data:
                self._show_plan_selection_dialog(data.get("plans", []))
            elif msg_type == "generation_ready" and data:
                self._show_generation_ready_card(data)
            elif msg_type == "postprocess_ready" and data:
                self._show_postprocess_ready_card(data)
            elif msg_type == "spec_update" and data:
                self._update_spec_display(data)
            elif msg_type == "plans" and data:
                self._display_plans(data.get("plans", []), data.get("recommended_id"))
            elif msg_type == "safe_fix" and data:
                self._show_safe_fix_notification(data)
            else:
                self._append_chat(msg_type, content)
        
        self.root.after(0, handle_message)
    
    def _show_approval_dialog(self, prompt: str, data: dict):
        """Show modal approval dialog for V5 workflows."""
        result = messagebox.askyesno(
            "Approval Required",
            prompt,
            icon="question"
        )
        self.workflow_manager.send_approval(result)
        self._append_chat("system", f"Approval: {'Approved' if result else 'Rejected'}")
    
    def _show_plan_selection_dialog(self, plans: list):
        """Show plan selection in chat (no popup) for V5 workflows."""
        if not plans:
            self.workflow_manager.send_input("")
            return
        
        # Build plan display for chat - use plan_id (not id)
        valid_plan_ids = set()
        plan_text = "Please select a plan by typing its ID:\n"
        
        for plan in plans:
            plan_id = plan.get("plan_id", plan.get("id", ""))
            valid_plan_ids.add(plan_id)
            name = plan.get("name", "Unknown")
            interpretation = plan.get("interpretation", "")
            cost = plan.get("cost_estimate", "")
            risks = plan.get("risks", [])
            is_recommended = plan.get("recommended", False)
            
            rec_marker = " (RECOMMENDED)" if is_recommended else ""
            plan_text += f"\n  [{plan_id}]{rec_marker} {name}"
            if interpretation:
                plan_text += f"\n    {interpretation}"
            if cost:
                plan_text += f"\n    Estimated: {cost}"
            if risks:
                plan_text += f"\n    Risks: {', '.join(risks[:2])}"
                if len(risks) > 2:
                    plan_text += f" (+{len(risks)-2} more)"
        
        plan_text += "\n\nType a plan ID (e.g., 'tree_balanced') or press Enter for recommended:"
        
        # Display in chat and store valid IDs for validation
        self._append_chat("prompt", plan_text)
        self._pending_plan_selection = valid_plan_ids
        self._awaiting_plan_selection = True
    
    def _show_generation_ready_card(self, data: dict):
        """Show generation ready notification with approval."""
        runtime = data.get("runtime_estimate", "unknown")
        outputs = data.get("expected_outputs", [])
        assumptions = data.get("assumptions", [])
        risks = data.get("risk_flags", [])
        
        info_text = f"Ready to generate\n\nEstimated runtime: {runtime}"
        if outputs:
            info_text += f"\n\nExpected outputs:\n- " + "\n- ".join(outputs)
        if assumptions:
            info_text += f"\n\nAssumptions:\n- " + "\n- ".join(assumptions)
        if risks:
            info_text += f"\n\nRisk flags:\n- " + "\n- ".join(risks)
        
        self._append_chat("generation_ready", info_text)
    
    def _show_postprocess_ready_card(self, data: dict):
        """Show postprocess ready notification."""
        runtime = data.get("runtime_estimate", "unknown")
        outputs = data.get("expected_outputs", [])
        steps = data.get("repair_steps", [])
        
        info_text = f"Ready to postprocess\n\nEstimated runtime: {runtime}"
        if steps:
            info_text += f"\n\nRepair steps:\n- " + "\n- ".join(steps)
        if outputs:
            info_text += f"\n\nExpected outputs:\n- " + "\n- ".join(outputs)
        
        self._append_chat("postprocess_ready", info_text)
    
    def _update_spec_display(self, spec_data: dict):
        """Update living spec display panel (if exists)."""
        spec_text = "Living Spec:\n"
        for key, value in spec_data.items():
            spec_text += f"  {key}: {value}\n"
        self._append_chat("spec", spec_text)
    
    def _display_plans(self, plans: list, recommended_id: str):
        """Display proposed plans in chat."""
        plans_text = "Proposed Plans:\n"
        for plan in plans:
            # Use plan_id (not id) - this is the correct key from Plan.to_dict()
            plan_id = plan.get("plan_id", plan.get("id", ""))
            name = plan.get("name", "")
            is_recommended = " (RECOMMENDED)" if plan_id == recommended_id else ""
            plans_text += f"\n  [{plan_id}] {name}{is_recommended}"
            if plan.get("interpretation"):
                plans_text += f"\n      {plan.get('interpretation')}"
        self._append_chat("plans", plans_text)
    
    def _show_safe_fix_notification(self, data: dict):
        """Show safe fix notification."""
        field = data.get("field", "")
        before = data.get("before", "")
        after = data.get("after", "")
        reason = data.get("reason", "")
        
        fix_text = f"Safe fix applied: {field}\n  {before} -> {after}"
        if reason:
            fix_text += f"\n  Reason: {reason}"
        self._append_chat("safe_fix", fix_text)
    
    def _on_status_change(self, status: WorkflowStatus, message: str):
        """Handle workflow status change."""
        def update():
            self.status_var.set(f"{status.value}: {message}" if message else status.value)
            
            if status in (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED):
                self.start_btn.config(state="normal")
                self.stop_btn.config(state="disabled")
                # Send button stays enabled - _send_input handles the no-workflow case
                self.progress.stop()
            elif status == WorkflowStatus.WAITING_INPUT:
                self.input_entry.focus_set()
        
        self.root.after(0, update)
    
    def _on_output(self, output_type: str, data: Any):
        """Handle output from workflow."""
        def update():
            if output_type == "stl_file":
                self._append_output(f"Generated STL: {data}")
                self.stl_viewer.load_stl(data)
                self.current_file_var.set(os.path.basename(data))
            else:
                self._append_output(f"{output_type}: {data}")
        
        self.root.after(0, update)
    
    def _append_chat(self, msg_type: str, content: str):
        """Append message to chat display."""
        self.chat_text.config(state="normal")
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = f"[{timestamp}] "
        
        if msg_type == "user":
            prefix += "You: "
        elif msg_type == "system":
            prefix += "System: "
        elif msg_type == "prompt":
            prefix += ">>> "
        elif msg_type == "error":
            prefix += "Error: "
        elif msg_type == "success":
            prefix += "Success: "
        else:
            prefix += f"{msg_type}: "
        
        self.chat_text.insert("end", prefix, msg_type)
        self.chat_text.insert("end", content + "\n", msg_type)
        self.chat_text.see("end")
        self.chat_text.config(state="disabled")
    
    def _append_output(self, content: str):
        """Append content to output display."""
        self.output_text.config(state="normal")
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.output_text.insert("end", f"[{timestamp}] {content}\n")
        self.output_text.see("end")
        self.output_text.config(state="disabled")
    
    def _clear_chat(self):
        """Clear chat display."""
        self.chat_text.config(state="normal")
        self.chat_text.delete("1.0", "end")
        self.chat_text.config(state="disabled")
    
    def _clear_output(self):
        """Clear output display."""
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.config(state="disabled")
    
    def _open_project(self):
        """Open existing project directory."""
        project_dir = filedialog.askdirectory(
            title="Select Project Directory",
        )
        
        if project_dir:
            stl_files = list(Path(project_dir).glob("**/*.stl"))
            if stl_files:
                self.stl_viewer.load_stl(str(stl_files[0]))
                self._append_output(f"Loaded project: {project_dir}")
                self._append_output(f"Found {len(stl_files)} STL file(s)")
    
    def _load_stl(self):
        """Load STL file directly."""
        file_path = filedialog.askopenfilename(
            title="Select STL File",
            filetypes=[("STL files", "*.stl"), ("All files", "*.*")],
        )
        
        if file_path:
            self.stl_viewer.load_stl(file_path)
    
    def _export_view(self):
        """Export current STL view as image."""
        file_path = filedialog.asksaveasfilename(
            title="Export View",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
            ],
        )
        
        if file_path:
            if self.stl_viewer.export_image(file_path):
                messagebox.showinfo("Success", f"View exported to {file_path}")
            else:
                messagebox.showerror("Error", "Failed to export view")
    
    def _open_output_folder(self):
        """Open output folder in file manager."""
        artifacts = self.workflow_manager.get_artifacts()
        output_dir = artifacts.get("output_dir")
        
        if output_dir and os.path.exists(output_dir):
            if sys.platform == "darwin":
                os.system(f'open "{output_dir}"')
            elif sys.platform == "win32":
                os.startfile(output_dir)
            else:
                os.system(f'xdg-open "{output_dir}"')
        else:
            messagebox.showinfo("Info", "No output folder available")
    
    def _view_master_script(self):
        """P3 #26: View master script in a read-only viewer."""
        artifacts = self.workflow_manager.get_artifacts()
        workspace_path = artifacts.get("workspace_path")
        
        if not workspace_path:
            messagebox.showinfo("Info", "No workspace available")
            return
        
        master_path = os.path.join(workspace_path, "master.py")
        if not os.path.exists(master_path):
            messagebox.showinfo("Info", "No master script found")
            return
        
        try:
            with open(master_path, 'r') as f:
                script_content = f.read()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read master script: {e}")
            return
        
        viewer = tk.Toplevel(self.root)
        viewer.title("Master Script Viewer")
        viewer.geometry("800x600")
        
        text_widget = scrolledtext.ScrolledText(
            viewer,
            wrap="none",
            font=("TkFixedFont", 10),
        )
        text_widget.pack(fill="both", expand=True, padx=5, pady=5)
        text_widget.insert("1.0", script_content)
        text_widget.config(state="disabled")
    
    def _show_run_history(self):
        """P3 #29: Show run history selector."""
        artifacts = self.workflow_manager.get_artifacts()
        workspace_path = artifacts.get("workspace_path")
        
        if not workspace_path:
            messagebox.showinfo("Info", "No workspace available")
            return
        
        runs_dir = os.path.join(workspace_path, "runs")
        if not os.path.exists(runs_dir):
            messagebox.showinfo("Info", "No runs found")
            return
        
        run_dirs = sorted([
            d for d in os.listdir(runs_dir)
            if os.path.isdir(os.path.join(runs_dir, d)) and d.startswith("run_")
        ], reverse=True)
        
        if not run_dirs:
            messagebox.showinfo("Info", "No runs found")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Run History")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Select a run to inspect:", font=("TkDefaultFont", 12, "bold")).pack(pady=10)
        
        listbox = tk.Listbox(dialog, font=("TkFixedFont", 10))
        listbox.pack(fill="both", expand=True, padx=10, pady=5)
        
        for run_dir in run_dirs:
            listbox.insert("end", run_dir)
        
        def on_select():
            selection = listbox.curselection()
            if not selection:
                return
            run_name = listbox.get(selection[0])
            run_path = os.path.join(runs_dir, run_name)
            dialog.destroy()
            self._inspect_run(run_path)
        
        ttk.Button(dialog, text="Inspect", command=on_select).pack(pady=10)
    
    def _inspect_run(self, run_path: str):
        """Inspect a specific run directory."""
        inspector = tk.Toplevel(self.root)
        inspector.title(f"Run Inspector: {os.path.basename(run_path)}")
        inspector.geometry("600x400")
        
        files = os.listdir(run_path) if os.path.exists(run_path) else []
        
        ttk.Label(inspector, text=f"Run: {os.path.basename(run_path)}", font=("TkDefaultFont", 12, "bold")).pack(pady=10)
        
        files_frame = ttk.LabelFrame(inspector, text="Files")
        files_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        listbox = tk.Listbox(files_frame, font=("TkFixedFont", 10))
        listbox.pack(fill="both", expand=True, padx=5, pady=5)
        
        for f in sorted(files):
            listbox.insert("end", f)
        
        def load_stl():
            selection = listbox.curselection()
            if not selection:
                return
            filename = listbox.get(selection[0])
            if filename.endswith(".stl"):
                stl_path = os.path.join(run_path, filename)
                self.stl_viewer.load_stl(stl_path)
                self._append_output(f"Loaded STL from run: {filename}")
        
        ttk.Button(inspector, text="Load STL", command=load_stl).pack(pady=10)
    
    def _show_verification_report(self):
        """P3 #31: Show verification report in GUI."""
        artifacts = self.workflow_manager.get_artifacts()
        verification_report = artifacts.get("verification_report")
        
        if not verification_report:
            messagebox.showinfo("Info", "No verification report available")
            return
        
        report_window = tk.Toplevel(self.root)
        report_window.title("Verification Report")
        report_window.geometry("600x500")
        
        text_widget = scrolledtext.ScrolledText(
            report_window,
            wrap="word",
            font=("TkFixedFont", 10),
        )
        text_widget.pack(fill="both", expand=True, padx=5, pady=5)
        
        report_text = "Verification Report\n" + "=" * 40 + "\n\n"
        
        if isinstance(verification_report, dict):
            report_text += f"Status: {'PASSED' if verification_report.get('passed', False) else 'FAILED'}\n\n"
            
            if "required_files" in verification_report:
                report_text += "Required Files:\n"
                for f, status in verification_report["required_files"].items():
                    report_text += f"  {f}: {'Found' if status else 'MISSING'}\n"
                report_text += "\n"
            
            if "mesh_stats" in verification_report:
                report_text += "Mesh Statistics:\n"
                stats = verification_report["mesh_stats"]
                report_text += f"  Vertices: {stats.get('vertices', 'N/A')}\n"
                report_text += f"  Faces: {stats.get('faces', 'N/A')}\n"
                report_text += f"  Watertight: {stats.get('watertight', 'N/A')}\n"
                report_text += f"  Manifold: {stats.get('manifold', 'N/A')}\n"
                report_text += "\n"
            
            if "bounds" in verification_report:
                bounds = verification_report["bounds"]
                report_text += f"Bounds: {bounds}\n\n"
            
            if "errors" in verification_report:
                report_text += "Errors:\n"
                for error in verification_report["errors"]:
                    report_text += f"  - {error}\n"
        else:
            report_text += str(verification_report)
        
        text_widget.insert("1.0", report_text)
        text_widget.config(state="disabled")
    
    def _show_workspace_update_summary(self, update_data: dict):
        """P3 #25: Show workspace update summary in GUI."""
        summary_text = "Workspace Update Summary\n" + "-" * 30 + "\n"
        
        if "files_updated" in update_data:
            summary_text += "\nFiles Updated:\n"
            for f in update_data["files_updated"]:
                summary_text += f"  - {f}\n"
        
        if "registry_updated" in update_data and update_data["registry_updated"]:
            summary_text += "\nTool Registry: Updated\n"
        
        if "spec_updated" in update_data and update_data["spec_updated"]:
            summary_text += "\nSpec: Updated\n"
        
        self._append_chat("workspace_update", summary_text)
    
    def _show_diff_view(self, old_content: str, new_content: str, filename: str):
        """P3 #27: Show diff view when master changes."""
        import difflib
        
        diff_window = tk.Toplevel(self.root)
        diff_window.title(f"Diff View: {filename}")
        diff_window.geometry("800x600")
        
        text_widget = scrolledtext.ScrolledText(
            diff_window,
            wrap="none",
            font=("TkFixedFont", 10),
        )
        text_widget.pack(fill="both", expand=True, padx=5, pady=5)
        
        text_widget.tag_configure("added", foreground="green")
        text_widget.tag_configure("removed", foreground="red")
        text_widget.tag_configure("header", foreground="blue")
        
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        
        diff = difflib.unified_diff(old_lines, new_lines, fromfile="old", tofile="new")
        
        for line in diff:
            if line.startswith("+++") or line.startswith("---"):
                text_widget.insert("end", line, "header")
            elif line.startswith("+"):
                text_widget.insert("end", line, "added")
            elif line.startswith("-"):
                text_widget.insert("end", line, "removed")
            else:
                text_widget.insert("end", line)
        
        text_widget.config(state="disabled")
    
    def _show_docs(self):
        """Show documentation."""
        import webbrowser
        webbrowser.open("https://github.com/ErickGross-19/Agentic-Organ-Generation")
    
    def _show_about(self):
        """Show about dialog."""
        from gui import __version__
        messagebox.showinfo(
            "About Organ Generator",
            f"Organ Generator v{__version__}\n\n"
            "A GUI for the Agentic Organ Generation system.\n\n"
            "Provides workflow selection, agent configuration,\n"
            "and 3D STL visualization.\n\n"
            "https://github.com/ErickGross-19/Agentic-Organ-Generation"
        )
    
    def _load_window_state(self):
        """Load saved window state."""
        geometry = self.secure_config.get_config("window_geometry")
        if geometry:
            try:
                self.root.geometry(geometry)
            except Exception:
                pass
    
    def _save_window_state(self):
        """Save window state."""
        self.secure_config.store_config("window_geometry", self.root.geometry())
    
    def _on_close(self):
        """Handle window close."""
        if self.workflow_manager.is_running:
            if not messagebox.askyesno(
                "Confirm Exit",
                "A workflow is running. Are you sure you want to exit?"
            ):
                return
            self.workflow_manager.stop_workflow()
        
        self._save_window_state()
        self.root.destroy()
    
    def run(self):
        """Run the main application loop."""
        self.root.mainloop()


def launch_gui():
    """Launch the Organ Generator GUI."""
    app = MainWindow()
    app.run()


if __name__ == "__main__":
    launch_gui()
