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
from .designspec_workflow_manager import DesignSpecWorkflowManager
from .designspec_panels import SpecPanel, PatchPanel, CompilePanel, RunPanel, ArtifactsPanel, LiveSpecViewer
from .configuration_wizard import ConfigurationWizard, WizardConfiguration
from .execution_progress_panel import ExecutionProgressPanel


class WorkflowSelectionDialog(tk.Toplevel):
    """
    Dialog for selecting workflow type and configuration.
    
    Currently only supports DesignSpec workflow.
    
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
        
        self.title("Start DesignSpec Workflow")
        self.geometry("400x200")
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
            text="DesignSpec Workflow",
            font=("TkDefaultFont", 14, "bold"),
        ).pack(pady=(0, 10))
        
        ttk.Label(
            main_frame,
            text="Conversation-driven spec editing with JSON patches.\nPrimary workflow for DesignSpec-first development.",
            foreground="gray",
            justify="center",
        ).pack(pady=(0, 20))
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(20, 0))
        
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
        self.result = WorkflowType.DESIGNSPEC
        self.callback(WorkflowType.DESIGNSPEC)
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
        self._wizard_config: Optional[WizardConfiguration] = None
        
        self._live_spec_viewer: Optional[LiveSpecViewer] = None
        self._execution_progress_panel: Optional[ExecutionProgressPanel] = None
        self._current_layout_mode: str = "conversation"
        
        self._setup_menu()
        self._setup_ui()
        self._setup_bindings()
        
        self._load_window_state()
        
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        self.root.after(100, self._show_configuration_wizard)
    
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
        """Set up three-panel layout with DesignSpec panels in a notebook."""
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
        
        # Create notebook for DesignSpec panels
        self.panels_notebook = ttk.Notebook(left_paned)
        left_paned.add(self.panels_notebook, weight=1)
        
        # Tab 1: Conversation / Log (existing output_text)
        output_frame = ttk.Frame(self.panels_notebook)
        self.panels_notebook.add(output_frame, text="Log")
        
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
        
        # Tab 2: Spec panel
        self.spec_panel = SpecPanel(
            self.panels_notebook,
            on_edit_request=self._on_spec_refresh,
        )
        self.panels_notebook.add(self.spec_panel, text="Spec")
        
        # Tab 3: Patches panel
        self.patch_panel = PatchPanel(
            self.panels_notebook,
            on_approve=self._on_patch_approve,
            on_reject=self._on_patch_reject,
        )
        self.panels_notebook.add(self.patch_panel, text="Patches")
        
        # Tab 4: Run panel with approval buttons
        self.run_panel = RunPanel(
            self.panels_notebook,
            on_run_until=self._on_run_until,
            on_run_full=self._on_run_full,
            on_approve_run=self._on_approve_run,
            on_reject_run=self._on_reject_run,
        )
        self.panels_notebook.add(self.run_panel, text="Run")
        
        # Tab 5: Artifacts panel
        self.artifacts_panel = ArtifactsPanel(
            self.panels_notebook,
            on_load_stl=self._load_stl,
            on_open_folder=self._open_artifact_folder,
        )
        self.panels_notebook.add(self.artifacts_panel, text="Artifacts")
        
        # Tab 6: Compile / Reports panel
        self.compile_panel = CompilePanel(
            self.panels_notebook,
            on_compile=self._on_compile,
        )
        self.panels_notebook.add(self.compile_panel, text="Reports")
        
        # STL Viewer panel
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
    
    def _show_configuration_wizard(self):
        """Show the configuration wizard on startup."""
        def on_complete(config: WizardConfiguration):
            self._wizard_config = config
            self._agent_config = config.agent_config
            self._append_chat(
                "system",
                f"Configuration complete. Provider: {config.agent_config.provider}, "
                f"Project: {config.project_name}, Mode: {config.workflow_mode}"
            )
            if config.workflow_mode == "llm_first":
                use_legacy = False
            else:
                use_legacy = True
            
            if config.import_path:
                self._open_designspec_project(config.import_path, use_legacy_agent=use_legacy)
            else:
                self._init_designspec_workflow(
                    project_root=config.project_location,
                    project_name=config.project_name,
                    use_legacy_agent=use_legacy,
                )
            self._switch_to_conversation_layout()
        
        def on_cancel():
            self._append_chat("system", "Configuration wizard cancelled. Use File > New Workflow to start.")
        
        wizard = ConfigurationWizard(
            self.root,
            self.secure_config,
            on_complete=on_complete,
            on_cancel=on_cancel,
        )
        self.root.wait_window(wizard)
    
    def _switch_to_conversation_layout(self):
        """Switch to 2-panel conversation layout (chat + live spec viewer)."""
        self._current_layout_mode = "conversation"
        
        if self._live_spec_viewer is None:
            self._live_spec_viewer = LiveSpecViewer(self.panels_notebook)
            self.panels_notebook.add(self._live_spec_viewer, text="Live Spec")
        
        self.panels_notebook.select(self._live_spec_viewer)
        
        if hasattr(self, '_designspec_manager') and self._designspec_manager:
            spec = self._designspec_manager.get_spec()
            if spec:
                self._live_spec_viewer.update_spec(spec)
    
    def _switch_to_execution_layout(self):
        """Switch to execution layout with progress panel."""
        self._current_layout_mode = "execution"
        
        if self._execution_progress_panel is None:
            self._execution_progress_panel = ExecutionProgressPanel(self.panels_notebook)
            self.panels_notebook.add(self._execution_progress_panel, text="Progress")
        
        self._execution_progress_panel.reset()
        self.panels_notebook.select(self._execution_progress_panel)
    
    def _switch_to_results_layout(self):
        """Switch to 3-panel results layout after run completes."""
        self._current_layout_mode = "results"
        
        self.panels_notebook.select(self.artifacts_panel)
        
        if hasattr(self, '_designspec_manager') and self._designspec_manager:
            artifacts_dir = self._designspec_manager.get_artifacts_dir()
            if artifacts_dir:
                self.artifacts_panel.update_artifacts(artifacts_dir)
                stl_files = list(Path(artifacts_dir).glob("*.stl"))
                if stl_files:
                    self._load_stl(str(stl_files[0]))
    
    def _on_run_progress(self, event_data: Dict[str, Any]):
        """Handle run progress events from the workflow."""
        if self._execution_progress_panel is None:
            return
        
        stage = event_data.get("stage", "")
        status = event_data.get("status", "")
        message = event_data.get("message", "")
        elapsed = event_data.get("elapsed", 0)
        stage_index = event_data.get("stage_index", 0)
        total_stages = event_data.get("total_stages", 11)
        
        if status == "starting":
            self._execution_progress_panel.update_progress(
                stage=stage,
                stage_index=stage_index,
                total_stages=total_stages,
                status="running",
                message=message,
            )
        elif status == "completed":
            self._execution_progress_panel.set_stage_completed(stage, elapsed)
        elif status == "failed":
            error = event_data.get("error", "Unknown error")
            self._execution_progress_panel.set_stage_failed(stage, error)
        
        if message:
            self._execution_progress_panel.add_log(message)
    
    def _update_live_spec_viewer(self, spec: Dict[str, Any]):
        """Update the live spec viewer with new spec data."""
        if self._live_spec_viewer is not None:
            self._live_spec_viewer.update_spec(spec)
    
    def _show_workflow_selection(self):
        """Show workflow selection dialog."""
        dialog = WorkflowSelectionDialog(self.root, self._on_workflow_selected)
        self.root.wait_window(dialog)
    
    def _on_workflow_selected(self, workflow_type: WorkflowType):
        """Handle workflow selection."""
        self._current_workflow_type = workflow_type
        
        if workflow_type == WorkflowType.DESIGNSPEC:
            self.workflow_label.config(text="DesignSpec Project")
            self._show_designspec_project_dialog()
            return
        
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
        """Start the selected workflow.
        
        For DesignSpec workflow, redirects to the DesignSpec project dialog.
        """
        if self._current_workflow_type is None:
            # Default to DesignSpec workflow
            self._current_workflow_type = WorkflowType.DESIGNSPEC
        
        # DesignSpec is the only supported workflow
        self._show_designspec_project_dialog()
    
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
            
            # Check if DesignSpec workflow is active
            if hasattr(self, '_designspec_manager') and self._current_workflow_type == WorkflowType.DESIGNSPEC:
                if not text:
                    self._append_chat("system", "Please enter some text to send.")
                    self.input_var.set("")
                    self.input_entry.delete(0, "end")
                    return
                
                self._append_chat("user", text)
                
                if text.lower() == "approve" and hasattr(self, '_pending_patch_id'):
                    self._designspec_manager.approve_patch(self._pending_patch_id)
                    delattr(self, '_pending_patch_id')
                elif text.lower() == "reject" and hasattr(self, '_pending_patch_id'):
                    self._designspec_manager.reject_patch(self._pending_patch_id, "User rejected")
                    delattr(self, '_pending_patch_id')
                else:
                    self._designspec_manager.send_message(text)
                
                self.input_var.set("")
                self.input_entry.delete(0, "end")
                return
            
            # Check if workflow is running
            if not self.workflow_manager.is_running:
                self._append_chat("system", "Please start a workflow first (File > New Workflow or Ctrl+N)")
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
        
        DesignSpec workflows emit structured messages with payloads that require
        special handling:
        - spec_update: Update living spec display (if panel exists)
        - plans: Display proposed plans
        - safe_fix: Show safe fix notification
        """
        def handle_message():
            msg_type = message.type
            content = message.content
            data = message.data
            
            if msg_type == "spec_update" and data:
                self._update_spec_display(data)
            elif msg_type == "plans" and data:
                self._display_plans(data.get("plans", []), data.get("recommended_id"))
            elif msg_type == "safe_fix" and data:
                self._show_safe_fix_notification(data)
            else:
                self._append_chat(msg_type, content)
        
        self.root.after(0, handle_message)
    
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
                self.progress.stop()
                
                if self._current_layout_mode == "execution":
                    if self._execution_progress_panel:
                        success = status == WorkflowStatus.COMPLETED
                        error_msg = message if status == WorkflowStatus.FAILED else None
                        self._execution_progress_panel.stop(success=success, error=error_msg)
                    self._switch_to_results_layout()
                    
            elif status == WorkflowStatus.WAITING_INPUT:
                self.input_entry.focus_set()
                if self._current_layout_mode == "execution":
                    if self._execution_progress_panel:
                        self._execution_progress_panel.stop(success=True)
                    self._switch_to_results_layout()
        
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
            spec_file = Path(project_dir) / "spec.json"
            if spec_file.exists():
                self._open_designspec_project(project_dir)
            else:
                stl_files = list(Path(project_dir).glob("**/*.stl"))
                if stl_files:
                    self.stl_viewer.load_stl(str(stl_files[0]))
                    self._append_output(f"Loaded project: {project_dir}")
                    self._append_output(f"Found {len(stl_files)} STL file(s)")
    
    def _show_designspec_project_dialog(self):
        """Show dialog for creating or opening a DesignSpec project."""
        dialog = tk.Toplevel(self.root)
        dialog.title("DesignSpec Project")
        dialog.geometry("450x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding=20)
        main_frame.pack(fill="both", expand=True)
        
        ttk.Label(
            main_frame,
            text="DesignSpec Project",
            font=("TkDefaultFont", 14, "bold"),
        ).pack(pady=(0, 10))
        
        ttk.Label(
            main_frame,
            text="Create a new project or open an existing one:",
        ).pack(pady=(0, 10))
        
        agent_frame = ttk.LabelFrame(main_frame, text="Agent Mode", padding=10)
        agent_frame.pack(fill="x", pady=(0, 10))
        
        use_llm_var = tk.BooleanVar(value=True)
        
        ttk.Radiobutton(
            agent_frame,
            text="LLM-First Agent (Recommended)",
            variable=use_llm_var,
            value=True,
        ).pack(anchor="w")
        ttk.Label(
            agent_frame,
            text="Uses LLM as primary interpreter for natural language understanding",
            foreground="gray",
            font=("TkDefaultFont", 9),
        ).pack(anchor="w", padx=20)
        
        ttk.Radiobutton(
            agent_frame,
            text="Legacy Rule-Based Agent",
            variable=use_llm_var,
            value=False,
        ).pack(anchor="w")
        ttk.Label(
            agent_frame,
            text="Uses regex/heuristic parsing (fallback mode)",
            foreground="gray",
            font=("TkDefaultFont", 9),
        ).pack(anchor="w", padx=20)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=10)
        
        def on_new():
            use_legacy = not use_llm_var.get()
            dialog.destroy()
            self._create_designspec_project_with_mode(use_legacy)
        
        def on_open():
            use_legacy = not use_llm_var.get()
            dialog.destroy()
            project_dir = filedialog.askdirectory(
                title="Select DesignSpec Project Directory",
            )
            if project_dir:
                self._open_designspec_project(project_dir, use_legacy_agent=use_legacy)
        
        ttk.Button(
            button_frame,
            text="New Project",
            command=on_new,
        ).pack(side="left", padx=5, expand=True)
        
        ttk.Button(
            button_frame,
            text="Open Project",
            command=on_open,
        ).pack(side="left", padx=5, expand=True)
        
        ttk.Button(
            button_frame,
            text="Cancel",
            command=dialog.destroy,
        ).pack(side="left", padx=5, expand=True)
        
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
    
    def _create_designspec_project(self):
        """Create a new DesignSpec project."""
        dialog = tk.Toplevel(self.root)
        dialog.title("New DesignSpec Project")
        dialog.geometry("450x280")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding=20)
        main_frame.pack(fill="both", expand=True)
        
        ttk.Label(main_frame, text="Project Name:").pack(anchor="w")
        name_var = tk.StringVar(value="my_organ_project")
        name_entry = ttk.Entry(main_frame, textvariable=name_var, width=40)
        name_entry.pack(fill="x", pady=(0, 10))
        
        ttk.Label(main_frame, text="Location:").pack(anchor="w")
        location_frame = ttk.Frame(main_frame)
        location_frame.pack(fill="x", pady=(0, 10))
        
        location_var = tk.StringVar(value=os.path.expanduser("~/projects"))
        location_entry = ttk.Entry(location_frame, textvariable=location_var)
        location_entry.pack(side="left", fill="x", expand=True)
        
        def browse():
            path = filedialog.askdirectory(title="Select Location")
            if path:
                location_var.set(path)
        
        ttk.Button(location_frame, text="Browse", command=browse).pack(side="left", padx=5)
        
        agent_frame = ttk.LabelFrame(main_frame, text="Agent Mode", padding=10)
        agent_frame.pack(fill="x", pady=(5, 10))
        
        use_llm_var = tk.BooleanVar(value=True)
        
        ttk.Radiobutton(
            agent_frame,
            text="LLM-First Agent (Recommended)",
            variable=use_llm_var,
            value=True,
        ).pack(anchor="w")
        ttk.Label(
            agent_frame,
            text="Uses LLM as primary interpreter for natural language understanding",
            foreground="gray",
            font=("TkDefaultFont", 9),
        ).pack(anchor="w", padx=20)
        
        ttk.Radiobutton(
            agent_frame,
            text="Legacy Rule-Based Agent",
            variable=use_llm_var,
            value=False,
        ).pack(anchor="w")
        ttk.Label(
            agent_frame,
            text="Uses regex/heuristic parsing (fallback mode)",
            foreground="gray",
            font=("TkDefaultFont", 9),
        ).pack(anchor="w", padx=20)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=10)
        
        def on_create():
            name = name_var.get().strip()
            location = location_var.get().strip()
            if not name:
                messagebox.showwarning("Warning", "Please enter a project name")
                return
            if not location:
                messagebox.showwarning("Warning", "Please select a location")
                return
            
            use_legacy = not use_llm_var.get()
            dialog.destroy()
            self._init_designspec_workflow(location, name, use_legacy_agent=use_legacy)
        
        ttk.Button(button_frame, text="Create", command=on_create).pack(side="right", padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side="right", padx=5)
        
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
    
    def _create_designspec_project_with_mode(self, use_legacy_agent: bool = False):
        """
        Create a new DesignSpec project with specified agent mode.
        
        This is called from the project dialog when agent mode is already selected.
        
        Parameters
        ----------
        use_legacy_agent : bool
            If True, use the legacy rule-based agent instead of the LLM-first agent.
        """
        dialog = tk.Toplevel(self.root)
        dialog.title("New DesignSpec Project")
        dialog.geometry("400x180")
        dialog.transient(self.root)
        dialog.grab_set()
        
        main_frame = ttk.Frame(dialog, padding=20)
        main_frame.pack(fill="both", expand=True)
        
        ttk.Label(main_frame, text="Project Name:").pack(anchor="w")
        name_var = tk.StringVar(value="my_organ_project")
        name_entry = ttk.Entry(main_frame, textvariable=name_var, width=40)
        name_entry.pack(fill="x", pady=(0, 10))
        
        ttk.Label(main_frame, text="Location:").pack(anchor="w")
        location_frame = ttk.Frame(main_frame)
        location_frame.pack(fill="x", pady=(0, 10))
        
        location_var = tk.StringVar(value=os.path.expanduser("~/projects"))
        location_entry = ttk.Entry(location_frame, textvariable=location_var)
        location_entry.pack(side="left", fill="x", expand=True)
        
        def browse():
            path = filedialog.askdirectory(title="Select Location")
            if path:
                location_var.set(path)
        
        ttk.Button(location_frame, text="Browse", command=browse).pack(side="left", padx=5)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=10)
        
        def on_create():
            name = name_var.get().strip()
            location = location_var.get().strip()
            if not name:
                messagebox.showwarning("Warning", "Please enter a project name")
                return
            if not location:
                messagebox.showwarning("Warning", "Please select a location")
                return
            
            dialog.destroy()
            self._init_designspec_workflow(location, name, use_legacy_agent=use_legacy_agent)
        
        ttk.Button(button_frame, text="Create", command=on_create).pack(side="right", padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side="right", padx=5)
        
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
    
    def _open_designspec_project(self, project_dir: str, use_legacy_agent: bool = False):
        """
        Open an existing DesignSpec project.
        
        Parameters
        ----------
        project_dir : str
            Path to the project directory
        use_legacy_agent : bool
            If True, use the legacy rule-based agent instead of the LLM-first agent.
        """
        self._init_designspec_workflow(project_dir=project_dir, use_legacy_agent=use_legacy_agent)
    
    def _init_designspec_workflow(
        self,
        project_root: str = None,
        project_name: str = None,
        project_dir: str = None,
        use_legacy_agent: bool = False,
    ):
        """
        Initialize the DesignSpec workflow.
        
        Parameters
        ----------
        project_root : str, optional
            Parent directory for new project
        project_name : str, optional
            Name for new project
        project_dir : str, optional
            Path to existing project directory
        use_legacy_agent : bool
            If True, use the legacy rule-based agent instead of the LLM-first agent.
            Default is False (use LLM-first agent, recommended).
        """
        from tkinter import messagebox
        # Note: AgentConfiguration is already imported at module level from .agent_config
        
        if not hasattr(self, '_designspec_manager') or self._designspec_manager is None:
            self._designspec_manager = DesignSpecWorkflowManager(
                message_callback=self._on_workflow_message,
                status_callback=self._on_status_change,
                output_callback=self._on_output,
                spec_callback=self._on_spec_update,
                patch_callback=self._on_patch_proposal,
                compile_callback=self._on_compile_status,
                run_progress_callback=self._on_run_progress,
                use_legacy_agent=use_legacy_agent,
            )
        else:
            self._designspec_manager._use_legacy_agent = use_legacy_agent
        
        # For LLM-first mode, require successful LLM initialization
        if not use_legacy_agent:
            # Determine config: use _agent_config if exists, otherwise use default
            if hasattr(self, '_agent_config') and self._agent_config:
                config = self._agent_config
            else:
                # Use default config (allows env-var-based keys to work)
                config = AgentConfiguration()
            
            llm_init_success = self._designspec_manager.initialize_llm(
                provider=config.provider,
                api_key=config.api_key,
                model=config.model,
                api_base=config.api_base,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            
            if not llm_init_success:
                # Show error dialog and abort
                error_msg = self._designspec_manager.last_llm_init_error or "Unknown error"
                messagebox.showerror(
                    "LLM Initialization Failed",
                    f"Failed to initialize LLM for LLM-first mode.\n\n"
                    f"Provider: {config.provider}\n"
                    f"Model: {config.model or 'default'}\n\n"
                    f"Error: {error_msg}\n\n"
                    f"Please open Agent Config and set a valid API key, "
                    f"or set the provider environment variable (e.g., OPENAI_API_KEY)."
                )
                # Also append error to chat for visibility
                self._append_chat(
                    "error",
                    f"LLM initialization failed: {error_msg}. "
                    f"Please configure a valid API key in Agent Config."
                )
                return  # Do NOT proceed to create/load project
        elif hasattr(self, '_agent_config') and self._agent_config:
            # Legacy mode with config available - still try to init LLM (optional)
            config = self._agent_config
            self._designspec_manager.initialize_llm(
                provider=config.provider,
                api_key=config.api_key,
                model=config.model,
                api_base=config.api_base,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        
        if project_dir:
            success = self._designspec_manager.load_project(project_dir)
        else:
            success = self._designspec_manager.create_project(project_root, project_name)
        
        if success:
            self._current_workflow_type = WorkflowType.DESIGNSPEC
            self.workflow_label.config(text="DesignSpec Project")
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self._append_chat("system", "DesignSpec project ready. Describe what you want to create!")
    
    def _on_spec_update(self, spec: Dict[str, Any]):
        """Handle spec update from DesignSpec workflow."""
        self._update_live_spec_viewer(spec)
        
        domains = spec.get("domains", {})
        components = spec.get("components", [])
        features = spec.get("features", {})
        
        summary_parts = []
        
        if domains:
            domain_names = list(domains.keys())
            for name in domain_names[:2]:
                domain = domains[name]
                dtype = domain.get("type", "unknown")
                if dtype == "box":
                    if "x_min" in domain and "x_max" in domain:
                        width = domain.get("x_max", 0) - domain.get("x_min", 0)
                        height = domain.get("y_max", 0) - domain.get("y_min", 0)
                        depth = domain.get("z_max", 0) - domain.get("z_min", 0)
                        summary_parts.append(f"Domain '{name}': box {width}x{height}x{depth}")
                    elif "size" in domain:
                        size = domain.get("size", [0, 0, 0])
                        summary_parts.append(f"Domain '{name}': box {size[0]}x{size[1]}x{size[2]}")
                    else:
                        summary_parts.append(f"Domain '{name}': box")
                elif dtype == "cylinder":
                    r = domain.get("radius", 0)
                    h = domain.get("height", 0)
                    summary_parts.append(f"Domain '{name}': cylinder r={r} h={h}")
                else:
                    summary_parts.append(f"Domain '{name}': {dtype}")
        
        if components:
            for comp in components[:3]:
                comp_id = comp.get("id", "unnamed")
                build_type = comp.get("build", {}).get("type", comp.get("build", {}).get("backend", "unknown"))
                summary_parts.append(f"Component '{comp_id}': {build_type}")
        
        if features:
            ridges = features.get("ridges", [])
            if ridges:
                if isinstance(ridges, dict):
                    faces = ridges.get("faces", [])
                else:
                    faces = [r.get("face", "?") for r in ridges]
                summary_parts.append(f"Ridges on faces: {', '.join(faces)}")
        
        if summary_parts:
            summary = "Spec updated:\n" + "\n".join(f"  - {p}" for p in summary_parts)
            self._append_chat("system", summary)
    
    def _on_patch_proposal(self, patch_data: Dict[str, Any]):
        """Handle patch proposal from DesignSpec workflow."""
        patch_id = patch_data.get("patch_id", "")
        explanation = patch_data.get("explanation", "")
        patches = patch_data.get("patches", [])
        
        self._pending_patch_id = patch_id
        
        msg = f"Proposed patch ({patch_id}):\n{explanation}\n"
        for i, patch in enumerate(patches[:3]):
            op = patch.get("op", "")
            path = patch.get("path", "")
            msg += f"  {i+1}. {op} {path}\n"
        if len(patches) > 3:
            msg += f"  ... and {len(patches) - 3} more\n"
        msg += "\nType 'approve' to apply or 'reject' to discard."
        
        self._append_chat("assistant", msg)
    
    def _on_compile_status(self, compile_data: Dict[str, Any]):
        """Handle compile status from DesignSpec workflow."""
        status = compile_data.get("status", "")
        message = compile_data.get("message", "")
        report = compile_data.get("report", {})
        
        if status == "running":
            self._append_chat("system", "Compiling...")
        elif status == "success":
            warnings = report.get("warnings", [])
            if warnings:
                warning_text = "\n".join(f"  - {w}" for w in warnings[:5])
                self._append_chat("success", f"Compile successful with warnings:\n{warning_text}")
            else:
                self._append_chat("success", "Compile successful")
        elif status == "failed":
            errors = report.get("errors", [])
            warnings = report.get("warnings", [])
            error_msg = f"Compile failed: {message}"
            if errors:
                error_text = "\n".join(f"  - {e}" for e in errors[:5])
                error_msg += f"\nErrors:\n{error_text}"
            if warnings:
                warning_text = "\n".join(f"  - {w}" for w in warnings[:3])
                error_msg += f"\nWarnings:\n{warning_text}"
            self._append_chat("error", error_msg)
    
    def _on_spec_refresh(self, action: str = None):
        """Handle spec refresh request from SpecPanel.
        
        Parameters
        ----------
        action : str, optional
            The action requested (e.g., "refresh"). Currently unused but
            required to match the callback signature from SpecPanel.
        """
        if hasattr(self, '_designspec_manager') and self._designspec_manager:
            spec = self._designspec_manager.get_spec()
            if spec and hasattr(self, 'spec_panel'):
                self.spec_panel.update_spec(spec)
    
    def _on_patch_approve(self, patch_id: str):
        """Handle patch approval from PatchPanel."""
        if hasattr(self, '_designspec_manager') and self._designspec_manager:
            self._designspec_manager.approve_patch(patch_id)
            self._append_chat("system", f"Patch {patch_id} approved and applied")
            if hasattr(self, 'patch_panel'):
                self.patch_panel.clear_patches()
    
    def _on_patch_reject(self, patch_id: str, reason: str = ""):
        """Handle patch rejection from PatchPanel."""
        if hasattr(self, '_designspec_manager') and self._designspec_manager:
            self._designspec_manager.reject_patch(patch_id, reason)
            self._append_chat("system", f"Patch {patch_id} rejected")
            if hasattr(self, 'patch_panel'):
                self.patch_panel.clear_patches()
    
    def _on_run_until(self, stage: str):
        """Handle run until request from RunPanel."""
        if hasattr(self, '_designspec_manager') and self._designspec_manager:
            self._append_chat("system", f"Running until stage: {stage}")
            self._switch_to_execution_layout()
            if self._execution_progress_panel:
                self._execution_progress_panel.start()
            if hasattr(self, 'run_panel'):
                self.run_panel.set_running(True, f"Running until {stage}...")
            self._designspec_manager.run_until(stage)
    
    def _on_run_full(self):
        """Handle full run request from RunPanel."""
        if hasattr(self, '_designspec_manager') and self._designspec_manager:
            self._append_chat("system", "Starting full run...")
            self._switch_to_execution_layout()
            if self._execution_progress_panel:
                self._execution_progress_panel.start()
            if hasattr(self, 'run_panel'):
                self.run_panel.set_running(True, "Running full pipeline...")
            self._designspec_manager.run_full()
    
    def _on_approve_run(self):
        """Handle run approval from RunPanel."""
        if hasattr(self, '_designspec_manager') and self._designspec_manager:
            self._append_chat("system", "Run approved, executing...")
            self._switch_to_execution_layout()
            if self._execution_progress_panel:
                self._execution_progress_panel.start()
            if hasattr(self, 'run_panel'):
                self.run_panel.set_waiting_approval(False)
                self.run_panel.set_running(True, "Executing approved run...")
            self._designspec_manager.approve_run()
    
    def _on_reject_run(self, reason: str = ""):
        """Handle run rejection from RunPanel."""
        if hasattr(self, '_designspec_manager') and self._designspec_manager:
            self._append_chat("system", "Run rejected")
            if hasattr(self, 'run_panel'):
                self.run_panel.set_waiting_approval(False)
            self._designspec_manager.reject_run(reason)
    
    def _open_artifact_folder(self, artifact_path: str):
        """Open folder containing an artifact."""
        if artifact_path and os.path.exists(artifact_path):
            folder = os.path.dirname(artifact_path)
            if sys.platform == "darwin":
                os.system(f'open "{folder}"')
            elif sys.platform == "win32":
                os.startfile(folder)
            else:
                os.system(f'xdg-open "{folder}"')
        else:
            messagebox.showinfo("Info", "Artifact folder not available")
    
    def _on_compile(self):
        """Handle compile request from CompilePanel."""
        if hasattr(self, '_designspec_manager') and self._designspec_manager:
            self._append_chat("system", "Compiling spec...")
            if hasattr(self, 'compile_panel'):
                self.compile_panel.update_status("running", "Compiling...")
            self._designspec_manager.compile_spec()
    
    def _load_stl(self, file_path: str = None):
        """Load STL file directly or via file dialog."""
        if not file_path:
            file_path = filedialog.askopenfilename(
                title="Select STL File",
                filetypes=[("STL files", "*.stl"), ("All files", "*.*")],
            )
        
        if file_path and os.path.exists(file_path):
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
