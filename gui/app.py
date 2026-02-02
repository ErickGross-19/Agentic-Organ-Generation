"""
DesignSpec GUI Application Entry Point

This module provides the main entry point for the DesignSpec GUI application.
It launches the configuration wizard and then starts the conversation-based workflow.

Usage:
    >>> from gui import launch_gui
    >>> launch_gui()

    Or run as module:
    $ python -m gui
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from .configuration_wizard import ConfigurationWizard, WizardConfiguration
from .designspec_workflow_manager import DesignSpecWorkflowManager
from .workflow_manager import WorkflowMessage, WorkflowStatus
from .designspec_panels import LiveSpecViewer
from .security import SecureConfig


class DesignSpecApp:
    """
    Simplified DesignSpec GUI Application.

    Shows only the configuration wizard and conversation layout.
    Avoids the deprecated tabbed multi-panel layout.
    """

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Organ Generator - DesignSpec Workflow")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)

        self.secure_config = SecureConfig()
        self._designspec_manager: Optional[DesignSpecWorkflowManager] = None
        self._wizard_config: Optional[WizardConfiguration] = None

        self._setup_ui()

        # Show configuration wizard on startup
        self.root.after(100, self._show_configuration_wizard)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_ui(self):
        """Set up the main UI."""
        # Main container
        self.main_frame = ttk.Frame(self.root, padding=5)
        self.main_frame.pack(fill="both", expand=True)

        # Paned window for chat and spec viewer
        self.paned_window = ttk.PanedWindow(self.main_frame, orient="horizontal")
        self.paned_window.pack(fill="both", expand=True)

        # Left: Chat panel
        self._setup_chat_panel()

        # Right: Live spec viewer
        self._setup_spec_panel()

        # Status bar
        self._setup_status_bar()

    def _setup_chat_panel(self):
        """Set up the chat panel."""
        chat_container = ttk.LabelFrame(self.paned_window, text="Chat")
        self.paned_window.add(chat_container, weight=1)

        chat_container.columnconfigure(0, weight=1)
        chat_container.rowconfigure(0, weight=1)

        # Chat text area
        self.chat_text = scrolledtext.ScrolledText(
            chat_container,
            wrap="word",
            state="disabled",
            font=("TkFixedFont", 10),
        )
        self.chat_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Configure tags for different message types
        self.chat_text.tag_configure("system", foreground="blue")
        self.chat_text.tag_configure("user", foreground="green")
        self.chat_text.tag_configure("assistant", foreground="purple")
        self.chat_text.tag_configure("error", foreground="red")
        self.chat_text.tag_configure("success", foreground="darkgreen")
        self.chat_text.tag_configure("prompt", foreground="orange")

        # Input frame
        input_frame = ttk.Frame(chat_container)
        input_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        input_frame.columnconfigure(0, weight=1)

        self.input_var = tk.StringVar()
        self.input_entry = ttk.Entry(
            input_frame,
            textvariable=self.input_var,
            font=("TkFixedFont", 10),
        )
        self.input_entry.grid(row=0, column=0, sticky="ew")
        self.input_entry.bind("<Return>", lambda e: self._send_message())

        self.send_btn = ttk.Button(
            input_frame,
            text="Send",
            command=self._send_message,
        )
        self.send_btn.grid(row=0, column=1, padx=(5, 0))

        self.run_btn = ttk.Button(
            input_frame,
            text="Run",
            command=self._run_pipeline,
        )
        self.run_btn.grid(row=0, column=2, padx=(5, 0))

    def _setup_spec_panel(self):
        """Set up the live spec viewer panel."""
        spec_container = ttk.LabelFrame(self.paned_window, text="Live Spec")
        self.paned_window.add(spec_container, weight=1)

        self.live_spec_viewer = LiveSpecViewer(spec_container)
        self.live_spec_viewer.pack(fill="both", expand=True, padx=5, pady=5)

    def _setup_status_bar(self):
        """Set up the status bar."""
        status_frame = ttk.Frame(self.main_frame)
        status_frame.pack(fill="x", pady=(5, 0))

        self.status_var = tk.StringVar(value="Ready - Please configure your project")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.status_label.pack(side="left")

        self.progress = ttk.Progressbar(
            status_frame,
            mode="indeterminate",
            length=100,
        )
        self.progress.pack(side="right", padx=5)

    def _show_configuration_wizard(self):
        """Show the configuration wizard."""
        def on_complete(config: WizardConfiguration):
            self._wizard_config = config
            self._initialize_workflow(config)

        def on_cancel():
            self._append_chat("system", "Configuration cancelled. Use File > New Workflow to start.")
            self.status_var.set("No project configured")

        wizard = ConfigurationWizard(
            self.root,
            self.secure_config,
            on_complete=on_complete,
            on_cancel=on_cancel,
        )
        self.root.wait_window(wizard)

    def _initialize_workflow(self, config: WizardConfiguration):
        """Initialize the DesignSpec workflow with the given configuration."""
        # Determine agent mode
        use_legacy = (config.workflow_mode == "legacy")

        # Create workflow manager
        self._designspec_manager = DesignSpecWorkflowManager(
            message_callback=self._on_workflow_message,
            status_callback=self._on_status_change,
            output_callback=self._on_output,
            spec_callback=self._on_spec_update,
            patch_callback=self._on_patch_proposal,
            compile_callback=self._on_compile_status,
            run_progress_callback=self._on_run_progress,
            use_legacy_agent=use_legacy,
        )

        # Initialize LLM
        if not use_legacy:
            llm_success = self._designspec_manager.initialize_llm(
                provider=config.agent_config.provider,
                api_key=config.agent_config.api_key,
                model=config.agent_config.model,
                api_base=config.agent_config.api_base,
                temperature=config.agent_config.temperature,
                max_tokens=config.agent_config.max_tokens,
            )

            if not llm_success:
                error_msg = self._designspec_manager.last_llm_init_error or "Unknown error"
                messagebox.showerror(
                    "LLM Initialization Failed",
                    f"Failed to initialize LLM.\n\nError: {error_msg}\n\n"
                    "Please check your API key and try again."
                )
                return

        # Load or create project
        if config.template == "open_project" and config.open_project_path:
            self._append_chat("system", f"Opening project: {config.open_project_path}")
            success = self._designspec_manager.load_project(config.open_project_path)
        elif config.import_path:
            self._append_chat("system", f"Importing spec from: {config.import_path}")
            success = self._designspec_manager.load_project(config.import_path)
        else:
            self._append_chat("system", f"Creating new project: {config.project_name}")
            success = self._designspec_manager.create_project(
                config.project_location,
                config.project_name
            )

        if success:
            mode_str = "LLM-First" if not use_legacy else "Legacy"
            self._append_chat("success", f"Project ready! Mode: {mode_str}")
            self._append_chat("system", "Describe what you want to create, or type 'run' to execute the pipeline.")
            self.status_var.set("Project loaded - Ready")
        else:
            self._append_chat("error", "Failed to initialize project")
            self.status_var.set("Project initialization failed")

    def _send_message(self):
        """Send user message to the workflow."""
        text = self.input_entry.get().strip()
        if not text:
            return

        if not self._designspec_manager:
            self._append_chat("error", "No project loaded. Please restart and configure a project.")
            return

        self._append_chat("user", text)
        self.input_var.set("")
        self.input_entry.delete(0, "end")

        # Send to workflow manager
        self._designspec_manager.send_message(text)

    def _run_pipeline(self):
        """Run the full pipeline."""
        if not self._designspec_manager:
            self._append_chat("error", "No project loaded. Please restart and configure a project.")
            return

        self._append_chat("user", "run")
        self._append_chat("system", "Starting full pipeline run...")
        self._designspec_manager.run_full()

    def _on_workflow_message(self, message: WorkflowMessage):
        """Handle message from workflow."""
        self.root.after(0, lambda: self._append_chat(message.type, message.content))

    def _on_status_change(self, status: WorkflowStatus, message: str):
        """Handle workflow status change."""
        def update():
            status_text = f"{status.value}"
            if message:
                status_text += f": {message}"
            self.status_var.set(status_text)

            if status == WorkflowStatus.RUNNING:
                self.progress.start()
            else:
                self.progress.stop()

        self.root.after(0, update)

    def _on_output(self, output_type: str, data: Any):
        """Handle output from workflow."""
        self.root.after(0, lambda: self._append_chat("system", f"{output_type}: {data}"))

    def _on_spec_update(self, spec: Dict[str, Any]):
        """Handle spec update from workflow."""
        self.root.after(0, lambda: self.live_spec_viewer.update_spec(spec))

    def _on_patch_proposal(self, patch_data: Dict[str, Any]):
        """Handle patch proposal from workflow."""
        def show_patch():
            patch_id = patch_data.get("patch_id", "")
            explanation = patch_data.get("explanation", "")
            patches = patch_data.get("patches", [])

            msg = f"Proposed patch ({patch_id}):\n{explanation}\n"
            for i, patch in enumerate(patches[:3]):
                op = patch.get("op", "")
                path = patch.get("path", "")
                msg += f"  {i+1}. {op} {path}\n"
            if len(patches) > 3:
                msg += f"  ... and {len(patches) - 3} more\n"
            msg += "\nType 'approve' to apply or 'reject' to discard."

            self._append_chat("assistant", msg)

        self.root.after(0, show_patch)

    def _on_compile_status(self, compile_data: Dict[str, Any]):
        """Handle compile status from workflow."""
        status = compile_data.get("status", "")
        message = compile_data.get("message", "")

        if status == "success":
            self.root.after(0, lambda: self._append_chat("success", f"Compile: {message}"))
        elif status == "failed":
            self.root.after(0, lambda: self._append_chat("error", f"Compile failed: {message}"))

    def _on_run_progress(self, progress_data: Dict[str, Any]):
        """Handle run progress updates."""
        stage = progress_data.get("stage", "")
        status = progress_data.get("status", "")

        if status == "completed":
            self.root.after(0, lambda: self._append_chat("success", f"Stage completed: {stage}"))
        elif status == "failed":
            self.root.after(0, lambda: self._append_chat("error", f"Stage failed: {stage}"))

    def _append_chat(self, msg_type: str, content: str):
        """Append message to chat."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = f"[{timestamp}] "

        if msg_type == "user":
            prefix += "You: "
        elif msg_type == "system":
            prefix += "System: "
        elif msg_type == "assistant":
            prefix += "Agent: "
        elif msg_type == "error":
            prefix += "Error: "
        elif msg_type == "success":
            prefix += "Success: "
        elif msg_type == "prompt":
            prefix += ">>> "

        self.chat_text.config(state="normal")
        self.chat_text.insert("end", prefix, msg_type)
        self.chat_text.insert("end", content + "\n", msg_type)
        self.chat_text.see("end")
        self.chat_text.config(state="disabled")

    def _on_close(self):
        """Handle window close."""
        if self._designspec_manager and self._designspec_manager.is_running:
            if not messagebox.askyesno(
                "Confirm Exit",
                "A workflow is running. Are you sure you want to exit?"
            ):
                return
            self._designspec_manager.stop()

        self.root.destroy()

    def run(self):
        """Run the application."""
        self.root.mainloop()


def launch_gui():
    """
    Launch the DesignSpec GUI application.

    This launches a simplified configuration-driven GUI that shows:
    1. Configuration wizard on startup
    2. Chat interface for natural language interaction
    3. Live spec viewer showing the current DesignSpec
    """
    app = DesignSpecApp()
    app.run()


if __name__ == "__main__":
    launch_gui()
