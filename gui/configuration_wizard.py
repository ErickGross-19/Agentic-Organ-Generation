"""
Configuration Wizard

Initial configuration screen that appears on startup before the workflow starts.
Collects agent configuration, project setup, and workflow mode selection.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import os

from .security import SecureConfig
from .agent_config import (
    SUPPORTED_PROVIDERS,
    PROVIDER_ENV_VARS,
    PROVIDER_MODELS,
    AgentConfiguration,
)


@dataclass
class WizardConfiguration:
    """Complete configuration from the wizard."""
    agent_config: AgentConfiguration = field(default_factory=AgentConfiguration)
    project_name: str = ""
    project_location: str = ""
    workflow_mode: str = "llm_first"
    template: str = "empty"
    import_path: Optional[str] = None


class ConfigurationWizard(tk.Toplevel):
    """
    Initial configuration screen before workflow starts.
    
    Collects:
    1. Agent Configuration (LLM provider, model, API key)
    2. Project Setup (name, location, template selection)
    3. DesignSpec Mode (LLM-first vs legacy)
    
    Parameters
    ----------
    parent : tk.Widget
        Parent widget
    secure_config : SecureConfig
        Secure configuration storage
    on_complete : Callable
        Callback with WizardConfiguration when wizard completes
    on_cancel : Callable, optional
        Callback when wizard is cancelled
    """
    
    def __init__(
        self,
        parent: tk.Widget,
        secure_config: SecureConfig,
        on_complete: Callable[[WizardConfiguration], None],
        on_cancel: Optional[Callable[[], None]] = None,
    ):
        super().__init__(parent)
        
        self.secure_config = secure_config
        self.on_complete = on_complete
        self.on_cancel_callback = on_cancel
        
        self._config = WizardConfiguration()
        self._current_step = 0
        self._steps = ["Agent Configuration", "Project Setup", "Workflow Mode"]
        
        self.title("DesignSpec Workflow Setup")
        self.geometry("600x500")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        
        self._setup_ui()
        self._load_saved_config()
        self._show_step(0)
        
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")
    
    def _setup_ui(self):
        """Set up the wizard UI."""
        main_frame = ttk.Frame(self, padding=20)
        main_frame.pack(fill="both", expand=True)
        
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill="x", pady=(0, 20))
        
        ttk.Label(
            header_frame,
            text="DesignSpec Workflow Setup",
            font=("TkDefaultFont", 16, "bold"),
        ).pack(side="left")
        
        self.step_label = ttk.Label(
            header_frame,
            text="Step 1 of 3",
            foreground="gray",
        )
        self.step_label.pack(side="right")
        
        self.progress = ttk.Progressbar(
            main_frame,
            mode="determinate",
            maximum=len(self._steps),
        )
        self.progress.pack(fill="x", pady=(0, 20))
        
        self.content_frame = ttk.Frame(main_frame)
        self.content_frame.pack(fill="both", expand=True)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(20, 0))
        
        self.cancel_btn = ttk.Button(
            button_frame,
            text="Cancel",
            command=self._on_cancel,
        )
        self.cancel_btn.pack(side="left")
        
        self.next_btn = ttk.Button(
            button_frame,
            text="Next",
            command=self._on_next,
        )
        self.next_btn.pack(side="right", padx=(5, 0))
        
        self.back_btn = ttk.Button(
            button_frame,
            text="Back",
            command=self._on_back,
            state="disabled",
        )
        self.back_btn.pack(side="right")
    
    def _show_step(self, step: int):
        """Show the specified step."""
        self._current_step = step
        
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        
        self.step_label.config(text=f"Step {step + 1} of {len(self._steps)}: {self._steps[step]}")
        self.progress["value"] = step + 1
        
        self.back_btn.config(state="normal" if step > 0 else "disabled")
        self.next_btn.config(text="Finish" if step == len(self._steps) - 1 else "Next")
        
        if step == 0:
            self._setup_agent_step()
        elif step == 1:
            self._setup_project_step()
        elif step == 2:
            self._setup_mode_step()
    
    def _setup_agent_step(self):
        """Set up the agent configuration step."""
        frame = ttk.LabelFrame(self.content_frame, text="LLM Agent Configuration", padding=15)
        frame.pack(fill="both", expand=True)
        
        frame.columnconfigure(1, weight=1)
        
        row = 0
        ttk.Label(frame, text="Provider:").grid(row=row, column=0, sticky="w", pady=5)
        self.provider_var = tk.StringVar(value=self._config.agent_config.provider)
        provider_combo = ttk.Combobox(
            frame,
            textvariable=self.provider_var,
            values=[p[0] for p in SUPPORTED_PROVIDERS],
            state="readonly",
            width=30,
        )
        provider_combo.grid(row=row, column=1, sticky="ew", pady=5, padx=(10, 0))
        provider_combo.bind("<<ComboboxSelected>>", self._on_provider_change)
        
        for i, (name, key, _) in enumerate(SUPPORTED_PROVIDERS):
            if key == self._config.agent_config.provider:
                provider_combo.current(i)
                break
        
        row += 1
        ttk.Label(frame, text="Model:").grid(row=row, column=0, sticky="w", pady=5)
        self.model_var = tk.StringVar(value=self._config.agent_config.model)
        self.model_combo = ttk.Combobox(
            frame,
            textvariable=self.model_var,
            values=PROVIDER_MODELS.get(self._config.agent_config.provider, []),
            width=30,
        )
        self.model_combo.grid(row=row, column=1, sticky="ew", pady=5, padx=(10, 0))
        
        row += 1
        ttk.Label(frame, text="API Key:").grid(row=row, column=0, sticky="w", pady=5)
        key_frame = ttk.Frame(frame)
        key_frame.grid(row=row, column=1, sticky="ew", pady=5, padx=(10, 0))
        key_frame.columnconfigure(0, weight=1)
        
        self.api_key_var = tk.StringVar(value=self._config.agent_config.api_key or "")
        self.api_key_entry = ttk.Entry(
            key_frame,
            textvariable=self.api_key_var,
            show="*",
            width=25,
        )
        self.api_key_entry.grid(row=0, column=0, sticky="ew")
        
        self.show_key_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            key_frame,
            text="Show",
            variable=self.show_key_var,
            command=self._toggle_key_visibility,
        ).grid(row=0, column=1, padx=(5, 0))
        
        row += 1
        env_var = PROVIDER_ENV_VARS.get(self._config.agent_config.provider)
        if env_var:
            env_value = os.environ.get(env_var)
            if env_value:
                ttk.Label(
                    frame,
                    text=f"(Using {env_var} from environment)",
                    foreground="green",
                    font=("TkDefaultFont", 9),
                ).grid(row=row, column=1, sticky="w", padx=(10, 0))
        
        row += 1
        ttk.Separator(frame, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=15
        )
        
        row += 1
        ttk.Label(frame, text="Temperature:").grid(row=row, column=0, sticky="w", pady=5)
        temp_frame = ttk.Frame(frame)
        temp_frame.grid(row=row, column=1, sticky="ew", pady=5, padx=(10, 0))
        
        self.temp_var = tk.DoubleVar(value=self._config.agent_config.temperature)
        temp_scale = ttk.Scale(
            temp_frame,
            from_=0.0,
            to=1.0,
            variable=self.temp_var,
            orient="horizontal",
        )
        temp_scale.pack(side="left", fill="x", expand=True)
        
        self.temp_label = ttk.Label(temp_frame, text=f"{self.temp_var.get():.2f}", width=5)
        self.temp_label.pack(side="left", padx=(5, 0))
        self.temp_var.trace_add("write", self._on_temp_change)
        
        row += 1
        ttk.Label(frame, text="Max Tokens:").grid(row=row, column=0, sticky="w", pady=5)
        self.max_tokens_var = tk.IntVar(value=self._config.agent_config.max_tokens)
        ttk.Spinbox(
            frame,
            from_=256,
            to=32768,
            textvariable=self.max_tokens_var,
            width=10,
        ).grid(row=row, column=1, sticky="w", pady=5, padx=(10, 0))
    
    def _setup_project_step(self):
        """Set up the project configuration step."""
        frame = ttk.LabelFrame(self.content_frame, text="Project Setup", padding=15)
        frame.pack(fill="both", expand=True)
        
        frame.columnconfigure(1, weight=1)
        
        row = 0
        ttk.Label(frame, text="Project Name:").grid(row=row, column=0, sticky="w", pady=5)
        self.project_name_var = tk.StringVar(value=self._config.project_name)
        ttk.Entry(
            frame,
            textvariable=self.project_name_var,
            width=40,
        ).grid(row=row, column=1, sticky="ew", pady=5, padx=(10, 0))
        
        row += 1
        ttk.Label(frame, text="Location:").grid(row=row, column=0, sticky="w", pady=5)
        loc_frame = ttk.Frame(frame)
        loc_frame.grid(row=row, column=1, sticky="ew", pady=5, padx=(10, 0))
        loc_frame.columnconfigure(0, weight=1)
        
        default_location = str(Path.home() / "designspec_projects")
        self.project_location_var = tk.StringVar(
            value=self._config.project_location or default_location
        )
        ttk.Entry(
            loc_frame,
            textvariable=self.project_location_var,
            width=30,
        ).grid(row=0, column=0, sticky="ew")
        
        ttk.Button(
            loc_frame,
            text="Browse...",
            command=self._browse_location,
        ).grid(row=0, column=1, padx=(5, 0))
        
        row += 1
        ttk.Separator(frame, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=15
        )
        
        row += 1
        ttk.Label(frame, text="Template:").grid(row=row, column=0, sticky="nw", pady=5)
        
        template_frame = ttk.Frame(frame)
        template_frame.grid(row=row, column=1, sticky="ew", pady=5, padx=(10, 0))
        
        self.template_var = tk.StringVar(value=self._config.template)
        
        templates = [
            ("empty", "Empty Project", "Start with a blank DesignSpec"),
            ("cylinder", "Cylinder Domain", "Pre-configured cylinder domain with inlets"),
            ("box", "Box Domain", "Pre-configured box domain"),
            ("import", "Import Existing", "Import an existing DesignSpec file"),
        ]
        
        for value, label, description in templates:
            rb = ttk.Radiobutton(
                template_frame,
                text=label,
                value=value,
                variable=self.template_var,
                command=self._on_template_change,
            )
            rb.pack(anchor="w")
            ttk.Label(
                template_frame,
                text=description,
                foreground="gray",
                font=("TkDefaultFont", 9),
            ).pack(anchor="w", padx=(20, 0), pady=(0, 5))
        
        row += 1
        self.import_frame = ttk.Frame(frame)
        self.import_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=5)
        self.import_frame.columnconfigure(1, weight=1)
        
        ttk.Label(self.import_frame, text="Import File:").grid(row=0, column=0, sticky="w", padx=(20, 0))
        self.import_path_var = tk.StringVar(value=self._config.import_path or "")
        ttk.Entry(
            self.import_frame,
            textvariable=self.import_path_var,
            width=30,
        ).grid(row=0, column=1, sticky="ew", padx=(10, 0))
        ttk.Button(
            self.import_frame,
            text="Browse...",
            command=self._browse_import,
        ).grid(row=0, column=2, padx=(5, 0))
        
        self._on_template_change()
    
    def _setup_mode_step(self):
        """Set up the workflow mode selection step."""
        frame = ttk.LabelFrame(self.content_frame, text="Workflow Mode", padding=15)
        frame.pack(fill="both", expand=True)
        
        ttk.Label(
            frame,
            text="Select how you want to interact with the DesignSpec workflow:",
            wraplength=500,
        ).pack(anchor="w", pady=(0, 15))
        
        self.mode_var = tk.StringVar(value=self._config.workflow_mode)
        
        modes = [
            (
                "llm_first",
                "LLM-First (Recommended)",
                "Use an AI agent to help you create and modify your DesignSpec through "
                "natural conversation. The agent will propose changes as JSON patches "
                "that you can approve or reject.",
            ),
            (
                "legacy",
                "Legacy Mode",
                "Direct editing of the DesignSpec JSON without AI assistance. "
                "Suitable for advanced users who prefer manual control.",
            ),
        ]
        
        for value, label, description in modes:
            mode_frame = ttk.Frame(frame)
            mode_frame.pack(fill="x", pady=5)
            
            rb = ttk.Radiobutton(
                mode_frame,
                text=label,
                value=value,
                variable=self.mode_var,
            )
            rb.pack(anchor="w")
            
            ttk.Label(
                mode_frame,
                text=description,
                wraplength=480,
                foreground="gray",
                font=("TkDefaultFont", 9),
            ).pack(anchor="w", padx=(20, 0), pady=(0, 10))
        
        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=15)
        
        summary_frame = ttk.LabelFrame(frame, text="Configuration Summary", padding=10)
        summary_frame.pack(fill="x")
        
        provider_name = self._config.agent_config.provider
        for name, key, _ in SUPPORTED_PROVIDERS:
            if key == provider_name:
                provider_name = name
                break
        
        summary_text = (
            f"Provider: {provider_name}\n"
            f"Model: {self._config.agent_config.model}\n"
            f"Project: {self._config.project_name or '(not set)'}\n"
            f"Template: {self._config.template}"
        )
        
        ttk.Label(
            summary_frame,
            text=summary_text,
            justify="left",
        ).pack(anchor="w")
    
    def _on_provider_change(self, event=None):
        """Handle provider selection change."""
        selected_name = self.provider_var.get()
        
        for name, key, default_model in SUPPORTED_PROVIDERS:
            if name == selected_name:
                self._config.agent_config.provider = key
                self._config.agent_config.model = default_model
                
                models = PROVIDER_MODELS.get(key, [])
                self.model_combo["values"] = models
                if default_model in models:
                    self.model_combo.current(models.index(default_model))
                elif models:
                    self.model_combo.current(0)
                
                self._load_api_key_for_provider(key)
                break
    
    def _load_api_key_for_provider(self, provider: str):
        """Load API key for the selected provider."""
        env_var = PROVIDER_ENV_VARS.get(provider)
        if env_var:
            env_value = os.environ.get(env_var)
            if env_value:
                self.api_key_var.set("")
                self._config.agent_config.api_key = env_value
                return
        
        saved_key = self.secure_config.get_api_key(provider)
        if saved_key:
            self.api_key_var.set(saved_key)
            self._config.agent_config.api_key = saved_key
        else:
            self.api_key_var.set("")
            self._config.agent_config.api_key = None
    
    def _toggle_key_visibility(self):
        """Toggle API key visibility."""
        if self.show_key_var.get():
            self.api_key_entry.config(show="")
        else:
            self.api_key_entry.config(show="*")
    
    def _on_temp_change(self, *args):
        """Handle temperature change."""
        self.temp_label.config(text=f"{self.temp_var.get():.2f}")
    
    def _on_template_change(self):
        """Handle template selection change."""
        if self.template_var.get() == "import":
            for child in self.import_frame.winfo_children():
                child.configure(state="normal")
        else:
            for child in self.import_frame.winfo_children():
                if isinstance(child, (ttk.Entry, ttk.Button)):
                    child.configure(state="disabled")
    
    def _browse_location(self):
        """Browse for project location."""
        path = filedialog.askdirectory(
            title="Select Project Location",
            initialdir=self.project_location_var.get() or str(Path.home()),
        )
        if path:
            self.project_location_var.set(path)
    
    def _browse_import(self):
        """Browse for import file."""
        path = filedialog.askopenfilename(
            title="Select DesignSpec File",
            filetypes=[
                ("DesignSpec Files", "*.json *.yaml *.yml"),
                ("JSON Files", "*.json"),
                ("YAML Files", "*.yaml *.yml"),
                ("All Files", "*.*"),
            ],
        )
        if path:
            self.import_path_var.set(path)
    
    def _load_saved_config(self):
        """Load saved configuration into _config (before UI is created)."""
        saved_provider = self.secure_config.get_config("last_provider")
        if saved_provider:
            self._config.agent_config.provider = saved_provider
        
        saved_model = self.secure_config.get_config("last_model")
        if saved_model:
            self._config.agent_config.model = saved_model
        
        saved_key = self.secure_config.get_api_key(self._config.agent_config.provider)
        if saved_key:
            self._config.agent_config.api_key = saved_key
    
    def _save_current_step(self):
        """Save the current step's configuration."""
        if self._current_step == 0:
            if hasattr(self, 'provider_var'):
                for name, key, _ in SUPPORTED_PROVIDERS:
                    if name == self.provider_var.get():
                        self._config.agent_config.provider = key
                        break
                
                self._config.agent_config.model = self.model_var.get()
                
                api_key = self.api_key_var.get().strip()
                if api_key:
                    self._config.agent_config.api_key = api_key
                    self.secure_config.store_api_key(
                        self._config.agent_config.provider,
                        api_key,
                    )
                
                self._config.agent_config.temperature = self.temp_var.get()
                self._config.agent_config.max_tokens = self.max_tokens_var.get()
                
                self.secure_config.store_config("last_provider", self._config.agent_config.provider)
                self.secure_config.store_config("last_model", self._config.agent_config.model)
            
        elif self._current_step == 1:
            self._config.project_name = self.project_name_var.get().strip()
            self._config.project_location = self.project_location_var.get().strip()
            self._config.template = self.template_var.get()
            
            if self._config.template == "import":
                self._config.import_path = self.import_path_var.get().strip()
            else:
                self._config.import_path = None
                
        elif self._current_step == 2:
            self._config.workflow_mode = self.mode_var.get()
    
    def _validate_current_step(self) -> bool:
        """Validate the current step."""
        if self._current_step == 0:
            if not self._config.agent_config.api_key:
                env_var = PROVIDER_ENV_VARS.get(self._config.agent_config.provider)
                if env_var and os.environ.get(env_var):
                    return True
                
                messagebox.showwarning(
                    "Missing API Key",
                    f"Please enter an API key for {self._config.agent_config.provider}.",
                )
                return False
            
        elif self._current_step == 1:
            if not self._config.project_name:
                messagebox.showwarning(
                    "Missing Project Name",
                    "Please enter a project name.",
                )
                return False
            
            if not self._config.project_location:
                messagebox.showwarning(
                    "Missing Location",
                    "Please select a project location.",
                )
                return False
            
            if self._config.template == "import" and not self._config.import_path:
                messagebox.showwarning(
                    "Missing Import File",
                    "Please select a DesignSpec file to import.",
                )
                return False
        
        return True
    
    def _on_next(self):
        """Handle next button click."""
        self._save_current_step()
        
        if not self._validate_current_step():
            return
        
        if self._current_step < len(self._steps) - 1:
            self._show_step(self._current_step + 1)
        else:
            self._on_finish()
    
    def _on_back(self):
        """Handle back button click."""
        self._save_current_step()
        
        if self._current_step > 0:
            self._show_step(self._current_step - 1)
    
    def _on_finish(self):
        """Handle finish button click."""
        self._save_current_step()
        self.on_complete(self._config)
        self.destroy()
    
    def _on_cancel(self):
        """Handle cancel button click."""
        if self.on_cancel_callback:
            self.on_cancel_callback()
        self.destroy()
