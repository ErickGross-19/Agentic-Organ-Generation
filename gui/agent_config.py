"""
Agent Configuration Panel

Provides UI for configuring LLM agents including provider selection,
API key management, and model configuration.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass

from .security import SecureConfig


SUPPORTED_PROVIDERS = [
    ("OpenAI", "openai", "gpt-4o"),
    ("Anthropic", "anthropic", "claude-3-5-sonnet-20241022"),
    ("Google Gemini", "google", "gemini-2.0-flash"),
    ("Mistral", "mistral", "mistral-large-latest"),
    ("xAI (Grok)", "xai", "grok-2"),
    ("Groq", "groq", "llama-3.3-70b-versatile"),
    ("Local (OpenAI-compatible)", "local", "local-model"),
]

PROVIDER_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "xai": "XAI_API_KEY",
    "groq": "GROQ_API_KEY",
    "local": None,
}

PROVIDER_MODELS = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "o1", "o1-mini", "o1-preview", "gpt-3.5-turbo"],
    "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
    "google": ["gemini-2.0-flash", "gemini-2.0-flash-exp", "gemini-2.0-flash-thinking-exp", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-pro"],
    "mistral": ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest", "codestral-latest"],
    "xai": ["grok-2", "grok-2-mini", "grok-beta"],
    "groq": ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    "local": ["local-model"],
}


@dataclass
class AgentConfiguration:
    """Configuration for an LLM agent."""
    provider: str = "openai"
    model: str = "gpt-4"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096


class AgentConfigPanel(ttk.Frame):
    """
    Panel for configuring LLM agent settings.
    
    Provides UI for:
    - Provider selection
    - Model selection
    - API key input and secure storage
    - Advanced settings (temperature, max tokens)
    
    Parameters
    ----------
    parent : tk.Widget
        Parent widget
    secure_config : SecureConfig
        Secure configuration storage
    on_config_change : Callable, optional
        Callback when configuration changes
    """
    
    def __init__(
        self,
        parent: tk.Widget,
        secure_config: SecureConfig,
        on_config_change: Optional[Callable[[AgentConfiguration], None]] = None,
    ):
        super().__init__(parent)
        
        self.secure_config = secure_config
        self.on_config_change = on_config_change
        self._config = AgentConfiguration()
        
        self._setup_ui()
        self._load_saved_config()
    
    def _setup_ui(self):
        """Set up the configuration panel UI."""
        self.columnconfigure(1, weight=1)
        
        row = 0
        
        ttk.Label(self, text="Provider:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.provider_var = tk.StringVar(value="openai")
        self.provider_combo = ttk.Combobox(
            self,
            textvariable=self.provider_var,
            values=[p[0] for p in SUPPORTED_PROVIDERS],
            state="readonly",
            width=30,
        )
        self.provider_combo.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        self.provider_combo.bind("<<ComboboxSelected>>", self._on_provider_change)
        self.provider_combo.current(0)
        
        row += 1
        
        ttk.Label(self, text="Model:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.model_var = tk.StringVar(value="gpt-4")
        self.model_combo = ttk.Combobox(
            self,
            textvariable=self.model_var,
            values=PROVIDER_MODELS["openai"],
            width=30,
        )
        self.model_combo.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_config_update)
        
        row += 1
        
        ttk.Label(self, text="API Key:").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        self.api_key_frame = ttk.Frame(self)
        self.api_key_frame.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        self.api_key_frame.columnconfigure(0, weight=1)
        
        self.api_key_var = tk.StringVar()
        self.api_key_entry = ttk.Entry(
            self.api_key_frame,
            textvariable=self.api_key_var,
            show="*",
            width=25,
        )
        self.api_key_entry.grid(row=0, column=0, sticky="ew")
        self.api_key_entry.bind("<KeyRelease>", self._on_config_update)
        
        self.show_key_var = tk.BooleanVar(value=False)
        self.show_key_btn = ttk.Checkbutton(
            self.api_key_frame,
            text="Show",
            variable=self.show_key_var,
            command=self._toggle_key_visibility,
        )
        self.show_key_btn.grid(row=0, column=1, padx=(5, 0))
        
        self.save_key_btn = ttk.Button(
            self.api_key_frame,
            text="Save",
            command=self._save_api_key,
            width=6,
        )
        self.save_key_btn.grid(row=0, column=2, padx=(5, 0))
        
        row += 1
        
        self.api_base_frame = ttk.Frame(self)
        self.api_base_frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        self.api_base_frame.columnconfigure(1, weight=1)
        
        ttk.Label(self.api_base_frame, text="API Base URL:").grid(row=0, column=0, sticky="w")
        self.api_base_var = tk.StringVar()
        self.api_base_entry = ttk.Entry(
            self.api_base_frame,
            textvariable=self.api_base_var,
            width=35,
        )
        self.api_base_entry.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        self.api_base_entry.bind("<KeyRelease>", self._on_config_update)
        self.api_base_frame.grid_remove()
        
        row += 1
        
        advanced_frame = ttk.LabelFrame(self, text="Advanced Settings")
        advanced_frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=5, pady=10)
        advanced_frame.columnconfigure(1, weight=1)
        advanced_frame.columnconfigure(3, weight=1)
        
        ttk.Label(advanced_frame, text="Temperature:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.temp_var = tk.DoubleVar(value=0.7)
        self.temp_scale = ttk.Scale(
            advanced_frame,
            from_=0.0,
            to=1.0,
            variable=self.temp_var,
            orient="horizontal",
            command=self._on_temp_change,
        )
        self.temp_scale.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        self.temp_label = ttk.Label(advanced_frame, text="0.7")
        self.temp_label.grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(advanced_frame, text="Max Tokens:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.max_tokens_var = tk.IntVar(value=4096)
        self.max_tokens_spin = ttk.Spinbox(
            advanced_frame,
            from_=256,
            to=32768,
            textvariable=self.max_tokens_var,
            width=10,
        )
        self.max_tokens_spin.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        self.max_tokens_spin.bind("<KeyRelease>", self._on_config_update)
        
        row += 1
        
        self.status_var = tk.StringVar(value="")
        self.status_label = ttk.Label(self, textvariable=self.status_var, foreground="gray")
        self.status_label.grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=5)
    
    def _on_provider_change(self, event=None):
        """Handle provider selection change."""
        display_name = self.provider_var.get()
        
        provider_id = None
        default_model = None
        for name, pid, model in SUPPORTED_PROVIDERS:
            if name == display_name:
                provider_id = pid
                default_model = model
                break
        
        if provider_id:
            self._config.provider = provider_id
            
            models = PROVIDER_MODELS.get(provider_id, ["default"])
            self.model_combo["values"] = models
            if default_model in models:
                self.model_var.set(default_model)
            elif models:
                self.model_var.set(models[0])
            
            if provider_id == "local":
                self.api_base_frame.grid()
            else:
                self.api_base_frame.grid_remove()
            
            saved_key = self.secure_config.get_api_key(provider_id)
            if saved_key:
                self.api_key_var.set(saved_key)
                self.status_var.set(f"Loaded saved API key for {display_name}")
            else:
                self.api_key_var.set("")
                env_var = PROVIDER_ENV_VARS.get(provider_id)
                if env_var:
                    self.status_var.set(f"Enter API key or set {env_var}")
                else:
                    self.status_var.set("")
        
        self._on_config_update()
    
    def _on_config_update(self, event=None):
        """Handle configuration update."""
        display_name = self.provider_var.get()
        for name, pid, _ in SUPPORTED_PROVIDERS:
            if name == display_name:
                self._config.provider = pid
                break
        
        self._config.model = self.model_var.get()
        self._config.api_key = self.api_key_var.get() or None
        self._config.api_base = self.api_base_var.get() or None
        self._config.temperature = self.temp_var.get()
        
        try:
            self._config.max_tokens = int(self.max_tokens_var.get())
        except (ValueError, tk.TclError):
            self._config.max_tokens = 4096
        
        if self.on_config_change:
            self.on_config_change(self._config)
    
    def _on_temp_change(self, value):
        """Handle temperature slider change."""
        temp = float(value)
        self.temp_label.config(text=f"{temp:.2f}")
        self._on_config_update()
    
    def _toggle_key_visibility(self):
        """Toggle API key visibility."""
        if self.show_key_var.get():
            self.api_key_entry.config(show="")
        else:
            self.api_key_entry.config(show="*")
    
    def _save_api_key(self):
        """Save API key to secure storage."""
        api_key = self.api_key_var.get()
        if not api_key:
            messagebox.showwarning("Warning", "Please enter an API key to save.")
            return
        
        if self.secure_config.store_api_key(self._config.provider, api_key):
            self.status_var.set(f"API key saved for {self._config.provider}")
            messagebox.showinfo("Success", "API key saved securely.")
        else:
            messagebox.showerror("Error", "Failed to save API key.")
    
    def _load_saved_config(self):
        """Load saved configuration."""
        saved_provider = self.secure_config.get_config("last_provider", "openai")
        saved_model = self.secure_config.get_config("last_model")
        
        for i, (name, pid, _) in enumerate(SUPPORTED_PROVIDERS):
            if pid == saved_provider:
                self.provider_combo.current(i)
                self._on_provider_change()
                break
        
        if saved_model:
            self.model_var.set(saved_model)
        
        saved_temp = self.secure_config.get_config("temperature", 0.7)
        self.temp_var.set(saved_temp)
        self.temp_label.config(text=f"{saved_temp:.2f}")
        
        saved_max_tokens = self.secure_config.get_config("max_tokens", 4096)
        self.max_tokens_var.set(saved_max_tokens)
    
    def save_current_config(self):
        """Save current configuration for next session."""
        self.secure_config.store_config("last_provider", self._config.provider)
        self.secure_config.store_config("last_model", self._config.model)
        self.secure_config.store_config("temperature", self._config.temperature)
        self.secure_config.store_config("max_tokens", self._config.max_tokens)
    
    def get_configuration(self) -> AgentConfiguration:
        """Get current agent configuration."""
        return self._config
    
    def validate_configuration(self) -> tuple:
        """
        Validate current configuration.
        
        Returns
        -------
        tuple
            (is_valid: bool, error_message: str)
        """
        if self._config.provider == "local":
            if not self._config.api_base:
                return False, "API Base URL is required for local provider"
        else:
            if not self._config.api_key:
                env_var = PROVIDER_ENV_VARS.get(self._config.provider)
                if env_var:
                    import os
                    if not os.environ.get(env_var):
                        return False, f"API key required. Enter key or set {env_var}"
        
        return True, ""
