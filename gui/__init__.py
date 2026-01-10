"""
Organ Generator GUI

A graphical user interface for the Agentic Organ Generation system.
Provides workflow selection, agent configuration, and STL visualization.

Main Components:
    - MainWindow: Primary application window with three-panel layout
    - WorkflowManager: Orchestrates Single Agent workflows
    - STLViewer: 3D visualization of generated STL files
    - SecureConfig: Encrypted API key storage

Usage:
    >>> from gui import launch_gui
    >>> launch_gui()
    
    Or run as module:
    $ python -m gui

Requirements:
    - Python 3.8+ with tkinter support
    - matplotlib (for 3D visualization)
    - trimesh (for STL loading)
"""

__version__ = "1.0.0"

from .security import SecureConfig

_import_error = None

try:
    from .main_window import MainWindow, launch_gui as _launch_gui
    from .workflow_manager import WorkflowManager
    from .stl_viewer import STLViewer
    from .agent_config import AgentConfigPanel
    
    def launch_gui():
        """Launch the Organ Generator GUI."""
        _launch_gui()
    
    __all__ = [
        "MainWindow",
        "launch_gui",
        "WorkflowManager",
        "STLViewer",
        "SecureConfig",
        "AgentConfigPanel",
        "__version__",
    ]
except ImportError as e:
    _import_error = e
    
    MainWindow = None
    WorkflowManager = None
    STLViewer = None
    AgentConfigPanel = None
    
    def launch_gui():
        """Fallback launch_gui that raises a clear error."""
        raise ImportError(
            f"GUI components not available: {_import_error}. "
            "Make sure tkinter is installed (usually included with Python on desktop systems). "
            "On Linux, you may need to install python3-tk: sudo apt-get install python3-tk"
        )
    
    __all__ = ["SecureConfig", "launch_gui", "__version__"]
