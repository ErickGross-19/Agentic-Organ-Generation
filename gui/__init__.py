"""
Organ Generator GUI

A graphical user interface for the Agentic Organ Generation system.
Provides workflow selection, agent configuration, and STL visualization.

Main Components:
    - MainWindow: Primary application window with three-panel layout
    - WorkflowManager: Orchestrates Single Agent and MOGS workflows
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

from .security import SecureConfig

try:
    from .main_window import MainWindow, launch_gui
    from .workflow_manager import WorkflowManager
    from .stl_viewer import STLViewer
    from .agent_config import AgentConfigPanel
    
    __all__ = [
        "MainWindow",
        "launch_gui",
        "WorkflowManager",
        "STLViewer",
        "SecureConfig",
        "AgentConfigPanel",
    ]
except ImportError as e:
    import warnings
    warnings.warn(
        f"GUI components not available: {e}. "
        "Make sure tkinter is installed (usually included with Python on desktop systems)."
    )
    
    MainWindow = None
    launch_gui = None
    WorkflowManager = None
    STLViewer = None
    AgentConfigPanel = None
    
    __all__ = ["SecureConfig"]
