"""
Legacy GUI components.

This module contains deprecated GUI components that are kept for backwards
compatibility but should not be used in new code.

The MainWindow class with its tabbed multi-panel layout has been deprecated
in favor of the simplified conversation layout in the configuration wizard.
"""

from .main_window import MainWindow, launch_gui

__all__ = ["MainWindow", "launch_gui"]
