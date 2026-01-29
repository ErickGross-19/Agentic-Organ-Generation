"""
DesignSpec GUI Application Entry Point

This module provides the main entry point for the DesignSpec GUI application.
It launches the configuration wizard and manages the workflow.

Usage:
    >>> from gui import launch_gui
    >>> launch_gui()
    
    Or run as module:
    $ python -m gui
"""

import warnings


def launch_gui():
    """
    Launch the DesignSpec GUI application.
    
    This is the recommended entry point for the GUI. It uses the legacy
    MainWindow internally but provides a clean interface for launching
    the application.
    """
    from ._legacy.main_window import MainWindow
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        app = MainWindow()
        app.run()


if __name__ == "__main__":
    launch_gui()
