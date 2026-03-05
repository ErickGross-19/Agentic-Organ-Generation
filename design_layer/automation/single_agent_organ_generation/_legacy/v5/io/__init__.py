"""
IO Adapters for V5 Controller

Provides unified interface for CLI and GUI without input() monkeypatching.
"""

from .base_io import BaseIOAdapter, IOMessage, IOMessageKind
from .cli_io import CLIIOAdapter
from .gui_io import GUIIOAdapter

__all__ = [
    "BaseIOAdapter",
    "IOMessage",
    "IOMessageKind",
    "CLIIOAdapter",
    "GUIIOAdapter",
]
