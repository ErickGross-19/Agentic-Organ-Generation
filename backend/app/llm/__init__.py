"""
LLM integration module for scaffold design assistant.
"""

from .agent import ScaffoldAgent, get_agent
from .prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_COMPACT
from .tools import (
    DEFAULT_SUGGESTIONS,
    GENERAL_SUGGESTIONS,
    SCAFFOLD_TOOLS,
    TOOL_TO_SCAFFOLD_TYPE,
)

__all__ = [
    # Agent
    "ScaffoldAgent",
    "get_agent",
    # Prompts
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_COMPACT",
    # Tools
    "SCAFFOLD_TOOLS",
    "TOOL_TO_SCAFFOLD_TYPE",
    "DEFAULT_SUGGESTIONS",
    "GENERAL_SUGGESTIONS",
]
