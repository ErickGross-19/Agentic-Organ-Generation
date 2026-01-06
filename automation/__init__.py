"""
Agentic Organ Generation - Automation Scripts (Part C)

This module provides automation scripts for LLM API calls and agent behavior.
It enables programmatic interaction with LLM APIs to generate and validate
organ structures.

Main Components:
    - agent_runner: Generic agent runner that works with standard LLM APIs
    - llm_client: Client for interacting with LLM APIs (OpenAI, Anthropic, etc.)
    - task_templates: Pre-built task prompts for common operations

Example:
    >>> from automation import AgentRunner, LLMClient
    >>> from automation.task_templates import generate_structure_prompt
    >>>
    >>> # Initialize client and runner
    >>> client = LLMClient(provider="openai", api_key="...")
    >>> runner = AgentRunner(client=client, repo_path="./")
    >>>
    >>> # Generate a structure
    >>> result = runner.run_task(
    ...     task=generate_structure_prompt(
    ...         organ_type="liver",
    ...         constraints={"plate_size": (200, 200, 200)}
    ...     )
    ... )
"""

from .agent_runner import AgentRunner, AgentConfig, TaskResult
from .llm_client import LLMClient, LLMConfig

__all__ = [
    "AgentRunner",
    "AgentConfig",
    "TaskResult",
    "LLMClient",
    "LLMConfig",
]
