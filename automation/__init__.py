"""
Agentic Organ Generation - Automation Scripts (Part C)

This module provides automation scripts for LLM API calls and agent behavior.
It enables programmatic interaction with LLM APIs to generate and validate
organ structures.

Main Components:
    - agent_runner: Generic agent runner that works with standard LLM APIs
    - llm_client: Client for interacting with LLM APIs (OpenAI, Anthropic, etc.)
    - task_templates: Pre-built task prompts for common operations
    - workflow: Single Agent Organ Generator V1 workflow

Workflows:
    - SingleAgentOrganGeneratorV1: Interactive workflow for organ structure generation
      that guides users through project setup, requirements, generation, review,
      and finalization.

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
    
    >>> # Or use the workflow
    >>> from automation import SingleAgentOrganGeneratorV1
    >>> workflow = SingleAgentOrganGeneratorV1(runner)
    >>> context = workflow.run()
"""

from .agent_runner import AgentRunner, AgentConfig, TaskResult, create_agent
from .llm_client import LLMClient, LLMConfig
from .workflow import (
    SingleAgentOrganGeneratorV1,
    WorkflowState,
    ProjectContext,
    ObjectContext,
    ObjectRequirements,
    IdentitySection,
    FrameOfReferenceSection,
    DomainSection,
    PortSpec,
    InletsOutletsSection,
    TopologySection,
    GeometrySection,
    ConstraintsSection,
    EmbeddingExportSection,
    AcceptanceCriteriaSection,
    QUESTION_GROUPS,
    run_single_agent_workflow,
)

__all__ = [
    "AgentRunner",
    "AgentConfig",
    "TaskResult",
    "LLMClient",
    "LLMConfig",
    "create_agent",
    "SingleAgentOrganGeneratorV1",
    "WorkflowState",
    "ProjectContext",
    "ObjectContext",
    "ObjectRequirements",
    "IdentitySection",
    "FrameOfReferenceSection",
    "DomainSection",
    "PortSpec",
    "InletsOutletsSection",
    "TopologySection",
    "GeometrySection",
    "ConstraintsSection",
    "EmbeddingExportSection",
    "AcceptanceCriteriaSection",
    "QUESTION_GROUPS",
    "run_single_agent_workflow",
]
