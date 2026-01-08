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
from .execution_modes import (
    ExecutionMode,
    DEFAULT_EXECUTION_MODE,
    parse_execution_mode,
    should_write_script,
    should_pause_for_review,
    should_execute,
    get_mode_description,
)
from .script_artifacts import (
    ArtifactProfile,
    ArtifactManifest,
    ArtifactsJson,
    generation_expected_files,
    analysis_expected_files,
    final_expected_files,
    get_artifact_profile,
)
from .script_writer import (
    write_script,
    get_run_command,
    extract_code_block,
    scan_for_suspicious_patterns,
    ScriptWriteResult,
)
from .review_gate import (
    ReviewAction,
    ReviewResult,
    run_review_gate,
    interactive_review,
    auto_review,
)
from .subprocess_runner import (
    run_script,
    RunResult,
    print_run_summary,
    DEFAULT_TIMEOUT_SECONDS,
)
from .artifact_verifier import (
    verify_artifacts,
    verify_generation_stage,
    verify_final_stage,
    save_manifest,
    print_verification_summary,
    VerificationResult,
    FileCheckResult,
)
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
    ORGAN_QUESTION_VARIANTS,
    detect_organ_type,
    get_tailored_questions,
    run_single_agent_workflow,
    RuleFlag,
    ProposedDefault,
    RuleEvaluationResult,
    PlannedQuestion,
    IntentParser,
    RuleEngine,
    QuestionPlanner,
    run_rule_based_capture,
)

__all__ = [
    # Core agent components
    "AgentRunner",
    "AgentConfig",
    "TaskResult",
    "LLMClient",
    "LLMConfig",
    "create_agent",
    # Execution modes
    "ExecutionMode",
    "DEFAULT_EXECUTION_MODE",
    "parse_execution_mode",
    "should_write_script",
    "should_pause_for_review",
    "should_execute",
    "get_mode_description",
    # Script artifacts
    "ArtifactProfile",
    "ArtifactManifest",
    "ArtifactsJson",
    "generation_expected_files",
    "analysis_expected_files",
    "final_expected_files",
    "get_artifact_profile",
    # Script writer
    "write_script",
    "get_run_command",
    "extract_code_block",
    "scan_for_suspicious_patterns",
    "ScriptWriteResult",
    # Review gate
    "ReviewAction",
    "ReviewResult",
    "run_review_gate",
    "interactive_review",
    "auto_review",
    # Subprocess runner
    "run_script",
    "RunResult",
    "print_run_summary",
    "DEFAULT_TIMEOUT_SECONDS",
    # Artifact verifier
    "verify_artifacts",
    "verify_generation_stage",
    "verify_final_stage",
    "save_manifest",
    "print_verification_summary",
    "VerificationResult",
    "FileCheckResult",
    # Workflow
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
    "ORGAN_QUESTION_VARIANTS",
    "detect_organ_type",
    "get_tailored_questions",
    "run_single_agent_workflow",
    "RuleFlag",
    "ProposedDefault",
    "RuleEvaluationResult",
    "PlannedQuestion",
    "IntentParser",
    "RuleEngine",
    "QuestionPlanner",
    "run_rule_based_capture",
]
