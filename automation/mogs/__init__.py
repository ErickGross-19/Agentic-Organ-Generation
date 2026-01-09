"""
MOGS - MultiAgentOrgan Generation System

A comprehensive system for LLM-driven automated design and validation
of 3D vascular organ structures using a three-agent workflow.

Key Components:
- Object Identity Model: UUID + human-readable name
- Three-Agent Workflow: CSA (Concept & Spec), CBA (Coding & Build), VQA (Validation & QA)
- Approval Gates: Spec, Code+Plan, Results
- Three-Script Contract: generate, analyze, finalize
- Version Retention: Keep last 5 versions with safe garbage collection
- Safety Constraints: Restricted writes, no network, no subprocess

Usage:
    from automation.mogs import MOGSRunner, create_mogs_runner
    
    # Create a runner
    runner = create_mogs_runner(objects_base_dir="./objects")
    
    # Create a new object
    folder_manager = runner.create_object("My Liver Structure")
    
    # Run the workflow
    requirements = {
        "domain": {"type": "box", "size_m": [0.02, 0.06, 0.03]},
        "topology": {"kind": "tree", "target_terminals": 100},
    }
    result = runner.run_workflow(
        object_uuid=folder_manager.object_uuid,
        requirements=requirements,
        user_description="Generate a liver vascular network",
    )
"""

# Models
from .models import (
    # Enums
    ApprovalStatus,
    VersionStatus,
    AgentType,
    GateType,
    # Data classes
    RetentionPolicy,
    ObjectManifest,
    RiskFlag,
    SpecVersion,
    ExpectedArtifact,
    ScriptManifest,
    RunEntry,
    RunIndex,
    ValidationCheck,
    ValidationReport,
    ApprovalRecord,
    FinalManifest,
    # Utilities
    generate_object_uuid,
    get_timestamp,
)

# Folder management
from .folder_manager import (
    FolderManager,
    create_new_object,
    load_object,
    FOLDER_STRUCTURE,
)

# Object registry
from .object_registry import (
    ObjectRegistry,
    ObjectRegistryEntry,
)

# Agents
from .agents import (
    ConceptSpecAgent,
    CodingBuildAgent,
    ValidationQAAgent,
)

# Approval gates
from .gates import (
    ApprovalChoice,
    GateContext,
    GateResult,
    ApprovalGate,
    SpecApprovalGate,
    CodeApprovalGate,
    ResultsApprovalGate,
    GateManager,
)

# Retention
from .retention import (
    RetentionAction,
    RetentionReport,
    RetentionManager,
)

# Safety
from .safety import (
    SafetyConfig,
    SafetyViolation,
    SafetyManager,
    ENV_ALLOWLIST,
    DEFAULT_LIMITS,
    create_safe_runner_script,
)

# Runner
from .runner import (
    WorkflowState,
    WorkflowContext,
    WorkflowResult,
    MOGSRunner,
    create_mogs_runner,
)

__all__ = [
    # Models - Enums
    "ApprovalStatus",
    "VersionStatus",
    "AgentType",
    "GateType",
    # Models - Data classes
    "RetentionPolicy",
    "ObjectManifest",
    "RiskFlag",
    "SpecVersion",
    "ExpectedArtifact",
    "ScriptManifest",
    "RunEntry",
    "RunIndex",
    "ValidationCheck",
    "ValidationReport",
    "ApprovalRecord",
    "FinalManifest",
    # Models - Utilities
    "generate_object_uuid",
    "get_timestamp",
    # Folder management
    "FolderManager",
    "create_new_object",
    "load_object",
    "FOLDER_STRUCTURE",
    # Object registry
    "ObjectRegistry",
    "ObjectRegistryEntry",
    # Agents
    "ConceptSpecAgent",
    "CodingBuildAgent",
    "ValidationQAAgent",
    # Approval gates
    "ApprovalChoice",
    "GateContext",
    "GateResult",
    "ApprovalGate",
    "SpecApprovalGate",
    "CodeApprovalGate",
    "ResultsApprovalGate",
    "GateManager",
    # Retention
    "RetentionAction",
    "RetentionReport",
    "RetentionManager",
    # Safety
    "SafetyConfig",
    "SafetyViolation",
    "SafetyManager",
    "ENV_ALLOWLIST",
    "DEFAULT_LIMITS",
    "create_safe_runner_script",
    # Runner
    "WorkflowState",
    "WorkflowContext",
    "WorkflowResult",
    "MOGSRunner",
    "create_mogs_runner",
]

__version__ = "1.0.0"
