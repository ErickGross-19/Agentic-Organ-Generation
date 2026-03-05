"""
DesignSpec LLM Agent Package

This package provides an LLM-driven agent for DesignSpec workflows that:
1. Interprets user messages directly via LLM (no regex parsing as primary path)
2. Asks clarification questions when needed
3. Proposes RFC 6902 JSON Patch operations to update the spec
4. Decides when to run the pipeline and how far (run_until stage)
5. Uses previous run data to propose targeted fixes
6. Guides the conversation and chooses the next action

Main Components:
- DesignSpecDirective: Structured LLM output contract
- ContextBuilder: Builds context packs from session and artifacts
- PromptBuilder: Constructs system and user prompts
- DesignSpecLLMAgent: LLM-first agent with fallback
- ArtifactIndexer: Indexes run history for context

Usage:
    from automation.designspec_llm import (
        DesignSpecLLMAgent,
        create_llm_agent,
        ContextBuilder,
        PromptBuilder,
    )
    
    # Create agent for a session
    agent = create_llm_agent(session, llm_client)
    
    # Process user message
    response = agent.process_message("Create a box domain 20mm x 60mm x 30mm")
"""

from .directive import (
    DesignSpecDirective,
    Question,
    RunRequest,
    ContextRequest,
    DirectiveParseError,
    from_json,
    validate_patches,
    validate_json_patch,
    create_error_directive,
    create_fallback_directive,
    PIPELINE_STAGES,
    VALID_PATCH_OPS,
)

from .context_builder import (
    ContextBuilder,
    ContextPack,
    SpecSummary,
    RunSummary,
    ValidationSummary,
    PatchHistoryEntry,
)

from .prompt_builder import (
    PromptBuilder,
    get_system_prompt,
    build_user_prompt,
    build_retry_prompt,
    build_context_request_prompt,
    SYSTEM_PROMPT,
)

from .agent import (
    DesignSpecLLMAgent,
    AgentTurnLog,
    create_llm_agent,
)

from .artifact_indexer import (
    ArtifactIndexer,
    ArtifactIndex,
    RunArtifactEntry,
)

__all__ = [
    # Directive
    "DesignSpecDirective",
    "Question",
    "RunRequest",
    "ContextRequest",
    "DirectiveParseError",
    "from_json",
    "validate_patches",
    "validate_json_patch",
    "create_error_directive",
    "create_fallback_directive",
    "PIPELINE_STAGES",
    "VALID_PATCH_OPS",
    # Context Builder
    "ContextBuilder",
    "ContextPack",
    "SpecSummary",
    "RunSummary",
    "ValidationSummary",
    "PatchHistoryEntry",
    # Prompt Builder
    "PromptBuilder",
    "get_system_prompt",
    "build_user_prompt",
    "build_retry_prompt",
    "build_context_request_prompt",
    "SYSTEM_PROMPT",
    # Agent
    "DesignSpecLLMAgent",
    "AgentTurnLog",
    "create_llm_agent",
    # Artifact Indexer
    "ArtifactIndexer",
    "ArtifactIndex",
    "RunArtifactEntry",
]
