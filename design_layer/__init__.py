"""
Design Orchestration Layer (Section 3).

This package consolidates the DesignSpec pipeline, policies, automation,
and GUI that drive the generation and embedding systems.

Subpackages
-----------
designspec : DesignSpec JSON schema, runner, and report generation
    - designspec.spec : DesignSpec dataclass
    - designspec.runner : DesignSpecRunner (the core 11-stage pipeline)
    - designspec.reports : run_report, serializers

aog_policies : Canonical policy dataclasses (30+)
    - GrowthPolicy, CollisionPolicy, EmbeddingPolicy, SpaceColonizationPolicy, etc.

automation : LLM-driven automation and CLI
    - automation.workflows.designspec_workflow : DesignSpecWorkflow
    - automation.designspec_llm : DesignSpecLLMAgent, SessionMemory, PatchGenerator
    - automation.llm_client : LLMClient
    - automation.cli : CLI entry point

gui : Desktop GUI (Tkinter-based)
    - gui.app : launch_gui()
    - gui.designspec_workflow_manager : DesignSpecWorkflowManager
"""
