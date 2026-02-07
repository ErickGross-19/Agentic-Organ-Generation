# Automation Module

LLM agent layer for AI-driven scaffold design and DesignSpec editing. Provides conversational interfaces for creating and iterating on DesignSpec JSON files, with structured patch proposals, self-correction, and session memory.

## Overview

| Component | Purpose |
|-----------|---------|
| **DesignSpec Workflow** | Conversation-driven spec editing with patch proposals, run management, and auto-analysis (recommended) |
| **DesignSpec LLM Agent** | LLM-first agent that outputs structured directives with patches, questions, and run requests |
| **LLM Client** | Unified interface for multiple LLM providers (OpenAI, Anthropic, xAI, Google, Mistral, Groq, local) |
| **Agent Runner** | Task orchestration with iteration control and artifact management |
| **V5 Goal-Driven Controller** | WorldModel-based agent with capabilities and policies |
| **Task Templates** | Pre-built prompts for generation, validation, and iteration workflows |

## Module Structure

```
automation/
├── __init__.py
├── llm_client.py              # Multi-provider LLM client
├── agent_runner.py            # Task orchestration and execution
├── workflow.py                # V3/V4 workflow (legacy)
├── cli.py                     # Command-line interface
├── designspec_session.py      # Project session management (spec, patches, artifacts)
├── designspec_agent.py        # Rule-based patch agent (legacy)
├── workflows/
│   └── designspec_workflow.py # Main DesignSpec workflow with auto-analysis
├── designspec_llm/            # LLM-first agent package (see designspec_llm/README.md)
│   ├── agent.py               # DesignSpecLLMAgent
│   ├── directive.py           # Structured output schema
│   ├── context_builder.py     # Context packing for LLM
│   ├── prompt_builder.py      # System/user prompt construction
│   ├── artifact_indexer.py    # Run history indexing
│   ├── error_taxonomy.py      # 30+ error patterns with fixes
│   ├── patch_generator.py     # Targeted RFC 6902 patch generation
│   ├── run_analyzer.py        # Auto-analyze failures and propose fixes
│   ├── task_context.py        # Persistent goal tracking
│   ├── session_memory.py      # Decision and error resolution memory
│   └── policy_reference.py    # Policy docs for LLM context
├── single_agent_organ_generation/
│   └── v5/                    # V5 Goal-Driven Controller
│       ├── controller.py      # Main agent loop
│       ├── world_model.py     # Single source of truth
│       ├── goals.py           # Goal definitions and tracking
│       ├── policies.py        # Safe fix and approval policies
│       ├── plan_synthesizer.py
│       └── io/                # CLI and GUI IO adapters
└── task_templates/            # Structured prompt templates
```

## DesignSpec Workflow (Recommended)

The primary way to interact with the system. Users describe changes in natural language, the agent proposes JSON patches (RFC 6902), and the DesignSpec pipeline executes the result.

### Integration Points

The workflow is accessible through two frontends:
- **Web UI**: The FastAPI `DesignSpecBridge` in `backend/app/api/designspec.py` wraps the workflow for HTTP
- **Desktop GUI**: The `DesignSpecWorkflowManager` in `gui/designspec_workflow_manager.py` wraps it for tkinter (legacy)

### Patch Approval Flow

1. User sends a message describing desired changes
2. Agent analyzes the request and current spec
3. Agent proposes JSON Patches (RFC 6902) with explanation
4. User reviews and approves or rejects
5. If approved, patch is applied and compile runs automatically
6. Results are reported back

### Auto-Analysis

When a pipeline run fails, the `RunAnalyzer` automatically:
1. Parses the error through the `ErrorParser` (30+ patterns in `error_taxonomy.py`)
2. Classifies the error type with confidence scores
3. Generates targeted fix patches via `PatchGenerator`
4. Proposes the fix to the user for approval

### Session State

Each project stores session state in `project_dir/session/`:
- `task_context.json` - Current goal, sub-tasks, blockers
- `memory.json` - Decisions, error resolutions, user preferences

### Usage

```python
from automation.workflows.designspec_workflow import DesignSpecWorkflow

workflow = DesignSpecWorkflow()
workflow.on_start(project_dir="/path/to/project")
workflow.on_user_message("Create a box domain 20mm x 60mm x 30mm")
workflow.approve_patch(patch_id)
workflow.run_full()
```

## LLM Client

Unified interface for multiple LLM providers. Configured via environment variables or explicit parameters.

### Supported Providers

| Provider | Environment Variable |
|----------|---------------------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |
| xAI/Grok | `XAI_API_KEY` |
| Google/Gemini | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |
| Groq | `GROQ_API_KEY` |
| Local | Any OpenAI-compatible API via `api_base` |

### Usage

```python
from automation.llm_client import LLMClient

client = LLMClient(provider="openai")
response = client.chat([{"role": "user", "content": "Design a hepatic lobule scaffold"}])
```

## V5 Goal-Driven Controller

WorldModel-based agent that maintains a single source of truth about the design state, tracks goals, and applies policies for safe fixes and approval workflows. Uses IO adapters for CLI or GUI interaction.

```python
from automation.single_agent_organ_generation.v5 import OrganGenerationControllerV5

controller = OrganGenerationControllerV5(provider="openai")
controller.run()
```

## CLI

```bash
python -m automation.cli generate --organ liver --segments 500 --output ./output
python -m automation.cli validate --input structure.stl --stage both
python -m automation.cli interactive
```
