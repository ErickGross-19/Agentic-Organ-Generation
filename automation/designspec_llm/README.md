# DesignSpec LLM Agent

This package implements an LLM-driven agent loop for iteratively designing and running DesignSpec JSON workflows through conversation with the user.

## Overview

The DesignSpec LLM Agent replaces the previous rule-based parsing system with a structured LLM-first approach where the LLM acts as the primary interpreter and planner. The agent receives user messages along with context about the current spec and recent artifacts, then returns structured directives that the system validates and executes.

## Architecture

The package consists of four main modules:

### directive.py - Structured Output Contract

Defines the `DesignSpecDirective` dataclass that represents the LLM's structured response. Every LLM output must conform to this schema:

```python
@dataclass
class DesignSpecDirective:
    assistant_message: str          # Message shown to user
    questions: List[Question]       # Clarification questions (optional)
    proposed_patches: List[dict]    # RFC 6902 JSON Patch operations (optional)
    run_request: RunRequest         # Pipeline run request (optional)
    context_requests: ContextRequest # Request for more context (optional)
    confidence: float               # 0-1 confidence score
    requires_approval: bool         # Whether user approval needed
    stop: bool                      # Whether workflow is complete
```

Key features:
- Robust JSON extraction from LLM output (handles markdown code blocks, leading/trailing text)
- RFC 6902 JSON Patch validation
- Field validation with detailed error messages
- Helper functions for creating error/fallback directives

### context_builder.py - Context Packing

The `ContextBuilder` class assembles context for the LLM from multiple sources:

- Current spec from `DesignSpecSession.get_spec()`
- Last runner result and validation/compile reports
- Recent runs from `project_dir/artifacts/run_*/*`
- Patch history summaries

Produces two context versions to manage token usage:
- `context_compact`: Spec summary + last run summary + key metrics (default)
- `context_full`: Includes full spec JSON and detailed artifact data

### prompt_builder.py - Prompt Construction

The `PromptBuilder` class creates prompts for the LLM:

- **System prompt**: Stable across session, defines agent role, constraints, and output schema
- **User prompt**: Per-turn prompt with user message + context pack + JSON output instructions

The system prompt includes:
- Agent role as "DesignSpec Iteration Agent"
- Hard constraints (JSON-only output, no code, respect units, minimal patches)
- Pipeline stages list
- Artifact reasoning guidelines
- Common failure patterns to detect
- Decision rules for patch minimalism, units sanity checks, etc.

### agent.py - LLM-First Agent

The `DesignSpecLLMAgent` class implements the agent loop:

1. Receives user message
2. Builds context pack via `ContextBuilder`
3. Constructs prompts via `PromptBuilder`
4. Calls LLM and parses response into `DesignSpecDirective`
5. Converts directive to `AgentResponse` for workflow integration
6. Logs all turns to `project_dir/logs/agent_turns.jsonl`

Falls back to legacy agent if LLM is unavailable.

### artifact_indexer.py - Run History Access

The `ArtifactIndexer` class maintains an index of run artifacts:

- Builds `project_dir/reports/artifact_index.json`
- Indexes: run_id, timestamp, stage outcomes, file list, extracted metrics
- Provides `select_relevant_runs()` for context building (last N runs + last successful)

## Directive Schema

The LLM must output a single JSON object with these fields:

```json
{
  "assistant_message": "string - message to show user",
  "questions": [
    {
      "id": "string - stable identifier",
      "question": "string - the question",
      "why_needed": "string - explanation",
      "default": "optional - suggested default"
    }
  ],
  "proposed_patches": [
    {
      "op": "add|remove|replace|move|copy|test",
      "path": "/json/pointer/path",
      "value": "required for add/replace/test",
      "from": "required for move/copy"
    }
  ],
  "run_request": {
    "run": true,
    "run_until": "stage_name",
    "reason": "why running",
    "expected_signal": "what to verify"
  },
  "context_requests": {
    "need_full_spec": false,
    "need_last_run_report": false,
    "need_validity_report": false,
    "need_network_artifact": false,
    "need_specific_files": [],
    "need_more_history": false,
    "why": "explanation"
  },
  "confidence": 0.85,
  "requires_approval": true,
  "stop": false
}
```

## Pipeline Stages

Valid `run_until` values:
- `compile_policies`
- `compile_domains`
- `component_build`
- `component_mesh`
- `mesh_merge`
- `union_void`
- `embed`
- `port_recarve`
- `validity`
- `full`

## Usage

### CLI Command

```bash
agentic-organ designspec --project-dir /path/to/project
```

Options:
- `--project-dir, -p`: Path to project directory (required)
- `--legacy-agent`: Use legacy rule-based agent instead of LLM-first
- `--provider`: LLM provider (default: openai)
- `--model`: Model name
- `--api-key`: API key

### Programmatic Usage

```python
from automation.workflows.designspec_workflow import DesignSpecWorkflow
from automation.llm_client import LLMClient, LLMConfig

# Create LLM client
config = LLMConfig(provider="openai")
llm_client = LLMClient(config)

# Create workflow with LLM agent (default)
workflow = DesignSpecWorkflow(
    llm_client=llm_client,
    use_legacy_agent=False,  # Use LLM-first agent
)

# Start workflow
workflow.on_start(project_dir="/path/to/project")

# Process user messages
workflow.on_user_message("Create a box domain 20x60x30mm")

# Approve patches
workflow.approve_patch(patch_id)

# Run pipeline
workflow.run_until("component_mesh")
```

## Extending

### Adding New Pipeline Stages

Update `PIPELINE_STAGES` in `directive.py`:

```python
PIPELINE_STAGES = [
    "compile_policies",
    "compile_domains",
    # ... existing stages ...
    "new_stage",  # Add new stage
]
```

### Customizing the System Prompt

Modify `SYSTEM_PROMPT` in `prompt_builder.py` to adjust agent behavior, add new constraints, or update failure pattern detection.

### Adding Context Sources

Extend `ContextBuilder` to pull from additional sources:

```python
class ContextBuilder:
    def build_custom_summary(self) -> dict:
        # Pull from new source
        pass
    
    def build_compact(self) -> ContextPack:
        # Include custom summary in pack
        custom = self.build_custom_summary()
        # ...
```

## Testing

Run tests with pytest:

```bash
pytest tests/automation/test_designspec_llm_directive.py -v
pytest tests/automation/test_designspec_llm_context_builder.py -v
pytest tests/automation/test_designspec_llm_workflow_integration.py -v
```

Test coverage includes:
- Directive parsing (valid/invalid JSON, patch validation)
- Context building (spec summaries, run summaries, artifact selection)
- Workflow integration (user request → patch → approval → spec update)

## Design Principles

1. **LLM-first**: The LLM is the primary interpreter; regex/heuristics only as fallback
2. **Strict validation**: All LLM outputs validated before use
3. **Minimal patches**: Prefer smallest change that addresses the issue
4. **Approval gates**: User confirmation before patches and runs
5. **Traceability**: All decisions logged to `agent_turns.jsonl`
6. **No hardcoding**: No malaria-specific or filename-specific exceptions
7. **Clean separation**: Context building ≠ prompts ≠ parsing ≠ execution
