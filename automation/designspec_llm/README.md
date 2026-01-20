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

## Run Approval Gating

The workflow implements explicit approval gating to prevent unintended pipeline executions. When the LLM requests a run with `requires_approval=true`, the workflow:

1. Stores the pending run request in `_pending_run_request`
2. Transitions to `WAITING_RUN_APPROVAL` state
3. Emits a `RUN_APPROVAL_REQUIRED` event with run details
4. Waits for user action before proceeding

The GUI displays "Approve Run" and "Reject Run" buttons when in this state. Users can:
- **Approve**: Calls `approve_pending_run()` which executes the stored run request
- **Reject**: Calls `reject_pending_run()` which clears the pending request and returns to IDLE

Workflow states:
- `IDLE`: Ready for user input
- `PROCESSING`: Agent is processing a message
- `WAITING_PATCH_APPROVAL`: Patches proposed, awaiting user approval
- `WAITING_RUN_APPROVAL`: Run requested, awaiting user approval
- `RUNNING`: Pipeline is executing
- `ERROR`: An error occurred

## Context Request Fulfillment

When the LLM needs more information to make a decision, it can request additional context via `context_requests`. The agent automatically fulfills these requests with a second LLM call:

1. First LLM call with compact context
2. If directive contains `context_requests` (e.g., `need_full_spec=true`), agent builds expanded context
3. Second LLM call with expanded context pack
4. Only the second directive is returned to the workflow

This "one internal hop" approach allows the LLM to request more information without user intervention, while preventing infinite loops by limiting to a single expansion.

Supported context requests:
- `need_full_spec`: Include complete spec JSON instead of summary
- `need_validity_report`: Include detailed validity check results
- `need_last_run_report`: Include full last run report
- `need_network_artifact`: Include network graph data
- `need_specific_files`: Request specific artifact files
- `need_more_history`: Include extended run history

## Compact Context Auto-Escalation

The context builder automatically escalates to "debug compact" mode when issues are detected:

- If last run failed OR validity has failed checks:
  - Includes detailed validation summary with failure names and reasons
  - Includes mesh statistics (component void, union void, domain with void)
  - Includes network statistics (node count, edge count, bbox, radius stats)

This provides the LLM with sufficient debugging information without requiring explicit context requests.

## GUI Panels

The DesignSpec workflow integrates with the GUI through a tabbed notebook interface:

### Conversation / Log Tab
Displays the conversation history and agent responses.

### Spec Tab (SpecPanel)
Shows the current spec summary and allows viewing/editing the full spec JSON.

### Patches Tab (PatchPanel)
Displays proposed patches with "Approve" and "Reject" buttons for each patch.

### Run Tab (RunPanel)
- Shows current workflow state and pending run details
- "Approve Run" button (enabled only in WAITING_RUN_APPROVAL state)
- "Reject Run" button (enabled only in WAITING_RUN_APPROVAL state)
- Stage selector for manual run requests
- Displays run_until, reason, and expected_signal for pending runs

### Artifacts Tab (ArtifactsPanel)
Lists generated artifacts with ability to open STL files in the viewer.

### Reports Tab (CompilePanel)
Shows compilation and validity reports.

### Panel Callbacks

Panels receive updates from `DesignSpecWorkflowManager` via callbacks:
- `on_spec_update`: Spec content changed
- `on_pending_patches`: New patches proposed
- `on_pending_run_request`: Run approval requested
- `on_state_change`: Workflow state transition
- `on_artifact_update`: New artifacts generated

## Design Principles

1. **LLM-first**: The LLM is the primary interpreter; regex/heuristics only as fallback
2. **Strict validation**: All LLM outputs validated before use
3. **Minimal patches**: Prefer smallest change that addresses the issue
4. **Approval gates**: User confirmation before patches and runs
5. **Traceability**: All decisions logged to `agent_turns.jsonl`
6. **No hardcoding**: No malaria-specific or filename-specific exceptions
7. **Clean separation**: Context building ≠ prompts ≠ parsing ≠ execution
8. **No silent runs**: Any patch or run must be explicitly approved by user action
