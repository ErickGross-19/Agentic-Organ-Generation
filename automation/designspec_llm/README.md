# DesignSpec LLM Agent

LLM-driven agent loop for iteratively designing and running DesignSpec JSON workflows through conversation. Replaces the previous rule-based parsing system with a structured LLM-first approach.

## Architecture

The agent receives user messages along with context about the current spec and recent artifacts, then returns structured directives that the system validates and executes.

### Core Modules

| Module | Purpose |
|--------|---------|
| `agent.py` | `DesignSpecLLMAgent` - main agent loop: context → prompt → LLM → directive → response |
| `directive.py` | `DesignSpecDirective` dataclass - structured output contract for LLM responses |
| `context_builder.py` | `ContextBuilder` - assembles spec, run history, and artifacts into context packs |
| `prompt_builder.py` | `PromptBuilder` - system and user prompt construction with all 44 scaffold types |
| `artifact_indexer.py` | `ArtifactIndexer` - indexes run artifacts for context building |
| `error_taxonomy.py` | 30+ classified error patterns with confidence scores and fix strategies |
| `patch_generator.py` | Targeted RFC 6902 patch generation from error classifications |
| `run_analyzer.py` | Auto-analyze pipeline failures and propose fix patches |
| `task_context.py` | Persistent goal tracking across turns |
| `session_memory.py` | Decision and error resolution memory |
| `policy_reference.py` | Policy documentation injected into LLM context |

## Directive Schema

Every LLM output must conform to the `DesignSpecDirective` schema:

```json
{
  "assistant_message": "Message shown to user",
  "questions": [{"id": "q1", "question": "...", "why_needed": "...", "default": "..."}],
  "proposed_patches": [{"op": "add|replace|remove", "path": "/json/pointer", "value": "..."}],
  "run_request": {"run": true, "run_until": "stage_name", "reason": "...", "expected_signal": "..."},
  "context_requests": {"need_full_spec": false, "need_validity_report": false, "...": "..."},
  "confidence": 0.85,
  "requires_approval": true,
  "stop": false
}
```

## Integration Points

### Web UI (Primary)
The FastAPI `DesignSpecBridge` in `backend/app/api/designspec.py` wraps the workflow for HTTP access from the Next.js frontend. The chat panel sends messages to `/api/designspec/message` and receives directives with patches, questions, and run requests.

### Desktop GUI (Legacy)
The `DesignSpecWorkflowManager` in `gui/designspec_workflow_manager.py` wraps the workflow for the tkinter GUI with tabbed panels for spec, patches, runs, and artifacts.

## Workflow States

| State | Description |
|-------|-------------|
| `IDLE` | Ready for user input |
| `PROCESSING` | Agent processing a message |
| `WAITING_PATCH_APPROVAL` | Patches proposed, awaiting approval |
| `WAITING_RUN_APPROVAL` | Run requested, awaiting approval |
| `RUNNING` | Pipeline executing |
| `ERROR` | Error occurred |

## Auto-Analysis

When a pipeline run fails, the system automatically:
1. `ErrorParser` classifies the error against 30+ patterns in `error_taxonomy.py`
2. `PatchGenerator` creates targeted fix patches based on the classification
3. `RunAnalyzer` proposes the fix to the user for approval

## Context Request Fulfillment

When the LLM needs more information, it requests additional context via `context_requests`. The agent fulfills these with a second LLM call (one internal hop, no infinite loops):

1. First call with compact context
2. If directive contains `context_requests`, agent builds expanded context
3. Second call with expanded context
4. Only the second directive is returned

## Compact Context Auto-Escalation

The context builder automatically escalates to "debug compact" mode when issues are detected (failed run or failed validity checks), including detailed validation summaries, mesh statistics, and network statistics.

## Scaffold Types in Prompt

The system prompt includes all 44 scaffold types organized by category so the LLM agent can propose specs for any generator:
- Vascular (space colonization, bifurcating tree, top-down scaffold)
- Lattice/TPMS (gyroid, schwarz-p, octet truss, voronoi, honeycomb)
- Skeletal, Organ, Soft Tissue, Tubular, Dental, Microfluidic categories

## Design Principles

1. **LLM-first**: The LLM is the primary interpreter; regex/heuristics only as fallback
2. **Strict validation**: All LLM outputs validated before use
3. **Minimal patches**: Prefer smallest change that addresses the issue
4. **Approval gates**: User confirmation before patches and runs
5. **Traceability**: All decisions logged to `agent_turns.jsonl`
6. **Clean separation**: Context building, prompts, parsing, and execution are separate concerns
