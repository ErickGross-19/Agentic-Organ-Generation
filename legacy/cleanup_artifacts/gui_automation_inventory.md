# GUI/Automation Inventory for DesignSpec-First Revamp

## Overview

This document summarizes the current architecture of `gui/` and `automation/` directories, identifying files to reuse, bypass, or deprecate for the DesignSpec-first revamp.

## Current Architecture Summary

### GUI Layer (`gui/`)

#### Entry Points

**`gui/main_window.py`** (1132 lines)
- `WorkflowSelectionDialog`: Dialog for selecting workflow type (currently only SINGLE_AGENT)
- `MainWindow`: Main application window with three-panel layout:
  - Chat panel (left): User input and conversation history
  - Output panel (center): Workflow output and status
  - STL viewer (right): 3D visualization of generated meshes
- Key methods:
  - `_setup_menu()`: File/Edit/Help menus
  - `_setup_toolbar()`: Quick action buttons
  - `_start_workflow()`: Initializes and starts selected workflow
  - `_send_input()`: Sends user messages to workflow

**`gui/workflow_manager.py`** (739 lines)
- `WorkflowType` enum: Currently only `SINGLE_AGENT`
- `WorkflowStatus` enum: IDLE, INITIALIZING, RUNNING, WAITING_INPUT, PAUSED, COMPLETED, FAILED, CANCELLED
- `WorkflowConfig` dataclass: Configuration for workflow execution
- `WorkflowManager` class: Orchestrates workflow execution in background threads
  - Uses IO adapters (GUIIOAdapter) for V5 workflow communication
  - Manages message passing between workflow and GUI

#### Supporting Files

- `gui/stl_viewer.py`: 3D visualization with matplotlib
- `gui/secure_config.py`: Encrypted API key storage
- `gui/chat_panel.py`: Chat interface components
- `gui/output_panel.py`: Output display components

### Automation Layer (`automation/`)

#### Core Workflow Files

**`automation/workflow.py`** (6050 lines)
- Contains V4 and V5 workflow implementations
- `ObjectRequirements`: 9-section schema for capturing user requirements
- `WorkflowState` enum: States for V4 state machine
- Imports `SingleAgentOrganGeneratorV5` from v5 subpackage
- Contains data classes for requirements sections (Identity, Domain, Topology, etc.)

**`automation/single_agent_organ_generation/v5/controller.py`** (3477 lines)
- `SingleAgentOrganGeneratorV5`: Main V5 controller class
- `ControllerConfig`: Configuration for V5 controller
- `ControllerStatus`: Derived status from world model
- 40+ capability methods (`_cap_*`)
- Goal-driven architecture with WorldModel as single source of truth

**`automation/single_agent_organ_generation/v5/world_model.py`**
- `WorldModel`: Single source of truth for all workflow state
- Tracks facts with provenance (USER, INFERRED, DEFAULT, SAFE_FIX, SYSTEM)
- Supports undo/backtracking

**`automation/single_agent_organ_generation/v5/goals.py`**
- `GoalTracker`: Manages 9 sequential goals
- Goal satisfaction replaces state machine transitions

**`automation/single_agent_organ_generation/v5/policies.py`**
- `SafeFixPolicy`: Classifies fixes as SAFE, NEEDS_CONFIRMATION, UNSAFE
- `ApprovalPolicy`: Determines when user approval is required
- `CapabilitySelectionPolicy`: Selects best capability from available set

**`automation/single_agent_organ_generation/v5/io/`**
- `BaseIOAdapter`: Abstract interface for I/O operations
- `CLIIOAdapter`: CLI implementation
- `GUIIOAdapter`: GUI implementation with callbacks

#### Schema and Validation

**`automation/schema_patch_validator.py`** (464 lines)
- `validate_module_patch()`: Validates module-level patches
- `validate_field_patch()`: Validates field-level patches
- `validate_field_value()`: Validates field values against schema
- `validate_schema_coherence()`: Validates overall schema coherence
- `validate_schema_patch()`: Main entry point for patch validation
- `suggest_patch_fixes()`: Suggests fixes for invalid patches

**`automation/schema_manager.py`** (774 lines)
- `FieldPatch`, `ModulePatch`, `SchemaPatch` dataclasses
- `Question` dataclass: Represents a question to ask the user
- `ActiveSchema`: Maintains current schema state
- `SchemaManager`: Manages schema lifecycle and patch application

#### Code Generation (TO BYPASS in DesignSpec workflow)

**`automation/script_writer.py`** (389 lines)
- `ScriptWriteResult` dataclass
- `extract_code_block()`: Extracts Python code from LLM responses
- `write_script()`: Writes extracted code to disk
- **NOTE: This module should NOT be used in DesignSpec workflow**

**`automation/review_gate.py`** (379 lines)
- `ReviewAction` enum: RUN, DONE, CANCEL, EDIT, VIEW
- `ReviewResult` dataclass
- `interactive_review()`: Human checkpoint for reviewing LLM-generated scripts
- **NOTE: May need adaptation for patch review instead of script review**

#### LLM Integration

**`automation/llm_client.py`**
- `LLMClient`: Unified interface for 8 LLM providers
- Retry logic with exponential backoff
- Conversation history management

**`automation/agent_runner.py`**
- `AgentRunner`: Orchestrates LLM task execution
- Code generation and sandboxed execution

**`automation/agent_dialogue.py`**
- `understand_object()`: Natural language understanding
- `propose_plans()`: Plan generation
- `format_user_summary()`: User-facing summaries

### DesignSpec Layer (`designspec/`)

**`designspec/spec.py`** (552 lines)
- `DesignSpec` class: Loads, validates, and normalizes specs
- Unit conversion (mm/um to meters)
- Schema validation
- Content hashing for reproducibility

**`designspec/runner.py`** (1447 lines)
- `DesignSpecRunner`: Core pipeline executor
- `StageReport`, `RunnerResult` dataclasses
- Pipeline stages:
  1. `compile_policies` - Compile policy dicts to aog_policies objects
  2. `compile_domains` - Compile domain dicts to runtime Domain objects
  3. `component_ports:<id>` - Resolve port positions
  4. `component_build:<id>` - Generate network/mesh
  5. `component_mesh:<id>` - Convert network to void mesh
  6. `union_voids` - Union all component void meshes
  7. `mesh_domain` - Generate domain mesh
  8. `embed` - Embed unified void into domain
  9. `port_recarve` - Recarve ports if enabled
  10. `validity` - Run validity checks
  11. `export` - Export outputs to files

**`designspec/plan.py`**
- `ExecutionPlan`: Controls partial execution
- `Stage` enum: All pipeline stages
- `STAGE_ORDER`: Ordered list of stages

**`designspec/context.py`**
- `RunnerContext`: Caching context for intermediate results
- `ArtifactStore`: Store for named artifacts

### Policy Layer (`aog_policies/`)

- `ResolutionPolicy`, `GrowthPolicy`, `CollisionPolicy`
- `EmbeddingPolicy`, `TissueSamplingPolicy`, `ValidationPolicy`
- `OpenPortPolicy`, `RepairPolicy`, `ComposePolicy`
- `PathfindingPolicy`, `PortPlacementPolicy`, `ChannelPolicy`
- `OutputPolicy`, `DomainMeshingPolicy`
- `OperationReport`: Base class for operation results

## Files to Reuse

### From `designspec/`
- **`designspec/spec.py`**: DesignSpec loading, validation, normalization
- **`designspec/runner.py`**: DesignSpecRunner for pipeline execution
- **`designspec/plan.py`**: ExecutionPlan for partial execution control
- **`designspec/context.py`**: RunnerContext and ArtifactStore

### From `aog_policies/`
- All policy classes for compile_policies stage
- `OperationReport` pattern for JSON-serializable results

### From `automation/`
- **`automation/schema_patch_validator.py`**: Patch validation logic (adapt for JSON Patch)
- **`automation/schema_manager.py`**: Schema management patterns (adapt for DesignSpec)
- **`automation/llm_client.py`**: LLM integration
- **`automation/single_agent_organ_generation/v5/io/`**: IO adapter patterns

### From `gui/`
- **`gui/stl_viewer.py`**: 3D visualization
- **`gui/secure_config.py`**: API key storage
- **`gui/workflow_manager.py`**: Background thread execution patterns

## Files to Bypass/Deprecate (Not Delete)

### Code Generation Modules (Bypass in DesignSpec workflow)
- **`automation/script_writer.py`**: Not used in DesignSpec workflow (no code generation)
- **`automation/review_gate.py`**: Replace with patch review mechanism

### V4 Workflow Components (Deprecate)
- V4 state machine code in `automation/workflow.py`
- Legacy workflow implementations in `automation/single_agent_organ_generation/_legacy/`

### Code Execution Modules (Bypass in DesignSpec workflow)
- **`automation/subprocess_runner.py`**: Not needed for DesignSpec workflow
- **`automation/single_agent_organ_generation/v5/workspace.py`**: LLM code sandbox (not needed)
- **`automation/single_agent_organ_generation/v5/brain.py`**: LLM decision engine for code gen

## New Modules to Create

### Phase 1: `automation/designspec_session.py`
- `DesignSpecSession` class
- Project directory management
- Spec persistence and history
- Patch application with auto-compile
- Runner execution wrapper

### Phase 2: `automation/designspec_agent.py`
- `DesignSpecAgent` class
- Question generation
- Patch proposal (JSON Patch format)
- Run request handling
- No code generation

### Phase 3: `automation/workflows/designspec_workflow.py`
- `DesignSpecWorkflow` class
- Integration with session and agent
- GUI workflow interface compatibility

### Phase 4: GUI Updates
- Project management (New/Open Project)
- Spec panel (JSON view + validation status)
- Patch panel (diff view + approve/reject)
- Compile status panel
- Run panel (stage selection)
- Artifacts panel

## Key Patterns to Follow

### OperationReport Pattern
All operations must emit JSON-serializable results with:
- `success: bool`
- `warnings: List[str]`
- `errors: List[str]`
- `metadata: Dict[str, Any]`

### JSON Patch (RFC 6902)
All patches must use standard JSON Patch format:
```json
[
  {"op": "add", "path": "/domains/main_domain", "value": {...}},
  {"op": "replace", "path": "/meta/seed", "value": 42},
  {"op": "remove", "path": "/components/0/policy_overrides/growth"}
]
```

### Auto-Compile After Patch
After every successful patch application:
1. Run `compile_policies` stage
2. Run `compile_domains` stage
3. Store results in `reports/compile_report.json`
4. Surface warnings/errors to user

### Project Directory Structure
```
project_dir/
  spec.json              # Current spec
  spec_history/          # Timestamped spec snapshots
  patches/               # Applied patches with metadata
  reports/               # Compile and run reports
  artifacts/             # Generated meshes and outputs
  logs/                  # Session logs
```

## Migration Path

1. Create new modules without modifying existing code
2. Add `DESIGNSPEC_PROJECT` workflow type to `WorkflowType` enum
3. Add project management UI to GUI
4. Integrate new workflow with existing `WorkflowManager`
5. Existing V5 workflow remains functional for backward compatibility
