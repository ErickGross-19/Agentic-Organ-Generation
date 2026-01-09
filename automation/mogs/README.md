# MOGS - MultiAgentOrgan Generation System

A comprehensive system for LLM-driven automated design and validation of 3D vascular organ structures using a three-agent workflow.

## Overview

MOGS introduces a structured approach to organ generation with:

- **Object Identity Model**: UUID (immutable) + human-readable name (editable)
- **Three-Agent Workflow**: CSA, CBA, and VQA agents with clear responsibilities
- **Approval Gates**: User approval at each stage before proceeding
- **Three-Script Contract**: Standardized generate, analyze, finalize scripts
- **Version Retention**: Keep last 5 versions with safe garbage collection
- **Safety Constraints**: Restricted writes, no network access, no subprocess spawning

## Architecture

### Folder Structure

Every object follows this canonical layout:

```
objects/<object_uuid>/
  project/
    00_admin/           # Administrative files
      object_manifest.json
      run_index.json
      README.md
      retention_log.md
    01_specs/           # Versioned specifications
      spec_v###.json
      spec_v###_summary.md
      spec_v###_risk_flags.json
      spec_changelog.md
    02_agent_docs/      # Agent session documents
      CSA/
      CBA/
      VQA/
    03_prompts/         # Prompt packs
    04_scripts/         # Generated scripts
      scripts_v###/
        01_generate.py
        02_analyze.py
        03_finalize.py
        expected_artifacts.json
        run_manifest.json
    05_runs/            # Execution logs
    06_outputs/         # Generated outputs
      v###/
        generation/
        analysis/
        final/
    07_validation/      # Validation reports
    99_logs/            # System logs
```

### Three-Agent Workflow

1. **CSA (Concept & Spec Agent)**: Converts user intent into versioned specifications
   - Creates spec_v###.json (canonical)
   - Generates summary and risk flags
   - Only agent that can increment versions

2. **CBA (Coding & Build Agent)**: Generates scripts from specifications
   - Creates three scripts: generate, analyze, finalize
   - Defines expected artifacts
   - Cannot modify spec (proposes changes back to CSA)

3. **VQA (Validation & QA Agent)**: Validates outputs and proposes refinements
   - Validates scripts before execution
   - Validates outputs after execution
   - Proposes refinements back to CSA

### Approval Gates

- **Gate A (Spec Approval)**: User reviews spec summary and risk flags
- **Gate B (Code + Plan Approval)**: User reviews build plan and all scripts
- **Gate C (Results Approval)**: User reviews validation report and confirms outputs

## Usage

### Basic Usage

```python
from automation.mogs import create_mogs_runner

# Create a runner
runner = create_mogs_runner(objects_base_dir="./objects")

# Create a new object
folder_manager = runner.create_object("My Liver Structure")

# Define requirements
requirements = {
    "domain": {
        "type": "box",
        "size_m": [0.02, 0.06, 0.03],
    },
    "topology": {
        "kind": "tree",
        "target_terminals": 100,
    },
    "ports": {
        "inlets": [{"position_m": [0, 0, 0.015], "radius_m": 0.001}],
    },
}

# Run the workflow
result = runner.run_workflow(
    object_uuid=folder_manager.object_uuid,
    requirements=requirements,
    user_description="Generate a liver vascular network",
)

if result.success:
    print(f"Success! Outputs at: {result.final_outputs}")
else:
    print(f"Failed: {result.message}")
```

### Auto-Approve Mode (for testing)

```python
runner = create_mogs_runner(
    objects_base_dir="./objects",
    auto_approve=True,  # Automatically approve all gates
)
```

### Custom Approval Callback

```python
from automation.mogs import GateContext, GateResult, ApprovalChoice

def my_approval_callback(context: GateContext) -> GateResult:
    # Review the context
    print(f"Gate: {context.gate_type.value}")
    print(f"Files to review: {context.files_to_review}")
    
    # Make decision
    return GateResult(
        gate_type=context.gate_type,
        spec_version=context.spec_version,
        choice=ApprovalChoice.APPROVE,
        comments="Looks good!",
    )

runner = create_mogs_runner(
    objects_base_dir="./objects",
    approval_callback=my_approval_callback,
)
```

### Working with Existing Objects

```python
# List all objects
objects = runner.list_objects()
for obj in objects:
    print(f"{obj['object_name']} ({obj['object_uuid']})")

# Load an existing object
folder_manager = runner.load_object("uuid-here")
manifest = folder_manager.load_manifest()
print(f"Active spec version: {manifest.active_spec_version}")
```

### Refinement Workflow

```python
# Run a refinement based on previous version
result = runner.run_refinement(
    object_uuid="uuid-here",
    refinement_notes="Increase terminal count to 200",
    parent_version=1,
)
```

## Components

### Models (`models.py`)

Core data structures:
- `ObjectManifest`: Object metadata and configuration
- `SpecVersion`: Versioned specification
- `RiskFlag`: Risk identified during spec creation
- `ValidationReport`: Validation results
- `RunIndex`: Index of all runs

### Folder Manager (`folder_manager.py`)

Manages the canonical folder structure:
- Creates and validates folder structure
- Provides path helpers for all components
- Enforces write restrictions

### Object Registry (`object_registry.py`)

Manages the registry of all objects:
- Create, list, search objects
- Track object metadata
- Refresh registry from disk

### Agents (`agents/`)

- `ConceptSpecAgent`: Spec creation and versioning
- `CodingBuildAgent`: Script generation
- `ValidationQAAgent`: Output validation

### Gates (`gates.py`)

Approval gate implementations:
- `SpecApprovalGate`: Gate A
- `CodeApprovalGate`: Gate B
- `ResultsApprovalGate`: Gate C
- `GateManager`: Orchestrates all gates

### Retention (`retention.py`)

Version retention management:
- Keep last N versions
- Pin/unpin versions
- Safe garbage collection
- Audit trail

### Safety (`safety.py`)

Safety constraints for script execution:
- Write path restrictions
- Environment sanitization
- Resource limits
- Script validation

### Runner (`runner.py`)

Main workflow orchestration:
- `MOGSRunner`: Complete workflow runner
- `create_mogs_runner`: Factory function

## Safety Constraints

Scripts executed by MOGS are subject to:

1. **Write Restrictions**: Can only write inside the object's project folder
2. **No Network Access**: Network modules are blocked
3. **No Subprocess Spawning**: subprocess module is blocked
4. **Resource Limits**: Memory, CPU time, file size limits
5. **Environment Sanitization**: Only allowlisted environment variables

## Version Retention

By default, MOGS keeps the last 5 spec versions. Older versions are deleted as a unit, including:
- Spec files
- Scripts
- Outputs
- Validation reports
- Run logs

Protected versions:
- Pinned versions (user can pin milestones)
- Most recent accepted version

All deletions are logged in `retention_log.md`.

## Integration with Existing Automation

MOGS integrates with the existing automation infrastructure:
- Uses `LLMClient` for agent interactions
- Follows patterns from `script_writer.py` and `subprocess_runner.py`
- Compatible with existing `AgentRunner` workflows
