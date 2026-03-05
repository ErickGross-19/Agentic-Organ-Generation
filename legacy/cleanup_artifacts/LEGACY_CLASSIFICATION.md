# Legacy Classification Rubric

This document defines objective criteria for classifying code as "legacy" in the Agentic-Organ-Generation repository.

## Current Architecture

The canonical execution flow is:
1. **DesignSpec** (JSON) → loaded via `designspec/spec.py`
2. **DesignSpecRunner** (`designspec/runner.py`) → orchestrates the pipeline
3. **aog_policies/** → all behavior controlled through policy objects
4. **generation/api/generate.py** and **generation/api/embed.py** → core generation operations
5. **validity/runner.py** → canonical validity checks

## Classification Criteria

### Code IS Legacy If:

1. **Depends on deprecated dataclass APIs** (`generation/specs/*`)
   - Files that import from `generation.specs.design_spec` (EllipsoidSpec, TreeSpec, ColonizationSpec, BoxSpec, InletSpec, OutletSpec, DualTreeSpec, DesignSpec dataclass)
   - These are superseded by the JSON DesignSpec + aog_policies architecture

2. **Old entrypoints not used by DesignSpecRunner**
   - `generation/api/design.py::design_from_spec` - old entry point
   - Any function that takes legacy spec dataclasses as input

3. **Old validity orchestrators not called by validity/runner.py**
   - `validity/orchestrators.py` - not imported by anything in the canonical path
   - `validity/pipeline.py` - only used by test files

4. **Files explicitly named with "legacy" suffix**
   - `generation/ops/collision_legacy.py`
   - `generation/ops/embedding_legacy.py`
   - `generation/ops/features_legacy.py`
   - `generation/ops/pathfinding_legacy.py`

5. **Pipeline functions not reachable by DesignSpecRunner**
   - Functions that are not imported (directly or transitively) by:
     - `designspec/runner.py`
     - `generation/api/generate.py`
     - `generation/api/embed.py`
     - `validity/runner.py`

6. **Duplicate implementations where one is clearly superseded**
   - Old implementations kept for backward compatibility but not used by the runner

### Code is NOT Legacy If:

1. **Imported by DesignSpecRunner path**
   - Any module imported (directly or transitively) by `designspec/runner.py`

2. **Used by current readiness gate tests**
   - Modules imported by tests in `tests/contract/`, `tests/unit/`, `tests/integration/`, `tests/regression/`, `tests/quality/`

3. **Core utilities used by generation/validity pipelines**
   - `generation/core/*` - core data structures (network, domain, types)
   - `generation/backends/*` - generation backends
   - `generation/ops/*` (excluding `*_legacy.py` files)
   - `generation/utils/*` - utility functions
   - `validity/checks/*` - validity check implementations
   - `validity/api/*` - validity API functions

4. **aog_policies package**
   - All of `aog_policies/*` is canonical and must not be moved

5. **designspec package**
   - All of `designspec/*` is canonical and must not be moved

## No-Cheating Rule

1. **Code cannot be kept in place merely by labeling it "non-legacy"**
   - If code is not used by the current system and fits the legacy criteria, it must be moved

2. **If code cannot be moved safely because of external users**
   - Create a compatibility shim at the old location
   - Move the original implementation to `/legacy/`
   - The shim must be < 30 lines and purely delegating

3. **All classification decisions must be justified**
   - Each moved file must have documented rationale in LEGACY_MOVE_LOG.md

## Directories That Must NOT Be Modified

Per user instructions, the following directories are excluded from this cleanup:
- `gui/`
- `automation/`
- `examples/`

These directories will not be moved, renamed, or edited.
