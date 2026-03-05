# Legacy Move Log

This document tracks all files moved to the `/legacy/` directory as part of the repository cleanup.

## Summary

| Category | Files Moved | Files Kept in Place |
|----------|-------------|---------------------|
| CFD modules | 11 | 0 |
| **Total** | **11** | See "Files Kept in Place" section |

## Files Moved

### generation/cfd/ (11 files)

The entire CFD module was moved to legacy because it is not used by the DesignSpecRunner or any readiness gate tests.

| Old Path | New Path | Shim Created | Reason |
|----------|----------|--------------|--------|
| `generation/cfd/__init__.py` | `legacy/generation/cfd/__init__.py` | No | CFD module not used by runner, not imported by any code |
| `generation/cfd/bcs.py` | `legacy/generation/cfd/bcs.py` | No | CFD boundary conditions, not imported |
| `generation/cfd/geometry.py` | `legacy/generation/cfd/geometry.py` | No | CFD geometry, not imported |
| `generation/cfd/meshing.py` | `legacy/generation/cfd/meshing.py` | No | CFD meshing, not imported |
| `generation/cfd/pipeline.py` | `legacy/generation/cfd/pipeline.py` | No | CFD pipeline, not imported |
| `generation/cfd/results.py` | `legacy/generation/cfd/results.py` | No | CFD results, not imported |
| `generation/cfd/solvers/__init__.py` | `legacy/generation/cfd/solvers/__init__.py` | No | CFD solvers package, not imported |
| `generation/cfd/solvers/base.py` | `legacy/generation/cfd/solvers/base.py` | No | Base CFD solver, not imported |
| `generation/cfd/solvers/sv_0d.py` | `legacy/generation/cfd/solvers/sv_0d.py` | No | 0D solver, not imported |
| `generation/cfd/solvers/sv_1d.py` | `legacy/generation/cfd/solvers/sv_1d.py` | No | 1D solver, not imported |
| `generation/cfd/solvers/sv_3d.py` | `legacy/generation/cfd/solvers/sv_3d.py` | No | 3D solver, not imported |

## Files Kept in Place

The following files were initially considered for migration but were kept in place because they are still actively imported by other code in the codebase:

### generation/api/design.py

| File | Reason Kept |
|------|-------------|
| `generation/api/design.py` | Imported by `generation/api/__init__.py` which re-exports `design_from_spec` |

### generation/ops/*_legacy.py (4 files)

| File | Reason Kept |
|------|-------------|
| `generation/ops/collision_legacy.py` | Imported by `generation/ops/collision/__init__.py` |
| `generation/ops/embedding_legacy.py` | Imported by `generation/ops/embedding/__init__.py` |
| `generation/ops/features_legacy.py` | Imported by `generation/ops/features/__init__.py` |
| `generation/ops/pathfinding_legacy.py` | Imported by `generation/ops/pathfinding/__init__.py` |

**Note**: These files have "legacy" in their names but are still actively used by the codebase through the `__init__.py` re-exports. They provide backward compatibility for the legacy API while the new unified API is being adopted.

### validity/orchestrators.py

| File | Reason Kept |
|------|-------------|
| `validity/orchestrators.py` | Imported by `validity/__init__.py` |

### validity/analysis/ (3 files)

| File | Reason Kept |
|------|-------------|
| `validity/analysis/centerline.py` | Imported by `validity/analysis/__init__.py` |
| `validity/analysis/connectivity.py` | Imported by `validity/analysis/__init__.py` |
| `validity/analysis/node_metrics.py` | Imported by `validity/analysis/__init__.py` |

### validity/post_embedding/ (2 files)

| File | Reason Kept |
|------|-------------|
| `validity/post_embedding/connectivity_checks.py` | Imported by `validity/post_embedding/__init__.py` |
| `validity/post_embedding/domain_checks.py` | Imported by `validity/post_embedding/__init__.py` |

### validity/pre_embedding/ (2 files)

| File | Reason Kept |
|------|-------------|
| `validity/pre_embedding/flow_checks.py` | Imported by `validity/pre_embedding/__init__.py` |
| `validity/pre_embedding/mesh_checks.py` | Imported by `validity/pre_embedding/__init__.py` |

### generation/analysis/ (5 files)

| File | Reason Kept |
|------|-------------|
| `generation/analysis/coverage.py` | Imported by `generation/analysis/__init__.py` |
| `generation/analysis/distance.py` | Imported by `generation/analysis/__init__.py` |
| `generation/analysis/flow.py` | Imported by `generation/analysis/__init__.py` |
| `generation/analysis/query.py` | Imported by `generation/analysis/__init__.py` |
| `generation/analysis/solver.py` | Imported by `generation/analysis/__init__.py` |

### generation/specs/ (kept with deprecation warning)

| File | Reason Kept |
|------|-------------|
| `generation/specs/__init__.py` | Imported by automation/workflow.py, automation/spec_compiler.py, automation/streamlined_workflow.py, examples/malaria_venule_*.py |
| `generation/specs/design_spec.py` | Imported by automation/ and examples/ |
| `generation/specs/compile.py` | Imported by automation/ and examples/ |
| `generation/specs/eval_result.py` | Part of specs package |

**Action taken**: Added deprecation warning to `generation/specs/__init__.py` that warns users to use JSON DesignSpec + DesignSpecRunner instead.

### generation/optimization/ (kept - no changes)

| File | Reason Kept |
|------|-------------|
| `generation/optimization/*` | Imported by examples/malaria_venule_*.py which cannot be modified |

### generation/organ_generators/liver/ (kept - no changes)

| File | Reason Kept |
|------|-------------|
| `generation/organ_generators/liver/*` | Imported by automation/task_templates/generate_structure.py and generation/adapters/liver_adapter.py |

## Classification Rubric Applied

All moved files were classified as legacy based on the criteria defined in `LEGACY_CLASSIFICATION.md`:

1. **Not imported by any code** (verified via ripgrep search)
2. **Not used by DesignSpecRunner path** (verified via dependency graph)
3. **Not used by readiness gate tests** (verified via test import analysis)

Files that were imported by other code (even through `__init__.py` re-exports) were kept in place to avoid breaking the codebase.

## Verification

After the cleanup, the readiness gate tests were run to verify no new failures were introduced:

- **Baseline**: 5 failed, 1056 passed, 11 skipped
- **After cleanup**: 5 failed, 1056 passed, 11 skipped

The 5 failures are pre-existing issues in the codebase and were not introduced by this cleanup.
