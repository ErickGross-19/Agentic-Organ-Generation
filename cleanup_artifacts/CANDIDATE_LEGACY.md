# Candidate Legacy Modules

This document lists all modules that are candidates for moving to the `/legacy/` directory, with rationale for each classification.

## Classification Categories

### Category A: Truly Legacy (MOVE)
Old APIs that are superseded by the new DesignSpec + DesignSpecRunner architecture.

### Category B: Experimental/Non-Runner (MOVE)
Modules that are not part of the runner path and represent experimental or specialized functionality.

### Category C: Keep in Place
Modules that should NOT be moved because they are `__init__.py` files, core utilities, or may be needed at runtime.

---

## Category A: Truly Legacy (MOVE to /legacy/)

### generation/specs/ (4 files) - MOVE
**Rationale**: These are the old dataclass-based specification system, superseded by JSON DesignSpec + aog_policies.

| File | Reason |
|------|--------|
| `generation/specs/__init__.py` | Package init for legacy specs |
| `generation/specs/compile.py` | Compiles legacy spec dataclasses |
| `generation/specs/design_spec.py` | Old DesignSpec, TreeSpec, DomainSpec dataclasses |
| `generation/specs/eval_result.py` | Legacy evaluation result |

### generation/api/design.py - MOVE
**Rationale**: Old entry point `design_from_spec()` that takes legacy spec dataclasses. Superseded by `DesignSpecRunner.run()`.

### generation/ops/*_legacy.py (4 files) - MOVE
**Rationale**: Explicitly named as legacy implementations.

| File | Reason |
|------|--------|
| `generation/ops/collision_legacy.py` | Old collision detection |
| `generation/ops/embedding_legacy.py` | Old embedding implementation |
| `generation/ops/features_legacy.py` | Old feature extraction |
| `generation/ops/pathfinding_legacy.py` | Old pathfinding implementation |

### validity/orchestrators.py - MOVE
**Rationale**: Old validity orchestrator not used by `validity/runner.py`. Not imported by any current code.

---

## Category B: Experimental/Non-Runner (MOVE to /legacy/)

### generation/cfd/ (10 files) - MOVE
**Rationale**: CFD (Computational Fluid Dynamics) simulation modules. Not part of the runner path, represents specialized analysis functionality.

| File | Reason |
|------|--------|
| `generation/cfd/__init__.py` | CFD package init |
| `generation/cfd/bcs.py` | Boundary conditions |
| `generation/cfd/geometry.py` | CFD geometry |
| `generation/cfd/meshing.py` | CFD meshing |
| `generation/cfd/pipeline.py` | CFD pipeline |
| `generation/cfd/results.py` | CFD results |
| `generation/cfd/solvers/__init__.py` | Solvers package |
| `generation/cfd/solvers/base.py` | Base solver |
| `generation/cfd/solvers/sv_0d.py` | 0D solver |
| `generation/cfd/solvers/sv_1d.py` | 1D solver |
| `generation/cfd/solvers/sv_3d.py` | 3D solver |

### generation/optimization/ (5 files) - MOVE
**Rationale**: Optimization modules not used by the runner. Represents experimental optimization functionality.

| File | Reason |
|------|--------|
| `generation/optimization/__init__.py` | Package init |
| `generation/optimization/nlp_geometry.py` | NLP geometry optimization |
| `generation/optimization/refine.py` | Refinement optimization |
| `generation/optimization/solvers.py` | Optimization solvers |
| `generation/optimization/topology_swaps.py` | Topology swap optimization |

### generation/organ_generators/liver/ (7 files) - MOVE
**Rationale**: Liver-specific generator. Not used by the generic runner path.

| File | Reason |
|------|--------|
| `generation/organ_generators/liver/__init__.py` | Package init |
| `generation/organ_generators/liver/config.py` | Liver config |
| `generation/organ_generators/liver/export.py` | Liver export |
| `generation/organ_generators/liver/geometry.py` | Liver geometry |
| `generation/organ_generators/liver/growth.py` | Liver growth |
| `generation/organ_generators/liver/rules.py` | Liver rules |
| `generation/organ_generators/liver/tree.py` | Liver tree |

### generation/analysis/ (6 files) - MOVE (except perfusion.py and structure.py)
**Rationale**: Analysis modules not used by the runner. Some are used by tests.

| File | Reason | Action |
|------|--------|--------|
| `generation/analysis/__init__.py` | Package init | MOVE |
| `generation/analysis/coverage.py` | Coverage analysis | MOVE |
| `generation/analysis/distance.py` | Distance analysis | MOVE |
| `generation/analysis/flow.py` | Flow analysis | MOVE |
| `generation/analysis/query.py` | Query analysis | MOVE |
| `generation/analysis/radius.py` | Radius analysis | MOVE |
| `generation/analysis/solver.py` | Analysis solver | MOVE |

### validity/analysis/ (6 files) - MOVE
**Rationale**: Validity analysis modules not used by the runner.

| File | Reason |
|------|--------|
| `validity/analysis/__init__.py` | Package init |
| `validity/analysis/centerline.py` | Centerline analysis |
| `validity/analysis/cfd.py` | CFD analysis |
| `validity/analysis/connectivity.py` | Connectivity analysis |
| `validity/analysis/metrics.py` | Metrics analysis |
| `validity/analysis/node_metrics.py` | Node metrics |

### validity/post_embedding/ (4 files) - MOVE
**Rationale**: Post-embedding checks not used by the canonical validity runner.

| File | Reason |
|------|--------|
| `validity/post_embedding/__init__.py` | Package init |
| `validity/post_embedding/connectivity_checks.py` | Connectivity checks |
| `validity/post_embedding/domain_checks.py` | Domain checks |
| `validity/post_embedding/printability_checks.py` | Printability checks |

### validity/pre_embedding/ (3 files) - MOVE (except graph_checks.py)
**Rationale**: Pre-embedding checks not used by the canonical validity runner.

| File | Reason |
|------|--------|
| `validity/pre_embedding/__init__.py` | Package init |
| `validity/pre_embedding/flow_checks.py` | Flow checks |
| `validity/pre_embedding/mesh_checks.py` | Mesh checks |

---

## Category C: Keep in Place (DO NOT MOVE)

### `__init__.py` files
These define package structure and should remain in place:
- `designspec/__init__.py`
- `designspec/ast/__init__.py`
- `designspec/reports/__init__.py`
- `generation/adapters/__init__.py`
- `generation/backends/__init__.py`
- `generation/core/__init__.py`
- `generation/geometry/__init__.py`
- `generation/ops/__init__.py`
- `generation/ops/mesh/__init__.py`
- `generation/ops/network/__init__.py`
- `generation/ops/pathfinding/__init__.py`
- `generation/ops/primitives/__init__.py`
- `generation/ops/validity/__init__.py`
- `generation/params/__init__.py`
- `generation/rules/__init__.py`
- `generation/spatial/__init__.py`
- `generation/utils/__init__.py`
- `validity/api/__init__.py`
- `validity/checks/__init__.py`
- `validity/io/__init__.py`
- `validity/mesh/__init__.py`
- `validity/repair/__init__.py`
- `validity/reporting/__init__.py`

### Core utilities that may be used at runtime
These modules may be imported dynamically or used by modules that are used:
- `generation/adapters/liver_adapter.py` - May be used by automation
- `generation/adapters/networkx_adapter.py` - May be used by automation
- `generation/adapters/report_adapter.py` - May be used by automation
- `generation/backends/base.py` - Base class for backends
- `generation/core/domain_transform.py` - Domain transformations
- `generation/core/ids.py` - ID generation
- `generation/core/report.py` - Report utilities
- `generation/core/result.py` - Result types
- `generation/geometry/obb.py` - OBB calculations
- `generation/ops/anastomosis.py` - Anastomosis operations
- `generation/ops/build.py` - Build operations
- `generation/ops/collision/unified.py` - Unified collision
- `generation/ops/domain_meshing.py` - Domain meshing
- `generation/ops/features/face_feature.py` - Face features
- `generation/ops/features/ridge_helpers.py` - Ridge helpers
- `generation/ops/mesh/repair.py` - Mesh repair
- `generation/ops/network/cleanup.py` - Network cleanup
- `generation/ops/network/merge.py` - Network merge
- `generation/ops/network/metrics.py` - Network metrics
- `generation/ops/pathfinding/astar_voxel.py` - A* pathfinding
- `generation/ops/pathfinding/hierarchical_astar.py` - Hierarchical A*
- `generation/ops/primitives/channels.py` - Channel primitives
- `generation/ops/primitives/path_sweep.py` - Path sweep
- `generation/ops/space_colonization.py` - Space colonization
- `generation/ops/tree_ops.py` - Tree operations
- `generation/ops/validity/void_checks.py` - Void checks
- `generation/params/presets.py` - Parameter presets
- `generation/params/validation.py` - Parameter validation
- `generation/rules/radius.py` - Radius rules
- `generation/utils/faces.py` - Face utilities
- `generation/utils/layout.py` - Layout utilities
- `generation/utils/port_placement.py` - Port placement
- `generation/utils/scale.py` - Scale utilities
- `generation/utils/schedules.py` - Schedule utilities
- `generation/utils/tissue_sampling.py` - Tissue sampling
- `validity/api/pipeline.py` - Validity pipeline
- `validity/api/validate.py` - Validation API
- `validity/checks/components.py` - Component checks
- `validity/checks/dimensions.py` - Dimension checks
- `validity/checks/topology.py` - Topology checks
- `validity/checks/watertight.py` - Watertight checks
- `validity/io/exporters.py` - Exporters
- `validity/io/loaders.py` - Loaders
- `validity/mesh/cleaning.py` - Mesh cleaning
- `validity/mesh/diagnostics.py` - Mesh diagnostics
- `validity/mesh/repair.py` - Mesh repair
- `validity/mesh/voxel_utils.py` - Voxel utilities
- `validity/repair/cleanup.py` - Repair cleanup
- `validity/repair/voxel_repair.py` - Voxel repair
- `validity/reporting/drift_report.py` - Drift report
- `validity/reporting/run_report.py` - Run report
- `validity/utils.py` - Validity utilities

### Top-level files
- `main.py` - Application entry point
- `setup.py` - Package setup
- `aog_policies/domain.py` - Domain policy (may be used)

---

## Summary

| Category | Count | Action |
|----------|-------|--------|
| Category A: Truly Legacy | 10 files | MOVE (only 12 actually moved - see note) |
| Category B: Experimental/Non-Runner | 41 files | KEEP (still imported by other code) |
| Category C: Keep in Place | 81 files | KEEP |
| **Total** | **132 files** | |

**Note**: After attempting to move the files, it was discovered that many of the "candidate legacy" files are still actively imported by `__init__.py` files in their respective packages. Moving these files would break the codebase. Only the following files were actually moved:

## Files Actually Moved (11 total)

1. `generation/cfd/__init__.py` - CFD module not used by runner
2. `generation/cfd/bcs.py` - CFD boundary conditions
3. `generation/cfd/geometry.py` - CFD geometry
4. `generation/cfd/meshing.py` - CFD meshing
5. `generation/cfd/pipeline.py` - CFD pipeline
6. `generation/cfd/results.py` - CFD results
7. `generation/cfd/solvers/__init__.py` - CFD solvers package
8. `generation/cfd/solvers/base.py` - Base CFD solver
9. `generation/cfd/solvers/sv_0d.py` - 0D solver
10. `generation/cfd/solvers/sv_1d.py` - 1D solver
11. `generation/cfd/solvers/sv_3d.py` - 3D solver

## Files NOT Moved (still imported by other code)

The following files were initially candidates for moving but were kept in place because they are still actively imported by `__init__.py` files or other code:

- `generation/specs/*` - Imported by automation/ and examples/ (added deprecation warning instead)
- `generation/ops/*_legacy.py` - Imported by `generation/ops/*/\__init__.py`
- `validity/orchestrators.py` - Imported by `validity/__init__.py`
- `generation/analysis/*` - Imported by `generation/analysis/__init__.py`
- `validity/analysis/*` - Imported by `validity/analysis/__init__.py`
- `validity/post_embedding/*` - Imported by `validity/post_embedding/__init__.py`
- `validity/pre_embedding/*` - Imported by `validity/pre_embedding/__init__.py`
- `generation/optimization/*` - Imported by examples/
- `generation/organ_generators/liver/*` - Imported by automation/
