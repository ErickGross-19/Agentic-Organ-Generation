# Used Modules

This document lists all modules that are actively used by the canonical DesignSpec + DesignSpecRunner architecture and the readiness gate tests.

## Methodology

Modules were identified by tracing all imports (including dynamic imports inside functions) from:
1. Canonical entrypoints: `designspec/runner.py`, `designspec/spec.py`, `generation/api/generate.py`, `generation/api/embed.py`, `validity/runner.py`
2. Readiness gate tests: `tests/contract/`, `tests/unit/`, `tests/integration/`, `tests/regression/`, `tests/quality/`

## Used Modules (60 total)

### aog_policies/ (9 files)
All policy modules are canonical and actively used:
- `aog_policies/__init__.py` - Main policy exports
- `aog_policies/base.py` - Base policy classes
- `aog_policies/collision.py` - Collision policy
- `aog_policies/composition.py` - Composition policy
- `aog_policies/features.py` - Feature policies
- `aog_policies/generation.py` - Generation policies
- `aog_policies/pathfinding.py` - Pathfinding policies
- `aog_policies/resolution.py` - Resolution policy
- `aog_policies/validity.py` - Validity policies

### designspec/ (10 files)
All designspec modules are canonical:
- `designspec/ast/compile.py` - AST compilation
- `designspec/ast/nodes.py` - AST node definitions
- `designspec/compat/__init__.py` - Compatibility layer
- `designspec/compat/v1_aliases.py` - V1 alias mappings
- `designspec/context.py` - Runner context
- `designspec/plan.py` - Execution plan
- `designspec/reports/run_report.py` - Run report
- `designspec/reports/serializers.py` - Report serializers
- `designspec/runner.py` - Main runner
- `designspec/schema.py` - Schema validation
- `designspec/spec.py` - Spec loader

### generation/ (26 files)
Core generation modules used by the runner:
- `generation/__init__.py` - Package init
- `generation/adapters/mesh_adapter.py` - Mesh adapters
- `generation/analysis/perfusion.py` - Perfusion analysis (used by tests)
- `generation/analysis/structure.py` - Structure analysis (used by tests)
- `generation/api/__init__.py` - API exports
- `generation/api/embed.py` - Embedding API
- `generation/api/evaluate.py` - Evaluation API
- `generation/api/export.py` - Export API
- `generation/api/generate.py` - Generation API
- `generation/backends/cco_hybrid_backend.py` - CCO backend
- `generation/backends/kary_tree_backend.py` - K-ary tree backend
- `generation/backends/programmatic_backend.py` - Programmatic backend
- `generation/backends/space_colonization_backend.py` - Space colonization backend
- `generation/core/domain.py` - Domain definitions
- `generation/core/domain_composite.py` - Composite domains
- `generation/core/domain_implicit.py` - Implicit domains
- `generation/core/domain_primitives.py` - Domain primitives
- `generation/core/network.py` - Vascular network
- `generation/core/types.py` - Core types
- `generation/ops/collision/__init__.py` - Collision ops
- `generation/ops/compose.py` - Composition ops
- `generation/ops/embedding/__init__.py` - Embedding ops
- `generation/ops/embedding/enhanced_embedding.py` - Enhanced embedding
- `generation/ops/growth.py` - Growth ops
- `generation/ops/mesh/merge.py` - Mesh merge
- `generation/ops/mesh/synthesis.py` - Mesh synthesis
- `generation/policies.py` - Policy re-exports
- `generation/rules/constraints.py` - Constraint rules
- `generation/spatial/grid_index.py` - Spatial indexing
- `generation/utils/geometry.py` - Geometry utilities
- `generation/utils/resolution_resolver.py` - Resolution resolver
- `generation/utils/units.py` - Unit conversion

### validity/ (10 files)
Core validity modules used by the runner:
- `validity/__init__.py` - Package init
- `validity/api/__init__.py` - API exports
- `validity/api/repair.py` - Repair API
- `validity/checks/open_ports.py` - Open port checks
- `validity/models.py` - Validity models
- `validity/pipeline.py` - Validity pipeline (used by tests)
- `validity/pre_embedding/graph_checks.py` - Graph checks
- `validity/runner.py` - Validity runner

## Notes

1. Some modules are used only by tests but are still considered "used" because they test functionality that should be preserved.
2. `__init__.py` files are generally kept in place even if they appear unused, as they define package structure.
3. Modules that re-export from other modules (like `generation/policies.py`) are kept because they provide backward-compatible import paths.
