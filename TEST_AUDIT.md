# Test Suite Audit Report

This document summarizes the test suite overhaul for the Agentic-Organ-Generation repository, aligning tests with the new DesignSpec + DesignSpecRunner architecture where all orchestration is policy-controlled via `aog_policies`.

## Executive Summary

The test suite has been audited and reorganized to ensure:
1. Everything is policy-controlled via `aog_policies`
2. Everything is JSON serializable and runner-orchestratable
3. Scale/budget behavior works (mm/um, 20um min diameter, budgets, pitch relaxation)
4. Hierarchical pathfinding is mandatory and correct
5. Multi-component voids are unioned then embedded once
6. Embedding preserves ports via voxel recarve
7. Validity orchestrates checks consistently including open ports
8. Reports and artifacts are JSON-clean and reproducible

## New Test Taxonomy

```
tests/
  contract/            # Policy ownership, JSON serializability, schema versioning, report schema
  unit/                # Domains, AST eval, resolution budgeting, port placement logic, tube sweep primitives
  integration/         # DesignSpecRunner end-to-end (golden spec), partial execution, multi-component union then embed once
  regression/          # Previously fixed bug regressions (stop_before_boundary applied once, no 1mm clamps, etc.)
  quality/             # Targeted code hygiene for runner-critical code only
  legacy_disabled/     # Quarantined tests, not run by default
  fixtures/            # Golden fixtures and test data
```

## File Categorization

### Files to KEEP (Move to New Structure)

#### Contract Tests (tests/contract/)

| Current Location | New Location | Rationale |
|------------------|--------------|-----------|
| `tests/runner_contract/test_end_to_end_policy_pipeline.py` | `tests/contract/test_end_to_end_policy_pipeline.py` | I1: Tests complete pipeline from dict spec + policies through generate, compose, embed, and validate |
| `tests/runner_contract/test_policy_ownership_imports.py` | `tests/contract/test_policy_ownership_imports.py` | A3: Verifies canonical code paths do not import policies from outside aog_policies |
| `tests/runner_contract/test_policy_serialization.py` | `tests/contract/test_policy_serialization.py` | A1, A2: Verifies all policy classes round-trip through JSON without data loss and contain no callables |
| `tests/designspec/test_run_report_json_clean.py` | `tests/contract/test_run_report_json_clean.py` | Validates JSON serialization of reports - core contract requirement |
| `tests/validity/test_reports_json_schema.py` | `tests/contract/test_reports_json_schema.py` | H3: Reports are JSON serializable and include requested vs effective policies |

#### Unit Tests (tests/unit/)

| Current Location | New Location | Rationale |
|------------------|--------------|-----------|
| `tests/designspec/test_alias_application.py` | `tests/unit/designspec/test_alias_application.py` | Tests alias application logic |
| `tests/designspec/test_ast_compile_eval.py` | `tests/unit/designspec/test_ast_compile_eval.py` | Tests AST compilation and evaluation |
| `tests/designspec/test_ast_validation.py` | `tests/unit/designspec/test_ast_validation.py` | Tests AST validation |
| `tests/designspec/test_context_cache_hits.py` | `tests/unit/designspec/test_context_cache_hits.py` | Tests context caching |
| `tests/designspec/test_named_artifact_registry.py` | `tests/unit/designspec/test_named_artifact_registry.py` | Tests artifact registry |
| `tests/designspec/test_plan_stage_filtering.py` | `tests/unit/designspec/test_plan_stage_filtering.py` | Tests plan stage filtering |
| `tests/designspec/test_schema_versioning.py` | `tests/unit/designspec/test_schema_versioning.py` | Tests schema versioning |
| `tests/designspec/test_spec_hash_stability.py` | `tests/unit/designspec/test_spec_hash_stability.py` | Tests spec hash stability |
| `tests/designspec/test_spec_load_and_normalize.py` | `tests/unit/designspec/test_spec_load_and_normalize.py` | Tests spec loading and normalization |
| `tests/domains/test_domain_from_dict.py` | `tests/unit/domains/test_domain_from_dict.py` | Tests domain creation from dict |
| `tests/domains/test_domain_interface_contract.py` | `tests/unit/domains/test_domain_interface_contract.py` | Tests domain interface contract |
| `tests/embedding/test_domain_to_mesh_used.py` | `tests/unit/embedding/test_domain_to_mesh_used.py` | Tests domain to mesh conversion |
| `tests/pathfinding/test_budgeted_coarse_and_fine.py` | `tests/unit/pathfinding/test_budgeted_coarse_and_fine.py` | D4: Tests budgeted coarse and fine stages |
| `tests/pathfinding/test_hierarchical_mandatory.py` | `tests/unit/pathfinding/test_hierarchical_mandatory.py` | D1: Tests mandatory hierarchical pathfinding |
| `tests/pathfinding/test_inflation_clearance_plus_local_radius.py` | `tests/unit/pathfinding/test_inflation_clearance_plus_local_radius.py` | D2: Tests inflation rule correctness |
| `tests/pathfinding/test_waypoint_skipping.py` | `tests/unit/pathfinding/test_waypoint_skipping.py` | D3: Tests waypoint skipping behavior |
| `tests/primitives/test_fang_hook_constraints.py` | `tests/unit/primitives/test_fang_hook_constraints.py` | E4: Tests fang hook constraints |
| `tests/primitives/test_path_channel_tube_sweep.py` | `tests/unit/primitives/test_path_channel_tube_sweep.py` | E3: Tests path channel tube sweep |
| `tests/resolution/test_operation_pitch_resolver_shared.py` | `tests/unit/resolution/test_operation_pitch_resolver_shared.py` | C4: Tests shared resolver behavior |
| `tests/resolution/test_pitch_resolution_budgeting.py` | `tests/unit/resolution/test_pitch_resolution_budgeting.py` | C1, C2, C3: Tests pitch resolution budgeting |
| `tests/validity/test_open_port_roi_budgeted_connectivity.py` | `tests/unit/validity/test_open_port_roi_budgeted_connectivity.py` | H2: Tests open-port ROI budgeted connectivity |
| `tests/validity/test_validity_runner_orchestrates_checks.py` | `tests/unit/validity/test_validity_runner_orchestrates_checks.py` | H1: Tests validity runner orchestration |
| `tests/test_adapters.py` | `tests/unit/test_adapters.py` | Tests mesh adapters and unit scaling |
| `tests/test_collision_swept.py` | `tests/unit/test_collision_swept.py` | Tests swept-volume collision detection |
| `tests/test_constraints.py` | `tests/unit/test_constraints.py` | Tests meter-scale constraints |
| `tests/test_drift_metrics.py` | `tests/unit/test_drift_metrics.py` | Tests drift-aware validation metrics |
| `tests/test_topology_smoke.py` | `tests/unit/test_topology_smoke.py` | Tests network topology operations |
| `tests/test_units.py` | `tests/unit/test_units.py` | Tests unit conversion system |
| `tests/test_units_sidecar.py` | `tests/unit/test_units_sidecar.py` | Tests sidecar file reading |
| `tests/test_validity_smoke.py` | `tests/unit/test_validity_smoke.py` | Tests validity checks |

#### Integration Tests (tests/integration/)

| Current Location | New Location | Rationale |
|------------------|--------------|-----------|
| `tests/designspec/test_golden_fixture_roundtrip.py` | `tests/integration/test_golden_fixture_roundtrip.py` | Uses golden_example_v1.json fixture |
| `tests/embedding/test_embedding_outputs_all_artifacts.py` | `tests/integration/test_embedding_outputs_all_artifacts.py` | Tests embedding outputs all artifacts |
| `tests/embedding/test_port_recarve_preserves_ports.py` | `tests/integration/test_port_recarve_preserves_ports.py` | G1: Tests port preservation via voxel recarve |
| `tests/composition/test_union_then_embed_once.py` | `tests/integration/test_union_then_embed_once.py` | F1: Tests union-before-embed guarantee |
| (NEW) | `tests/integration/test_runner_end_to_end_union_only.py` | Run until union_voids, assert union void exists and is a mesh |
| (NEW) | `tests/integration/test_runner_end_to_end_full.py` | Assert domain_with_void, void, optional shell exist, validity report exists and is JSON-serializable |
| (NEW) | `tests/integration/test_runner_policy_overrides_and_units.py` | Component-level policy overrides are applied and normalized from input_units |
| (NEW) | `tests/integration/test_runner_backend_params_plumbing.py` | backend_params actually affect runtime behavior or appear in effective policy snapshot/metrics |

#### Regression Tests (tests/regression/)

| Current Location | New Location | Rationale |
|------------------|--------------|-----------|
| `tests/primitives/test_channel_length_modes.py` | `tests/regression/test_channel_length_modes.py` | E1, E2: Tests stop_before_boundary applied once (regression for "double subtract" bug) and no hidden 1mm minimum clamp |
| `tests/test_fix_plan_acceptance.py` | `tests/regression/test_fix_plan_acceptance.py` | Tests for Agent Fix Plan (v39) implementations - regression tests for P0-P2 fixes |

#### Quality Tests (tests/quality/)

| Current Location | New Location | Rationale |
|------------------|--------------|-----------|
| `tests/quality/test_no_hardcoded_magic_lengths.py` | `tests/quality/test_no_hardcoded_magic_lengths.py` | J1: Searches for suspicious constants in runner-critical paths |
| `tests/quality/test_no_numpy_in_json_reports.py` | `tests/quality/test_no_numpy_in_json_reports.py` | J2: Ensures no numpy scalars/arrays leak into reports |

### Files to DELETE (Duplicates)

| File | Reason |
|------|--------|
| `tests/test_designspec_runner_contract.py` | DUPLICATE: Overlaps significantly with `tests/runner_contract/*`. The runner_contract directory has more comprehensive and better-organized tests. |
| `tests/test_designspec_readiness_smoke.py` | DUPLICATE: Overlaps with `tests/runner_contract/test_policy_serialization.py` and `tests/runner_contract/test_end_to_end_policy_pipeline.py`. |

### Files to QUARANTINE (Legacy)

| File | New Location | Reason |
|------|--------------|--------|
| `tests/test_generation.py` | `tests/legacy_disabled/test_generation.py` | LEGACY: Imports `generation/specs/design_spec.py` dataclasses (EllipsoidSpec, TreeSpec, ColonizationSpec, etc.) which are deprecated in favor of the new DesignSpec + aog_policies architecture. |
| `tests/test_api_refactor.py` | `tests/legacy_disabled/test_api_refactor.py` | LEGACY: Imports `generation/specs/design_spec.py`, `generation/policies`, `validity/api/*` - deprecated APIs. |
| `tests/test_designspec_readiness.py` (partial) | `tests/legacy_disabled/test_deprecated_design_from_spec.py` | LEGACY: The `TestDeprecatedDesignFromSpec` class tests deprecated `design_from_spec` API. Other tests in this file should be reviewed and merged into integration tests. |

### Files to EXCLUDE from Readiness Gate (Non-Runner)

| File | Reason | Action |
|------|--------|--------|
| `tests/test_subprocess_runner.py` | Tests automation/subprocess_runner module - not part of runner-based architecture | Keep but exclude from readiness gate |
| `tests/test_workflow.py` | Tests automation/workflow module - not part of runner-based architecture | Keep but exclude from readiness gate |

## Duplicate/Conflict Analysis

### 1. tests/runner_contract/* vs tests/test_designspec_runner_contract.py

**Overlap Analysis:**
- `test_designspec_runner_contract.py` tests: policy-only control surface, JSON-friendly inputs, budget behavior, effective policy snapshots
- `tests/runner_contract/test_policy_serialization.py` tests: JSON serialization, no callables, round-trip
- `tests/runner_contract/test_end_to_end_policy_pipeline.py` tests: complete pipeline from dict spec + policies
- `tests/runner_contract/test_policy_ownership_imports.py` tests: import purity

**Decision:** REMOVE `test_designspec_runner_contract.py` - the `tests/runner_contract/` directory has more comprehensive, better-organized tests that cover the same contracts.

### 2. tests/designspec/* vs tests/test_designspec_readiness*

**Overlap Analysis:**
- `test_designspec_readiness.py` contains `TestDeprecatedDesignFromSpec` (tests deprecated API) and other readiness tests
- `test_designspec_readiness_smoke.py` overlaps with runner_contract tests for policy serialization and pipeline
- `tests/designspec/` contains focused unit tests for specific functionality

**Decision:**
- REMOVE `test_designspec_readiness_smoke.py` (duplicate)
- QUARANTINE `TestDeprecatedDesignFromSpec` from `test_designspec_readiness.py` to legacy_disabled
- KEEP other tests from `test_designspec_readiness.py` and merge into integration tests if needed

### 3. Tests importing generation/specs/* dataclasses

**Files identified:**
- `test_generation.py` - imports EllipsoidSpec, TreeSpec, ColonizationSpec, BoxSpec, InletSpec, OutletSpec, DualTreeSpec, DesignSpec
- `test_api_refactor.py` - imports generation/specs/design_spec.py, generation/policies, validity/api/*

**Decision:** QUARANTINE to `tests/legacy_disabled/` with `@pytest.mark.legacy` marker. These test deprecated APIs that are not part of the new runner-based architecture.

## New Integration Tests to Add

### test_runner_end_to_end_union_only.py
- Run DesignSpecRunner with `run_until="union_voids"`
- Assert union void exists and is a mesh
- Assert intermediate artifacts are created
- Use temp dirs (tmp_path) and do not write to repo

### test_runner_end_to_end_full.py
- Run DesignSpecRunner through full pipeline
- Assert domain_with_void, void, optional shell exist
- Assert validity report exists and is JSON-serializable
- Verify all artifacts are JSON-clean

### test_runner_policy_overrides_and_units.py
- Test component-level policy overrides are applied
- Test input_units normalization
- Verify effective policy reflects overrides

### test_runner_backend_params_plumbing.py
- Test backend_params actually affect runtime behavior
- Verify backend_params appear in effective policy snapshot/metrics

## pytest.ini Configuration

```ini
[pytest]
markers =
    legacy: marks tests as legacy (not run by default)
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests

addopts = -m "not legacy"

testpaths = tests
```

## Test Commands (Windows-Friendly)

### Default Fast Suite (CI Safe)
```bash
pytest -q tests/contract tests/unit tests/regression tests/quality
```

### Full Readiness Gate
```bash
pytest -q tests/contract tests/unit tests/integration tests/regression tests/quality
```

### Optional Legacy Tests
```bash
pytest -q -m legacy tests/legacy_disabled
```

### Run All Tests Including Legacy
```bash
pytest -q tests/
```

## Migration Checklist

- [ ] Create new directory structure
- [ ] Move contract tests to tests/contract/
- [ ] Move unit tests to tests/unit/
- [ ] Move integration tests to tests/integration/
- [ ] Move regression tests to tests/regression/
- [ ] Keep quality tests in tests/quality/
- [ ] Create tests/legacy_disabled/ and move legacy tests
- [ ] Delete duplicate test files
- [ ] Add @pytest.mark.legacy to quarantined tests
- [ ] Create pytest.ini with markers and defaults
- [ ] Add missing integration tests
- [ ] Update quality suite scope to runner-critical code only
- [ ] Add determinism test
- [ ] Verify all test commands pass

## Summary Statistics

| Category | File Count | Status |
|----------|------------|--------|
| Contract Tests | 5 | Keep |
| Unit Tests | 31 | Keep |
| Integration Tests | 4 existing + 4 new = 8 | Keep/Add |
| Regression Tests | 2 | Keep |
| Quality Tests | 2 | Keep |
| Legacy/Quarantine | 3 | Quarantine |
| Duplicates | 2 | Delete |
| Non-Runner (Exclude) | 2 | Exclude from gate |
| **Total** | **57 original** | |
