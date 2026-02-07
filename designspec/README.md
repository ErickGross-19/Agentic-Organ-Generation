# DesignSpec Package

Declarative JSON specification and execution pipeline for scaffold generation. This is the canonical workflow for all generation in the AOG system.

## Purpose

The DesignSpec package provides a unified, JSON-driven specification format for defining generation pipelines. It enables:

- **Reproducibility**: Stable spec hashes, versioned schemas, and complete execution metadata
- **Policy-driven execution**: All behavior controlled through `aog_policies` objects
- **Multi-component support**: Union all void contributions, then embed once
- **Multiple build types**: Vascular backends (scaffold_topdown, space_colonization, programmatic) and ManifoldBackend (44 geometry generators)
- **Partial execution**: Run specific stages or subsets of components
- **Artifact management**: Named intermediate outputs with caching

## Pipeline Stages

The runner executes these stages in order:

| # | Stage | Description |
|---|-------|-------------|
| 1 | `compile_policies` | Compile policy dicts to `aog_policies` objects (ResolutionPolicy, EmbeddingPolicy, ChannelPolicy, GrowthPolicy, MeshSynthesisPolicy, ManifoldGeneratorPolicy, etc.) |
| 2 | `compile_domains` | Compile domain dicts to runtime Domain objects (EllipsoidDomain, BoxDomain, CylinderDomain) |
| 3 | `component_ports:<id>` | Resolve port positions for each component |
| 4 | `component_build:<id>` | Generate network/mesh for each component using the configured backend |
| 5 | `component_mesh:<id>` | Convert network to void mesh for each component |
| 6 | `union_voids` | Union all component void meshes into a single void |
| 7 | `mesh_domain` | Generate domain mesh |
| 8 | `embed` | Embed unified void into domain (boolean subtraction) |
| 9 | `port_recarve` | Recarve ports if enabled |
| 10 | `validity` | Run pre/post embedding validation checks |
| 11 | `export` | Export outputs to STL/JSON files |

## Build Types

The `component_build` stage supports multiple build types:

| Build Type | Backend | Description |
|------------|---------|-------------|
| `scaffold_topdown` | `ScaffoldTopdownBackend` | Recursive bifurcating tree with collision avoidance |
| `space_colonization` | `SpaceColonizationBackend` | Attractor-based organic vascular growth |
| `programmatic` | `ProgrammaticBackend` | DSL-based generation with A* pathfinding |
| `primitive_channels` | (inline) | Simple channel geometry (taper, fang-hook) |
| `manifold_generator` | `ManifoldBackend` | 44 MorphoStruct geometry generators |

## Core Components

| Module | Class | Purpose |
|--------|-------|---------|
| `spec.py` | `DesignSpec` | Load, validate, normalize JSON specs; compute stable content hashes |
| `runner.py` | `DesignSpecRunner` | Execute the 11-stage pipeline with partial execution controls |
| `context.py` | `RunnerContext` | Cache expensive computations and manage artifact storage |
| `plan.py` | `ExecutionPlan` | Define `run_until`, `run_only`, `skip`, and `components_subset` |
| `reports/run_report.py` | `RunReport` | Capture full reproducibility state including environment and hashes |

## Invariants

1. **Schema versioning**: All specs must declare `schema.name` = "aog_designspec" and a compatible version
2. **Unit normalization**: All geometric values converted to meters internally from `meta.input_units`
3. **Union-before-embed**: Multi-component specs union all voids before a single embedding operation
4. **Policy ownership**: All policies come from `aog_policies` package, not duplicated here
5. **Deterministic execution**: Same spec + seed produces identical outputs

## Usage

```python
from designspec import DesignSpec, DesignSpecRunner

spec = DesignSpec.from_json("my_spec.json")

runner = DesignSpecRunner(
    spec,
    run_until="embed",
    components_subset=["net_1"],
)

result = runner.run()
print(f"Spec hash: {spec.spec_hash}")
print(f"Success: {result.success}")
```

## Schema Version

Current schema: `aog_designspec` version `1.0.0`
