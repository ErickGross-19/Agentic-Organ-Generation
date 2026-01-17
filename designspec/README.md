# DesignSpec Package

Declarative specification and runner for vascular network generation pipelines.

## Purpose

The DesignSpec package provides a unified, JSON-driven specification format for defining
vascular network generation pipelines. It enables:

- **Reproducibility**: Stable spec hashes, versioned schemas, and complete execution metadata
- **Policy-driven execution**: All behavior controlled through aog_policies objects
- **Multi-component support**: Union all void contributions, then embed once
- **Partial execution**: Run specific stages or subsets of components
- **Artifact management**: Named intermediate outputs with caching

## Invariants

1. **Schema versioning**: All specs must declare `schema.name` = "aog_designspec" and a compatible version
2. **Unit normalization**: All geometric values are converted to meters internally from `meta.input_units`
3. **Union-before-embed**: Multi-component specs union all voids before a single embedding operation
4. **Policy ownership**: All policies come from `aog_policies` package, not duplicated here
5. **Deterministic execution**: Same spec + seed produces identical outputs

## Core Components

### DesignSpec (`spec.py`)
Loads, validates, and normalizes JSON specifications. Computes stable content hashes.

### DesignSpecRunner (`runner.py`)
Executes the pipeline through defined stages, respecting partial execution controls.

### RunnerContext (`context.py`)
Manages caching of expensive computations and artifact storage.

### ExecutionPlan (`plan.py`)
Defines stable stage names and supports `run_until`, `run_only`, `skip`, and `components_subset`.

### RunReport (`reports/run_report.py`)
Captures full reproducibility state including environment, hashes, and stage reports.

## Stages

The runner executes these stages in order:

1. `compile_policies` - Compile policy dicts to aog_policies objects
2. `compile_domains` - Compile domain dicts to runtime Domain objects
3. `component_ports:<id>` - Resolve port positions for each component
4. `component_build:<id>` - Generate network/mesh for each component
5. `component_mesh:<id>` - Convert network to void mesh for each component
6. `union_voids` - Union all component void meshes
7. `mesh_domain` - Generate domain mesh
8. `embed` - Embed unified void into domain
9. `port_recarve` - Recarve ports if enabled
10. `validity` - Run validity checks
11. `export` - Export outputs to files

## Usage

```python
from designspec import DesignSpec, DesignSpecRunner

# Load spec from JSON file
spec = DesignSpec.from_json("my_spec.json")

# Create runner with options
runner = DesignSpecRunner(
    spec,
    run_until="embed",  # Stop after embedding
    components_subset=["net_1"],  # Only process net_1
)

# Execute pipeline
result = runner.run()

# Access outputs
print(f"Spec hash: {spec.spec_hash}")
print(f"Success: {result.success}")
print(result.run_report.to_json())
```

## Schema Version

Current schema: `aog_designspec` version `1.0.0`

Compatible versions follow semver rules:
- Major version must match exactly
- Minor version must match exactly
- Patch versions are compatible if listed in `compatible_with`
