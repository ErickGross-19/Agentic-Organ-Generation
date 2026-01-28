# Agentic Organ Generation

A toolkit for generating 3D organ structures with embedded vascular networks. The system produces printable scaffolds with vascular channels carved as negative space, suitable for biomedical research, tissue engineering, and additive manufacturing.

## What This Repo Does Today

The canonical workflow uses **DesignSpec JSON files** loaded via `designspec.DesignSpec` and executed by `designspec.DesignSpecRunner`. The runner orchestrates all stages of the pipeline: policy compilation, domain setup, network generation, mesh synthesis, void union, domain embedding, port recarving, and validity checking.

Primary outputs:
- **domain_with_void.stl**: Solid scaffold with vascular network carved as negative space (ready for 3D printing)
- **structure.stl**: Surface mesh of the vascular network (for visualization)
- **network.json**: Graph representation with node positions and segment properties
- **run_report.json**: Full execution report with stage timings, policies used, and validation results

## Installation

```bash
git clone https://github.com/ErickGross-19/Agentic-Organ-Generation.git
cd Agentic-Organ-Generation
pip install -r requirements.txt
```

## Canonical Workflow

The recommended way to use this system is through JSON DesignSpec files:

```python
from designspec import DesignSpec, DesignSpecRunner
from designspec.plan import ExecutionPlan

# Load a JSON design specification
spec = DesignSpec.from_json("examples/designspec/golden_example_v1.json")

# Run the full pipeline
runner = DesignSpecRunner(spec=spec, output_dir="./output")
result = runner.run()

print(f"Success: {result.success}")
print(f"Spec hash: {result.spec_hash}")
print(f"Stages completed: {result.stages_completed}")
```

### Partial Execution

Run until a specific stage using `run_until`:

```python
from designspec.plan import ExecutionPlan

# Run until union_voids (before embedding)
plan = ExecutionPlan(run_until="union_voids")
runner = DesignSpecRunner(spec=spec, plan=plan, output_dir="./output")
result = runner.run()
```

Available stages (in order): `compile_policies`, `compile_domains`, `component_ports`, `component_build`, `component_mesh`, `union_voids`, `mesh_domain`, `embed`, `port_recarve`, `validity`, `export`.

You can also use `run_only` to execute specific stages or `components_subset` to process only certain components.

## Policies (aog_policies)

All runner behavior is controlled through policies defined in `aog_policies/`. Policies are specified in the JSON spec under the `policies` section and compiled at runtime:

```json
{
  "policies": {
    "resolution": {
      "min_pitch": 0.0001,
      "max_pitch": 0.001,
      "max_voxel_budget": 1000000
    },
    "embedding": {
      "recarve_ports": true,
      "shell_thickness": 0.001
    },
    "channels": {
      "length_mode": "explicit",
      "length": 0.005
    }
  }
}
```

Key policy types:
- **ResolutionPolicy**: Controls voxel pitch and budget for mesh operations
- **EmbeddingPolicy**: Controls domain embedding and port recarving
- **ChannelPolicy**: Controls primitive channel generation
- **GrowthPolicy**: Controls space colonization parameters
- **MeshSynthesisPolicy**: Controls network-to-mesh conversion

Components can override top-level policies using `policy_overrides` for per-component customization.

## Outputs and Reproducibility

Every run produces deterministic outputs when a seed is specified:

```json
{
  "meta": {
    "name": "my_design",
    "seed": 42,
    "input_units": "m"
  }
}
```

The runner computes a `spec_hash` from the normalized specification, ensuring identical specs produce identical hashes. The `run_report.json` captures:
- Spec hash for reproducibility verification
- Stage-by-stage timing and success status
- Requested vs effective policies for each stage
- Validation results and any warnings

## Testing

Run the readiness gate tests:

```bash
pytest -q tests/contract tests/unit tests/integration tests/regression tests/quality
```

Or run specific test categories:

```bash
# Integration tests for runner and designspec
pytest -q tests/integration -k "runner or designspec"

# Contract tests for API compatibility
pytest -q tests/contract
```

## Legacy Note

Older APIs exist in `generation/specs/` (Python dataclasses) and `generation/api/design.py` but are deprecated. These trigger deprecation warnings when imported. New code should use JSON DesignSpec files with `designspec.DesignSpec`. Once the `/legacy` directory is created, deprecated implementations will be moved there.

## Not Yet Migrated

The following directories have not been updated to use the new DesignSpec workflow:
- `gui/` - Graphical user interface
- `automation/` - LLM integration and agent workflows
- `examples/` - Example scripts and notebooks

These continue to use older APIs and may not reflect the current canonical workflow.

## Project Structure

```
Agentic-Organ-Generation/
├── designspec/           # Canonical: JSON spec loading and runner
│   ├── spec.py           # DesignSpec class
│   ├── runner.py         # DesignSpecRunner orchestration
│   └── plan.py           # ExecutionPlan for partial runs
│
├── aog_policies/         # Policy surface for runner behavior
│   ├── resolution.py     # ResolutionPolicy (voxel pitch and budget)
│   ├── generation.py     # ChannelPolicy, GrowthPolicy, EmbeddingPolicy, MeshSynthesisPolicy
│   ├── validity.py       # ValidationPolicy, RepairPolicy
│   └── base.py           # OperationReport for stage feedback
│
├── generation/           # Core generation library
│   ├── core/             # VascularNetwork, Node, Segment
│   ├── ops/              # Operations (space colonization, mesh synthesis)
│   ├── api/              # High-level API (generate, embed)
│   └── adapters/         # Export adapters (STL, mesh)
│
├── validity/             # Validation system
│   ├── pre_embedding/    # Topology, flow, mesh checks
│   └── post_embedding/   # Manufacturability, connectivity checks
│
├── tests/                # Test suites
│   ├── contract/         # API contract tests
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   ├── regression/       # Regression tests
│   └── quality/          # Quality checks
│
├── gui/                  # (Not yet migrated)
├── automation/           # (Not yet migrated)
├── examples/             # (Not yet migrated)
└── legacy/               # Deprecated code (when created)
```

## License

See LICENSE file for details.
