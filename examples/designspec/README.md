# DesignSpec Examples

This directory contains a comprehensive set of DesignSpec JSON examples that showcase the capabilities of the Agentic Organ Generation system and serve as runnable "spec fixtures" to exercise the DesignSpec + DesignSpecRunner pipeline.

## How to Run an Example

### Using the Runner Script

```bash
# Run a specific example
python scripts/run_designspec_example.py --spec examples/designspec/01_minimal_box_network.json --out ./output

# Run until a specific stage (for debugging)
python scripts/run_designspec_example.py --spec examples/designspec/01_minimal_box_network.json --out ./output --run-until compile_domains

# Available stages: compile_policies, compile_domains, place_ports, build_components, 
#                   synthesize_meshes, merge_meshes, union_voids, embed, validate, export
```

### Using Python Directly

```python
from designspec.spec import DesignSpec
from designspec.runner import DesignSpecRunner
from designspec.plan import ExecutionPlan
from pathlib import Path

# Load the spec
spec = DesignSpec.from_file("examples/designspec/01_minimal_box_network.json")

# Create output directory
output_dir = Path("./output/01_minimal_box_network")
output_dir.mkdir(parents=True, exist_ok=True)

# Run the full pipeline
runner = DesignSpecRunner(spec, output_dir=output_dir)
result = runner.run()

# Check results
print(f"Success: {result.success}")
print(f"Stages completed: {result.stages_completed}")
```

## Where Outputs Go

Each example specifies its output directory in `policies.output.output_dir`. When running via the runner script with `--out`, outputs are placed in:

```
<output_dir>/
├── artifacts/
│   ├── <component>_network.json    # Network graph data
│   ├── <component>_void.stl        # Component void mesh
│   └── union_void.stl              # Combined void mesh
├── domain_with_void.stl            # Final embedded domain
├── void_union.stl                  # Final void mesh
├── shell.stl                       # Optional shell mesh
├── run_report.json                 # Execution report
└── validity_report.json            # Validation results
```

## What "Success" Means

A successful run produces:

1. **run_report.json**: Contains stage-by-stage execution details, timing, and any warnings
2. **validity_report.json**: Contains validation results (when validity is enabled)
3. **Output meshes**: domain_with_void.stl, void_union.stl, and optionally shell.stl
4. **No fatal errors**: All stages complete without exceptions

The validity report should show:
- `watertight: true` - The mesh is closed
- `components: 1` - Single connected component
- `open_ports: all_open` - All ports are accessible
- `void_inside_domain: true` - Void is properly contained

## Partial Execution for Debugging

Use the `--run-until` flag to stop at a specific stage:

```bash
# Stop after domain compilation (fast, tests domain parsing)
python scripts/run_designspec_example.py --spec example.json --out ./out --run-until compile_domains

# Stop after void union (tests generation without embedding)
python scripts/run_designspec_example.py --spec example.json --out ./out --run-until union_voids

# Stop after embedding (tests full pipeline without validation)
python scripts/run_designspec_example.py --spec example.json --out ./out --run-until embed
```

## Examples Overview

| # | Example | Description |
|---|---------|-------------|
| 01 | minimal_box_network | Simplest complete example: box domain, space colonization, embedding, validity |
| 02 | multicomponent_union_embed | Two components (network + channels) unioned before single embedding |
| 03 | transform_domain | TransformDomain with 4x4 row-major rotation/translation matrix |
| 04 | composite_domain_boolean | CompositeDomain with difference operation (cylinder - box) |
| 05 | implicit_ast_domain | ImplicitDomain with JSON AST SDF (sphere) |
| 06 | mesh_domain_user_faces | Box domain with user-defined face planes for port placement |
| 07 | fang_hook_channels | Fang-hook curved channels with radial outward bending |
| 08 | path_channel_tube_sweep | Taper channels with explicit length and taper schedule |
| 09 | hierarchical_pathfinding_waypoints | Pathfinding with obstacles, waypoints, and corridor search |
| 10 | programmatic_backend_dsl | Programmatic backend with DSL via backend_params |
| 11 | kary_backend | K-ary recursive tree backend with stable ID allocation |
| 12 | cco_hybrid_backend | CCO hybrid backend with Murray's Law optimization |
| 13 | validity_open_ports_focus | Open-port validation with voxel recarve preservation |
| 14 | budget_relaxation_showcase | Budget relaxation with max_voxels and pitch warnings |

## Capability Coverage Matrix

The following matrix shows which capabilities each example exercises:

### Domain Types

| Example | Box | Cylinder | Sphere | Ellipsoid | Capsule | Frustum | Transform | Composite | Implicit | Mesh |
|---------|-----|----------|--------|-----------|---------|---------|-----------|-----------|----------|------|
| 01 | X | | | | | | | | | |
| 02 | | X | | | | | | | | |
| 03 | | X | | | | | X | | | |
| 04 | | X | | | | | | X | | |
| 05 | | | | | | | | | X | |
| 06 | X | | | | | | | | | X* |
| 07 | | X | | | | | | | | |
| 08 | X | | | | | | | | | |
| 09 | X | | | | | | | X | | |
| 10 | X | | | | | | | | | |
| 11 | | X | | | | | | | | |
| 12 | | X | | | | | | | | |
| 13 | | X | | | | | | | | |
| 14 | X | | | | | | | | | |

*Example 06 uses box with user-defined faces (mesh-like behavior)

### Generation Backends

| Example | Space Colonization | K-ary Tree | CCO Hybrid | Programmatic |
|---------|-------------------|------------|------------|--------------|
| 01 | X | | | |
| 02 | X | | | |
| 03 | X | | | |
| 04 | X | | | |
| 05 | X | | | |
| 06 | X | | | |
| 07 | | | | |
| 08 | | | | |
| 09 | X | | | |
| 10 | | | | X |
| 11 | | X | | |
| 12 | | | X | |
| 13 | X | | | |
| 14 | X | | | |

### Channel Types

| Example | Straight | Taper | Fang-Hook | Path Channel |
|---------|----------|-------|-----------|--------------|
| 01 | | | | |
| 02 | | X | | |
| 03 | | | | |
| 04 | | | | |
| 05 | | | | |
| 06 | | | | |
| 07 | | | X | |
| 08 | | X | | |
| 09 | | | | |
| 10 | | | | |
| 11 | | | | |
| 12 | | | | |
| 13 | | | | |
| 14 | | | | |

### Port & Embedding Features

| Example | Face Placement | Ridge Constraint | Port Recarve | Multi-Port | Shell Output |
|---------|---------------|------------------|--------------|------------|--------------|
| 01 | X | X | X | | X |
| 02 | X | X | X | | X |
| 03 | X | X | X | | |
| 04 | X | X | X | | |
| 05 | X | | X | | |
| 06 | X | X | X | | |
| 07 | X | X | X | X | |
| 08 | X | X | X | X | X |
| 09 | X | X | X | | |
| 10 | X | X | X | | |
| 11 | X | X | X | | |
| 12 | X | X | X | | |
| 13 | X | X | X | X | |
| 14 | X | X | X | | X |

### Validation & Special Features

| Example | Watertight | Open Ports | Components | Pathfinding | Waypoints | Budget Relaxation |
|---------|------------|------------|------------|-------------|-----------|-------------------|
| 01 | X | X | X | | | |
| 02 | X | X | X | | | |
| 03 | X | X | X | | | |
| 04 | X | X | X | | | |
| 05 | X | X | X | | | |
| 06 | X | X | X | | | |
| 07 | X | X | X | | | |
| 08 | X | X | X | | | |
| 09 | X | X | X | X | X | |
| 10 | X | X | X | | | |
| 11 | X | X | X | | | |
| 12 | X | X | X | | | |
| 13 | X | X | X | | | |
| 14 | X | X | X | | | X |

## Running Integration Tests

```bash
# Run fast smoke tests (CI subset)
pytest -q tests/integration/test_examples_smoke.py

# Run all examples including slow ones
pytest -q -m slow tests/integration/test_examples_smoke.py

# Run specific example test
pytest -q tests/integration/test_examples_smoke.py::test_01_minimal_box_network
```

## Design Rules

All examples follow these design rules:

1. **Top-level structure**: schema, meta, policies, domains, components, composition, embedding, validity, outputs
2. **Deterministic**: Every example includes `meta.seed` for reproducibility
3. **Fast execution**: Small domains (5-30 mm), low iteration counts, reasonable voxel budgets
4. **Union-before-embed**: Multi-component examples union all voids before single embedding
5. **Port preservation**: Uses voxel recarve mode for port preservation
6. **Validity enabled**: All examples enable validity checking by default

## Troubleshooting

### Common Issues

1. **"max_voxels exceeded"**: The domain is too large for the voxel budget. Either increase `max_voxels` or use a smaller domain.

2. **"pitch relaxation warning"**: The requested pitch was too fine for the voxel budget. The system automatically relaxes the pitch. Check `run_report.json` for the effective pitch used.

3. **"open port validation failed"**: A port is blocked after embedding. Ensure `preserve_mode: "recarve"` is enabled and `carve_depth` is sufficient.

4. **"watertight check failed"**: The mesh has holes. Enable `repair.fill_holes_enabled: true` and `repair.voxel_repair_enabled: true`.

### Getting Help

For issues with examples, check:
1. The `run_report.json` for stage-by-stage details and warnings
2. The `validity_report.json` for specific validation failures
3. The intermediate artifacts in the `artifacts/` directory
