# DesignSpec Examples

This directory contains malaria venule insert DesignSpec JSON examples that showcase the capabilities of the Agentic Organ Generation system and serve as runnable "spec fixtures" to exercise the DesignSpec + DesignSpecRunner pipeline.

## How to Run an Example

### Using the Runner Script

```bash
python scripts/run_designspec_example.py --spec examples/designspec/malaria_venule_bifurcating_tree.json --out ./output

python scripts/run_designspec_example.py --spec examples/designspec/malaria_venule_bifurcating_tree.json --out ./output --run-until compile_domains
```

### Using Python Directly

```python
from designspec.spec import DesignSpec
from designspec.runner import DesignSpecRunner
from designspec.plan import ExecutionPlan
from pathlib import Path

spec = DesignSpec.from_file("examples/designspec/malaria_venule_bifurcating_tree.json")

output_dir = Path("./output/malaria_venule_bifurcating_tree")
output_dir.mkdir(parents=True, exist_ok=True)

runner = DesignSpecRunner(spec, output_dir=output_dir)
result = runner.run()

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

## Malaria Venule Insert Examples

| Example | Description | Backend |
|---------|-------------|---------|
| malaria_venule_bifurcating_tree | Bifurcating tree with scaffold_topdown backend, 5 inlets | scaffold_topdown |
| malaria_venule_bifurcating_tree_with_merge | Bifurcating tree with merge on collision enabled | scaffold_topdown |
| malaria_venule_control_ridge_only | Control spec: ridged cylinder only, no network/void | none |
| malaria_venule_vertical_channels | 9 straight tapered vertical channels | primitive_channels |
| malaria_venule_fang_hook_channels | 9 radial-out fang-hook curved channels | primitive_channels |
| malaria_venule_space_colonization | Dense multi-inlet space colonization network | space_colonization |

All malaria venule insert examples share common characteristics:
- Cylinder domain: R=5mm, H=2mm
- Ridge enabled on top face
- Multiple inlets for multi-channel flow
- Consistent composition policies (`keep_largest_component: false`)

## Backend Status

See `docs/status.md` for the current status of generation backends:
- **scaffold_topdown**: Active, preferred for bifurcating trees
- **space_colonization**: Active, recommended for organic growth
- **programmatic**: Active, for DSL-based generation
- **kary_tree**: DEPRECATED - use scaffold_topdown instead
- **cco_hybrid**: BLOCKED - not finished, do not use
- **NLP optimization**: BLOCKED - not finished, do not use

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
