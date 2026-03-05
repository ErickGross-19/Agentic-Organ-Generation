# DesignSpec Examples

Runnable DesignSpec JSON files that exercise the DesignSpec + DesignSpecRunner pipeline. These serve as spec fixtures for testing and as starting points for new designs.

## How to Run

### Using the Runner Script

```bash
python scripts/run_designspec_example.py --spec examples/designspec/malaria_venule_bifurcating_tree.json --out ./output

python scripts/run_designspec_example.py --spec examples/designspec/malaria_venule_bifurcating_tree.json --out ./output --run-until compile_domains
```

### Using Python

```python
from designspec.spec import DesignSpec
from designspec.runner import DesignSpecRunner
from pathlib import Path

spec = DesignSpec.from_file("examples/designspec/malaria_venule_bifurcating_tree.json")
runner = DesignSpecRunner(spec, output_dir=Path("./output"))
result = runner.run()
print(f"Success: {result.success}")
```

### Using the Web UI

1. Start the backend: `cd backend && python run.py`
2. Start the frontend: `cd frontend && npm run dev`
3. Open http://localhost:3000
4. Switch to "DesignSpec Agent" mode
5. Describe your scaffold in the chat — the LLM agent will create and iterate on DesignSpec JSON

## Output Structure

```
<output_dir>/
├── artifacts/
│   ├── <component>_network.json
│   ├── <component>_void.stl
│   └── union_void.stl
├── domain_with_void.stl
├── void_union.stl
├── shell.stl (optional)
├── run_report.json
└── validity_report.json
```

## Malaria Venule Insert Examples

| Example | Backend | Description |
|---------|---------|-------------|
| malaria_venule_bifurcating_tree | scaffold_topdown | 5-inlet bifurcating tree |
| malaria_venule_bifurcating_tree_with_merge | scaffold_topdown | Bifurcating tree with merge on collision |
| malaria_venule_control_ridge_only | none | Control spec: ridged cylinder only |
| malaria_venule_vertical_channels | primitive_channels | 9 straight tapered vertical channels |
| malaria_venule_fang_hook_channels | primitive_channels | 9 radial-out fang-hook curved channels |
| malaria_venule_space_colonization | space_colonization | Dense multi-inlet space colonization |

All malaria venule examples use a cylinder domain (R=5mm, H=2mm) with ridge enabled and multiple inlets.

## Available Build Types

These examples use vascular backends. The pipeline also supports `manifold_generator` build type for 44 geometry generators (lattice, skeletal, organ, soft tissue, tubular, dental, microfluidic). See `generation/README.md` for the full list.

## Backend Status

See [docs/status.md](../../docs/status.md) for current backend status.

## Partial Execution

Use `--run-until` to stop at a specific stage for debugging:

```bash
python scripts/run_designspec_example.py --spec example.json --out ./out --run-until compile_domains
python scripts/run_designspec_example.py --spec example.json --out ./out --run-until union_voids
python scripts/run_designspec_example.py --spec example.json --out ./out --run-until embed
```

## Design Rules

1. **Deterministic**: Every example includes `meta.seed` for reproducibility
2. **Fast execution**: Small domains, low iteration counts
3. **Union-before-embed**: Multi-component examples union all voids before embedding
4. **Validity enabled**: All examples enable validity checking by default
