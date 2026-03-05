# Validity Checking Library

Comprehensive validation for generated scaffolds. Runs as stage 10 of the DesignSpec pipeline and can also be used standalone. Checks are split into pre-embedding (intrinsic mesh/network quality) and post-embedding (manufacturability of the domain-with-void).

## Validation Stages

### Pre-Embedding Validation

Checks the vascular network mesh and graph topology before embedding into a manufacturing domain.

| Category | Checks |
|----------|--------|
| **Mesh** | Watertightness, manifoldness, surface quality (aspect ratios), degenerate faces |
| **Graph** | Murray's law compliance, branch order distribution, segment collisions, self-intersections |
| **Flow** | Mass conservation at junctions, Reynolds number (laminar flow), pressure monotonicity |

### Post-Embedding Validation

Checks the embedded structure (domain with void) for manufacturability.

| Category | Checks |
|----------|--------|
| **Connectivity** | Port accessibility, trapped fluid detection, channel continuity |
| **Printability** | Minimum channel diameter, wall thickness, unsupported overhangs |
| **Domain** | Outlet openings, void fraction coverage |

## Directory Structure

```
validity/
├── __init__.py              # High-level exports
├── orchestrators.py         # run_pre_embedding_validation(), run_post_embedding_validation()
├── pre_embedding/
│   ├── mesh_checks.py       # Mesh quality checks
│   ├── graph_checks.py      # Network topology checks
│   └── flow_checks.py       # Hemodynamic flow checks
├── post_embedding/
│   ├── connectivity_checks.py  # Fluid connectivity
│   ├── printability_checks.py  # Manufacturing constraints
│   └── domain_checks.py       # Domain-specific checks
├── mesh/                    # Mesh repair utilities
└── analysis/                # Analysis utilities
```

## Usage

### In DesignSpec Pipeline

Validation runs automatically as stage 10 (`validity`) of the pipeline. Configure via the spec's policies:

```json
{
  "policies": {
    "resolution": {"voxel_pitch": 0.1},
    "manufacturing": {
      "min_channel_diameter": 0.5,
      "min_wall_thickness": 0.3
    }
  }
}
```

### Standalone

```python
from validity import run_pre_embedding_validation, run_post_embedding_validation

pre_report = run_pre_embedding_validation(mesh_path="structure.stl", network=vascular_network)
print(f"Status: {pre_report.status}")  # "ok", "warnings", or "fail"

post_report = run_post_embedding_validation(
    mesh_path="domain_with_void.stl",
    manufacturing_config={"min_channel_diameter": 0.5, "min_wall_thickness": 0.3}
)
```

## Validation Reports

Reports provide:
- `passed` (bool) - overall pass/fail
- `status` - "ok", "warnings", or "fail"
- `stage` - "pre_embedding" or "post_embedding"
- Individual check results with details, messages, and warnings
- Summary statistics (total, passed, failed, warning counts)
- JSON export via `report.save_json()`

## Integration with LLM Agent

When validation fails during a DesignSpec pipeline run, the `RunAnalyzer` in `automation/designspec_llm/` automatically classifies the failure and proposes fix patches. The LLM agent uses validation reports as context for suggesting spec modifications.
