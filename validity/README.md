# Validity Checking Library (Part B)

The Validity Checking Library provides comprehensive validation checks for organ structures. It includes both pre-embedding checks (before the structure is embedded into a domain) and post-embedding checks (after embedding into a domain for manufacturing).

## Overview

Validation is split into two stages:

1. **Pre-Embedding Validation**: Checks the intrinsic properties of the vascular network mesh and graph topology before it is embedded into a manufacturing domain. These checks ensure the structure is geometrically valid and biophysically plausible.

2. **Post-Embedding Validation**: Checks the embedded structure (domain with void) for manufacturability, connectivity, and physical constraints. These checks ensure the structure can be successfully manufactured and will function as intended.

## Directory Structure

```
validity/
├── __init__.py              # Main entry point with high-level exports
├── orchestrators.py         # Main validation orchestrators
├── pre_embedding/           # Pre-embedding validation checks
│   ├── __init__.py
│   ├── mesh_checks.py       # Mesh quality checks
│   ├── graph_checks.py      # Network topology checks
│   └── flow_checks.py       # Hemodynamic flow checks
├── post_embedding/          # Post-embedding validation checks
│   ├── __init__.py
│   ├── connectivity_checks.py  # Fluid connectivity checks
│   ├── printability_checks.py  # Manufacturing checks
│   └── domain_checks.py        # Domain-specific checks
├── mesh/                    # Mesh repair utilities (from vascular_network)
└── analysis/                # Analysis utilities (from vascular_network)
```

## Main Entry Points

### High-Level Orchestrators

#### `run_pre_embedding_validation()`

Run all pre-embedding validation checks on a mesh and/or network.

```python
from validity import run_pre_embedding_validation

# Validate a mesh file
report = run_pre_embedding_validation(mesh_path="structure.stl")
print(f"Status: {report.status}")  # "ok", "warnings", or "fail"
print(f"Passed: {report.passed}")

# Validate with network for graph/flow checks
report = run_pre_embedding_validation(
    mesh_path="structure.stl",
    network=vascular_network,
)

# Save report to JSON
report.save_json("validation_report.json")
```

#### `run_post_embedding_validation()`

Run all post-embedding validation checks on an embedded mesh.

```python
from validity import run_post_embedding_validation

# Validate with manufacturing constraints
report = run_post_embedding_validation(
    mesh_path="domain_with_void.stl",
    manufacturing_config={
        "min_channel_diameter": 0.5,  # mm
        "min_wall_thickness": 0.3,    # mm
        "plate_size": (200, 200, 200),  # mm
    }
)

print(f"Status: {report.status}")
print(f"Summary: {report.summary}")
```

## Pre-Embedding Checks

### Mesh Checks (`pre_embedding/mesh_checks.py`)

Validates the geometric quality of the mesh surface.

| Check | Description | Pass Criteria |
|-------|-------------|---------------|
| `check_watertightness()` | Mesh is closed with no holes | Euler number = 2 |
| `check_manifoldness()` | Each edge shared by exactly 2 faces | No non-manifold edges |
| `check_surface_quality()` | Face aspect ratios acceptable | Max aspect ratio < 10 |
| `check_degenerate_faces()` | No zero-area faces | No degenerate faces |

```python
from validity.pre_embedding.mesh_checks import run_all_mesh_checks
import trimesh

mesh = trimesh.load("structure.stl")
report = run_all_mesh_checks(mesh)

for check in report.checks:
    print(f"{check.check_name}: {'PASS' if check.passed else 'FAIL'}")
    if check.warnings:
        print(f"  Warnings: {check.warnings}")
```

### Graph Checks (`pre_embedding/graph_checks.py`)

Validates the network topology and biophysical constraints.

| Check | Description | Pass Criteria |
|-------|-------------|---------------|
| `check_murrays_law()` | Bifurcation radii follow Murray's law | Deviation < tolerance |
| `check_branch_order()` | Branch order distribution reasonable | Max order < threshold |
| `check_collisions()` | No segment collisions | No collisions detected |
| `check_self_intersections()` | No zero-length segments | No self-intersections |

```python
from validity.pre_embedding.graph_checks import run_all_graph_checks

report = run_all_graph_checks(
    network,
    murray_gamma=3.0,
    murray_tolerance=0.15,
    max_branch_order=20,
    min_clearance=0.0,
)

print(f"Murray's law compliance: {report.checks[0].details['compliance_rate']:.1%}")
```

### Flow Checks (`pre_embedding/flow_checks.py`)

Validates hemodynamic flow properties.

| Check | Description | Pass Criteria |
|-------|-------------|---------------|
| `check_flow_plausibility()` | Mass conserved at junctions | Balance error < threshold |
| `check_reynolds_number()` | Flow is laminar | Re < 2300 |
| `check_pressure_monotonicity()` | Pressure decreases along flow | Monotonic decrease |

```python
from validity.pre_embedding.flow_checks import run_all_flow_checks

report = run_all_flow_checks(
    network,
    max_flow_balance_error=0.05,
    max_reynolds=2300.0,
)

print(f"Max Reynolds number: {report.checks[1].details['max_reynolds']:.0f}")
```

## Post-Embedding Checks

### Connectivity Checks (`post_embedding/connectivity_checks.py`)

Validates fluid connectivity in the embedded structure.

| Check | Description | Pass Criteria |
|-------|-------------|---------------|
| `check_port_accessibility()` | Ports accessible from exterior | Min components found |
| `check_trapped_fluid()` | No isolated fluid regions | Trapped fraction < threshold |
| `check_channel_continuity()` | Channels are connected | Single fluid component |

```python
from validity.post_embedding.connectivity_checks import run_all_connectivity_checks

report = run_all_connectivity_checks(
    mesh,
    pitch=0.1,
    min_port_components=1,
    max_trapped_fraction=0.05,
)

print(f"Fluid components: {report.checks[2].details['num_components']}")
```

### Printability Checks (`post_embedding/printability_checks.py`)

Validates manufacturability constraints.

| Check | Description | Pass Criteria |
|-------|-------------|---------------|
| `check_min_channel_diameter()` | Channels wide enough | All > min diameter |
| `check_wall_thickness()` | Walls thick enough | All > min thickness |
| `check_unsupported_features()` | No excessive overhangs | Angle < max overhang |

```python
from validity.post_embedding.printability_checks import (
    run_all_printability_checks,
    ManufacturingConfig,
)

config = ManufacturingConfig(
    min_channel_diameter=0.5,  # mm
    min_wall_thickness=0.3,    # mm
    max_overhang_angle=45.0,   # degrees
    plate_size=(200, 200, 200),  # mm
    units="mm",
    printer_type="SLA",
)

report = run_all_printability_checks(mesh, config=config)

print(f"Min channel found: {report.checks[0].details['min_channel_diameter']:.2f} mm")
```

### Domain Checks (`post_embedding/domain_checks.py`)

Validates domain-specific constraints.

| Check | Description | Pass Criteria |
|-------|-------------|---------------|
| `check_outlets_open()` | Outlets not covered | Expected outlets found |
| `check_domain_coverage()` | Appropriate void fraction | Within min/max range |

```python
from validity.post_embedding.domain_checks import run_all_domain_checks

report = run_all_domain_checks(
    mesh,
    expected_outlets=2,
    pitch=0.1,
    min_coverage_fraction=0.01,
    max_coverage_fraction=0.5,
)

print(f"Outlets found: {report.checks[0].details['total_openings']}")
print(f"Void fraction: {report.checks[1].details['void_fraction']:.1%}")
```

## Configuration

### ValidationConfig

Centralized configuration for all validation checks:

```python
from validity import ValidationConfig
from validity.post_embedding.printability_checks import ManufacturingConfig

config = ValidationConfig(
    # Voxelization
    voxel_pitch=0.1,
    
    # Murray's law
    murray_gamma=3.0,
    murray_tolerance=0.15,
    
    # Branch order
    max_branch_order=20,
    
    # Collision detection
    min_clearance=0.0,
    max_collisions=0,
    
    # Flow checks
    max_flow_balance_error=0.05,
    max_reynolds=2300.0,
    
    # Connectivity
    min_port_components=1,
    max_trapped_fraction=0.05,
    expected_outlets=2,
    
    # Manufacturing
    manufacturing=ManufacturingConfig(
        min_channel_diameter=0.5,
        min_wall_thickness=0.3,
        max_overhang_angle=45.0,
        plate_size=(200, 200, 200),
        units="mm",
    ),
)
```

### ManufacturingConfig

Manufacturing constraints provided by the user at runtime:

```python
from validity.post_embedding.printability_checks import ManufacturingConfig

config = ManufacturingConfig(
    min_channel_diameter=0.5,   # Minimum printable channel diameter (mm)
    min_wall_thickness=0.3,     # Minimum wall thickness (mm)
    max_overhang_angle=45.0,    # Maximum unsupported overhang angle (degrees)
    plate_size=(200, 200, 200), # Build plate dimensions (mm)
    units="mm",                 # Unit system
    printer_type="SLA",         # Printer type (SLA, FDM, etc.)
)
```

## Validation Reports

### ValidationReport

The main report object returned by orchestrators:

```python
report = run_pre_embedding_validation(mesh_path="structure.stl")

# Access properties
print(f"Passed: {report.passed}")
print(f"Status: {report.status}")  # "ok", "warnings", "fail"
print(f"Stage: {report.stage}")    # "pre_embedding" or "post_embedding"

# Access individual reports
for name, subreport in report.reports.items():
    print(f"\n{name}:")
    for check in subreport.checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"  {check.check_name}: {status}")
        print(f"    {check.message}")

# Summary statistics
print(f"\nTotal checks: {report.summary['total_checks']}")
print(f"Passed: {report.summary['passed_checks']}")
print(f"Failed: {report.summary['failed_checks']}")
print(f"Warnings: {report.summary['total_warnings']}")

# Export to JSON
report.save_json("validation_report.json")

# Convert to dict
report_dict = report.to_dict()
```

## Integration with Generation Library

The validity library is designed to work seamlessly with the generation library:

```python
from generation.api import design_from_spec
from generation.ops import embed_in_domain
from validity import (
    run_pre_embedding_validation,
    run_post_embedding_validation,
    ValidationConfig,
)

# Generate network
network = design_from_spec(spec)

# Pre-embedding validation
pre_report = run_pre_embedding_validation(
    mesh=network.to_mesh(),
    network=network,
)

if not pre_report.passed:
    print("Pre-embedding validation failed!")
    for name, report in pre_report.reports.items():
        for check in report.checks:
            if not check.passed:
                print(f"  {check.check_name}: {check.message}")
else:
    # Embed and validate
    domain_mesh = embed_in_domain(network, domain)
    
    post_report = run_post_embedding_validation(
        mesh=domain_mesh,
        manufacturing_config={
            "min_channel_diameter": 0.5,
            "min_wall_thickness": 0.3,
        }
    )
    
    if post_report.passed:
        print("Structure is valid and manufacturable!")
    else:
        print("Post-embedding validation failed!")
```

## Examples

See the `examples/` directory for working examples:

- `examples/single_agent_organgenerator_v1.ipynb` - Interactive Jupyter notebook that demonstrates the complete workflow including validation integration with the generation library
