# Generation Library (Part A)

The Generation Library provides LLM/agent-callable functions for 3D organ structure generation. It includes operations for space colonization, vascular network growth, bifurcation, collision avoidance, anastomosis creation, and embedding structures into domains.

## Primary Output

The primary output of the generation library is a **domain-with-void scaffold** (a solid domain mesh with the vascular structure carved out as negative space) plus a **supplementary surface mesh** of the vascular network itself.

## Directory Structure

```
generation/
├── __init__.py          # Main entry point with high-level exports
├── core/                # Core data structures
│   ├── network.py       # VascularNetwork, Node, Segment classes
│   ├── domain.py        # Domain representation
│   └── types.py         # Type definitions
├── ops/                 # Low-level operations
│   ├── space_colonization.py  # Organic growth algorithm
│   ├── growth.py        # Branch growth operations
│   ├── bifurcate.py     # Bifurcation operations
│   ├── collision.py     # Collision detection/avoidance
│   ├── anastomosis.py   # Anastomosis creation
│   └── embedding.py     # Domain embedding operations
├── api/                 # High-level LLM-friendly interface
│   ├── design.py        # design_from_spec()
│   └── experiment.py    # run_experiment()
├── specs/               # Design specifications
│   ├── design_spec.py   # DesignSpec dataclass
│   └── tree_spec.py     # TreeSpec for individual trees
├── params/              # Parameter presets
│   └── presets.py       # liver_arterial_dense, etc.
├── adapters/            # Export adapters
│   ├── stl_export.py    # STL export
│   ├── networkx_adapter.py  # NetworkX conversion
│   └── reports.py       # Report generation
├── organ_generators/    # Organ-specific generators
│   └── liver.py         # Liver vascular generator
└── utils/               # Utility functions
    └── geometry.py      # Geometric utilities
```

## Main Entry Points

### High-Level API

#### `design_from_spec(spec: DesignSpec) -> VascularNetwork`

Build a vascular network from a declarative JSON specification. This is the recommended entry point for LLM-driven generation.

```python
from generation.api import design_from_spec
from generation.specs import DesignSpec, TreeSpec

spec = DesignSpec(
    domain_type="ellipsoid",
    domain_params={"a": 50.0, "b": 40.0, "c": 30.0},
    trees=[
        TreeSpec(
            name="arterial",
            inlet_position=(0, 0, 30),
            inlet_radius=2.0,
            target_segments=500,
            growth_method="space_colonization",
        )
    ],
)

network = design_from_spec(spec)
```

#### `run_experiment(spec: DesignSpec, output_dir: str) -> ExperimentResult`

One-call orchestration of design → evaluate → export. Handles the complete workflow including validation and file export.

```python
from generation.api import run_experiment
from generation.specs import DesignSpec

result = run_experiment(
    spec=my_spec,
    output_dir="./output",
    export_formats=["stl", "json"],
    run_validation=True,
)

print(f"Domain mesh: {result.domain_mesh_path}")
print(f"Surface mesh: {result.surface_mesh_path}")
print(f"Validation: {result.validation_report}")
```

### Low-Level Operations

For fine-grained control, use the operations directly:

```python
from generation.core import VascularNetwork, Node
from generation.ops import (
    create_network,
    add_inlet,
    space_colonization_step,
    grow_branch,
    bifurcate,
    check_collision,
    create_anastomosis,
    embed_in_domain,
)

# Create network
network = create_network()

# Add inlet
inlet = add_inlet(network, position=(0, 0, 30), radius=2.0)

# Grow using space colonization
attractors = generate_attractors(domain, n=1000)
for _ in range(100):
    network = space_colonization_step(network, attractors)

# Embed in domain
domain_mesh = embed_in_domain(network, domain_type="box", domain_size=(100, 100, 100))
```

## Organ-Specific Generators

### Liver Generator

```python
from generation.organ_generators.liver import (
    generate_liver_vasculature,
    LiverVascularConfig,
)

config = LiverVascularConfig(
    arterial_segments=500,
    venous_segments=500,
    domain_size=(100, 80, 60),
    inlet_radius=2.0,
    murray_gamma=3.0,
)

result = generate_liver_vasculature(config)
print(f"Arterial tree: {len(result.arterial_network.segments)} segments")
print(f"Venous tree: {len(result.venous_network.segments)} segments")
```

## Design Specifications

The `DesignSpec` dataclass provides a declarative way to specify network designs:

```python
from generation.specs import DesignSpec, TreeSpec

spec = DesignSpec(
    # Domain configuration
    domain_type="ellipsoid",  # or "box"
    domain_params={
        "a": 50.0,  # semi-axis x
        "b": 40.0,  # semi-axis y
        "c": 30.0,  # semi-axis z
    },
    
    # Tree specifications
    trees=[
        TreeSpec(
            name="arterial",
            inlet_position=(0, 0, 30),
            inlet_radius=2.0,
            target_segments=500,
            growth_method="space_colonization",
            growth_params={
                "attraction_distance": 5.0,
                "kill_distance": 1.0,
                "step_size": 1.0,
            },
            murray_gamma=3.0,
        ),
    ],
    
    # Output configuration
    output_format="domain_with_void",
    units="mm",
)
```

## Parameter Presets

Pre-configured parameter sets for common use cases:

```python
from generation.params import (
    liver_arterial_dense,
    liver_arterial_sparse,
    kidney_cortical,
    generic_vascular,
)

# Use a preset
spec = DesignSpec(
    domain_type="ellipsoid",
    domain_params={"a": 50, "b": 40, "c": 30},
    trees=[liver_arterial_dense],
)
```

## Export Formats

### STL Export

```python
from generation.adapters import export_stl

# Export surface mesh
export_stl(network, "vascular_surface.stl", mesh_type="surface")

# Export domain with void
export_stl(network, "domain_with_void.stl", mesh_type="domain_void", domain=domain)
```

### NetworkX Export

```python
from generation.adapters import to_networkx

G = to_networkx(network)
# Now use NetworkX for graph analysis
```

## Manufacturing Constraints

Manufacturing constraints are provided by the user at runtime and should be passed to the generation functions:

```python
from generation.api import run_experiment

result = run_experiment(
    spec=my_spec,
    output_dir="./output",
    manufacturing_config={
        "min_channel_diameter": 0.5,  # mm
        "min_wall_thickness": 0.3,    # mm
        "plate_size": (200, 200, 200),  # mm
        "units": "mm",
    },
)
```

## Integration with Validity Library

The generation library is designed to work seamlessly with the validity library:

```python
from generation.api import design_from_spec
from validity import run_pre_embedding_validation, run_post_embedding_validation

# Generate network
network = design_from_spec(spec)

# Validate before embedding
pre_report = run_pre_embedding_validation(
    mesh=network.to_mesh(),
    network=network,
)

if pre_report.passed:
    # Embed and validate
    domain_mesh = embed_in_domain(network, domain)
    post_report = run_post_embedding_validation(mesh=domain_mesh)
```

## Examples

See the `examples/` directory for complete working examples:

- `examples/generate_liver.py` - Generate a liver vascular network
- `examples/custom_organ.py` - Generate a custom organ structure
- `examples/iterative_design.py` - Iteratively improve a design
