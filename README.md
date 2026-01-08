# Agentic Organ Generation

A comprehensive toolkit for AI-driven generation and validation of 3D organ structures with vascular networks. This repository enables large language models (LLMs) and autonomous agents to design, generate, validate, and iteratively refine organ structures for biomedical research, tissue engineering, and additive manufacturing.

## Overview

This toolkit provides a complete pipeline for creating printable organ scaffolds with embedded vascular networks. It is organized into three integrated modules:

| Module | Directory | Purpose |
|--------|-----------|---------|
| **Generation** | `generation/` | Core library for programmatic 3D vascular network generation |
| **Validation** | `validity/` | Two-stage validation system (pre-embedding and post-embedding) |
| **Automation** | `automation/` | LLM integration layer with multi-provider support |

### Primary Outputs

The system produces two primary artifacts:

1. **Domain-with-void scaffold**: A solid domain mesh with the vascular structure carved as negative space, ready for 3D printing
2. **Surface mesh**: A supplementary mesh of the vascular network itself for visualization and analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/ErickGross-19/Agentic-Organ-Generation.git
cd Agentic-Organ-Generation

# Install dependencies
pip install -r requirements.txt

# Or install as an editable package for development
pip install -e .
```

## Quick Start

### Generating a Vascular Network

The following example demonstrates how to generate a liver vascular network using the high-level API:

```python
from generation.api import run_experiment
from generation.specs import DesignSpec, TreeSpec

# Define the design specification
spec = DesignSpec(
    domain_type="ellipsoid",
    domain_params={"a": 50.0, "b": 40.0, "c": 30.0},
    trees=[
        TreeSpec(
            name="arterial",
            inlet_position=(0, 0, 30),
            inlet_radius=2.0,
            target_segments=500,
        )
    ],
)

# Execute the generation pipeline
result = run_experiment(
    spec=spec,
    output_dir="./output",
    manufacturing_config={
        "min_channel_diameter": 0.5,  # mm
        "min_wall_thickness": 0.3,    # mm
        "plate_size": (200, 200, 200), # mm
    },
)

print(f"Domain mesh: {result.domain_mesh_path}")
print(f"Surface mesh: {result.surface_mesh_path}")
```

### Validating a Structure

The validation system provides comprehensive checks at two stages:

```python
from validity import run_pre_embedding_validation, run_post_embedding_validation

# Pre-embedding validation (topology, flow, mesh integrity)
pre_report = run_pre_embedding_validation(mesh_path="structure.stl")
print(f"Pre-embedding status: {pre_report.status}")

# Post-embedding validation (manufacturability, connectivity)
post_report = run_post_embedding_validation(
    mesh_path="domain_with_void.stl",
    manufacturing_config={
        "min_channel_diameter": 0.5,  # mm
        "min_wall_thickness": 0.3,    # mm
    }
)
print(f"Post-embedding status: {post_report.status}")
```

### LLM-Assisted Generation

The automation module enables natural language interaction with the generation system:

```python
from automation import AgentRunner, LLMClient
from automation.task_templates import generate_liver_prompt

# Initialize the LLM client
client = LLMClient(provider="openai")
runner = AgentRunner(client=client)

# Generate using LLM-guided workflow
result = runner.run_task(
    generate_liver_prompt(
        arterial_segments=500,
        venous_segments=500,
        plate_size=(200, 200, 200),
    )
)
```

### Interactive Workflow

The Single Agent Organ Generator V3 provides a guided, interactive workflow for organ structure generation. It features an adaptive rule engine that intelligently determines which questions to ask based on the user's input, along with an agent dialogue system that follows an Interpret → Plan → Ask pattern:

```python
from automation.workflow import run_single_agent_workflow

# Launch the interactive workflow
context = run_single_agent_workflow(
    provider="openai",
    model="gpt-4",
    base_output_dir="./my_projects"
)
```

The workflow progresses through the following stages:

1. **Project Initialization**: Configure project name and output directory
2. **Object Planning**: Define the number and types of structures to generate
3. **Coordinate Convention**: Establish spatial reference frame
4. **Requirements Capture**: Adaptive questioning based on rule engine analysis
5. **Specification Compilation**: Convert requirements to design specification
6. **Generation**: Execute the vascular network generation algorithm
7. **Validation**: Run pre-embedding and post-embedding checks
8. **Iteration**: Refine the design based on validation results
9. **Finalization**: Embed and export production-ready artifacts

#### Adaptive Rule Engine

The rule engine minimizes user interaction by intelligently determining which questions to ask:

| Rule Family | Purpose | Example Triggers |
|-------------|---------|------------------|
| **Completeness** | Ensures required fields are populated | Missing inlet count, undefined domain size |
| **Ambiguity** | Resolves unclear specifications | Spatial terms ("left/right"), vague quantifiers ("dense") |
| **Conflict** | Identifies feasibility issues early | Incompatible constraints, unrealistic parameters |

The system follows a three-tier approach: infer from user text, propose sensible defaults, then ask targeted questions only when necessary.

### Command-Line Interface

The CLI provides convenient access to all major operations:

```bash
# Generate a vascular structure
python -m automation.cli generate --organ liver --segments 500

# Validate an existing structure
python -m automation.cli validate --input structure.stl --stage both

# Launch the interactive workflow
python -m automation.cli workflow

# Start an interactive LLM session
python -m automation.cli interactive
```

## Project Structure

```
Agentic-Organ-Generation/
├── generation/                 # Part A: Generation Library
│   ├── core/                   # Data structures (Network, Node, Segment)
│   ├── ops/                    # Operations (space colonization, growth, etc.)
│   ├── api/                    # High-level API (design_from_spec, run_experiment)
│   ├── specs/                  # Design specifications
│   ├── params/                 # Parameter presets
│   ├── adapters/               # Export adapters (STL, NetworkX)
│   ├── organ_generators/       # Organ-specific generators
│   └── README.md               # Detailed documentation
│
├── validity/                   # Part B: Validity Checking Library
│   ├── pre_embedding/          # Pre-embedding checks
│   │   ├── mesh_checks.py      # Watertightness, manifoldness, etc.
│   │   ├── graph_checks.py     # Murray's law, collisions, etc.
│   │   └── flow_checks.py      # Flow plausibility, Reynolds, etc.
│   ├── post_embedding/         # Post-embedding checks
│   │   ├── connectivity_checks.py  # Port accessibility, trapped fluid
│   │   ├── printability_checks.py  # Channel diameter, wall thickness
│   │   └── domain_checks.py        # Outlet openness, coverage
│   ├── orchestrators.py        # Main entry points
│   └── README.md               # Detailed documentation
│
├── automation/                 # Part C: Automation Scripts
│   ├── llm_client.py           # LLM API client (OpenAI, Anthropic)
│   ├── agent_runner.py         # Agent orchestration
│   ├── task_templates/         # Pre-built prompts
│   ├── cli.py                  # Command-line interface
│   └── README.md               # Detailed documentation
│
├── examples/                   # Example scripts
├── tests/                      # Test suites
├── README.md                   # This file
├── requirements.txt            # Dependencies
└── setup.py                    # Package setup
```

## Generation Library

The generation module provides a comprehensive API for programmatic 3D vascular network generation.

### API Reference

| Function | Description |
|----------|-------------|
| `design_from_spec()` | Generate networks from declarative specifications |
| `run_experiment()` | Execute complete design, validation, and export pipeline |
| `create_network()` | Initialize a new vascular network |
| `space_colonization_step()` | Execute one iteration of the organic growth algorithm |

### Core Algorithms

The generation library implements several biologically-inspired algorithms:

| Algorithm | Purpose |
|-----------|---------|
| **Space Colonization** | Organic growth algorithm that produces realistic vascular morphology |
| **Murray's Law Bifurcation** | Creates physiologically accurate branching points |
| **Collision Detection** | Prevents self-intersections during network growth |
| **Anastomosis** | Connects arterial and venous trees at capillary beds |
| **Domain Embedding** | Carves vascular structures into solid domain volumes |

For detailed API documentation, see [generation/README.md](generation/README.md).

## Validation Library

The validation module implements a comprehensive two-stage quality assurance system.

### Pre-Embedding Validation

Performed before embedding the network into the domain volume:

| Category | Validation Checks |
|----------|-------------------|
| **Mesh Integrity** | Watertightness, manifold geometry, surface quality, degenerate face detection |
| **Network Topology** | Murray's law compliance, branch order consistency, collision detection, self-intersection prevention |
| **Flow Properties** | Flow plausibility, Reynolds number estimation, pressure monotonicity verification |

### Post-Embedding Validation

Performed after carving the network into the domain:

| Category | Validation Checks |
|----------|-------------------|
| **Connectivity** | Port accessibility, trapped fluid detection, channel continuity verification |
| **Manufacturability** | Minimum channel diameter, wall thickness requirements, unsupported feature detection |
| **Domain Coverage** | Outlet openness, tissue perfusion coverage analysis |

For detailed validation documentation, see [validity/README.md](validity/README.md).

## Automation Module

The automation module provides LLM integration for intelligent generation and validation workflows.

### Supported LLM Providers

| Provider | Configuration | Models |
|----------|---------------|--------|
| **OpenAI** | `provider="openai"` | GPT-4, GPT-3.5, GPT-5 |
| **Anthropic** | `provider="anthropic"` | Claude 3 family |
| **xAI/Grok** | `provider="xai"` or `"grok"` | Grok models |
| **Google** | `provider="google"` or `"gemini"` | Gemini family |
| **Mistral** | `provider="mistral"` | Mistral models |
| **Groq** | `provider="groq"` | Groq-hosted models |
| **Local** | `provider="local"` | OpenAI-compatible APIs |

### Task Templates

The module includes pre-built prompt templates for common operations:

| Template | Purpose |
|----------|---------|
| `generate_structure_prompt()` | Generate organ structures from specifications |
| `validate_structure_prompt()` | Validate existing structures |
| `iterate_design_prompt()` | Iteratively refine designs based on feedback |

For detailed automation documentation, see [automation/README.md](automation/README.md).

## Unit System

The library maintains a clear separation between specification, runtime, and output units to ensure consistency and prevent conversion errors.

### Unit Conventions

| Context | Units | Description |
|---------|-------|-------------|
| **Specification** | Meters | All values in spec classes (EllipsoidSpec, BoxSpec, InletSpec) |
| **Runtime** | Meters | Internal domain objects and geometric operations |
| **Output** | Configurable | STL and JSON exports (default: millimeters) |

For example, `EllipsoidSpec(semi_axes=(0.05, 0.045, 0.035))` defines a domain of 50mm × 45mm × 35mm using meter values internally.

### Coordinate Frame

The default coordinate frame follows standard conventions:

| Axis | Direction | Anatomical Reference |
|------|-----------|---------------------|
| **Origin** | Domain center (0, 0, 0) | Geometric centroid |
| **X-axis** | Left-right | Width dimension |
| **Y-axis** | Front-back | Depth dimension |
| **Z-axis** | Bottom-top | Height dimension |

For organ-specific coordinate frames (e.g., anatomical orientation), apply a custom transform using the `transform` parameter in `compile_domain()`.

### Domain Compilation

Use `compile_domain()` to convert spec classes to runtime domain objects:

```python
from generation.specs import EllipsoidSpec, compile_domain

# Create a spec (values in meters)
spec = EllipsoidSpec(center=(0, 0, 0), semi_axes=(0.05, 0.045, 0.035))

# Compile to runtime domain
domain = compile_domain(spec)

# Optionally apply a coordinate transform
from generation.specs import make_translation_transform
transform = make_translation_transform(0.01, 0.02, 0.03)
domain_transformed = compile_domain(spec, transform=transform)
```

### Output Units

```python
from generation.specs import DesignSpec

# Specify output units in the design spec
spec = DesignSpec(
    domain_type="ellipsoid",
    domain_params={"a": 0.05, "b": 0.04, "c": 0.03},  # Internal: meters
    output_units="mm",  # Output STL will be in mm (50mm, 40mm, 30mm)
)

# Or use different output units
spec_meters = DesignSpec(
    domain_type="ellipsoid",
    domain_params={"a": 0.05, "b": 0.04, "c": 0.03},
    output_units="m",  # Output STL will be in meters (0.05, 0.04, 0.03)
)
```

Supported output units: `"m"`, `"mm"`, `"cm"`, `"um"`

## Manufacturing Constraints

Manufacturing constraints are specified at runtime to ensure printability on the target fabrication system:

```python
manufacturing_config = {
    "min_channel_diameter": 0.5,   # mm - minimum printable channel diameter
    "min_wall_thickness": 0.3,     # mm - minimum structural wall thickness
    "max_overhang_angle": 45.0,    # degrees - maximum unsupported overhang angle
    "plate_size": (200, 200, 200), # mm - build volume dimensions (X, Y, Z)
    "units": "mm",                 # output unit system
    "printer_type": "SLA",         # fabrication technology
}
```

These constraints are validated during post-embedding checks to ensure the generated structure can be successfully manufactured.

## Dependencies

### Required

| Package | Purpose |
|---------|---------|
| Python 3.8+ | Runtime environment |
| numpy | Numerical computations |
| scipy | Scientific algorithms |
| trimesh | 3D mesh operations |
| networkx | Graph data structures |

### Optional

| Package | Purpose |
|---------|---------|
| openai | OpenAI API integration |
| anthropic | Anthropic Claude integration |

## Examples

The `examples/` directory contains working demonstrations of the system's capabilities:

### Single Agent Organ Generator V3 Notebook

`examples/single_agent_organgenerator_v2.ipynb` provides an interactive tutorial covering:

| Topic | Description |
|-------|-------------|
| Quick Start | Launch the workflow with `run_single_agent_workflow()` |
| Workflow Control | Fine-grained control via `SingleAgentOrganGeneratorV3` class |
| State Transitions | Programmatic progression using the `step()` method |
| Persistence | Save and restore workflow state across sessions |
| Visualization | Render and inspect generated STL meshes |
| Provider Configuration | Configure different LLM providers |
| Direct Generation | Generate structures without LLM assistance |

## Contributing

Contributions are welcome. Please refer to the individual module documentation for architectural details and coding conventions.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

This project builds upon the [Vascular-Network-Validity](https://github.com/ErickGross-19/Vascular-Network-Validity-) repository for validation algorithms and mesh analysis.
