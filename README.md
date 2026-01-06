# Agentic Organ Generation

A comprehensive toolkit for LLM/agent-driven generation and validation of 3D organ structures with vascular networks. This repository enables AI agents to design, generate, validate, and iterate on organ structures for biomedical research, tissue engineering, and manufacturing.

## Overview

The toolkit is organized into three main parts:

| Part | Module | Description |
|------|--------|-------------|
| **A** | `generation/` | Library of LLM-callable functions for 3D structure generation |
| **B** | `validity/` | Validation checks for pre-embedding and post-embedding stages |
| **C** | `automation/` | Scripts for LLM API calls and agent behavior |

### Primary Output

The primary output is a **domain-with-void scaffold** (solid domain mesh with vascular structure as negative space) plus a **supplementary surface mesh** of the vascular network.

## Installation

```bash
# Clone the repository
git clone https://github.com/ErickGross-19/Agentic-Organ-Generation.git
cd Agentic-Organ-Generation

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Quick Start

### Generate a Liver Vascular Network

```python
from generation.api import run_experiment
from generation.specs import DesignSpec, TreeSpec

# Define specification
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

# Run experiment
result = run_experiment(
    spec=spec,
    output_dir="./output",
    manufacturing_config={
        "min_channel_diameter": 0.5,
        "min_wall_thickness": 0.3,
        "plate_size": (200, 200, 200),
    },
)

print(f"Domain mesh: {result.domain_mesh_path}")
print(f"Surface mesh: {result.surface_mesh_path}")
```

### Validate a Structure

```python
from validity import run_pre_embedding_validation, run_post_embedding_validation

# Pre-embedding validation
pre_report = run_pre_embedding_validation(mesh_path="structure.stl")
print(f"Pre-embedding: {pre_report.status}")

# Post-embedding validation
post_report = run_post_embedding_validation(
    mesh_path="domain_with_void.stl",
    manufacturing_config={
        "min_channel_diameter": 0.5,
        "min_wall_thickness": 0.3,
    }
)
print(f"Post-embedding: {post_report.status}")
```

### Use LLM Agent for Generation

```python
from automation import AgentRunner, LLMClient
from automation.task_templates import generate_liver_prompt

# Initialize
client = LLMClient(provider="openai")
runner = AgentRunner(client=client)

# Generate with LLM guidance
result = runner.run_task(
    generate_liver_prompt(
        arterial_segments=500,
        venous_segments=500,
        plate_size=(200, 200, 200),
    )
)
```

### Command-Line Interface

```bash
# Generate a structure
python -m automation.cli generate --organ liver --segments 500

# Validate a structure
python -m automation.cli validate --input structure.stl --stage both

# Interactive session
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

## Part A: Generation Library

The generation library provides LLM-callable functions for 3D organ structure generation.

### Main Entry Points

| Function | Description |
|----------|-------------|
| `design_from_spec()` | Build networks from declarative specifications |
| `run_experiment()` | Complete design → validate → export workflow |
| `create_network()` | Low-level network initialization |
| `space_colonization_step()` | Organic growth algorithm |

### Key Operations

- **Space Colonization**: Organic growth algorithm for realistic vascular networks
- **Bifurcation**: Create branching points following Murray's law
- **Collision Avoidance**: Detect and resolve segment collisions
- **Anastomosis**: Connect arterial and venous trees
- **Embedding**: Carve vascular structures into domain volumes

See [generation/README.md](generation/README.md) for detailed documentation.

## Part B: Validity Checking Library

The validity library provides comprehensive validation checks.

### Pre-Embedding Checks (Before Embedding)

| Category | Checks |
|----------|--------|
| Mesh | Watertightness, manifoldness, surface quality, degenerate faces |
| Graph | Murray's law, branch order, collisions, self-intersections |
| Flow | Flow plausibility, Reynolds number, pressure monotonicity |

### Post-Embedding Checks (After Embedding)

| Category | Checks |
|----------|--------|
| Connectivity | Port accessibility, trapped fluid, channel continuity |
| Printability | Min channel diameter, wall thickness, unsupported features |
| Domain | Outlet openness, domain coverage |

See [validity/README.md](validity/README.md) for detailed documentation.

## Part C: Automation Scripts

The automation module enables LLM-driven generation and validation.

### Supported LLM Providers

- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude 3)
- Local models (OpenAI-compatible APIs)

### Task Templates

- `generate_structure_prompt()` - Generate organ structures
- `validate_structure_prompt()` - Validate structures
- `iterate_design_prompt()` - Iteratively improve designs

See [automation/README.md](automation/README.md) for detailed documentation.

## Manufacturing Constraints

Manufacturing constraints are provided by the user at runtime:

```python
manufacturing_config = {
    "min_channel_diameter": 0.5,   # mm - minimum printable channel
    "min_wall_thickness": 0.3,     # mm - minimum wall thickness
    "max_overhang_angle": 45.0,    # degrees - maximum unsupported overhang
    "plate_size": (200, 200, 200), # mm - build plate dimensions
    "units": "mm",                 # unit system
    "printer_type": "SLA",         # printer type
}
```

## Dependencies

- Python 3.8+
- numpy
- scipy
- trimesh
- networkx
- openai (optional, for LLM automation)
- anthropic (optional, for Claude support)

## Examples

See the `examples/` directory for complete working examples:

- `examples/generate_liver.py` - Generate a liver vascular network
- `examples/validate_structure.py` - Validate a structure
- `examples/llm_generation.py` - Use LLM for generation
- `examples/iterative_design.py` - Iteratively improve a design

## Contributing

Contributions are welcome! Please see the individual module READMEs for architecture details.

## License

MIT License

## Acknowledgments

This project builds upon the [Vascular-Network-Validity](https://github.com/ErickGross-19/Vascular-Network-Validity-) repository.
