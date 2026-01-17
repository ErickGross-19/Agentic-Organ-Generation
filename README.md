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

### Generating a Vascular Network (Recommended: JSON DesignSpec)

The recommended way to generate vascular networks is using JSON DesignSpec files with the DesignSpecRunner:

```python
from designspec import DesignSpec, DesignSpecRunner
from designspec.plan import ExecutionPlan

# Load a JSON design specification
spec = DesignSpec.from_json("examples/designspec/golden_example_v1.json")

# Create a runner and execute the full pipeline
runner = DesignSpecRunner(spec=spec, output_dir="./output")
result = runner.run()

print(f"Success: {result.success}")
print(f"Stages completed: {result.stages_completed}")

# Or run until a specific stage (partial execution)
plan = ExecutionPlan(run_until="union_voids")
runner = DesignSpecRunner(spec=spec, plan=plan, output_dir="./output")
partial_result = runner.run()
```

The JSON DesignSpec format provides full control over policies, domains, and components. See `examples/designspec/golden_example_v1.json` for a complete example.

### Running the Readiness Gate Tests

```bash
# Run all readiness gate tests
pytest -q tests/contract tests/unit tests/integration tests/regression tests/quality
```

### Legacy API (Deprecated)

The older Python dataclass-based API (`generation.specs`) is deprecated but still available for backward compatibility:

```python
# DEPRECATED - use JSON DesignSpec instead
from generation.api import run_experiment
from generation.specs import DesignSpec, TreeSpec  # Triggers deprecation warning

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

result = run_experiment(
    spec=spec,
    output_dir="./output",
    manufacturing_config={
        "min_channel_diameter": 0.5,  # mm
        "min_wall_thickness": 0.3,    # mm
        "plate_size": (200, 200, 200), # mm
    },
)
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

The repository provides two workflow implementations for interactive organ structure generation:

#### V5 Goal-Driven Controller (Recommended)

The Single Agent Organ Generator V5 is a goal-driven controller that feels like a confident engineer. It replaces the state-machine approach with a goal-driven architecture:

```python
from automation.single_agent_organ_generation.v5 import (
    SingleAgentOrganGeneratorV5,
    ControllerConfig,
    CLIIOAdapter,
)

# Initialize the V5 controller
io_adapter = CLIIOAdapter()
config = ControllerConfig(
    max_iterations=1000,
    max_safe_fixes_per_run=10,
    auto_select_plan_if_confident=True,
    verbose=True,
)

controller = SingleAgentOrganGeneratorV5(
    io_adapter=io_adapter,
    config=config,
)

# Run the workflow
controller.run()
```

V5 features include:

| Feature | Description |
|---------|-------------|
| **Goal-Driven Progress** | Progress measured by goal satisfaction, not state transitions |
| **WorldModel** | Single source of truth for facts, approvals, artifacts, and history |
| **Intelligent Questioning** | Automatically generates questions for missing required fields |
| **Safe Fix Policy** | Applies safe fixes one at a time with user confirmation for non-safe changes |
| **Approval Tracking** | Always asks before generation and postprocess; tracks denied approvals |
| **Interrupt Support** | Supports undo, backtracking, and revisiting previous decisions |
| **Spam Prevention** | Only summarizes spec when meaningful changes occur |

V5 goals progress through the following sequence:

1. **spec_minimum_complete**: All required fields for the chosen topology are filled
2. **spec_compiled**: Design spec has been compiled to executable form
3. **pregen_verified**: Feasibility and schema checks passed
4. **generation_approved**: User has approved generation
5. **generation_done**: Vascular network has been generated
6. **postprocess_approved**: User has approved postprocessing
7. **postprocess_done**: Embedding/voxelization/repair/export complete
8. **validation_passed**: All validation checks passed
9. **outputs_packaged**: Final deliverables are packaged

#### V3 Legacy Workflow

The Single Agent Organ Generator V3 provides a state-machine-based workflow with an adaptive rule engine:

```python
from automation.workflow import run_single_agent_workflow

# Launch the interactive workflow
context = run_single_agent_workflow(
    provider="openai",
    model="gpt-4",
    base_output_dir="./my_projects"
)
```

The V3 workflow progresses through the following stages:

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

### Graphical User Interface

The GUI provides a visual interface for workflow execution with integrated STL visualization.

#### Launching the GUI

```bash
# Launch the GUI
python main.py

# Or as a module
python -m gui
```

#### Getting Started with the GUI

Follow these steps to generate your first organ structure using the GUI:

**Step 1: Configure the LLM Agent**

Click the "Agent Config" button in the toolbar to open the configuration dialog:

| Setting | Description | Example |
|---------|-------------|---------|
| **Provider** | LLM provider to use | OpenAI, Anthropic, Google, Mistral, xAI, Groq, or Local |
| **API Key** | Your API key for the selected provider | `sk-...` (stored securely) |
| **Model** | Model name | `gpt-4`, `claude-3-opus`, etc. |
| **Temperature** | Sampling temperature (0.0-1.0) | `0.7` (default) |
| **Max Tokens** | Maximum response length | `4096` (default) |

For local providers (e.g., Ollama), set the API Base URL instead of an API key.

**Step 2: Select a Workflow**

Click "Select Workflow" in the toolbar or use File > New Workflow (Ctrl+N). Currently available:

- **Single Agent Organ Generator**: Interactive workflow with topology-first questioning. Best for guided organ structure design.

**Step 3: Choose Output Directory**

After clicking "Start", you'll be prompted to select an output directory where generated files will be saved.

**Step 4: Interact with the Agent**

The agent will guide you through the design process in the chat panel:

1. **Project Description**: Describe what you want to build (e.g., "I want to create a liver scaffold for perfusion testing, approximately 20x30x40 mm")
2. **Topology Selection**: Choose the vascular topology (tree, dual_trees, path, backbone, loop)
3. **Domain Configuration**: Specify domain type and dimensions
4. **Inlet/Outlet Placement**: Define where fluid enters and exits
5. **Plan Selection**: Review and select from generated plans
6. **Generation Approval**: Approve the generation step
7. **Postprocessing**: Approve mesh embedding and export

**Step 5: View Results**

Generated STL files appear in the STL Viewer panel on the right. You can:

- Rotate the view by dragging
- Zoom with the scroll wheel
- Load additional STL files via File > Load STL
- Export the current view via File > Export View

#### GUI Features

| Feature | Description |
|---------|-------------|
| **Three-Panel Layout** | Chat (interaction), Output (logs), STL Viewer (3D visualization) |
| **Secure API Key Storage** | API keys are encrypted and stored locally |
| **Real-time Progress** | Status bar shows current workflow state |
| **Undo Support** | Backtrack decisions during the workflow |
| **Run History** | View previous generation runs and their outputs |
| **Verification Reports** | Inspect validation results for generated structures |

#### Troubleshooting

| Issue | Solution |
|-------|----------|
| "Agent not configured" | Click "Agent Config" and enter your API key |
| "Please start a workflow first" | Click "Select Workflow" then "Start" |
| STL not loading | Ensure the file path is valid and the file is not corrupted |
| Workflow stuck | Check the Output panel for error messages; try stopping and restarting |

For detailed GUI documentation, see [gui/README.md](gui/README.md).

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

## Current Architecture: DesignSpec + DesignSpecRunner

The recommended way to use this system is through the **DesignSpec + DesignSpecRunner** architecture, which provides a JSON-based specification format and policy-controlled execution:

```python
from designspec import DesignSpec, DesignSpecRunner
from aog_policies import ResolutionPolicy, EmbeddingPolicy

# Load a JSON design specification
spec = DesignSpec.from_json("my_design.json")

# Create a runner with custom policies
runner = DesignSpecRunner(
    spec=spec,
    resolution_policy=ResolutionPolicy(min_diameter_um=20, max_voxel_budget=1_000_000),
    embedding_policy=EmbeddingPolicy(recarve_ports=True),
)

# Run the full pipeline
result = runner.run()

# Or run until a specific stage
partial_result = runner.run_until("union_voids")
```

The runner orchestrates all geometry generation, embedding, and validity checking through `aog_policies/`, ensuring consistent behavior and JSON-serializable reports.

**Note**: The older `generation/specs/` module (Python dataclasses) is deprecated. New code should use JSON DesignSpec files loaded via `designspec.DesignSpec`. The `gui/`, `automation/`, and `examples/` directories have not yet been migrated to the new architecture.

## Project Structure

```
Agentic-Organ-Generation/
├── designspec/                 # NEW: JSON-based design specification system
│   ├── spec.py                 # DesignSpec class for loading JSON specs
│   ├── runner.py               # DesignSpecRunner for orchestrating execution
│   ├── ast/                    # AST evaluation for spec expressions
│   └── reports/                # JSON-serializable run reports
│
├── aog_policies/               # NEW: Policy surface for controlling runner behavior
│   ├── resolution.py           # Resolution and voxel budget policies
│   ├── embedding.py            # Embedding and port recarve policies
│   └── validity.py             # Validity check policies
│
├── legacy/                     # Deprecated code moved here for reference
│   └── generation/cfd/         # CFD simulation modules (not used by runner)
│
├── generation/                 # Part A: Generation Library
│   ├── core/                   # Data structures (Network, Node, Segment)
│   ├── ops/                    # Operations (space colonization, growth, etc.)
│   ├── api/                    # High-level API (design_from_spec, run_experiment)
│   ├── specs/                  # Design specifications (DEPRECATED - use designspec/)
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
│   ├── single_agent_organ_generation/
│   │   └── v5/                 # V5 Goal-Driven Controller
│   │       ├── controller.py   # Main agent loop
│   │       ├── world_model.py  # Single source of truth
│   │       ├── goals.py        # Goal definitions and tracking
│   │       ├── policies.py     # Safe fix and approval policies
│   │       ├── plan_synthesizer.py  # Object-specific plan generation
│   │       └── io/             # IO adapters (CLI, GUI)
│   └── README.md               # Detailed documentation
│
├── gui/                        # Part D: Graphical User Interface
│   ├── main_window.py          # Primary application window
│   ├── workflow_manager.py     # Workflow orchestration
│   ├── stl_viewer.py           # 3D STL visualization
│   ├── agent_config.py         # LLM configuration panel
│   ├── security.py             # Encrypted API key storage
│   └── README.md               # Detailed documentation
│
├── examples/                   # Example scripts
├── tests/                      # Test suites
├── main.py                     # GUI entry point
├── build.spec                  # PyInstaller configuration
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

### Single Agent Organ Generator Notebooks

`examples/single_agent_organgenerator_v2.ipynb` provides an interactive tutorial covering:

| Topic | Description |
|-------|-------------|
| Quick Start | Launch the workflow with V5 goal-driven controller or V3 state machine |
| V5 Controller | Goal-driven workflow with WorldModel, capabilities, and policies |
| V3 Workflow | State-machine-based workflow with adaptive rule engine |
| Persistence | Save and restore workflow state across sessions |
| Visualization | Render and inspect generated STL meshes |
| Provider Configuration | Configure different LLM providers |
| Direct Generation | Generate structures without LLM assistance |

## Running Tests

The repository includes a comprehensive test suite organized into the following categories:

```bash
# Fast suite (CI-safe, runs in seconds)
pytest -q tests/contract tests/unit tests/regression tests/quality

# Full readiness gate (includes integration tests)
pytest -q tests/contract tests/unit tests/integration tests/regression tests/quality

# Legacy tests only (not run by default)
pytest -q -m legacy tests/legacy_disabled
```

| Test Category | Purpose |
|---------------|---------|
| `tests/contract/` | Policy ownership, JSON serializability, schema versioning |
| `tests/unit/` | Domains, AST eval, resolution budgeting, pathfinding, primitives |
| `tests/integration/` | DesignSpecRunner end-to-end tests with golden fixtures |
| `tests/regression/` | Previously fixed bug regressions |
| `tests/quality/` | Code hygiene for runner-critical code |
| `tests/legacy_disabled/` | Quarantined legacy tests, not run by default |

## Contributing

Contributions are welcome. Please refer to the individual module documentation for architectural details and coding conventions.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

This project builds upon the [Vascular-Network-Validity](https://github.com/ErickGross-19/Vascular-Network-Validity-) repository for validation algorithms and mesh analysis.
