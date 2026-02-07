# Agentic Organ Generation

A full-stack platform for generating 3D organ scaffolds with embedded vascular networks. The system combines an AI-driven DesignSpec pipeline with 44 parametric geometry generators, a Next.js web frontend, and a FastAPI backend to produce printable scaffolds for tissue engineering, biomedical research, and additive manufacturing.

## Architecture

The platform has three layers:

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Next.js + React + Three.js | Web UI with 3D viewer, chat interface, scaffold controls |
| **Backend** | FastAPI + manifold3d | REST API, 44 geometry generators, LLM agent integration |
| **Core Pipeline** | Python (DesignSpec) | 11-stage pipeline: policy compilation through validity checking and STL export |

### How It Works

1. **Web Chat or Direct Controls**: Users interact through a conversational chat (DesignSpec Agent mode) or direct parameter sliders (Direct Generate mode)
2. **DesignSpec Pipeline**: The AI agent proposes JSON patches to a DesignSpec, which the 11-stage runner executes: compile policies, compile domains, build components, synthesize meshes, union voids, embed in domain, recarve ports, validate, and export
3. **Geometry Generators**: 44 generators across 9 categories produce scaffolds using manifold3d for robust boolean geometry
4. **Output**: STL files (domain with void, surface mesh), network JSON, and run reports

## Quick Start

### Backend

```bash
git clone https://github.com/ErickGross-19/Agentic-Organ-Generation.git
cd Agentic-Organ-Generation
pip install -r requirements.txt

cd backend
pip install -r requirements.txt
python run.py
# API available at http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# Web UI at http://localhost:3000
```

### Environment Variables

Create `backend/.env`:
```bash
LLM_PROVIDER=anthropic          # or openai
ANTHROPIC_API_KEY=sk-ant-...    # for Anthropic
OPENAI_API_KEY=sk-...           # for OpenAI
LLM_MODEL=claude-sonnet-4-20250514       # optional model override
```

## Scaffold Types (44 Generators)

Organized into 9 categories:

| Category | Generators |
|----------|-----------|
| **Original** | Vascular Network, Porous Disc, Lattice, Primitive |
| **Advanced Lattice / TPMS** | Gyroid, Schwarz-P, Octet Truss, Voronoi, Honeycomb |
| **Skeletal** | Trabecular Bone, Osteochondral, Haversian Bone, Articular Cartilage, Intervertebral Disc, Meniscus, Tendon/Ligament |
| **Organ** | Hepatic Lobule, Cardiac Patch, Kidney Tubule, Lung Alveoli, Pancreatic Islet, Liver Sinusoid |
| **Soft Tissue** | Multilayer Skin, Skeletal Muscle, Cornea, Adipose |
| **Tubular** | Blood Vessel, Nerve Conduit, Trachea, Spinal Cord, Bladder, Simple Conduit |
| **Dental / Craniofacial** | Dentin-Pulp, Ear Auricle, Nasal Septum |
| **Microfluidic** | Organ-on-Chip, Gradient Scaffold, Perfusable Network |
| **Vascular Backends** | Space Colonization, Bifurcating Tree (Top-Down Scaffold) |

## DesignSpec Pipeline

The canonical workflow uses JSON DesignSpec files executed by the `DesignSpecRunner`. The runner orchestrates 11 stages:

```
compile_policies -> compile_domains -> component_ports -> component_build ->
component_mesh -> union_voids -> mesh_domain -> embed -> port_recarve ->
validity -> export
```

### Programmatic Usage

```python
from designspec import DesignSpec, DesignSpecRunner

spec = DesignSpec.from_json("examples/designspec/golden_example_v1.json")
runner = DesignSpecRunner(spec=spec, output_dir="./output")
result = runner.run()

print(f"Success: {result.success}")
print(f"Stages completed: {result.stages_completed}")
```

### Partial Execution

```python
from designspec.plan import ExecutionPlan

plan = ExecutionPlan(run_until="union_voids")
runner = DesignSpecRunner(spec=spec, plan=plan, output_dir="./output")
result = runner.run()
```

Available stages (in order): `compile_policies`, `compile_domains`, `component_ports`, `component_build`, `component_mesh`, `union_voids`, `mesh_domain`, `embed`, `port_recarve`, `validity`, `export`.

You can also use `run_only` to execute specific stages or `components_subset` to process only certain components.

### ManifoldBackend Integration

The 44 MorphoStruct geometry generators are accessible through the DesignSpec pipeline via the `manifold_generator` build type:

```python
from generation.backends.manifold_backend import ManifoldBackend

backend = ManifoldBackend()
mesh, stats = backend.generate(
    generator_type="hepatic_lobule",
    params={"num_lobules": 7, "lobule_radius": 1.5},
    convert_units=True,
)
```

In a DesignSpec JSON, use the `manifold_generator` build type:

```json
{
  "components": {
    "my_scaffold": {
      "build": {
        "type": "manifold_generator",
        "generator_type": "hepatic_lobule",
        "generator_params": { "num_lobules": 7 }
      }
    }
  }
}
```

## Web Interface

### Two Operating Modes

**Direct Generate Mode**: Select a scaffold type from the grouped selector, adjust parameters with sliders, and generate immediately. The 3D viewer displays the result with rotate/zoom/pan controls.

**DesignSpec Agent Mode**: Chat with the AI agent to iteratively design organ specifications. The agent proposes JSON patches (RFC 6902), which you approve or reject. Approved patches trigger automatic compilation. Run the pipeline through the chat interface and view results in the 3D viewer.

### Key Frontend Components

- **ScaffoldTypeSelector**: Grouped dropdown with all 44 types across 9 categories
- **ChatPanel**: Conversational interface with patch approval cards and pipeline progress
- **SpecViewer**: Displays current DesignSpec JSON with expand/collapse and copy-to-clipboard
- **ParameterPanel**: Dynamic controls per scaffold type
- **3D Viewer**: Three.js-based STL viewer with wireframe/solid toggle

## Policies

All pipeline behavior is controlled through policies in `aog_policies/`. Policies are specified in the JSON spec under the `policies` section and compiled at runtime:

```json
{
  "policies": {
    "resolution": { "min_pitch": 0.0001, "max_pitch": 0.001, "max_voxel_budget": 1000000 },
    "embedding": { "recarve_ports": true, "shell_thickness": 0.001 },
    "channels": { "length_mode": "explicit", "length": 0.005 },
    "manifold_generator": { "generator_type": "gyroid", "convert_units": true }
  }
}
```

Key policy types:
- **ResolutionPolicy**: Voxel pitch and budget for mesh operations
- **EmbeddingPolicy**: Domain embedding and port recarving
- **ChannelPolicy**: Primitive channel generation
- **GrowthPolicy**: Space colonization parameters
- **MeshSynthesisPolicy**: Network-to-mesh conversion
- **ManifoldGeneratorPolicy**: Manifold generator type, params, and unit conversion

Components can override top-level policies using `policy_overrides` for per-component customization.

## Outputs and Reproducibility

Every run produces deterministic outputs when a seed is specified:

```json
{
  "meta": { "name": "my_design", "seed": 42, "input_units": "m" }
}
```

The runner computes a `spec_hash` from the normalized specification, ensuring identical specs produce identical hashes. The `run_report.json` captures spec hash, stage-by-stage timing and success status, requested vs effective policies, and validation results.

## LLM Agent System

The DesignSpec LLM Agent (`automation/designspec_llm/`) provides:

- **Structured directives**: LLM outputs JSON with patches, run requests, and questions
- **Self-correction**: Auto-analyzes run failures and proposes fixes using 30+ error patterns
- **Session memory**: Tracks decisions, error resolutions, and user preferences across turns
- **Task context**: Persistent goal tracking across conversation turns
- **Artifact awareness**: Knows about mesh statistics, network data, and run history

The agent is accessible through:
- The web chat interface (FastAPI `/api/designspec/*` endpoints via the `DesignSpecBridge`)
- The legacy desktop GUI (`gui/`)
- Programmatic usage via `DesignSpecWorkflow`

## Testing

```bash
# Backend integration tests (71 tests)
pytest tests/test_manifold_backend.py tests/test_designspec_bridge.py tests/test_prompt_scaffold_types.py -v

# Full readiness gate
pytest -q tests/contract tests/unit tests/integration tests/regression tests/quality

# Frontend build verification
cd frontend && npm run build
```

## Project Structure

```
Agentic-Organ-Generation/
├── frontend/                 # Next.js web application
│   ├── app/                  # Pages (generator, dashboard, settings, etc.)
│   ├── components/           # React components (chat, controls, viewer, designspec)
│   └── lib/                  # API clients, stores (Zustand), types, utilities
│
├── backend/                  # FastAPI backend
│   ├── app/
│   │   ├── api/              # REST endpoints (chat, scaffolds, designspec, auth)
│   │   ├── geometry/         # 44 generators organized by category
│   │   │   ├── lattice/      # Gyroid, Schwarz-P, Octet Truss, Voronoi, Honeycomb
│   │   │   ├── skeletal/     # Trabecular Bone, Osteochondral, Haversian, etc.
│   │   │   ├── organ/        # Hepatic Lobule, Cardiac Patch, Kidney Tubule, etc.
│   │   │   ├── soft_tissue/  # Multilayer Skin, Skeletal Muscle, Cornea, Adipose
│   │   │   ├── tubular/      # Blood Vessel, Nerve Conduit, Trachea, etc.
│   │   │   ├── dental/       # Dentin-Pulp, Ear Auricle, Nasal Septum
│   │   │   └── microfluidic/ # Organ-on-Chip, Gradient Scaffold, Perfusable Network
│   │   ├── llm/              # Multi-provider LLM abstraction (Anthropic, OpenAI)
│   │   └── vascular/         # Space Colonization, Bifurcating Tree wrappers
│   └── tests/                # Backend-specific tests
│
├── designspec/               # DesignSpec pipeline (spec, runner, plan, context)
├── aog_policies/             # Policy surface (resolution, embedding, channels, manifold, etc.)
├── generation/               # Core generation library
│   ├── core/                 # VascularNetwork, Node, Segment, Domain
│   ├── ops/                  # Space colonization, growth, collision, embedding
│   ├── backends/             # Generation backends + ManifoldBackend (44 generators)
│   └── adapters/             # STL export, NetworkX, reports
│
├── automation/               # LLM agent system
│   ├── designspec_llm/       # DesignSpec LLM Agent (directive, context, prompts, errors)
│   └── workflows/            # DesignSpec workflow manager
│
├── validity/                 # Pre- and post-embedding validation
├── gui/                      # Legacy tkinter GUI (superseded by frontend/)
├── agentic_scaffolding/      # Standalone scaffold generation tools
├── tests/                    # Test suites (contract, unit, integration, regression, quality)
├── examples/                 # DesignSpec JSON examples and stress tests
├── docs/                     # Additional documentation
└── scripts/                  # Utility scripts
```

## Legacy Notes

- Older APIs in `generation/specs/` (Python dataclasses) and `generation/api/design.py` are deprecated in favor of JSON DesignSpec files
- The tkinter GUI (`gui/`) is superseded by the Next.js frontend (`frontend/`) but remains functional
- The `agentic_scaffolding/` module provides standalone scaffold generation tools independent of the main pipeline

## License

See LICENSE file for details.
