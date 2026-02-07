# Implementation Summary

This document summarizes the major components and integration points of the Agentic Organ Generation platform.

## System Architecture

The platform is a full-stack application combining a Python-based organ generation pipeline with a web interface:

```
frontend/ (Next.js)  <-->  backend/ (FastAPI)  <-->  Core Pipeline (DesignSpec + generation/)
                                                       |
                                                  44 Geometry Generators
                                                  (ManifoldBackend)
```

## Core Pipeline: DesignSpec

The `designspec/` package defines a JSON-driven specification format executed by the `DesignSpecRunner` through 11 stages:

1. **compile_policies** - Compile policy dicts to `aog_policies` objects
2. **compile_domains** - Compile domain dicts to runtime Domain objects
3. **component_ports** - Resolve port positions for each component
4. **component_build** - Generate network/mesh for each component (dispatches to backends)
5. **component_mesh** - Convert network to void mesh
6. **union_voids** - Union all component void meshes
7. **mesh_domain** - Generate domain mesh
8. **embed** - Boolean-subtract unified void from domain
9. **port_recarve** - Recarve ports if enabled
10. **validity** - Run pre/post-embedding validation checks
11. **export** - Write STL, JSON, and report files

Build types supported: `scaffold_topdown`, `space_colonization`, `programmatic`, `primitive_channels`, `manifold_generator`.

## Generation Backends

The `generation/backends/` directory contains:

| Backend | Purpose |
|---------|---------|
| `scaffold_topdown_backend.py` | Recursive top-down bifurcating tree generation |
| `space_colonization_backend.py` | Attractor-based organic growth with multi-inlet support |
| `programmatic_backend.py` | DSL-based generation with A* pathfinding |
| `manifold_backend.py` | Dispatches to 44 MorphoStruct geometry generators |
| `kary_tree_backend.py` | Deprecated, use scaffold_topdown instead |
| `cco_hybrid_backend.py` | Not finished, blocked from use |

## ManifoldBackend and 44 Geometry Generators

The `ManifoldBackend` wraps 44 geometry generators from the MorphoStruct integration. These use manifold3d for boolean geometry and are organized by category in `backend/app/geometry/`:

| Category | Location | Generators |
|----------|----------|-----------|
| Lattice / TPMS | `geometry/lattice/` | Gyroid, Schwarz-P, Octet Truss, Voronoi, Honeycomb |
| Skeletal | `geometry/skeletal/` | Trabecular Bone, Osteochondral, Haversian Bone, Articular Cartilage, Intervertebral Disc, Meniscus, Tendon/Ligament |
| Organ | `geometry/organ/` | Hepatic Lobule, Cardiac Patch, Kidney Tubule, Lung Alveoli, Pancreatic Islet, Liver Sinusoid |
| Soft Tissue | `geometry/soft_tissue/` | Multilayer Skin, Skeletal Muscle, Cornea, Adipose |
| Tubular | `geometry/tubular/` | Blood Vessel, Nerve Conduit, Trachea, Spinal Cord, Bladder, Simple Conduit |
| Dental / Craniofacial | `geometry/dental/` | Dentin-Pulp, Ear Auricle, Nasal Septum |
| Microfluidic | `geometry/microfluidic/` | Organ-on-Chip, Gradient Scaffold, Perfusable Network |

Each generator follows a consistent pattern: dataclass parameters with defaults, a `generate_*()` function returning `(manifold, stats)`, and a `generate_*_from_dict()` function for API integration.

## LLM Agent System

### DesignSpec LLM Agent (`automation/designspec_llm/`)

The agent provides a conversational interface for creating and editing DesignSpec files:

- **Structured directives**: LLM outputs JSON containing patches (RFC 6902), run requests, and questions
- **Error taxonomy**: 30+ error patterns with regex matching, confidence scores, and fix templates (`error_taxonomy.py`)
- **Patch generator**: Produces targeted JSON patches from structured errors (`patch_generator.py`)
- **Run analyzer**: Auto-analyzes pipeline failures and proposes fixes (`run_analyzer.py`)
- **Context builder**: Builds compact prompts with artifact summaries, task context, and session memory (`context_builder.py`)
- **Task context**: Persistent goal tracking across conversation turns (`task_context.py`)
- **Session memory**: Records decisions, error resolutions, and user preferences (`session_memory.py`)

### Workflow Integration

The `DesignSpecWorkflow` (`automation/workflows/designspec_workflow.py`) integrates the agent with both UIs:
- **Web interface**: The FastAPI `DesignSpecBridge` (`backend/app/api/designspec.py`) wraps the workflow for HTTP access
- **Desktop GUI**: The `DesignSpecWorkflowManager` (`gui/designspec_workflow_manager.py`) wraps it for tkinter

### Other Agent Implementations

- **V5 Goal-Driven Controller**: WorldModel-based agent with capabilities and policies (`automation/single_agent_organ_generation/v5/`)
- **V3/V4 State Machine**: Earlier workflow implementations, now legacy

## Frontend (Next.js)

The `frontend/` directory contains the MorphoStruct web application:

- **Pages**: Generator (main scaffold creation), Dashboard, Library, Settings, Auth
- **Components**: ScaffoldTypeSelector (grouped 44-type dropdown), ChatPanel (with patch approval), SpecViewer, ParameterPanel, 3D Viewer
- **State**: Zustand stores for scaffold parameters, chat state, auth
- **Two modes**: Direct Generate (parameter sliders) and DesignSpec Agent (conversational chat)

## Backend (FastAPI)

The `backend/app/` directory contains the REST API:

- **`api/scaffolds.py`**: Scaffold generation endpoints
- **`api/chat.py`**: Chat endpoints for the ScaffoldAgent
- **`api/designspec.py`**: DesignSpec bridge endpoints (`/api/designspec/message`, `/api/designspec/approve-patch`, `/api/designspec/run`)
- **`api/auth.py`**: Authentication endpoints
- **`llm/`**: Multi-provider LLM abstraction (Anthropic, OpenAI) with `ScaffoldAgent`
- **`vascular/`**: Space Colonization and Bifurcating Tree wrappers calling into root-level `generation/`

## Policy System

All pipeline behavior is controlled through `aog_policies/`:

| Policy | File | Purpose |
|--------|------|---------|
| `ResolutionPolicy` | `resolution.py` | Voxel pitch, budget, min diameter |
| `EmbeddingPolicy` | `generation.py` | Shell thickness, port recarving |
| `ChannelPolicy` | `generation.py` | Primitive channel dimensions |
| `GrowthPolicy` | `generation.py` | Backend selection, step sizes, radii |
| `MeshSynthesisPolicy` | `generation.py` | Network-to-mesh conversion params |
| `ManifoldGeneratorPolicy` | `manifold.py` | Generator type, params, unit conversion |
| `ValidationPolicy` | `validity.py` | Which validity checks to run |
| `CollisionPolicy` | `collision.py` | Collision detection thresholds |
| `CompositionPolicy` | `composition.py` | Void union and merge settings |

## Validation System

The `validity/` package provides pre- and post-embedding checks:

- **Pre-embedding**: Mesh quality (watertight, manifold), network topology (Murray's law, branch order), hemodynamic flow (mass conservation, Reynolds number)
- **Post-embedding**: Connectivity (port accessibility, trapped fluid), printability (min channel diameter, wall thickness), domain coverage

The `validity/runner.py` orchestrates all checks and is invoked by the DesignSpec pipeline at the `validity` stage.

## Unit Normalization

All geometric values are normalized to meters internally. Input units are specified via `meta.input_units` in DesignSpec JSON. See [NORMALIZATION.md](NORMALIZATION.md) for the complete list of unitful fields and conversion rules.

## Testing

The test suite is organized into categories:

| Category | Location | Purpose |
|----------|----------|---------|
| Contract | `tests/contract/` | Policy ownership, JSON serializability, report schema |
| Unit | `tests/unit/` | Domains, resolution budgeting, port placement, tube sweep |
| Integration | `tests/integration/` | DesignSpecRunner end-to-end, partial execution |
| Regression | `tests/regression/` | Previously fixed bug regressions |
| Quality | `tests/quality/` | Code hygiene for runner-critical code |
| Backend | `tests/test_manifold_backend.py`, `tests/test_designspec_bridge.py`, `tests/test_prompt_scaffold_types.py` | ManifoldBackend, DesignSpec API bridge, prompt/scaffold type consistency |

See [TEST_AUDIT.md](TEST_AUDIT.md) for the complete test taxonomy and migration checklist.
