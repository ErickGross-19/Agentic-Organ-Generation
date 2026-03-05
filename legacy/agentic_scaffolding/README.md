# Agentic Scaffolding (Standalone Tools)

Standalone scaffold generation tools using manifold3d boolean geometry. These are independent from the main DesignSpec pipeline and provide direct web-based and command-line interfaces for generating 3D printable scaffold STL files.

> **Note**: For DesignSpec-based generation (the primary workflow), see the main [README.md](../README.md). The tools below are standalone utilities for quick scaffold prototyping.

## Tools

| Tool | Interface | Purpose |
|------|-----------|---------|
| `scaffold_web_collision.py` | Web UI (localhost:8080) | Full-control scaffold generation with sliders, collision detection, 3D preview |
| `scaffold_agent_web.py` | Conversational web (localhost:8081) | Natural language scaffold design |
| `scaffold_agent.py` | Command line | Batch generation and scripting |

## scaffold_web_collision.py (Primary)

Web application for generating vascular network scaffolds with real-time preview and collision detection.

```bash
python scaffold_web_collision.py
# Opens at http://localhost:8080
```

### Architecture

The scaffold consists of an outer ring (cylindrical wall), a solid body, and branching channels created by boolean subtraction. The collision detection system uses:
- **Spatial Hash Grid** for O(1) average neighbor lookup
- **Vectorized Branch Tracking** with NumPy arrays
- **Numba JIT** compiled distance functions (10-100x speedup)
- **Geometry Caching** (60-80% hit rate)

### Key Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Inlets | 1-25 | 4 | Number of inlet ports (Fibonacci spiral for >9) |
| Levels | 0-8 | 2 | Branching depth |
| Splits | 1-6 | 2 | Branches per junction |
| Spread | 0.1-0.8 | 0.35 | Horizontal distance per level (mm) |
| Ratio | 0.5-0.95 | 0.79 | Child/parent radius ratio (0.79 = Murray's law) |
| Cone Angle | 10-180 | 60 | Angular spread of child branches |
| Curvature | 0-1 | 0.3 | Branch curvature |

### Default Dimensions
- Outer radius: 4.875mm, Inner radius: 4.575mm
- Height: 2.0mm, Wall thickness: 0.3mm, Inlet radius: 0.35mm

## scaffold_agent_web.py

Conversational interface that interprets natural language to generate scaffolds.

```bash
python scaffold_agent_web.py
# Opens at http://localhost:8081
```

Supports exact values ("3 inlets", "2mm tall"), relative modifiers ("thicker", "broader"), and style keywords ("organic", "dense", "uniform").

## scaffold_agent.py

Command-line tool for scaffold STL generation.

```bash
export ANTHROPIC_API_KEY="your-key"
python scaffold_agent.py "dense organic network with 9 inlets"
python scaffold_agent.py --params '{"inlets": 9, "levels": 4}'
python scaffold_agent.py --interactive
```

## Installation

```bash
conda env create -f environment.yml
conda activate agentic-scaffolding
```

Or via pip:
```bash
pip install numpy>=1.21 manifold3d>=2.3 pyvista>=0.42 trame>=3.0 trame-vuetify>=2.4 trame-vtk>=2.6
pip install numba>=0.57  # optional, 10-100x speedup
pip install anthropic>=0.18  # optional, for natural language
```
