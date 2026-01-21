# Agentic Scaffolding

LLM-powered and interactive tools for generating vascular network scaffolds for biomedical applications. These tools create 3D printable STL files of branching channel networks that can be used for tissue engineering, organ scaffolds, and microfluidic devices.

## Overview

This module provides three approaches to scaffold generation:

| Tool | Interface | Best For |
|------|-----------|----------|
| `scaffold_web_collision.py` | Web UI with sliders | Full control, exploration, fine-tuning |
| `scaffold_agent_web.py` | Conversational web | Natural language design, iterative refinement |
| `scaffold_agent.py` | Command line | Batch generation, scripting, automation |

---

## scaffold_web_collision.py

**The primary scaffold generation tool** - a comprehensive web application for generating vascular network scaffolds with real-time preview and collision detection.

### Quick Start

```bash
python scaffold_web_collision.py
# Opens at http://localhost:8080
```

### Architecture

The scaffold consists of:
- **Outer ring**: Cylindrical wall (4.875mm outer radius, 4.575mm inner radius)
- **Body**: Solid cylinder that gets channels subtracted from it
- **Channels**: Branching vascular network created by boolean subtraction

```
     ┌─────────────────┐  ← Inlet ports on top
     │  ○   ○   ○   ○  │
     │   ╲ ╱ ╲ ╱ ╲ ╱   │  ← Branches spread and split
     │    ╳   ╳   ╳    │
     │   ╱ ╲ ╱ ╲ ╱ ╲   │  ← Multiple levels of branching
     │  ↓   ↓   ↓   ↓  │  ← Tips (optional: point down)
     └─────────────────┘
```

### Collision Detection System

The collision detection prevents branches from intersecting, ensuring valid 3D printable geometry.

#### How It Works

1. **Spatial Hash Grid** (`OptimizedSpatialGrid`)
   - Divides 3D space into cells for O(1) average neighbor lookup
   - Adaptive cell sizing based on average branch radius
   - Eliminates O(n²) pairwise distance checks

2. **Vectorized Branch Tracking** (`VectorizedBranchTracker`)
   - Stores branch segments in contiguous NumPy arrays
   - Pre-allocated buffers for zero-copy operations
   - Batch distance calculations using vectorized math

3. **JIT-Compiled Distance Functions**
   - `_point_to_segment_dist_sq_jit`: Point-to-line-segment distance
   - `_segment_to_segment_dist_jit`: Segment-to-segment distance
   - `_batch_segment_distances_jit`: Parallel batch computation
   - Compiled to machine code via Numba for 10-100x speedup

4. **Collision Resolution**
   ```
   When placing a new branch:
   1. Check if proposed position collides with existing branches
   2. If collision detected:
      a. Try rotating around parent junction (±22.5°, ±45°, ±67.5°, etc.)
      b. If no safe angle found, reduce spread distance
      c. Try random angles at reduced spread
   3. Record final branch path for future collision checks
   ```

#### Geometry Caching (`GeometryCache`)

Frequently-used primitives are cached to avoid recreation:
- Spheres quantized to 0.02mm radius increments
- Cylinders quantized to 0.05mm length increments
- Typical cache hit rate: 60-80%

### Branch Generation Algorithm

```python
def branch(x, y, z, radius, angle, remaining_levels):
    # 1. Calculate target position
    z_step = (z - bottom) / (remaining_levels + 1)
    target_x = x + spread * cos(angle)
    target_y = y + spread * sin(angle)
    target_z = z - z_step

    # 2. Apply randomness (unless deterministic mode)
    if not deterministic:
        z_step *= random(0.65, 1.35)
        angle += random(-0.4, 0.4)
        radius *= random(0.75, 1.25)

    # 3. Find collision-free position
    safe_position = find_safe_position(start, target, radius)

    # 4. Create curved geometry (Bezier curve)
    geometry = make_curved_branch(start, safe_position, ...)

    # 5. Recurse for child branches
    for i in range(splits):
        child_angle = compute_child_angle(i, splits)
        branch(safe_x, safe_y, safe_z, child_radius, child_angle, remaining_levels - 1)
```

### UI Controls Reference

#### Toolbar

| Button | Function |
|--------|----------|
| **Fast** | Toggle fast preview mode (skips boolean subtraction) |
| **Deterministic** | Grid-aligned channels with no randomness |
| **Tips Down** | Terminal branches curve to point straight down |
| **Auto** | Automatically rebuild when parameters change |
| **PREVIEW** | Quick rebuild with current settings |
| **FULL BUILD** | Complete boolean operations (slower, accurate) |
| **RESEED** | Generate new random variation with same parameters |
| **Export Scaffold** | Save scaffold STL (high resolution) |
| **Export Channels** | Save channels-only STL |

#### Parameters Panel

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Inlets** | 1-25 | 4 | Number of inlet ports. Uses Fibonacci spiral for optimal packing when >9 |
| **Levels** | 0-8 | 2 | Branching depth. 0 = straight channels, higher = more branching |
| **Splits** | 1-6 | 2 | Child branches per junction. 2 = binary tree, 3+ = more complex |
| **Spread** | 0.1-0.8 | 0.35 | Horizontal distance per branch level (mm) |
| **Ratio** | 0.5-0.95 | 0.79 | Child/parent radius ratio. 0.79 = Murray's law (optimal flow) |
| **Cone Angle** | 10-180° | 60° | Angular spread of child branches |
| **Curvature** | 0-1 | 0.3 | Branch curvature. 0 = straight, 1 = very curved |
| **Even Spread** | toggle | on | Even 360° distribution vs directional cone |

#### Randomness Panel

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| **Radius Variation** | 0-100% | 25% | Random thickness variation |
| **Flip Chance** | 0-0.5 | 0.30 | Probability of direction flip |
| **Z Variation** | 0-0.5 | 0.35 | Vertical step randomness |
| **Angle Variation** | 0-0.5 | 0.40 | Branch angle randomness |
| **Collision Buffer** | 0-0.3mm | 0.08 | Extra spacing between branches |

#### View Modes

| Mode | Description |
|------|-------------|
| **Normal** | Scaffold body (blue) with channel overlay (red) |
| **Inverted** | Channels only - useful for visualizing flow paths |
| **Section** | Cross-section view through the middle |

### Performance Modes

| Mode | Speed | Quality | Use Case |
|------|-------|---------|----------|
| **Fast + Deterministic** | ~0.1s | Preview | Quick exploration |
| **Fast + Organic** | ~0.5s | Preview | Design iteration |
| **Full + Deterministic** | ~2s | Export | Grid-like designs |
| **Full + Organic** | ~5-15s | Export | Final production |

### Technical Specifications

**Default Scaffold Dimensions:**
- Outer radius: 4.875 mm
- Inner radius: 4.575 mm
- Total height: 2.0 mm
- Scaffold body height: 1.92 mm
- Wall thickness: 0.3 mm
- Inlet radius: 0.35 mm

**Resolution Settings:**
- Preview: 6 circular segments
- Export: 16 circular segments

**Performance Optimizations:**
- Spatial hash grid: O(1) average collision lookup
- Numba JIT: 10-100x speedup on distance calculations
- Geometry cache: 60-80% hit rate reduces object creation
- Tree reduction: O(log n) boolean union depth
- Optional CuPy GPU acceleration

---

## scaffold_agent_web.py

A conversational web interface that interprets natural language to generate scaffolds.

### Quick Start

```bash
python scaffold_agent_web.py
# Opens at http://localhost:8081
```

### Supported Commands

**Exact values:**
```
"3 inlets"           → inlets = 3
"4 layers"           → levels = 4
"2mm tall"           → height = 2.0
"seed 12345"         → seed = 12345
```

**Relative modifiers:**
```
"thicker"            → ratio += 0.08, inlet_radius += 0.08
"thinner"            → ratio -= 0.08, inlet_radius -= 0.08
"taller"             → height += 0.5
"shorter"            → height -= 0.5
"broader"            → spread += 0.15
"narrower"           → spread -= 0.1
"curvier"            → curvature += 0.2
"straighter"         → curvature -= 0.2
```

**Style keywords:**
```
"organic"            → curvature = 0.7, deterministic = false
"uniform"            → deterministic = true
"tips down"          → tips_down = true
"dense"              → inlets += 5, levels += 2
"simple"             → inlets = 4, levels = 2
```

---

## scaffold_agent.py

Command-line tool for generating scaffold STL files.

### Usage

```bash
# Natural language (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY="your-key"
python scaffold_agent.py "dense organic network with 9 inlets"

# Direct parameters
python scaffold_agent.py --params '{"inlets": 9, "levels": 4, "tips_down": true}'

# Interactive mode
python scaffold_agent.py --interactive

# Custom output directory
python scaffold_agent.py "simple scaffold" -o ./output
```

### Output Files

```
output/
├── scaffold_9in_4lvl_tips_s42.stl          # Complete scaffold
└── scaffold_9in_4lvl_tips_s42_channels.stl  # Channels only
```

---

## Installation

```bash
# Core dependencies
pip install numpy manifold3d pyvista

# Web interface
pip install trame trame-vuetify trame-vtk

# LLM features (optional)
pip install anthropic

# Performance optimizations (optional but recommended)
pip install numba        # 10-100x speedup on collision detection
pip install cupy-cuda12x # GPU acceleration (requires CUDA)
```

## Parameter Reference

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `inlets` | int | 1-25 | 4 | Number of inlet ports |
| `levels` | int | 0-8 | 2 | Branching depth |
| `splits` | int | 1-6 | 2 | Branches per junction |
| `spread` | float | 0.1-0.8 | 0.35 | Horizontal spread (mm) |
| `ratio` | float | 0.5-0.95 | 0.79 | Child/parent radius ratio |
| `curvature` | float | 0-1 | 0.3 | Branch curvature |
| `cone_angle` | float | 10-180 | 60 | Child branch spread (degrees) |
| `tips_down` | bool | - | false | Tips point straight down |
| `deterministic` | bool | - | false | Disable randomness |
| `seed` | int | any | 42 | Random seed |

---

## License

See repository LICENSE file.
