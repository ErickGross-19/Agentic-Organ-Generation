# Malaria Venule Insert Examples

This document describes how to run the malaria venule insert DesignSpec examples and what outputs to expect.

## Overview

The malaria venule examples demonstrate different vascular network generation strategies for creating microfluidic inserts used in malaria research. All examples use a cylindrical domain (R=5mm, H=2mm) with a ridge on the top face.

## Examples

### malaria_venule_bifurcating_tree.json

Generates a bifurcating tree network using the `scaffold_topdown` backend with 5 inlets (center + 4 cardinal positions). Features online collision avoidance and postpass collision cleanup.

```bash
python scripts/run_designspec_example.py \
    --spec examples/designspec/malaria_venule_bifurcating_tree.json \
    --out ./out/malaria_bifurcating
```

Key parameters: `splits=2`, `levels=6`, collision avoidance enabled.

### malaria_venule_space_colonization.json

Dense organic network using the `space_colonization` backend with 5 inlets targeting 1024 terminals. Uses attractor-based growth for biologically-inspired branching patterns.

```bash
python scripts/run_designspec_example.py \
    --spec examples/designspec/malaria_venule_space_colonization.json \
    --out ./out/malaria_space_col
```

Key parameters: 150,000 attraction points, `influence_radius=0.9mm`, `kill_radius=0.25mm`.

### malaria_venule_cco.json

Optimized tree structure using the `cco_hybrid` backend with Murray's Law optimization. Single inlet with 256 target terminals.

```bash
python scripts/run_designspec_example.py \
    --spec examples/designspec/malaria_venule_cco.json \
    --out ./out/malaria_cco
```

Key parameters: `murray_exponent=3.0`, `num_outlets=32`.

### malaria_venule_fang_hook_channels.json

9 radial-out fang-hook channels using the `primitive_channels` build type. Channels curve outward with a hook shape.

```bash
python scripts/run_designspec_example.py \
    --spec examples/designspec/malaria_venule_fang_hook_channels.json \
    --out ./out/malaria_fang_hook
```

Key parameters: `profile=fang_hook`, `hook_depth=0.7mm`, `hook_angle_deg=90`.

### malaria_venule_vertical_channels.json

9 straight tapered channels using the `primitive_channels` build type. Simple vertical channels that stop 1mm before the bottom.

```bash
python scripts/run_designspec_example.py \
    --spec examples/designspec/malaria_venule_vertical_channels.json \
    --out ./out/malaria_vertical
```

Key parameters: `profile=taper`, `taper_factor=0.2`.

## Running Examples

### Basic Usage

Run any example with the runner script:

```bash
python scripts/run_designspec_example.py --spec <path_to_json> --out <output_dir>
```

### Run Until Specific Stage

To run only through a specific stage (useful for debugging):

```bash
python scripts/run_designspec_example.py \
    --spec examples/designspec/malaria_venule_bifurcating_tree.json \
    --out ./out/test \
    --run-until component_build:bifurcating_tree_5in
```

Common stages: `compile_policies`, `compile_domains`, `component_build:<id>`, `union_voids`, `embed_voids`, `validity`.

## Output Artifacts

After running, check the output directory for:

| File | Description |
|------|-------------|
| `run_report.json` | Execution summary with stage reports and timing |
| `artifacts/<component>_network.json` | Network graph data (nodes, segments) |
| `artifacts/<component>_void.stl` | Void mesh for the component |
| `domain_with_void.stl` | Final embedded scaffold (3D-printable) |
| `void_union.stl` | Union of all void meshes |

## Configuration Notes

### Merge Policy

All malaria examples use `keep_largest_component=false` to preserve multi-inlet forest structures without discarding disconnected trees.

### Unit System

Most malaria examples use meters (`input_units: "m"`) except `malaria_venule_cco.json` which uses millimeters (`input_units: "mm"`). All values are normalized to meters internally.

### Multi-Inlet Support

The bifurcating tree and space colonization examples support multiple inlets, generating a forest of trees (one per inlet) without merging. This is the intended behavior for malaria inserts.

## Testing

Run the malaria venule test suite:

```bash
# Regression tests
pytest tests/regression/test_malaria_venule_examples.py -v

# Integration tests
pytest tests/integration/test_malaria_venule_examples_runner.py -v
```
