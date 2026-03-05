# Generation Library

3D scaffold and vascular network generation. Provides multiple backends for the DesignSpec pipeline including vascular tree growth, space colonization, programmatic paths, and 44 manifold3d-based geometry generators from MorphoStruct.

## Backends

| Backend | Module | Purpose |
|---------|--------|---------|
| `scaffold_topdown` | `backends/scaffold_topdown_backend.py` | Recursive top-down bifurcating tree with collision avoidance |
| `space_colonization` | `backends/space_colonization_backend.py` | Attractor-based organic vascular growth |
| `programmatic` | `backends/programmatic_backend.py` | DSL-based generation with A* pathfinding |
| `manifold` | `backends/manifold_backend.py` | 44 MorphoStruct geometry generators via ManifoldBackend |
| `kary_tree` | `backends/kary_tree_backend.py` | K-ary tree (deprecated, use scaffold_topdown) |
| `cco_hybrid` | `backends/cco_hybrid_backend.py` | CCO optimization (not finished, blocked) |

## Directory Structure

```
generation/
├── __init__.py
├── core/
│   ├── network.py          # VascularNetwork, Node, Segment
│   ├── domain.py           # EllipsoidDomain, BoxDomain, CylinderDomain
│   └── types.py            # Type definitions
├── ops/
│   ├── space_colonization.py  # Space colonization algorithm (single-step, KD-tree)
│   ├── growth.py           # Branch growth operations
│   ├── bifurcate.py        # Bifurcation with Murray's law
│   ├── collision.py        # Collision detection/avoidance
│   ├── anastomosis.py      # Anastomosis creation
│   └── embedding.py        # Domain embedding (boolean subtraction)
├── backends/
│   ├── __init__.py         # Backend registry
│   ├── scaffold_topdown_backend.py
│   ├── space_colonization_backend.py
│   ├── programmatic_backend.py
│   ├── manifold_backend.py # ManifoldBackend (44 generators)
│   ├── kary_tree_backend.py
│   └── cco_hybrid_backend.py
├── spatial/
│   └── grid_index.py       # DynamicSpatialIndex for fast neighbor queries
├── api/
│   ├── design.py           # design_from_spec()
│   └── experiment.py       # run_experiment()
├── specs/
│   ├── design_spec.py      # DesignSpec, EllipsoidSpec, BoxSpec
│   ├── compile.py          # compile_domain()
│   └── tree_spec.py        # TreeSpec for individual trees
├── params/
│   └── presets.py          # Parameter presets (liver_arterial_dense, etc.)
├── adapters/
│   ├── mesh_adapter.py     # STL export
│   ├── networkx_adapter.py # NetworkX conversion
│   ├── liver_adapter.py    # Liver VascularTree conversion
│   └── report_adapter.py   # Report generation
├── organ_generators/
│   └── liver.py            # Liver vascular generator
└── utils/
    └── geometry.py         # Geometric utilities
```

## ManifoldBackend (44 Generators)

The `ManifoldBackend` wraps all MorphoStruct geometry generators for use in the DesignSpec pipeline. Each generator produces manifold3d geometry converted to trimesh.

### Generator Categories

| Category | Count | Generators |
|----------|-------|------------|
| Lattice/TPMS | 5 | Gyroid, Schwarz-P, Octet Truss, Voronoi, Honeycomb |
| Skeletal | 7 | Trabecular Bone, Osteochondral, Haversian Bone, Articular Cartilage, Intervertebral Disc, Meniscus, Tendon/Ligament |
| Organ | 6 | Hepatic Lobule, Cardiac Patch, Kidney Tubule, Lung Alveoli, Pancreatic Islet, Liver Sinusoid |
| Soft Tissue | 4 | Multilayer Skin, Skeletal Muscle, Cornea, Adipose |
| Tubular | 6 | Blood Vessel, Nerve Conduit, Trachea, Spinal Cord, Bladder, Simple Conduit |
| Dental | 3 | Dentin-Pulp, Ear Auricle, Nasal Septum |
| Microfluidic | 3 | Organ-on-Chip, Gradient Scaffold, Perfusable Network |
| Vascular | 2 | Space Colonization Network, Bifurcating Tree |
| Original | 8 | Vascular Network, Porous Disc, Primitive, BioScaffold, Lattice, etc. |

### Usage via DesignSpec

```json
{
  "components": {
    "scaffold_1": {
      "build": {
        "build_type": "manifold_generator",
        "generator_type": "hepatic_lobule",
        "generator_params": {
          "lobule_radius": 0.5,
          "lobule_height": 0.3,
          "central_vein_radius": 0.05
        }
      }
    }
  },
  "policies": {
    "manifold_generator": {
      "generator_type": "hepatic_lobule",
      "generator_params": {}
    }
  }
}
```

### Programmatic Usage

```python
from generation.backends.manifold_backend import ManifoldBackend

backend = ManifoldBackend()
mesh, stats = backend.generate(
    generator_type="hepatic_lobule",
    params={"lobule_radius": 0.5, "lobule_height": 0.3}
)
```

## Vascular Backends

### scaffold_topdown (Preferred for Trees)

Recursive top-down bifurcating tree with online collision avoidance, post-pass resolution, multi-inlet forest mode.

### space_colonization

Attractor-based organic growth with single-step architecture, KD-tree caching, stall detection, and multi-inlet support (blended/partitioned modes).

### programmatic

DSL-based generation for explicit path definitions and waypoint-based routing with A* voxel pathfinding.

## Legacy API

The older `design_from_spec()` and `run_experiment()` entry points in `generation/api/` still work for direct generation without the DesignSpec pipeline:

```python
from generation.api import design_from_spec
from generation.specs import DesignSpec, TreeSpec

spec = DesignSpec(
    domain_type="ellipsoid",
    domain_params={"a": 50.0, "b": 40.0, "c": 30.0},
    trees=[TreeSpec(name="arterial", inlet_position=(0, 0, 30), inlet_radius=2.0, target_segments=500)],
)
network = design_from_spec(spec)
```

## Unit System

All geometric values are in **meters** internally. Specs accept `input_units` (default "mm") and convert on load. Export uses configurable output units via `UnitContext`.

| Stage | Units |
|-------|-------|
| Spec input | Configurable (`input_units`) |
| Internal | Meters |
| Export | Configurable (`output_units`, default mm) |
