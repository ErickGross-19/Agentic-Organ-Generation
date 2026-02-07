# Backend Status

Generation backends available in the AOG system. These are used by the DesignSpec pipeline's `component_build` stage to produce vascular networks and scaffold geometry.

## DesignSpec Build Types

| Build Type | Backend | Status | Purpose |
|------------|---------|--------|---------|
| `scaffold_topdown` | `scaffold_topdown_backend.py` | Active | Recursive top-down bifurcating tree generation with collision avoidance |
| `space_colonization` | `space_colonization_backend.py` | Active | Attractor-based organic growth, multi-inlet forests |
| `programmatic` | `programmatic_backend.py` | Active | DSL-based generation with A* pathfinding and waypoints |
| `primitive_channels` | (inline in runner) | Active | Simple channel geometry (taper, fang-hook profiles) |
| `manifold_generator` | `manifold_backend.py` | Active | 44 MorphoStruct geometry generators via ManifoldBackend |

## Vascular Backends (Detail)

### scaffold_topdown (Preferred for Trees)

Recursive top-down tree generation with online collision avoidance, post-pass collision resolution, multiple inlets (forest mode), and configurable cone angles, jitter, and curvature. Used via `GrowthPolicy(backend="scaffold_topdown")`.

### space_colonization

Attractor-based organic growth for dense vascular networks. Supports multi-inlet blended and partitioned modes, stall detection, KD-tree caching, and interleaving strategies. See [space_colonization_refactor.md](space_colonization_refactor.md) for the single-step architecture.

### programmatic

DSL-based generation for explicit path definitions, waypoint-based routing, and custom network topologies with A* voxel pathfinding.

## ManifoldBackend (44 Generators)

The `ManifoldBackend` dispatches to 44 geometry generators from the MorphoStruct integration. Each generator produces manifold3d geometry that is converted to trimesh for the DesignSpec pipeline. Generators are organized by category in `backend/app/geometry/`:

- **Lattice/TPMS**: Gyroid, Schwarz-P, Octet Truss, Voronoi, Honeycomb
- **Skeletal**: Trabecular Bone, Osteochondral, Haversian Bone, Articular Cartilage, Intervertebral Disc, Meniscus, Tendon/Ligament
- **Organ**: Hepatic Lobule, Cardiac Patch, Kidney Tubule, Lung Alveoli, Pancreatic Islet, Liver Sinusoid
- **Soft Tissue**: Multilayer Skin, Skeletal Muscle, Cornea, Adipose
- **Tubular**: Blood Vessel, Nerve Conduit, Trachea, Spinal Cord, Bladder, Simple Conduit
- **Dental/Craniofacial**: Dentin-Pulp, Ear Auricle, Nasal Septum
- **Microfluidic**: Organ-on-Chip, Gradient Scaffold, Perfusable Network

Used via `ManifoldGeneratorPolicy(generator_type="hepatic_lobule", generator_params={...})` in DesignSpec.

## Deprecated / Blocked

| Backend | Status | Notes |
|---------|--------|-------|
| `kary_tree` | Deprecated | Use `scaffold_topdown` instead |
| `cco_hybrid` | Blocked | Not finished, raises error on use |
| NLP optimization | Blocked | Not finished, raises error on use |
