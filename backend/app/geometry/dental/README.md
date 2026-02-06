# Dental and Craniofacial Scaffold Generators

This module provides specialized scaffold generators for dental and craniofacial tissue engineering applications.

## Generators

### 1. Dentin-Pulp (dentin_pulp.py)
Tooth-like structures with outer mineralized dentin shell and inner pulp chamber.

**Parameters:**
- `tooth_height` (10-15mm): Total tooth height
- `crown_diameter` (8-12mm): Width of crown
- `root_length` (10-15mm): Length of root portion
- `root_diameter` (4-6mm): Diameter at root tip
- `pulp_chamber_size` (0.3-0.5): Relative size of pulp chamber
- `dentin_thickness` (2-4mm): Thickness of dentin shell
- `resolution` (default 32): Mesh resolution

**Structure:**
- Crown: Dome-shaped (hemisphere section)
- Root: Tapered cylinder (cone frustum)
- Hollow interior: Pulp chamber for tissue integration

**Usage:**
```python
from app.geometry.dental import DentinPulpParams, generate_dentin_pulp

params = DentinPulpParams(
    tooth_height=12.0,
    crown_diameter=10.0,
    root_length=10.0,
    dentin_thickness=3.0
)
manifold, stats = generate_dentin_pulp(params)
```

### 2. Ear Auricle (ear_auricle.py)
Complex curved cartilage framework for ear reconstruction.

**Parameters:**
- `scale_factor` (0.5-1.5): Overall size multiplier
- `thickness` (1-3mm): Cartilage thickness
- `helix_definition` (0.5-1.0): Prominence of helix rim
- `antihelix_depth` (0-1): Depth of antihelix ridge
- `resolution` (default 32): Mesh resolution

**Structure:**
- Main body: Ellipsoid base (60mm H × 30mm W × 15mm D)
- Helix rim: Torus section forming outer curved edge
- Conchal bowl: Depression for ear canal
- Antihelix: Internal ridge feature
- Shell structure: Hollow with specified thickness

**Usage:**
```python
from app.geometry.dental import EarAuricleParams, generate_ear_auricle

params = EarAuricleParams(
    scale_factor=1.0,
    thickness=2.0,
    helix_definition=0.7
)
manifold, stats = generate_ear_auricle(params)
```

### 3. Nasal Septum (nasal_septum.py)
Curved cartilage sheet for nasal reconstruction.

**Parameters:**
- `length` (30-50mm): Anterior-posterior length
- `height` (25-35mm): Superior-inferior height
- `thickness` (2-4mm): Cartilage thickness
- `curvature_radius` (50-100mm): Radius of curvature
- `curve_type` ('single' or 's_curve'): Type of deviation
- `resolution` (default 32): Mesh resolution

**Structure:**
- Base: Rectangular plate with rounded edges
- Curvature: Single curve or S-curve deviation
- Edge treatment: Rounded corners and smooth edges

**Usage:**
```python
from app.geometry.dental import NasalSeptumParams, generate_nasal_septum

params = NasalSeptumParams(
    length=40.0,
    height=30.0,
    thickness=3.0,
    curvature_radius=75.0,
    curve_type='single'
)
manifold, stats = generate_nasal_septum(params)
```

## Implementation Pattern

All generators follow the standard SHED pattern:

1. **Dataclass Parameters**: Type-safe configuration with sensible defaults
2. **Generate Function**: Returns `(manifold, stats)` tuple
3. **Dict Compatibility**: `generate_X_from_dict()` for API integration
4. **Manifold3D**: Uses manifold3d for robust geometry operations
5. **Statistics**: Returns triangle count, volume, and scaffold-specific metadata

## Dependencies

- `manifold3d`: 3D geometry operations
- `numpy`: Numerical computations
- `dataclasses`: Parameter structures
- `..core.batch_union`: Efficient boolean operations (for complex features)

## Design Notes

### Dentin-Pulp
- Crown uses sphere section to create anatomical dome shape
- Root tapers from crown diameter to root tip
- Pulp chamber created by scaling outer shape and subtracting

### Ear Auricle
- Simplified parametric approach rather than anatomical accuracy
- Helix rim uses torus section for natural curved edge
- Shell structure provides cartilage-like thickness
- Antihelix adds internal support structure

### Nasal Septum
- Base plate with comprehensive edge rounding
- Curvature applied through rotation/bending
- S-curve creates natural septal deviation
- Simple geometry suitable for cartilage regeneration

## Integration

These generators are exported through `app.geometry.dental.__init__.py` and will be integrated into:

1. **Backend API**: FastAPI endpoints for scaffold generation
2. **Frontend UI**: Parameter controls and 3D preview
3. **LLM Tools**: Natural language scaffold specification
