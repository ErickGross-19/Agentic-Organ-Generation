# Organ-Specific Scaffold Generators

This directory contains 6 specialized scaffold generators for organ tissue engineering applications.

## Implemented Generators

### 1. **Hepatic Lobule** (`hepatic_lobule.py`)
Hexagonal prism structures mimicking liver lobule architecture.

**Parameters:**
- `num_lobules`: Number of hexagonal units (default: 1)
- `lobule_radius`: Radius of each hexagon (default: 1.5mm)
- `lobule_height`: Height of prism (default: 3.0mm)
- `wall_thickness`: Thickness of walls (default: 0.1mm)
- `central_vein_radius`: Radius of central vein (default: 0.15mm)
- `portal_vein_radius`: Radius of portal veins at corners (default: 0.12mm)
- `show_sinusoids`: Add radial sinusoidal channels (default: False)
- `sinusoid_radius`: Radius of sinusoid channels (default: 0.025mm)
- `resolution`: Mesh resolution (default: 8)

**Features:**
- Honeycomb arrangement of hexagonal lobules
- Central vein through each lobule center
- Portal triads at hexagon corners
- Optional sinusoidal channels connecting periphery to center

---

### 2. **Cardiac Patch** (`cardiac_patch.py`)
Aligned microfibrous scaffolds for cardiomyocyte culture.

**Parameters:**
- `fiber_spacing`: Distance between fibers (default: 300µm, range: 100-500µm)
- `fiber_diameter`: Diameter of each fiber (default: 100µm, range: 50-200µm)
- `patch_size`: Overall dimensions (x, y, z) (default: 10×10×1mm)
- `layer_count`: Number of fiber layers (default: 3)
- `alignment_angle`: Angular offset between layers (default: 0°)
- `resolution`: Fiber mesh resolution (default: 8)

**Features:**
- Parallel fibers in grid pattern
- Multi-layer architecture with angular offset
- Controlled fiber spacing and diameter
- Mimics cardiac tissue anisotropy

---

### 3. **Kidney Tubule** (`kidney_tubule.py`)
Convoluted proximal tubule structures with perfusable lumen.

**Parameters:**
- `tubule_diameter`: Outer diameter (default: 100µm, range: 50-200µm)
- `wall_thickness`: Thickness of tubule wall (default: 15µm)
- `convolution_amplitude`: Curvature amplitude (default: 500µm)
- `convolution_frequency`: Oscillation frequency (default: 3.0)
- `length`: Total tubule length (default: 10mm)
- `resolution`: Mesh resolution (default: 16)

**Features:**
- Sinusoidal convoluted path
- Hollow interior for fluid flow
- Mimics nephron geometry
- Adjustable convolution parameters

---

### 4. **Lung Alveoli** (`lung_alveoli.py`)
Branching airway structures terminating in alveolar sacs.

**Parameters:**
- `branch_generations`: Branching depth (default: 3, range: 1-5)
- `alveoli_diameter`: Terminal sac size (default: 200µm)
- `airway_diameter`: Initial airway diameter (default: 1.0mm)
- `branch_angle`: Angle between branches (default: 35°)
- `bounding_box`: Overall dimensions (x, y, z) (default: 10×10×10mm)
- `resolution`: Mesh resolution (default: 12)

**Features:**
- Recursive dichotomous branching tree
- Airways narrow with each generation (×0.7)
- Spherical alveoli at terminal branches
- Mimics respiratory tree architecture

---

### 5. **Pancreatic Islet** (`pancreatic_islet.py`)
Spherical cluster structures with porous architecture.

**Parameters:**
- `islet_diameter`: Diameter of each islet (default: 200µm, range: 100-300µm)
- `shell_thickness`: Thickness of shell (default: 50µm)
- `pore_size`: Diameter of pores (default: 20µm)
- `cluster_count`: Number of islets (default: 7)
- `spacing`: Distance between islet centers (default: 300µm)
- `resolution`: Mesh resolution (default: 16)

**Features:**
- Core-shell spherical structure
- Evenly distributed pores (Fibonacci sphere distribution)
- Cluster arrangement in 3D
- Mimics islets of Langerhans

---

### 6. **Liver Sinusoid** (`liver_sinusoid.py`)
Fenestrated tubular channels mimicking hepatic sinusoids.

**Parameters:**
- `sinusoid_diameter`: Tube diameter (default: 20µm, range: 10-30µm)
- `length`: Tube length (default: 5mm)
- `fenestration_size`: Pore size (default: 1µm, scaled from 50-150nm)
- `fenestration_density`: Pore coverage fraction (default: 0.2, range: 0.0-1.0)
- `resolution`: Mesh resolution (default: 16)

**Features:**
- Hollow tube with wall fenestrations
- Small pores distributed evenly on surface
- Perfusable lumen
- Mimics hepatic sinusoidal endothelium

---

## Usage

All generators follow the same API pattern:

```python
from app.geometry.organ import generate_hepatic_lobule_from_dict

# Using dictionary parameters
params = {
    'num_lobules': 7,
    'lobule_radius': 1.5,
    'lobule_height': 3.0,
    'show_sinusoids': True
}

manifold, stats = generate_hepatic_lobule_from_dict(params)

# Stats include:
# - triangle_count: Number of mesh triangles
# - volume_mm3: Scaffold volume
# - scaffold_type: Generator identifier
# - Additional generator-specific metrics
```

## Implementation Notes

- All generators use **manifold3d** for robust boolean operations
- Batch union operations from `geometry.core` for efficient combining
- Parameters with µm units are automatically converted to mm internally
- All generators return both manifold geometry and statistics dictionary
- Statistics always include: `triangle_count`, `volume_mm3`, `scaffold_type`

## Dependencies

- `manifold3d`: Core geometry engine
- `numpy`: Numerical operations
- `dataclasses`: Parameter management
- `geometry.core`: Shared utilities (batch_union, tree_union)

## Design Philosophy

Each generator:
1. Uses biologically-inspired parameters (diameters, spacing from literature)
2. Provides sensible defaults based on physiological ranges
3. Returns validated manifold geometry
4. Includes comprehensive statistics for analysis
5. Supports both dataclass and dictionary parameter interfaces
