# Microfluidic Scaffold Generators

This module provides three specialized microfluidic scaffold generators for tissue engineering and lab-on-chip applications.

## Implemented Generators

### 1. Organ-on-Chip (`organ_on_chip.py`)

Microfluidic channels with tissue chambers for organ-on-chip applications.

**Key Features:**
- Configurable inlet channels leading to tissue chambers
- Rectangular tissue chambers (larger cavities for cell culture)
- Output channels for perfusion
- Created as negative space (channels carved from solid block)

**Parameters:**
- `channel_width_mm`: Channel width (100-500µm typical, default 0.3mm)
- `channel_depth_mm`: Channel depth (50-200µm typical, default 0.1mm)
- `chamber_size_mm`: Chamber dimensions (x, y, z) - default (3.0, 2.0, 0.15)mm
- `chamber_count`: Number of tissue chambers (default 2)
- `inlet_count`: Number of inlet channels (1-4, default 2)
- `chip_size_mm`: Overall chip dimensions (x, y, z) - default (15.0, 10.0, 2.0)mm
- `resolution`: Mesh resolution (default 8)

**Usage:**
```python
from app.geometry.microfluidic import OrganOnChipParams, generate_organ_on_chip

params = OrganOnChipParams(
    channel_width_mm=0.3,
    chamber_count=3,
    inlet_count=2
)
manifold, stats = generate_organ_on_chip(params)
```

---

### 2. Gradient Scaffold (`gradient_scaffold.py`)

Continuous porosity/stiffness gradient scaffolds with varying pore sizes.

**Key Features:**
- Linear, exponential, or sigmoid gradient functions
- Configurable gradient direction (x, y, or z axis)
- Continuous variation in pore size based on position
- Grid-based pore arrangement

**Parameters:**
- `dimensions_mm`: Scaffold dimensions (x, y, z) - default (10.0, 10.0, 10.0)
- `gradient_direction`: Axis for gradient ('x', 'y', or 'z') - default 'z'
- `start_porosity`: Porosity at start (0.0-1.0, default 0.2)
- `end_porosity`: Porosity at end (0.0-1.0, default 0.8)
- `gradient_type`: 'linear', 'exponential', or 'sigmoid' (default 'linear')
- `pore_base_size_mm`: Base pore diameter (default 0.5mm)
- `grid_spacing_mm`: Distance between pore centers (default 1.5mm)
- `resolution`: Mesh resolution (default 12)

**Gradient Functions:**
- **Linear**: `pore_size = start + (end - start) * position/length`
- **Exponential**: `pore_size = start * exp(log(end/start) * position/length)`
- **Sigmoid**: `pore_size = start + (end - start) / (1 + exp(-10*(position/length - 0.5)))`

**Usage:**
```python
from app.geometry.microfluidic import GradientScaffoldParams, generate_gradient_scaffold

params = GradientScaffoldParams(
    gradient_direction='z',
    start_porosity=0.3,
    end_porosity=0.7,
    gradient_type='sigmoid'
)
manifold, stats = generate_gradient_scaffold(params)
```

---

### 3. Perfusable Network (`perfusable_network.py`)

Branching vascular tree for thick tissue perfusion following Murray's law.

**Key Features:**
- Murray's law branching: r_parent³ = Σ r_child³
- Configurable branch generations (depth of tree)
- Binary branching for vascular networks
- Smooth tapered vessel segments
- Optimized for tissue perfusion

**Parameters:**
- `inlet_diameter_mm`: Main inlet diameter (1-3mm typical, default 2.0mm)
- `branch_generations`: Number of branching levels (2-5, default 3)
- `murray_ratio`: Child/parent radius ratio (default 0.79 - Murray's law optimal)
- `network_type`: 'arterial', 'venous', or 'capillary' (default 'arterial')
- `bounding_box_mm`: Network bounds (x, y, z) - default (10.0, 10.0, 10.0)
- `branching_angle_deg`: Angle between parent and child (default 30°)
- `resolution`: Mesh resolution (default 12)

**Murray's Law:**
The optimal branching ratio for minimal energy dissipation in vascular networks.
For symmetric binary branching: `r_child = r_parent × 0.79`

**Usage:**
```python
from app.geometry.microfluidic import PerfusableNetworkParams, generate_perfusable_network

params = PerfusableNetworkParams(
    inlet_diameter_mm=2.0,
    branch_generations=4,
    murray_ratio=0.79,
    network_type='arterial'
)
manifold, stats = generate_perfusable_network(params)
```

---

## Design Patterns

All generators follow the established patterns from `lattice.py` and `vascular.py`:

1. **Dataclass parameters** with sensible defaults
2. **Type hints** for all functions and parameters
3. **Manifold3d** geometry generation
4. **batch_union** from core utilities for efficient boolean operations
5. **Dict-based API compatibility** functions (`generate_*_from_dict`)
6. **Statistics** returned with each geometry (triangle count, volume, etc.)
7. **Docstrings** with detailed parameter descriptions

## Integration

The module exports all parameters and generator functions via `__init__.py`:

```python
from app.geometry.microfluidic import (
    OrganOnChipParams,
    GradientScaffoldParams,
    PerfusableNetworkParams,
    generate_organ_on_chip,
    generate_gradient_scaffold,
    generate_perfusable_network,
    # Dict-based versions for API
    generate_organ_on_chip_from_dict,
    generate_gradient_scaffold_from_dict,
    generate_perfusable_network_from_dict,
)
```

## File Structure

```
microfluidic/
├── __init__.py                 # Package exports
├── organ_on_chip.py           # Organ-on-chip generator
├── gradient_scaffold.py       # Gradient scaffold generator
├── perfusable_network.py      # Perfusable network generator
└── README.md                  # This file
```

## References

- Murray's Law: Murray, C. D. (1926). "The physiological principle of minimum work"
- Organ-on-chip design: Bhatia, S. N., & Ingber, D. E. (2014). "Microfluidic organs-on-chips"
- Gradient scaffolds: Sant, S., et al. (2010). "Hybrid PGS-PCL microfibrous scaffolds"
