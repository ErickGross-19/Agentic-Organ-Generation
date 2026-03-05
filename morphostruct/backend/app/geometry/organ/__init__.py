"""
Organ-specific scaffold generators.

This module provides specialized scaffold generators for various organ systems:

1. **hepatic_lobule** - Hexagonal liver lobule units with portal triads and central veins
2. **cardiac_patch** - Aligned microfibrous scaffolds for cardiomyocytes
3. **kidney_tubule** - Convoluted tubules with perfusable lumen
4. **lung_alveoli** - Branching airways with terminal alveolar sacs
5. **pancreatic_islet** - Spherical clusters with porous architecture
6. **liver_sinusoid** - Fenestrated tubular channels for hepatic blood flow

Each generator follows the standard pattern:
- Dataclass for parameters
- generate_X(params) -> (manifold, stats)
- generate_X_from_dict(params_dict) -> (manifold, stats)
"""

from .hepatic_lobule import (
    HepaticLobuleParams,
    generate_hepatic_lobule,
    generate_hepatic_lobule_from_dict
)

from .cardiac_patch import (
    CardiacPatchParams,
    generate_cardiac_patch,
    generate_cardiac_patch_from_dict
)

from .kidney_tubule import (
    KidneyTubuleParams,
    generate_kidney_tubule,
    generate_kidney_tubule_from_dict
)

from .lung_alveoli import (
    LungAlveoliParams,
    generate_lung_alveoli,
    generate_lung_alveoli_from_dict
)

from .pancreatic_islet import (
    PancreaticIsletParams,
    generate_pancreatic_islet,
    generate_pancreatic_islet_from_dict
)

from .liver_sinusoid import (
    LiverSinusoidParams,
    generate_liver_sinusoid,
    generate_liver_sinusoid_from_dict
)


__all__ = [
    # Hepatic lobule
    'HepaticLobuleParams',
    'generate_hepatic_lobule',
    'generate_hepatic_lobule_from_dict',

    # Cardiac patch
    'CardiacPatchParams',
    'generate_cardiac_patch',
    'generate_cardiac_patch_from_dict',

    # Kidney tubule
    'KidneyTubuleParams',
    'generate_kidney_tubule',
    'generate_kidney_tubule_from_dict',

    # Lung alveoli
    'LungAlveoliParams',
    'generate_lung_alveoli',
    'generate_lung_alveoli_from_dict',

    # Pancreatic islet
    'PancreaticIsletParams',
    'generate_pancreatic_islet',
    'generate_pancreatic_islet_from_dict',

    # Liver sinusoid
    'LiverSinusoidParams',
    'generate_liver_sinusoid',
    'generate_liver_sinusoid_from_dict',
]
